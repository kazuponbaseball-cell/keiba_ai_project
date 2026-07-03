param(
    [string]$Date = (Get-Date -Format "yyyy-MM-dd"),
    [string[]]$Venues = @(),
    [string]$Races = "",
    [string[]]$RaceKeys = @(),
    [string]$Sid = "UNKNOWN",
    [ValidateSet("all", "win_place_frame", "umaren", "wide", "umatan", "trio", "trifecta")]
    [string[]]$BetTypes = @("win_place_frame", "umaren", "wide"),
    [int]$IntervalSeconds = 60,
    [int]$Count = 1,
    [switch]$Loop,
    [switch]$SkipOddsFetch,
    [string]$ScoredCsv = "outputs\analysis\risk_models_v1\investment_features_with_risk_models.csv",
    [string]$TicketsCsv = "outputs\analysis\operational_ticket_profiles_v1\ticket_profiles.csv",
    [string]$RawBodyWeightCsv = "",
    [string]$OutputHtml = "outputs\ui\keiba_dashboard.html",
    [string]$LiveOddsOutputHtml = "outputs\ui\live_odds_dashboard.html",
    [string]$EntryCsvForLiveOddsDashboard = "",
    [int]$MaxRaces = 120
)

$ErrorActionPreference = "Continue"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$snapshotLoop = Join-Path $projectRoot "scripts\run_jv_realtime_odds_snapshot_loop.ps1"
$normalize = Join-Path $projectRoot "scripts\normalize_jv_realtime_pair_odds.py"
$normalizeSingle = Join-Path $projectRoot "scripts\normalize_live_single_odds.py"
$normalizeBody = Join-Path $projectRoot "scripts\normalize_live_body_weight.py"
$dashboard = Join-Path $projectRoot "scripts\build_keiba_dashboard_html.py"
$liveOddsDashboard = Join-Path $projectRoot "scripts\build_live_odds_dashboard_html.py"
$selectKeys = Join-Path $projectRoot "scripts\select_live_race_keys.py"
$selectedJson = "outputs\ui\live_dashboard_selected_races.json"
$livePairOddsCsv = "data\processed\live_odds\realtime_pair_odds_latest.csv"
$liveSingleOddsCsv = "data\processed\live_odds\realtime_single_odds_latest.csv"
$liveBodyWeightCsv = "data\processed\live_body_weight\body_weight_latest.csv"

if (-not (Test-Path $python)) {
    throw "Python executable was not found: $python"
}

function Resolve-RaceKeys {
    if ($RaceKeys.Count -gt 0) {
        return $RaceKeys
    }
    $args = @($selectKeys, "--scored-csv", $ScoredCsv, "--output-json", $selectedJson)
    if ($Date -ne "") {
        $args += @("--date", $Date)
    }
    if ($Venues.Count -gt 0) {
        $args += @("--venues")
        $args += $Venues
    }
    $raceNumbers = @()
    if ($Races -ne "") {
        $raceNumbers = $Races -split "[,\s]+" | Where-Object { $_ -ne "" }
    }
    if ($raceNumbers.Count -gt 0) {
        $args += @("--races")
        $args += $raceNumbers
    }
    $raw = & $python @args
    if ($LASTEXITCODE -ne 0) {
        throw "Race key selection failed."
    }
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return @()
    }
    return $raw.Split(",") | Where-Object { $_ -ne "" }
}

function Invoke-OneUpdate {
    Set-Location $projectRoot
    $keys = @(Resolve-RaceKeys)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$stamp] Dashboard live update"
    if ($keys.Count -eq 0) {
        Write-Warning "No race keys selected. Check -Date, -Venues, and -Races."
    } else {
        Write-Host "RaceKeys: $($keys -join ',')"
    }

    if (-not $SkipOddsFetch -and $keys.Count -gt 0) {
        Write-Host "Fetching realtime odds..."
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $snapshotLoop `
            -RaceKeys $keys `
            -BetTypes $BetTypes `
            -Sid $Sid `
            -IntervalSeconds 1 `
            -Count 1
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Realtime odds fetch returned exit code $LASTEXITCODE. Continuing with the latest available odds CSV."
        }

        Write-Host "Normalizing realtime pair odds..."
        & $python $normalize --output-csv $livePairOddsCsv
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Realtime odds normalization returned exit code $LASTEXITCODE. Continuing with dashboard build."
        }

        Write-Host "Normalizing realtime single odds..."
        & $python $normalizeSingle --output-csv $liveSingleOddsCsv
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Realtime single odds normalization returned exit code $LASTEXITCODE. Continuing with dashboard build."
        }
    }

    $bodyCsvForDashboard = ""
    if ($RawBodyWeightCsv -ne "") {
        Write-Host "Normalizing body weight..."
        & $python $normalizeBody --manual-csv $RawBodyWeightCsv --output-csv $liveBodyWeightCsv
        if ($LASTEXITCODE -eq 0) {
            $bodyCsvForDashboard = $liveBodyWeightCsv
        } else {
            Write-Warning "Body weight normalization returned exit code $LASTEXITCODE. Continuing without live body weight."
        }
    }

    Write-Host "Building dashboard HTML..."
    $dashArgs = @(
        $dashboard,
        "--scored-csv", $ScoredCsv,
        "--tickets-csv", $TicketsCsv,
        "--live-pair-odds-csv", $livePairOddsCsv,
        "--live-single-odds-csv", $liveSingleOddsCsv,
        "--output-html", $OutputHtml,
        "--max-races", "$MaxRaces"
    )
    if ($bodyCsvForDashboard -ne "") {
        $dashArgs += @("--body-weight-csv", $bodyCsvForDashboard)
    }
    & $python @dashArgs

    if (Test-Path $liveOddsDashboard) {
        Write-Host "Building live odds dashboard HTML..."
        $liveArgs = @(
            $liveOddsDashboard,
            "--single-odds-csv", $liveSingleOddsCsv,
            "--pair-odds-csv", $livePairOddsCsv,
            "--output-html", $LiveOddsOutputHtml
        )
        if ($EntryCsvForLiveOddsDashboard -ne "") {
            $liveArgs += @("--entry-csv", $EntryCsvForLiveOddsDashboard)
        }
        & $python @liveArgs
    }
}

do {
    Invoke-OneUpdate
    if ($Loop) {
        Start-Sleep -Seconds $IntervalSeconds
    }
} while ($Loop)
