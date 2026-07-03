param(
    [string]$Date = (Get-Date -Format "yyyy-MM-dd"),

    [string[]]$Venues = @(),

    [string]$Races = "",

    [string[]]$RaceKeys = @(),

    [string]$Sid = "",

    [ValidateSet("all", "win_place_frame", "umaren", "wide", "umatan", "trio", "trifecta")]
    [string[]]$BetTypes = @("win_place_frame", "umaren", "wide"),

    [ValidateSet("T-10", "T-5", "T-3", "final_check", "manual")]
    [string]$DecisionLabel = "manual",

    [int]$IntervalSeconds = 1,

    [int]$Count = 1,

    [switch]$SkipOddsFetch,

    [switch]$SkipTimeline,

    [switch]$SkipJraOfficialFallback,

    [switch]$ProxyWhenMissing,

    [switch]$OfficialOnly,

    [string]$ScoredCsv = "outputs\analysis\risk_models_v1\investment_features_with_risk_models.csv",

    [string]$BaseTicketsCsv = "outputs\analysis\robust_expansion_runtime_ready_v1\standard_plus_robust_runtime_ready_tickets.csv",

    [string]$PairLiveCsv = "data\processed\live_odds\realtime_pair_odds_latest.csv",

    [string]$SingleLiveCsv = "data\processed\live_odds\realtime_single_odds_latest.csv",

    [string]$BodyWeightCsv = "data\processed\live_body_weight\body_weight_latest.csv",

    [string]$McsPboPolicy = "mcs_full_margin095_s0304_skip03119",

    [string]$ManualPairOddsCsv = "",

    [string]$ManualSingleOddsCsv = "",

    [string]$OutputRoot = "outputs\analysis\race_day_runtime_operation_latest",

    [string]$DashboardHtml = "outputs\ui\keiba_dashboard_runtime.html",

    [string]$LiveOddsDashboardHtml = "outputs\ui\live_odds_dashboard.html",

    [string]$EntryCsvForLiveOddsDashboard = "",

    [int]$MaxDashboardRaces = 180,

    [switch]$LineNotify,

    [switch]$LineNotifyIfConfigured,

    [string]$LineDashboardUrl = "",

    [int]$LineMaxRaces = 8,

    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$SnapshotLoop = Join-Path $ProjectRoot "scripts\run_jv_realtime_odds_snapshot_loop.ps1"
$FetchJraOfficialOdds = Join-Path $ProjectRoot "scripts\fetch_jra_official_odds.py"
$SelectKeys = Join-Path $ProjectRoot "scripts\select_live_race_keys.py"
$RuntimePipeline = Join-Path $ProjectRoot "scripts\run_race_day_runtime_pipeline.py"
$AppendTimeline = Join-Path $ProjectRoot "scripts\append_live_odds_timeline.py"
$BuildTimelineFeatures = Join-Path $ProjectRoot "scripts\build_odds_timeline_features.py"
$EvaluateFixedTimePairEdge = Join-Path $ProjectRoot "scripts\evaluate_fixed_time_pair_edge.py"
$BuildLiveOddsDashboard = Join-Path $ProjectRoot "scripts\build_live_odds_dashboard_html.py"
$LineAlert = Join-Path $ProjectRoot "scripts\send_line_keiba_alert.ps1"

. (Join-Path $ProjectRoot "scripts\Resolve-JvSid.ps1")

if (-not (Test-Path $PythonExe)) {
    throw "Python executable was not found: $PythonExe"
}
if (-not (Test-Path $RuntimePipeline)) {
    throw "Runtime pipeline was not found: $RuntimePipeline"
}

Set-Location $ProjectRoot

function Resolve-ProjectPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path $ProjectRoot $Path)
}

function Resolve-RaceKeys {
    if ($RaceKeys.Count -gt 0) {
        return @($RaceKeys)
    }

    if (-not (Test-Path $SelectKeys)) {
        Write-Warning "Race key selector was not found. Pass -RaceKeys explicitly to fetch JV odds."
        return @()
    }

    $selectedJson = Join-Path (Resolve-ProjectPath $OutputRoot) "selected_races.json"
    $args = @(
        $SelectKeys,
        "--scored-csv", $ScoredCsv,
        "--output-json", $selectedJson
    )
    if ($Date -ne "") {
        $args += @("--date", $Date)
    }
    if ($Venues.Count -gt 0) {
        $args += "--venues"
        $args += $Venues
    }
    $raceNumbers = @()
    if ($Races -ne "") {
        $raceNumbers = @($Races -split "[,\s]+" | Where-Object { $_ -ne "" })
    }
    if ($raceNumbers.Count -gt 0) {
        $args += "--races"
        $args += $raceNumbers
    }

    $raw = & $PythonExe @args
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Race key selection failed. Pass -RaceKeys explicitly if this is a race-day TARGET snapshot."
        return @()
    }
    if ([string]::IsNullOrWhiteSpace($raw)) {
        return @()
    }
    return @($raw.Split(",") | Where-Object { $_ -ne "" })
}

$sidResolution = Resolve-JvSid -Sid $Sid
$Sid = $sidResolution.Sid
if ([string]::IsNullOrWhiteSpace($Sid)) {
    $Sid = "UNKNOWN"
}
if ($OfficialOnly -and -not $SkipOddsFetch -and $Sid -eq "UNKNOWN") {
    throw "OfficialOnly requires JV SID. Set `$env:JV_SID, pass -Sid, or configure JRA-VAN Data Lab servicekey."
}

$resolvedOutputRoot = Resolve-ProjectPath $OutputRoot
New-Item -ItemType Directory -Force -Path $resolvedOutputRoot | Out-Null

$keys = @(Resolve-RaceKeys)
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Write-Host "[$stamp] Race-day runtime operation"
Write-Host "Mode: $(if ($ProxyWhenMissing) { 'proxy/debug' } else { 'strict live' })"
Write-Host "OutputRoot: $OutputRoot"
if ($keys.Count -gt 0) {
    Write-Host "RaceKeys: $($keys -join ',')"
} else {
    Write-Warning "No race keys selected. Odds fetch will be skipped unless -RaceKeys is supplied."
    if ($OfficialOnly -and -not $SkipOddsFetch) {
        throw "OfficialOnly requires RaceKeys. Pass -RaceKeys or provide a scored CSV/date selection that can derive them."
    }
}

if (-not $SkipOddsFetch -and $keys.Count -gt 0) {
    if (-not (Test-Path $SnapshotLoop)) {
        throw "Snapshot loop was not found: $SnapshotLoop"
    }
    Write-Host "Fetching JV realtime odds..."
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $SnapshotLoop `
        -RaceKeys $keys `
        -BetTypes $BetTypes `
        -Sid $Sid `
        -IntervalSeconds $IntervalSeconds `
        -Count $Count
    if ($LASTEXITCODE -ne 0) {
        if ($OfficialOnly -and $SkipJraOfficialFallback) {
            throw "JV realtime odds fetch failed in OfficialOnly mode."
        }
        Write-Warning "JV realtime odds fetch returned exit code $LASTEXITCODE. Continuing to JRA official fallback/latest available odds."
    }
}

if (-not $SkipOddsFetch -and -not $SkipJraOfficialFallback -and $keys.Count -gt 0 -and (Test-Path $FetchJraOfficialOdds)) {
    $officialPairCsv = Join-Path $resolvedOutputRoot "jra_official_pair_odds_latest.csv"
    $officialSingleCsv = Join-Path $resolvedOutputRoot "jra_official_single_odds_latest.csv"
    $officialSummaryJson = Join-Path $resolvedOutputRoot "jra_official_odds_summary.json"
    $officialRawDir = Join-Path $resolvedOutputRoot "jra_official_odds_raw"
    $jraBetTypes = @()
    foreach ($betType in $BetTypes) {
        if ($betType -eq "all") {
            $jraBetTypes += @("win_place_frame", "umaren", "wide")
        } elseif (@("win_place_frame", "umaren", "wide") -contains $betType) {
            $jraBetTypes += $betType
        }
    }
    $jraBetTypes = @($jraBetTypes | Select-Object -Unique)
    if ($jraBetTypes.Count -gt 0) {
        Write-Host "Fetching JRA official odds fallback..."
        $jraArgs = @(
            $FetchJraOfficialOdds,
            "--date", $Date,
            "--race-keys", ($keys -join ","),
            "--raw-dir", $officialRawDir,
            "--pair-output-csv", $officialPairCsv,
            "--single-output-csv", $officialSingleCsv,
            "--summary-json", $officialSummaryJson
        )
        $jraArgs += "--bet-types"
        $jraArgs += $jraBetTypes
        & $PythonExe @jraArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "JRA official odds fallback returned exit code $LASTEXITCODE. Strict mode will still wait if no live odds are available."
        } else {
            if ($ManualPairOddsCsv -eq "" -and (Test-Path $officialPairCsv)) {
                $ManualPairOddsCsv = $officialPairCsv
            }
            if ($ManualSingleOddsCsv -eq "" -and (Test-Path $officialSingleCsv)) {
                $ManualSingleOddsCsv = $officialSingleCsv
            }
        }
    }
}

Write-Host "Running calibrated runtime pipeline..."
$pipelineArgs = @(
    $RuntimePipeline,
    "--base-tickets-csv", $BaseTicketsCsv,
    "--output-root", $OutputRoot,
    "--dashboard-html", $DashboardHtml,
    "--scored-csv", $ScoredCsv,
    "--pair-live-csv", $PairLiveCsv,
    "--single-live-csv", $SingleLiveCsv,
    "--body-weight-csv", $BodyWeightCsv,
    "--max-dashboard-races", "$MaxDashboardRaces",
    "--normalize-live-odds"
)
if ($ManualPairOddsCsv -ne "") {
    $pipelineArgs += @("--manual-pair-odds-csv", $ManualPairOddsCsv)
}
if ($ManualSingleOddsCsv -ne "") {
    $pipelineArgs += @("--manual-single-odds-csv", $ManualSingleOddsCsv)
}
if ($ProxyWhenMissing) {
    $pipelineArgs += "--proxy-when-missing"
}
if ($OfficialOnly) {
    $pipelineArgs += "--skip-netkeiba-export"
}
if ($McsPboPolicy -ne "") {
    $pipelineArgs += @("--mcs-pbo-policy", $McsPboPolicy)
}
& $PythonExe @pipelineArgs
if ($LASTEXITCODE -ne 0) {
    throw "Calibrated runtime pipeline failed with exit code $LASTEXITCODE."
}

if (-not $SkipTimeline) {
    Write-Host "Appending latest odds to timeline..."
    & $PythonExe $AppendTimeline `
        --pair-latest-csv $PairLiveCsv `
        --single-latest-csv $SingleLiveCsv `
        --decision-label $DecisionLabel `
        --summary-json (Join-Path $OutputRoot "odds_timeline_append_summary.json")
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Odds timeline append returned exit code $LASTEXITCODE."
    }

    Write-Host "Building live odds timeline features..."
    & $PythonExe $BuildTimelineFeatures `
        --timeline-csv "data\processed\live_odds\realtime_single_odds_timeline.csv" `
        --output-csv "data\processed\live_odds\realtime_single_odds_timeline_features.csv"
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Single odds timeline feature build returned exit code $LASTEXITCODE."
    }

    $summaryPathForFixedEdge = Join-Path $resolvedOutputRoot "summary.json"
    if ((Test-Path $EvaluateFixedTimePairEdge) -and (Test-Path $summaryPathForFixedEdge)) {
        try {
            $operationSummaryForFixedEdge = Get-Content -Path $summaryPathForFixedEdge -Raw -Encoding UTF8 | ConvertFrom-Json
            $fixedEdgeTicketsCsv = $operationSummaryForFixedEdge.final_tickets_csv
            if (-not [string]::IsNullOrWhiteSpace($fixedEdgeTicketsCsv)) {
                Write-Host "Evaluating fixed-time pair edge..."
                & $PythonExe $EvaluateFixedTimePairEdge `
                    --tickets-csv $fixedEdgeTicketsCsv `
                    --pair-timeline-csv "data\processed\live_odds\realtime_pair_odds_timeline.csv" `
                    --output-dir (Join-Path $OutputRoot "fixed_time_pair_edge")
                if ($LASTEXITCODE -ne 0) {
                    Write-Warning "Fixed-time pair edge evaluation returned exit code $LASTEXITCODE."
                }
            }
        } catch {
            Write-Warning "Fixed-time pair edge evaluation could not be started: $($_.Exception.Message)"
        }
    }
}

if (Test-Path $BuildLiveOddsDashboard) {
    Write-Host "Building live odds dashboard HTML..."
    $liveOddsDashboardArgs = @(
        $BuildLiveOddsDashboard,
        "--single-odds-csv", $SingleLiveCsv,
        "--pair-odds-csv", $PairLiveCsv,
        "--output-html", $LiveOddsDashboardHtml
    )
    if ($EntryCsvForLiveOddsDashboard -ne "") {
        $liveOddsDashboardArgs += @("--entry-csv", $EntryCsvForLiveOddsDashboard)
    }
    & $PythonExe @liveOddsDashboardArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Live odds dashboard build returned exit code $LASTEXITCODE."
    }
}

$summaryPath = Join-Path $resolvedOutputRoot "summary.json"
if (Test-Path $summaryPath) {
    $summary = Get-Content -Path $summaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    Write-Host "Selected tickets: $($summary.selected_metrics.tickets)"
    Write-Host "Selected races: $($summary.selected_metrics.races)"
    Write-Host "Stake yen: $($summary.selected_metrics.stake_yen)"
    Write-Host "Dashboard: $DashboardHtml"
    Write-Host "Live odds dashboard: $LiveOddsDashboardHtml"
    if ($summary.netkeiba_plan_dir -and -not $OfficialOnly) {
        Write-Host "Netkeiba plan: $($summary.netkeiba_plan_dir)"
    }

    if (($LineNotify -or $LineNotifyIfConfigured) -and (Test-Path $LineAlert)) {
        try {
            $ticketsForLine = ""
            if ($summary.selected_csv -and (Test-Path (Resolve-ProjectPath $summary.selected_csv))) {
                $ticketsForLine = $summary.selected_csv
            } elseif ($summary.final_tickets_csv -and (Test-Path (Resolve-ProjectPath $summary.final_tickets_csv))) {
                $ticketsForLine = $summary.final_tickets_csv
            }

            if (-not [string]::IsNullOrWhiteSpace($ticketsForLine)) {
                $lineUrl = $LineDashboardUrl
                if ([string]::IsNullOrWhiteSpace($lineUrl)) {
                    $publicInfoPath = Join-Path $ProjectRoot "outputs\runtime\public_dashboard_tunnel.json"
                    if (Test-Path $publicInfoPath) {
                        $publicInfo = Get-Content -Path $publicInfoPath -Raw -Encoding UTF8 | ConvertFrom-Json
                        if ($publicInfo.public_url) {
                            $dashboardPathForUrl = $DashboardHtml.Replace("\", "/")
                            $lineUrl = "$($publicInfo.public_url)/$dashboardPathForUrl"
                        }
                    }
                }

                Write-Host "Sending LINE final-ticket alert..."
                $lineArgs = @(
                    "-File", $LineAlert,
                    "-Mode", "final",
                    "-TicketsCsv", $ticketsForLine,
                    "-DashboardUrl", $lineUrl,
                    "-MaxRaces", "$LineMaxRaces",
                    "-PythonExe", $PythonExe
                )
                if ($LineNotify) {
                    $lineArgs += "-Send"
                }
                if ($LineNotifyIfConfigured) {
                    $lineArgs += "-SendIfConfigured"
                }
                powershell.exe -NoProfile -ExecutionPolicy Bypass @lineArgs
            } else {
                Write-Warning "LINE final-ticket alert skipped because no selected/final ticket CSV was found."
            }
        } catch {
            Write-Warning "LINE final-ticket alert failed: $($_.Exception.Message)"
        }
    }
}

Write-Host "Done."
