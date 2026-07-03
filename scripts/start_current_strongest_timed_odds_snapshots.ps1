param(
    [string]$Date = (Get-Date -Format "yyyyMMdd"),

    [int[]]$OffsetsMinutes = @(10, 5, 3),

    [int]$PollSeconds = 15,

    [int]$GroupWindowSeconds = 45,

    [string]$EntryCsv = "",

    [string]$PredictionCsv = "",

    [string]$CurrentInputsJson = "outputs\runtime\current_dashboard_inputs.json",

    [string]$ScheduleCsv = "data\processed\live_odds\current_strongest_timed_schedule.csv",

    [string]$StateJson = "data\processed\live_odds\current_strongest_timed_snapshot_state.json",

    [string]$LogPath = "outputs\analysis\current_strongest_line_update\timed_snapshot_log.txt",

    [switch]$SkipOddsFetch,

    [switch]$SkipTrackFetch,

    [switch]$SkipBodyWeight,

    [switch]$SkipResultFetch,

    [switch]$SkipWin5,

    [switch]$SkipDashboard,

    [switch]$FastT3,

    [switch]$SendIfConfigured,

    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",

    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
}

Set-Location -LiteralPath $ProjectRoot

$BuildSchedule = Join-Path $ProjectRoot "scripts\build_live_odds_race_schedule.py"
$UpdateScript = Join-Path $ProjectRoot "scripts\run_current_strongest_line_update.ps1"

function Resolve-ProjectPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path $ProjectRoot $Path)
}

function Get-StateValue {
    param(
        [object]$State,
        [string]$Name,
        [object]$Default = ""
    )
    if ($null -ne $State -and $State.PSObject.Properties[$Name]) {
        return $State.$Name
    }
    return $Default
}

function Write-AutoLog {
    param([string]$Message)
    $resolved = Resolve-ProjectPath $LogPath
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $resolved) | Out-Null
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Write-Host $line
    Add-Content -LiteralPath $resolved -Value $line -Encoding UTF8
}

function Read-State {
    $resolved = Resolve-ProjectPath $StateJson
    if (-not (Test-Path -LiteralPath $resolved)) {
        return @{}
    }
    $text = Get-Content -LiteralPath $resolved -Raw -Encoding UTF8
    if (-not $text.Trim()) {
        return @{}
    }
    $obj = $text | ConvertFrom-Json
    $state = @{}
    foreach ($prop in $obj.PSObject.Properties) {
        $state[$prop.Name] = [bool]$prop.Value
    }
    return $state
}

function Save-State {
    param([hashtable]$State)
    $resolved = Resolve-ProjectPath $StateJson
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $resolved) | Out-Null
    $State | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $resolved -Encoding UTF8
}

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable was not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $BuildSchedule)) {
    throw "Schedule builder was not found: $BuildSchedule"
}
if (-not (Test-Path -LiteralPath $UpdateScript)) {
    throw "Current strongest update script was not found: $UpdateScript"
}

if ([string]::IsNullOrWhiteSpace($EntryCsv) -or [string]::IsNullOrWhiteSpace($PredictionCsv)) {
    $inputsPath = Resolve-ProjectPath $CurrentInputsJson
    if (Test-Path -LiteralPath $inputsPath) {
        try {
            $inputs = Get-Content -LiteralPath $inputsPath -Raw -Encoding UTF8 | ConvertFrom-Json
            $inputDate = Get-StateValue -State $inputs -Name "date" -Default ""
            if ([string]::IsNullOrWhiteSpace($inputDate) -or $inputDate -eq $Date) {
                if ([string]::IsNullOrWhiteSpace($EntryCsv)) {
                    $EntryCsv = Get-StateValue -State $inputs -Name "entry_csv" -Default ""
                }
                if ([string]::IsNullOrWhiteSpace($PredictionCsv)) {
                    $PredictionCsv = Get-StateValue -State $inputs -Name "prediction_csv" -Default ""
                }
            }
        } catch {
            Write-AutoLog "WARN could not read current input json: $($_.Exception.Message)"
        }
    }
}

if ([string]::IsNullOrWhiteSpace($EntryCsv)) {
    throw "EntryCsv is required. Pass -EntryCsv or set outputs\runtime\current_dashboard_inputs.json."
}

$scheduleArgs = @(
    $BuildSchedule,
    "--entry-csv", $EntryCsv,
    "--date", $Date,
    "--output-csv", $ScheduleCsv,
    "--output-json", ($ScheduleCsv -replace "\.csv$", ".json")
)

Write-AutoLog "Building current strongest timed schedule date=$Date entry=$EntryCsv"
& $PythonExe @scheduleArgs
if ($LASTEXITCODE -ne 0) {
    throw "Schedule build failed with exit code $LASTEXITCODE."
}

$schedulePath = Resolve-ProjectPath $ScheduleCsv
if (-not (Test-Path -LiteralPath $schedulePath)) {
    throw "Schedule CSV was not found: $schedulePath"
}

$schedule = @(Import-Csv -LiteralPath $schedulePath)
if ($schedule.Count -eq 0) {
    throw "Schedule CSV has no rows: $schedulePath"
}

$state = Read-State
$jobs = @()
foreach ($row in $schedule) {
    $postTime = [datetime]::Parse($row.post_time)
    foreach ($offset in $OffsetsMinutes) {
        $label = "T-$offset"
        $jobs += [pscustomobject]@{
            race_key = [string]$row.race_key
            venue = [string]$row.venue
            race_no = [int]$row.race_no
            post_time = $postTime
            due_time = $postTime.AddMinutes(-1 * $offset)
            decision_label = $label
            state_key = "$($row.race_key)|$label"
        }
    }
}

$jobs = $jobs | Sort-Object due_time, race_key
$lastDue = ($jobs | Sort-Object due_time -Descending | Select-Object -First 1).due_time
Write-AutoLog "Started current strongest timed odds snapshots. jobs=$($jobs.Count)"

while ($true) {
    $now = Get-Date
    $staleJobs = @(
        $jobs | Where-Object {
            (-not $state.ContainsKey($_.state_key)) -and
            ($_.post_time -le $now)
        }
    )
    if ($staleJobs.Count -gt 0) {
        foreach ($job in $staleJobs) {
            $state[$job.state_key] = $true
        }
        Save-State -State $state
        Write-AutoLog "Skipped stale timed snapshot jobs after post time. jobs=$($staleJobs.Count)"
    }

    $dueJobs = @(
        $jobs | Where-Object {
            (-not $state.ContainsKey($_.state_key)) -and
            ($_.due_time -le $now) -and
            ($_.due_time -ge $now.AddMinutes(-20)) -and
            ($_.post_time -gt $now)
        }
    )

    if ($dueJobs.Count -gt 0) {
        foreach ($group in ($dueJobs | Group-Object decision_label)) {
            $label = $group.Name
            $raceKeys = @($group.Group | Select-Object -ExpandProperty race_key -Unique)
            Write-AutoLog "Running current strongest update label=$label races=$($raceKeys -join ',')"
            try {
                $args = @(
                    "-File", $UpdateScript,
                    "-Date", $Date,
                    "-RaceKeys"
                )
                $args += $raceKeys
                $args += @(
                    "-DecisionLabel", $label,
                    "-EntryCsv", $EntryCsv,
                    "-PythonExe", $PythonExe
                )
                if (-not [string]::IsNullOrWhiteSpace($PredictionCsv)) {
                    $args += @("-PredictionCsv", $PredictionCsv)
                }
                if ($SkipOddsFetch) {
                    $args += "-SkipOddsFetch"
                }
                if ($SkipTrackFetch -or ($FastT3 -and $label -eq "T-3")) {
                    $args += "-SkipTrackFetch"
                }
                if ($SkipBodyWeight -or ($FastT3 -and $label -eq "T-3")) {
                    $args += "-SkipBodyWeight"
                }
                if ($SkipResultFetch -or ($FastT3 -and $label -eq "T-3")) {
                    $args += "-SkipResultFetch"
                }
                if ($SkipWin5 -or ($FastT3 -and $label -eq "T-3")) {
                    $args += "-SkipWin5"
                }
                if ($SkipDashboard -or ($FastT3 -and $label -eq "T-3")) {
                    $args += "-SkipDashboard"
                }
                if ($FastT3 -and $label -eq "T-3") {
                    $args += "-SkipExternalAudit"
                    $args += "-ForceNotify"
                }
                if ($SendIfConfigured) {
                    $args += "-SendIfConfigured"
                }
                $output = @(powershell.exe -NoProfile -ExecutionPolicy Bypass @args)
                foreach ($line in $output) {
                    Write-AutoLog "label=$label $line"
                }
                foreach ($job in $group.Group) {
                    $state[$job.state_key] = $true
                }
                Save-State -State $state
                Write-AutoLog "Completed current strongest update label=$label races=$($raceKeys -join ',')"
            } catch {
                Write-AutoLog "ERROR label=$label races=$($raceKeys -join ',') message=$($_.Exception.Message)"
            }
            Start-Sleep -Seconds $GroupWindowSeconds
        }
    }

    $remaining = @($jobs | Where-Object { -not $state.ContainsKey($_.state_key) })
    if ($remaining.Count -eq 0 -or $now -gt $lastDue.AddMinutes(20)) {
        Write-AutoLog "Finished current strongest timed odds snapshots. remaining=$($remaining.Count)"
        break
    }
    Start-Sleep -Seconds $PollSeconds
}
