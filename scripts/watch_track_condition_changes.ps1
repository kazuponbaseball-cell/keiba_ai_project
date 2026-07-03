param(
    [string]$Date = (Get-Date -Format "yyyyMMdd"),
    [int]$IntervalSeconds = 90,
    [string]$UntilTime = "17:10",
    [switch]$Notify,
    [switch]$SendIfConfigured,
    [string]$TrackConditionCsv = "data\processed\live_track_conditions\current_track_conditions.csv",
    [string]$StateJson = "data\processed\notifications\track_condition_change_state.json",
    [string]$ProbeSummaryJson = "outputs\analysis\live_track_conditions\track_condition_change_probe_summary.json",
    [string]$LogPath = "outputs\analysis\live_track_conditions\track_condition_watch_log.txt",
    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
}

Set-Location -LiteralPath $ProjectRoot

$FetchTrackConditions = Join-Path $ProjectRoot "scripts\fetch_jra_current_track_conditions.py"
$DetectTrackChanges = Join-Path $ProjectRoot "scripts\detect_track_condition_changes.py"
$RunCurrentUpdate = Join-Path $ProjectRoot "scripts\run_current_strongest_line_update.ps1"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable was not found: $PythonExe"
}
foreach ($scriptPath in @($FetchTrackConditions, $DetectTrackChanges, $RunCurrentUpdate)) {
    if (-not (Test-Path -LiteralPath $scriptPath)) {
        throw "Required script was not found: $scriptPath"
    }
}

function Resolve-ProjectPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path $ProjectRoot $Path)
}

function Write-WatchLog {
    param([string]$Message)
    $resolvedLog = Resolve-ProjectPath $LogPath
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $resolvedLog) | Out-Null
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Write-Host $line
    Add-Content -LiteralPath $resolvedLog -Value $line -Encoding UTF8
}

$today = [datetime]::ParseExact($Date, "yyyyMMdd", $null)
$untilParts = $UntilTime.Split(":")
$until = Get-Date -Year $today.Year -Month $today.Month -Day $today.Day -Hour ([int]$untilParts[0]) -Minute ([int]$untilParts[1]) -Second 0

Write-WatchLog "Started track condition watcher date=$Date interval=${IntervalSeconds}s until=$($until.ToString('HH:mm'))"

# Establish a baseline without sending a notification.
& $PythonExe $FetchTrackConditions `
    --output-csv $TrackConditionCsv `
    --summary-json "outputs\analysis\live_track_conditions\current_track_conditions_summary.json"
if ($LASTEXITCODE -ne 0) {
    throw "Initial track condition fetch failed with exit code $LASTEXITCODE."
}
& $PythonExe $DetectTrackChanges `
    --track-csv $TrackConditionCsv `
    --state-json $StateJson `
    --output-json "outputs\analysis\live_track_conditions\track_condition_change_summary.json" | Out-Host
if ($LASTEXITCODE -ne 0) {
    throw "Initial track condition baseline failed with exit code $LASTEXITCODE."
}

while ((Get-Date) -le $until) {
    try {
        & $PythonExe $FetchTrackConditions `
            --output-csv $TrackConditionCsv `
            --summary-json "outputs\analysis\live_track_conditions\current_track_conditions_summary.json"
        if ($LASTEXITCODE -ne 0) {
            Write-WatchLog "WARN fetch failed exit=$LASTEXITCODE"
            Start-Sleep -Seconds $IntervalSeconds
            continue
        }

        & $PythonExe $DetectTrackChanges `
            --track-csv $TrackConditionCsv `
            --state-json $StateJson `
            --output-json $ProbeSummaryJson `
            --message-text "outputs\notifications\track_condition_change_probe_latest.txt" `
            --no-state-write | Out-Host
        if ($LASTEXITCODE -ne 0) {
            Write-WatchLog "WARN probe failed exit=$LASTEXITCODE"
            Start-Sleep -Seconds $IntervalSeconds
            continue
        }

        $probePath = Resolve-ProjectPath $ProbeSummaryJson
        $probe = Get-Content -LiteralPath $probePath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($probe.changed) {
            $changeText = @($probe.changes | ForEach-Object { "$($_.venue) $($_.label) $($_.old)->$($_.new)" }) -join "; "
            Write-WatchLog "Track condition changed: $changeText"
            $updateArgs = @(
                "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-File", $RunCurrentUpdate,
                "-Date", $Date,
                "-SkipTrackFetch",
                "-ForceNotify",
                "-TrackChangeStateJson", $StateJson
            )
            if ($Notify) {
                $updateArgs += "-Notify"
            }
            if ($SendIfConfigured) {
                $updateArgs += "-SendIfConfigured"
            }
            & powershell.exe @updateArgs
            if ($LASTEXITCODE -ne 0) {
                Write-WatchLog "WARN full update after track change failed exit=$LASTEXITCODE"
            } else {
                Write-WatchLog "Completed full update after track change."
            }
        } else {
            Write-WatchLog "No track condition change."
        }
    } catch {
        Write-WatchLog "ERROR $($_.Exception.Message)"
    }
    Start-Sleep -Seconds $IntervalSeconds
}

Write-WatchLog "Finished track condition watcher."
