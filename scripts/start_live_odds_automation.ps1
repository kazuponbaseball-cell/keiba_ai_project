param(
    [string]$Date = "",

    [string[]]$Venues = @(),

    [int[]]$Races = @(),

    [string]$Sid = "",

    [int[]]$OffsetsMinutes = @(10, 5, 3),

    [string]$EntryCsv = "data\datasets\inference\weekly\entry_snapshot.csv",

    [string]$ScheduleCsv = "data\processed\live_odds\live_odds_race_schedule.csv",

    [string]$TicketsCsv = "outputs\analysis\roi_mode_stake_sizing_v1\stake_sized_ticket_profiles.csv",

    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$BuildSchedule = Join-Path $ProjectRoot "scripts\build_live_odds_race_schedule.py"
$Watcher = Join-Path $ProjectRoot "scripts\watch_live_odds_schedule.ps1"

. (Join-Path $ProjectRoot "scripts\Resolve-JvSid.ps1")
$sidResolution = Resolve-JvSid -Sid $Sid
$Sid = $sidResolution.Sid
if ([string]::IsNullOrWhiteSpace($Sid)) {
    throw "JV SID is required. Pass -Sid, set environment variable JV_SID, or configure JRA-VAN Data Lab servicekey."
}
if (-not (Test-Path $PythonExe)) {
    throw "Python executable was not found: $PythonExe"
}

Set-Location $ProjectRoot

$buildArgs = @(
    $BuildSchedule,
    "--entry-csv", $EntryCsv,
    "--output-csv", $ScheduleCsv
)
if ($Date) {
    $buildArgs += @("--date", $Date)
}
if ($Venues.Count -gt 0) {
    $buildArgs += "--venues"
    $buildArgs += $Venues
}
if ($Races.Count -gt 0) {
    $buildArgs += "--races"
    $buildArgs += ($Races | ForEach-Object { [string]$_ })
}

Write-Host "Building live odds schedule..."
& $PythonExe @buildArgs

Write-Host "Starting live odds watcher..."
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Watcher `
    -ScheduleCsv $ScheduleCsv `
    -Sid $Sid `
    -OffsetsMinutes $OffsetsMinutes `
    -TicketsCsv $TicketsCsv
