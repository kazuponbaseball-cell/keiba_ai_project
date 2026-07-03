param(
    [Parameter(Mandatory = $true)]
    [string[]]$RaceKeys,

    [string]$Sid = "",

    [int]$IntervalSeconds = 45,

    [int]$Count = 3,

    [string]$TicketsCsv = "outputs\analysis\priority_s_betting_policy_ticket_choice_v1\walkforward_selected_tickets.csv",

    [string]$OutputDir = "outputs\analysis\priority_s_live_pipeline_latest",

    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$SnapshotLoop = Join-Path $ProjectRoot "scripts\run_jv_realtime_odds_snapshot_loop.ps1"
$Normalize = Join-Path $ProjectRoot "scripts\normalize_jv_realtime_pair_odds.py"
$Gate = Join-Path $ProjectRoot "scripts\apply_live_odds_gate_to_priority_s_tickets.py"
$LiveOddsCsv = Join-Path $ProjectRoot "data\processed\live_odds\realtime_pair_odds_latest.csv"
$GateOutput = Join-Path $ProjectRoot $OutputDir

. (Join-Path $ProjectRoot "scripts\Resolve-JvSid.ps1")
$sidResolution = Resolve-JvSid -Sid $Sid
$Sid = $sidResolution.Sid
if ([string]::IsNullOrWhiteSpace($Sid)) {
    throw "JV SID is required. Pass -Sid, set environment variable JV_SID, or configure JRA-VAN Data Lab servicekey."
}

if (-not (Test-Path $PythonExe)) {
    throw "Python executable was not found: $PythonExe"
}

Write-Host "Fetching JV realtime pair odds..."
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $SnapshotLoop `
    -RaceKeys $RaceKeys `
    -BetTypes umaren,wide `
    -Sid $Sid `
    -IntervalSeconds $IntervalSeconds `
    -Count $Count

Write-Host "Normalizing realtime pair odds..."
& $PythonExe $Normalize `
    --output-csv "data\processed\live_odds\realtime_pair_odds_latest.csv"

Write-Host "Applying priority-S live odds gate..."
& $PythonExe $Gate `
    --tickets-csv $TicketsCsv `
    --live-odds-csv "data\processed\live_odds\realtime_pair_odds_latest.csv" `
    --output-dir $OutputDir

Write-Host "Done."
Write-Host "Live odds CSV: $LiveOddsCsv"
Write-Host "Gate output: $GateOutput"
