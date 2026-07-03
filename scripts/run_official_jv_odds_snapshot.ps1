param(
    [Parameter(Mandatory = $true)]
    [string[]]$RaceKeys,

    [string]$Sid = "",

    [ValidateSet("all", "win_place_frame", "umaren", "wide", "umatan", "trio", "trifecta")]
    [string[]]$BetTypes = @("win_place_frame", "umaren", "wide"),

    [int]$IntervalSeconds = 1,
    [int]$Count = 1,

    [string]$OutputDir = "data\raw\jv_realtime_odds",

    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $ProjectRoot

. (Join-Path $ProjectRoot "scripts\Resolve-JvSid.ps1")
$sidResolution = Resolve-JvSid -Sid $Sid
$Sid = $sidResolution.Sid
if ([string]::IsNullOrWhiteSpace($Sid)) {
    throw "JV SID is required for official JV odds. Set `$env:JV_SID, pass -Sid, or configure JRA-VAN Data Lab servicekey."
}
Write-Host "JV SID source: $($sidResolution.Source)"

$Check = Join-Path $ProjectRoot "scripts\check_jv_realtime_setup.ps1"
$SnapshotLoop = Join-Path $ProjectRoot "scripts\run_jv_realtime_odds_snapshot_loop.ps1"
$NormalizePair = Join-Path $ProjectRoot "scripts\normalize_jv_realtime_pair_odds.py"
$NormalizeSingle = Join-Path $ProjectRoot "scripts\normalize_live_single_odds.py"

Write-Host "[1/4] Checking JV-Link setup"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Check -Sid $Sid -RequireSid | Write-Host
if ($LASTEXITCODE -ne 0) {
    throw "JV-Link setup check failed."
}

Write-Host "[2/4] Fetching official JV realtime odds"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $SnapshotLoop `
    -RaceKeys $RaceKeys `
    -BetTypes $BetTypes `
    -Sid $Sid `
    -IntervalSeconds $IntervalSeconds `
    -Count $Count `
    -OutputDir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "Official JV realtime odds fetch failed."
}

Write-Host "[3/4] Normalizing pair odds"
& $PythonExe $NormalizePair `
    --raw-dir $OutputDir `
    --output-csv "data\processed\live_odds\realtime_pair_odds_latest.csv"
if ($LASTEXITCODE -ne 0) {
    throw "Pair odds normalization failed."
}

Write-Host "[4/4] Normalizing win/place odds"
& $PythonExe $NormalizeSingle `
    --raw-dir $OutputDir `
    --output-csv "data\processed\live_odds\realtime_single_odds_latest.csv"
if ($LASTEXITCODE -ne 0) {
    throw "Single odds normalization failed."
}

Write-Host "Official JV odds update complete."
Write-Host "Pair odds: data\processed\live_odds\realtime_pair_odds_latest.csv"
Write-Host "Single odds: data\processed\live_odds\realtime_single_odds_latest.csv"
