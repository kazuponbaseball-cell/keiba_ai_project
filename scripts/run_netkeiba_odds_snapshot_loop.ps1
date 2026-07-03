param(
    [Parameter(Mandatory = $true)]
    [string[]]$RaceId,

    [int]$IntervalSeconds = 300,

    [int]$Count = 12,

    [string]$OddsType = "b1",

    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Fetcher = Join-Path $ProjectRoot "scripts\fetch_netkeiba_odds_snapshot.py"

for ($i = 1; $i -le $Count; $i++) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    foreach ($id in $RaceId) {
        Write-Host "[$stamp] Fetching odds snapshot race_id=$id type=$OddsType ($i/$Count)"
        & $PythonExe $Fetcher --race-id $id --odds-type $OddsType
    }
    if ($i -lt $Count) {
        Start-Sleep -Seconds $IntervalSeconds
    }
}
