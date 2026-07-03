param(
    [Parameter(Mandatory = $true)]
    [string]$RaceKey,

    [string]$Sid = "UNKNOWN",

    [string]$OutputDir = "data\raw\jv_realtime_odds"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Fetcher = Join-Path $ProjectRoot "scripts\fetch_jv_realtime_odds.ps1"

$PowerShell32 = "$env:WINDIR\SysWOW64\WindowsPowerShell\v1.0\powershell.exe"
if (-not (Test-Path $PowerShell32)) {
    throw "32-bit PowerShell was not found: $PowerShell32"
}

& $PowerShell32 -NoProfile -ExecutionPolicy Bypass -File $Fetcher -RaceKey $RaceKey -BetType wide -Sid $Sid -OutputDir $OutputDir
exit $LASTEXITCODE
