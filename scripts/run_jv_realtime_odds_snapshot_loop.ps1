param(
    [Parameter(Mandatory = $true)]
    [string[]]$RaceKeys,

    [ValidateSet("all", "win_place_frame", "umaren", "wide", "umatan", "trio", "trifecta")]
    [string[]]$BetTypes = @("win_place_frame", "wide"),

    [string]$Sid = "",

    [int]$IntervalSeconds = 60,

    [int]$Count = 5,

    [string]$OutputDir = "data\raw\jv_realtime_odds"
)

$ErrorActionPreference = "Continue"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Fetcher = Join-Path $ProjectRoot "scripts\fetch_jv_realtime_odds.ps1"
$PowerShell32 = "$env:WINDIR\SysWOW64\WindowsPowerShell\v1.0\powershell.exe"

if (-not (Test-Path $PowerShell32)) {
    throw "32-bit PowerShell was not found: $PowerShell32"
}
if (-not (Test-Path $Fetcher)) {
    throw "Fetcher was not found: $Fetcher"
}

$NormalizedRaceKeys = @()
foreach ($raceKey in $RaceKeys) {
    $NormalizedRaceKeys += @($raceKey -split "[,\s]+" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}
$RaceKeys = @($NormalizedRaceKeys | Select-Object -Unique)

for ($i = 1; $i -le $Count; $i++) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$stamp] JV realtime odds snapshot loop $i/$Count"
    foreach ($raceKey in $RaceKeys) {
        foreach ($betType in $BetTypes) {
            Write-Host "  race=$raceKey bet=$betType"
            & $PowerShell32 -NoProfile -ExecutionPolicy Bypass -File $Fetcher `
                -RaceKey $raceKey `
                -BetType $betType `
                -Sid $Sid `
                -OutputDir $OutputDir
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "Fetch failed race=$raceKey bet=$betType exit=$LASTEXITCODE"
            }
        }
    }
    if ($i -lt $Count) {
        Start-Sleep -Seconds $IntervalSeconds
    }
}
