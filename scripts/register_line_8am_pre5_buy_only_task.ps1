param(
    [string]$TaskName = "KeibaLine8AMPre5BuyOnly",
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
}

$RegisterScript = Join-Path $ProjectRoot "scripts\register_race_day_timed_line_task.ps1"
if (-not (Test-Path -LiteralPath $RegisterScript)) {
    throw "Register script was not found: $RegisterScript"
}

& powershell.exe `
    -NoProfile `
    -ExecutionPolicy Bypass `
    -File $RegisterScript `
    -TaskName $TaskName `
    -StartTime "08:00" `
    -IntervalSeconds 60 `
    -ExecutionHours 10 `
    -PreRacePolicy "buy" `
    -DisableFinal3 `
    -DisableDailySummary `
    -Register `
    -ProjectRoot $ProjectRoot
