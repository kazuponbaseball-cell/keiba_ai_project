param(
    [string]$TaskName = "KeibaRaceDayTimedLineWatcher",
    [string]$StartTime = "08:00",
    [int]$IntervalSeconds = 60,
    [int]$ExecutionHours = 10,
    [ValidateSet("buy", "buy_or_watch", "all")]
    [string]$PreRacePolicy = "buy",
    [switch]$DisableFinal3,
    [switch]$DisableDailySummary,
    [switch]$Register,
    [switch]$Unregister,
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
}

$Watcher = Join-Path $ProjectRoot "scripts\watch_race_day_timed_line_alerts.ps1"
if (-not (Test-Path -LiteralPath $Watcher)) {
    throw "Watcher script was not found: $Watcher"
}

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Unregistered scheduled task: $TaskName"
    exit 0
}

if (-not $Register) {
    Write-Host "This script registers the race-day timed LINE notification task."
    Write-Host "Preview:"
    Write-Host "  TaskName: $TaskName"
    Write-Host "  StartTime: $StartTime"
    Write-Host "  IntervalSeconds: $IntervalSeconds"
    Write-Host "  ExecutionHours: $ExecutionHours"
    Write-Host "  PreRacePolicy: $PreRacePolicy"
    Write-Host "  DisableFinal3: $([bool]$DisableFinal3)"
    Write-Host "  DisableDailySummary: $([bool]$DisableDailySummary)"
    Write-Host ""
    Write-Host "Register:"
    Write-Host "  powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\register_race_day_timed_line_task.ps1 -Register"
    Write-Host ""
    Write-Host "Unregister:"
    Write-Host "  powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\register_race_day_timed_line_task.ps1 -Unregister"
    exit 0
}

$actionArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$Watcher`"",
    "-SendIfConfigured",
    "-IntervalSeconds", "$IntervalSeconds",
    "-PreRacePolicy", "$PreRacePolicy"
) -join " "

if ($DisableFinal3) {
    $actionArgs += " -DisableFinal3"
}
if ($DisableDailySummary) {
    $actionArgs += " -DisableDailySummary"
}

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument $actionArgs `
    -WorkingDirectory $ProjectRoot

$trigger = New-ScheduledTaskTrigger -Daily -At $StartTime
$settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours $ExecutionHours) `
    -StartWhenAvailable

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Refresh Keiba AI tickets and send timed LINE notices." `
    -Force | Out-Null

Write-Host "Registered scheduled task: $TaskName"
Write-Host "It starts daily at $StartTime and stops after about $ExecutionHours hours."
