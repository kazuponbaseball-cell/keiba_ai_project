param(
    [string]$TaskName = "KeibaCurrentStrongestLineWatcher",
    [string]$StartTime = "09:00",
    [int]$IntervalSeconds = 60,
    [int]$ExecutionHours = 9,
    [switch]$Register,
    [switch]$Unregister,
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
}

$Watcher = Join-Path $ProjectRoot "scripts\watch_current_strongest_line_alerts.ps1"
if (-not (Test-Path -LiteralPath $Watcher)) {
    throw "Watcher script was not found: $Watcher"
}

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Unregistered scheduled task: $TaskName"
    exit 0
}

if (-not $Register) {
    Write-Host "This script registers a Windows scheduled task for automatic LINE watcher startup."
    Write-Host "Preview:"
    Write-Host "  TaskName: $TaskName"
    Write-Host "  StartTime: $StartTime"
    Write-Host "  IntervalSeconds: $IntervalSeconds"
    Write-Host "  ExecutionHours: $ExecutionHours"
    Write-Host ""
    Write-Host "Register:"
    Write-Host "  powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\register_current_strongest_line_watcher_task.ps1 -Register"
    Write-Host ""
    Write-Host "Unregister:"
    Write-Host "  powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\register_current_strongest_line_watcher_task.ps1 -Unregister"
    exit 0
}

$actionArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$Watcher`"",
    "-SendIfConfigured",
    "-IntervalSeconds", "$IntervalSeconds"
) -join " "

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
    -Description "Refresh Keiba strongest tickets and send LINE when the ticket set changes." `
    -Force | Out-Null

Write-Host "Registered scheduled task: $TaskName"
Write-Host "It starts daily at $StartTime and stops after about $ExecutionHours hours."
