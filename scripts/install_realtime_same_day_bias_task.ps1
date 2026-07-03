param(
    [string]$TaskName = "KeibaAIRealtimeSameDayBias",
    [string]$ProjectRoot = "",
    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    [string]$TargetRoot = "C:\Users\kazup\Data Lab",
    [string]$StartAt = "09:00",
    [int]$IntervalMinutes = 3,
    [int]$DurationHours = 8,
    [switch]$RunPreflight
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}

$refreshScript = Join-Path $ProjectRoot "scripts\run_realtime_same_day_bias_refresh.ps1"
if (-not (Test-Path -LiteralPath $refreshScript)) {
    throw "Refresh script not found: $refreshScript"
}

$argsList = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$refreshScript`"",
    "-ProjectRoot", "`"$ProjectRoot`"",
    "-PythonExe", "`"$PythonExe`"",
    "-TargetRoot", "`"$TargetRoot`"",
    "-Once"
)

if ($RunPreflight) {
    $argsList += "-RunPreflight"
}

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument ($argsList -join " ") `
    -WorkingDirectory $ProjectRoot

$startTime = [datetime]::ParseExact($StartAt, "HH:mm", $null)
$startDateTime = (Get-Date).Date.Add($startTime.TimeOfDay)
if ($startDateTime -lt (Get-Date)) {
    $startDateTime = $startDateTime.AddDays(1)
}

$trigger = New-ScheduledTaskTrigger `
    -Once `
    -At $startDateTime `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) `
    -RepetitionDuration (New-TimeSpan -Hours $DurationHours)

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Refresh Keiba AI same-day bias features and predictions from TARGET/normalized data." `
    -Force | Out-Null

Write-Host "Registered task: $TaskName"
Write-Host "Schedule: from $($startDateTime.ToString('yyyy-MM-dd HH:mm')), every $IntervalMinutes minutes for $DurationHours hours"
Write-Host "Refresh script: $refreshScript"
