param(
    [string]$Date = (Get-Date -Format "yyyyMMdd"),
    [int]$IntervalSeconds = 90,
    [string]$UntilTime = "17:10",
    [switch]$Notify,
    [switch]$SendIfConfigured,
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
}

$Watcher = Join-Path $ProjectRoot "scripts\watch_track_condition_changes.ps1"
if (-not (Test-Path -LiteralPath $Watcher)) {
    throw "Track condition watcher was not found: $Watcher"
}

$logDir = Join-Path $ProjectRoot "outputs\analysis\live_track_conditions"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdout = Join-Path $logDir "track_condition_watcher_background_stdout.log"
$stderr = Join-Path $logDir "track_condition_watcher_background_stderr.log"

$argsList = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$Watcher`"",
    "-Date", $Date,
    "-IntervalSeconds", [string]$IntervalSeconds,
    "-UntilTime", $UntilTime
)
if ($Notify) {
    $argsList += "-Notify"
}
if ($SendIfConfigured) {
    $argsList += "-SendIfConfigured"
}

$process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $argsList `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -PassThru

$summary = [ordered]@{
    started_at = (Get-Date).ToString("s")
    pid = $process.Id
    date = $Date
    interval_seconds = $IntervalSeconds
    until_time = $UntilTime
    notify = [bool]$Notify
    send_if_configured = [bool]$SendIfConfigured
    stdout = $stdout
    stderr = $stderr
}

$summaryPath = Join-Path $logDir "track_condition_watcher_background_summary.json"
$summary | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
$summary | ConvertTo-Json -Depth 4
