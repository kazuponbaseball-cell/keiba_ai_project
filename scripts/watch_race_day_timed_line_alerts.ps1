param(
    [string]$Date = (Get-Date -Format "yyyyMMdd"),
    [int]$IntervalSeconds = 60,
    [int]$MaxRuns = 0,
    [switch]$SkipOddsFetch,
    [switch]$SkipTrackFetch,
    [switch]$IncludeResultFetch,
    [switch]$SendIfConfigured,
    [switch]$Force,
    [switch]$DisableFinal3,
    [switch]$DisableDailySummary,
    [string]$DashboardUrl = "",
    [ValidateSet("buy", "buy_or_watch", "all")]
    [string]$PreRacePolicy = "buy",
    [string]$LogPath = "outputs\notifications\race_day_line\watch_log.txt",
    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
}

Set-Location -LiteralPath $ProjectRoot

$userLineToken = [Environment]::GetEnvironmentVariable("LINE_CHANNEL_ACCESS_TOKEN", "User")
$userLineTo = [Environment]::GetEnvironmentVariable("LINE_USER_ID", "User")
if ([string]::IsNullOrWhiteSpace($env:LINE_CHANNEL_ACCESS_TOKEN) -and -not [string]::IsNullOrWhiteSpace($userLineToken)) {
    $env:LINE_CHANNEL_ACCESS_TOKEN = $userLineToken
}
if ([string]::IsNullOrWhiteSpace($env:LINE_USER_ID) -and -not [string]::IsNullOrWhiteSpace($userLineTo)) {
    $env:LINE_USER_ID = $userLineTo
}

$UpdateScript = Join-Path $ProjectRoot "scripts\run_current_strongest_line_update.ps1"
$NotifyScript = Join-Path $ProjectRoot "scripts\send_race_day_line_notifications.py"
$ResolvedLogPath = if ([System.IO.Path]::IsPathRooted($LogPath)) { $LogPath } else { Join-Path $ProjectRoot $LogPath }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ResolvedLogPath) | Out-Null

function Write-WatchLog {
    param([string]$Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Write-Host $line
    Add-Content -LiteralPath $ResolvedLogPath -Value $line -Encoding UTF8
}

if (-not (Test-Path -LiteralPath $UpdateScript)) {
    throw "Update script was not found: $UpdateScript"
}
if (-not (Test-Path -LiteralPath $NotifyScript)) {
    throw "Notification script was not found: $NotifyScript"
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable was not found: $PythonExe"
}

$run = 0
Write-WatchLog "Started timed LINE watcher. interval=$IntervalSeconds maxRuns=$MaxRuns date=$Date"
Write-WatchLog "Pre-race notification policy: $PreRacePolicy"
if ($DisableFinal3) {
    Write-WatchLog "Final 3-minute notifications: disabled"
}
if ($DisableDailySummary) {
    Write-WatchLog "Daily summary notifications: disabled"
}

while ($true) {
    $run += 1
    try {
        Write-WatchLog "Run $run update start"
        $updateArgs = @(
            "-File", $UpdateScript,
            "-Date", $Date,
            "-PythonExe", $PythonExe
        )
        if ($SkipOddsFetch) {
            $updateArgs += "-SkipOddsFetch"
        }
        if ($SkipTrackFetch) {
            $updateArgs += "-SkipTrackFetch"
        }
        if (-not $IncludeResultFetch) {
            $updateArgs += "-SkipResultFetch"
        }
        $updateOutput = @(powershell.exe -NoProfile -ExecutionPolicy Bypass @updateArgs)
        foreach ($line in $updateOutput) {
            Write-WatchLog "Run $run update: $line"
        }

        Write-WatchLog "Run $run notify start"
        $notifyArgs = @(
            $NotifyScript,
            "--date", $Date,
            "--pre-race-policy", $PreRacePolicy
        )
        if ($SendIfConfigured) {
            $notifyArgs += "--send-if-configured"
        }
        if ($Force) {
            $notifyArgs += "--force"
        }
        if ($DisableFinal3) {
            $notifyArgs += "--disable-final3"
        }
        if ($DisableDailySummary) {
            $notifyArgs += "--disable-daily-summary"
        }
        if (-not [string]::IsNullOrWhiteSpace($DashboardUrl)) {
            $notifyArgs += "--dashboard-url"
            $notifyArgs += $DashboardUrl
        }
        $notifyOutput = @(& $PythonExe @notifyArgs)
        foreach ($line in $notifyOutput) {
            Write-WatchLog "Run $run notify: $line"
        }
        Write-WatchLog "Run $run complete"
    } catch {
        Write-WatchLog "Run $run ERROR: $($_.Exception.Message)"
    }

    if ($MaxRuns -gt 0 -and $run -ge $MaxRuns) {
        Write-WatchLog "Finished timed LINE watcher. runs=$run"
        break
    }
    Start-Sleep -Seconds $IntervalSeconds
}
