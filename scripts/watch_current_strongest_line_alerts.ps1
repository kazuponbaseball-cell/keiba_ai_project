param(
    [string]$Date = (Get-Date -Format "yyyyMMdd"),
    [string[]]$RaceKeys = @(),
    [int]$IntervalSeconds = 60,
    [int]$MaxRuns = 0,
    [switch]$SkipOddsFetch,
    [switch]$Notify,
    [switch]$SendIfConfigured,
    [switch]$ForceFirstNotify,
    [string]$LogPath = "outputs\analysis\current_strongest_line_update\watch_log.txt",
    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
}

Set-Location -LiteralPath $ProjectRoot

$Runner = Join-Path $ProjectRoot "scripts\run_current_strongest_line_update.ps1"
$ResolvedLogPath = if ([System.IO.Path]::IsPathRooted($LogPath)) { $LogPath } else { Join-Path $ProjectRoot $LogPath }
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $ResolvedLogPath) | Out-Null

function Write-WatchLog {
    param([string]$Message)
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $Message"
    Write-Host $line
    Add-Content -LiteralPath $ResolvedLogPath -Value $line -Encoding UTF8
}

if (-not (Test-Path -LiteralPath $Runner)) {
    throw "Runner was not found: $Runner"
}

$run = 0
Write-WatchLog "Started current strongest LINE watcher. interval=$IntervalSeconds maxRuns=$MaxRuns date=$Date"

while ($true) {
    $run += 1
    $argsList = @(
        "-File", $Runner,
        "-Date", $Date,
        "-PythonExe", $PythonExe
    )
    if ($RaceKeys.Count -gt 0) {
        $argsList += "-RaceKeys"
        $argsList += $RaceKeys
    }
    if ($SkipOddsFetch) {
        $argsList += "-SkipOddsFetch"
    }
    if ($Notify) {
        $argsList += "-Notify"
    }
    if ($SendIfConfigured) {
        $argsList += "-SendIfConfigured"
    }
    if ($ForceFirstNotify -and $run -eq 1) {
        $argsList += "-ForceNotify"
    }

    try {
        Write-WatchLog "Run $run start"
        $output = @(powershell.exe -NoProfile -ExecutionPolicy Bypass @argsList)
        foreach ($line in $output) {
            Write-WatchLog "Run $run output: $line"
        }
        Write-WatchLog "Run $run complete"
    } catch {
        Write-WatchLog "Run $run ERROR: $($_.Exception.Message)"
    }

    if ($MaxRuns -gt 0 -and $run -ge $MaxRuns) {
        Write-WatchLog "Finished current strongest LINE watcher. runs=$run"
        break
    }
    Start-Sleep -Seconds $IntervalSeconds
}
