param(
    [string]$Date = (Get-Date -Format "yyyyMMdd"),
    [int[]]$OffsetsMinutes = @(10, 5, 3),
    [int]$PollSeconds = 15,
    [int]$GroupWindowSeconds = 25,
    [switch]$SendIfConfigured,
    [switch]$ResetState,
    [string]$EntryCsv = "",
    [string]$PredictionCsv = "",
    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
}

$Runner = Join-Path $ProjectRoot "scripts\start_current_strongest_timed_odds_snapshots.ps1"
$RuntimeDir = Join-Path $ProjectRoot "outputs\runtime"
$InfoJson = Join-Path $RuntimeDir "current_strongest_timed_snapshots_background.json"
$StdoutLog = Join-Path $RuntimeDir "current_strongest_timed_snapshots_stdout.log"
$StderrLog = Join-Path $RuntimeDir "current_strongest_timed_snapshots_stderr.log"
$StateJson = Join-Path $ProjectRoot "data\processed\live_odds\current_strongest_timed_snapshot_state.json"

if (-not (Test-Path -LiteralPath $Runner)) {
    throw "Timed snapshot runner was not found: $Runner"
}
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable was not found: $PythonExe"
}

New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null

if ($ResetState -and (Test-Path -LiteralPath $StateJson)) {
    Remove-Item -LiteralPath $StateJson -Force
}

if (Test-Path -LiteralPath $InfoJson) {
    try {
        $old = Get-Content -LiteralPath $InfoJson -Raw -Encoding UTF8 | ConvertFrom-Json
        $oldProc = Get-Process -Id $old.pid -ErrorAction SilentlyContinue
        if ($oldProc -and -not $oldProc.HasExited) {
            Write-Host "Timed snapshot watcher already running. pid=$($old.pid)"
            Write-Host "Info: $InfoJson"
            exit 0
        }
    } catch {
        Write-Warning "Could not inspect previous timed snapshot watcher info: $($_.Exception.Message)"
    }
}

Remove-Item -Force -ErrorAction SilentlyContinue $StdoutLog, $StderrLog

$processPath = [Environment]::GetEnvironmentVariable("Path", "Process")
$processPATH = [Environment]::GetEnvironmentVariable("PATH", "Process")
if (-not [string]::IsNullOrWhiteSpace($processPath) -and -not [string]::IsNullOrWhiteSpace($processPATH)) {
    # Windows environment names are case-insensitive, but some parent shells can
    # expose both Path and PATH. PowerShell Start-Process then fails while copying
    # the environment dictionary, so keep one canonical process-level value.
    [Environment]::SetEnvironmentVariable("PATH", $null, "Process")
    [Environment]::SetEnvironmentVariable("Path", $processPath, "Process")
}

$argList = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $Runner,
    "-Date", $Date,
    "-PollSeconds", "$PollSeconds",
    "-GroupWindowSeconds", "$GroupWindowSeconds",
    "-SkipResultFetch",
    "-PythonExe", $PythonExe,
    "-ProjectRoot", $ProjectRoot
)

# The runner already defaults to T-10/T-5/T-3. Passing an int-array through
# Start-Process is surprisingly easy to mis-bind, so keep the runner default
# for the race-day launcher.
if (-not [string]::IsNullOrWhiteSpace($EntryCsv)) {
    $argList += @("-EntryCsv", $EntryCsv)
}
if (-not [string]::IsNullOrWhiteSpace($PredictionCsv)) {
    $argList += @("-PredictionCsv", $PredictionCsv)
}
if ($SendIfConfigured) {
    $argList += "-SendIfConfigured"
}

$proc = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList $argList `
    -PassThru `
    -WindowStyle Hidden `
    -WorkingDirectory $ProjectRoot

$info = [ordered]@{
    pid = $proc.Id
    date = $Date
    offsets_minutes = $OffsetsMinutes
    poll_seconds = $PollSeconds
    group_window_seconds = $GroupWindowSeconds
    send_if_configured = [bool]$SendIfConfigured
    started_at = (Get-Date).ToString("s")
    stdout = $StdoutLog
    stderr = $StderrLog
    runner = $Runner
    state_json = $StateJson
}
$info | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $InfoJson -Encoding UTF8

Write-Host "Current strongest timed odds snapshots started."
Write-Host "PID: $($proc.Id)"
Write-Host "Info: $InfoJson"
Write-Host "Log: outputs\analysis\current_strongest_line_update\timed_snapshot_log.txt"
