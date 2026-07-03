param(
    [int]$Port = 8766,
    [string]$Username = "keiba",
    [string]$Password = "kazu",
    [string]$RuntimeRoot = "C:\keiba_ai_runtime",
    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
}

$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)
$Server = Join-Path $ProjectRoot "scripts\serve_dashboard_with_basic_auth.py"
$SyncScript = Join-Path $ProjectRoot "scripts\sync_dashboard_runtime_mirror.ps1"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable was not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $Server)) {
    throw "Dashboard server script was not found: $Server"
}

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $SyncScript -RuntimeRoot $RuntimeRoot -ProjectRoot $ProjectRoot

$logDir = Join-Path $RuntimeRoot "logs"
$stateDir = Join-Path $RuntimeRoot "state"
New-Item -ItemType Directory -Force -Path $logDir, $stateDir | Out-Null

$existing = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
if ($existing) {
    $pids = ($existing | Select-Object -ExpandProperty OwningProcess -Unique) -join ", "
    throw "Port $Port is already in use by PID(s): $pids"
}

$stdout = Join-Path $logDir "dashboard_server_stdout.log"
$stderr = Join-Path $logDir "dashboard_server_stderr.log"
Remove-Item -Force -ErrorAction SilentlyContinue $stdout, $stderr

$args = @(
    $Server,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--username", $Username,
    "--password", $Password,
    "--directory", $RuntimeRoot,
    "--project-root", $ProjectRoot,
    "--quiet"
)

$process = Start-Process -FilePath $PythonExe -ArgumentList $args -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
Start-Sleep -Seconds 2

if ($process.HasExited) {
    $err = if (Test-Path $stderr) { Get-Content $stderr -Raw -ErrorAction SilentlyContinue } else { "" }
    throw "Lightweight dashboard server exited immediately. $err"
}

$info = [ordered]@{
    ok = $true
    url = "http://127.0.0.1:$Port/outputs/ui/live_odds_dashboard.html"
    short_url = "http://127.0.0.1:$Port/kazu"
    username = $Username
    password = $Password
    pid = $process.Id
    project_root = $ProjectRoot
    runtime_root = $RuntimeRoot
    started_at = (Get-Date).ToString("s")
    stdout = $stdout
    stderr = $stderr
}

$infoPath = Join-Path $stateDir "dashboard_server.json"
$info | ConvertTo-Json -Depth 5 | Set-Content -Path $infoPath -Encoding UTF8

Write-Host "Lightweight dashboard server is ready."
Write-Host "URL:"
Write-Host "  $($info.url)"
Write-Host "Short:"
Write-Host "  $($info.short_url)"
Write-Host "Login:"
Write-Host "  user: $Username"
Write-Host "  pass: $Password"
Write-Host "Info:"
Write-Host "  $infoPath"
