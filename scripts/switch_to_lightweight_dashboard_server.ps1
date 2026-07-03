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

$listeners = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue
foreach ($listener in $listeners) {
    $pidValue = [int]$listener.OwningProcess
    $p = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
    if (-not $p) { continue }
    if ($p.ProcessName -notmatch "^pythonw?$") {
        throw "Port $Port is used by $($p.ProcessName) PID=$pidValue. Refusing to stop a non-Python process."
    }
    Stop-Process -Id $pidValue -Force
    Write-Host "Stopped previous dashboard server $($p.ProcessName) PID=$pidValue"
}

$startScript = Join-Path $ProjectRoot "scripts\start_lightweight_dashboard_server.ps1"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $startScript `
    -Port $Port `
    -Username $Username `
    -Password $Password `
    -RuntimeRoot $RuntimeRoot `
    -PythonExe $PythonExe `
    -ProjectRoot $ProjectRoot
