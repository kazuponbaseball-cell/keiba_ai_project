param(
    [int]$Port = 8767,
    [string]$Username = "keiba",
    [Parameter(Mandatory = $true)]
    [string]$Password,
    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
}

Set-Location -LiteralPath $ProjectRoot

$Server = Join-Path $ProjectRoot "scripts\serve_dashboard_with_basic_auth.py"
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable was not found: $PythonExe"
}
if (-not (Test-Path -LiteralPath $Server)) {
    throw "Public dashboard server script was not found: $Server"
}

& $PythonExe `
    $Server `
    --host 127.0.0.1 `
    --port $Port `
    --username $Username `
    --password $Password `
    --directory $ProjectRoot `
    --quiet
