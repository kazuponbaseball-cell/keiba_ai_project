param(
    [int]$Port = 8765,
    [string]$Bind = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

Set-Location $projectRoot
Write-Host "Starting dashboard server..."
Write-Host "URL: http://${Bind}:$Port/outputs/ui/keiba_dashboard.html"
& $python -m http.server $Port --bind $Bind
