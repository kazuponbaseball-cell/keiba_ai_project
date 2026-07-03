param(
    [int]$Port = 8766,
    [string]$Username = "keiba",
    [string]$Password = "",
    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ToolsDir = Join-Path $ProjectRoot "tools"
$Cloudflared = Join-Path $ToolsDir "cloudflared.exe"
$AuthServer = Join-Path $ProjectRoot "scripts\serve_dashboard_with_basic_auth.py"
$DashboardPath = "outputs/ui/live_odds_dashboard.html"

if (-not (Test-Path $PythonExe)) {
    throw "Python executable was not found: $PythonExe"
}

if ([string]::IsNullOrWhiteSpace($Password)) {
    $Password = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 14 | ForEach-Object {[char]$_})
}

if (-not (Test-Path $Cloudflared)) {
    throw "cloudflared.exe was not found: $Cloudflared. Run scripts\install_cloudflared.ps1 first."
}

Set-Location $ProjectRoot

Write-Host "Starting password-protected local dashboard server..."
$serverArgs = @(
    $AuthServer,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--username", $Username,
    "--password", $Password,
    "--directory", $ProjectRoot
)
$server = Start-Process -FilePath $PythonExe -ArgumentList $serverArgs -PassThru -WindowStyle Hidden
Start-Sleep -Seconds 2

Write-Host ""
Write-Host "Dashboard path:"
Write-Host "  /$DashboardPath"
Write-Host ""
Write-Host "Login:"
Write-Host "  user: $Username"
Write-Host "  pass: $Password"
Write-Host ""
Write-Host "Starting Cloudflare temporary tunnel..."
Write-Host "Open the generated https://*.trycloudflare.com URL on your phone, then append:"
Write-Host "  /$DashboardPath"
Write-Host ""
Write-Host "Press Ctrl+C in this terminal to stop the public URL."

try {
    & $Cloudflared tunnel --url "http://127.0.0.1:$Port"
}
finally {
    if ($server -and -not $server.HasExited) {
        Stop-Process -Id $server.Id -Force
    }
}
