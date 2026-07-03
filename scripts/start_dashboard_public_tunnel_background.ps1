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
$RuntimeDir = Join-Path $ProjectRoot "outputs\runtime"
$DashboardPath = "kazu"
$InfoJson = Join-Path $RuntimeDir "public_dashboard_tunnel.json"
$ServerOut = Join-Path $RuntimeDir "public_dashboard_server_stdout.log"
$ServerErr = Join-Path $RuntimeDir "public_dashboard_server_stderr.log"
$TunnelOut = Join-Path $RuntimeDir "cloudflared_stdout.log"
$TunnelErr = Join-Path $RuntimeDir "cloudflared_stderr.log"

if (-not (Test-Path $PythonExe)) {
    throw "Python executable was not found: $PythonExe"
}
if (-not (Test-Path $Cloudflared)) {
    throw "cloudflared.exe was not found: $Cloudflared. Run scripts\install_cloudflared.ps1 first."
}
if ([string]::IsNullOrWhiteSpace($Password)) {
    $Password = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 14 | ForEach-Object {[char]$_})
}

# Some PowerShell hosts expose both Path and PATH in the process environment.
# Start-Process can fail on that duplicate, so normalize it before spawning helpers.
$processPath = [Environment]::GetEnvironmentVariable("Path", "Process")
if (-not [string]::IsNullOrWhiteSpace($processPath)) {
    [Environment]::SetEnvironmentVariable("PATH", $null, "Process")
    [Environment]::SetEnvironmentVariable("Path", $processPath, "Process")
}

New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
Remove-Item -Force -ErrorAction SilentlyContinue $TunnelOut, $TunnelErr, $ServerOut, $ServerErr

Set-Location $ProjectRoot

$serverArgs = @(
    $AuthServer,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--username", $Username,
    "--password", $Password,
    "--directory", $ProjectRoot
)
$server = Start-Process -FilePath $PythonExe -ArgumentList $serverArgs -PassThru -WindowStyle Hidden -RedirectStandardOutput $ServerOut -RedirectStandardError $ServerErr
Start-Sleep -Seconds 2

$tunnelArgs = @("tunnel", "--url", "http://127.0.0.1:$Port")
$tunnel = Start-Process -FilePath $Cloudflared -ArgumentList $tunnelArgs -PassThru -WindowStyle Hidden -RedirectStandardOutput $TunnelOut -RedirectStandardError $TunnelErr

$publicUrl = ""
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Seconds 1
    $text = ""
    if (Test-Path $TunnelOut) { $text += Get-Content $TunnelOut -Raw -ErrorAction SilentlyContinue }
    if (Test-Path $TunnelErr) { $text += Get-Content $TunnelErr -Raw -ErrorAction SilentlyContinue }
    $match = [regex]::Match($text, "https://(?!api\.)[-a-zA-Z0-9]+\.trycloudflare\.com")
    if ($match.Success) {
        $publicUrl = $match.Value
        break
    }
    if ($tunnel.HasExited) {
        break
    }
}

if ([string]::IsNullOrWhiteSpace($publicUrl)) {
    if ($server -and -not $server.HasExited) { Stop-Process -Id $server.Id -Force }
    if ($tunnel -and -not $tunnel.HasExited) { Stop-Process -Id $tunnel.Id -Force }
    throw "Could not detect the public Cloudflare Tunnel URL. See $TunnelErr"
}

$dashboardUrl = "$publicUrl/$DashboardPath"
$info = [ordered]@{
    public_url = $publicUrl
    dashboard_url = $dashboardUrl
    username = $Username
    password = $Password
    server_pid = $server.Id
    tunnel_pid = $tunnel.Id
    started_at = (Get-Date).ToString("s")
    server_stdout = $ServerOut
    server_stderr = $ServerErr
    tunnel_stdout = $TunnelOut
    tunnel_stderr = $TunnelErr
}
$info | ConvertTo-Json | Set-Content -Path $InfoJson -Encoding UTF8

Write-Host "Public dashboard tunnel is ready."
Write-Host "URL:"
Write-Host "  $dashboardUrl"
Write-Host "Login:"
Write-Host "  user: $Username"
Write-Host "  pass: $Password"
Write-Host "Info:"
Write-Host "  $InfoJson"
Write-Host "Stop:"
Write-Host "  powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\stop_dashboard_public_tunnel.ps1"
