$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$ToolsDir = Join-Path $ProjectRoot "tools"
$Cloudflared = Join-Path $ToolsDir "cloudflared.exe"
$Url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"

New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null

Write-Host "Downloading cloudflared..."
Invoke-WebRequest -Uri $Url -OutFile $Cloudflared

if (-not (Test-Path $Cloudflared)) {
    throw "Download failed: $Cloudflared"
}

Write-Host "Installed:"
Write-Host "  $Cloudflared"
& $Cloudflared --version
