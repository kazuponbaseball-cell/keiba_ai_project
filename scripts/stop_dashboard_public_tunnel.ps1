$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$InfoJson = Join-Path $ProjectRoot "outputs\runtime\public_dashboard_tunnel.json"

if (-not (Test-Path $InfoJson)) {
    Write-Host "No public tunnel info found: $InfoJson"
    exit 0
}

$info = Get-Content $InfoJson -Raw | ConvertFrom-Json
foreach ($pidValue in @($info.server_pid, $info.tunnel_pid)) {
    if ($null -eq $pidValue) { continue }
    $proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
    if ($proc) {
        Stop-Process -Id $pidValue -Force
        Write-Host "Stopped process $pidValue"
    }
}

Remove-Item -Force -ErrorAction SilentlyContinue $InfoJson
Write-Host "Public dashboard tunnel stopped."
