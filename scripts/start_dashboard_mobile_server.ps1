param(
    [int]$Port = 8765,
    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

if (-not (Test-Path $PythonExe)) {
    throw "Python executable was not found: $PythonExe"
}

Set-Location $ProjectRoot

$ips = Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object {
        $_.IPAddress -notlike "127.*" -and
        $_.IPAddress -notlike "169.254.*" -and
        $_.PrefixOrigin -ne "WellKnown"
    } |
    Select-Object -ExpandProperty IPAddress -Unique

Write-Host "Starting dashboard server for phone/LAN access..."
Write-Host "Local PC URL:"
Write-Host "  http://127.0.0.1:$Port/outputs/ui/keiba_dashboard_aggressive_stake.html"
Write-Host ""
Write-Host "Phone URLs on the same Wi-Fi:"
foreach ($ip in $ips) {
    Write-Host "  http://$ip`:$Port/outputs/ui/keiba_dashboard_aggressive_stake.html"
}
Write-Host ""
Write-Host "If the phone cannot open it, allow Python through Windows Firewall for private networks."

& $PythonExe -m http.server $Port --bind 0.0.0.0
