param(
    [string]$TicketsCsv = "outputs\analysis\current_strongest_runtime_v1\selected_after_live_safety.csv",
    [string]$DashboardUrl = "",
    [int]$MaxRaces = 8,
    [switch]$Send,
    [switch]$SendIfConfigured,
    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
}

Set-Location -LiteralPath $ProjectRoot

$LineAlert = Join-Path $ProjectRoot "scripts\send_line_keiba_alert.ps1"
if (-not (Test-Path $LineAlert)) {
    throw "LINE alert wrapper was not found: $LineAlert"
}

function Resolve-DashboardUrl {
    if (-not [string]::IsNullOrWhiteSpace($DashboardUrl)) {
        return $DashboardUrl
    }

    $publicInfoPath = Join-Path $ProjectRoot "outputs\runtime\public_dashboard_tunnel.json"
    if (Test-Path $publicInfoPath) {
        try {
            $publicInfo = Get-Content -Path $publicInfoPath -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($publicInfo.dashboard_url) {
                return $publicInfo.dashboard_url
            }
            if ($publicInfo.public_url) {
                return "$($publicInfo.public_url)/kazu"
            }
        } catch {
            Write-Warning "Could not read public tunnel info: $($_.Exception.Message)"
        }
    }

    $dashboardPath = Join-Path $ProjectRoot "outputs\ui\live_odds_dashboard.html"
    return ([System.Uri]$dashboardPath).AbsoluteUri
}

$resolvedUrl = Resolve-DashboardUrl
$argsList = @(
    "-File", $LineAlert,
    "-Mode", "final",
    "-TicketsCsv", $TicketsCsv,
    "-DashboardUrl", $resolvedUrl,
    "-MaxRaces", "$MaxRaces",
    "-PythonExe", $PythonExe
)

if ($Send) {
    $argsList += "-Send"
}
if ($SendIfConfigured) {
    $argsList += "-SendIfConfigured"
}

powershell.exe -NoProfile -ExecutionPolicy Bypass @argsList
