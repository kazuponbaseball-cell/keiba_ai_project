param(
    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    [string]$ProjectRoot = "",
    [string]$Today = "",
    [ValidateSet("preday", "buyable", "skip", "all", "final")]
    [string]$Mode = "preday",
    [string]$DashboardUrl = "",
    [string]$BodyWeightCsv = "",
    [string]$TicketsCsv = "",
    [int]$MaxRaces = 8,
    [switch]$RequireBodyWeight,
    [switch]$Send,
    [switch]$SendIfConfigured
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
}

Set-Location -LiteralPath $ProjectRoot

$userLineToken = [Environment]::GetEnvironmentVariable("LINE_CHANNEL_ACCESS_TOKEN", "User")
$userLineTo = [Environment]::GetEnvironmentVariable("LINE_USER_ID", "User")
if ([string]::IsNullOrWhiteSpace($env:LINE_CHANNEL_ACCESS_TOKEN) -and -not [string]::IsNullOrWhiteSpace($userLineToken)) {
    $env:LINE_CHANNEL_ACCESS_TOKEN = $userLineToken
}
if ([string]::IsNullOrWhiteSpace($env:LINE_USER_ID) -and -not [string]::IsNullOrWhiteSpace($userLineTo)) {
    $env:LINE_USER_ID = $userLineTo
}

$argsList = @(
    "scripts\send_line_keiba_alert.py",
    "--mode", $Mode,
    "--max-races", "$MaxRaces"
)

if (-not [string]::IsNullOrWhiteSpace($Today)) {
    $argsList += @("--today", $Today)
}
if (-not [string]::IsNullOrWhiteSpace($DashboardUrl)) {
    $argsList += @("--dashboard-url", $DashboardUrl)
}
if (-not [string]::IsNullOrWhiteSpace($BodyWeightCsv)) {
    $argsList += @("--body-weight-csv", $BodyWeightCsv)
}
if (-not [string]::IsNullOrWhiteSpace($TicketsCsv)) {
    $argsList += @("--tickets-csv", $TicketsCsv)
}
if ($RequireBodyWeight) {
    $argsList += "--require-body-weight"
}
$shouldSend = $Send
if ($SendIfConfigured) {
    $shouldSend = -not [string]::IsNullOrWhiteSpace($env:LINE_CHANNEL_ACCESS_TOKEN) -and -not [string]::IsNullOrWhiteSpace($env:LINE_USER_ID)
    if (-not $shouldSend) {
        Write-Warning "LINE credentials are not configured. Running dry-run only."
    }
}
if ($shouldSend) {
    $argsList += "--send"
}

& $PythonExe @argsList
