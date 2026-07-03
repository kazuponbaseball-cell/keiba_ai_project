param(
    [string]$RuntimeRoot = "C:\keiba_ai_runtime",
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
}

$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$RuntimeRoot = [System.IO.Path]::GetFullPath($RuntimeRoot)

$files = @(
    "outputs\ui\live_odds_dashboard.html",
    "outputs\ui\live_odds_dashboard.summary.json",
    "outputs\runtime\current_dashboard_inputs.json",
    "outputs\analysis\win5_runtime\win5_plan.json"
)

$copied = @()
$missing = @()

foreach ($rel in $files) {
    $source = Join-Path $ProjectRoot $rel
    $target = Join-Path $RuntimeRoot $rel
    if (Test-Path -LiteralPath $source) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
        Copy-Item -LiteralPath $source -Destination $target -Force
        $copied += $rel
    } else {
        $missing += $rel
    }
}

$stateDir = Join-Path $RuntimeRoot "state"
New-Item -ItemType Directory -Force -Path $stateDir | Out-Null

$status = [ordered]@{
    ok = $true
    project_root = $ProjectRoot
    runtime_root = $RuntimeRoot
    copied = $copied
    missing = $missing
    synced_at = (Get-Date).ToString("s")
}

$statusPath = Join-Path $stateDir "runtime_mirror_status.json"
$status | ConvertTo-Json -Depth 5 | Set-Content -Path $statusPath -Encoding UTF8

Write-Host "Dashboard runtime mirror synced."
Write-Host "Runtime:"
Write-Host "  $RuntimeRoot"
Write-Host "Status:"
Write-Host "  $statusPath"
