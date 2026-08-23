param(
    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    [string]$ProjectRoot = "",
    [string]$TargetDate = "20260823",
    [int[]]$Budgets = @(5000, 10000, 25000),
    [switch]$SkipTargetRefresh
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}

Set-Location -LiteralPath $ProjectRoot

Write-Host "[WIN5 MVP] date=$TargetDate"

if (-not $SkipTargetRefresh) {
    Write-Host "[1/3] Refresh TARGET entry + inference"
    powershell -ExecutionPolicy Bypass -File "scripts\run_target_weekly_update.ps1" `
        -PythonExe $PythonExe `
        -ProjectRoot $ProjectRoot `
        -TargetDate $TargetDate
    if ($LASTEXITCODE -ne 0) {
        throw "TARGET weekly update failed with exit code $LASTEXITCODE"
    }
} else {
    Write-Host "[1/3] TARGET refresh skipped"
}

Write-Host "[2/3] Resolve latest prediction"
$prediction = Get-ChildItem -LiteralPath "outputs\predictions" -Filter "baseline_predictions_*.csv" -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1

if ($null -eq $prediction) {
    throw "No outputs\predictions\baseline_predictions_*.csv found"
}

Write-Host "  prediction=$($prediction.FullName)"

Write-Host "[3/3] Build same-day WIN5 report"
$argsList = @(
    "scripts\build_win5_today_mvp.py",
    "--date", $TargetDate,
    "--prediction-csv", $prediction.FullName,
    "--budgets"
)
$argsList += ($Budgets | ForEach-Object { [string]$_ })

& $PythonExe @argsList
if ($LASTEXITCODE -ne 0) {
    throw "WIN5 MVP report failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Outputs:"
Write-Host "  outputs\analysis\win5_runtime\win5_today_report_$TargetDate.md"
Write-Host "  outputs\analysis\win5_runtime\win5_candidates_$TargetDate.csv"
Write-Host "  outputs\analysis\win5_runtime\win5_plan_$TargetDate.json"
