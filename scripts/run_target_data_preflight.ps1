param(
    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    [string]$ProjectRoot = "",
    [string]$TargetRoot = "C:\Users\kazup\Data Lab",
    [string]$TargetDate = "",
    [int]$LookbackDays = 45,
    [switch]$RequireReady
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}

Set-Location -LiteralPath $ProjectRoot

if ([string]::IsNullOrWhiteSpace($TargetDate)) {
    $TargetDate = (Get-Date).AddDays(-1).ToString("yyyyMMdd")
}

$startDate = ([datetime]::ParseExact($TargetDate, "yyyyMMdd", $null)).AddDays(-1 * $LookbackDays).ToString("yyyyMMdd")
$endDate = (Get-Date).ToString("yyyyMMdd")

Write-Host "[preflight] TARGET data availability"
Write-Host "  target date: $TargetDate"
Write-Host "  target root: $TargetRoot"

$argsList = @(
    "scripts\diagnose_target_data_availability.py",
    "--target-root", $TargetRoot,
    "--start-date", $startDate,
    "--end-date", $endDate,
    "--target-date", $TargetDate
)

if ($RequireReady) {
    $argsList += "--require-target-ready"
}

& $PythonExe @argsList
$preflightExitCode = $LASTEXITCODE

Write-Host "[preflight] report: outputs\analysis\target_data_availability_report.md"
Write-Host "[preflight] csv: outputs\analysis\target_data_availability.csv"

if ($preflightExitCode -ne 0) {
    exit $preflightExitCode
}
