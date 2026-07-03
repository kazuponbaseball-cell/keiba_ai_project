param(
    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    [string]$ProjectRoot = "",
    [string]$TargetDate = "",
    [switch]$SkipPreflight,
    [switch]$RequireReady
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}

Set-Location -LiteralPath $ProjectRoot

if (-not $SkipPreflight) {
    Write-Host "[0/4] Check TARGET data availability"
    $preflightArgs = @(
        "-File", "scripts\run_target_data_preflight.ps1",
        "-PythonExe", $PythonExe,
        "-ProjectRoot", $ProjectRoot
    )
    if (-not [string]::IsNullOrWhiteSpace($TargetDate)) {
        $preflightArgs += @("-TargetDate", $TargetDate)
    }
    if ($RequireReady) {
        $preflightArgs += "-RequireReady"
    }
    powershell -ExecutionPolicy Bypass @preflightArgs
}

Write-Host "[1/3] Import latest TARGET entry CSV"
& $PythonExe -m src.pipelines.import_latest_target_entry

Write-Host "[2/3] Validate/build weekly inference dataset"
& $PythonExe -m src.pipelines.build_weekly_inference_dataset

Write-Host "[3/3] Run daily inference"
& $PythonExe -m src.pipelines.run_daily_inference
