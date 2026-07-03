param(
    [string]$TargetDate = "",
    [string]$Today = "",
    [switch]$RequireTargetReady,
    [int]$MaxRaces = 72,
    [string]$OutputHtml = "outputs\ui\keiba_preday_dashboard_official.html",
    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $ProjectRoot

if (-not $TargetDate) {
    $TargetDate = (Get-Date).AddDays(1).ToString("yyyyMMdd")
}
if (-not $Today) {
    $Today = Get-Date -Format "yyyy-MM-dd"
}

Write-Host "[1/5] TARGET preflight"
$preflightArgs = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass",
    "-File", "scripts\run_target_data_preflight.ps1",
    "-TargetDate", $TargetDate
)
if ($RequireTargetReady) {
    $preflightArgs += "-RequireReady"
}
powershell.exe @preflightArgs
if ($LASTEXITCODE -ne 0 -and $RequireTargetReady) {
    throw "TARGET preflight failed for $TargetDate."
}

Write-Host "[2/5] Sync latest TARGET entry export"
$oldErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$syncOutput = & $PythonExe -m src.pipelines.sync_target_exports --mode entry --run-import 2>&1
$syncExit = $LASTEXITCODE
$ErrorActionPreference = $oldErrorActionPreference
if ($syncExit -ne 0) {
    $message = "TARGET entry sync was not ready. No current TARGET entry export matched the expected columns."
    if ($RequireTargetReady) {
        throw "$message $($syncOutput -join ' ')"
    }
    Write-Warning $message
} elseif ($syncOutput) {
    $syncOutput | Write-Host
}

Write-Host "[3/5] Validate/build weekly inference dataset"
$oldErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
$buildOutput = & $PythonExe -m src.pipelines.build_weekly_inference_dataset 2>&1
$buildExit = $LASTEXITCODE
$ErrorActionPreference = $oldErrorActionPreference
if ($buildExit -ne 0) {
    $message = "Weekly inference dataset is not ready. The current TARGET snapshot is missing required columns or target dates."
    if ($RequireTargetReady) {
        throw "$message $($buildOutput -join ' ')"
    }
    Write-Warning $message
} elseif ($buildOutput) {
    $buildOutput | Write-Host
}

Write-Host "[4/5] Run inference if TARGET snapshot is ready"
$snapshotPath = "data\datasets\inference\weekly\entry_snapshot.csv"
$readyDates = & $PythonExe scripts\check_entry_snapshot_ready.py --entry-csv $snapshotPath --target-date $TargetDate
$targetSnapshotReady = $LASTEXITCODE -eq 0
if ($targetSnapshotReady) {
    try {
        & $PythonExe -m src.pipelines.run_daily_inference
        if ($LASTEXITCODE -ne 0 -and $RequireTargetReady) {
            throw "Daily inference failed."
        }
    } catch {
        if ($RequireTargetReady) {
            throw
        }
        Write-Warning "Daily inference skipped/not ready: $($_.Exception.Message)"
    }
} else {
    $message = "TARGET snapshot does not contain $TargetDate. dates=$readyDates"
    if ($RequireTargetReady) {
        throw $message
    }
    Write-Warning $message
}

Write-Host "[5/5] Build official-only preday dashboard"
& $PythonExe scripts\build_preday_dashboard_html.py `
    --today $Today `
    --official-only `
    --output-html $OutputHtml `
    --max-races $MaxRaces
if ($LASTEXITCODE -ne 0) {
    throw "Official-only dashboard build failed."
}

Write-Host "Official-only preday update complete."
Write-Host "Dashboard: $OutputHtml"
