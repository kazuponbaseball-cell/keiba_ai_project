param(
    [string]$DecisionLabel = "manual_post_weekend_review",
    [string[]]$PnlDetailCsvs = @(
        "outputs\analysis\current_live_pnl\current_live_pnl_detail.csv"
    ),
    [string]$SnapshotsCsv = "data\processed\live_decision_snapshots\current_strongest_decision_snapshots.csv",
    [string]$PairTimelineCsv = "data\processed\live_odds\realtime_pair_odds_timeline.csv",
    [string]$OutputRoot = "outputs\analysis\post_day_shadow_review",
    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
}

Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable was not found: $PythonExe"
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runDir = Join-Path $ProjectRoot (Join-Path $OutputRoot $stamp)
$riskDir = Join-Path $runDir "decision_snapshot_risk_coverage"
$survivalDir = Join-Path $runDir "odds_value_survival"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

function Resolve-ProjectPath {
    param([string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return $Path
    }
    return (Join-Path $ProjectRoot $Path)
}

function Invoke-Checked {
    param(
        [string]$Label,
        [scriptblock]$Command
    )
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Label"
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

$riskScript = Join-Path $ProjectRoot "scripts\analyze_decision_snapshot_risk_coverage.py"
$survivalScript = Join-Path $ProjectRoot "scripts\analyze_odds_value_survival.py"

$resolvedPnl = @()
foreach ($pnlItem in $PnlDetailCsvs) {
    foreach ($pnl in ($pnlItem -split ",")) {
        if ([string]::IsNullOrWhiteSpace($pnl)) {
            continue
        }
        $path = Resolve-ProjectPath $pnl.Trim()
        if (Test-Path -LiteralPath $path) {
            $resolvedPnl += $path
        } else {
            Write-Warning "PnL detail CSV was not found and will be skipped: $path"
        }
    }
}

if ($resolvedPnl.Count -eq 0) {
    throw "No PnL detail CSVs were found."
}

$riskOk = $false
if ((Test-Path -LiteralPath (Resolve-ProjectPath $SnapshotsCsv)) -and (Test-Path -LiteralPath $riskScript)) {
    Invoke-Checked -Label "Analyze decision snapshot risk coverage" -Command {
        & $PythonExe $riskScript `
            --snapshots-csv $SnapshotsCsv `
            --pnl-detail-csv $resolvedPnl[-1] `
            --decision-label $DecisionLabel `
            --output-dir $riskDir
    }
    $riskOk = $true
} else {
    Write-Warning "Skipping decision snapshot risk coverage; snapshots or script were not found."
}

$survivalArgs = @($survivalScript, "--pnl-detail-csv") + $resolvedPnl + @(
    "--pair-timeline-csv", $PairTimelineCsv,
    "--output-dir", $survivalDir
)
Invoke-Checked -Label "Analyze T-5/T-3 odds value survival" -Command {
    & $PythonExe @survivalArgs
}

$summary = [ordered]@{
    generated_at = (Get-Date).ToString("s")
    decision_label = $DecisionLabel
    output_dir = $runDir
    pnl_detail_csvs = $resolvedPnl
    risk_coverage_ran = $riskOk
    risk_coverage_dir = $riskDir
    odds_value_survival_dir = $survivalDir
}

$summaryPath = Join-Path $runDir "summary.json"
$summary | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $summaryPath -Encoding UTF8
$summary | ConvertTo-Json -Depth 6
