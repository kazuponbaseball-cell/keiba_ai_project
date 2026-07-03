param(
    [Parameter(Mandatory = $true)]
    [string[]]$RaceKeys,

    [ValidateSet("T-10", "T-5", "T-3", "final_check", "manual")]
    [string]$DecisionLabel = "manual",

    [string]$Sid = "",

    [ValidateSet("all", "win_place_frame", "umaren", "wide", "umatan", "trio", "trifecta")]
    [string[]]$BetTypes = @("win_place_frame", "wide", "umaren"),

    [int]$IntervalSeconds = 1,

    [int]$Count = 1,

    [string]$TicketsCsv = "outputs\analysis\roi_mode_stake_sizing_v1\stake_sized_ticket_profiles.csv",

    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$SnapshotLoop = Join-Path $ProjectRoot "scripts\run_jv_realtime_odds_snapshot_loop.ps1"
$NormalizePair = Join-Path $ProjectRoot "scripts\normalize_jv_realtime_pair_odds.py"
$NormalizeSingle = Join-Path $ProjectRoot "scripts\normalize_live_single_odds.py"
$AppendTimeline = Join-Path $ProjectRoot "scripts\append_live_odds_timeline.py"
$BuildSingleFeatures = Join-Path $ProjectRoot "scripts\build_odds_timeline_features.py"
$EvaluateSlippage = Join-Path $ProjectRoot "scripts\evaluate_odds_timeline_slippage.py"
$ApplyRuntimeDecisions = Join-Path $ProjectRoot "scripts\apply_runtime_odds_decision_rules.py"
$ApplyPriorityContext = Join-Path $ProjectRoot "scripts\apply_priority_context_factor_overlay.py"
$ApplyStandardStaking = Join-Path $ProjectRoot "scripts\apply_standard_staking_plan.py"
$ApplyLiveSafety = Join-Path $ProjectRoot "scripts\apply_live_runtime_safety_overlay.py"
$ApplyPriorityA = Join-Path $ProjectRoot "scripts\apply_priority_a_ticket_type_overlay.py"
$EvaluatePriorityB = Join-Path $ProjectRoot "scripts\evaluate_priority_b_context_factors.py"
$ApplyEquipmentOverlay = Join-Path $ProjectRoot "scripts\apply_equipment_pair_blinker_overlay.py"
$AddOperationalExplainability = Join-Path $ProjectRoot "scripts\add_operational_explainability.py"
$BuildDashboard = Join-Path $ProjectRoot "scripts\build_keiba_dashboard_html.py"
$ExportNetkeibaPlan = Join-Path $ProjectRoot "scripts\export_netkeiba_bet_plan.py"

. (Join-Path $ProjectRoot "scripts\Resolve-JvSid.ps1")
$sidResolution = Resolve-JvSid -Sid $Sid
$Sid = $sidResolution.Sid
if ([string]::IsNullOrWhiteSpace($Sid)) {
    throw "JV SID is required. Pass -Sid, set environment variable JV_SID, or configure JRA-VAN Data Lab servicekey."
}

if (-not (Test-Path $PythonExe)) {
    throw "Python executable was not found: $PythonExe"
}

Set-Location $ProjectRoot

Write-Host "Fetching JV realtime odds snapshots..."
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $SnapshotLoop `
    -RaceKeys $RaceKeys `
    -BetTypes $BetTypes `
    -Sid $Sid `
    -IntervalSeconds $IntervalSeconds `
    -Count $Count

Write-Host "Normalizing pair odds..."
& $PythonExe $NormalizePair `
    --output-csv "data\processed\live_odds\realtime_pair_odds_latest.csv"

Write-Host "Normalizing win/place odds..."
& $PythonExe $NormalizeSingle `
    --output-csv "data\processed\live_odds\realtime_single_odds_latest.csv"

Write-Host "Appending to odds timelines..."
& $PythonExe $AppendTimeline `
    --decision-label $DecisionLabel

Write-Host "Building single-odds timeline features..."
& $PythonExe $BuildSingleFeatures `
    --timeline-csv "data\processed\live_odds\realtime_single_odds_timeline.csv" `
    --output-csv "data\processed\live_odds\realtime_single_odds_timeline_features.csv"

Write-Host "Evaluating live-vs-final odds slippage..."
& $PythonExe $EvaluateSlippage `
    --tickets-csv $TicketsCsv

Write-Host "Applying runtime buy/reduce/wait/skip decisions..."
& $PythonExe $ApplyRuntimeDecisions `
    --tickets-csv "outputs\analysis\min_odds_ticket_prob_gate_v1\min_odds_annotated_ticket_profiles.csv" `
    --no-proxy-when-missing

Write-Host "Applying priority context factor overlay..."
& $PythonExe $ApplyPriorityContext `
    --tickets-csv "outputs\analysis\runtime_odds_decision_rules_v1\runtime_ticket_decisions.csv"

Write-Host "Applying standard 2x staking plan..."
& $PythonExe $ApplyStandardStaking `
    --tickets-csv "outputs\analysis\priority_context_factor_overlay_v1\priority_context_selected_tickets.csv"

Write-Host "Applying live runtime safety overlay..."
& $PythonExe $ApplyLiveSafety `
    --tickets-csv "outputs\analysis\standard_staking_plan_v1\standard_staked_tickets.csv"

Write-Host "Applying optimized A overlay..."
& $PythonExe $ApplyPriorityA `
    --tickets-csv "outputs\analysis\live_runtime_safety_overlay_v1\live_safety_overlaid_tickets.csv" `
    --output-dir "outputs\analysis\live_priority_a_overlay_v1" `
    --mode "boost_umaren_a_top_115"

Write-Host "Applying optimized B overlay..."
& $PythonExe $EvaluatePriorityB `
    --tickets-csv "outputs\analysis\live_priority_a_overlay_v1\priority_a_ticket_type_overlaid_tickets.csv" `
    --output-dir "outputs\analysis\live_priority_ab_overlay_v1"

Write-Host "Applying equipment risk overlay..."
& $PythonExe $ApplyEquipmentOverlay `
    --tickets-csv "outputs\analysis\live_priority_ab_overlay_v1\boost_high_b_110_tickets.csv" `
    --output-dir "outputs\analysis\live_optimized_strategy_v1" `
    --mode "reduce_pair_first_blinker_50"

Write-Host "Adding dashboard explanations..."
& $PythonExe $AddOperationalExplainability `
    --tickets-csv "outputs\analysis\live_optimized_strategy_v1\reduce_pair_first_blinker_50_tickets.csv" `
    --output-csv "outputs\analysis\live_optimized_strategy_v1\standard_explained_tickets.csv" `
    --mode-label "standard"

Write-Host "Rebuilding dashboard with optimized strategy overlay..."
& $PythonExe $BuildDashboard `
    --tickets-csv "outputs\analysis\live_optimized_strategy_v1\standard_explained_tickets.csv" `
    --body-weight-csv "data\processed\live_body_weight\body_weight_latest.csv" `
    --output-html "outputs\ui\keiba_dashboard_aggressive_stake.html"

Write-Host "Exporting netkeiba bet plan..."
& $PythonExe $ExportNetkeibaPlan `
    --tickets-csv "outputs\analysis\live_optimized_strategy_v1\standard_explained_tickets.csv"

Write-Host "Done."
Write-Host "Pair timeline: data\processed\live_odds\realtime_pair_odds_timeline.csv"
Write-Host "Single timeline: data\processed\live_odds\realtime_single_odds_timeline.csv"
Write-Host "Slippage summary: outputs\analysis\live_odds_slippage_v1\slippage_summary.csv"
Write-Host "Runtime decisions: outputs\analysis\runtime_odds_decision_rules_v1\runtime_ticket_decisions.csv"
Write-Host "Priority context tickets: outputs\analysis\priority_context_factor_overlay_v1\priority_context_selected_tickets.csv"
Write-Host "Standard staked tickets: outputs\analysis\standard_staking_plan_v1\standard_staked_tickets.csv"
Write-Host "Live safety tickets: outputs\analysis\live_runtime_safety_overlay_v1\live_safety_overlaid_tickets.csv"
Write-Host "Optimized strategy tickets: outputs\analysis\live_optimized_strategy_v1\standard_explained_tickets.csv"
Write-Host "netkeiba plan: outputs\integration\netkeiba_bet_plan\netkeiba_bet_plan.csv"
Write-Host "Dashboard: outputs\ui\keiba_dashboard_aggressive_stake.html"
