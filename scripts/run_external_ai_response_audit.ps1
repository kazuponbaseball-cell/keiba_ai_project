param(
    [string]$PythonExe = "C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

function Invoke-Step {
    param(
        [string]$Label,
        [scriptblock]$Command
    )
    Write-Host "==> $Label"
    & $Command
}

Invoke-Step "Low-probability high-odds guard" {
    & $PythonExe scripts\analyze_low_prob_high_odds_guard.py `
        --tickets-csv outputs\analysis\mcs_pbo_runtime_overlay_v4_operational_gates_default\recommended_runtime_tickets.csv `
        --output-dir outputs\analysis\low_prob_high_odds_guard_v1 `
        --min-races 220
}

Invoke-Step "Champion freeze manifest" {
    & $PythonExe scripts\freeze_champion_strategy_manifest.py `
        --output-dir outputs\analysis\champion_strategy_freeze_v1
}

Invoke-Step "Final odds survival dataset" {
    & $PythonExe scripts\build_final_odds_survival_dataset.py `
        --candidates-csv outputs\analysis\current_strongest_runtime_v1\current_strongest_all_candidates.csv `
        --pair-timeline-csv data\processed\live_odds\realtime_pair_odds_timeline.csv `
        --output-dir outputs\analysis\final_odds_survival_model_v1
}

Invoke-Step "Candidate rejection ledger" {
    & $PythonExe scripts\build_candidate_rejection_ledger.py `
        --candidates-csv outputs\analysis\current_strongest_runtime_v1\current_strongest_all_candidates.csv `
        --selected-csv outputs\analysis\current_strongest_runtime_v1\selected_after_live_safety.csv `
        --output-dir outputs\analysis\candidate_rejection_ledger_v1
}

Invoke-Step "Candidate rejection ledger without post-time lock for analysis" {
    & $PythonExe scripts\build_candidate_rejection_ledger.py `
        --candidates-csv outputs\analysis\current_strongest_runtime_v1\current_strongest_all_candidates.csv `
        --selected-csv outputs\analysis\current_strongest_runtime_v1\selected_after_live_safety.csv `
        --output-dir outputs\analysis\candidate_rejection_ledger_prepost_sim_v1 `
        --ignore-post-time-lock
}

Invoke-Step "Shadow Challenger candidates" {
    & $PythonExe scripts\build_shadow_challenger_candidates.py `
        --ledger-csv outputs\analysis\candidate_rejection_ledger_prepost_sim_v1\candidate_rejection_ledger.csv `
        --survival-csv outputs\analysis\final_odds_survival_model_v1\final_odds_survival_dataset.csv `
        --pnl-detail-csv outputs\analysis\current_live_pnl\current_live_pnl_detail.csv `
        --output-dir outputs\analysis\shadow_challenger_candidates_v1
}

Invoke-Step "Gate dropout audit" {
    & $PythonExe scripts\build_gate_dropout_audit.py `
        --ledger-csv outputs\analysis\candidate_rejection_ledger_prepost_sim_v1\candidate_rejection_ledger.csv `
        --pnl-detail-csv outputs\analysis\current_live_pnl\current_live_pnl_detail.csv `
        --output-dir outputs\analysis\gate_dropout_audit_v1
}

Invoke-Step "Shadow promotion readiness" {
    & $PythonExe scripts\build_shadow_promotion_readiness.py `
        --shadow-csv outputs\analysis\shadow_challenger_candidates_v1\shadow_challenger_candidates.csv `
        --output-dir outputs\analysis\shadow_promotion_readiness_v1
}

Write-Host "External AI response audit completed."
