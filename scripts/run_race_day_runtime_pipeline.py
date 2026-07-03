from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _project_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _run_step(name: str, cmd: list[str]) -> dict:
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    return {
        "name": name,
        "command": cmd,
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }


def _read_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _selected_metrics(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {"tickets": 0, "races": 0, "stake_yen": 0.0}
    df = pd.read_csv(path, dtype={"race_id": str}, low_memory=False)
    stake_col = "runtime_stake_yen" if "runtime_stake_yen" in df.columns else "stake_yen"
    return {
        "tickets": int(len(df)),
        "races": int(df["race_id"].nunique()) if "race_id" in df.columns else 0,
        "stake_yen": float(pd.to_numeric(df.get(stake_col), errors="coerce").fillna(0.0).sum()) if stake_col in df.columns else 0.0,
        "actions": df["runtime_action"].value_counts().to_dict() if "runtime_action" in df.columns else {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Race-day one-command runtime pipeline: pair probability calibration, live odds normalization, strict decision, safety overlay, dashboard, and netkeiba handoff."
    )
    parser.add_argument("--base-tickets-csv", default="outputs/analysis/robust_expansion_runtime_ready_v1/standard_plus_robust_runtime_ready_tickets.csv")
    parser.add_argument("--output-root", default="outputs/analysis/race_day_runtime_pipeline_v1")
    parser.add_argument("--dashboard-html", default="outputs/ui/keiba_dashboard_runtime.html")
    parser.add_argument("--scored-csv", default="outputs/analysis/risk_models_v1/investment_features_with_risk_models.csv")
    parser.add_argument("--pair-live-csv", default="data/processed/live_odds/realtime_pair_odds_latest.csv")
    parser.add_argument("--single-live-csv", default="data/processed/live_odds/realtime_single_odds_latest.csv")
    parser.add_argument("--body-weight-csv", default="data/processed/live_body_weight/body_weight_latest.csv")
    parser.add_argument("--manual-pair-odds-csv", default="")
    parser.add_argument("--manual-single-odds-csv", default="")
    parser.add_argument("--normalize-live-odds", action="store_true")
    parser.add_argument("--proxy-when-missing", action="store_true", help="Historical/debug mode. Real operation should leave this off.")
    parser.add_argument("--skip-pair-calibration", action="store_true")
    parser.add_argument("--skip-live-safety", action="store_true")
    parser.add_argument(
        "--mcs-pbo-policy",
        default="",
        help="Optional policy such as mcs_full_margin095_s0304_skip03119, mcs_full_margin095_s0304, reduce_wide_win_50, or mcs_margin_s0078. Empty disables this overlay.",
    )
    parser.add_argument("--skip-dashboard", action="store_true")
    parser.add_argument("--skip-netkeiba-export", action="store_true")
    parser.add_argument("--max-dashboard-races", type=int, default=180)
    args = parser.parse_args()

    out_root = _project_path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    steps: list[dict] = []

    pair_live_csv = args.pair_live_csv
    single_live_csv = args.single_live_csv

    if args.normalize_live_odds:
        pair_cmd = [
            sys.executable,
            "scripts/normalize_jv_realtime_pair_odds.py",
            "--output-csv",
            pair_live_csv,
            "--summary-json",
            str(out_root / "pair_odds_normalization_summary.json"),
        ]
        if args.manual_pair_odds_csv:
            pair_cmd.extend(["--manual-csv", args.manual_pair_odds_csv])
        steps.append(_run_step("normalize_pair_live_odds", pair_cmd))

        single_cmd = [
            sys.executable,
            "scripts/normalize_live_single_odds.py",
            "--output-csv",
            single_live_csv,
            "--summary-json",
            str(out_root / "single_odds_normalization_summary.json"),
        ]
        if args.manual_single_odds_csv:
            single_cmd.extend(["--manual-csv", args.manual_single_odds_csv])
        steps.append(_run_step("normalize_single_live_odds", single_cmd))

    tickets_for_runtime = args.base_tickets_csv
    if not args.skip_pair_calibration:
        pair_dir = out_root / "pair_probability_runtime"
        cmd = [
            sys.executable,
            "scripts/apply_pair_probability_calibration_to_runtime.py",
            "--tickets-csv",
            args.base_tickets_csv,
            "--output-dir",
            str(pair_dir),
        ]
        if args.proxy_when_missing:
            cmd.append("--evaluate-proxy")
        steps.append(_run_step("apply_pair_probability_calibration", cmd))
        tickets_for_runtime = str(pair_dir / "pair_calibrated_runtime_tickets.csv")

    runtime_dir = out_root / ("runtime_proxy" if args.proxy_when_missing else "runtime_strict")
    runtime_cmd = [
        sys.executable,
        "scripts/apply_runtime_odds_decision_rules.py",
        "--tickets-csv",
        tickets_for_runtime,
        "--pair-live-csv",
        pair_live_csv,
        "--single-live-csv",
        single_live_csv,
        "--output-dir",
        str(runtime_dir),
    ]
    if not args.proxy_when_missing:
        runtime_cmd.append("--no-proxy-when-missing")
    steps.append(_run_step("apply_runtime_odds_decisions", runtime_cmd))

    final_tickets_csv = runtime_dir / "runtime_ticket_decisions.csv"
    if not args.skip_live_safety:
        safety_dir = out_root / "live_safety"
        safety_cmd = [
            sys.executable,
            "scripts/apply_live_runtime_safety_overlay.py",
            "--tickets-csv",
            str(final_tickets_csv),
            "--body-weight-csv",
            args.body_weight_csv,
            "--output-dir",
            str(safety_dir),
        ]
        steps.append(_run_step("apply_live_safety_overlay", safety_cmd))
        final_tickets_csv = safety_dir / "live_safety_overlaid_tickets.csv"

    if args.mcs_pbo_policy:
        overlay_dir = out_root / "mcs_pbo_overlay"
        overlay_cmd = [
            sys.executable,
            "scripts/apply_mcs_pbo_survivor_overlay.py",
            "--tickets-csv",
            str(final_tickets_csv),
            "--output-dir",
            str(overlay_dir),
            "--selected-policy",
            args.mcs_pbo_policy,
        ]
        steps.append(_run_step("apply_mcs_pbo_survivor_overlay", overlay_cmd))
        final_tickets_csv = overlay_dir / "recommended_all_tickets.csv"

    bet_plan_dir = out_root / "netkeiba_bet_plan"
    if not args.skip_netkeiba_export:
        steps.append(
            _run_step(
                "export_netkeiba_bet_plan",
                [
                    sys.executable,
                    "scripts/export_netkeiba_bet_plan.py",
                    "--tickets-csv",
                    str(final_tickets_csv),
                    "--output-dir",
                    str(bet_plan_dir),
                ],
            )
        )

    dashboard_html = _project_path(args.dashboard_html)
    if not args.skip_dashboard:
        steps.append(
            _run_step(
                "build_dashboard",
                [
                    sys.executable,
                    "scripts/build_keiba_dashboard_html.py",
                    "--scored-csv",
                    args.scored_csv,
                    "--tickets-csv",
                    str(final_tickets_csv),
                    "--live-pair-odds-csv",
                    pair_live_csv,
                    "--live-single-odds-csv",
                    single_live_csv,
                    "--body-weight-csv",
                    args.body_weight_csv,
                    "--output-html",
                    str(dashboard_html),
                    "--max-races",
                    str(args.max_dashboard_races),
                ],
            )
        )

    selected_csv = final_tickets_csv.parent / (
        "runtime_selected_tickets.csv" if final_tickets_csv.name == "runtime_ticket_decisions.csv" else "selected_after_live_safety.csv"
    )
    if final_tickets_csv.name != "runtime_ticket_decisions.csv":
        final_df = pd.read_csv(final_tickets_csv, dtype={"race_id": str}, low_memory=False)
        selected = final_df[pd.to_numeric(final_df.get("runtime_stake_yen"), errors="coerce").fillna(0.0).gt(0)].copy()
        selected.to_csv(selected_csv, index=False, encoding="utf-8-sig")

    payload = {
        "output_root": str(out_root),
        "mode": "proxy_debug" if args.proxy_when_missing else "strict_live",
        "base_tickets_csv": args.base_tickets_csv,
        "tickets_for_runtime": tickets_for_runtime,
        "final_tickets_csv": str(final_tickets_csv),
        "selected_csv": str(selected_csv),
        "selected_metrics": _selected_metrics(selected_csv),
        "runtime_summary": _read_summary(runtime_dir / "summary.json"),
        "dashboard_html": str(dashboard_html) if not args.skip_dashboard else "",
        "netkeiba_plan_dir": str(bet_plan_dir) if not args.skip_netkeiba_export else "",
        "steps": steps,
        "failed_steps": [s for s in steps if s["returncode"] != 0],
        "note": "Strict live mode intentionally waits when live odds are absent. Use --proxy-when-missing only for historical/debug checks.",
    }
    (out_root / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["failed_steps"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
