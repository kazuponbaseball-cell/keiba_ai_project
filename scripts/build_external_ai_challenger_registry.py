from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"race_id": str}, low_memory=False)


def num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def text(frame: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=str)
    return frame[col].astype("string").fillna(default).astype(str)


def add_challenger_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    reasons = text(out, "rejection_reasons")
    single_reason = text(out, "single_rejection_reason")
    rejection_count = num(out, "rejection_reason_count", 99)
    margin = num(out, "min_odds_margin_ratio", 0.0)
    score = num(out, "strongest_current_score", 0.0)
    live_odds = num(out, "live_odds", np.nan)
    champion_buy = text(out, "champion_selected_buy").str.lower().isin({"true", "1", "1.0"})

    c1 = champion_buy | reasons.eq("PASS")
    c2 = rejection_count.eq(1) & (single_reason.eq("SCORE_FAIL") | reasons.eq("SCORE_FAIL"))
    c3 = rejection_count.eq(1) & (
        single_reason.isin(["MARGIN_FAIL", "MARGIN_BELOW"])
        | reasons.isin(["MARGIN_FAIL", "MARGIN_BELOW"])
        | (reasons.str.contains("MARGIN", regex=False) & ~reasons.str.contains("|", regex=False))
    )

    out["external_ai_challenger_id"] = np.select(
        [c1, c2, c3],
        [
            "C1_CHAMPION_FULL_PASS_BY_PURCHASE_TIME",
            "C2_SCORE_ONLY_NEAR_MISS",
            "C3_MARGIN_ONLY_NEAR_MISS",
        ],
        default="NOT_REGISTERED",
    )
    out["external_ai_challenger_registered"] = out["external_ai_challenger_id"].ne("NOT_REGISTERED")
    out["registry_note"] = np.select(
        [c1, c2, c3],
        [
            "Original Champion gates passed; do not loosen. Use as baseline/pass-through only.",
            "Only SCORE gate failed. Shadow-only until multi-day OOS and fixed T-5/T-3 snapshots support promotion.",
            "Only margin gate failed. Shadow-only; currently valuable only if final odds survival proves conservative edge.",
        ],
        default="Multiple fail or hard stop; not a Challenger candidate.",
    )
    out["registry_safety_stop"] = (
        live_odds.gt(120)
        | text(out, "rejection_reasons").str.contains("ODDS_TOO_HIGH", regex=False)
        | text(out, "rejection_reasons").str.contains("LIVE_ODDS_MISSING", regex=False)
    )
    out["score_gap_to_champion"] = (0.86 - score).clip(lower=0.0)
    out["margin_gap_to_champion"] = (0.95 - margin).clip(lower=0.0)
    return out


def metric(group: pd.DataFrame) -> dict[str, Any]:
    stake = len(group) * 100.0
    ret_col = None
    for col in ["proxy_return_yen", "return_yen", "runtime_return_yen", "eval_return_yen"]:
        if col in group.columns:
            ret_col = col
            break
    returns = float(num(group, ret_col, 0.0).sum()) if ret_col else None
    hit_col = None
    for col in ["proxy_hit", "hit", "runtime_hit"]:
        if col in group.columns:
            hit_col = col
            break
    hits = int(num(group, hit_col, 0.0).gt(0).sum()) if hit_col else None
    return {
        "candidates": int(len(group)),
        "races": int(group["race_id"].nunique()) if "race_id" in group.columns else 0,
        "avg_live_odds": float(num(group, "live_odds", np.nan).replace([np.inf, -np.inf], np.nan).mean()),
        "avg_score": float(num(group, "strongest_current_score", np.nan).replace([np.inf, -np.inf], np.nan).mean()),
        "avg_margin": float(num(group, "min_odds_margin_ratio", np.nan).replace([np.inf, -np.inf], np.nan).mean()),
        "stake_yen_flat100": stake,
        "return_yen_flat100": returns,
        "hits": hits,
        "roi_pct_flat100_proxy": float(returns / stake * 100.0) if returns is not None and stake else None,
        "performance_source": ret_col or "not_in_ledger_use_shadow_promotion_readiness",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Register exactly the external-AI approved Challenger candidate families.")
    parser.add_argument(
        "--ledger-csv",
        default="outputs/analysis/candidate_rejection_ledger_prepost_sim_v1/candidate_rejection_ledger.csv",
    )
    parser.add_argument("--output-dir", default="outputs/analysis/external_ai_challenger_registry_v1")
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = read_csv(project_path(args.ledger_csv))
    if ledger.empty:
        payload = {"generated_at": datetime.now().isoformat(timespec="seconds"), "rows": 0, "status": "FAIL"}
        (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    registry = add_challenger_labels(ledger)
    registered = registry[registry["external_ai_challenger_registered"]].copy()
    registry.to_csv(out_dir / "candidate_registry_all_rows.csv", index=False, encoding="utf-8-sig")
    registered.to_csv(out_dir / "registered_challenger_candidates.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for cid in [
        "C1_CHAMPION_FULL_PASS_BY_PURCHASE_TIME",
        "C2_SCORE_ONLY_NEAR_MISS",
        "C3_MARGIN_ONLY_NEAR_MISS",
    ]:
        group = registry[registry["external_ai_challenger_id"].eq(cid)]
        row = {"challenger_id": cid}
        row.update(metric(group))
        row["promotion_allowed_now"] = False
        row["promotion_policy"] = "shadow_only_until_fixed_time_multi_day_oos"
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "challenger_registry_summary.csv", index=False, encoding="utf-8-sig")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "ledger_csv": args.ledger_csv,
        "rows": int(len(registry)),
        "registered_rows": int(len(registered)),
        "summary": summary_df.to_dict(orient="records"),
        "policy": {
            "allowed_candidate_families": [
                "C1_CHAMPION_FULL_PASS_BY_PURCHASE_TIME",
                "C2_SCORE_ONLY_NEAR_MISS",
                "C3_MARGIN_ONLY_NEAR_MISS",
            ],
            "champion_changed": False,
            "live_buy_change": False,
            "promotion_allowed_now": False,
            "reason": "External-AI guidance requires immutable T-5/T-3 snapshots and multi-day OOS before any Challenger promotion.",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(payload), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
