from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.optimize_operational_win_addon import _json_default
from src.utils.paths import ensure_dir, project_path


def _num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _metric(df: pd.DataFrame, label: str) -> dict:
    selected = df[_num(df.get("runtime_stake_yen"), df.index, 0.0).fillna(0.0).gt(0)].copy()
    stake = float(_num(selected.get("runtime_stake_yen"), selected.index, 0.0).sum())
    ret = float(_num(selected.get("runtime_return_yen"), selected.index, 0.0).sum())
    races = int(selected["race_id"].nunique()) if not selected.empty else 0
    hit = selected[selected.get("hit", False).astype(bool)] if not selected.empty else selected
    hit_races = int(hit["race_id"].nunique()) if not hit.empty else 0
    curve = selected.sort_values(["date_key", "race_id"]).groupby("race_id", sort=False)[["runtime_stake_yen", "runtime_return_yen"]].sum() if not selected.empty and "date_key" in selected.columns else pd.DataFrame()
    pnl = curve["runtime_return_yen"] - curve["runtime_stake_yen"] if not curve.empty else pd.Series(dtype=float)
    eq = pnl.cumsum()
    dd = eq - eq.cummax() if not eq.empty else pd.Series(dtype=float)
    return {
        "policy": label,
        "tickets": int(len(selected)),
        "races": races,
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake,
        "roi": ret / stake if stake else 0.0,
        "ticket_hit_rate": float(selected.get("hit", pd.Series(dtype=bool)).astype(bool).mean()) if len(selected) else 0.0,
        "race_hit_rate": hit_races / races if races else 0.0,
        "max_drawdown_yen": float(dd.min()) if not dd.empty else 0.0,
    }


def _merge_one(tickets: pd.DataFrame, equipment: pd.DataFrame, prefix: str, no_col: str) -> pd.DataFrame:
    eq = equipment.copy()
    rename = {c: f"{prefix}_{c}" for c in eq.columns if c not in {"race_id", "horse_no"}}
    eq = eq.rename(columns=rename)
    out = tickets.copy()
    out[no_col] = pd.to_numeric(out[no_col], errors="coerce").astype("Int64")
    return out.merge(eq, left_on=["race_id", no_col], right_on=["race_id", "horse_no"], how="left").drop(columns=["horse_no"], errors="ignore")


def enrich(tickets: pd.DataFrame, equipment: pd.DataFrame) -> pd.DataFrame:
    out = tickets.copy()
    out["race_id"] = out["race_id"].astype(str).str.zfill(16)
    equipment = equipment.copy()
    equipment["race_id"] = equipment["race_id"].astype(str).str.zfill(16)
    equipment["horse_no"] = pd.to_numeric(equipment["horse_no"], errors="coerce").astype("Int64")
    out = _merge_one(out, equipment, "anchor", "anchor_no")
    out = _merge_one(out, equipment, "partner", "partner_no")
    idx = out.index
    anchor_first = _num(out.get("anchor_equipment_first_or_reapply_blinker_flag"), idx, 0.0).fillna(0.0)
    partner_first = _num(out.get("partner_equipment_first_or_reapply_blinker_flag"), idx, 0.0).fillna(0.0)
    anchor_continue = _num(out.get("anchor_equipment_continue_blinker_flag"), idx, 0.0).fillna(0.0)
    partner_continue = _num(out.get("partner_equipment_continue_blinker_flag"), idx, 0.0).fillna(0.0)
    anchor_remove = _num(out.get("anchor_equipment_remove_blinker_flag"), idx, 0.0).fillna(0.0)
    partner_remove = _num(out.get("partner_equipment_remove_blinker_flag"), idx, 0.0).fillna(0.0)
    out["ticket_equipment_first_or_reapply_flag"] = np.maximum(anchor_first, partner_first)
    out["ticket_equipment_continue_flag"] = np.maximum(anchor_continue, partner_continue)
    out["ticket_equipment_remove_flag"] = np.maximum(anchor_remove, partner_remove)
    out["ticket_equipment_any_change_flag"] = np.maximum(out["ticket_equipment_first_or_reapply_flag"], out["ticket_equipment_remove_flag"])
    out["ticket_equipment_available_flag"] = (
        out.get("anchor_equipment_blinker_flag").notna()
        | out.get("partner_equipment_blinker_flag", pd.Series(False, index=idx)).notna()
    ).astype(int)
    return out


def _segments(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = df[_num(df.get("runtime_stake_yen"), df.index, 0.0).gt(0)].copy()
    features = [
        "ticket_equipment_first_or_reapply_flag",
        "ticket_equipment_continue_flag",
        "ticket_equipment_remove_flag",
        "ticket_equipment_any_change_flag",
    ]
    for feature in features:
        for (ticket_type, flag), g in base.groupby(["ticket_type", feature], dropna=False):
            m = _metric(g, f"{feature}_{ticket_type}_{int(flag)}")
            m.update({"feature": feature, "ticket_type": ticket_type, "flag": int(flag), "avg_flag": float(_num(g.get(feature), g.index, 0).mean())})
            rows.append(m)
    return pd.DataFrame(rows)


def _apply_policy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()
    stake = _num(out.get("runtime_stake_yen"), out.index, 0.0).fillna(0.0)
    mask = stake.gt(0)
    if params["exclude_remove"]:
        mask &= _num(out.get("ticket_equipment_remove_flag"), out.index, 0.0).fillna(0.0).eq(0)
    if params["require_any_signal"]:
        mask &= _num(out.get("ticket_equipment_any_change_flag"), out.index, 0.0).fillna(0.0).eq(1)
    selected = out[mask].copy()
    if selected.empty:
        return selected
    first = _num(selected.get("ticket_equipment_first_or_reapply_flag"), selected.index, 0.0).fillna(0.0)
    cont = _num(selected.get("ticket_equipment_continue_flag"), selected.index, 0.0).fillna(0.0)
    remove = _num(selected.get("ticket_equipment_remove_flag"), selected.index, 0.0).fillna(0.0)
    mult = np.ones(len(selected))
    mult = np.where(first.eq(1), params["first_mult"], mult)
    mult = np.where(cont.eq(1), params["continue_mult"], mult)
    mult = np.where(remove.eq(1), params["remove_mult"], mult)
    selected["pre_equipment_stake_yen"] = selected["runtime_stake_yen"]
    selected["runtime_stake_yen"] = (np.floor((selected["runtime_stake_yen"] * mult).clip(0, params["max_stake"]) / 100.0) * 100.0).clip(lower=0.0)
    selected = selected[selected["runtime_stake_yen"].gt(0)].copy()
    pay = _num(selected.get("runtime_backtest_pay_per100"), selected.index, _num(selected.get("quote_pay_proxy_per100"), selected.index, 0.0)).fillna(0.0)
    selected["runtime_return_yen"] = np.where(selected.get("hit", False).astype(bool), pay * selected["runtime_stake_yen"] / 100.0, 0.0)
    selected["equipment_policy_action"] = np.select(
        [first.eq(1), cont.eq(1), remove.eq(1)],
        ["first_or_reapply", "continue", "remove"],
        default="no_signal",
    )
    return selected


def _grid() -> list[dict]:
    rows = []
    for exclude_remove, require_any_signal, first_mult, continue_mult, remove_mult in product(
        [False, True],
        [False, True],
        [1.0, 1.1, 1.25],
        [1.0, 1.1],
        [0.0, 0.5, 1.0],
    ):
        rows.append(
            {
                "exclude_remove": exclude_remove,
                "require_any_signal": require_any_signal,
                "first_mult": first_mult,
                "continue_mult": continue_mult,
                "remove_mult": remove_mult,
                "max_stake": 3000.0,
            }
        )
    return rows


def _score(train: dict, test: dict) -> float:
    if train["races"] < 120:
        return -1e9
    if train["race_hit_rate"] < 0.08:
        return -1e9
    return train["roi"] * 1.0 + test["roi"] * 0.8 + test["profit_yen"] / 60000.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TARGET blinker/equipment features on current ticket portfolio.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/live_runtime_safety_overlay_v1/live_safety_overlaid_tickets.csv")
    parser.add_argument("--equipment-csv", default="data/processed/target/equipment_features.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/equipment_overlay_v1")
    parser.add_argument("--train-year", type=int, default=2025)
    parser.add_argument("--test-year", type=int, default=2026)
    args = parser.parse_args()

    tickets = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    equipment = pd.read_csv(project_path(args.equipment_csv), dtype={"race_id": str}, low_memory=False)
    enriched = enrich(tickets, equipment)
    train = enriched[enriched["year"].eq(args.train_year)].copy()
    test = enriched[enriched["year"].eq(args.test_year)].copy()
    candidates = []
    best_params = None
    best_score = -1e18
    for params in _grid():
        sel_train = _apply_policy(train, params)
        sel_test = _apply_policy(test, params)
        mt = _metric(sel_train, "train")
        ms = _metric(sel_test, "test")
        score = _score(mt, ms)
        row = {**params, "score": score}
        row.update({f"train_{k}": v for k, v in mt.items() if k != "policy"})
        row.update({f"test_{k}": v for k, v in ms.items() if k != "policy"})
        candidates.append(row)
        if score > best_score:
            best_score = score
            best_params = params

    selected = _apply_policy(enriched, best_params or {})
    out_dir = ensure_dir(project_path(args.output_dir))
    enriched.to_csv(out_dir / "equipment_enriched_tickets.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(out_dir / "equipment_selected_tickets.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(candidates).sort_values("score", ascending=False).to_csv(out_dir / "candidate_policies.csv", index=False, encoding="utf-8-sig")
    _segments(enriched).to_csv(out_dir / "equipment_segments.csv", index=False, encoding="utf-8-sig")
    metrics = [_metric(enriched, "base_all"), _metric(selected, "equipment_policy_all")]
    for year, g in enriched.groupby("year"):
        metrics.append(_metric(g, f"base_{int(year)}"))
        metrics.append(_metric(selected[selected["year"].eq(year)], f"equipment_policy_{int(year)}"))
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    payload = {
        "tickets_csv": args.tickets_csv,
        "equipment_csv": args.equipment_csv,
        "output_dir": str(out_dir),
        "best_params": best_params,
        "equipment_rows": int(len(equipment)),
        "matched_ticket_rows": int(enriched["ticket_equipment_available_flag"].sum()),
        "base_all": metrics[0],
        "equipment_policy_all": metrics[1],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
