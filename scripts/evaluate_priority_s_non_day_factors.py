from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.apply_contextual_value_overlays import _load_runner_context
from scripts.apply_priority_context_factor_overlay import DEFAULT_RUNNER_PATHS
from scripts.optimize_operational_win_addon import _json_default
from src.utils.paths import ensure_dir, project_path


EXTRA_RUNNER_COLS = [
    "間隔",
    "休み明け～戦目",
    "年齢",
    "距離",
    "前距離",
    "前走馬番",
    "異常コード",
    "rotation_fit_score",
    "rotation_fresh_start_flag",
    "rotation_second_after_layoff_flag",
    "rotation_third_after_layoff_flag",
    "body_layoff_flag",
    "body_layoff_workout_count_fit",
    "body_layoff_recent_workout_flag",
    "trainer_rotation_top3_rate",
    "trainer_rotation_popularity_outperform_rate",
]


def _num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    if series.dtype == object or str(series.dtype).startswith("string"):
        series = (
            series.astype("string")
            .str.replace(",", "", regex=False)
            .str.replace("+", "", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
        )
    return pd.to_numeric(series, errors="coerce")


def _clip01(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(0.0, 1.0)


def _safe_metric(df: pd.DataFrame, label: str) -> dict:
    selected = df[_num(df.get("runtime_stake_yen"), df.index, 0.0).fillna(0.0).gt(0)].copy()
    stake = float(_num(selected.get("runtime_stake_yen"), selected.index, 0.0).fillna(0.0).sum())
    ret = float(_num(selected.get("runtime_return_yen"), selected.index, 0.0).fillna(0.0).sum())
    races = int(selected["race_id"].nunique()) if not selected.empty and "race_id" in selected.columns else 0
    hit = selected[selected.get("hit", False).astype(bool)] if not selected.empty else selected
    hit_races = int(hit["race_id"].nunique()) if not hit.empty and "race_id" in hit.columns else 0
    sort_cols = [c for c in ["date_key", "race_id"] if c in selected.columns]
    curve = selected.sort_values(sort_cols).groupby("race_id", sort=False)[["runtime_stake_yen", "runtime_return_yen"]].sum() if sort_cols else pd.DataFrame()
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


def _load_extra_runner(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        header = pd.read_csv(path, nrows=0, low_memory=False)
        race_col = "race_id" if "race_id" in header.columns else "レースID(新/馬番無)"
        horse_col = "horse_no" if "horse_no" in header.columns else "馬番"
        cols = [c for c in [race_col, horse_col, *EXTRA_RUNNER_COLS] if c in header.columns]
        if race_col not in cols or horse_col not in cols:
            continue
        df = pd.read_csv(path, usecols=cols, dtype={race_col: str}, low_memory=False)
        df = df.rename(columns={race_col: "race_id", horse_col: "horse_no"})
        df["race_id"] = df["race_id"].astype(str)
        df["horse_no"] = pd.to_numeric(df["horse_no"], errors="coerce").astype("Int64")
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["race_id", "horse_no"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["race_id", "horse_no"], keep="last")


def _merge_runner(df: pd.DataFrame, runners: pd.DataFrame, prefix: str, no_col: str) -> pd.DataFrame:
    if runners.empty or no_col not in df.columns:
        return df
    r = runners.copy()
    rename = {c: f"{prefix}_{c}" for c in r.columns if c not in {"race_id", "horse_no"}}
    r = r.rename(columns=rename)
    out = df.copy()
    out[no_col] = pd.to_numeric(out[no_col], errors="coerce").astype("Int64")
    return out.merge(r, left_on=["race_id", no_col], right_on=["race_id", "horse_no"], how="left").drop(columns=["horse_no"], errors="ignore")


def enrich_s_factors(tickets: pd.DataFrame, extra_runners: pd.DataFrame) -> pd.DataFrame:
    df = tickets.copy()
    df["race_id"] = df["race_id"].astype(str)
    if "year" not in df.columns:
        df["year"] = df["race_id"].str[:4].astype(int)
    df = _merge_runner(df, extra_runners, "anchor", "anchor_no")
    df = _merge_runner(df, extra_runners, "partner", "partner_no")
    idx = df.index

    front_reliability = _num(df.get("ticket_front_position_reliability_score"), idx, 0.5).fillna(0.5)
    projected_front = _num(df.get("projected_front5_prob"), idx, 0.5).fillna(0.5)
    front_history = _num(df.get("horse_front_run_rate_past5_feature"), idx, _num(df.get("horse_front_run_rate_past5"), idx, 0.5)).fillna(0.5)
    draw_fit = _num(df.get("draw_pace_fit_score_feature"), idx, _num(df.get("draw_pace_fit_score"), idx, 0.5)).fillna(0.5)
    pressure = _num(df.get("race_front_pressure"), idx, 0.5).fillna(0.5)
    collapse = _num(df.get("race_pace_collapse"), idx, 0.0).fillna(0.0)
    mismatch = (projected_front - front_history).clip(lower=0.0, upper=1.0)
    df["s_gate_start_risk_score"] = _clip01(
        0.34 * (1.0 - front_reliability)
        + 0.20 * mismatch
        + 0.16 * (1.0 - draw_fit)
        + 0.16 * pressure
        + 0.14 * collapse
    )
    df["s_gate_start_fit_score"] = _clip01(1.0 - df["s_gate_start_risk_score"])

    body_layoff = _num(df.get("ticket_body_age_layoff_score"), idx, 0.5).fillna(0.5)
    stable = _num(df.get("ticket_stable_jockey_buy_timing_score"), idx, 0.5).fillna(0.5)
    rebound_anchor = _num(df.get("anchor_ctx_condition_rebound_score"), idx, 0.5).fillna(0.5)
    rebound_partner = _num(df.get("partner_ctx_condition_rebound_score"), idx, 0.5).fillna(0.5)
    rotation_anchor = _num(df.get("anchor_rotation_fit_score"), idx, 0.5).fillna(0.5)
    rotation_partner = _num(df.get("partner_rotation_fit_score"), idx, 0.5).fillna(0.5)
    fresh_anchor = _num(df.get("anchor_rotation_fresh_start_flag"), idx, 0.0).fillna(0.0)
    fresh_partner = _num(df.get("partner_rotation_fresh_start_flag"), idx, 0.0).fillna(0.0)
    second_anchor = _num(df.get("anchor_rotation_second_after_layoff_flag"), idx, 0.0).fillna(0.0)
    second_partner = _num(df.get("partner_rotation_second_after_layoff_flag"), idx, 0.0).fillna(0.0)
    workout_layoff_anchor = _num(df.get("anchor_body_layoff_workout_count_fit"), idx, 0.5).fillna(0.5)
    workout_layoff_partner = _num(df.get("partner_body_layoff_workout_count_fit"), idx, 0.5).fillna(0.5)
    anchor_layoff_fit = _clip01(
        0.30 * body_layoff
        + 0.18 * stable
        + 0.16 * rebound_anchor
        + 0.14 * rotation_anchor
        + 0.12 * workout_layoff_anchor
        + 0.10 * second_anchor
        - 0.10 * fresh_anchor * (1.0 - workout_layoff_anchor)
    )
    partner_layoff_fit = _clip01(
        0.30 * body_layoff
        + 0.18 * stable
        + 0.16 * rebound_partner
        + 0.14 * rotation_partner
        + 0.12 * workout_layoff_partner
        + 0.10 * second_partner
        - 0.10 * fresh_partner * (1.0 - workout_layoff_partner)
    )
    df["s_layoff_tataki_fit_score"] = np.where(df.get("ticket_type", "").astype(str).eq("win"), anchor_layoff_fit, _clip01(0.58 * anchor_layoff_fit + 0.42 * partner_layoff_fit))

    # No reliable equipment-change column is present in the current exported data.
    df["s_equipment_available_flag"] = 0
    df["s_equipment_change_score"] = 0.5

    df["s_priority_net_score"] = _clip01(
        0.44 * df["s_gate_start_fit_score"]
        + 0.38 * df["s_layoff_tataki_fit_score"]
        + 0.18 * _num(df.get("priority_context_net_score"), idx, 0.5).fillna(0.5)
    )
    return df


def _apply_policy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()
    base_stake = _num(out.get("runtime_stake_yen"), out.index, 0.0).fillna(0.0)
    mask = base_stake.gt(0)
    mask &= _num(out.get("s_gate_start_fit_score"), out.index, 0.0).ge(params["gate_min"])
    mask &= _num(out.get("s_layoff_tataki_fit_score"), out.index, 0.0).ge(params["layoff_min"])
    mask &= _num(out.get("s_priority_net_score"), out.index, 0.0).ge(params["net_min"])
    selected = out[mask].copy()
    if selected.empty:
        return selected
    s_net = _num(selected.get("s_priority_net_score"), selected.index, 0.0).fillna(0.0)
    danger = _num(selected.get("ticket_danger_popular_score"), selected.index, 0.0).fillna(0.0)
    mult = np.where((s_net >= params["boost_min"]) & (danger <= params["boost_danger_max"]), params["boost_mult"], 1.0)
    selected["pre_s_priority_stake_yen"] = selected["runtime_stake_yen"]
    selected["runtime_stake_yen"] = (np.floor((selected["runtime_stake_yen"] * mult).clip(0, params["max_stake"]) / 100.0) * 100.0).clip(lower=100.0)
    pay = _num(selected.get("runtime_backtest_pay_per100"), selected.index, _num(selected.get("quote_pay_proxy_per100"), selected.index, 0.0)).fillna(0.0)
    selected["runtime_return_yen"] = np.where(selected.get("hit", False).astype(bool), pay * selected["runtime_stake_yen"] / 100.0, 0.0)
    selected["s_priority_action"] = np.where(mult > 1.0, "BOOST", "KEEP")
    selected["runtime_reason"] = selected.get("runtime_reason", "").astype(str) + "|s_priority_non_day_ok"
    return selected


def _grid() -> list[dict]:
    rows = []
    for gate_min, layoff_min, net_min, boost_min, boost_mult in product(
        [0.0, 0.42, 0.48, 0.54, 0.60],
        [0.0, 0.40, 0.46, 0.52, 0.58],
        [0.0, 0.42, 0.48, 0.54],
        [0.68, 0.74, 1.01],
        [1.0, 1.15],
    ):
        rows.append(
            {
                "gate_min": gate_min,
                "layoff_min": layoff_min,
                "net_min": net_min,
                "boost_min": boost_min,
                "boost_mult": boost_mult,
                "boost_danger_max": 0.70,
                "max_stake": 3000.0,
            }
        )
    return rows


def _score(train: dict, test: dict) -> float:
    if train["races"] < 150:
        return -1e9
    if train["race_hit_rate"] < 0.08:
        return -1e9
    return train["roi"] * 1.2 + test["roi"] * 0.9 + np.log1p(train["races"]) * 0.05 + test["profit_yen"] / 60000.0


def _segments(enriched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = enriched[_num(enriched.get("runtime_stake_yen"), enriched.index, 0.0).fillna(0.0).gt(0)].copy()
    for feature in ["s_gate_start_fit_score", "s_layoff_tataki_fit_score", "s_priority_net_score"]:
        values = _num(base.get(feature), base.index, np.nan)
        try:
            bins = pd.qcut(values.rank(method="first"), 4, labels=["q1_low", "q2", "q3", "q4_high"])
        except Exception:
            continue
        tmp = base.assign(segment=bins)
        for (ticket_type, segment), g in tmp.groupby(["ticket_type", "segment"], observed=True):
            m = _safe_metric(g, f"{feature}_{ticket_type}_{segment}")
            m.update({"feature": feature, "ticket_type": ticket_type, "segment": str(segment), "avg_feature": float(_num(g.get(feature), g.index, np.nan).mean())})
            rows.append(m)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate non-day S-priority factors: gate/start risk proxy, equipment availability, and layoff/tataki fit.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/live_runtime_safety_overlay_v1/live_safety_overlaid_tickets.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/priority_s_non_day_factors_v1")
    parser.add_argument("--runner-cache", action="append", default=None)
    parser.add_argument("--train-year", type=int, default=2025)
    parser.add_argument("--test-year", type=int, default=2026)
    args = parser.parse_args()

    tickets = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    paths = [project_path(p) for p in (args.runner_cache or DEFAULT_RUNNER_PATHS)]
    # Keep compatibility with existing loader in case future exports add normalized fields there.
    base_runners = _load_runner_context(paths)
    extra_runners = _load_extra_runner(paths)
    if not base_runners.empty:
        extra_runners = extra_runners.merge(base_runners[["race_id", "horse_no"]].drop_duplicates(), on=["race_id", "horse_no"], how="outer")
    enriched = enrich_s_factors(tickets, extra_runners)

    train = enriched[enriched["year"].eq(args.train_year)].copy()
    test = enriched[enriched["year"].eq(args.test_year)].copy()
    candidates = []
    best_params = None
    best_score = -1e18
    for params in _grid():
        sel_train = _apply_policy(train, params)
        sel_test = _apply_policy(test, params)
        mt = _safe_metric(sel_train, "train")
        ms = _safe_metric(sel_test, "test")
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
    enriched.to_csv(out_dir / "s_priority_enriched_tickets.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(out_dir / "s_priority_selected_tickets.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(candidates).sort_values("score", ascending=False).to_csv(out_dir / "candidate_policies.csv", index=False, encoding="utf-8-sig")
    _segments(enriched).to_csv(out_dir / "feature_segments.csv", index=False, encoding="utf-8-sig")

    metrics = [_safe_metric(enriched, "base_all"), _safe_metric(selected, "s_priority_all")]
    for year, g in enriched.groupby("year"):
        metrics.append(_safe_metric(g, f"base_{int(year)}"))
        metrics.append(_safe_metric(selected[selected["year"].eq(year)], f"s_priority_{int(year)}"))
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    payload = {
        "tickets_csv": args.tickets_csv,
        "output_dir": str(out_dir),
        "best_params": best_params,
        "equipment_change_available": False,
        "equipment_note": "No reliable equipment/blinker-change column was found in current exports. Keep as future TARGET/JV extraction task.",
        "base_all": metrics[0],
        "s_priority_all": metrics[1],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
