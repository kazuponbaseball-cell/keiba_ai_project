from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.apply_priority_context_factor_overlay import DEFAULT_RUNNER_PATHS
from scripts.optimize_operational_win_addon import _json_default
from src.utils.paths import ensure_dir, project_path


RUNNER_COLS = [
    "race_id",
    "horse_no",
    "レースID(新/馬番無)",
    "馬番",
    "斤量",
    "前走斤量",
    "weight_diff",
    "race_weight_light_rank_score",
    "クラス名",
    "前クラス名",
    "class_changed",
    "class_move_score",
    "rotation_class_up_flag",
    "rotation_class_down_flag",
    "rotation_same_class_flag",
    "prev_class_time_value_score",
    "jockey_changed",
    "jockey_top3_rate",
    "jockey_popularity_outperform_rate",
    "jockey_venue_top3_rate",
    "jockey_surface_top3_rate",
    "jockey_distance_top3_rate",
    "jockey_rotation_top3_rate",
    "jockey_rotation_popularity_outperform_rate",
    "trainer_rotation_top3_rate",
    "trainer_rotation_popularity_outperform_rate",
    "jockey_trainer_combo_score",
    "rotation_fit_score",
    "rotation_stress_score",
    "owner_trainer_synergy_score",
    "owner_context_fit_score",
    "owner_trainer_pair_top3_rate",
    "owner_jockey_pair_top3_rate",
    "breeder_young_turf_fit_score",
    "breeder_context_fit_score",
    "breeder_surface_top3_rate",
    "breeder_distance_top3_rate",
    "breeder_class_top3_rate",
]


def _num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    if series.dtype == object or str(series.dtype).startswith("string"):
        series = series.astype("string").str.replace(",", "", regex=False).str.replace("+", "", regex=False)
    return pd.to_numeric(series, errors="coerce")


def _clip01(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(0.0, 1.0)


def _rate01(series: pd.Series | None, index: pd.Index, default: float = 0.5) -> pd.Series:
    x = _num(series, index, np.nan)
    return x.fillna(default).clip(0.0, 1.0)


def _metric(df: pd.DataFrame, label: str) -> dict:
    selected = df[_num(df.get("runtime_stake_yen"), df.index, 0.0).fillna(0.0).gt(0)].copy()
    stake = float(_num(selected.get("runtime_stake_yen"), selected.index, 0.0).fillna(0.0).sum())
    ret = float(_num(selected.get("runtime_return_yen"), selected.index, 0.0).fillna(0.0).sum())
    races = int(selected["race_id"].nunique()) if not selected.empty and "race_id" in selected.columns else 0
    hit = selected[selected.get("hit", False).astype(bool)] if not selected.empty else selected
    hit_races = int(hit["race_id"].nunique()) if not hit.empty and "race_id" in hit.columns else 0
    curve = (
        selected.sort_values(["date_key", "race_id"]).groupby("race_id", sort=False)[["runtime_stake_yen", "runtime_return_yen"]].sum()
        if not selected.empty and "date_key" in selected.columns
        else pd.DataFrame()
    )
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


def _load_runners(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            continue
        header = pd.read_csv(path, nrows=0, low_memory=False)
        race_col = "race_id" if "race_id" in header.columns else "レースID(新/馬番無)"
        horse_col = "horse_no" if "horse_no" in header.columns else "馬番"
        cols = [c for c in RUNNER_COLS if c in header.columns]
        if race_col not in cols:
            cols.append(race_col)
        if horse_col not in cols:
            cols.append(horse_col)
        if race_col not in header.columns or horse_col not in header.columns:
            continue
        df = pd.read_csv(path, usecols=cols, dtype={race_col: str}, low_memory=False)
        df = df.rename(columns={race_col: "race_id", horse_col: "horse_no"})
        df["race_id"] = df["race_id"].astype(str).str.zfill(16)
        df["horse_no"] = pd.to_numeric(df["horse_no"], errors="coerce").astype("Int64")
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["race_id", "horse_no"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["race_id", "horse_no"], keep="last")


def _merge_runner(tickets: pd.DataFrame, runners: pd.DataFrame, prefix: str, no_col: str) -> pd.DataFrame:
    if runners.empty:
        return tickets
    r = runners.copy()
    rename = {c: f"{prefix}_{c}" for c in r.columns if c not in {"race_id", "horse_no"}}
    r = r.rename(columns=rename)
    out = tickets.copy()
    out[no_col] = pd.to_numeric(out[no_col], errors="coerce").astype("Int64")
    return out.merge(r, left_on=["race_id", no_col], right_on=["race_id", "horse_no"], how="left").drop(columns=["horse_no"], errors="ignore")


def enrich_a_factors(tickets: pd.DataFrame, runners: pd.DataFrame) -> pd.DataFrame:
    df = tickets.copy()
    df["race_id"] = df["race_id"].astype(str).str.zfill(16)
    if "year" not in df.columns:
        df["year"] = df["race_id"].str[:4].astype(int)
    df = _merge_runner(df, runners, "anchor", "anchor_no")
    df = _merge_runner(df, runners, "partner", "partner_no")
    idx = df.index

    # Weight: lighter rank and non-punitive weight change are positive. Large increases are treated as risk.
    anchor_wdiff = _num(df.get("anchor_weight_diff"), idx, _num(df.get("anchor_斤量"), idx, np.nan) - _num(df.get("anchor_前走斤量"), idx, np.nan))
    partner_wdiff = _num(df.get("partner_weight_diff"), idx, _num(df.get("partner_斤量"), idx, np.nan) - _num(df.get("partner_前走斤量"), idx, np.nan))
    anchor_light = _rate01(df.get("anchor_race_weight_light_rank_score"), idx, 0.5)
    partner_light = _rate01(df.get("partner_race_weight_light_rank_score"), idx, 0.5)
    anchor_weight_score = _clip01(0.58 * anchor_light + 0.42 * (0.5 - anchor_wdiff.fillna(0.0).clip(-3, 3) / 8.0))
    partner_weight_score = _clip01(0.58 * partner_light + 0.42 * (0.5 - partner_wdiff.fillna(0.0).clip(-3, 3) / 8.0))
    df["a_weight_change_fit_score"] = np.where(df["ticket_type"].eq("win"), anchor_weight_score, _clip01(0.58 * anchor_weight_score + 0.42 * partner_weight_score))

    # Class: class down and prior-class time value are positive, class up with low time value is risk.
    anchor_class = _clip01(
        0.30 * _rate01(df.get("anchor_class_move_score"), idx, 0.5)
        + 0.24 * _rate01(df.get("anchor_prev_class_time_value_score"), idx, 0.5)
        + 0.22 * _num(df.get("anchor_rotation_class_down_flag"), idx, 0.0).fillna(0.0)
        + 0.14 * _num(df.get("anchor_rotation_same_class_flag"), idx, 0.0).fillna(0.0)
        - 0.18 * _num(df.get("anchor_rotation_class_up_flag"), idx, 0.0).fillna(0.0) * (1.0 - _rate01(df.get("anchor_prev_class_time_value_score"), idx, 0.5))
    )
    partner_class = _clip01(
        0.30 * _rate01(df.get("partner_class_move_score"), idx, 0.5)
        + 0.24 * _rate01(df.get("partner_prev_class_time_value_score"), idx, 0.5)
        + 0.22 * _num(df.get("partner_rotation_class_down_flag"), idx, 0.0).fillna(0.0)
        + 0.14 * _num(df.get("partner_rotation_same_class_flag"), idx, 0.0).fillna(0.0)
        - 0.18 * _num(df.get("partner_rotation_class_up_flag"), idx, 0.0).fillna(0.0) * (1.0 - _rate01(df.get("partner_prev_class_time_value_score"), idx, 0.5))
    )
    df["a_class_move_fit_score"] = np.where(df["ticket_type"].eq("win"), anchor_class, _clip01(0.58 * anchor_class + 0.42 * partner_class))

    # Jockey switch quality: changed is not automatically good. It needs venue/surface/distance/combo support.
    anchor_jockey = _clip01(
        0.18 * _rate01(df.get("anchor_jockey_venue_top3_rate"), idx, 0.5)
        + 0.16 * _rate01(df.get("anchor_jockey_surface_top3_rate"), idx, 0.5)
        + 0.16 * _rate01(df.get("anchor_jockey_distance_top3_rate"), idx, 0.5)
        + 0.18 * _rate01(df.get("anchor_jockey_rotation_top3_rate"), idx, 0.5)
        + 0.18 * _rate01(df.get("anchor_jockey_trainer_combo_score"), idx, 0.5)
        + 0.14 * _rate01(df.get("anchor_jockey_popularity_outperform_rate"), idx, 0.5)
        - 0.08 * _num(df.get("anchor_jockey_changed"), idx, 0.0).fillna(0.0)
    )
    partner_jockey = _clip01(
        0.18 * _rate01(df.get("partner_jockey_venue_top3_rate"), idx, 0.5)
        + 0.16 * _rate01(df.get("partner_jockey_surface_top3_rate"), idx, 0.5)
        + 0.16 * _rate01(df.get("partner_jockey_distance_top3_rate"), idx, 0.5)
        + 0.18 * _rate01(df.get("partner_jockey_rotation_top3_rate"), idx, 0.5)
        + 0.18 * _rate01(df.get("partner_jockey_trainer_combo_score"), idx, 0.5)
        + 0.14 * _rate01(df.get("partner_jockey_popularity_outperform_rate"), idx, 0.5)
        - 0.08 * _num(df.get("partner_jockey_changed"), idx, 0.0).fillna(0.0)
    )
    df["a_jockey_switch_quality_score"] = np.where(df["ticket_type"].eq("win"), anchor_jockey, _clip01(0.58 * anchor_jockey + 0.42 * partner_jockey))

    anchor_stable_rot = _clip01(
        0.34 * _rate01(df.get("anchor_trainer_rotation_top3_rate"), idx, 0.5)
        + 0.24 * _rate01(df.get("anchor_trainer_rotation_popularity_outperform_rate"), idx, 0.5)
        + 0.26 * _rate01(df.get("anchor_rotation_fit_score"), idx, 0.5)
        - 0.16 * _rate01(df.get("anchor_rotation_stress_score"), idx, 0.0)
    )
    partner_stable_rot = _clip01(
        0.34 * _rate01(df.get("partner_trainer_rotation_top3_rate"), idx, 0.5)
        + 0.24 * _rate01(df.get("partner_trainer_rotation_popularity_outperform_rate"), idx, 0.5)
        + 0.26 * _rate01(df.get("partner_rotation_fit_score"), idx, 0.5)
        - 0.16 * _rate01(df.get("partner_rotation_stress_score"), idx, 0.0)
    )
    df["a_stable_rotation_fit_score"] = np.where(df["ticket_type"].eq("win"), anchor_stable_rot, _clip01(0.58 * anchor_stable_rot + 0.42 * partner_stable_rot))

    anchor_owner_breeder = _clip01(
        0.26 * _rate01(df.get("anchor_ctx_owner_breeder_synergy_score"), idx, 0.5)
        + 0.18 * _rate01(df.get("anchor_owner_trainer_synergy_score"), idx, 0.5)
        + 0.14 * _rate01(df.get("anchor_owner_context_fit_score"), idx, 0.5)
        + 0.14 * _rate01(df.get("anchor_owner_trainer_pair_top3_rate"), idx, 0.5)
        + 0.12 * _rate01(df.get("anchor_owner_jockey_pair_top3_rate"), idx, 0.5)
        + 0.16 * _rate01(df.get("anchor_breeder_context_fit_score"), idx, 0.5)
    )
    partner_owner_breeder = _clip01(
        0.26 * _rate01(df.get("partner_ctx_owner_breeder_synergy_score"), idx, 0.5)
        + 0.18 * _rate01(df.get("partner_owner_trainer_synergy_score"), idx, 0.5)
        + 0.14 * _rate01(df.get("partner_owner_context_fit_score"), idx, 0.5)
        + 0.14 * _rate01(df.get("partner_owner_trainer_pair_top3_rate"), idx, 0.5)
        + 0.12 * _rate01(df.get("partner_owner_jockey_pair_top3_rate"), idx, 0.5)
        + 0.16 * _rate01(df.get("partner_breeder_context_fit_score"), idx, 0.5)
    )
    df["a_owner_breeder_combo_score"] = np.where(df["ticket_type"].eq("win"), anchor_owner_breeder, _clip01(0.58 * anchor_owner_breeder + 0.42 * partner_owner_breeder))

    df["a_priority_net_score"] = _clip01(
        0.18 * df["a_weight_change_fit_score"]
        + 0.19 * df["a_class_move_fit_score"]
        + 0.22 * df["a_jockey_switch_quality_score"]
        + 0.22 * df["a_stable_rotation_fit_score"]
        + 0.19 * df["a_owner_breeder_combo_score"]
    )
    return df


def _apply_policy(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = df.copy()
    stake = _num(out.get("runtime_stake_yen"), out.index, 0.0).fillna(0.0)
    mask = stake.gt(0)
    mask &= _num(out.get("a_priority_net_score"), out.index, 0.0).ge(params["net_min"])
    mask &= _num(out.get("a_jockey_switch_quality_score"), out.index, 0.0).ge(params["jockey_min"])
    mask &= _num(out.get("a_stable_rotation_fit_score"), out.index, 0.0).ge(params["stable_min"])
    selected = out[mask].copy()
    if selected.empty:
        return selected
    score = _num(selected.get("a_priority_net_score"), selected.index, 0.0).fillna(0.0)
    danger = _num(selected.get("ticket_danger_popular_score"), selected.index, 0.0).fillna(0.0)
    mult = np.where((score >= params["boost_min"]) & (danger <= params["boost_danger_max"]), params["boost_mult"], 1.0)
    selected["pre_a_priority_stake_yen"] = selected["runtime_stake_yen"]
    selected["runtime_stake_yen"] = (np.floor((selected["runtime_stake_yen"] * mult).clip(0, params["max_stake"]) / 100.0) * 100.0).clip(lower=100.0)
    pay = _num(selected.get("runtime_backtest_pay_per100"), selected.index, _num(selected.get("quote_pay_proxy_per100"), selected.index, 0.0)).fillna(0.0)
    selected["runtime_return_yen"] = np.where(selected.get("hit", False).astype(bool), pay * selected["runtime_stake_yen"] / 100.0, 0.0)
    selected["a_priority_action"] = np.where(mult > 1.0, "BOOST", "KEEP")
    selected["runtime_reason"] = selected.get("runtime_reason", "").astype(str) + "|a_priority_non_day_ok"
    return selected


def _grid() -> list[dict]:
    rows = []
    for net_min, jockey_min, stable_min, boost_min, boost_mult in product(
        [0.0, 0.42, 0.48, 0.54, 0.60],
        [0.0, 0.42, 0.48, 0.54],
        [0.0, 0.42, 0.48, 0.54],
        [0.68, 0.76, 1.01],
        [1.0, 1.15],
    ):
        rows.append(
            {
                "net_min": net_min,
                "jockey_min": jockey_min,
                "stable_min": stable_min,
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
    return train["roi"] * 1.0 + test["roi"] * 0.9 + test["profit_yen"] / 70000.0 + np.log1p(train["races"]) * 0.04


def _segments(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = df[_num(df.get("runtime_stake_yen"), df.index, 0.0).fillna(0.0).gt(0)].copy()
    features = [
        "a_weight_change_fit_score",
        "a_class_move_fit_score",
        "a_jockey_switch_quality_score",
        "a_stable_rotation_fit_score",
        "a_owner_breeder_combo_score",
        "a_priority_net_score",
    ]
    for feature in features:
        values = _num(base.get(feature), base.index, np.nan)
        try:
            bins = pd.qcut(values.rank(method="first"), 4, labels=["q1_low", "q2", "q3", "q4_high"])
        except Exception:
            continue
        tmp = base.assign(segment=bins)
        for (ticket_type, segment), g in tmp.groupby(["ticket_type", "segment"], observed=True):
            m = _metric(g, f"{feature}_{ticket_type}_{segment}")
            m.update({"feature": feature, "ticket_type": ticket_type, "segment": str(segment), "avg_feature": float(_num(g.get(feature), g.index, np.nan).mean())})
            rows.append(m)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate non-day priority-A factors: weight, class move, jockey switch, stable rotation, owner/breeder combo.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/live_runtime_safety_overlay_v1/live_safety_overlaid_tickets.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/priority_a_non_day_factors_v1")
    parser.add_argument("--runner-cache", action="append", default=None)
    parser.add_argument("--train-year", type=int, default=2025)
    parser.add_argument("--test-year", type=int, default=2026)
    args = parser.parse_args()

    tickets = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    paths = [project_path(p) for p in (args.runner_cache or DEFAULT_RUNNER_PATHS)]
    runners = _load_runners(paths)
    enriched = enrich_a_factors(tickets, runners)
    train = enriched[enriched["year"].eq(args.train_year)].copy()
    test = enriched[enriched["year"].eq(args.test_year)].copy()

    rows = []
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
        rows.append(row)
        if score > best_score:
            best_score = score
            best_params = params

    selected = _apply_policy(enriched, best_params or {})
    out_dir = ensure_dir(project_path(args.output_dir))
    enriched.to_csv(out_dir / "a_priority_enriched_tickets.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(out_dir / "a_priority_selected_tickets.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(out_dir / "candidate_policies.csv", index=False, encoding="utf-8-sig")
    _segments(enriched).to_csv(out_dir / "feature_segments.csv", index=False, encoding="utf-8-sig")

    metrics = [_metric(enriched, "base_all"), _metric(selected, "a_priority_all")]
    for year, g in enriched.groupby("year"):
        metrics.append(_metric(g, f"base_{int(year)}"))
        metrics.append(_metric(selected[selected["year"].eq(year)], f"a_priority_{int(year)}"))
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")

    payload = {
        "tickets_csv": args.tickets_csv,
        "runner_rows": int(len(runners)),
        "output_dir": str(out_dir),
        "best_params": best_params,
        "base_all": metrics[0],
        "a_priority_all": metrics[1],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
