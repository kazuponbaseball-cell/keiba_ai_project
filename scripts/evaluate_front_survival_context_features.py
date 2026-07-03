from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / (
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/"
    "body_weight_backfilled/owner_breeder_enriched/train_features_with_same_day_bias_v3_retro_body_breeder.csv"
)
DEFAULT_TEST = ROOT / (
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/"
    "body_weight_backfilled/owner_breeder_enriched/test_features_with_same_day_bias_v3_retro_body_breeder.csv"
)
DEFAULT_TICKETS = ROOT / "outputs/analysis/lap_pair_refinement_candidates_v1/s_priority_tickets_with_lap_pair_refinement.csv"
DEFAULT_OUT = ROOT / "outputs/analysis/front_survival_context_v1"

RACE_COL = "レースID(新/馬番無)"


def read_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, nrows=nrows)


def num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    values = frame[col]
    if values.dtype == object:
        values = values.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(values, errors="coerce")


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def norm_race_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True)


def bucket_distance(distance: pd.Series) -> pd.Series:
    d = pd.to_numeric(distance, errors="coerce")
    bins = [-np.inf, 1200, 1400, 1600, 1800, 2000, 2400, np.inf]
    labels = ["<=1200", "1201-1400", "1401-1600", "1601-1800", "1801-2000", "2001-2400", "2401+"]
    return pd.cut(d, bins=bins, labels=labels).astype(str).replace("nan", "unknown")


def class_group(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str)
    out = pd.Series("other", index=series.index, dtype=object)
    out[s.str.contains("新馬", regex=False)] = "newcomer"
    out[s.str.contains("未勝利", regex=False)] = "maiden"
    out[s.str.contains("1勝|500万", regex=True)] = "1win"
    out[s.str.contains("2勝|1000万", regex=True)] = "2win"
    out[s.str.contains("3勝|1600万", regex=True)] = "3win"
    out[s.str.contains("オープン|OP|L|リステッド", regex=True)] = "open"
    out[s.str.contains("G1|Ｇ１|G2|Ｇ２|G3|Ｇ３|重賞", regex=True)] = "graded"
    return out


def going_group(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str)
    out = pd.Series("unknown", index=series.index, dtype=object)
    out[s.str.contains("良", regex=False)] = "firm"
    out[s.str.contains("稍", regex=False)] = "yielding"
    out[s.str.contains("重", regex=False)] = "soft"
    out[s.str.contains("不", regex=False)] = "heavy"
    return out


def build_race_table(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    work = frame.copy()
    work["race_id"] = norm_race_id(work[RACE_COL])
    work["_corner4"] = num(work, "4角")
    work["_finish"] = num(work, "確定着順")
    work["_field"] = num(work, "出走頭数").fillna(num(work, "頭数"))
    work["_actual_front5"] = work["_corner4"].le(5)
    work["_actual_front_third"] = work["_corner4"].le(np.maximum(3, np.floor(work["_field"] * 0.33)))
    work["_is_win"] = num(work, "target_win", 0).fillna(0).eq(1)
    work["_is_top3"] = num(work, "target_top3", 0).fillna(0).eq(1)

    first_cols = [
        "日付",
        "日付S",
        "場所",
        "Ｒ",
        "レース名",
        "クラス名",
        "芝・ダ",
        "距離",
        "馬場状態",
        "頭数",
        "出走頭数",
        "RPCI",
        "PCI3",
        "race_front_runner_count",
        "race_front_runner_ratio",
        "race_early_pressure_score",
        "race_need_lead_count",
        "race_stalker_count_deep",
        "race_pace_collapse_risk",
        "race_slow_pace_risk",
        "solo_lead_potential",
    ]
    rows: list[dict[str, Any]] = []
    for race_id, group in work.groupby("race_id", sort=False):
        first = group.iloc[0]
        field = float(pd.to_numeric(first.get("出走頭数", first.get("頭数", np.nan)), errors="coerce"))
        top3 = group[group["_is_top3"]]
        winner = group[group["_is_win"]]
        top3_front5_count = int(top3["_actual_front5"].sum()) if not top3.empty else 0
        top3_front_third_count = int(top3["_actual_front_third"].sum()) if not top3.empty else 0
        winner_front5 = bool(winner["_actual_front5"].any()) if not winner.empty else False
        winner_front_third = bool(winner["_actual_front_third"].any()) if not winner.empty else False
        row = {"race_id": race_id, "split": split}
        for col in first_cols:
            if col in work.columns:
                row[col] = first.get(col)
        row.update(
            {
                "field_size": field,
                "top3_front5_count": top3_front5_count,
                "top3_front_third_count": top3_front_third_count,
                "winner_front5": int(winner_front5),
                "winner_front_third": int(winner_front_third),
                "actual_front_survival": int(winner_front5 or top3_front5_count >= 2),
                "actual_front_collapse": int(top3_front5_count == 0),
                "actual_front_third_survival": int(winner_front_third or top3_front_third_count >= 2),
                "actual_top3_front5_share": safe_div(top3_front5_count, len(top3)) if len(top3) else np.nan,
            }
        )
        rows.append(row)

    races = pd.DataFrame(rows)
    races["race_id"] = norm_race_id(races["race_id"])
    races["date_key"] = pd.to_numeric(races.get("日付"), errors="coerce").fillna(pd.to_numeric(races.get("日付S"), errors="coerce"))
    races["year"] = (races["date_key"] // 10000).astype("Int64").astype(str)
    races["venue"] = races.get("場所", pd.Series("", index=races.index)).fillna("").astype(str)
    races["surface"] = races.get("芝・ダ", pd.Series("", index=races.index)).fillna("").astype(str)
    races["distance_num"] = pd.to_numeric(races.get("距離"), errors="coerce")
    races["distance_bin"] = bucket_distance(races["distance_num"])
    races["class_group"] = class_group(races.get("クラス名", pd.Series("", index=races.index)))
    races["going_group"] = going_group(races.get("馬場状態", pd.Series("", index=races.index)))
    collapse = pd.to_numeric(races.get("race_pace_collapse_risk"), errors="coerce").fillna(0.0).clip(0, 1)
    pressure = pd.to_numeric(races.get("race_early_pressure_score"), errors="coerce").fillna(0.0).clip(0, 1)
    front_ratio = pd.to_numeric(races.get("race_front_runner_ratio"), errors="coerce").fillna(0.0).clip(0, 1)
    need_lead = pd.to_numeric(races.get("race_need_lead_count"), errors="coerce").fillna(0.0)
    slow = pd.to_numeric(races.get("race_slow_pace_risk"), errors="coerce").fillna(0.0).clip(0, 1)
    solo = pd.to_numeric(races.get("solo_lead_potential"), errors="coerce").fillna(0.0).clip(0, 1)
    races["pre_high_pressure_signal"] = (
        0.34 * collapse
        + 0.28 * pressure
        + 0.18 * front_ratio
        + 0.12 * (need_lead / 4.0).clip(0, 1)
        + 0.08 * (1.0 - solo)
        - 0.10 * slow
    ).clip(0, 1)
    return races.sort_values(["date_key", "venue", "Ｒ", "race_id"], kind="mergesort")


def rolling_prior(frame: pd.DataFrame, keys: list[str], target: str, min_periods: int) -> tuple[pd.Series, pd.Series]:
    grouped = frame.groupby(keys, sort=False)[target]
    prior = grouped.transform(lambda s: s.shift().expanding(min_periods=min_periods).mean())
    count = frame.groupby(keys, sort=False).cumcount()
    return prior, count


def add_context_priors(races: pd.DataFrame) -> pd.DataFrame:
    out = races.copy().sort_values(["date_key", "venue", "Ｒ", "race_id"], kind="mergesort")
    out["global_survival_prior"] = out["actual_front_survival"].shift().expanding(min_periods=30).mean()
    out["global_collapse_prior"] = out["actual_front_collapse"].shift().expanding(min_periods=30).mean()
    out["global_top3_front5_share_prior"] = out["actual_top3_front5_share"].shift().expanding(min_periods=30).mean()

    specs = [
        (["venue", "surface", "distance_bin", "class_group", "going_group"], "full", 8),
        (["venue", "surface", "distance_bin", "class_group"], "course_class", 10),
        (["venue", "surface", "distance_bin"], "course", 12),
        (["venue", "surface"], "venue_surface", 20),
        (["surface", "distance_bin"], "surface_distance", 25),
    ]
    for keys, prefix, min_periods in specs:
        out[f"{prefix}_survival_prior"], out[f"{prefix}_prior_count"] = rolling_prior(
            out, keys, "actual_front_survival", min_periods
        )
        out[f"{prefix}_collapse_prior"], _ = rolling_prior(out, keys, "actual_front_collapse", min_periods)
        out[f"{prefix}_top3_front5_share_prior"], _ = rolling_prior(out, keys, "actual_top3_front5_share", min_periods)

    for target in ["survival", "collapse", "top3_front5_share"]:
        out[f"front_{target}_context_prior"] = out[f"full_{target}_prior"]
        out[f"front_{target}_context_count"] = out["full_prior_count"]
        for prefix in ["course_class", "course", "venue_surface", "surface_distance"]:
            missing = out[f"front_{target}_context_prior"].isna()
            out.loc[missing, f"front_{target}_context_prior"] = out.loc[missing, f"{prefix}_{target}_prior"]
            out.loc[missing, f"front_{target}_context_count"] = out.loc[missing, f"{prefix}_prior_count"]
        out[f"front_{target}_context_prior"] = out[f"front_{target}_context_prior"].fillna(
            out[f"global_{target}_prior"]
        )

    out["front_survival_context_prior"] = out["front_survival_context_prior"].fillna(out["actual_front_survival"].mean())
    out["front_collapse_context_prior"] = out["front_collapse_context_prior"].fillna(out["actual_front_collapse"].mean())
    out["front_top3_front5_share_context_prior"] = out["front_top3_front5_share_context_prior"].fillna(
        out["actual_top3_front5_share"].mean()
    )
    out["front_survival_edge_vs_global"] = (
        out["front_survival_context_prior"] - out["global_survival_prior"].fillna(out["actual_front_survival"].mean())
    )
    out["front_collapse_edge_vs_global"] = (
        out["front_collapse_context_prior"] - out["global_collapse_prior"].fillna(out["actual_front_collapse"].mean())
    )
    out["front_survival_despite_pressure_score"] = (
        out["pre_high_pressure_signal"] * (0.70 * out["front_survival_context_prior"] + 0.30 * out["front_top3_front5_share_context_prior"])
        - 0.30 * out["front_collapse_context_prior"]
    )
    out["front_collapse_reinforced_score"] = (
        out["pre_high_pressure_signal"] * (0.75 * out["front_collapse_context_prior"] + 0.25 * (1 - out["front_survival_context_prior"]))
    )
    out["front_context_readability_score"] = (
        (out["front_survival_context_prior"] - out["front_collapse_context_prior"]).abs()
        * (out["front_survival_context_count"].fillna(0).clip(upper=50) / 50.0)
    )
    return out


def rank_auc(y_true: pd.Series, score: pd.Series) -> float:
    data = pd.DataFrame({"y": pd.to_numeric(y_true, errors="coerce"), "s": pd.to_numeric(score, errors="coerce")}).dropna()
    if data.empty or data["y"].nunique() < 2:
        return float("nan")
    pos = data["y"].eq(1)
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    ranks = data["s"].rank(method="average")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def race_score_report(test_races: pd.DataFrame) -> pd.DataFrame:
    score_cols = {
        "collapse_risk_inverse": 1.0 - pd.to_numeric(test_races.get("race_pace_collapse_risk"), errors="coerce"),
        "pre_high_pressure_inverse": 1.0 - test_races["pre_high_pressure_signal"],
        "context_survival_prior": test_races["front_survival_context_prior"],
        "survival_despite_pressure_score": test_races["front_survival_despite_pressure_score"],
        "collapse_reinforced_inverse": 1.0 - test_races["front_collapse_reinforced_score"],
        "readability_score": test_races["front_context_readability_score"],
    }
    rows = []
    for name, score in score_cols.items():
        rows.append(
            {
                "score": name,
                "auc_actual_front_survival": rank_auc(test_races["actual_front_survival"], score),
                "auc_winner_front5": rank_auc(test_races["winner_front5"], score),
                "top20_survival_rate": float(test_races.loc[score.rank(pct=True).ge(0.80), "actual_front_survival"].mean()),
                "bottom20_survival_rate": float(test_races.loc[score.rank(pct=True).le(0.20), "actual_front_survival"].mean()),
            }
        )
    return pd.DataFrame(rows)


def surface_type(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str)
    return pd.Series(
        np.select(
            [
                s.str.contains("芝", regex=False) | s.str.lower().isin(["turf", "grass"]),
                s.str.contains("ダ", regex=False) | s.str.lower().isin(["dirt", "sand"]),
            ],
            ["turf", "dirt"],
            default="unknown",
        ),
        index=series.index,
        dtype=object,
    )


def build_context_lookup(races: pd.DataFrame) -> pd.DataFrame:
    use = races.copy()
    use["venue_code"] = use["race_id"].astype(str).str.zfill(16).str.slice(8, 10)
    use["surface_type"] = surface_type(use["surface"])
    specs = [
        ("full", ["venue_code", "surface_type", "distance_bin", "class_group", "going_group"]),
        ("course_class", ["venue_code", "surface_type", "distance_bin", "class_group"]),
        ("course", ["venue_code", "surface_type", "distance_bin"]),
        ("venue_surface", ["venue_code", "surface_type"]),
        ("surface_distance", ["surface_type", "distance_bin"]),
        ("global", []),
    ]
    rows: list[pd.DataFrame] = []
    for level, keys in specs:
        if keys:
            grouped = (
                use.groupby(keys, dropna=False)
                .agg(
                    context_races=("race_id", "nunique"),
                    front_survival_rate=("actual_front_survival", "mean"),
                    front_collapse_rate=("actual_front_collapse", "mean"),
                    top3_front5_share=("actual_top3_front5_share", "mean"),
                    avg_high_pressure_signal=("pre_high_pressure_signal", "mean"),
                )
                .reset_index()
            )
        else:
            grouped = pd.DataFrame(
                [
                    {
                        "context_races": use["race_id"].nunique(),
                        "front_survival_rate": use["actual_front_survival"].mean(),
                        "front_collapse_rate": use["actual_front_collapse"].mean(),
                        "top3_front5_share": use["actual_top3_front5_share"].mean(),
                        "avg_high_pressure_signal": use["pre_high_pressure_signal"].mean(),
                    }
                ]
            )
        grouped["lookup_level"] = level
        for col in ["venue_code", "surface_type", "distance_bin", "class_group", "going_group"]:
            if col not in grouped.columns:
                grouped[col] = "*"
            grouped[col] = grouped[col].fillna("*").astype(str)
        rows.append(grouped)
    out = pd.concat(rows, ignore_index=True, sort=False)
    out["front_survival_edge_vs_global"] = out["front_survival_rate"] - float(use["actual_front_survival"].mean())
    out["front_collapse_edge_vs_global"] = out["front_collapse_rate"] - float(use["actual_front_collapse"].mean())
    out["front_context_readability_score"] = (
        (out["front_survival_rate"] - out["front_collapse_rate"]).abs()
        * (out["context_races"].clip(upper=50) / 50.0)
    ).clip(0.0, 1.0)
    return out[
        [
            "lookup_level",
            "venue_code",
            "surface_type",
            "distance_bin",
            "class_group",
            "going_group",
            "context_races",
            "front_survival_rate",
            "front_collapse_rate",
            "top3_front5_share",
            "avg_high_pressure_signal",
            "front_survival_edge_vs_global",
            "front_collapse_edge_vs_global",
            "front_context_readability_score",
        ]
    ]


def ticket_metrics(frame: pd.DataFrame, policy: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "policy": policy,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi": np.nan,
            "hit_rate": np.nan,
            "top_return_share": np.nan,
            "roi_ex_top1": np.nan,
        }
    stake = pd.to_numeric(frame.get("runtime_stake_yen"), errors="coerce").fillna(
        pd.to_numeric(frame.get("stake_yen"), errors="coerce")
    ).fillna(0.0)
    ret = pd.to_numeric(frame.get("runtime_return_yen"), errors="coerce").fillna(
        pd.to_numeric(frame.get("return_yen"), errors="coerce")
    ).fillna(0.0)
    stake_sum = float(stake.sum())
    ret_sum = float(ret.sum())
    if ret_sum > 0 and len(frame) > 1:
        top_i = int(ret.to_numpy().argmax())
        top_return_share = float(ret.max() / ret_sum)
        roi_ex_top1 = safe_div(ret_sum - float(ret.iloc[top_i]), stake_sum - float(stake.iloc[top_i]))
    else:
        top_return_share = float("nan")
        roi_ex_top1 = float("nan")
    return {
        "policy": policy,
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()) if "race_id" in frame else 0,
        "stake_yen": stake_sum,
        "return_yen": ret_sum,
        "profit_yen": ret_sum - stake_sum,
        "roi": safe_div(ret_sum, stake_sum),
        "hit_rate": float(ret.gt(0).mean()) if len(frame) else float("nan"),
        "top_return_share": top_return_share,
        "roi_ex_top1": roi_ex_top1,
    }


def evaluate_ticket_policies(tickets: pd.DataFrame) -> pd.DataFrame:
    front_dep = pd.to_numeric(tickets.get("projected_front5_prob"), errors="coerce").fillna(
        pd.to_numeric(tickets.get("front_advantage_score_feature"), errors="coerce")
    ).fillna(0.0)
    survival = pd.to_numeric(tickets["front_survival_despite_pressure_score"], errors="coerce")
    collapse = pd.to_numeric(tickets["front_collapse_reinforced_score"], errors="coerce")
    pressure = pd.to_numeric(tickets["pre_high_pressure_signal"], errors="coerce")
    readability = pd.to_numeric(tickets["front_context_readability_score"], errors="coerce")

    q = {
        "survival_q60": float(survival.quantile(0.60)),
        "survival_q70": float(survival.quantile(0.70)),
        "collapse_q70": float(collapse.quantile(0.70)),
        "collapse_q80": float(collapse.quantile(0.80)),
        "pressure_q60": float(pressure.quantile(0.60)),
        "front_dep_q60": float(front_dep.quantile(0.60)),
        "readability_q50": float(readability.quantile(0.50)),
    }
    masks = {
        "base_all": pd.Series(True, index=tickets.index),
        "front_survival_context_q60": survival.ge(q["survival_q60"]),
        "front_survival_context_q70": survival.ge(q["survival_q70"]),
        "high_pressure_survival_support": pressure.ge(q["pressure_q60"]) & survival.ge(q["survival_q60"]),
        "avoid_collapse_context_q80": collapse.le(q["collapse_q80"]),
        "avoid_collapse_context_q70": collapse.le(q["collapse_q70"]),
        "front_pair_survival_support": front_dep.ge(q["front_dep_q60"]) & survival.ge(q["survival_q60"]),
        "front_pair_collapse_avoid": ~(front_dep.ge(q["front_dep_q60"]) & collapse.ge(q["collapse_q70"])),
        "readable_survival_support": readability.ge(q["readability_q50"]) & survival.ge(q["survival_q60"]),
        "pressure_context_gate": ~(
            pressure.ge(q["pressure_q60"]) & survival.lt(q["survival_q60"]) & collapse.ge(q["collapse_q70"])
        ),
    }
    rows = []
    by_year_rows = []
    for name, mask in masks.items():
        sub = tickets.loc[mask.fillna(False)].copy()
        rows.append(ticket_metrics(sub, name))
        for year, group in sub.groupby("year", dropna=False):
            row = ticket_metrics(group, name)
            row["year"] = str(year)
            by_year_rows.append(row)
    out = pd.DataFrame(rows)
    out["year"] = "ALL"
    by_year = pd.DataFrame(by_year_rows)
    if not by_year.empty:
        by_year = by_year[["policy", "year", *[c for c in by_year.columns if c not in {"policy", "year"}]]]
    out.attrs["thresholds"] = q
    return out, by_year


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--tickets", type=Path, default=DEFAULT_TICKETS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train = read_csv(args.train)
    test = read_csv(args.test)
    tickets = read_csv(args.tickets)

    race_all = pd.concat([build_race_table(train, "train"), build_race_table(test, "test")], ignore_index=True, sort=False)
    race_all = add_context_priors(race_all)
    test_races = race_all[race_all["split"].eq("test")].copy()
    race_scores = race_score_report(test_races)

    tickets = tickets.copy()
    tickets["race_id"] = norm_race_id(tickets["race_id"])
    merge_cols = [
        "race_id",
        "date_key",
        "venue",
        "surface",
        "distance_bin",
        "class_group",
        "going_group",
        "pre_high_pressure_signal",
        "front_survival_context_prior",
        "front_collapse_context_prior",
        "front_top3_front5_share_context_prior",
        "front_survival_context_count",
        "front_survival_edge_vs_global",
        "front_collapse_edge_vs_global",
        "front_survival_despite_pressure_score",
        "front_collapse_reinforced_score",
        "front_context_readability_score",
        "actual_front_survival",
        "actual_front_collapse",
        "actual_top3_front5_share",
    ]
    ticket_ctx = tickets.merge(test_races[[c for c in merge_cols if c in test_races.columns]], on="race_id", how="left")
    policies, by_year = evaluate_ticket_policies(ticket_ctx)

    segment_cols = [
        "venue",
        "surface",
        "distance_bin",
        "class_group",
        "going_group",
    ]
    segment_rows = []
    for col in segment_cols:
        for value, group in test_races.groupby(col, dropna=False):
            if len(group) < 20:
                continue
            segment_rows.append(
                {
                    "segment": col,
                    "value": str(value),
                    "races": int(len(group)),
                    "actual_front_survival_rate": float(group["actual_front_survival"].mean()),
                    "actual_front_collapse_rate": float(group["actual_front_collapse"].mean()),
                    "avg_high_pressure_signal": float(group["pre_high_pressure_signal"].mean()),
                    "avg_context_survival_prior": float(group["front_survival_context_prior"].mean()),
                    "avg_context_collapse_prior": float(group["front_collapse_context_prior"].mean()),
                }
            )
    segments = pd.DataFrame(segment_rows).sort_values(
        ["actual_front_survival_rate", "races"], ascending=[False, False]
    )
    lookup = build_context_lookup(race_all)

    race_all.to_csv(args.out_dir / "race_front_survival_context.csv", index=False, encoding="utf-8-sig")
    ticket_ctx.to_csv(args.out_dir / "tickets_with_front_survival_context.csv", index=False, encoding="utf-8-sig")
    race_scores.to_csv(args.out_dir / "race_score_auc.csv", index=False, encoding="utf-8-sig")
    policies.to_csv(args.out_dir / "policy_metrics.csv", index=False, encoding="utf-8-sig")
    by_year.to_csv(args.out_dir / "policy_metrics_by_year.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(args.out_dir / "front_survival_segments.csv", index=False, encoding="utf-8-sig")
    lookup.to_csv(args.out_dir / "front_survival_context_lookup.csv", index=False, encoding="utf-8-sig")

    summary = {
        "output_dir": str(args.out_dir.relative_to(ROOT)),
        "train_races": int(race_all["split"].eq("train").sum()),
        "test_races": int(race_all["split"].eq("test").sum()),
        "test_front_survival_rate": float(test_races["actual_front_survival"].mean()),
        "test_front_collapse_rate": float(test_races["actual_front_collapse"].mean()),
        "race_score_auc": race_scores.to_dict(orient="records"),
        "ticket_policy_metrics": policies.to_dict(orient="records"),
        "thresholds": policies.attrs.get("thresholds", {}),
        "top_segments": segments.head(20).to_dict(orient="records"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Front Survival Context v1",
                "",
                "Purpose: verify whether high-pace races can be separated into front-survival and front-collapse contexts.",
                "Prediction-side features use only historical course/surface/distance/class/going priors and pre-race pressure signals.",
                "Actual 4th-corner position is used only as the post-race label.",
                "",
                "Key files:",
                "- race_front_survival_context.csv",
                "- tickets_with_front_survival_context.csv",
                "- race_score_auc.csv",
                "- policy_metrics.csv",
                "- policy_metrics_by_year.csv",
                "- front_survival_segments.csv",
                "- front_survival_context_lookup.csv",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
