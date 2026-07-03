from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "analysis" / "queue_shape_race_quality_v1"


DEFAULT_FEATURE_CSVS = [
    ROOT / "data" / "datasets" / "cache" / "pace_style_time" / "train_features.csv",
    ROOT / "data" / "datasets" / "cache" / "pace_style_time" / "test_features.csv",
]

DEFAULT_TICKET_CSVS = [
    ROOT / "outputs" / "analysis" / "purged_walkforward_mcs_pbo_rebuilt_20260623" / "purged_walkforward_selected_tickets.csv",
    ROOT / "outputs" / "analysis" / "dynamic_pair_ticket_allocation_rebuilt_20260623" / "walkforward_selected_tickets.csv",
    ROOT
    / "outputs"
    / "analysis"
    / "race_day_runtime_operation_skip03119_smoke_v1"
    / "mcs_pbo_overlay"
    / "hybrid_mcs_core_pbo_boost_selected_tickets.csv",
    ROOT
    / "outputs"
    / "analysis"
    / "race_day_runtime_operation_skip03119_smoke_v1"
    / "mcs_pbo_overlay"
    / "pbo_front_overlay_s0334_selected_tickets.csv",
    ROOT / "outputs" / "analysis" / "dynamic_pair_stake_sizing_v1" / "stake_sizing_tickets.csv",
]


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, **kwargs)


def pick_col(frame: pd.DataFrame, candidates: list[str], contains: str | None = None) -> str | None:
    for col in candidates:
        if col in frame.columns:
            return col
    if contains:
        for col in frame.columns:
            if contains in str(col):
                return col
    return None


def num(series: pd.Series | Any, default: float = 0.0) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").fillna(default)
    return pd.Series(default)


def sigmoid(x: pd.Series | np.ndarray) -> pd.Series:
    arr = np.asarray(x, dtype=float)
    return pd.Series(1.0 / (1.0 + np.exp(-np.clip(arr, -30, 30))))


def second_largest(s: pd.Series) -> float:
    vals = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) < 2:
        return 0.0
    vals = np.sort(vals)
    return float(vals[-2])


def top3_mean(s: pd.Series) -> float:
    vals = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return 0.0
    vals = np.sort(vals)
    return float(vals[-min(3, len(vals)) :].mean())


def build_feature_frame(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            continue
        frame = read_csv(path, dtype=str)
        frame["_source_file"] = str(path.relative_to(ROOT))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    race_col = pick_col(df, ["race_id", "レースID(新/馬番無)"], contains="レースID")
    horse_col = pick_col(df, ["horse_no", "馬番"])
    finish_col = pick_col(df, ["finish", "確定着順", "着順"])
    corner4_col = pick_col(df, ["corner4", "4角.1", "4角"])
    field_col = pick_col(df, ["field_size", "出走頭数", "頭数"])
    if not race_col or not horse_col:
        raise ValueError("Feature CSV does not contain race/horse identifiers.")

    out = pd.DataFrame(
        {
            "race_id": df[race_col].astype(str).str.replace(r"\.0$", "", regex=True),
            "horse_no": num(df[horse_col], np.nan),
            "finish": num(df[finish_col], np.nan) if finish_col else np.nan,
            "corner4": num(df[corner4_col], np.nan) if corner4_col else np.nan,
            "field_size": num(df[field_col], np.nan) if field_col else np.nan,
            "front_running_tendency": num(df.get("front_running_tendency", pd.Series(np.nan, index=df.index)), 0.0),
            "horse_front_run_rate_past5": num(df.get("horse_front_run_rate_past5", pd.Series(np.nan, index=df.index)), np.nan),
            "prev_corner4_position_rate": num(df.get("prev_corner4_position_rate", pd.Series(np.nan, index=df.index)), 0.5),
            "race_early_pressure_score": num(df.get("race_early_pressure_score", pd.Series(np.nan, index=df.index)), 0.0),
            "race_slow_pace_risk": num(df.get("race_slow_pace_risk", pd.Series(np.nan, index=df.index)), 0.0),
            "race_pace_collapse_risk": num(df.get("race_pace_collapse_risk", pd.Series(np.nan, index=df.index)), 0.0),
            "horse_closer_rate_past5": num(df.get("horse_closer_rate_past5", pd.Series(np.nan, index=df.index)), 0.0),
            "closing_tendency": num(df.get("closing_tendency", pd.Series(np.nan, index=df.index)), 0.0),
            "pace_fit_score": num(df.get("pace_fit_score", pd.Series(np.nan, index=df.index)), 0.0),
            "front_advantage_score": num(df.get("front_advantage_score", pd.Series(np.nan, index=df.index)), 0.0),
            "draw_pace_fit_score": num(df.get("draw_pace_fit_score", pd.Series(np.nan, index=df.index)), 0.0),
            "target_score": num(df.get("target_score", pd.Series(np.nan, index=df.index)), np.nan),
            "target_top3": num(df.get("target_top3", pd.Series(np.nan, index=df.index)), np.nan),
            "target_win": num(df.get("target_win", pd.Series(np.nan, index=df.index)), np.nan),
            "popularity": num(df.get("人気", pd.Series(np.nan, index=df.index)), np.nan),
            "win_odds": num(df.get("単勝オッズ", pd.Series(np.nan, index=df.index)), np.nan),
            "win_pay": num(df.get("単勝配当", pd.Series(np.nan, index=df.index)), 0.0),
            "place_pay": num(df.get("複勝配当", pd.Series(np.nan, index=df.index)), 0.0),
        }
    )
    out = out[out["race_id"].str.len().ge(12) & out["horse_no"].notna()].copy()
    out["horse_no"] = out["horse_no"].astype(int)
    out["field_size"] = out["field_size"].fillna(out.groupby("race_id")["horse_no"].transform("size")).clip(lower=1.0)
    return out


def add_queue_shape(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    race_id = out["race_id"].astype(str)
    front_intent = out["horse_front_run_rate_past5"].fillna(out["front_running_tendency"]).fillna(0.0).clip(0.0, 1.0)
    prev_c4 = out["prev_corner4_position_rate"].fillna(0.5).clip(0.0, 1.0)
    front_rank = front_intent.groupby(race_id).rank(pct=True).fillna(0.5).clip(0.0, 1.0)
    # Historical approximation of the runtime "lead/tempo assertion" score.
    lead_score = (0.60 * front_intent + 0.25 * front_rank + 0.15 * (1.0 - prev_c4)).clip(0.0, 1.0)
    field = out["field_size"].fillna(lead_score.groupby(race_id).transform("size")).clip(lower=1.0)
    top1 = lead_score.groupby(race_id).transform("max").fillna(0.0).clip(0.0, 1.0)
    top2 = lead_score.groupby(race_id).transform(second_largest).fillna(0.0).clip(0.0, 1.0)
    top3 = lead_score.groupby(race_id).transform(top3_mean).fillna(0.0).clip(0.0, 1.0)
    gap = (top1 - top2).clip(0.0, 1.0)
    candidate_count = lead_score.ge(0.52).astype(float).groupby(race_id).transform("sum").fillna(0.0)
    near_count = lead_score.ge(0.45).astype(float).groupby(race_id).transform("sum").fillna(0.0)
    density = (near_count / field).clip(0.0, 1.0)
    pressure = out["race_early_pressure_score"].fillna(0.0).clip(0.0, 1.0)
    clarity = sigmoid(
        5.2 * (gap - 0.12)
        + 1.5 * (top1 - 0.55)
        - 1.15 * (density - 0.25)
        - 0.42 * candidate_count.sub(1.0).clip(lower=0.0)
    ).set_axis(out.index)
    duel = sigmoid(
        4.4 * (0.13 - gap)
        + 0.80 * candidate_count.sub(1.0)
        + 1.15 * pressure
        + 0.55 * (top3 - 0.50)
        - 0.55 * (top1 - 0.78).clip(lower=0.0)
    ).set_axis(out.index)
    projected_load = (0.42 * pressure + 0.30 * duel + 0.18 * (lead_score.groupby(race_id).transform("sum") / field).clip(0.0, 1.0) + 0.10 * top3).clip(0.0, 1.0)
    labels = np.select(
        [
            (top1.lt(0.42) | candidate_count.le(0)),
            (top1.ge(0.56) & gap.ge(0.18) & candidate_count.le(2)),
            (candidate_count.ge(3) & gap.le(0.12)),
            (candidate_count.ge(2) & gap.le(0.10)),
        ],
        ["no_clear_leader", "single_leader_clear", "front_duel_dense", "matched_speed_duel"],
        default="mixed_queue",
    )
    out["queue_lead_score"] = lead_score
    out["queue_top_gap"] = gap
    out["queue_candidate_count"] = candidate_count
    out["queue_clarity_score"] = clarity.clip(0.0, 1.0)
    out["queue_duel_risk_score"] = duel.clip(0.0, 1.0)
    out["queue_front_load_score"] = projected_load
    out["queue_shape_label"] = labels
    closer = (0.55 * out["horse_closer_rate_past5"].fillna(0.0).clip(0.0, 1.0) + 0.45 * out["closing_tendency"].fillna(0.0).clip(0.0, 1.0)).clip(0.0, 1.0)
    pace_fit = out["pace_fit_score"].fillna(0.0).clip(0.0, 1.0)
    front_adv = out["front_advantage_score"].fillna(0.0).clip(0.0, 1.0)
    draw_fit = out["draw_pace_fit_score"].fillna(0.0).clip(0.0, 1.0)
    collapse = out["race_pace_collapse_risk"].fillna(0.0).clip(0.0, 1.0)
    slow = out["race_slow_pace_risk"].fillna(0.0).clip(0.0, 1.0)
    front_fit = (0.42 * lead_score + 0.22 * front_adv + 0.18 * pace_fit + 0.12 * draw_fit + 0.06 * slow).clip(0.0, 1.0)
    closer_fit = (0.38 * closer + 0.24 * collapse + 0.16 * duel + 0.12 * pace_fit + 0.10 * draw_fit).clip(0.0, 1.0)
    shape = pd.Series(labels, index=out.index)
    shape_fit = pd.Series(0.5, index=out.index, dtype=float)
    shape_fit = shape_fit.where(~shape.eq("single_leader_clear"), front_fit)
    shape_fit = shape_fit.where(~shape.eq("no_clear_leader"), (0.70 * front_fit + 0.30 * pace_fit).clip(0.0, 1.0))
    shape_fit = shape_fit.where(~shape.eq("front_duel_dense"), (0.68 * closer_fit + 0.32 * front_fit).clip(0.0, 1.0))
    shape_fit = shape_fit.where(~shape.eq("matched_speed_duel"), (0.58 * closer_fit + 0.42 * front_fit).clip(0.0, 1.0))
    shape_fit = shape_fit.where(~shape.eq("mixed_queue"), (0.50 * closer_fit + 0.50 * front_fit).clip(0.0, 1.0))
    front_burn_risk = (duel * lead_score * (1.0 - closer)).clip(0.0, 1.0)
    dead_slow_closer_risk = ((1.0 - duel) * slow * closer * (1.0 - lead_score)).clip(0.0, 1.0)
    out["runner_shape_front_fit_score"] = front_fit
    out["runner_shape_closer_fit_score"] = closer_fit
    out["runner_shape_fit_score"] = shape_fit.clip(0.0, 1.0)
    out["runner_shape_mismatch_risk_score"] = np.maximum(front_burn_risk, dead_slow_closer_risk).clip(0.0, 1.0)
    return out


def race_level_summary(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for race_id, g in features.groupby("race_id"):
        field = float(g["field_size"].dropna().iloc[0]) if g["field_size"].notna().any() else float(len(g))
        top3 = g[g["finish"].between(1, 3)].copy()
        if top3.empty:
            continue
        front_line = max(5.0, field * 0.40)
        closer_line = field * 0.65
        front_count = int(top3["corner4"].le(front_line).sum())
        closer_count = int(top3["corner4"].ge(closer_line).sum())
        avg_c4 = float(top3["corner4"].mean()) if top3["corner4"].notna().any() else np.nan
        actual_shape = "mixed"
        if front_count >= 2 or (pd.notna(avg_c4) and avg_c4 <= max(5.0, field * 0.45)):
            actual_shape = "front_stalker"
        elif closer_count >= 2 or (pd.notna(avg_c4) and avg_c4 >= field * 0.55):
            actual_shape = "closer"
        first = g.iloc[0]
        rows.append(
            {
                "race_id": race_id,
                "field_size": field,
                "queue_shape_label": first["queue_shape_label"],
                "queue_clarity_score": float(first["queue_clarity_score"]),
                "queue_duel_risk_score": float(first["queue_duel_risk_score"]),
                "queue_front_load_score": float(first["queue_front_load_score"]),
                "queue_top_gap": float(first["queue_top_gap"]),
                "queue_candidate_count": float(first["queue_candidate_count"]),
                "actual_shape": actual_shape,
                "actual_front_stalker": int(actual_shape == "front_stalker"),
                "actual_closer": int(actual_shape == "closer"),
                "top3_avg_corner4": avg_c4,
                "top3_avg_corner4_rate": avg_c4 / field if pd.notna(avg_c4) and field else np.nan,
                "winner_corner4": float(g.loc[g["finish"].eq(1), "corner4"].iloc[0]) if g["finish"].eq(1).any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize_races(races: pd.DataFrame) -> pd.DataFrame:
    def block(name: str, frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "segment": name,
            "races": int(frame["race_id"].nunique()),
            "front_stalker_rate_pct": round(float(frame["actual_front_stalker"].mean()) * 100, 1) if len(frame) else 0.0,
            "closer_rate_pct": round(float(frame["actual_closer"].mean()) * 100, 1) if len(frame) else 0.0,
            "avg_top3_corner4_rate_pct": round(float(frame["top3_avg_corner4_rate"].mean()) * 100, 1) if len(frame) else np.nan,
            "avg_duel_risk": round(float(frame["queue_duel_risk_score"].mean()), 3) if len(frame) else np.nan,
            "avg_clarity": round(float(frame["queue_clarity_score"].mean()), 3) if len(frame) else np.nan,
        }

    rows = [block("all", races)]
    for label, frame in races.groupby("queue_shape_label"):
        rows.append(block(f"shape:{label}", frame))
    q_clarity = races["queue_clarity_score"].quantile(0.75)
    q_duel = races["queue_duel_risk_score"].quantile(0.75)
    rows.append(block("clarity_top25", races[races["queue_clarity_score"].ge(q_clarity)]))
    rows.append(block("duel_risk_top25", races[races["queue_duel_risk_score"].ge(q_duel)]))
    rows.append(block("duel_low25", races[races["queue_duel_risk_score"].le(races["queue_duel_risk_score"].quantile(0.25))]))
    return pd.DataFrame(rows)


def first_existing(frame: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def ticket_metrics(frame: pd.DataFrame, name: str) -> dict[str, Any]:
    if frame.empty:
        return {"segment": name, "tickets": 0, "races": 0, "stake_yen": 0.0, "return_yen": 0.0, "roi_pct": 0.0, "hit_rate_pct": 0.0}
    stake = frame["_stake"].sum()
    ret = frame["_return"].sum()
    return {
        "segment": name,
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()),
        "stake_yen": round(float(stake), 1),
        "return_yen": round(float(ret), 1),
        "roi_pct": round(float(ret / stake * 100), 1) if stake > 0 else 0.0,
        "hit_rate_pct": round(float(frame["_hit"].mean() * 100), 1),
    }


def load_ticket_file(path: Path) -> pd.DataFrame:
    df = read_csv(path, dtype={"race_id": str})
    if df.empty or "race_id" not in df.columns:
        return pd.DataFrame()
    if "stake_policy" in df.columns:
        preferred = df[df["stake_policy"].astype(str).eq("hit_balanced")]
        if not preferred.empty:
            df = preferred.copy()
    if "budget_yen" in df.columns:
        preferred = df[pd.to_numeric(df["budget_yen"], errors="coerce").fillna(0).eq(5000)]
        if not preferred.empty:
            df = preferred.copy()
    stake_col = first_existing(df, ["policy_stake_yen", "runtime_stake_yen", "stake_yen", "eval_stake_yen"])
    ret_col = first_existing(df, ["policy_return_yen", "return_yen", "actual_return_yen", "eval_return_yen"])
    hit_col = first_existing(df, ["hit", "wide_hit", "umaren_hit"])
    if stake_col is None or ret_col is None:
        return pd.DataFrame()
    out = df.copy()
    out["_stake"] = pd.to_numeric(out[stake_col], errors="coerce").fillna(0.0)
    out["_return"] = pd.to_numeric(out[ret_col], errors="coerce").fillna(0.0)
    if hit_col:
        out["_hit"] = out[hit_col].astype(str).str.lower().isin(["true", "1", "yes"])
        out["_hit"] = out["_hit"] | out["_return"].gt(0)
    else:
        out["_hit"] = out["_return"].gt(0)
    out = out[out["_stake"].gt(0)].copy()
    out["_ticket_source"] = str(path.relative_to(ROOT))
    return out


def summarize_tickets(ticket_paths: list[Path], races: pd.DataFrame) -> pd.DataFrame:
    rows = []
    race_cols = [
        "race_id",
        "queue_shape_label",
        "queue_clarity_score",
        "queue_duel_risk_score",
        "queue_front_load_score",
        "queue_top_gap",
        "queue_candidate_count",
        "actual_shape",
    ]
    q_clarity = races["queue_clarity_score"].quantile(0.75)
    q_duel = races["queue_duel_risk_score"].quantile(0.75)
    for path in ticket_paths:
        if not path.exists():
            continue
        tickets = load_ticket_file(path)
        if tickets.empty:
            continue
        merged = tickets.merge(races[race_cols], on="race_id", how="inner")
        if merged.empty:
            continue
        source = path.parent.name + "/" + path.name
        segments = {
            "all": merged,
            "single_leader_clear": merged[merged["queue_shape_label"].eq("single_leader_clear")],
            "front_duel_dense": merged[merged["queue_shape_label"].eq("front_duel_dense")],
            "matched_speed_duel": merged[merged["queue_shape_label"].eq("matched_speed_duel")],
            "no_clear_leader": merged[merged["queue_shape_label"].eq("no_clear_leader")],
            "mixed_queue": merged[merged["queue_shape_label"].eq("mixed_queue")],
            "clarity_top25": merged[merged["queue_clarity_score"].ge(q_clarity)],
            "duel_risk_top25": merged[merged["queue_duel_risk_score"].ge(q_duel)],
            "front_load_top25": merged[merged["queue_front_load_score"].ge(races["queue_front_load_score"].quantile(0.75))],
        }
        front_col = first_existing(merged, ["projected_front5_prob", "partner_front5_model_prob", "front5_model_prob"])
        if front_col:
            front_mask = pd.to_numeric(merged[front_col], errors="coerce").fillna(0.0).ge(0.60)
            segments["front_ticket_in_clear_queue"] = merged[front_mask & merged["queue_shape_label"].eq("single_leader_clear")]
            segments["front_ticket_in_duel_dense"] = merged[front_mask & merged["queue_shape_label"].eq("front_duel_dense")]
            segments["front_ticket_in_low_duel"] = merged[front_mask & merged["queue_duel_risk_score"].le(races["queue_duel_risk_score"].quantile(0.25))]
        for seg, frame in segments.items():
            metric = ticket_metrics(frame, seg)
            metric["source"] = source
            rows.append(metric)
    return pd.DataFrame(rows)


def runner_pick_metrics(frame: pd.DataFrame, name: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "policy": name,
            "races": 0,
            "win_rate_pct": 0.0,
            "top3_rate_pct": 0.0,
            "win_roi_pct": 0.0,
            "place_roi_pct": 0.0,
            "avg_popularity": np.nan,
            "avg_win_odds": np.nan,
        }
    races = int(frame["race_id"].nunique())
    stake = races * 100.0
    win_ret = np.where(frame["finish"].eq(1), pd.to_numeric(frame["win_pay"], errors="coerce").fillna(0.0), 0.0).sum()
    place_ret = np.where(frame["finish"].between(1, 3), pd.to_numeric(frame["place_pay"], errors="coerce").fillna(0.0), 0.0).sum()
    return {
        "policy": name,
        "races": races,
        "win_rate_pct": round(float(frame["finish"].eq(1).mean() * 100), 1),
        "top3_rate_pct": round(float(frame["finish"].between(1, 3).mean() * 100), 1),
        "win_roi_pct": round(float(win_ret / stake * 100), 1) if stake > 0 else 0.0,
        "place_roi_pct": round(float(place_ret / stake * 100), 1) if stake > 0 else 0.0,
        "avg_popularity": round(float(frame["popularity"].mean()), 2),
        "avg_win_odds": round(float(frame["win_odds"].mean()), 2),
    }


def evaluate_runner_pick_shift(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = features.copy()
    work = work[work["target_score"].notna() & work["finish"].notna()].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()
    race_id = work["race_id"].astype(str)
    base_rank_score = work.groupby("race_id")["target_score"].rank(pct=True).fillna(0.5)
    shape_fit = work["runner_shape_fit_score"].fillna(0.5).clip(0.0, 1.0)
    shape_risk = work["runner_shape_mismatch_risk_score"].fillna(0.0).clip(0.0, 1.0)

    rows = []
    picks = []
    for weight in [0.00, 0.04, 0.08, 0.12, 0.16, 0.20]:
        score_col = f"_shape_score_{weight:.2f}"
        work[score_col] = (
            (1.0 - weight) * base_rank_score
            + weight * shape_fit
            - min(0.10, weight * 0.65) * shape_risk
        )
        top = (
            work.sort_values(["race_id", score_col, "target_score"], ascending=[True, False, False])
            .drop_duplicates("race_id", keep="first")
            .copy()
        )
        policy = f"base_plus_shape_w{weight:.2f}" if weight else "base_target_score"
        metric = runner_pick_metrics(top, policy)
        metric["shape_weight"] = weight
        rows.append(metric)
        top["policy"] = policy
        top["shape_weight"] = weight
        picks.append(
            top[
                [
                    "policy",
                    "shape_weight",
                    "race_id",
                    "horse_no",
                    "finish",
                    "popularity",
                    "win_odds",
                    "win_pay",
                    "place_pay",
                    "target_score",
                    "runner_shape_fit_score",
                    "runner_shape_mismatch_risk_score",
                    "queue_shape_label",
                ]
            ]
        )

    base_pick = picks[0][["race_id", "horse_no"]].rename(columns={"horse_no": "base_horse_no"})
    detail = pd.concat(picks, ignore_index=True)
    detail = detail.merge(base_pick, on="race_id", how="left")
    detail["changed_from_base"] = detail["horse_no"].ne(detail["base_horse_no"])
    changed_rows = []
    for policy, g in detail.groupby("policy"):
        changed_metric = runner_pick_metrics(g[g["changed_from_base"]], f"{policy}_changed_only")
        changed_rows.append(
            {
                "policy": policy,
                "changed_races": int(g["changed_from_base"].sum()),
                "changed_rate_pct": round(float(g["changed_from_base"].mean() * 100), 1),
                "changed_win_rate_pct": changed_metric["win_rate_pct"],
                "changed_top3_rate_pct": changed_metric["top3_rate_pct"],
                "changed_win_roi_pct": changed_metric["win_roi_pct"],
                "changed_place_roi_pct": changed_metric["place_roi_pct"],
            }
        )
    summary = pd.DataFrame(rows).merge(pd.DataFrame(changed_rows), on="policy", how="left")
    return summary, detail


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate runtime queue-shape race quality features on historical races and tickets.")
    parser.add_argument("--feature-csv", action="append", default=[], help="Historical runner feature CSV. Can be repeated.")
    parser.add_argument("--ticket-csv", action="append", default=[], help="Historical selected ticket CSV. Can be repeated.")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    feature_paths = [Path(p) for p in args.feature_csv] if args.feature_csv else DEFAULT_FEATURE_CSVS
    ticket_paths = [Path(p) for p in args.ticket_csv] if args.ticket_csv else DEFAULT_TICKET_CSVS
    feature_paths = [p if p.is_absolute() else ROOT / p for p in feature_paths]
    ticket_paths = [p if p.is_absolute() else ROOT / p for p in ticket_paths]
    out_dir = Path(args.output_dir)
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    features = build_feature_frame(feature_paths)
    features = add_queue_shape(features)
    races = race_level_summary(features)
    race_summary = summarize_races(races).sort_values(["segment"])
    ticket_summary = summarize_tickets(ticket_paths, races)
    runner_pick_summary, runner_pick_detail = evaluate_runner_pick_shift(features)
    if not ticket_summary.empty:
        ticket_summary = ticket_summary.sort_values(["source", "segment"])

    feature_light = features[
        [
            "race_id",
            "horse_no",
            "finish",
            "corner4",
            "field_size",
            "queue_lead_score",
            "queue_shape_label",
            "queue_clarity_score",
            "queue_duel_risk_score",
            "queue_front_load_score",
            "queue_top_gap",
            "queue_candidate_count",
            "runner_shape_front_fit_score",
            "runner_shape_closer_fit_score",
            "runner_shape_fit_score",
            "runner_shape_mismatch_risk_score",
        ]
    ]
    feature_light.to_csv(out_dir / "runner_queue_shape_features.csv", index=False, encoding="utf-8-sig")
    races.to_csv(out_dir / "race_queue_shape_validation.csv", index=False, encoding="utf-8-sig")
    race_summary.to_csv(out_dir / "race_queue_shape_summary.csv", index=False, encoding="utf-8-sig")
    ticket_summary.to_csv(out_dir / "ticket_queue_shape_roi_summary.csv", index=False, encoding="utf-8-sig")
    runner_pick_summary.to_csv(out_dir / "runner_shape_adjusted_pick_summary.csv", index=False, encoding="utf-8-sig")
    runner_pick_detail.to_csv(out_dir / "runner_shape_adjusted_pick_detail.csv", index=False, encoding="utf-8-sig")

    def top_records(frame: pd.DataFrame, n: int = 20) -> list[dict[str, Any]]:
        return frame.head(n).replace({np.nan: None}).to_dict(orient="records")

    summary = {
        "feature_rows": int(len(features)),
        "race_rows": int(len(races)),
        "ticket_sources_evaluated": sorted(ticket_summary["source"].unique().tolist()) if not ticket_summary.empty else [],
        "shape_counts": races["queue_shape_label"].value_counts().to_dict() if not races.empty else {},
        "race_summary_top": top_records(race_summary),
        "ticket_summary_top": top_records(ticket_summary.sort_values(["roi_pct", "tickets"], ascending=[False, False])) if not ticket_summary.empty else [],
        "runner_pick_summary": top_records(runner_pick_summary, n=20),
        "note": "Queue-shape features are pre-race approximations. Use them first as diagnostics/guards; do not promote to BUY gates without walk-forward stability.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
