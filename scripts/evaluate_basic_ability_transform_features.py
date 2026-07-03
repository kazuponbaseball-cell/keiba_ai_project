from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


CONTENT_DIR = ROOT / "outputs/analysis/content_bridge_member_features_v1"
SELECTED_TICKETS = ROOT / "outputs/analysis/optimized_stack_ab_s_gate_v1/s_priority_selected_tickets.csv"


def normalize_date(value: object) -> str:
    if pd.isna(value):
        return ""
    digits = re.sub(r"\D", "", str(value))
    if len(digits) == 6:
        return "20" + digits
    if len(digits) >= 8:
        return digits[:8]
    return ""


def to_num(series: pd.Series | None, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default)
    return pd.to_numeric(series, errors="coerce")


def clip01(values: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return np.clip(values, 0.0, 1.0)


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def available_usecols(path: Path, wanted: list[str]) -> list[str]:
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    return [c for c in wanted if c in header.columns]


def percentile_rank_train_based(train: pd.Series, values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    ref = pd.to_numeric(train, errors="coerce").dropna().sort_values().to_numpy()
    vals = pd.to_numeric(values, errors="coerce")
    if len(ref) == 0:
        return pd.Series(0.5, index=values.index)
    ranks = np.searchsorted(ref, vals.fillna(np.nanmedian(ref)).to_numpy(), side="right") / len(ref)
    out = pd.Series(ranks, index=values.index).clip(0.0, 1.0)
    return out if higher_is_better else 1.0 - out


def roc_auc(y_true: pd.Series, score: pd.Series) -> float:
    y = pd.to_numeric(y_true, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    mask = y.notna() & s.notna()
    y = y.loc[mask].astype(int)
    s = s.loc[mask]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = s.rank(method="average")
    pos_rank_sum = float(ranks.loc[y == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def summarize_runner_segment(df: pd.DataFrame, mask: pd.Series, segment: str) -> dict[str, object]:
    seg = df.loc[mask.fillna(False)].copy()
    if seg.empty:
        return {
            "segment": segment,
            "races": 0,
            "horses": 0,
            "win_rate": np.nan,
            "top3_rate": np.nan,
            "win_roi": np.nan,
            "place_roi": np.nan,
            "avg_popularity": np.nan,
            "avg_odds": np.nan,
        }
    stake = len(seg) * 100.0
    return {
        "segment": segment,
        "races": int(seg["race_id"].nunique()),
        "horses": int(len(seg)),
        "win_rate": float(seg["is_win"].mean()),
        "top3_rate": float(seg["is_top3"].mean()),
        "win_roi": safe_div(float(seg["win_return_100"].sum()), stake),
        "place_roi": safe_div(float(seg["place_return_100"].sum()), stake),
        "avg_popularity": float(seg["popularity"].mean()),
        "avg_odds": float(seg["odds"].mean()),
    }


def summarize_ticket_segment(df: pd.DataFrame, mask: pd.Series, segment: str) -> dict[str, object]:
    seg = df.loc[mask.fillna(False)].copy()
    if seg.empty:
        return {
            "segment": segment,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "roi": np.nan,
            "hit_rate": np.nan,
        }
    stake = float(seg["stake_yen"].sum())
    ret = float(seg["return_yen"].sum())
    return {
        "segment": segment,
        "tickets": int(len(seg)),
        "races": int(seg["race_id"].nunique()),
        "stake_yen": stake,
        "return_yen": ret,
        "roi": safe_div(ret, stake),
        "hit_rate": float((seg["return_yen"] > 0).mean()),
    }


def load_content_features() -> pd.DataFrame:
    wanted = [
        "レースID(新/馬番無)",
        "血統登録番号",
        "日付",
        "日付S",
        "場所",
        "Ｒ",
        "馬番",
        "枠番",
        "人気",
        "単勝オッズ",
        "芝・ダ",
        "距離",
        "クラス名",
        "馬場状態",
        "頭数",
        "出走頭数",
        "斤量",
        "年齢",
        "性別",
        "キャリア",
        "確定着順",
        "target_score",
        "target_win",
        "target_top3",
        "単勝配当",
        "複勝配当",
        "前走人気",
        "前走出走頭数",
        "前走頭数",
        "前走着差タイム",
        "前走上3F地点差",
        "前4角.1",
        "前走上り3F順",
        "前走斤量",
        "前走馬体重",
        "前走馬体重増減",
        "prev_margin_sec",
        "past3_avg_margin_sec",
        "past3_best_margin_sec",
        "prev_race_time_value",
        "prev_time_z_course_distance",
        "prev_time_adjusted_by_day_bias",
        "prev_class_time_value_score",
        "past3_avg_time_value",
        "past3_best_time_value",
        "past3_avg_time_z",
        "prev_performance_vs_member_level",
        "past3_avg_performance_vs_member_level",
        "confirmed_member_level_adjusted_score",
        "prev_content_performance_score",
        "past3_content_performance_score",
        "content_performance_score",
        "prev_race_member_level",
        "past3_avg_race_member_level",
    ]
    frames: list[pd.DataFrame] = []
    for source in ["train", "test"]:
        path = CONTENT_DIR / f"{source}_features_with_content_bridge.csv"
        if not path.exists():
            continue
        usecols = available_usecols(path, wanted)
        df = read_csv(path, dtype=str, usecols=usecols)
        df["source"] = source
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No content bridge feature files found under {CONTENT_DIR}")
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"レースID(新/馬番無)": "race_id"})
    out["race_id"] = out["race_id"].astype(str)
    out["horse_id"] = out.get("血統登録番号", pd.Series("", index=out.index)).fillna("").astype(str)
    out["date_key"] = out.get("日付", pd.Series("", index=out.index)).map(normalize_date)
    out["race_no"] = to_num(out.get("Ｒ"))
    out["horse_no"] = to_num(out.get("馬番"))
    out["popularity"] = to_num(out.get("人気"))
    out["odds"] = to_num(out.get("単勝オッズ"))
    out["finish"] = to_num(out.get("確定着順"))
    out["field_size"] = to_num(out.get("出走頭数")).fillna(to_num(out.get("頭数")))
    out["win_return_100"] = np.where(out["finish"] == 1, to_num(out.get("単勝配当")).fillna(0), 0.0)
    out["place_return_100"] = np.where(out["finish"] <= 3, to_num(out.get("複勝配当")).fillna(0), 0.0)
    out["is_win"] = out["finish"] == 1
    out["is_top3"] = out["finish"] <= 3
    return out


def add_past_score_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_score_num"] = to_num(out.get("target_score"))
    out = out.sort_values(["horse_id", "date_key", "race_no", "horse_no"]).reset_index(drop=True)
    grouped = out.groupby("horse_id", sort=False)
    shifted = []
    for lag in range(1, 6):
        col = f"prev{lag}_target_score_from_history"
        out[col] = grouped["target_score_num"].shift(lag)
        shifted.append(col)
    past3 = out[shifted[:3]]
    past5 = out[shifted]
    weights = np.array([0.50, 0.30, 0.20])
    present = past3.notna().astype(float)
    weighted_sum = past3.fillna(0.0).to_numpy() @ weights
    weight_sum = present.to_numpy() @ weights
    out["recent_weighted_score_3"] = np.divide(
        weighted_sum,
        weight_sum,
        out=np.full_like(weighted_sum, np.nan, dtype=float),
        where=weight_sum > 0,
    )
    out["recent_score_mean_3"] = past3.mean(axis=1)
    out["recent_score_std_3"] = past3.std(axis=1, ddof=0)
    out["recent_score_best_3"] = past3.max(axis=1)
    out["recent_score_worst_3"] = past3.min(axis=1)
    out["recent_score_best_5"] = past5.max(axis=1)
    out["recent_score_worst_5"] = past5.min(axis=1)
    out["recent_score_count_3"] = past3.notna().sum(axis=1)
    out["recent_score_slope_3"] = out["prev1_target_score_from_history"] - out["prev3_target_score_from_history"]
    out["recent_score_jump_vs_mean"] = out["prev1_target_score_from_history"] - out["recent_score_mean_3"]
    out["ability_stability_score_3"] = out["recent_score_mean_3"] - out["recent_score_std_3"].fillna(0.0)
    out["ability_one_shot_score_3"] = out["recent_score_best_3"] - out["recent_score_mean_3"]
    out["ability_ceiling_score_5"] = out["recent_score_best_5"]
    out["ability_floor_score_5"] = out["recent_score_worst_5"]
    return out


def add_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_past_score_features(df)
    prev_field = to_num(out.get("前走出走頭数")).fillna(to_num(out.get("前走頭数"))).replace(0, np.nan)
    denom = (prev_field - 1).replace(0, np.nan)
    prev_pop = to_num(out.get("前走人気"))
    prev_corner4 = to_num(out.get("前4角.1"))
    prev_final3f_rank = to_num(out.get("前走上り3F順"))
    out["prev_popularity_rate"] = prev_pop / prev_field
    out["prev_market_rank_score"] = clip01(1.0 - (prev_pop - 1.0) / denom)
    out["prev_corner4_position_rate"] = prev_corner4 / prev_field
    out["prev_corner4_front_rate"] = clip01(1.0 - (prev_corner4 - 1.0) / denom)
    out["prev_final3f_rank_rate"] = prev_final3f_rank / prev_field
    out["prev_final3f_excellence_rate"] = clip01(1.0 - (prev_final3f_rank - 1.0) / denom)

    prev_margin = to_num(out.get("prev_margin_sec")).fillna(to_num(out.get("前走着差タイム")))
    prev_3f_gap = to_num(out.get("前走上3F地点差"))
    out["prev_margin_score"] = -prev_margin
    out["prev_stretch_gain_sec"] = prev_3f_gap - prev_margin
    out["prev_stretch_gain_positive_flag"] = (out["prev_stretch_gain_sec"] > 0).astype(float)
    out["prev_late_improvement_score"] = out["prev_stretch_gain_sec"] * out["prev_final3f_excellence_rate"].fillna(0.5)

    out["prev_market_underestimated_score"] = (
        out["prev1_target_score_from_history"] - out["prev_market_rank_score"]
    )
    out["prev_market_overestimated_risk"] = (
        out["prev_market_rank_score"] - out["prev1_target_score_from_history"]
    )

    out["current_weight"] = to_num(out.get("斤量"))
    out["prev_weight"] = to_num(out.get("前走斤量"))
    out["prev_body_weight"] = to_num(out.get("前走馬体重"))
    out["weight_burden_ratio_prev_body"] = out["current_weight"] / out["prev_body_weight"].replace(0, np.nan)
    out["prev_weight_burden_ratio"] = out["prev_weight"] / out["prev_body_weight"].replace(0, np.nan)
    out["weight_burden_ratio_change"] = (
        out["weight_burden_ratio_prev_body"] - out["prev_weight_burden_ratio"]
    )

    out["career"] = to_num(out.get("キャリア"))
    out["age"] = to_num(out.get("年齢"))
    out["career_shallow_flag"] = out["career"].between(1, 2).astype(float)
    out["career_growth_zone_flag"] = out["career"].between(3, 5).astype(float)
    out["career_exposed_flag"] = (out["career"] >= 11).astype(float)
    out["age_career_density"] = out["career"] / out["age"].replace(0, np.nan)

    # Build a conservative condition-adjusted ability score from existing safe-ish past-performance fields.
    # Confirmed/member bridge scores are included as separate candidates below because they can be less available in real time.
    train = out["source"].eq("train")
    safe_parts = []
    for col, high_good in [
        ("recent_weighted_score_3", True),
        ("ability_stability_score_3", True),
        ("prev_margin_score", True),
        ("prev_time_adjusted_by_day_bias", True),
        ("prev_class_time_value_score", True),
        ("prev_performance_vs_member_level", True),
    ]:
        if col in out.columns:
            safe_parts.append(percentile_rank_train_based(out.loc[train, col], out[col], high_good))
    if safe_parts:
        out["condition_adjusted_recent_ability_score"] = pd.concat(safe_parts, axis=1).mean(axis=1)
    else:
        out["condition_adjusted_recent_ability_score"] = np.nan

    bridge_parts = []
    for col in [
        "confirmed_member_level_adjusted_score",
        "prev_content_performance_score",
        "past3_content_performance_score",
    ]:
        if col in out.columns:
            bridge_parts.append(percentile_rank_train_based(out.loc[train, col], out[col], True))
    if bridge_parts:
        out["retrospective_member_bridge_ability_score"] = pd.concat(bridge_parts, axis=1).mean(axis=1)
    else:
        out["retrospective_member_bridge_ability_score"] = np.nan

    out["wide_axis_reliability_ability_score"] = (
        0.45 * out["condition_adjusted_recent_ability_score"]
        + 0.35 * percentile_rank_train_based(out.loc[train, "ability_stability_score_3"], out["ability_stability_score_3"], True)
        + 0.20 * percentile_rank_train_based(out.loc[train, "ability_floor_score_5"], out["ability_floor_score_5"], True)
    )
    out["win_ceiling_ability_score"] = (
        0.50 * percentile_rank_train_based(out.loc[train, "ability_ceiling_score_5"], out["ability_ceiling_score_5"], True)
        + 0.30 * percentile_rank_train_based(out.loc[train, "recent_score_slope_3"], out["recent_score_slope_3"], True)
        + 0.20 * out["condition_adjusted_recent_ability_score"]
    )
    out["collapse_risk_score"] = (
        0.45 * percentile_rank_train_based(out.loc[train, "recent_score_std_3"], out["recent_score_std_3"], True)
        + 0.35 * percentile_rank_train_based(out.loc[train, "prev_market_overestimated_risk"], out["prev_market_overestimated_risk"], True)
        + 0.20 * percentile_rank_train_based(out.loc[train, "weight_burden_ratio_prev_body"], out["weight_burden_ratio_prev_body"], True)
    )
    return out


CANDIDATE_FEATURES = [
    "condition_adjusted_recent_ability_score",
    "retrospective_member_bridge_ability_score",
    "recent_weighted_score_3",
    "recent_score_slope_3",
    "recent_score_jump_vs_mean",
    "recent_score_std_3",
    "ability_stability_score_3",
    "ability_one_shot_score_3",
    "ability_ceiling_score_5",
    "ability_floor_score_5",
    "prev_corner4_front_rate",
    "prev_final3f_excellence_rate",
    "prev_stretch_gain_sec",
    "prev_late_improvement_score",
    "prev_market_underestimated_score",
    "prev_market_overestimated_risk",
    "weight_burden_ratio_prev_body",
    "weight_burden_ratio_change",
    "career_shallow_flag",
    "career_growth_zone_flag",
    "career_exposed_flag",
    "age_career_density",
    "wide_axis_reliability_ability_score",
    "win_ceiling_ability_score",
    "collapse_risk_score",
]


def evaluate_runner_candidates(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source, group in df.groupby("source", sort=False):
        group = group[group["finish"].notna()].copy()
        for feature in CANDIDATE_FEATURES:
            if feature not in group.columns:
                continue
            values = to_num(group[feature])
            valid = values.notna()
            if valid.sum() < 200:
                continue
            q90 = values.loc[valid].quantile(0.90)
            q80 = values.loc[valid].quantile(0.80)
            q20 = values.loc[valid].quantile(0.20)
            base = {
                "source": source,
                "feature": feature,
                "valid_rows": int(valid.sum()),
                "auc_top3_raw": roc_auc(group.loc[valid, "is_top3"], values.loc[valid]),
                "auc_win_raw": roc_auc(group.loc[valid, "is_win"], values.loc[valid]),
            }
            for label, mask in [
                ("top10pct", values >= q90),
                ("top20pct", values >= q80),
                ("bottom20pct", values <= q20),
            ]:
                rec = summarize_runner_segment(group, mask & valid, label)
                rows.append({**base, **rec})
            if feature in {
                "prev_market_underestimated_score",
                "wide_axis_reliability_ability_score",
                "win_ceiling_ability_score",
            }:
                rec = summarize_runner_segment(group, (values >= q80) & valid & (group["popularity"] >= 4), "top20pct_pop4plus")
                rows.append({**base, **rec})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["auc_top3_edge"] = out["auc_top3_raw"].map(lambda x: max(x, 1 - x) if pd.notna(x) else np.nan)
        out["auc_win_edge"] = out["auc_win_raw"].map(lambda x: max(x, 1 - x) if pd.notna(x) else np.nan)
    return out


def load_selected_tickets_with_candidates(runner: pd.DataFrame) -> pd.DataFrame:
    if not SELECTED_TICKETS.exists():
        return pd.DataFrame()
    wanted = ["race_id", "anchor_no", "partner_no", "ticket_type", "stake_yen", "return_yen"]
    usecols = available_usecols(SELECTED_TICKETS, wanted)
    tickets = read_csv(SELECTED_TICKETS, dtype=str, usecols=usecols)
    tickets["race_id"] = tickets["race_id"].astype(str)
    tickets["anchor_no"] = to_num(tickets.get("anchor_no"))
    tickets["partner_no"] = to_num(tickets.get("partner_no"))
    tickets["stake_yen"] = to_num(tickets.get("stake_yen")).fillna(0.0)
    tickets["return_yen"] = to_num(tickets.get("return_yen")).fillna(0.0)

    cols = ["race_id", "horse_no", "source"] + [c for c in CANDIDATE_FEATURES if c in runner.columns]
    r = runner[cols].copy()
    # Prefer the test rows when duplicated; selected strongest tickets are mainly OOS-style outputs.
    source_order = {"test": 0, "train": 1}
    r["_source_order"] = r["source"].map(source_order).fillna(9)
    r = r.sort_values("_source_order").drop_duplicates(["race_id", "horse_no"], keep="first")
    anchor = r.add_prefix("anchor_").rename(columns={"anchor_race_id": "race_id", "anchor_horse_no": "anchor_no"})
    partner = r.add_prefix("partner_").rename(columns={"partner_race_id": "race_id", "partner_horse_no": "partner_no"})
    out = tickets.merge(anchor, on=["race_id", "anchor_no"], how="left").merge(
        partner, on=["race_id", "partner_no"], how="left"
    )
    for feature in CANDIDATE_FEATURES:
        a = to_num(out.get(f"anchor_{feature}"))
        p = to_num(out.get(f"partner_{feature}"))
        out[f"pair_avg_{feature}"] = pd.concat([a, p], axis=1).mean(axis=1)
        out[f"pair_min_{feature}"] = pd.concat([a, p], axis=1).min(axis=1)
        out[f"pair_max_{feature}"] = pd.concat([a, p], axis=1).max(axis=1)
    return out


PAIR_FEATURES = [
    "pair_min_wide_axis_reliability_ability_score",
    "pair_avg_condition_adjusted_recent_ability_score",
    "pair_avg_retrospective_member_bridge_ability_score",
    "pair_max_win_ceiling_ability_score",
    "pair_min_ability_floor_score_5",
    "pair_avg_prev_market_underestimated_score",
    "pair_max_prev_market_overestimated_risk",
    "pair_avg_prev_stretch_gain_sec",
    "pair_max_collapse_risk_score",
    "pair_max_weight_burden_ratio_prev_body",
]


def evaluate_ticket_candidates(tickets: pd.DataFrame) -> pd.DataFrame:
    if tickets.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for feature in PAIR_FEATURES:
        if feature not in tickets.columns:
            continue
        values = to_num(tickets[feature])
        valid = values.notna()
        if valid.sum() < 20:
            continue
        q80 = values.loc[valid].quantile(0.80)
        q20 = values.loc[valid].quantile(0.20)
        rows.append({"feature": feature, **summarize_ticket_segment(tickets, valid, "all_valid")})
        rows.append({"feature": feature, **summarize_ticket_segment(tickets, (values >= q80) & valid, "top20pct")})
        rows.append({"feature": feature, **summarize_ticket_segment(tickets, (values <= q20) & valid, "bottom20pct")})
    return pd.DataFrame(rows)


def write_summary(out_dir: Path, runner_eval: pd.DataFrame, ticket_eval: pd.DataFrame, runner: pd.DataFrame) -> dict[str, object]:
    def top_runner_rows(source: str, segment: str, metric: str, n: int = 8) -> list[dict[str, object]]:
        if runner_eval.empty:
            return []
        df = runner_eval[(runner_eval["source"] == source) & (runner_eval["segment"] == segment)].copy()
        if metric not in df.columns:
            return []
        cols = ["feature", "horses", "races", "win_rate", "top3_rate", "win_roi", "place_roi", "auc_top3_raw"]
        return df.sort_values(metric, ascending=False)[cols].head(n).to_dict(orient="records")

    def top_ticket_rows(segment: str, metric: str, n: int = 8) -> list[dict[str, object]]:
        if ticket_eval.empty:
            return []
        df = ticket_eval[ticket_eval["segment"] == segment].copy()
        cols = ["feature", "tickets", "races", "roi", "hit_rate", "stake_yen", "return_yen"]
        return df.sort_values(metric, ascending=False)[cols].head(n).to_dict(orient="records")

    summary = {
        "coverage": {
            "runner_rows": int(len(runner)),
            "test_rows": int((runner["source"] == "test").sum()),
            "train_rows": int((runner["source"] == "train").sum()),
            "races": int(runner["race_id"].nunique()),
            "horses": int(runner["horse_id"].nunique()),
            "date_min": str(runner["date_key"].min()),
            "date_max": str(runner["date_key"].max()),
        },
        "runner_test_top20_by_win_roi": top_runner_rows("test", "top20pct", "win_roi"),
        "runner_test_top20_by_place_roi": top_runner_rows("test", "top20pct", "place_roi"),
        "runner_test_top20_pop4_by_win_roi": top_runner_rows("test", "top20pct_pop4plus", "win_roi"),
        "selected_ticket_top20_by_roi": top_ticket_rows("top20pct", "roi"),
        "notes": [
            "This verification turns existing raw past-performance fields into rank-rate, trend, stability, stretch-gain, market-mismatch, and weight-burden candidates.",
            "It is a feature-candidate/overlay check, not a replacement for the current strongest model.",
            "Retrospective member bridge features may be less available for very recent previous races; treat them separately from safe pre-race fields.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs/analysis/basic_ability_transform_candidates_v1"),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = add_candidate_features(load_content_features())
    runner_eval = evaluate_runner_candidates(runner)
    tickets = load_selected_tickets_with_candidates(runner)
    ticket_eval = evaluate_ticket_candidates(tickets)

    runner_cols = [
        "source",
        "race_id",
        "date_key",
        "場所",
        "Ｒ",
        "horse_id",
        "horse_no",
        "popularity",
        "odds",
        "finish",
        "is_win",
        "is_top3",
        "win_return_100",
        "place_return_100",
    ] + [c for c in CANDIDATE_FEATURES if c in runner.columns]
    runner[runner_cols].to_csv(out_dir / "runner_basic_ability_candidate_features.csv", index=False, encoding="utf-8-sig")
    runner_eval.to_csv(out_dir / "runner_candidate_feature_eval.csv", index=False, encoding="utf-8-sig")
    tickets.to_csv(out_dir / "selected_tickets_with_basic_ability_candidates.csv", index=False, encoding="utf-8-sig")
    ticket_eval.to_csv(out_dir / "selected_ticket_candidate_feature_eval.csv", index=False, encoding="utf-8-sig")
    summary = write_summary(out_dir, runner_eval, ticket_eval, runner)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
