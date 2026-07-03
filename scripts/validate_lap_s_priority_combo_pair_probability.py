from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.strict_pair_probability_roi_protocol import (  # noqa: E402
    build_raw_probability_features,
    clip01,
    metrics,
    norm01,
    num,
    train_val_holdout,
    walkforward,
)


PAIR_UNIVERSE = Path("outputs/analysis/horse_lap_decomp_pair_probability_v1/pair_universe_with_lap_decomp_features.csv")
PAST3_PAIR = Path("outputs/analysis/past3_lap_profile_overlay_v1/past3_lap_pair_scores.csv")
WAVE_LOAD = Path("outputs/analysis/lap_waveform_load_tolerance_v1/lap_waveform_load_enriched_tickets.csv")
WAVE_ROLE = Path("outputs/analysis/lap_waveform_role_goodrun_v1/lap_waveform_role_goodrun_enriched_tickets.csv")
RUNNER_FEATURE_DIR = Path("data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus")
OUT = Path("outputs/analysis/lap_s_priority_combo_pair_probability_v1")


VARIANTS = {
    "baseline": {"wave": 0.000, "class": 0.000, "distance": 0.000, "caution": 0.000, "market": 0.000},
    "waveform_light": {"wave": 0.014, "class": 0.000, "distance": 0.000, "caution": 0.010, "market": 0.006},
    "waveform_mid": {"wave": 0.026, "class": 0.000, "distance": 0.000, "caution": 0.018, "market": 0.010},
    "class_lap_light": {"wave": 0.000, "class": 0.014, "distance": 0.000, "caution": 0.010, "market": 0.006},
    "class_lap_mid": {"wave": 0.000, "class": 0.026, "distance": 0.000, "caution": 0.018, "market": 0.010},
    "distance_lap_light": {"wave": 0.000, "class": 0.000, "distance": 0.014, "caution": 0.010, "market": 0.006},
    "distance_lap_mid": {"wave": 0.000, "class": 0.000, "distance": 0.026, "caution": 0.018, "market": 0.010},
    "s_combo_light": {"wave": 0.010, "class": 0.010, "distance": 0.010, "caution": 0.010, "market": 0.006},
    "s_combo_mid": {"wave": 0.018, "class": 0.018, "distance": 0.018, "caution": 0.018, "market": 0.010},
    "s_combo_strong": {"wave": 0.032, "class": 0.032, "distance": 0.032, "caution": 0.032, "market": 0.016},
    "s_guard_only": {"wave": 0.000, "class": 0.000, "distance": 0.000, "caution": 0.024, "market": 0.000},
}


def pair_key(df: pd.DataFrame, left: str, right: str) -> pd.DataFrame:
    out = df.copy()
    a = num(out[left])
    b = num(out[right])
    out["_pair_lo"] = np.minimum(a, b).astype("Int64")
    out["_pair_hi"] = np.maximum(a, b).astype("Int64")
    return out


def read_pair_universe() -> pd.DataFrame:
    if not PAIR_UNIVERSE.exists():
        raise FileNotFoundError(f"missing pair universe: {PAIR_UNIVERSE}")
    df = pd.read_csv(PAIR_UNIVERSE, dtype={"race_id": str}, low_memory=False)
    df["race_id"] = df["race_id"].astype(str)
    if "year" not in df.columns:
        df["year"] = df["race_id"].str[:4].astype(int)
    return pair_key(df, "anchor_no", "partner_no")


def load_past3_pair_features() -> pd.DataFrame:
    if not PAST3_PAIR.exists():
        return pd.DataFrame()
    keep = {
        "race_id",
        "anchor_no",
        "partner_no",
        "race_quality_fast_need_score",
        "race_quality_slow_need_score",
        "race_quality_sustain_need_score",
        "past3_lap_pair_fit_score",
        "past3_lap_pair_min_score",
        "past3_lap_evidence_count",
    }
    df = pd.read_csv(PAST3_PAIR, dtype={"race_id": str}, usecols=lambda c: c in keep, low_memory=False)
    df["race_id"] = df["race_id"].astype(str)
    df = pair_key(df, "anchor_no", "partner_no")
    agg_cols = {
        "race_quality_fast_need_score": "max",
        "race_quality_slow_need_score": "max",
        "race_quality_sustain_need_score": "max",
        "past3_lap_pair_fit_score": "max",
        "past3_lap_pair_min_score": "max",
        "past3_lap_evidence_count": "max",
    }
    return df.groupby(["race_id", "_pair_lo", "_pair_hi"], as_index=False).agg(agg_cols)


def load_waveform_features(path: Path, columns: dict[str, str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    keep = {"race_id", "anchor_no", "partner_no"} | set(columns)
    df = pd.read_csv(path, dtype={"race_id": str}, usecols=lambda c: c in keep, low_memory=False)
    df["race_id"] = df["race_id"].astype(str)
    df = pair_key(df, "anchor_no", "partner_no")
    rename = {old: new for old, new in columns.items() if old in df.columns}
    df = df.rename(columns=rename)
    agg = {new: "max" for new in rename.values()}
    if not agg:
        return pd.DataFrame()
    return df.groupby(["race_id", "_pair_lo", "_pair_hi"], as_index=False).agg(agg)


def load_runner_transition_features() -> pd.DataFrame:
    paths = [RUNNER_FEATURE_DIR / "train_features.csv", RUNNER_FEATURE_DIR / "test_features.csv"]
    cols = {
        "レースID(新/馬番無)",
        "馬番",
        "クラス名",
        "前クラス名",
        "距離",
        "前距離",
        "distance_diff",
        "rotation_distance_up_flag",
        "rotation_distance_down_flag",
        "rotation_big_distance_change_flag",
        "class_move_score",
        "rotation_class_up_flag",
        "rotation_class_down_flag",
        "rotation_same_class_flag",
        "前PCI",
        "前走PCI3",
        "前走RPCI",
        "past3_avg_time_value",
    }
    frames = []
    for path in paths:
        if path.exists():
            frames.append(pd.read_csv(path, dtype={"レースID(新/馬番無)": str}, usecols=lambda c: c in cols, low_memory=False))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    out = pd.DataFrame(
        {
            "race_id": df["レースID(新/馬番無)"].astype(str),
            "horse_no": num(df["馬番"]).astype("Int64"),
            "runner_class_name": df.get("クラス名", pd.Series("", index=df.index)).fillna("").astype(str),
            "runner_prev_class_name": df.get("前クラス名", pd.Series("", index=df.index)).fillna("").astype(str),
            "runner_distance": num(df.get("距離"), df.index, np.nan),
            "runner_prev_distance": num(df.get("前距離"), df.index, np.nan),
            "runner_distance_diff": num(df.get("distance_diff"), df.index, 0.0).fillna(0.0),
            "runner_distance_up": num(df.get("rotation_distance_up_flag"), df.index, 0.0).fillna(0.0).clip(0.0, 1.0),
            "runner_distance_down": num(df.get("rotation_distance_down_flag"), df.index, 0.0).fillna(0.0).clip(0.0, 1.0),
            "runner_big_distance_change": num(df.get("rotation_big_distance_change_flag"), df.index, 0.0).fillna(0.0).clip(0.0, 1.0),
            "runner_class_move_score": num(df.get("class_move_score"), df.index, 0.0).fillna(0.0),
            "runner_class_up": num(df.get("rotation_class_up_flag"), df.index, 0.0).fillna(0.0).clip(0.0, 1.0),
            "runner_class_down": num(df.get("rotation_class_down_flag"), df.index, 0.0).fillna(0.0).clip(0.0, 1.0),
            "runner_same_class": num(df.get("rotation_same_class_flag"), df.index, 0.0).fillna(0.0).clip(0.0, 1.0),
            "runner_prev_pci": num(df.get("前PCI"), df.index, np.nan),
            "runner_prev_pci3": num(df.get("前走PCI3"), df.index, np.nan),
            "runner_prev_rpci": num(df.get("前走RPCI"), df.index, np.nan),
            "runner_past3_time_value": num(df.get("past3_avg_time_value"), df.index, 0.0).fillna(0.0),
        }
    )
    return out.dropna(subset=["race_id", "horse_no"]).drop_duplicates(["race_id", "horse_no"], keep="last")


def attach_runner_side_features(df: pd.DataFrame) -> pd.DataFrame:
    runner = load_runner_transition_features()
    if runner.empty:
        return df
    anchor = runner.add_prefix("anchor_tr_").rename(
        columns={"anchor_tr_race_id": "race_id", "anchor_tr_horse_no": "anchor_no"}
    )
    partner = runner.add_prefix("partner_tr_").rename(
        columns={"partner_tr_race_id": "race_id", "partner_tr_horse_no": "partner_no"}
    )
    out = df.copy()
    out["anchor_no"] = num(out["anchor_no"]).astype("Int64")
    out["partner_no"] = num(out["partner_no"]).astype("Int64")
    out = out.merge(anchor, on=["race_id", "anchor_no"], how="left")
    out = out.merge(partner, on=["race_id", "partner_no"], how="left")
    return out


def attach_s_priority_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for extra in [
        load_past3_pair_features(),
        load_waveform_features(
            WAVE_LOAD,
            {
                "pair_waveform_combo_score": "wave_pair_combo_score",
                "pair_waveform_gap_similarity": "wave_pair_gap_similarity",
                "pair_load_collapse_risk_score": "wave_load_collapse_risk",
                "pair_load_role_balance_score": "wave_load_role_balance",
                "pair_wave_load_combo_score": "wave_load_combo_score",
            },
        ),
        load_waveform_features(
            WAVE_ROLE,
            {
                "strict_waveform_pair_min_score": "role_strict_wave_min_score",
                "strict_waveform_pair_gap_score": "role_strict_wave_gap_score",
                "goodrun_lap_pair_min_score": "role_goodrun_pair_min_score",
                "lap_role_front_front_collision_risk": "role_front_collision_risk",
                "lap_role_pair_probability_proxy": "role_pair_probability_proxy",
                "lap_advanced_combo_score": "role_advanced_combo_score",
            },
        ),
    ]:
        if not extra.empty:
            out = out.merge(extra, on=["race_id", "_pair_lo", "_pair_hi"], how="left")
    out = attach_runner_side_features(out)
    return build_scores(out)


def _n(out: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    return num(out.get(col), out.index, default).fillna(default)


def build_scores(out: pd.DataFrame) -> pd.DataFrame:
    idx = out.index
    for col in [
        "past3_lap_pair_fit_score",
        "past3_lap_pair_min_score",
        "past3_lap_evidence_count",
        "wave_pair_combo_score",
        "wave_pair_gap_similarity",
        "wave_load_collapse_risk",
        "wave_load_role_balance",
        "wave_load_combo_score",
        "role_strict_wave_min_score",
        "role_strict_wave_gap_score",
        "role_goodrun_pair_min_score",
        "role_front_collision_risk",
        "role_pair_probability_proxy",
        "role_advanced_combo_score",
    ]:
        out[col] = _n(out, col, 0.0)

    lap_context = clip01(out.get("lap_decomp_context", pd.Series(0.0, index=idx)))
    lap_fit = clip01(out.get("lap_decomp_fit", pd.Series(0.0, index=idx)))
    lap_caution = clip01(out.get("lap_decomp_caution", pd.Series(0.0, index=idx)))
    past_fit = clip01(out["past3_lap_pair_fit_score"])
    past_min = clip01(out["past3_lap_pair_min_score"])
    evidence = norm01(out["past3_lap_evidence_count"], lo=0.0, hi=6.0)
    wave_combo = clip01(out["wave_pair_combo_score"]).where(out["wave_pair_combo_score"].gt(0), np.nan)
    wave_load_combo = clip01(out["wave_load_combo_score"]).where(out["wave_load_combo_score"].gt(0), np.nan)
    role_combo = clip01(out["role_advanced_combo_score"]).where(out["role_advanced_combo_score"].gt(0), np.nan)
    strict_wave = clip01(out["role_strict_wave_min_score"]).where(out["role_strict_wave_min_score"].gt(0), np.nan)
    goodrun = clip01(out["role_goodrun_pair_min_score"]).where(out["role_goodrun_pair_min_score"].gt(0), np.nan)
    wave_known = pd.concat([wave_combo, wave_load_combo, role_combo, strict_wave, goodrun], axis=1).notna().any(axis=1).astype(float)
    waveform_mean = pd.concat([wave_combo, wave_load_combo, role_combo, strict_wave, goodrun], axis=1).mean(axis=1).fillna(0.0)
    wave_gap_low = (1.0 - clip01(out["wave_pair_gap_similarity"])).where(out["wave_pair_gap_similarity"].gt(0), 0.0)
    role_gap_low = (1.0 - clip01(out["role_strict_wave_gap_score"])).where(out["role_strict_wave_gap_score"].gt(0), 0.0)
    load_risk = clip01(out["wave_load_collapse_risk"])
    collision_risk = clip01(out["role_front_collision_risk"])

    out["s_waveform_fit_score"] = (
        0.28 * past_fit
        + 0.18 * past_min
        + 0.12 * evidence
        + 0.18 * lap_context
        + 0.12 * lap_fit
        + 0.12 * waveform_mean * wave_known
        + 0.05 * wave_gap_low
        + 0.05 * role_gap_low
    ).clip(0.0, 1.0)
    out["s_waveform_caution_score"] = (
        0.30 * lap_caution
        + 0.22 * load_risk
        + 0.22 * collision_risk
        + 0.16 * (1.0 - past_min) * evidence
        + 0.10 * (1.0 - waveform_mean) * wave_known
    ).clip(0.0, 1.0)

    # Runner-level class/distance features.
    def side(col: str) -> tuple[pd.Series, pd.Series]:
        return _n(out, f"anchor_tr_{col}", 0.0), _n(out, f"partner_tr_{col}", 0.0)

    a_class_up, b_class_up = side("runner_class_up")
    a_class_down, b_class_down = side("runner_class_down")
    a_same_class, b_same_class = side("runner_same_class")
    a_class_move, b_class_move = side("runner_class_move_score")
    a_time, b_time = side("runner_past3_time_value")
    a_distance_up, b_distance_up = side("runner_distance_up")
    a_distance_down, b_distance_down = side("runner_distance_down")
    a_big_dist, b_big_dist = side("runner_big_distance_change")
    a_dist_diff, b_dist_diff = side("runner_distance_diff")
    a_prev_pci, b_prev_pci = side("runner_prev_pci")
    a_prev_rpci, b_prev_rpci = side("runner_prev_rpci")

    pair_class_up_any = np.maximum(a_class_up, b_class_up).clip(0.0, 1.0)
    pair_class_down_any = np.maximum(a_class_down, b_class_down).clip(0.0, 1.0)
    pair_same_class_both = (a_same_class * b_same_class).clip(0.0, 1.0)
    class_move_abs = norm01(pd.concat([a_class_move.abs(), b_class_move.abs()], axis=1).max(axis=1), lo=0.0, hi=3.0)
    time_value = norm01(pd.concat([a_time, b_time], axis=1).mean(axis=1), lo=-0.08, hi=0.08)

    out["s_class_lap_fit_score"] = (
        0.26 * lap_context
        + 0.18 * past_fit
        + 0.16 * time_value
        + 0.14 * pair_same_class_both
        + 0.12 * pair_class_down_any
        + 0.08 * pair_class_up_any * np.maximum(lap_fit, past_min)
        + 0.06 * evidence
    ).clip(0.0, 1.0)
    out["s_class_lap_caution_score"] = (
        0.28 * pair_class_up_any * (1.0 - np.maximum(lap_fit, past_min))
        + 0.18 * class_move_abs * (1.0 - time_value)
        + 0.18 * lap_caution
        + 0.14 * (1.0 - past_min) * evidence
        + 0.12 * pair_class_up_any * (1.0 - evidence)
        + 0.10 * pair_class_down_any * lap_caution
    ).clip(0.0, 1.0)

    dist_change_any = np.maximum.reduce([a_distance_up, b_distance_up, a_distance_down, b_distance_down, a_big_dist, b_big_dist]).clip(0.0, 1.0)
    shortening_any = np.maximum(a_distance_down, b_distance_down).clip(0.0, 1.0)
    extension_any = np.maximum(a_distance_up, b_distance_up).clip(0.0, 1.0)
    big_dist_any = np.maximum(a_big_dist, b_big_dist).clip(0.0, 1.0)
    abs_dist = norm01(pd.concat([a_dist_diff.abs(), b_dist_diff.abs()], axis=1).max(axis=1), lo=0.0, hi=600.0)
    prev_fast_or_tough = (
        pd.concat([a_prev_rpci, b_prev_rpci], axis=1).min(axis=1).lt(48.0)
        | pd.concat([a_prev_pci, b_prev_pci], axis=1).min(axis=1).lt(48.0)
    ).astype(float)
    prev_slow = (
        pd.concat([a_prev_rpci, b_prev_rpci], axis=1).max(axis=1).gt(52.0)
        | pd.concat([a_prev_pci, b_prev_pci], axis=1).max(axis=1).gt(52.0)
    ).astype(float)
    fast_type = (
        out.get("anchor_lap_computed_horse_lap_type", pd.Series("", index=idx)).fillna("").astype(str).eq("fast")
        | out.get("partner_lap_computed_horse_lap_type", pd.Series("", index=idx)).fillna("").astype(str).eq("fast")
    ).astype(float)
    sustain_or_long = (
        out.get("anchor_lap_computed_horse_lap_type", pd.Series("", index=idx)).fillna("").astype(str).isin(["sustain", "long_spurt"])
        | out.get("partner_lap_computed_horse_lap_type", pd.Series("", index=idx)).fillna("").astype(str).isin(["sustain", "long_spurt"])
    ).astype(float)

    out["s_distance_lap_fit_score"] = (
        0.24 * lap_context
        + 0.18 * past_fit
        + 0.14 * time_value
        + 0.13 * shortening_any * np.maximum(prev_fast_or_tough, fast_type)
        + 0.12 * extension_any * np.maximum(prev_slow, sustain_or_long)
        + 0.08 * dist_change_any * evidence
        + 0.06 * (1.0 - big_dist_any)
        + 0.05 * (1.0 - abs_dist)
    ).clip(0.0, 1.0)
    out["s_distance_lap_caution_score"] = (
        0.26 * big_dist_any * (1.0 - np.maximum(lap_fit, past_min))
        + 0.18 * abs_dist * (1.0 - time_value)
        + 0.16 * dist_change_any * lap_caution
        + 0.14 * shortening_any * prev_slow * (1.0 - fast_type)
        + 0.12 * extension_any * prev_fast_or_tough * (1.0 - sustain_or_long)
        + 0.08 * dist_change_any * (1.0 - evidence)
        + 0.06 * (1.0 - past_min) * evidence
    ).clip(0.0, 1.0)

    out["s_priority_lap_support_score"] = (
        0.40 * out["s_waveform_fit_score"]
        + 0.30 * out["s_class_lap_fit_score"]
        + 0.30 * out["s_distance_lap_fit_score"]
    ).clip(0.0, 1.0)
    out["s_priority_lap_caution_score"] = (
        0.40 * out["s_waveform_caution_score"]
        + 0.30 * out["s_class_lap_caution_score"]
        + 0.30 * out["s_distance_lap_caution_score"]
    ).clip(0.0, 1.0)
    out["s_priority_lap_net_score"] = (
        out["s_priority_lap_support_score"] - out["s_priority_lap_caution_score"]
    ).clip(-1.0, 1.0)
    out["s_priority_lap_label"] = np.select(
        [
            out["s_priority_lap_net_score"].ge(0.22),
            out["s_priority_lap_net_score"].le(-0.10),
            out["s_waveform_fit_score"].ge(0.62),
            out["s_class_lap_fit_score"].ge(0.62),
            out["s_distance_lap_fit_score"].ge(0.62),
        ],
        [
            "s_lap_total_support",
            "s_lap_caution",
            "s_waveform_support",
            "s_class_lap_support",
            "s_distance_lap_support",
        ],
        default="s_lap_neutral",
    )
    return out


def apply_variant(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()
    params = VARIANTS[name]
    if name == "baseline":
        out["s_priority_variant"] = name
        out["s_priority_probability_adjustment"] = 0.0
        return out
    support = (
        params["wave"] * clip01(out["s_waveform_fit_score"])
        + params["class"] * clip01(out["s_class_lap_fit_score"])
        + params["distance"] * clip01(out["s_distance_lap_fit_score"])
    )
    caution = params["caution"] * clip01(out["s_priority_lap_caution_score"])
    net = support - caution
    out["pair_quinella_score"] = (num(out["pair_quinella_score"]).fillna(0.0) + net).clip(0.0, 1.0)
    out["pair_score"] = (num(out["pair_score"]).fillna(0.0) + 0.70 * net).clip(0.0, 1.0)
    out["market_overlay_score"] = (
        num(out["market_overlay_score"]).fillna(0.0)
        + params["market"] * (0.72 * clip01(out["s_priority_lap_support_score"]) - 0.52 * clip01(out["s_priority_lap_caution_score"]))
    ).clip(0.0, 1.0)
    out["s_priority_probability_adjustment"] = net
    out["s_priority_variant"] = name
    return out


def aggregate_tickets(tickets: pd.DataFrame, label: str) -> dict:
    m = metrics(tickets, label)
    if not tickets.empty:
        year_rows = []
        for year, part in tickets.groupby("year", sort=True):
            y = metrics(part, f"{label}_{year}")
            year_rows.append({"year": int(year), "roi": y["roi"], "races": y["races"], "profit_yen": y["profit_yen"]})
        m["min_year_roi"] = min((x["roi"] for x in year_rows), default=0.0)
        m["year_metrics"] = year_rows
    else:
        m["min_year_roi"] = 0.0
        m["year_metrics"] = []
    return m


def run_variant(source: pd.DataFrame, name: str) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scored = build_raw_probability_features(apply_variant(source, name))
    train_grid, wf_summary, wf_tickets = walkforward(scored)
    holdout = train_val_holdout(scored)
    total = aggregate_tickets(wf_tickets, f"{name}_walkforward_total")
    total["variant"] = name
    return total, train_grid, wf_summary, holdout, wf_tickets


def segment_metrics(tickets: pd.DataFrame, label: str) -> pd.DataFrame:
    if tickets.empty:
        return pd.DataFrame()
    rows = []
    for col in ["s_priority_lap_label", "ticket_type", "venue", "going"]:
        if col not in tickets.columns:
            continue
        for val, part in tickets.groupby(col, dropna=False):
            stake = float(pd.to_numeric(part.get("stake_yen"), errors="coerce").fillna(0.0).sum())
            ret = float(pd.to_numeric(part.get("return_yen"), errors="coerce").fillna(0.0).sum())
            rows.append(
                {
                    "variant": label,
                    "segment_col": col,
                    "segment_value": str(val),
                    "tickets": int(len(part)),
                    "races": int(part["race_id"].nunique()),
                    "stake_yen": stake,
                    "return_yen": ret,
                    "profit_yen": ret - stake,
                    "roi": ret / stake if stake else 0.0,
                }
            )
    for score_col in [
        "s_waveform_fit_score",
        "s_class_lap_fit_score",
        "s_distance_lap_fit_score",
        "s_priority_lap_support_score",
        "s_priority_lap_caution_score",
        "s_priority_lap_net_score",
    ]:
        if score_col not in tickets.columns:
            continue
        work = tickets.copy()
        score = pd.to_numeric(work[score_col], errors="coerce").fillna(0.0)
        if score.nunique() < 4:
            continue
        try:
            work["_bin"] = pd.qcut(score.rank(method="first"), q=4, labels=["q1_low", "q2", "q3", "q4_high"])
        except ValueError:
            continue
        for val, part in work.groupby("_bin", observed=True):
            stake = float(pd.to_numeric(part.get("stake_yen"), errors="coerce").fillna(0.0).sum())
            ret = float(pd.to_numeric(part.get("return_yen"), errors="coerce").fillna(0.0).sum())
            rows.append(
                {
                    "variant": label,
                    "segment_col": f"{score_col}_quartile",
                    "segment_value": str(val),
                    "tickets": int(len(part)),
                    "races": int(part["race_id"].nunique()),
                    "stake_yen": stake,
                    "return_yen": ret,
                    "profit_yen": ret - stake,
                    "roi": ret / stake if stake else 0.0,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = attach_s_priority_features(read_pair_universe())
    source.to_csv(OUT / "pair_universe_with_s_priority_lap_features.csv", index=False, encoding="utf-8-sig")

    summaries: list[dict] = []
    all_years: list[dict] = []
    all_segments: list[pd.DataFrame] = []
    holdouts: list[pd.DataFrame] = []
    best_variant = ""
    best_roi = -1.0

    for variant in VARIANTS:
        total, train_grid, wf_summary, holdout, tickets = run_variant(source, variant)
        summaries.append(total)
        for row in total.get("year_metrics", []):
            row["variant"] = variant
            all_years.append(row)
        train_grid.to_csv(OUT / f"{variant}_walkforward_train_top100.csv", index=False, encoding="utf-8-sig")
        wf_summary.to_csv(OUT / f"{variant}_walkforward_summary.csv", index=False, encoding="utf-8-sig")
        holdout["variant"] = variant
        holdout.to_csv(OUT / f"{variant}_train2024_val2025_hold2026_grid.csv", index=False, encoding="utf-8-sig")
        tickets.to_csv(OUT / f"{variant}_walkforward_selected_tickets.csv", index=False, encoding="utf-8-sig")
        seg = segment_metrics(tickets, variant)
        if not seg.empty:
            all_segments.append(seg)
        if total["roi"] > best_roi:
            best_roi = float(total["roi"])
            best_variant = variant
        holdouts.append(holdout)

    comparison = pd.DataFrame(summaries).drop(columns=["year_metrics"], errors="ignore")
    comparison = comparison.sort_values(["roi", "profit_yen"], ascending=False)
    comparison.to_csv(OUT / "variant_walkforward_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_years).to_csv(OUT / "variant_yearly_walkforward_summary.csv", index=False, encoding="utf-8-sig")
    if all_segments:
        pd.concat(all_segments, ignore_index=True, sort=False).to_csv(
            OUT / "variant_selected_ticket_segment_metrics.csv", index=False, encoding="utf-8-sig"
        )
    if holdouts:
        hold = pd.concat(holdouts, ignore_index=True, sort=False)
        hold.to_csv(OUT / "variant_holdout_grid_all.csv", index=False, encoding="utf-8-sig")
        hold.sort_values(["dev_score", "val_profit_yen"], ascending=False).groupby("variant", as_index=False).head(5).to_csv(
            OUT / "variant_holdout_top5_by_variant.csv", index=False, encoding="utf-8-sig"
        )

    summary = {
        "output_dir": str(OUT),
        "pair_universe_rows": int(len(source)),
        "race_count": int(source["race_id"].nunique()),
        "variants": list(VARIANTS),
        "best_variant": best_variant,
        "comparison": comparison.to_dict(orient="records"),
        "note": (
            "Combined validation for S-priority lap factors: section/waveform profile, lap x class move, "
            "and lap x distance change. All are tested as thin modifiers on the strict pair-probability protocol."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# Lap S Priority Combo Pair Probability v1",
                "",
                "Purpose: validate three S-priority lap factors together and separately:",
                "1. section/waveform lap aptitude",
                "2. lap aptitude x class up/down",
                "3. lap aptitude x distance shortening/extension",
                "",
                "Key outputs:",
                "- pair_universe_with_s_priority_lap_features.csv",
                "- variant_walkforward_comparison.csv",
                "- variant_yearly_walkforward_summary.csv",
                "- variant_selected_ticket_segment_metrics.csv",
                "- summary.json",
            ]
        ),
        encoding="utf-8",
    )
    display_cols = [
        "variant",
        "races",
        "tickets",
        "stake_yen",
        "return_yen",
        "profit_yen",
        "roi",
        "race_hit_rate",
        "max_drawdown_yen",
        "min_year_roi",
    ]
    print(comparison[display_cols].to_string(index=False))
    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()
