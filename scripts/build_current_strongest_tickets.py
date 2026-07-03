from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.gelding_transition import (  # noqa: E402
    GELDING_FEATURE_COLUMNS,
    enrich_current_entries_with_gelding_context,
    read_csv_any,
)
from src.features.odds_timeline import clean_odds_series  # noqa: E402


VENUE_CODE_BY_NAME = {
    "札幌": "01",
    "函館": "02",
    "福島": "03",
    "新潟": "04",
    "東京": "05",
    "中山": "06",
    "中京": "07",
    "京都": "08",
    "阪神": "09",
    "小倉": "10",
}

MAX_FINAL_BUY_UMAREN_ODDS = 120.0
POSITION_FRONT_VALUE_MIN = 0.51
FAST_CLOCK_BUY_MIN = 0.24
FAST_CLOCK_STRONG_SUPPORT = 0.50
CORNER_SHAPE_STRONG_SUPPORT = 0.70
PAST3_LAP_BUY_MIN = 0.281
PAST3_LAP_BUY_MAX = 0.424
TIME_REFINEMENT_SHADOW_PAIR_MIN = 0.62
TIME_REFINEMENT_SHADOW_RELATIVE_MIN = 0.42
TIME_RELATIVE_LOW_CAUTION_MAX = 0.42
LAP_PAIR_FIT_SHADOW_MIN = 0.277
LAP_PAIR_CONFIDENCE_SHADOW_MIN = 0.342
LAP_PAIR_CONTRADICTION_CAUTION_MAX = 0.581
LAP_POPULAR_MISMATCH_CAUTION_MIN = 0.343
LAP_LIGHT_SAFETY_CAUTION_MAX = 0.60
LAP_LIGHT_SAFETY_EDGE_MARGIN = 3.20
LAP_LIGHT_SAFETY_EDGE_ROI = 1.75
LAP_ADV_GOODRUN_STRONG_MIN = 0.525
LAP_ADV_ROLE_STRONG_MIN = 0.520
LAP_ADV_COMBO_WATCH_MIN = 0.580
LAP_ADV_COMBO_STRONG_MIN = 0.630
LAP_ADV_COLLISION_SAFE_MAX = 0.350
WORKOUT_AUTO_CANDIDATES_DEFAULT = (
    "outputs/analysis/workout_knowledge_auto_mine_v1/workout_auto_knowledge_candidates_priority.csv"
)
TRACK_CONDITION_METRICS_DEFAULT = ROOT / "data/raw/track_condition_metrics.csv"
TARGET_RA_OFFICIAL_LAP_HISTORY_DEFAULT = (
    ROOT
    / "outputs"
    / "analysis"
    / "current_strongest_runtime_v1"
    / "current_target_ra_lap_history_features.csv"
)
TARGET_RA_LAP_AXES = [
    "front_load",
    "slow_finish",
    "l1_instant",
    "l2_sustain",
    "l3_long_spurt",
]
TARGET_RA_LAP_DEFAULTS = {
    "target_ra_lap_pair_fit_score": 0.0,
    "target_ra_lap_pair_fit_min_score": 0.0,
    "target_ra_lap_pair_meanfit_score": 0.0,
    "target_ra_lap_pair_need_match_score": 0.0,
    "target_ra_lap_pair_strength_score": 0.0,
    "target_ra_lap_pair_ready_score": 0.0,
    "target_ra_lap_pair_ready_both": 0.0,
    "target_ra_lap_mismatch_risk_score": 1.0,
    "target_ra_lap_shadow_label": "target_ra_lap_no_history",
    "target_ra_lap_shadow_note": "",
}
VENUE_NAME_BY_CODE = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}
LAP_TRACK_STRICT_GAP_LOW_MAX = 0.008
LAP_TRACK_GOODRUN_MIN_HIGH = 0.821
LAP_TRACK_ROLE_LOW_COLLISION_MIN = 0.666
LAP_TRACK_COLLISION_SAFE_MAX = 0.046
RACE_QUALITY_V2_MODEL_PATH = ROOT / "outputs/analysis/race_quality_prediction_v2/model_params.json"
RACE_QUALITY_V2_CONF_LOW = 0.279
RACE_QUALITY_V2_CONF_MID = 0.293
RACE_QUALITY_V2_CONF_HIGH = 0.350

# Historical high-pressure race outcomes by venue and surface.
# These are deliberately coarse and used only as a safety gate: they should
# stop front-leaning tickets in regimes where pressure often converts to
# closer outcomes, not create new BUY candidates.
PACE_REGIME_BY_VENUE_SURFACE = {
    ("01", "turf"): (0.889, 0.037),
    ("01", "dirt"): (0.844, 0.031),
    ("02", "turf"): (0.737, 0.000),
    ("02", "dirt"): (1.000, 0.000),
    ("03", "turf"): (0.700, 0.133),
    ("03", "dirt"): (0.886, 0.023),
    ("04", "turf"): (0.629, 0.171),
    ("04", "dirt"): (0.806, 0.028),
    ("05", "turf"): (0.500, 0.281),
    ("05", "dirt"): (0.606, 0.167),
    ("06", "turf"): (0.776, 0.103),
    ("06", "dirt"): (0.915, 0.000),
    ("07", "turf"): (0.656, 0.094),
    ("07", "dirt"): (0.564, 0.255),
    ("08", "turf"): (0.643, 0.119),
    ("08", "dirt"): (0.817, 0.024),
    ("09", "turf"): (0.692, 0.128),
    ("09", "dirt"): (0.662, 0.052),
    ("10", "turf"): (0.758, 0.113),
    ("10", "dirt"): (0.868, 0.038),
}
PACE_REGIME_SURFACE_DEFAULT = {
    "turf": (0.700, 0.120),
    "dirt": (0.780, 0.070),
    "unknown": (0.746, 0.089),
}
FRONT_SURVIVAL_CONTEXT_LOOKUP_PATH = ROOT / "outputs/analysis/front_survival_context_v1/front_survival_context_lookup.csv"
FRONT_SURVIVAL_CONTEXT_DEFAULT = {
    "context_races": 0.0,
    "front_survival_rate": 0.856,
    "front_collapse_rate": 0.035,
    "top3_front5_share": 0.56,
    "front_context_readability_score": 0.0,
}


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def latest_file(pattern: str) -> Path | None:
    files = list(ROOT.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def prediction_race_ids(frame: pd.DataFrame) -> set[str]:
    if frame.empty:
        return set()
    ids: set[str] = set()
    for _, row in frame.iterrows():
        race_id = row_to_official_race_id(row)
        if race_id:
            ids.add(race_id)
    return ids


def assert_prediction_odds_overlap(prediction: pd.DataFrame, single: pd.DataFrame, prediction_path: Path, single_path: Path) -> None:
    if prediction.empty or single.empty or "race_id" not in single.columns:
        return
    pred_ids = prediction_race_ids(prediction)
    odds_ids = set(single["race_id"].dropna().astype(str))
    if pred_ids and odds_ids and pred_ids.isdisjoint(odds_ids):
        pred_sample = ", ".join(sorted(list(pred_ids))[:3])
        odds_sample = ", ".join(sorted(list(odds_ids))[:3])
        raise ValueError(
            "Prediction CSV and live odds CSV have no overlapping race_id values. "
            "Refusing to overwrite current strongest tickets with an empty mismatch result. "
            f"prediction_csv={prediction_path} prediction_sample=[{pred_sample}] "
            f"single_odds_csv={single_path} odds_sample=[{odds_sample}]"
        )


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, **kwargs)


def read_gelding_history(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    history = read_csv_any(path)
    if "race_date" in history.columns:
        history["race_date"] = pd.to_datetime(history["race_date"], errors="coerce")
    return history


def text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def num(value: object, default: float = np.nan) -> float:
    try:
        if value is None or pd.isna(value) or text(value) == "":
            return default
        return float(str(value).replace(",", ""))
    except Exception:
        return default


def series_num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def first_series(frame: pd.DataFrame, names: list[str], default: float = np.nan) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def first_text_series(frame: pd.DataFrame, names: list[str], default: str = "") -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name].astype("string").fillna("").astype(str)
    return pd.Series(default, index=frame.index, dtype=str)


def compact_text_series(series: pd.Series | None, index: pd.Index, default: str = "unknown") -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype="string")
    out = series.astype("string").fillna(default)
    return out.mask(out.str.len().fillna(0).eq(0), default).astype("string")


def clip01(value: pd.Series | np.ndarray | float) -> pd.Series:
    if isinstance(value, pd.Series):
        s = value
    else:
        s = pd.Series(value)
    return pd.to_numeric(s, errors="coerce").fillna(0.0).clip(0.0, 1.0)


def norm01(series: pd.Series, lo: float | None = None, hi: float | None = None) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if lo is None:
        lo = float(x.quantile(0.05)) if x.notna().any() else 0.0
    if hi is None:
        hi = float(x.quantile(0.95)) if x.notna().any() else 1.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.5, index=x.index)
    return ((x.clip(lo, hi) - lo) / (hi - lo)).fillna(0.5).clip(0.0, 1.0)


def normalize_profile(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    values = values.clip(lower=0.0)
    total = values.sum(axis=1).replace(0.0, np.nan)
    return values.div(total, axis=0).fillna(1.0 / max(1, len(values.columns))).clip(0.0, 1.0)


def target_ra_clean_race_id(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.replace(r"\.0$", "", regex=True).str.strip().str.zfill(16)


def max_numeric_series(frame: pd.DataFrame, names: list[str], default: float = 0.0) -> pd.Series:
    values: list[pd.Series] = []
    for name in names:
        if name in frame.columns:
            values.append(pd.to_numeric(frame[name], errors="coerce"))
    if not values:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.concat(values, axis=1).max(axis=1).fillna(default)


def load_target_ra_lap_history(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    wanted = [
        "race_id",
        "horse_no",
        "official_lap_history_count_past3",
        "official_lap_history_ready",
        "official_lap_profile_strength",
        "official_lap_profile_versatility",
        *[f"official_{axis}_goodrun_score_past3_mean" for axis in TARGET_RA_LAP_AXES],
        *[f"official_{axis}_goodrun_score_past3_max" for axis in TARGET_RA_LAP_AXES],
        *[f"official_{axis}_need_past3_mean" for axis in TARGET_RA_LAP_AXES],
    ]
    try:
        header = read_csv(path, nrows=0)
        usecols = [col for col in wanted if col in header.columns]
        history = read_csv(path, usecols=usecols)
    except Exception:
        return pd.DataFrame()
    if history.empty or "race_id" not in history.columns or "horse_no" not in history.columns:
        return pd.DataFrame()
    history["race_id"] = target_ra_clean_race_id(history["race_id"])
    history["horse_no"] = pd.to_numeric(history["horse_no"], errors="coerce").astype("Int64")
    for col in history.columns:
        if col not in {"race_id", "horse_no"}:
            history[col] = pd.to_numeric(history[col], errors="coerce")
    return history.dropna(subset=["race_id", "horse_no"]).drop_duplicates(["race_id", "horse_no"], keep="last")


def target_ra_lap_need_vector(frame: pd.DataFrame) -> pd.DataFrame:
    idx = frame.index
    fast = max_numeric_series(
        frame,
        ["race_quality_v2_prob_fast", "race_quality_fast_need_score", "shape_fast_signal", "v2_prob_fast"],
        0.0,
    ).clip(0.0, 1.0)
    slow = max_numeric_series(
        frame,
        ["race_quality_v2_prob_slow", "race_quality_slow_need_score", "shape_slow_signal", "v2_prob_slow"],
        0.0,
    ).clip(0.0, 1.0)
    instant = max_numeric_series(
        frame,
        ["race_quality_v2_prob_instant", "shape_instant_signal", "v2_prob_instant"],
        0.0,
    ).clip(0.0, 1.0)
    sustain = max_numeric_series(
        frame,
        ["race_quality_v2_prob_sustain", "shape_sustain_signal", "v2_prob_sustain"],
        0.0,
    ).clip(0.0, 1.0)
    queue_front = max_numeric_series(
        frame,
        ["race_projected_front_load_score", "queue_front_load_score"],
        0.0,
    ).clip(0.0, 1.0)
    front_load = (0.70 * fast + 0.30 * queue_front).clip(0.0, 1.0)
    long_spurt = (0.55 * sustain + 0.35 * fast + 0.10 * queue_front).clip(0.0, 1.0)
    need = pd.DataFrame(
        {
            "front_load": front_load,
            "slow_finish": slow,
            "l1_instant": instant,
            "l2_sustain": sustain,
            "l3_long_spurt": long_spurt,
        },
        index=idx,
    )
    row_sum = need.sum(axis=1).replace(0.0, np.nan)
    return need.div(row_sum, axis=0).fillna(1.0 / len(TARGET_RA_LAP_AXES)).clip(0.0, 1.0)


def merge_target_ra_lap_side(frame: pd.DataFrame, history: pd.DataFrame, side: str, no_col: str) -> pd.DataFrame:
    out = frame.copy()
    if no_col not in out.columns:
        return out
    out[no_col] = pd.to_numeric(out[no_col], errors="coerce").astype("Int64")
    rename = {col: f"{side}_{col}" for col in history.columns if col not in {"race_id", "horse_no"}}
    side_history = history.rename(columns={"horse_no": no_col, **rename})
    return out.merge(side_history, on=["race_id", no_col], how="left")


def apply_target_ra_official_lap_overlay(candidates: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    for col, default in TARGET_RA_LAP_DEFAULTS.items():
        if col not in out.columns:
            out[col] = default
    if out.empty or history.empty or "race_id" not in out.columns:
        return out

    out["race_id"] = target_ra_clean_race_id(out["race_id"])
    out = merge_target_ra_lap_side(out, history, "anchor", "anchor_horse_no")
    out = merge_target_ra_lap_side(out, history, "partner", "partner_horse_no")
    need = target_ra_lap_need_vector(out)

    for side in ("anchor", "partner"):
        max_fit = pd.Series(0.0, index=out.index, dtype=float)
        mean_fit = pd.Series(0.0, index=out.index, dtype=float)
        need_match = pd.Series(0.0, index=out.index, dtype=float)
        for axis in TARGET_RA_LAP_AXES:
            weight = need[axis]
            max_score = clip01(first_series(out, [f"{side}_official_{axis}_goodrun_score_past3_max"], 0.0))
            mean_score = clip01(first_series(out, [f"{side}_official_{axis}_goodrun_score_past3_mean"], 0.0))
            need_score = clip01(first_series(out, [f"{side}_official_{axis}_need_past3_mean"], 0.0))
            max_fit += weight * max_score
            mean_fit += weight * mean_score
            need_match += weight * need_score
        out[f"{side}_target_ra_lap_fit_max"] = max_fit.clip(0.0, 1.0)
        out[f"{side}_target_ra_lap_fit_mean"] = mean_fit.clip(0.0, 1.0)
        out[f"{side}_target_ra_lap_need_match"] = need_match.clip(0.0, 1.0)
        out[f"{side}_target_ra_lap_ready"] = (
            first_series(out, [f"{side}_official_lap_history_ready"], 0.0).fillna(0.0).gt(0.0)
        )
        out[f"{side}_target_ra_lap_strength"] = clip01(
            first_series(out, [f"{side}_official_lap_profile_strength"], 0.0)
        )

    ready_any = out["anchor_target_ra_lap_ready"] | out["partner_target_ra_lap_ready"]
    ready_both = out["anchor_target_ra_lap_ready"] & out["partner_target_ra_lap_ready"]
    fit_avg = ((out["anchor_target_ra_lap_fit_max"] + out["partner_target_ra_lap_fit_max"]) / 2.0).clip(0.0, 1.0)
    fit_min = np.minimum(out["anchor_target_ra_lap_fit_max"], out["partner_target_ra_lap_fit_max"]).clip(0.0, 1.0)
    meanfit_avg = ((out["anchor_target_ra_lap_fit_mean"] + out["partner_target_ra_lap_fit_mean"]) / 2.0).clip(0.0, 1.0)
    need_match_avg = (
        (out["anchor_target_ra_lap_need_match"] + out["partner_target_ra_lap_need_match"]) / 2.0
    ).clip(0.0, 1.0)
    strength_avg = (
        (out["anchor_target_ra_lap_strength"] + out["partner_target_ra_lap_strength"]) / 2.0
    ).clip(0.0, 1.0)
    mismatch = (0.55 * (1.0 - fit_avg) + 0.25 * (1.0 - need_match_avg) + 0.20 * (1.0 - strength_avg)).clip(0.0, 1.0)

    out["target_ra_lap_pair_fit_score"] = fit_avg
    out["target_ra_lap_pair_fit_min_score"] = fit_min
    out["target_ra_lap_pair_meanfit_score"] = meanfit_avg
    out["target_ra_lap_pair_need_match_score"] = need_match_avg
    out["target_ra_lap_pair_strength_score"] = strength_avg
    out["target_ra_lap_pair_ready_score"] = (ready_any.astype(float) + ready_both.astype(float)) / 2.0
    out["target_ra_lap_pair_ready_both"] = ready_both.astype(float)
    out["target_ra_lap_mismatch_risk_score"] = mismatch
    out["target_ra_lap_shadow_label"] = np.select(
        [
            ~ready_any,
            fit_avg.ge(0.55) & mismatch.le(0.55) & ready_both,
            fit_avg.ge(0.55) & mismatch.le(0.55),
            fit_avg.ge(0.45) & mismatch.le(0.65),
            ready_any & fit_avg.lt(0.30),
        ],
        [
            "target_ra_lap_no_history",
            "target_ra_lap_fit_strong_both",
            "target_ra_lap_fit_strong",
            "target_ra_lap_fit_watch",
            "target_ra_lap_caution",
        ],
        default="target_ra_lap_neutral",
    )
    out["target_ra_lap_shadow_note"] = np.select(
        [
            out["target_ra_lap_shadow_label"].eq("target_ra_lap_fit_strong_both"),
            out["target_ra_lap_shadow_label"].eq("target_ra_lap_fit_strong"),
            out["target_ra_lap_shadow_label"].eq("target_ra_lap_fit_watch"),
            out["target_ra_lap_shadow_label"].eq("target_ra_lap_caution"),
        ],
        [
            "公式RAラップ履歴: 2頭とも今回想定の流れに合う",
            "公式RAラップ履歴: 片方以上に今回想定への裏付けあり",
            "公式RAラップ履歴: ラップ適性は参考プラス",
            "公式RAラップ履歴: 今回想定への裏付けは薄め",
        ],
        default="",
    )
    return out


def load_race_quality_v2_model(path: Path = RACE_QUALITY_V2_MODEL_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _race_quality_v2_softmax_neg_distance(distances: np.ndarray, temperature: float = 0.85) -> np.ndarray:
    scaled = -distances / max(float(temperature), 1e-6)
    scaled = scaled - np.nanmax(scaled, axis=1, keepdims=True)
    exp = np.exp(np.clip(scaled, -60, 60))
    denom = exp.sum(axis=1, keepdims=True)
    return exp / np.where(denom == 0, 1.0, denom)


def apply_race_quality_v2_runtime(runners: pd.DataFrame) -> pd.DataFrame:
    """Attach shadow race-quality v2 diagnosis.

    This is intentionally not a BUY gate. It exposes a calibrated race-shape read
    for dashboard, LINE, and post-race audit while leaving final ticket logic intact.
    """
    out = runners.copy()
    if out.empty or "race_id" not in out.columns:
        return out
    model = load_race_quality_v2_model()
    classes = (model or {}).get("classes", ["fast", "slow", "instant", "sustain"])
    if model is None:
        out["race_quality_v2_predicted_lap_mode"] = "unknown"
        out["race_quality_v2_confidence"] = 0.0
        out["race_quality_v2_margin"] = 0.0
        for klass in classes:
            out[f"race_quality_v2_prob_{klass}"] = 0.0
        out["race_quality_v2_runtime_tag"] = "v2_model_missing"
        return out

    idx = out.index
    expected = out.get("expected_pace", pd.Series("", index=idx)).astype(str).str.lower()
    expected_fast = expected.str.contains("fast").astype(float)
    expected_slow = expected.str.contains("slow").astype(float)
    expected_middle = expected.str.contains("middle|mid", regex=True).astype(float)
    pressure = first_series(out, ["race_early_pressure_score", "race_front_runner_ratio"], 0.0).fillna(0.0).clip(0.0, 1.0)
    collapse = first_series(out, ["race_pace_collapse_risk"], 0.0).fillna(0.0).clip(0.0, 1.0)
    slow = first_series(out, ["race_slow_pace_risk"], 0.0).fillna(0.0).clip(0.0, 1.0)
    front_adv = first_series(out, ["front_advantage_score"], 0.0).fillna(0.0).clip(0.0, 1.0)
    closer_adv = first_series(out, ["closer_advantage_score"], 0.0).fillna(0.0).clip(0.0, 1.0)
    pos_adv = first_series(out, ["positioning_advantage_score"], 0.5).fillna(0.5).clip(0.0, 1.0)

    lap_profile = normalize_profile(
        pd.DataFrame(
            {
                "fast": 0.38 * pressure + 0.30 * collapse + 0.20 * expected_fast + 0.12 * (1.0 - slow),
                "slow": 0.42 * slow + 0.20 * expected_slow + 0.18 * (1.0 - pressure) + 0.20 * front_adv,
                "instant": 0.46 * slow + 0.20 * expected_slow + 0.18 * closer_adv + 0.16 * (1.0 - pressure),
                "sustain": 0.34 * pressure + 0.26 * collapse + 0.18 * expected_middle + 0.12 * pos_adv + 0.10 * (1.0 - slow),
                "long_spurt": 0.24 * pressure + 0.22 * collapse + 0.18 * closer_adv + 0.18 * pos_adv + 0.18 * expected_middle,
            },
            index=idx,
        )
    )
    profile_conc = lap_profile.max(axis=1)
    profile_conf = (
        0.58 * ((profile_conc - 0.20) / 0.80).clip(0.0, 1.0)
        + 0.22 * (pressure - slow).abs().clip(0.0, 1.0)
        + 0.20 * (1.0 - (1.0 - (collapse - slow).abs().clip(0.0, 1.0)).clip(0.0, 1.0) * 0.35)
    ).clip(0.0, 1.0)

    race = pd.DataFrame({"race_id": out["race_id"].astype(str)}, index=idx)
    for mode in ["fast", "slow", "instant", "sustain", "long_spurt"]:
        race[f"avg_race_need_{mode}"] = lap_profile[mode]
    race["v1_confidence"] = profile_conf
    race["v1_concentration"] = profile_conc
    race["field_size_lap_rows"] = first_series(out, ["field_size"], 0.0).fillna(0.0)
    for mode in ["fast", "slow", "instant", "sustain", "long_spurt"]:
        col = f"horse_{mode}_lap_score_past5"
        race[f"horse_lap_{mode}"] = first_series(out, [col], 0.0).fillna(0.0).clip(0.0, 1.0)

    grouped = race.groupby("race_id", sort=False)
    race_features = grouped[
        ["field_size_lap_rows", "v1_confidence", "v1_concentration"]
        + [f"avg_race_need_{m}" for m in ["fast", "slow", "instant", "sustain", "long_spurt"]]
    ].mean()
    for mode in ["fast", "slow", "instant", "sustain", "long_spurt"]:
        race_features[f"field_mean_horse_lap_{mode}"] = grouped[f"horse_lap_{mode}"].mean()
        race_features[f"field_max_horse_lap_{mode}"] = grouped[f"horse_lap_{mode}"].max()
        race_features[f"field_std_horse_lap_{mode}"] = grouped[f"horse_lap_{mode}"].std(ddof=0).fillna(0.0)

    race_level = (
        out[
            [
                "race_id",
                *[
                    c
                    for c in [
                        "runtime_queue_clarity_score",
                        "runtime_front_duel_risk_score",
                        "runtime_projected_front_load_score",
                        "runtime_lead_top_gap",
                        "runtime_lead_candidate_count",
                        "runtime_front5_candidate_count",
                        "runtime_front5_top_gap",
                        "hist_avg_front3f_sec",
                        "hist_condition_sample_count",
                    ]
                    if c in out.columns
                ],
            ]
        ]
        .drop_duplicates("race_id", keep="first")
        .set_index("race_id")
    )
    race_features = race_features.join(race_level, how="left")
    def rf_numeric(name: str, default: float = np.nan) -> pd.Series:
        if name in race_features.columns:
            return pd.to_numeric(race_features[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return pd.Series(default, index=race_features.index, dtype=float)

    race_features["queue_clarity_score"] = rf_numeric("runtime_queue_clarity_score").fillna(0.5)
    race_features["queue_duel_risk_score"] = rf_numeric("runtime_front_duel_risk_score").fillna(0.5)
    race_features["queue_front_load_score"] = rf_numeric("runtime_projected_front_load_score").fillna(0.5)
    race_features["queue_top_gap"] = rf_numeric("runtime_lead_top_gap").fillna(
        rf_numeric("runtime_front5_top_gap")
    ).fillna(0.0)
    race_features["queue_candidate_count"] = rf_numeric("runtime_lead_candidate_count").fillna(
        rf_numeric("runtime_front5_candidate_count")
    ).fillna(0.0)
    race_features["course_front3f_prior_sec"] = rf_numeric("hist_avg_front3f_sec")
    race_features["course_front3f_prior_std"] = np.nan
    race_features["course_front3f_prior_count"] = rf_numeric("hist_condition_sample_count").fillna(0.0)
    race_features["race_course_adj_ten_pressure_score"] = race_features["queue_front_load_score"]
    race_features["race_course_adj_fast_start_count"] = race_features["queue_candidate_count"]
    race_features["race_course_adj_ten_speed_gap_top2"] = race_features["queue_top_gap"]
    race_features["race_course_adj_queue_clarity_score"] = race_features["queue_clarity_score"]

    duel = race_features["queue_duel_risk_score"].clip(0.0, 1.0)
    clarity = race_features["queue_clarity_score"].clip(0.0, 1.0)
    front_load = race_features["queue_front_load_score"].clip(0.0, 1.0)
    course_pressure = race_features["race_course_adj_ten_pressure_score"].clip(0.0, 1.0)
    course_clarity = race_features["race_course_adj_queue_clarity_score"].clip(0.0, 1.0)
    fast_count = (race_features["race_course_adj_fast_start_count"].clip(0.0, 8.0) / 8.0).fillna(0.0)
    ten_gap = race_features["race_course_adj_ten_speed_gap_top2"].clip(0.0, 1.0).fillna(0.0)
    race_features["shape_fast_signal"] = (0.34 * duel + 0.26 * front_load + 0.22 * course_pressure + 0.18 * fast_count).clip(0.0, 1.0)
    race_features["shape_slow_signal"] = (0.34 * clarity + 0.26 * course_clarity + 0.22 * ten_gap + 0.18 * (1.0 - duel)).clip(0.0, 1.0)
    race_features["shape_sustain_signal"] = (
        0.34 * front_load + 0.24 * duel + 0.22 * course_pressure + 0.20 * (1.0 - (clarity - 0.5).abs() * 2.0)
    ).clip(0.0, 1.0)
    race_features["shape_instant_signal"] = (
        0.38 * race_features["shape_slow_signal"] + 0.24 * (1.0 - front_load) + 0.20 * clarity + 0.18 * (1.0 - course_pressure)
    ).clip(0.0, 1.0)
    race_features["shape_uncertainty_signal"] = (
        1.0 - (race_features["shape_fast_signal"] - race_features["shape_slow_signal"]).abs()
    ).clip(0.0, 1.0)

    features = model.get("features", [])
    x = pd.DataFrame(index=race_features.index)
    for feature in features:
        x[feature] = pd.to_numeric(race_features.get(feature), errors="coerce")
    med = pd.Series(model.get("med", {}), dtype=float)
    mean = pd.Series(model.get("mean", {}), dtype=float)
    std = pd.Series(model.get("std", {}), dtype=float).replace(0.0, 1.0)
    x = x.fillna(med).fillna(0.0)
    z = (x - mean.reindex(features).fillna(0.0)) / std.reindex(features).fillna(1.0)
    dists = []
    for klass in classes:
        centroid = pd.Series(model.get("centroids", {}).get(klass, {}), dtype=float).reindex(features).fillna(0.0)
        diff = z - centroid
        prior = max(float(model.get("priors", {}).get(klass, 0.0)), 1e-4)
        dists.append((diff * diff).mean(axis=1).to_numpy(dtype=float) - 0.06 * np.log(prior))
    probs = _race_quality_v2_softmax_neg_distance(np.vstack(dists).T)
    best_idx = probs.argmax(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    diag = pd.DataFrame(index=race_features.index)
    for i, klass in enumerate(classes):
        diag[f"race_quality_v2_prob_{klass}"] = probs[:, i]
    diag["race_quality_v2_predicted_lap_mode"] = [classes[i] for i in best_idx]
    diag["race_quality_v2_confidence"] = probs.max(axis=1)
    diag["race_quality_v2_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    diag["race_quality_v2_runtime_tag"] = np.select(
        [
            diag["race_quality_v2_confidence"].ge(RACE_QUALITY_V2_CONF_HIGH),
            diag["race_quality_v2_confidence"].ge(RACE_QUALITY_V2_CONF_MID),
            diag["race_quality_v2_confidence"].ge(RACE_QUALITY_V2_CONF_LOW),
        ],
        ["v2_high_conf", "v2_mid_conf", "v2_low_conf"],
        default="v2_uncertain",
    )
    diag = diag.reset_index().rename(columns={"index": "race_id"})
    out = out.merge(diag, on="race_id", how="left")
    for klass in classes:
        out[f"race_quality_v2_prob_{klass}"] = pd.to_numeric(
            out.get(f"race_quality_v2_prob_{klass}"), errors="coerce"
        ).fillna(0.0)
    out["race_quality_v2_predicted_lap_mode"] = out["race_quality_v2_predicted_lap_mode"].fillna("unknown")
    out["race_quality_v2_confidence"] = pd.to_numeric(out["race_quality_v2_confidence"], errors="coerce").fillna(0.0)
    out["race_quality_v2_margin"] = pd.to_numeric(out["race_quality_v2_margin"], errors="coerce").fillna(0.0)
    out["race_quality_v2_runtime_tag"] = out["race_quality_v2_runtime_tag"].fillna("v2_unknown")
    return out


def pct_rank(series: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.notna().sum() <= 1 or x.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=x.index, dtype=float)
    ranked = x.rank(pct=True, ascending=True)
    if not higher_is_better:
        ranked = 1.0 - ranked
    return ranked.fillna(0.5).clip(0.0, 1.0)


def score_fast_clock_runtime(frame: pd.DataFrame) -> pd.Series:
    """Runtime proxy for whether a horse can handle a relatively fast clock."""
    avg_time = first_series(frame, ["past3_avg_time_value", "past3_avg_time_z"], 0.0).fillna(0.0)
    best_time = first_series(frame, ["past3_best_time_value", "prev_race_time_value"], 0.0).fillna(0.0)
    class_time = first_series(frame, ["prev_class_time_value_score"], 0.0).fillna(0.0)
    time_margin = first_series(frame, ["horse_time_value_plus_margin"], 0.0).fillna(0.0)
    fast_lap = first_series(frame, ["horse_fast_lap_score_past5"], 0.0).fillna(0.0)
    lap_fit = first_series(frame, ["lap_aptitude_fit_score"], 0.0).fillna(0.0)
    return (
        0.28 * norm01(avg_time, lo=-0.12, hi=0.12)
        + 0.18 * norm01(best_time, lo=-0.10, hi=0.18)
        + 0.18 * norm01(class_time, lo=-0.10, hi=0.18)
        + 0.16 * norm01(time_margin, lo=-0.18, hi=0.18)
        + 0.12 * norm01(fast_lap, lo=0.0, hi=0.75)
        + 0.08 * norm01(lap_fit, lo=0.0, hi=0.40)
    ).clip(0.0, 1.0)


def score_corner_shape_runtime(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Runtime proxy for venue/distance-shape fit; low-sample is kept visible."""
    source_cols = {
        "same_venue_starts",
        "same_venue_top3_rate",
        "same_venue_avg_score",
        "same_distance_category_starts",
        "same_distance_category_top3_rate",
        "same_distance_category_avg_score",
    }
    if not any(col in frame.columns for col in source_cols):
        return (
            pd.Series(0.5, index=frame.index, dtype=float),
            pd.Series(0.0, index=frame.index, dtype=float),
        )
    venue_starts = first_series(frame, ["same_venue_starts"], 0.0).fillna(0.0)
    venue_top3 = first_series(frame, ["same_venue_top3_rate"], 0.0).fillna(0.0).clip(0.0, 1.0)
    venue_score = first_series(frame, ["same_venue_avg_score"], 0.0).fillna(0.0)
    dist_starts = first_series(frame, ["same_distance_category_starts"], 0.0).fillna(0.0)
    dist_top3 = first_series(frame, ["same_distance_category_top3_rate"], 0.0).fillna(0.0).clip(0.0, 1.0)
    dist_score = first_series(frame, ["same_distance_category_avg_score"], 0.0).fillna(0.0)
    experience = np.maximum(norm01(venue_starts, lo=0.0, hi=5.0), norm01(dist_starts, lo=0.0, hi=5.0))
    fit = (
        0.35 * norm01(venue_score, lo=0.0, hi=0.65)
        + 0.25 * venue_top3
        + 0.20 * norm01(dist_score, lo=0.0, hi=0.65)
        + 0.10 * dist_top3
        + 0.10 * experience
    ).clip(0.0, 1.0)
    low_sample = (venue_starts.lt(2.0) & dist_starts.lt(2.0)).astype(float)
    return fit, low_sample


def runtime_class_group(value: object) -> str:
    raw = text(value)
    if "新馬" in raw:
        return "新馬"
    if "未勝利" in raw:
        return "未勝利"
    if "1勝" in raw or "500万" in raw:
        return "1勝"
    if "2勝" in raw or "1000万" in raw:
        return "2勝"
    if "3勝" in raw or "1600万" in raw:
        return "3勝"
    if any(token in raw for token in ["Ｇ１", "G1", "GⅠ", "ＧⅠ"]):
        return "G1"
    if any(token in raw for token in ["Ｇ２", "G2", "GⅡ", "ＧⅡ"]):
        return "G2"
    if any(token in raw for token in ["Ｇ３", "G3", "GⅢ", "ＧⅢ"]):
        return "G3"
    if "ｵｰﾌﾟﾝ" in raw or "オープン" in raw or raw.upper() == "OP":
        return "OP"
    if "L" in raw or "リステッド" in raw:
        return "L"
    return raw or "不明"


def runtime_surface_key(value: object) -> str:
    raw = text(value)
    if raw.startswith("芝") or raw.lower() in {"turf", "grass"}:
        return "芝"
    if raw.startswith("ダ") or raw.lower() in {"dirt", "sand"}:
        return "ダ"
    if raw.startswith("障"):
        return "障"
    return raw


def load_historical_condition_context(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    frame = read_csv(path)
    required = {"years", "scope", "surface", "distance", "sample_count"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()
    out = frame.copy()
    for col in [
        "years",
        "distance",
        "sample_count",
        "avg_winning_time_sec",
        "avg_front3f_sec",
        "avg_last3f_sec",
        "avg_1000m_sec",
        "avg_rpci",
        "avg_pci3",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["venue", "surface", "class_group", "going", "scope"]:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("__ALL__").astype(str)
    out["surface"] = out["surface"].map(runtime_surface_key)
    return out


def best_historical_condition_row(context: pd.DataFrame, race: pd.Series, years: int = 5) -> pd.Series | None:
    if context.empty:
        return None
    venue = text(race.get("場所"))
    surface = runtime_surface_key(race.get("芝・ダ"))
    distance = num(race.get("距離"))
    class_key = runtime_class_group(race.get("クラス名") or race.get("レース名"))
    going = text(race.get("runtime_going") or race.get("馬場状態"))
    if not surface or not np.isfinite(distance):
        return None
    base = context[
        context["years"].eq(years)
        & context["surface"].eq(surface)
        & context["distance"].eq(int(distance))
    ].copy()
    if base.empty:
        return None
    candidate_defs = [
        ("同場同距離×クラス×馬場", venue, class_key, going, 1),
        ("同場同距離×クラス", venue, class_key, "__ALL__", 2),
        ("同場同距離×馬場", venue, "__ALL__", going, 3),
        ("同場同距離", venue, "__ALL__", "__ALL__", 4),
        ("同芝ダ同距離×クラス", "__ALL__", class_key, "__ALL__", 5),
        ("同芝ダ同距離", "__ALL__", "__ALL__", "__ALL__", 6),
    ]
    for scope, venue_key, class_group, going_key, priority in candidate_defs:
        part = base[base["scope"].eq(scope)].copy()
        if venue_key != "__ALL__":
            part = part[part["venue"].eq(venue_key)]
        if class_group != "__ALL__":
            part = part[part["class_group"].eq(class_group)]
        if going_key != "__ALL__":
            part = part[part["going"].eq(going_key)]
        if part.empty:
            continue
        part["_priority"] = priority
        return part.sort_values(["_priority", "sample_count"], ascending=[True, False]).iloc[0]
    return None


def apply_historical_race_quality_context(runners: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    out = runners.copy()
    defaults = {
        "hist_condition_scope": "",
        "hist_condition_sample_count": 0.0,
        "hist_avg_winning_time_sec": np.nan,
        "hist_avg_front3f_sec": np.nan,
        "hist_avg_last3f_sec": np.nan,
        "hist_avg_1000m_sec": np.nan,
        "hist_avg_rpci": np.nan,
        "hist_avg_pci3": np.nan,
        "race_quality_expected_front3f_sec": np.nan,
        "race_quality_front3f_delta_sec": 0.0,
        "race_quality_expected_rpci": np.nan,
        "race_quality_rpci_delta": 0.0,
        "race_quality_fast_need_score": 0.0,
        "race_quality_slow_need_score": 0.0,
        "race_quality_fit_score": 0.5,
        "race_quality_label": "unknown",
        "race_quality_context_ready": 0.0,
    }
    for col, value in defaults.items():
        out[col] = value
    if context.empty or out.empty:
        return out

    updates: list[pd.DataFrame] = []
    for race_id, part in out.groupby("race_id", sort=False):
        race = part.iloc[0]
        row = best_historical_condition_row(context, race, years=5)
        if row is None:
            continue
        idx = part.index
        sample = num(row.get("sample_count"), 0.0)
        hist_front = num(row.get("avg_front3f_sec"), np.nan)
        hist_rpci = num(row.get("avg_rpci"), np.nan)
        pressure = pd.to_numeric(part.get("race_early_pressure_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
        collapse = pd.to_numeric(part.get("race_pace_collapse_risk"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
        slow = pd.to_numeric(part.get("race_slow_pace_risk"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
        race_pressure = float(pressure.max()) if pressure.notna().any() else 0.0
        race_collapse = float(collapse.max()) if collapse.notna().any() else 0.0
        race_slow = float(slow.max()) if slow.notna().any() else 0.0
        front_adj = float(np.clip(-0.80 * race_pressure - 0.55 * race_collapse + 0.70 * race_slow, -1.8, 1.8))
        rpci_adj = float(np.clip(-2.8 * race_collapse - 1.0 * race_pressure + 2.4 * race_slow, -5.0, 5.0))
        expected_front = hist_front + front_adj if np.isfinite(hist_front) else np.nan
        expected_rpci = hist_rpci + rpci_adj if np.isfinite(hist_rpci) else np.nan
        front_delta = expected_front - hist_front if np.isfinite(expected_front) and np.isfinite(hist_front) else 0.0
        rpci_delta = expected_rpci - hist_rpci if np.isfinite(expected_rpci) and np.isfinite(hist_rpci) else 0.0
        fast_need = float(np.clip(((-front_delta) / 1.4 + (-rpci_delta) / 5.0) / 2.0, 0.0, 1.0))
        slow_need = float(np.clip((front_delta / 1.4 + rpci_delta / 5.0) / 2.0, 0.0, 1.0))
        sustain_need = float(np.clip(1.0 - max(fast_need, slow_need), 0.0, 1.0))
        fast_clock = pd.to_numeric(part.get("fast_clock_runtime_score"), errors="coerce").fillna(0.5).clip(0.0, 1.0)
        fast_lap = pd.to_numeric(part.get("horse_fast_lap_score_past5"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
        slow_lap = pd.to_numeric(part.get("horse_slow_lap_score_past5"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
        sustain_lap = pd.to_numeric(part.get("horse_sustain_lap_score_past5"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
        long_lap = pd.to_numeric(part.get("horse_long_spurt_lap_score_past5"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
        pace_fit = pd.to_numeric(part.get("pace_fit_runtime_score"), errors="coerce").fillna(
            pd.to_numeric(part.get("pace_fit_score"), errors="coerce")
        ).fillna(0.0).clip(0.0, 1.0)
        front_rate = pd.to_numeric(part.get("horse_front_run_rate_past5"), errors="coerce").fillna(
            pd.to_numeric(part.get("front_running_tendency"), errors="coerce")
        ).fillna(0.0).clip(0.0, 1.0)
        fast_fit = (0.52 * fast_clock + 0.30 * fast_lap + 0.18 * pace_fit).clip(0.0, 1.0)
        slow_fit = (0.42 * slow_lap + 0.25 * front_rate + 0.20 * pace_fit + 0.13 * fast_clock).clip(0.0, 1.0)
        sustain_fit = (0.34 * sustain_lap + 0.24 * long_lap + 0.24 * pace_fit + 0.18 * fast_clock).clip(0.0, 1.0)
        quality_fit = (
            fast_need * fast_fit
            + slow_need * slow_fit
            + sustain_need * sustain_fit
        ).clip(0.0, 1.0)
        if sample < 3:
            quality_fit = (0.70 * quality_fit + 0.30 * 0.5).clip(0.0, 1.0)
        label = "fast" if fast_need >= 0.38 else ("slow" if slow_need >= 0.38 else "standard")
        race_update = pd.DataFrame(
            {
                "hist_condition_scope": text(row.get("scope")),
                "hist_condition_sample_count": sample,
                "hist_avg_winning_time_sec": num(row.get("avg_winning_time_sec")),
                "hist_avg_front3f_sec": hist_front,
                "hist_avg_last3f_sec": num(row.get("avg_last3f_sec")),
                "hist_avg_1000m_sec": num(row.get("avg_1000m_sec")),
                "hist_avg_rpci": hist_rpci,
                "hist_avg_pci3": num(row.get("avg_pci3")),
                "race_quality_expected_front3f_sec": expected_front,
                "race_quality_front3f_delta_sec": front_delta,
                "race_quality_expected_rpci": expected_rpci,
                "race_quality_rpci_delta": rpci_delta,
                "race_quality_fast_need_score": fast_need,
                "race_quality_slow_need_score": slow_need,
                "race_quality_fit_score": quality_fit,
                "race_quality_label": label,
                "race_quality_context_ready": 1.0,
            },
            index=idx,
        )
        updates.append(race_update)
    if updates:
        update = pd.concat(updates, axis=0)
        for col in update.columns:
            out.loc[update.index, col] = update[col]
    return out


def infer_surface_type(frame: pd.DataFrame) -> pd.Series:
    surface_cols = [
        c
        for c in frame.columns
        if c.startswith("anchor_") and ("surface" in c.lower() or "芝" in c or "闃" in c)
    ]
    if not surface_cols:
        surface_cols = [
            c
            for c in frame.columns
            if c.startswith("partner_") and ("surface" in c.lower() or "芝" in c or "闃" in c)
        ]
    raw = first_text_series(frame, surface_cols[:1], "")
    is_turf = raw.str.contains("芝", regex=False) | raw.str.contains("ЋЕ", regex=False)
    is_dirt = raw.str.contains("ダ", regex=False) | raw.str.contains("ѓ_", regex=False)
    return pd.Series(np.select([is_turf, is_dirt], ["turf", "dirt"], default="unknown"), index=frame.index)


def add_pair_pace_regime_scores(
    frame: pd.DataFrame,
    pressure_pair: pd.Series,
    front_pair: pd.Series,
) -> pd.DataFrame:
    out = frame.copy()
    venue_code = out["race_id"].astype(str).str.zfill(16).str.slice(8, 10)
    surface_type = infer_surface_type(out)
    front_scores: list[float] = []
    closer_scores: list[float] = []
    for code, surface in zip(venue_code, surface_type):
        front, closer = PACE_REGIME_BY_VENUE_SURFACE.get(
            (str(code), str(surface)),
            PACE_REGIME_SURFACE_DEFAULT.get(str(surface), PACE_REGIME_SURFACE_DEFAULT["unknown"]),
        )
        front_scores.append(float(front))
        closer_scores.append(float(closer))
    out["pace_regime_surface_type"] = surface_type
    out["pace_regime_front_survival_score"] = pd.Series(front_scores, index=out.index).clip(0.0, 1.0)
    out["pace_regime_collapse_conversion_score"] = pd.Series(closer_scores, index=out.index).clip(0.0, 1.0)
    pressure = pd.to_numeric(pressure_pair, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    front = pd.to_numeric(front_pair, errors="coerce").fillna(0.5).clip(0.0, 1.0)
    out["pace_regime_position_lock_score"] = (
        0.50
        + 0.55 * (out["pace_regime_front_survival_score"] - out["pace_regime_collapse_conversion_score"])
        + 0.08 * venue_code.isin(["01", "02", "03", "10"]).astype(float)
    ).clip(0.0, 1.0)
    out["pace_regime_front_fit_score"] = (
        front * out["pace_regime_front_survival_score"] * pressure
    ).clip(0.0, 1.0)
    out["pace_regime_front_collapse_risk_score"] = (
        front * out["pace_regime_collapse_conversion_score"] * pressure
    ).clip(0.0, 1.0)
    out["pace_regime_collapse_warning_flag"] = (
        pressure.ge(0.55)
        & front.ge(0.55)
        & out["pace_regime_collapse_conversion_score"].ge(0.13)
    ).astype(float)
    return out


def distance_bin_from_number(values: pd.Series) -> pd.Series:
    d = pd.to_numeric(values, errors="coerce")
    bins = [-np.inf, 1200, 1400, 1600, 1800, 2000, 2400, np.inf]
    labels = ["<=1200", "1201-1400", "1401-1600", "1601-1800", "1801-2000", "2001-2400", "2401+"]
    return pd.cut(d, bins=bins, labels=labels).astype("string").fillna("unknown").astype(str)


def runtime_class_group(values: pd.Series | object) -> pd.Series | str:
    scalar_input = not isinstance(values, pd.Series)
    if scalar_input:
        values = pd.Series([values])
    s = values.fillna("").astype(str)
    out = pd.Series("other", index=values.index, dtype=object)
    out[s.str.contains("新馬", regex=False)] = "newcomer"
    out[s.str.contains("未勝利", regex=False)] = "maiden"
    out[s.str.contains("1勝|500万", regex=True)] = "1win"
    out[s.str.contains("2勝|1000万", regex=True)] = "2win"
    out[s.str.contains("3勝|1600万", regex=True)] = "3win"
    out[s.str.contains("オープン|OP|L|リステッド", regex=True)] = "open"
    out[s.str.contains("G1|Ｇ１|G2|Ｇ２|G3|Ｇ３|重賞", regex=True)] = "graded"
    out = out.astype(str)
    return str(out.iloc[0]) if scalar_input else out


def runtime_going_group(values: pd.Series) -> pd.Series:
    s = values.fillna("").astype(str)
    out = pd.Series("unknown", index=values.index, dtype=object)
    out[s.str.contains("良", regex=False)] = "firm"
    out[s.str.contains("稍", regex=False)] = "yielding"
    out[s.str.contains("重", regex=False)] = "soft"
    out[s.str.contains("不", regex=False)] = "heavy"
    return out.astype(str)


def load_front_survival_context_lookup(path: Path = FRONT_SURVIVAL_CONTEXT_LOOKUP_PATH) -> dict[tuple[str, str, str, str, str], dict[str, float]]:
    if not path.exists():
        return {}
    try:
        lookup = read_csv(path, dtype=str)
    except Exception:
        return {}
    out: dict[tuple[str, str, str, str, str], dict[str, float]] = {}
    for _, row in lookup.iterrows():
        key = (
            text(row.get("venue_code")) or "*",
            text(row.get("surface_type")) or "*",
            text(row.get("distance_bin")) or "*",
            text(row.get("class_group")) or "*",
            text(row.get("going_group")) or "*",
        )
        out[key] = {
            "context_races": num(row.get("context_races"), 0.0),
            "front_survival_rate": num(row.get("front_survival_rate"), FRONT_SURVIVAL_CONTEXT_DEFAULT["front_survival_rate"]),
            "front_collapse_rate": num(row.get("front_collapse_rate"), FRONT_SURVIVAL_CONTEXT_DEFAULT["front_collapse_rate"]),
            "top3_front5_share": num(row.get("top3_front5_share"), FRONT_SURVIVAL_CONTEXT_DEFAULT["top3_front5_share"]),
            "front_context_readability_score": num(row.get("front_context_readability_score"), 0.0),
        }
    return out


def add_front_survival_context_scores(
    frame: pd.DataFrame,
    pressure_pair: pd.Series,
    front_pair: pd.Series,
) -> pd.DataFrame:
    """Attach shadow front-survival/collapse context from historical course buckets.

    The lookup is generated by scripts/evaluate_front_survival_context_features.py.
    These columns are diagnostics only; they must not promote tickets to BUY.
    """
    out = frame.copy()
    idx = out.index
    lookup = load_front_survival_context_lookup()
    venue_code = out["race_id"].astype(str).str.zfill(16).str.slice(8, 10)
    surface_type = infer_surface_type(out).astype(str)
    distance = first_series(out, ["anchor_距離", "partner_距離", "anchor_霍晞屬", "partner_霍晞屬"], np.nan)
    distance_bin = distance_bin_from_number(distance)
    class_text = first_text_series(out, ["anchor_クラス名", "partner_クラス名", "anchor_繧ｯ繝ｩ繧ｹ蜷・", "partner_繧ｯ繝ｩ繧ｹ蜷・"], "")
    class_bucket = runtime_class_group(class_text)
    going_text = first_text_series(out, ["anchor_runtime_going", "partner_runtime_going", "anchor_馬場状態", "partner_馬場状態"], "")
    going_bucket = runtime_going_group(going_text)

    context_rows: list[dict[str, float | str]] = []
    for code, surface, dist, cls, going in zip(venue_code, surface_type, distance_bin, class_bucket, going_bucket):
        key_candidates = [
            (str(code), str(surface), str(dist), str(cls), str(going)),
            (str(code), str(surface), str(dist), str(cls), "*"),
            (str(code), str(surface), str(dist), "*", "*"),
            (str(code), str(surface), "*", "*", "*"),
            ("*", str(surface), str(dist), "*", "*"),
            ("*", "*", "*", "*", "*"),
        ]
        found = None
        level = "default"
        for key in key_candidates:
            if key in lookup:
                found = lookup[key]
                level = "|".join(key)
                break
        if found is None:
            found = FRONT_SURVIVAL_CONTEXT_DEFAULT
        context_rows.append(
            {
                "front_context_lookup_key": level,
                "front_context_races": float(found.get("context_races", 0.0)),
                "front_context_survival_rate": float(found.get("front_survival_rate", FRONT_SURVIVAL_CONTEXT_DEFAULT["front_survival_rate"])),
                "front_context_collapse_rate": float(found.get("front_collapse_rate", FRONT_SURVIVAL_CONTEXT_DEFAULT["front_collapse_rate"])),
                "front_context_top3_front5_share": float(found.get("top3_front5_share", FRONT_SURVIVAL_CONTEXT_DEFAULT["top3_front5_share"])),
                "front_context_readability_score": float(found.get("front_context_readability_score", 0.0)),
            }
        )
    context = pd.DataFrame(context_rows, index=idx)
    for col in context.columns:
        out[col] = context[col]

    pressure = pd.to_numeric(pressure_pair, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    front = pd.to_numeric(front_pair, errors="coerce").fillna(0.5).clip(0.0, 1.0)
    collapse = np.maximum(
        pd.to_numeric(out.get("anchor_race_pace_collapse_risk"), errors="coerce").fillna(0.0),
        pd.to_numeric(out.get("partner_race_pace_collapse_risk"), errors="coerce").fillna(0.0),
    ).clip(0.0, 1.0)
    slow = np.maximum(
        pd.to_numeric(out.get("anchor_race_slow_pace_risk"), errors="coerce").fillna(0.0),
        pd.to_numeric(out.get("partner_race_slow_pace_risk"), errors="coerce").fillna(0.0),
    ).clip(0.0, 1.0)
    out["front_context_high_pressure_signal"] = (
        0.42 * collapse
        + 0.34 * pressure
        + 0.16 * front
        - 0.12 * slow
    ).clip(0.0, 1.0)
    out["front_context_survival_support_score"] = (
        out["front_context_high_pressure_signal"]
        * (0.70 * out["front_context_survival_rate"] + 0.30 * out["front_context_top3_front5_share"])
        - 0.30 * out["front_context_collapse_rate"]
    ).clip(0.0, 1.0)
    out["front_context_collapse_risk_score"] = (
        out["front_context_high_pressure_signal"]
        * (0.75 * out["front_context_collapse_rate"] + 0.25 * (1.0 - out["front_context_survival_rate"]))
        * (0.70 + 0.30 * front)
    ).clip(0.0, 1.0)
    out["front_context_gate_label"] = np.select(
        [
            out["front_context_collapse_risk_score"].ge(0.018) & front.ge(0.60),
            out["front_context_collapse_risk_score"].ge(0.012) & front.ge(0.60),
            out["front_context_readability_score"].ge(0.46) & out["front_context_survival_support_score"].ge(0.161),
        ],
        [
            "front_context_collapse_alert",
            "front_context_collapse_watch",
            "front_context_survival_watch",
        ],
        default="front_context_neutral",
    )
    out["front_context_gate_note"] = np.select(
        [
            out["front_context_gate_label"].eq("front_context_collapse_alert"),
            out["front_context_gate_label"].eq("front_context_collapse_watch"),
            out["front_context_gate_label"].eq("front_context_survival_watch"),
        ],
        [
            "前崩れ文脈が強く、前目ペアは慎重",
            "前崩れ文脈に注意",
            "前が残る文脈は比較的読みやすい",
        ],
        default="",
    )
    out["front_context_lookup_available"] = out["front_context_races"].gt(0).astype(float)
    return out


def parse_date_key(value: object) -> str:
    raw = text(value)
    if not raw:
        return ""
    for fmt in ("%Y.%m.%d", "%Y-%m-%d", "%Y%m%d", "%y%m%d"):
        try:
            parsed = datetime.strptime(raw, fmt)
            if 2000 <= parsed.year <= 2099:
                return parsed.strftime("%Y%m%d")
        except Exception:
            pass
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 6:
        return "20" + digits
    return digits[:8] if len(digits) >= 8 else ""


def parse_race_post_datetime(race_id: object, start_time: object) -> datetime | None:
    rid = text(race_id)
    date_key = rid[:8] if len(rid) >= 8 and rid[:8].isdigit() else ""
    match = re.search(r"(\d{1,2}):(\d{2})", text(start_time))
    if not date_key or not match:
        return None
    try:
        return datetime(
            int(date_key[:4]),
            int(date_key[4:6]),
            int(date_key[6:8]),
            int(match.group(1)),
            int(match.group(2)),
        )
    except Exception:
        return None


def add_race_post_time_flags(runners: pd.DataFrame, now: datetime | None = None) -> pd.DataFrame:
    out = runners.copy()
    current = now or datetime.now()
    start_series = first_text_series(out, ["発走時刻", "start_time"], "")
    post_times = [parse_race_post_datetime(rid, start) for rid, start in zip(out["race_id"], start_series)]
    out["race_post_at"] = [dt.isoformat(timespec="minutes") if dt is not None else "" for dt in post_times]
    out["race_minutes_to_post"] = [
        (dt - current).total_seconds() / 60.0 if dt is not None else np.nan for dt in post_times
    ]
    out["race_started_flag"] = [
        1.0 if dt is not None and current >= dt else 0.0 for dt in post_times
    ]
    out["runtime_decision_generated_at"] = current.isoformat(timespec="seconds")
    return out


def going_class(value: object) -> str:
    raw = text(value)
    if raw in {"良", "Good"}:
        return "Good"
    if raw in {"稍", "稍重", "Yielding"}:
        return "Yielding"
    if raw in {"重", "Soft"}:
        return "Soft"
    if raw in {"不", "不良", "Heavy"}:
        return "Heavy"
    return "Unknown"


def apply_runtime_track_conditions(runners: pd.DataFrame, track: pd.DataFrame) -> pd.DataFrame:
    out = runners.copy()
    out["runtime_track_condition_available"] = 0.0
    out["runtime_going"] = first_text_series(out, ["馬場状態"], "")
    out["runtime_going_class"] = out["runtime_going"].map(going_class)
    out["runtime_soft_heavy_flag"] = out["runtime_going_class"].isin(["Soft", "Heavy"]).astype(float)
    out["runtime_hakodate_flag"] = first_text_series(out, ["場所"], "").eq("函館").astype(float)
    if track.empty:
        return out

    needed = {"effective_date", "venue", "turf_going", "dirt_going"}
    if not needed.issubset(track.columns):
        return out

    current = track.copy()
    current["_date_key"] = current["effective_date"].map(parse_date_key)
    current["_venue_key"] = current["venue"].astype("string").fillna("").str.strip()
    current = current.drop_duplicates(["_date_key", "_venue_key"], keep="last")
    out["_date_key"] = first_text_series(out, ["日付S", "日付"], "").map(parse_date_key)
    out["_venue_key"] = first_text_series(out, ["場所"], "").str.strip()
    merged = out.merge(
        current[["_date_key", "_venue_key", "turf_going", "dirt_going", "timing", "weather"]],
        on=["_date_key", "_venue_key"],
        how="left",
    )
    surface = first_text_series(merged, ["芝・ダ"], "")
    turf_mask = surface.str.contains("芝|障", regex=True, na=False)
    dirt_mask = surface.str.contains("ダ", regex=True, na=False)
    turf = merged["turf_going"].astype("string").fillna("")
    dirt = merged["dirt_going"].astype("string").fillna("")
    runtime = np.select([turf_mask, dirt_mask], [turf, dirt], default="")
    merged["runtime_going"] = pd.Series(runtime, index=merged.index).replace("", np.nan)
    if "馬場状態" in merged.columns:
        merged["runtime_going"] = merged["runtime_going"].fillna(merged["馬場状態"])
        merged["馬場状態"] = merged["馬場状態"].where(merged["馬場状態"].astype("string").fillna("").ne(""), merged["runtime_going"])
    else:
        merged["馬場状態"] = merged["runtime_going"]
    merged["runtime_going"] = merged["runtime_going"].fillna("")
    merged["runtime_going_class"] = merged["runtime_going"].map(going_class)
    merged["runtime_track_condition_available"] = merged["runtime_going_class"].ne("Unknown").astype(float)
    merged["runtime_soft_heavy_flag"] = merged["runtime_going_class"].isin(["Soft", "Heavy"]).astype(float)
    merged["runtime_hakodate_flag"] = first_text_series(merged, ["場所"], "").eq("函館").astype(float)
    return merged.drop(columns=["_date_key", "_venue_key"], errors="ignore")


def normalize_venue_name(value: object) -> str:
    return text(value).replace(" ", "").replace("　", "")


def bucket_by_group(values: pd.Series, groups: pd.Series, labels: tuple[str, str, str]) -> pd.Series:
    out = pd.Series("unknown", index=values.index, dtype="object")
    for _, idx in groups.groupby(groups).groups.items():
        s = pd.to_numeric(values.loc[idx], errors="coerce")
        valid = s.notna()
        if valid.sum() < 20 or s.nunique(dropna=True) < 3:
            out.loc[idx[valid]] = labels[1]
            continue
        q1 = s.loc[valid].quantile(1 / 3)
        q2 = s.loc[valid].quantile(2 / 3)
        out.loc[idx[valid & s.le(q1)]] = labels[0]
        out.loc[idx[valid & s.gt(q1) & s.le(q2)]] = labels[1]
        out.loc[idx[valid & s.gt(q2)]] = labels[2]
    return out


def load_track_condition_metric_context(path: Path = TRACK_CONDITION_METRICS_DEFAULT) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        track = read_csv(path)
    except Exception:
        return pd.DataFrame()
    required = {
        "date",
        "venue",
        "cushion_value",
        "moisture_turf_goal",
        "moisture_turf_back",
        "moisture_dirt_goal",
        "moisture_dirt_back",
    }
    if not required.issubset(track.columns):
        return pd.DataFrame()
    out = track.copy()
    out["date_yyyymmdd"] = out["date"].astype(str).str.replace(r"\D", "", regex=True).str.slice(0, 8)
    out["venue_norm"] = out["venue"].map(normalize_venue_name)
    for col in [
        "cushion_value",
        "moisture_turf_goal",
        "moisture_turf_back",
        "moisture_dirt_goal",
        "moisture_dirt_back",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["turf_moisture_avg"] = out[["moisture_turf_goal", "moisture_turf_back"]].mean(axis=1)
    out["dirt_moisture_avg"] = out[["moisture_dirt_goal", "moisture_dirt_back"]].mean(axis=1)
    out["course_setting"] = out.get("course", "").astype("string").fillna("").str.strip()
    out["cushion_bucket"] = bucket_by_group(out["cushion_value"], out["venue_norm"], ("low_cushion", "mid_cushion", "high_cushion"))
    turf = out.copy()
    turf["surface_norm"] = "turf"
    turf["moisture_surface_avg"] = turf["turf_moisture_avg"]
    dirt = out.copy()
    dirt["surface_norm"] = "dirt"
    dirt["moisture_surface_avg"] = dirt["dirt_moisture_avg"]
    long = pd.concat([turf, dirt], ignore_index=True, sort=False)
    long["venue_surface"] = long["venue_norm"] + "_" + long["surface_norm"]
    long["moisture_bucket"] = bucket_by_group(long["moisture_surface_avg"], long["venue_surface"], ("dry_moisture", "mid_moisture", "wet_moisture"))
    keep = [
        "date_yyyymmdd",
        "venue_norm",
        "surface_norm",
        "course_setting",
        "cushion_value",
        "turf_moisture_avg",
        "dirt_moisture_avg",
        "moisture_surface_avg",
        "cushion_bucket",
        "moisture_bucket",
    ]
    return long[keep].drop_duplicates(["date_yyyymmdd", "venue_norm", "surface_norm"], keep="last")


def add_lap_track_shadow_features(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    if out.empty or "race_id" not in out.columns:
        out["lap_track_shadow_label"] = "neutral"
        out["lap_track_shadow_score"] = 0.0
        out["lap_track_shadow_note"] = ""
        return out

    race_id = out["race_id"].astype(str).str.zfill(16)
    out["_lap_track_date"] = race_id.str.slice(0, 8)
    out["_lap_track_venue_code"] = race_id.str.slice(8, 10)
    out["_lap_track_venue_norm"] = out["_lap_track_venue_code"].map(VENUE_NAME_BY_CODE).fillna("")
    surface = first_text_series(out, ["pace_regime_surface_type"], "").str.lower()
    missing_surface = surface.eq("") | surface.eq("unknown")
    if missing_surface.any():
        inferred = infer_surface_type(out).astype(str).str.lower()
        surface = surface.where(~missing_surface, inferred)
    out["_lap_track_surface_norm"] = surface.where(surface.isin(["turf", "dirt"]), "unknown")

    metrics = load_track_condition_metric_context()
    if not metrics.empty:
        out = out.drop(
            columns=[
                "date_yyyymmdd",
                "venue_norm",
                "surface_norm",
                "course_setting",
                "cushion_value",
                "turf_moisture_avg",
                "dirt_moisture_avg",
                "moisture_surface_avg",
                "cushion_bucket",
                "moisture_bucket",
            ],
            errors="ignore",
        )
        out = out.merge(
            metrics,
            left_on=["_lap_track_date", "_lap_track_venue_norm", "_lap_track_surface_norm"],
            right_on=["date_yyyymmdd", "venue_norm", "surface_norm"],
            how="left",
        )
    else:
        for col, default in {
            "course_setting": "",
            "cushion_value": np.nan,
            "turf_moisture_avg": np.nan,
            "dirt_moisture_avg": np.nan,
            "moisture_surface_avg": np.nan,
            "cushion_bucket": "unknown",
            "moisture_bucket": "unknown",
        }.items():
            out[col] = default

    going_class_pair = first_text_series(out, ["anchor_runtime_going_class", "partner_runtime_going_class"], "")
    wet_by_going = going_class_pair.isin(["Yielding", "Soft", "Heavy"])
    good_by_going = going_class_pair.eq("Good")
    out["moisture_bucket"] = out["moisture_bucket"].fillna("unknown").astype(str)
    out["cushion_bucket"] = out["cushion_bucket"].fillna("unknown").astype(str)
    out["moisture_bucket"] = out["moisture_bucket"].where(~(out["moisture_bucket"].eq("unknown") & wet_by_going), "wet_moisture")
    out["moisture_bucket"] = out["moisture_bucket"].where(~(out["moisture_bucket"].eq("unknown") & good_by_going), "dry_moisture")

    is_turf = out["_lap_track_surface_norm"].eq("turf")
    is_dirt = out["_lap_track_surface_norm"].eq("dirt")
    out["track_lap_regime"] = np.select(
        [
            is_turf & out["cushion_bucket"].eq("high_cushion") & out["moisture_bucket"].eq("dry_moisture"),
            is_turf & out["cushion_bucket"].eq("high_cushion"),
            is_turf & out["moisture_bucket"].eq("wet_moisture"),
            is_turf & out["cushion_bucket"].eq("low_cushion"),
            is_dirt & out["moisture_bucket"].eq("wet_moisture"),
            is_dirt & out["moisture_bucket"].eq("dry_moisture"),
        ],
        [
            "turf_fast_high_cushion_dry",
            "turf_high_cushion",
            "turf_wet_moisture",
            "turf_low_cushion",
            "dirt_wet_moisture",
            "dirt_dry_moisture",
        ],
        default="track_mid_or_unknown",
    )

    strict_gap = first_series(out, ["strict_waveform_pair_gap_score"], np.nan)
    goodrun_min = first_series(out, ["goodrun_lap_pair_min_score"], np.nan)
    role_proxy = first_series(out, ["lap_role_pair_probability_proxy"], np.nan)
    collision = first_series(out, ["lap_role_front_front_collision_risk"], np.nan)
    out["lap_signal_strict_gap_low"] = strict_gap.le(LAP_TRACK_STRICT_GAP_LOW_MAX).fillna(False)
    out["lap_signal_goodrun_min_high"] = goodrun_min.ge(LAP_TRACK_GOODRUN_MIN_HIGH).fillna(False)
    out["lap_signal_role_low_collision"] = (role_proxy.ge(LAP_TRACK_ROLE_LOW_COLLISION_MIN) & collision.le(LAP_TRACK_COLLISION_SAFE_MAX)).fillna(False)

    ticket_type = first_text_series(out, ["ticket_type"], "").str.lower()
    regime = out["track_lap_regime"].astype(str)
    positive_wet_strict = regime.eq("turf_wet_moisture") & out["lap_signal_strict_gap_low"] & ticket_type.eq("umaren")
    positive_fast_role = regime.eq("turf_fast_high_cushion_dry") & out["lap_signal_role_low_collision"]
    positive_fast_goodrun = regime.eq("turf_fast_high_cushion_dry") & out["lap_signal_goodrun_min_high"] & ticket_type.eq("umaren")
    positive_dirt_wet_wide = regime.eq("dirt_wet_moisture") & out["lap_signal_role_low_collision"] & ticket_type.eq("wide")
    caution = (
        (regime.eq("turf_wet_moisture") & out["lap_signal_role_low_collision"] & ticket_type.eq("umaren"))
        | (regime.eq("turf_low_cushion") & out["lap_signal_strict_gap_low"] & ticket_type.eq("umaren"))
        | (regime.eq("dirt_dry_moisture") & out["lap_signal_strict_gap_low"] & ticket_type.eq("umaren"))
    )

    label = pd.Series("neutral", index=out.index, dtype="object")
    score = pd.Series(0.0, index=out.index, dtype=float)
    note = pd.Series("", index=out.index, dtype="object")
    label.loc[positive_wet_strict] = "positive"
    score.loc[positive_wet_strict] = 0.90
    note.loc[positive_wet_strict] = "湿った芝でラップ波形がかみ合う馬連。過去検証では年別にも安定"
    label.loc[positive_fast_role] = "positive"
    score.loc[positive_fast_role] = np.maximum(score.loc[positive_fast_role], 0.82)
    note.loc[positive_fast_role] = "高速寄りの芝で役割衝突が少ないラップ適性"
    label.loc[positive_fast_goodrun] = "positive"
    score.loc[positive_fast_goodrun] = np.maximum(score.loc[positive_fast_goodrun], 0.78)
    note.loc[positive_fast_goodrun] = "高速寄りの芝で好走時ラップ適性が強い馬連"
    label.loc[positive_dirt_wet_wide] = "positive_soft"
    score.loc[positive_dirt_wet_wide] = np.maximum(score.loc[positive_dirt_wet_wide], 0.74)
    note.loc[positive_dirt_wet_wide] = "湿ったダートで役割衝突が少ないワイド。サンプルは小さめ"
    caution_only = caution & label.eq("neutral")
    label.loc[caution_only] = "caution"
    score.loc[caution_only] = 0.35
    note.loc[caution_only] = "総合ROIは高いが年別ブレあり。買い昇格ではなく注視止まり"

    out["lap_track_shadow_label"] = label
    out["lap_track_shadow_score"] = score
    out["lap_track_shadow_note"] = note
    return out.drop(
        columns=["_lap_track_date", "_lap_track_venue_code", "_lap_track_venue_norm", "_lap_track_surface_norm"],
        errors="ignore",
    )



def row_to_official_race_id(row: pd.Series) -> str:
    source_url = text(row.get("source_url"))
    match = re.search(r"race_id=(\d{12})", source_url)
    netkeiba_id = match.group(1) if match else ""
    if not netkeiba_id:
        raw_id = text(row.get("race_id") or row.get("レースID(新/馬番無)") or "")
        digits = re.sub(r"\D", "", raw_id)
        if len(digits) == 16:
            return digits
        if len(digits) >= 8 and len(digits) != 12:
            date_key = parse_date_key(row.get("日付S") or row.get("日付"))
            if not date_key:
                return ""
            tail = digits[-8:]
            venue = digits[:-8].zfill(2)
            return date_key + venue + tail[2:8]
        if len(digits) >= 12:
            netkeiba_id = digits[:12]
    date_key = parse_date_key(row.get("日付S") or row.get("日付"))
    if not date_key or not netkeiba_id or len(netkeiba_id) < 12:
        return ""
    return date_key + netkeiba_id[4:12]


def softmax_by_race(df: pd.DataFrame, score_col: str) -> pd.Series:
    out = pd.Series(0.0, index=df.index, dtype=float)
    for _, idx in df.groupby("race_id").groups.items():
        scores = pd.to_numeric(df.loc[idx, score_col], errors="coerce").fillna(df[score_col].median()).astype(float)
        scaled = (scores - scores.max()) / 0.11
        exp = np.exp(np.clip(scaled, -30.0, 0.0))
        denom = exp.sum()
        out.loc[idx] = exp / denom if denom else 1.0 / max(len(idx), 1)
    return out


STRONGEST_REQUIRED_COLUMNS = [
    "horse_front_run_rate_past5",
    "race_pace_collapse_risk",
    "race_slow_pace_risk",
    "pace_fit_score",
    "front_advantage_score",
    "draw_pace_fit_score",
    "workout_knowledge_grade_score",
    "workout_load_density_score",
]
DEFAULTABLE_REQUIRED_COLUMNS = {
    "workout_knowledge_grade_score",
    "workout_load_density_score",
}


FRONT5_MODEL_PARAMS_PATH = ROOT / "outputs" / "analysis" / "front5_position_model_v1" / "front5_model_params.json"


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def distance_category_from_meters(distance: pd.Series, *, extended: bool = False) -> pd.Series:
    d = pd.to_numeric(distance, errors="coerce")
    out = pd.Series("NA", index=distance.index, dtype=object)
    out.loc[d.le(1400)] = "sprint"
    out.loc[d.eq(1600)] = "mile"
    out.loc[d.between(1700, 1900, inclusive="both")] = "middle"
    out.loc[d.between(2000, 2400, inclusive="both")] = "classic" if not extended else "long"
    out.loc[d.gt(2400)] = "long" if not extended else "extended"
    return out


def field_bin_from_size(size: pd.Series) -> pd.Series:
    n = pd.to_numeric(size, errors="coerce")
    out = pd.Series("medium", index=size.index, dtype=object)
    out.loc[n.le(9)] = "small"
    out.loc[n.between(10, 13, inclusive="both")] = "medium"
    out.loc[n.between(14, 16, inclusive="both")] = "large"
    out.loc[n.ge(17)] = "full"
    return out


def class_group_from_text(value: pd.Series) -> pd.Series:
    raw = value.astype("string").fillna("").astype(str)
    out = pd.Series("other", index=value.index, dtype=object)
    out.loc[raw.str.contains("新馬|メイクデビュー", regex=True, na=False)] = "newcomer"
    out.loc[raw.str.contains("未勝利", regex=False, na=False)] = "maiden"
    out.loc[raw.str.contains("1勝|１勝|500万", regex=True, na=False)] = "1win"
    out.loc[raw.str.contains("2勝|２勝|1000万", regex=True, na=False)] = "2win"
    out.loc[raw.str.contains("3勝|３勝|1600万", regex=True, na=False)] = "3win"
    out.loc[raw.str.contains("OP|オープン|L|リステッド|G1|G2|G3|重賞", regex=True, na=False)] = "open"
    return out


def front5_numeric_feature(runners: pd.DataFrame, feature: str) -> tuple[pd.Series, bool]:
    alias_map = {
        "horse_no": ["horse_no", "馬番"],
        "枠番": ["枠番"],
        "頭数": ["field_size", "頭数", "出走頭数"],
        "出走頭数": ["field_size", "出走頭数", "頭数"],
        "popularity": ["live_popularity", "人気", "popularity"],
        "pop_rank_num": ["live_popularity", "人気", "pop_rank_num"],
        "odds": ["live_win_odds", "単勝オッズ", "odds"],
        "odds_num": ["live_win_odds", "単勝オッズ", "odds_num"],
        "market_win_prob_norm": ["market_prob", "market_win_prob_norm"],
        "front_running_tendency_x": ["front_running_tendency"],
        "front_running_tendency_y": ["front_running_tendency"],
        "closing_tendency_x": ["closing_tendency"],
        "closing_tendency_y": ["closing_tendency"],
        "race_front_runner_count_x": ["race_front_runner_count"],
        "race_front_runner_count_y": ["race_front_runner_count"],
        "horse_front_run_rate_past5_feature": ["horse_front_run_rate_past5"],
        "horse_closer_rate_past5_feature": ["horse_closer_rate_past5"],
    }
    names = alias_map.get(feature, [feature])
    for name in names:
        if name in runners.columns:
            return pd.to_numeric(runners[name], errors="coerce"), True
    return pd.Series(np.nan, index=runners.index, dtype=float), False


def front5_categorical_feature(runners: pd.DataFrame, feature: str) -> tuple[pd.Series, bool]:
    if feature == "venue":
        return first_text_series(runners, ["venue", "場所"], "NA"), any(c in runners.columns for c in ["venue", "場所"])
    if feature == "surface":
        return first_text_series(runners, ["surface", "芝・ダ"], "NA"), any(c in runners.columns for c in ["surface", "芝・ダ"])
    if feature == "distance_bin":
        if "distance_bin" in runners.columns:
            return runners["distance_bin"].astype("string").fillna("NA").astype(str), True
        return distance_category_from_meters(first_series(runners, ["距離"], np.nan), extended=True), "距離" in runners.columns
    if feature == "distance_category_eval":
        if "distance_category_eval" in runners.columns:
            return runners["distance_category_eval"].astype("string").fillna("NA").astype(str), True
        if "distance_category" in runners.columns:
            return runners["distance_category"].astype("string").fillna("NA").astype(str), True
        return distance_category_from_meters(first_series(runners, ["距離"], np.nan), extended=False), "距離" in runners.columns
    if feature == "class_group":
        if "class_group" in runners.columns:
            return runners["class_group"].astype("string").fillna("NA").astype(str), True
        return class_group_from_text(first_text_series(runners, ["クラス名", "レース名"], "")), any(c in runners.columns for c in ["クラス名", "レース名"])
    if feature == "field_bin":
        if "field_bin" in runners.columns:
            return runners["field_bin"].astype("string").fillna("NA").astype(str), True
        return field_bin_from_size(first_series(runners, ["field_size", "出走頭数", "頭数"], np.nan)), any(c in runners.columns for c in ["field_size", "出走頭数", "頭数"])
    if feature == "馬場状態":
        return first_text_series(runners, ["runtime_going", "馬場状態"], "NA"), any(c in runners.columns for c in ["runtime_going", "馬場状態"])
    if feature in runners.columns:
        return runners[feature].astype("string").fillna("NA").astype(str), True
    return pd.Series("NA", index=runners.index, dtype=str), False


def apply_front5_model_runtime(runners: pd.DataFrame, model_path: Path = FRONT5_MODEL_PARAMS_PATH) -> pd.DataFrame:
    out = runners.copy()
    out["projected_front5_prob_heuristic"] = pd.to_numeric(out.get("projected_front5_prob"), errors="coerce").fillna(0.5)
    out["front5_model_prob"] = np.nan
    out["front5_model_prob_raw"] = np.nan
    out["front5_model_feature_coverage"] = 0.0
    out["front5_model_blend_weight"] = 0.0
    if not model_path.exists():
        return out
    try:
        params = json.loads(model_path.read_text(encoding="utf-8"))
    except Exception:
        return out

    numeric_features = list(params.get("numeric_features") or [])
    categorical_specs = list(params.get("categorical_specs") or [])
    beta = np.asarray(params.get("beta") or [], dtype=float)
    fill = np.asarray(params.get("numeric_fill") or [], dtype=float)
    mean = np.asarray(params.get("numeric_mean") or [], dtype=float)
    scale = np.asarray(params.get("numeric_scale") or [], dtype=float)
    if len(beta) != 1 + len(numeric_features) + len(categorical_specs):
        return out

    numeric_arrays = []
    available = 0
    for feature in numeric_features:
        series, ok = front5_numeric_feature(out, feature)
        numeric_arrays.append(pd.to_numeric(series, errors="coerce").to_numpy(dtype=float))
        available += int(ok)
    if numeric_arrays:
        x_num = np.vstack(numeric_arrays).T
        if len(fill) != x_num.shape[1] or len(mean) != x_num.shape[1] or len(scale) != x_num.shape[1]:
            return out
        x_num = np.where(np.isnan(x_num), fill, x_num)
        x_num = (x_num - mean) / np.where(scale < 1e-6, 1.0, scale)
    else:
        x_num = np.empty((len(out), 0))

    cat_arrays = []
    cat_available = 0
    cat_cache: dict[str, pd.Series] = {}
    cat_ok: dict[str, bool] = {}
    for spec in categorical_specs:
        col = str(spec.get("column", ""))
        val = str(spec.get("value", ""))
        if col not in cat_cache:
            cat_cache[col], cat_ok[col] = front5_categorical_feature(out, col)
        cat_arrays.append((cat_cache[col].astype(str) == val).to_numpy(dtype=float))
        cat_available += int(cat_ok[col])
    x_cat = np.vstack(cat_arrays).T if cat_arrays else np.empty((len(out), 0))

    x = np.hstack([np.ones((len(out), 1), dtype=float), x_num, x_cat])
    raw = sigmoid_np(x @ beta)
    mids = np.asarray(params.get("calibrator_mids") or [], dtype=float)
    values = np.asarray(params.get("calibrator_values") or [], dtype=float)
    if len(mids) and len(values) == len(mids):
        prob = np.interp(raw, mids, values, left=values[0], right=values[-1])
    else:
        prob = raw
    coverage_den = max(1, len(numeric_features) + len(categorical_specs))
    coverage = (available + cat_available) / coverage_den
    out["front5_model_prob_raw"] = raw
    out["front5_model_prob"] = np.clip(prob, 0.001, 0.999)
    out["front5_model_feature_coverage"] = float(coverage)

    heuristic = out["projected_front5_prob_heuristic"].clip(0.03, 0.97)
    # Conservative adoption: the OOS front5 model is better calibrated, but
    # ticket-level ROI tests still favored the original strongest gate.
    blend = 0.18 if coverage >= 0.65 else 0.0
    out["front5_model_blend_weight"] = blend
    out["front5_model_disagreement_score"] = (out["front5_model_prob"] - heuristic).abs()
    out["projected_front5_prob"] = ((1.0 - blend) * heuristic + blend * out["front5_model_prob"]).clip(0.03, 0.97)
    return out


def add_runtime_queue_shape(runners: pd.DataFrame) -> pd.DataFrame:
    """Describe whether the likely front positions are clear or contested.

    This deliberately does not change the final BUY gate. It gives the live
    dashboard/LINE and shadow analysis a richer race-shape view than a simple
    front-runner count.
    """
    out = runners.copy()
    if "race_id" not in out.columns:
        return out
    idx = out.index
    race_id = out["race_id"].astype(str)
    front5 = pd.to_numeric(out.get("projected_front5_prob"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    front_intent = first_series(out, ["horse_front_run_rate_past5", "front_running_tendency"], 0.0).fillna(0.0).clip(0.0, 1.0)
    prev_c4 = first_series(out, ["prev_corner4_position_rate"], 0.5).fillna(0.5).clip(0.0, 1.0)
    front_rank = pd.to_numeric(out.get("front_rank_score"), errors="coerce").fillna(
        front_intent.groupby(race_id).rank(pct=True)
    ).fillna(0.5).clip(0.0, 1.0)
    lead_score = (
        0.54 * front_intent
        + 0.22 * front_rank
        + 0.14 * (1.0 - prev_c4)
        + 0.10 * front5
    ).clip(0.0, 1.0)
    field = pd.to_numeric(out.get("field_size"), errors="coerce").fillna(0.0)
    field = field.where(field.gt(0), lead_score.groupby(race_id).transform("size").astype(float)).clip(lower=1.0)

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
        return float(vals[-min(3, len(vals)):].mean())

    top1 = lead_score.groupby(race_id).transform("max").fillna(0.0).clip(0.0, 1.0)
    top2 = lead_score.groupby(race_id).transform(second_largest).fillna(0.0).clip(0.0, 1.0)
    top3 = lead_score.groupby(race_id).transform(top3_mean).fillna(0.0).clip(0.0, 1.0)
    gap = (top1 - top2).clip(0.0, 1.0)
    candidate_count = lead_score.ge(0.52).astype(float).groupby(race_id).transform("sum").fillna(0.0)
    near_front_count = lead_score.ge(0.45).astype(float).groupby(race_id).transform("sum").fillna(0.0)
    front_sum = lead_score.groupby(race_id).transform("sum").fillna(0.0)
    front_density = (near_front_count / field).clip(0.0, 1.0)
    existing_pressure = first_series(out, ["race_early_pressure_score", "race_front_runner_ratio"], 0.0).fillna(0.0).clip(0.0, 1.0)

    clarity = sigmoid_np(
        5.2 * (gap - 0.12)
        + 1.5 * (top1 - 0.55)
        - 1.15 * (front_density - 0.25)
        - 0.42 * candidate_count.sub(1.0).clip(lower=0.0)
    )
    duel = sigmoid_np(
        4.4 * (0.13 - gap)
        + 0.80 * candidate_count.sub(1.0)
        + 1.15 * existing_pressure
        + 0.55 * (top3 - 0.50)
        - 0.55 * (top1 - 0.78).clip(lower=0.0)
    )
    no_clear = sigmoid_np(4.0 * (0.48 - top1) + 1.6 * (0.08 - gap))
    projected_load = (
        0.42 * existing_pressure
        + 0.30 * duel
        + 0.18 * (front_sum / field).clip(0.0, 1.0)
        + 0.10 * top3
    ).clip(0.0, 1.0)

    labels = np.select(
        [
            (top1.lt(0.42) | candidate_count.le(0)),
            (top1.ge(0.56) & gap.ge(0.18) & candidate_count.le(2)),
            (candidate_count.ge(3) & gap.le(0.12)),
            (candidate_count.ge(2) & gap.le(0.10)),
        ],
        [
            "no_clear_leader",
            "single_leader_clear",
            "front_duel_dense",
            "matched_speed_duel",
        ],
        default="mixed_queue",
    )

    out["runtime_lead_score"] = lead_score
    out["runtime_lead_top_score"] = top1
    out["runtime_lead_second_score"] = top2
    out["runtime_lead_top_gap"] = gap
    out["runtime_lead_top3_mean_score"] = top3
    out["runtime_lead_candidate_count"] = candidate_count
    out["runtime_lead_near_count"] = near_front_count
    out["runtime_front5_top_prob"] = top1
    out["runtime_front5_second_prob"] = top2
    out["runtime_front5_top_gap"] = gap
    out["runtime_front5_top3_mean_prob"] = top3
    out["runtime_front5_candidate_count"] = candidate_count
    out["runtime_front5_near_count"] = near_front_count
    out["runtime_front5_density_score"] = front_density
    out["runtime_queue_clarity_score"] = pd.Series(clarity, index=idx).clip(0.0, 1.0)
    out["runtime_front_duel_risk_score"] = pd.Series(duel, index=idx).clip(0.0, 1.0)
    out["runtime_no_clear_leader_score"] = pd.Series(no_clear, index=idx).clip(0.0, 1.0)
    out["runtime_projected_front_load_score"] = projected_load
    out["runtime_pace_shape_label"] = pd.Series(labels, index=idx, dtype="string").fillna("mixed_queue").astype(str)
    return out


def apply_first_condition_uncertainty(runners: pd.DataFrame) -> pd.DataFrame:
    out = runners.copy()
    idx = out.index
    same_distance = first_series(out, ["same_distance_category_starts"], 0.0).fillna(0.0)
    same_venue = first_series(out, ["same_venue_starts"], 0.0).fillna(0.0)
    turf_starts = first_series(out, ["horse_turf_starts"], 0.0).fillna(0.0)
    dirt_starts = first_series(out, ["horse_dirt_starts"], 0.0).fillna(0.0)
    career_raw = first_series(out, ["\u30ad\u30e3\u30ea\u30a2"], np.nan)
    career_proxy = (turf_starts + dirt_starts).replace([np.inf, -np.inf], np.nan)
    career = career_raw.fillna(career_proxy).fillna(99.0)
    surface = first_text_series(out, ["\u829d\u30fb\u30c0", "surface"], "")
    is_turf = surface.str.contains("\u829d|\u969c", regex=True, na=False)
    is_dirt = surface.str.contains("\u30c0", regex=False, na=False)
    surface_starts = pd.Series(np.nan, index=idx, dtype=float)
    surface_starts.loc[is_turf] = turf_starts.loc[is_turf]
    surface_starts.loc[is_dirt] = dirt_starts.loc[is_dirt]
    surface_starts = surface_starts.fillna(np.minimum(turf_starts, dirt_starts))
    live_pop = first_series(out, ["live_popularity", "\u4eba\u6c17"], 99.0).fillna(99.0)
    live_win_odds = pd.to_numeric(out.get("live_win_odds"), errors="coerce").fillna(999.0)
    market_prob = pd.to_numeric(out.get("market_prob"), errors="coerce").fillna(0.0)
    prev_time_value = np.maximum(
        norm01(first_series(out, ["prev_class_time_value_score"], 0.0).fillna(0.0), lo=-0.08, hi=0.22),
        norm01(first_series(out, ["prev_race_time_value"], 0.0).fillna(0.0), lo=-0.10, hi=0.25),
    )
    prev_time_margin = norm01(
        first_series(out, ["horse_time_value_plus_margin"], 0.0).fillna(0.0),
        lo=-0.55,
        hi=0.18,
    )
    prev_final3f_rank = first_series(out, ["\u524d\u8d70\u4e0a\u308a3F\u9806", "prev_final3f_rank"], np.nan)
    prev_finish_rank_score = (1.0 - norm01(prev_final3f_rank.fillna(9.0), lo=1.0, hi=9.0)).clip(0.0, 1.0)
    prev_pop = first_series(out, ["\u524d\u8d70\u4eba\u6c17"], 99.0).fillna(99.0)
    prev_market_score = (1.0 - norm01(prev_pop, lo=1.0, hi=9.0)).clip(0.0, 1.0)

    out["first_condition_debutish_flag"] = career.le(1).astype(float)
    out["first_condition_low_career_flag"] = career.le(2).astype(float)
    out["first_distance_category_flag"] = same_distance.le(0).astype(float)
    out["low_distance_category_sample_flag"] = same_distance.le(1).astype(float)
    out["first_venue_flag"] = same_venue.le(0).astype(float)
    out["first_surface_flag"] = surface_starts.le(0).astype(float)
    out["low_surface_sample_flag"] = surface_starts.le(1).astype(float)
    out["first_condition_market_supported_flag"] = (
        live_pop.le(2) | live_win_odds.le(4.0) | market_prob.ge(0.18)
    ).astype(float)
    out["first_condition_market_respected_flag"] = (
        live_pop.le(4) | live_win_odds.le(8.0) | market_prob.ge(0.10)
    ).astype(float)
    out["first_condition_prev_impressive_score"] = (
        out["first_condition_low_career_flag"]
        * out["first_condition_market_respected_flag"]
        * (
            0.46 * prev_time_value
            + 0.28 * prev_time_margin
            + 0.16 * prev_finish_rank_score
            + 0.10 * prev_market_score
        )
    ).clip(0.0, 1.0)
    out["first_condition_impressive_supported_flag"] = (
        out["first_condition_prev_impressive_score"].ge(0.60)
        & out["first_condition_market_supported_flag"].ge(1.0)
    ).astype(float)
    out["first_condition_any_flag"] = out[
        [
            "first_condition_debutish_flag",
            "first_distance_category_flag",
            "first_venue_flag",
            "first_surface_flag",
        ]
    ].max(axis=1)
    out["first_condition_uncertainty_score"] = (
        0.35 * out["first_condition_debutish_flag"]
        + 0.20 * out["first_condition_low_career_flag"]
        + 0.25 * out["first_distance_category_flag"]
        + 0.15 * out["first_venue_flag"]
        + 0.25 * out["first_surface_flag"]
        + 0.10 * out["low_surface_sample_flag"]
    ).clip(0.0, 1.0)
    out["first_condition_net_uncertainty_score"] = (
        out["first_condition_uncertainty_score"] * (1.0 - 0.42 * out["first_condition_prev_impressive_score"])
    ).clip(0.0, 1.0)
    out["first_condition_supported_uncertain_flag"] = (
        out["first_condition_market_supported_flag"].ge(1.0)
        & out["first_condition_net_uncertainty_score"].ge(0.35)
        & out["first_condition_impressive_supported_flag"].lt(1.0)
    ).astype(float)
    out["race_first_condition_supported_uncertain_count"] = out.groupby("race_id")[
        "first_condition_supported_uncertain_flag"
    ].transform("sum")
    out["first_condition_note_tag"] = np.select(
        [
            out["first_condition_debutish_flag"].ge(1.0),
            out["first_surface_flag"].ge(1.0),
            out["first_distance_category_flag"].ge(1.0),
            out["first_venue_flag"].ge(1.0),
            out["first_condition_low_career_flag"].ge(1.0),
        ],
        ["low_career", "first_surface", "first_distance", "first_venue", "thin_history"],
        default="none",
    )
    return out


def enrich_runner_context_from_entry(runners: pd.DataFrame, entry: pd.DataFrame) -> pd.DataFrame:
    if entry.empty:
        return runners
    if "馬番" not in entry.columns:
        return runners
    context = entry.copy()
    context["race_id"] = [row_to_official_race_id(row) for _, row in context.iterrows()]
    context["horse_no"] = pd.to_numeric(context.get("馬番"), errors="coerce").astype("Int64")
    context = context[context["race_id"].ne("") & context["horse_no"].notna()].copy()
    if context.empty:
        return runners
    context["horse_no"] = context["horse_no"].astype(int)
    keep = ["race_id", "horse_no"]
    keep.extend([col for col in GELDING_FEATURE_COLUMNS if col in context.columns])
    keep.extend(
        [
            col
            for col in ["場所", "芝・ダ", "距離", "馬場状態", "レース名", "Ｒ", "発走時刻", "調教師コード", "クラス名"]
            if col in context.columns
        ]
    )
    context = context[keep].drop_duplicates(["race_id", "horse_no"], keep="last")
    return runners.merge(context, on=["race_id", "horse_no"], how="left", suffixes=("", "_entry"))


def _workout_auto_z_bucket(frame: pd.DataFrame, col: str, prefix: str) -> pd.Series:
    idx = frame.index
    z = first_series(frame, [col], np.nan)
    return pd.Series(
        np.select(
            [z.le(-1.00), z.le(-0.50), z.lt(0.50), z.lt(1.00), z.ge(1.00)],
            [
                f"{prefix}_very_fast",
                f"{prefix}_fast",
                f"{prefix}_normal",
                f"{prefix}_slow",
                f"{prefix}_very_slow",
            ],
            default=f"{prefix}_unknown",
        ),
        index=idx,
        dtype="string",
    )


def apply_auto_workout_knowledge(runners: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    out = runners.copy()
    defaults = {
        "workout_auto_knowledge_score": 0.0,
        "workout_auto_candidate_flag": 0.0,
        "workout_auto_shadow_flag": 0.0,
        "workout_auto_runtime_tag": "auto_workout_none",
    }
    for col, value in defaults.items():
        out[col] = value
    if out.empty or candidates.empty or "action" not in candidates.columns or "rule_key" not in candidates.columns:
        return out

    idx = out.index
    trainer = first_series(out, ["調教師コード"], -1).fillna(-1).astype("Int64").astype("string")
    course = compact_text_series(out.get("workout_latest_course_bucket"), idx)
    lap = compact_text_series(out.get("workout_latest_lap_group"), idx)
    pattern = compact_text_series(out.get("workout_latest_pattern_bucket"), idx)
    surface = compact_text_series(out.get("芝・ダ"), idx)
    latest_total_z = _workout_auto_z_bucket(out, "workout_latest_total_vs_trainer_z", "latest_total")
    latest_final1_z = _workout_auto_z_bucket(out, "workout_latest_final1_vs_trainer_z", "latest_final1")
    best_total_z = _workout_auto_z_bucket(out, "workout_best_total_vs_trainer_z", "best_total")
    best_final1_z = _workout_auto_z_bucket(out, "workout_best_final1_vs_trainer_z", "best_final1")
    days = first_series(out, ["workout_latest_days_before_race"], np.nan)
    days_bucket = pd.Series(
        np.select(
            [days.le(3.0), days.le(5.0), days.le(8.0), days.gt(8.0)],
            ["days_0_3", "days_4_5", "days_6_8", "days_9plus"],
            default="days_unknown",
        ),
        index=idx,
        dtype="string",
    )
    count = first_series(out, ["workout_count"], 0.0).fillna(0.0)
    count_bucket = pd.Series(
        np.select(
            [count.le(1.0), count.le(3.0), count.le(7.0), count.gt(7.0)],
            ["count_0_1", "count_2_3", "count_4_7", "count_8plus"],
            default="count_unknown",
        ),
        index=idx,
        dtype="string",
    )

    def flag_key(source: str, name: str) -> pd.Series:
        yes = first_series(out, [source], 0.0).fillna(0.0).ge(1.0)
        return pd.Series(np.where(yes, f"{name}_yes", f"{name}_no"), index=idx, dtype="string")

    family_keys = {
        "lap": trainer + "|" + lap,
        "course_lap": trainer + "|" + course + "|" + lap,
        "surface_lap": trainer + "|" + surface + "|" + lap,
        "pattern": trainer + "|" + pattern,
        "course_pattern": trainer + "|" + course + "|" + pattern,
        "surface_course_lap": trainer + "|" + surface + "|" + course + "|" + lap,
        "latest_total_z": trainer + "|" + latest_total_z,
        "latest_final1_z": trainer + "|" + latest_final1_z,
        "best_total_z": trainer + "|" + best_total_z,
        "best_final1_z": trainer + "|" + best_final1_z,
        "course_total_z": trainer + "|" + course + "|" + latest_total_z,
        "course_final1_z": trainer + "|" + course + "|" + latest_final1_z,
        "days_lap": trainer + "|" + days_bucket + "|" + lap,
        "count_lap": trainer + "|" + count_bucket + "|" + lap,
        "flag_a1": trainer + "|" + flag_key("workout_a1_flag", "a1"),
        "flag_a2": trainer + "|" + flag_key("workout_a2_flag", "a2"),
        "flag_a3": trainer + "|" + flag_key("workout_a3_flag", "a3"),
        "flag_b1": trainer + "|" + flag_key("workout_b1_flag", "b1"),
        "flag_b2": trainer + "|" + flag_key("workout_b2_flag", "b2"),
        "flag_b3": trainer + "|" + flag_key("workout_b3_flag", "b3"),
        "flag_fast_final": trainer + "|" + flag_key("workout_fast_final_flag", "fast_final"),
        "flag_strong_finish": trainer + "|" + flag_key("workout_strong_finish_flag", "strong_finish"),
        "flag_partner_win": trainer + "|" + flag_key("workout_partner_win_flag", "partner_win"),
    }
    candidate_flag = pd.Series(False, index=idx)
    shadow_flag = pd.Series(False, index=idx)
    for family, keys in family_keys.items():
        family_candidates = candidates[candidates["family"].astype("string").eq(family)]
        if family_candidates.empty:
            continue
        candidate_rules = set(
            family_candidates.loc[
                family_candidates["action"].astype("string").eq("candidate_rule"), "rule_key"
            ].astype("string")
        )
        shadow_rules = set(
            family_candidates.loc[
                family_candidates["action"].astype("string").eq("shadow_only"), "rule_key"
            ].astype("string")
        )
        candidate_flag |= keys.isin(candidate_rules)
        shadow_flag |= keys.isin(shadow_rules)
    shadow_flag = shadow_flag & ~candidate_flag
    score = np.select([candidate_flag, shadow_flag], [1.0, 0.45], default=0.0)
    out["workout_auto_candidate_flag"] = candidate_flag.astype(float)
    out["workout_auto_shadow_flag"] = shadow_flag.astype(float)
    out["workout_auto_knowledge_score"] = score
    out["workout_auto_runtime_tag"] = np.select(
        [candidate_flag, shadow_flag],
        ["auto_workout_candidate", "auto_workout_shadow"],
        default="auto_workout_none",
    )
    if "workout_runtime_score" in out.columns:
        out["workout_runtime_score"] = (
            pd.to_numeric(out["workout_runtime_score"], errors="coerce").fillna(0.0)
            + 0.10 * out["workout_auto_knowledge_score"]
        ).clip(0.0, 1.0)
    return out


def merge_live_odds_timeline_features(runners: pd.DataFrame, odds_features: pd.DataFrame) -> pd.DataFrame:
    if runners.empty or odds_features.empty:
        out = runners.copy()
    else:
        features = odds_features.copy()
        if not {"race_id", "horse_number"}.issubset(features.columns):
            out = runners.copy()
        else:
            features["race_id"] = features["race_id"].astype(str)
            features["horse_no"] = pd.to_numeric(features["horse_number"], errors="coerce").astype("Int64")
            features = features[features["horse_no"].notna()].copy()
            features["horse_no"] = features["horse_no"].astype(int)
            keep = [
                "race_id",
                "horse_no",
                "odds_snapshot_count",
                "odds_valid_snapshot_count",
                "odds_first_snapshot_at",
                "odds_latest_snapshot_at",
                "odds_elapsed_minutes",
                "odds_win_change_from_first_pct",
                "odds_win_change_from_prev_pct",
                "odds_drop_from_first_pct",
                "odds_drop_from_prev_pct",
                "odds_drop_velocity_per_min",
                "odds_steam_flag",
                "odds_drift_flag",
            ]
            keep = [col for col in keep if col in features.columns]
            features = features[keep].drop_duplicates(["race_id", "horse_no"], keep="last")
            out = runners.merge(features, on=["race_id", "horse_no"], how="left")

    idx = out.index
    out["late_odds_drop_rate"] = series_num(out.get("odds_drop_from_prev_pct"), idx, 0.0).fillna(0.0).clip(0.0, 3.0)
    out["late_odds_drift_rate"] = series_num(out.get("odds_win_change_from_prev_pct"), idx, 0.0).fillna(0.0).clip(0.0, 3.0)
    out["session_odds_drop_rate"] = series_num(out.get("odds_drop_from_first_pct"), idx, 0.0).fillna(0.0).clip(0.0, 5.0)
    out["session_odds_drift_rate"] = series_num(out.get("odds_win_change_from_first_pct"), idx, 0.0).fillna(0.0).clip(0.0, 5.0)
    out["odds_timeline_valid_snapshot_count"] = pd.to_numeric(
        out.get("odds_valid_snapshot_count", pd.Series(0, index=idx)), errors="coerce"
    ).fillna(0.0)
    out["odds_timeline_ready"] = out["odds_timeline_valid_snapshot_count"].ge(2).astype(float)
    out["odds_steam_flag"] = pd.to_numeric(out.get("odds_steam_flag", pd.Series(0, index=idx)), errors="coerce").fillna(0.0)
    out["odds_drift_flag"] = pd.to_numeric(out.get("odds_drift_flag", pd.Series(0, index=idx)), errors="coerce").fillna(0.0)
    return out


def prepare_runners(
    prediction: pd.DataFrame,
    single: pd.DataFrame,
    track: pd.DataFrame | None = None,
    entry: pd.DataFrame | None = None,
    odds_timeline_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    runners = prediction.copy()
    missing_required_before_defaults = [col for col in STRONGEST_REQUIRED_COLUMNS if col not in runners.columns]
    runners["race_id"] = [row_to_official_race_id(row) for _, row in runners.iterrows()]
    runners["horse_no"] = pd.to_numeric(runners.get("馬番"), errors="coerce").astype("Int64")
    runners = runners[runners["race_id"].ne("") & runners["horse_no"].notna()].copy()
    runners["horse_no"] = runners["horse_no"].astype(int)
    single_small = single.copy()
    single_small["race_id"] = single_small["race_id"].astype(str)
    single_small["horse_no"] = pd.to_numeric(single_small["horse_no"], errors="coerce").astype("Int64")
    single_small = single_small[single_small["horse_no"].notna()].copy()
    single_small["horse_no"] = single_small["horse_no"].astype(int)
    runners = runners.merge(single_small, on=["race_id", "horse_no"], how="inner")
    runners = enrich_runner_context_from_entry(runners, entry if entry is not None else pd.DataFrame())
    runners = add_race_post_time_flags(runners)
    runners = apply_runtime_track_conditions(runners, track if track is not None else pd.DataFrame())
    runners = merge_live_odds_timeline_features(
        runners, odds_timeline_features if odds_timeline_features is not None else pd.DataFrame()
    )

    runners["ai_score_num"] = pd.to_numeric(runners["ai_score"], errors="coerce").fillna(0.0)
    runners["ai_rank_num"] = pd.to_numeric(runners["ai_rank"], errors="coerce").fillna(999).astype(int)
    runners["field_size"] = runners.groupby("race_id")["horse_no"].transform("size").astype(float)
    runners["ai_rank_score"] = ((runners["field_size"] + 1.0 - runners["ai_rank_num"]) / runners["field_size"]).clip(0.0, 1.0)
    runners["ai_prob"] = softmax_by_race(runners, "ai_score_num")
    runners["live_win_odds"] = clean_odds_series(runners["live_win_odds"])
    implied = 1.0 / runners["live_win_odds"].replace(0, np.nan)
    runners["market_prob"] = implied.groupby(runners["race_id"]).transform(lambda s: s / s.sum() if s.sum() else s).fillna(0.0)
    top_market = (
        runners.sort_values(["race_id", "market_prob", "live_win_odds"], ascending=[True, False, True])
        .drop_duplicates("race_id", keep="first")[
            ["race_id", "horse_no", "market_prob", "live_win_odds", "ai_rank_num"]
        ]
        .rename(
            columns={
                "horse_no": "race_top_market_horse_no",
                "market_prob": "race_top_market_prob",
                "live_win_odds": "race_top_market_odds",
                "ai_rank_num": "race_top_market_ai_rank_num",
            }
        )
    )
    runners = runners.merge(top_market, on="race_id", how="left")
    overlay_raw = runners["ai_prob"] - runners["market_prob"]
    runners["market_overlay_score"] = (0.50 + 4.0 * overlay_raw).clip(0.0, 1.0)
    runners = apply_first_condition_uncertainty(runners)
    front = first_series(runners, ["horse_front_run_rate_past5", "front_running_tendency"], 0.0).fillna(0.0)
    stalker = first_series(runners, ["horse_stalker_rate_past5"], 0.0).fillna(0.0)
    prev_c4 = first_series(runners, ["prev_corner4_position_rate"], 0.5).fillna(0.5).clip(0.0, 1.0)
    pressure = first_series(runners, ["race_early_pressure_score", "race_front_runner_ratio"], 0.0).fillna(0.0)
    slow_risk = first_series(runners, ["race_slow_pace_risk"], 0.0).fillna(0.0).clip(0.0, 1.0)
    pace_fit = first_series(runners, ["pace_fit_score"], 0.0).fillna(0.0)
    draw_fit = first_series(runners, ["draw_pace_fit_score"], 0.0).fillna(0.0)
    front_adv = first_series(runners, ["front_advantage_score"], 0.0).fillna(0.0)
    workout_grade = first_series(runners, ["workout_knowledge_grade_score"], 2.0).fillna(2.0)
    workout_load = first_series(runners, ["workout_load_density_score"], 0.0).fillna(0.0)
    runners["workout_knowledge_grade_score"] = workout_grade
    runners["workout_load_density_score"] = workout_load
    frame = first_series(runners, ["枠番"], np.nan)
    frame_inner = (1.0 - (frame - 1.0) / 7.0).clip(0.0, 1.0).fillna(0.5)
    runners["front_rank_score"] = front.groupby(runners["race_id"]).rank(pct=True).fillna(0.5)
    expected_fast = runners.get("expected_pace", pd.Series("", index=runners.index)).astype(str).eq("fast").astype(float)
    front_logit = (
        -1.20
        + 2.35 * front
        + 0.80 * stalker
        + 1.05 * (1.0 - prev_c4)
        + 0.40 * frame_inner
        + 0.50 * runners["front_rank_score"]
        + 0.32 * slow_risk
        + 0.22 * norm01(pace_fit)
        + 0.16 * norm01(draw_fit)
        + 0.12 * norm01(front_adv)
        - 0.62 * pressure
        - 0.32 * expected_fast
        - 0.12 * (runners["field_size"].ge(16)).astype(float)
    )
    runners["projected_front5_prob"] = (1.0 / (1.0 + np.exp(-np.clip(front_logit, -30, 30)))).clip(0.03, 0.97)
    runners = apply_front5_model_runtime(runners)
    runners = add_runtime_queue_shape(runners)
    runners = apply_race_quality_v2_runtime(runners)
    runners["pace_fit_runtime_score"] = norm01(pace_fit)
    runners["draw_pace_fit_runtime_score"] = norm01(draw_fit)
    runners["workout_runtime_score"] = (
        0.62 * norm01(workout_grade, lo=0.0, hi=5.0)
        + 0.38 * norm01(workout_load, lo=0.0, hi=1.4)
    ).clip(0.0, 1.0)
    runners["fast_clock_runtime_score"] = score_fast_clock_runtime(runners)
    corner_shape_fit, corner_shape_low_sample = score_corner_shape_runtime(runners)
    runners["corner_shape_fit_runtime_score"] = corner_shape_fit
    runners["corner_shape_low_sample_flag"] = corner_shape_low_sample
    prev_early_move = first_series(runners, ["prev_early_move", "horse_early_move_avg_past5"], 0.0).fillna(0.0)
    early_move_avg = first_series(runners, ["horse_early_move_avg_past5"], 0.0).fillna(0.0)
    runners["early_start_shadow_score"] = (
        0.55 * front.clip(0.0, 1.0)
        + 0.25 * first_series(runners, ["front_running_tendency"], 0.0).fillna(0.0).clip(0.0, 1.0)
        + 0.20 * (1.0 - prev_c4)
    ).clip(0.0, 1.0)
    runners["second_leg_shadow_score"] = (
        0.55 * norm01(early_move_avg, lo=-2.0, hi=3.0)
        + 0.25 * norm01(prev_early_move, lo=-2.0, hi=3.0)
        + 0.20 * runners["pace_fit_runtime_score"]
    ).clip(0.0, 1.0)
    gelding_risk = first_series(runners, ["gelding_risk_score"], 0.0).fillna(0.0).clip(0.0, 1.0)
    gelding_value = first_series(runners, ["gelding_value_score"], 0.0).fillna(0.0).clip(0.0, 1.0)
    local_risk = first_series(runners, ["prev_local_transition_risk_score"], 0.0).fillna(0.0).clip(0.0, 1.0)
    local_value = first_series(runners, ["prev_local_transition_value_score"], 0.0).fillna(0.0).clip(0.0, 1.0)
    runners["gelding_risk_score"] = gelding_risk
    runners["gelding_value_score"] = gelding_value
    runners["local_transition_risk_score"] = local_risk
    runners["local_transition_value_score"] = local_value
    for col in [
        "prev_local_race_flag",
        "prev_local_good_run_score",
        "prev_local_transition_risk_score",
        "prev_local_transition_value_score",
    ]:
        if col not in runners.columns:
            runners[col] = 0.0
    missing_required = [
        col for col in missing_required_before_defaults if col not in DEFAULTABLE_REQUIRED_COLUMNS
    ]
    defaulted_required = [
        col
        for col in missing_required_before_defaults
        if col in DEFAULTABLE_REQUIRED_COLUMNS and col in runners.columns
    ]
    runners["strongest_feature_parity_ready"] = 0.0 if missing_required else 1.0
    runners["strongest_missing_required_features"] = "|".join(missing_required)
    runners["strongest_defaulted_required_features"] = "|".join(defaulted_required)
    confidence = pd.to_numeric(runners.get("ai_confidence_score"), errors="coerce").fillna(0.20)
    runners["danger_popular_score"] = (
        0.55 * (runners["market_prob"] - runners["ai_prob"]).clip(lower=0.0) / 0.18
        + 0.25 * (1.0 - confidence)
        + 0.13 * (1.0 - runners["ai_rank_score"])
        + 0.05 * (1.0 - runners["pace_fit_runtime_score"])
        + 0.02 * (1.0 - runners["workout_runtime_score"])
        + 0.08 * runners["gelding_risk_score"]
        + 0.04 * runners["local_transition_risk_score"]
    ).clip(0.0, 1.0)
    live_popularity = first_series(runners, ["live_popularity", "人気"], 99.0).fillna(99.0)
    live_odds_support = norm01(1.0 / runners["live_win_odds"].replace(0, np.nan), lo=0.02, hi=0.35)
    runners["market_support_score"] = (
        0.45 * norm01(runners["market_prob"], lo=0.02, hi=0.24)
        + 0.35 * (1.0 - ((live_popularity - 1.0) / 7.0).clip(0.0, 1.0))
        + 0.20 * live_odds_support
    ).clip(0.0, 1.0)
    runners["danger_popular_inclusion_score"] = (
        runners["danger_popular_score"] * runners["market_support_score"]
    ).clip(0.0, 1.0)
    race_market_top = runners.groupby("race_id")["market_prob"].transform("max").fillna(0.0)
    race_conf_top = confidence.groupby(runners["race_id"]).transform("max").fillna(0.20)
    race_gap_top = first_series(runners, ["ai_score_gap_to_second"], 0.0).groupby(runners["race_id"]).transform("max").fillna(0.0)
    race_first_condition_count = first_series(
        runners, ["race_first_condition_supported_uncertain_count"], 0.0
    ).fillna(0.0)
    collapse_risk = first_series(runners, ["race_pace_collapse_risk"], 0.0).fillna(0.0).clip(0.0, 1.0)
    pressure_uncertainty = (pressure - 0.42).abs().clip(0.0, 0.42) / 0.42
    runners["race_difficulty_score"] = (
        0.22 * (1.0 - race_conf_top.clip(0.0, 1.0))
        + 0.17 * norm01(runners["field_size"], lo=9.0, hi=18.0)
        + 0.16 * (1.0 - norm01(race_gap_top, lo=0.0, hi=0.10))
        + 0.15 * pressure_uncertainty
        + 0.12 * collapse_risk
        + 0.10 * (1.0 - norm01(race_market_top, lo=0.08, hi=0.35))
        + 0.08 * (race_first_condition_count / 3.0).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    runners["quinella_score"] = (
        0.34 * runners["ai_rank_score"]
        + 0.20 * norm01(runners["ai_prob"], lo=0.02, hi=0.25)
        + 0.16 * runners["market_overlay_score"]
        + 0.11 * runners["projected_front5_prob"]
        + 0.08 * runners["pace_fit_runtime_score"]
        + 0.06 * runners["workout_runtime_score"]
        + 0.05 * (1.0 - runners["danger_popular_score"])
        + 0.03 * runners["gelding_value_score"]
        + 0.025 * runners["local_transition_value_score"]
        - 0.08 * runners["gelding_risk_score"]
        - 0.04 * runners["local_transition_risk_score"]
    ).clip(0.0, 1.0)
    runners["place_score"] = (
        0.45 * runners["ai_rank_score"]
        + 0.25 * norm01(runners["ai_prob"], lo=0.02, hi=0.25)
        + 0.18 * (1.0 - runners["danger_popular_score"])
        + 0.07 * runners["projected_front5_prob"]
        + 0.05 * runners["pace_fit_runtime_score"]
        + 0.02 * runners["gelding_value_score"]
        + 0.02 * runners["local_transition_value_score"]
        - 0.06 * runners["gelding_risk_score"]
        - 0.035 * runners["local_transition_risk_score"]
    ).clip(0.0, 1.0)
    return runners


def build_pair_candidates(runners: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
    pair = pair[pair["ticket_type"].astype(str).eq("umaren")].copy()
    pair["race_id"] = pair["race_id"].astype(str)
    for col in ["a_no", "b_no"]:
        pair[col] = pd.to_numeric(pair[col], errors="coerce").astype("Int64")
    pair = pair[pair["a_no"].notna() & pair["b_no"].notna()].copy()
    pair["a_no"] = pair["a_no"].astype(int)
    pair["b_no"] = pair["b_no"].astype(int)

    left_cols = [
        "race_id",
        "horse_no",
        "馬名",
        "ai_rank_num",
        "ai_prob",
        "ai_rank_score",
        "market_overlay_score",
        "projected_front5_prob",
        "projected_front5_prob_heuristic",
        "front5_model_prob",
        "front5_model_prob_raw",
        "front5_model_feature_coverage",
        "front5_model_blend_weight",
        "front5_model_disagreement_score",
        "danger_popular_score",
        "market_support_score",
        "danger_popular_inclusion_score",
        "race_difficulty_score",
        "quinella_score",
        "place_score",
        "live_win_odds",
        "race_top_market_horse_no",
        "race_top_market_prob",
        "race_top_market_odds",
        "race_top_market_ai_rank_num",
        "late_odds_drop_rate",
        "late_odds_drift_rate",
        "session_odds_drop_rate",
        "session_odds_drift_rate",
        "odds_timeline_ready",
        "odds_steam_flag",
        "odds_drift_flag",
        "race_early_pressure_score",
        "race_front_runner_count",
        "runtime_lead_score",
        "runtime_lead_top_score",
        "runtime_lead_second_score",
        "runtime_lead_top_gap",
        "runtime_lead_top3_mean_score",
        "runtime_lead_candidate_count",
        "runtime_lead_near_count",
        "runtime_front5_top_prob",
        "runtime_front5_second_prob",
        "runtime_front5_top_gap",
        "runtime_front5_top3_mean_prob",
        "runtime_front5_candidate_count",
        "runtime_front5_near_count",
        "runtime_front5_density_score",
        "runtime_queue_clarity_score",
        "runtime_front_duel_risk_score",
        "runtime_no_clear_leader_score",
        "runtime_projected_front_load_score",
        "runtime_pace_shape_label",
        "ai_confidence_score",
        "ai_score_gap_to_second",
        "field_size",
        "斤量",
        "前走斤量",
        "weight_diff",
        "race_weight_light_rank_score",
        "発走時刻",
        "race_post_at",
        "race_minutes_to_post",
        "race_started_flag",
        "runtime_decision_generated_at",
        "場所",
        "芝・ダ",
        "距離",
        "前距離",
        "distance_diff",
        "previous_distance_category",
        "クラス名",
        "馬場状態",
        "runtime_going",
        "runtime_going_class",
        "runtime_track_condition_available",
        "runtime_soft_heavy_flag",
        "runtime_hakodate_flag",
        "expected_pace",
        "horse_front_run_rate_past5",
        "horse_closer_rate_past5",
        "closing_tendency",
        "race_pace_collapse_risk",
        "race_slow_pace_risk",
        "same_day_bias_fit_score",
        "pace_fit_score",
        "front_advantage_score",
        "closer_advantage_score",
        "positioning_advantage_score",
        "draw_pace_fit_score",
        "pace_fit_runtime_score",
        "draw_pace_fit_runtime_score",
        "workout_knowledge_grade_score",
        "workout_load_density_score",
        "workout_runtime_score",
        "workout_auto_knowledge_score",
        "workout_auto_candidate_flag",
        "workout_auto_shadow_flag",
        "workout_auto_runtime_tag",
        "fast_clock_runtime_score",
        "time_value_relative_rank_score",
        "recency_weighted_time_score",
        "best_time_reproducibility",
        "time_score_consistency",
        "condition_matched_time_score",
        "today_fast_clock_likelihood",
        "fast_clock_x_today_likelihood",
        "pace_tracking_score",
        "late_speed_value",
        "sustained_speed_score",
        "time_refinement_composite",
        "hist_condition_scope",
        "hist_condition_sample_count",
        "hist_avg_winning_time_sec",
        "hist_avg_front3f_sec",
        "hist_avg_last3f_sec",
        "hist_avg_1000m_sec",
        "hist_avg_rpci",
        "hist_avg_pci3",
        "race_quality_expected_front3f_sec",
        "race_quality_front3f_delta_sec",
        "race_quality_expected_rpci",
        "race_quality_rpci_delta",
        "race_quality_fast_need_score",
        "race_quality_slow_need_score",
        "race_quality_fit_score",
        "race_quality_label",
        "race_quality_context_ready",
        "race_quality_v2_predicted_lap_mode",
        "race_quality_v2_confidence",
        "race_quality_v2_margin",
        "race_quality_v2_prob_fast",
        "race_quality_v2_prob_slow",
        "race_quality_v2_prob_instant",
        "race_quality_v2_prob_sustain",
        "race_quality_v2_runtime_tag",
        "past1_lap_fast_success",
        "past1_lap_slow_success",
        "past1_lap_instant_success",
        "past1_lap_sustain_success",
        "past1_lap_long_spurt_success",
        "past1_lap_rpci",
        "past2_lap_fast_success",
        "past2_lap_slow_success",
        "past2_lap_instant_success",
        "past2_lap_sustain_success",
        "past2_lap_long_spurt_success",
        "past2_lap_rpci",
        "past3_lap_fast_success",
        "past3_lap_slow_success",
        "past3_lap_instant_success",
        "past3_lap_sustain_success",
        "past3_lap_long_spurt_success",
        "past3_lap_rpci",
        "horse_fast_lap_score_past5",
        "horse_slow_lap_score_past5",
        "horse_instant_lap_score_past5",
        "horse_sustain_lap_score_past5",
        "horse_long_spurt_lap_score_past5",
        "lap_pace_versatility_score",
        "lap_aptitude_reliability_score",
        "corner_shape_fit_runtime_score",
        "corner_shape_low_sample_flag",
        "early_start_shadow_score",
        "second_leg_shadow_score",
        "gelding_phase",
        "gelding_start_no_since_transition",
        "gelding_risk_score",
        "gelding_value_score",
        "gelding_context_note",
        "known_gelding_debut_flag",
        "known_gelding_second_start_flag",
        "gelding_debut_unpopular_flag",
        "gelding_debut_surface_switch_flag",
        "gelding_second_shorten_flag",
        "prev_local_race_flag",
        "prev_local_good_run_score",
        "prev_local_transition_risk_score",
        "prev_local_transition_value_score",
        "local_transition_risk_score",
        "local_transition_value_score",
        "confirmed_member_level_adjusted_score",
        "prev_race_member_level",
        "past3_max_race_member_level",
        "prev_confirmed_opponent_good_run_score",
        "prev_class_time_value_score",
        "rotation_class_up_flag",
        "rotation_class_down_flag",
        "rotation_same_class_flag",
        "basic_ability_history_ready",
        "recent_weighted_score_3",
        "recent_score_slope_3",
        "recent_score_jump_vs_mean",
        "recent_score_std_3",
        "ability_stability_score_3",
        "ability_ceiling_score_5",
        "ability_floor_score_5",
        "recent_score_count_3",
        "recent_score_count_5",
        "prev_corner4_front_rate",
        "prev_final3f_excellence_rate",
        "prev_stretch_gain_sec",
        "prev_late_improvement_score",
        "prev_market_underestimated_score",
        "prev_market_overestimated_risk",
        "weight_burden_ratio_prev_body",
        "weight_burden_ratio_change",
        "first_condition_debutish_flag",
        "first_condition_low_career_flag",
        "first_distance_category_flag",
        "low_distance_category_sample_flag",
        "first_venue_flag",
        "first_surface_flag",
        "low_surface_sample_flag",
        "first_condition_market_supported_flag",
        "first_condition_market_respected_flag",
        "first_condition_prev_impressive_score",
        "first_condition_impressive_supported_flag",
        "first_condition_any_flag",
        "first_condition_uncertainty_score",
        "first_condition_net_uncertainty_score",
        "first_condition_supported_uncertain_flag",
        "race_first_condition_supported_uncertain_count",
        "first_condition_note_tag",
        "strongest_feature_parity_ready",
        "strongest_missing_required_features",
        "strongest_defaulted_required_features",
    ]
    text_defaults = {
        "馬名",
        "場所",
        "芝・ダ",
        "クラス名",
        "馬場状態",
        "runtime_going",
        "runtime_going_class",
        "hist_condition_scope",
        "race_quality_label",
        "race_quality_v2_predicted_lap_mode",
        "race_quality_v2_runtime_tag",
        "expected_pace",
        "発走時刻",
        "race_post_at",
        "runtime_decision_generated_at",
        "gelding_phase",
        "gelding_context_note",
        "first_condition_note_tag",
        "workout_auto_runtime_tag",
        "strongest_missing_required_features",
    }
    nan_defaults = {
        "basic_ability_history_ready",
        "recent_weighted_score_3",
        "recent_score_slope_3",
        "recent_score_jump_vs_mean",
        "recent_score_std_3",
        "ability_stability_score_3",
        "ability_ceiling_score_5",
        "ability_floor_score_5",
        "recent_score_count_3",
        "recent_score_count_5",
        "prev_corner4_front_rate",
        "prev_final3f_excellence_rate",
        "prev_stretch_gain_sec",
        "prev_late_improvement_score",
        "prev_market_underestimated_score",
        "prev_market_overestimated_risk",
        "weight_burden_ratio_prev_body",
        "weight_burden_ratio_change",
        "time_value_relative_rank_score",
        "recency_weighted_time_score",
        "best_time_reproducibility",
        "time_score_consistency",
        "距離",
        "condition_matched_time_score",
        "today_fast_clock_likelihood",
        "fast_clock_x_today_likelihood",
        "pace_tracking_score",
        "late_speed_value",
        "sustained_speed_score",
        "time_refinement_composite",
    }
    for col in left_cols:
        if col not in runners.columns:
            runners[col] = "" if col in text_defaults else (np.nan if col in nan_defaults else 0.0)
    a = runners[left_cols].rename(columns={c: f"anchor_{c}" for c in left_cols if c not in {"race_id"}})
    b = runners[left_cols].rename(columns={c: f"partner_{c}" for c in left_cols if c not in {"race_id"}})
    merged = pair.merge(a, left_on=["race_id", "a_no"], right_on=["race_id", "anchor_horse_no"], how="inner")
    merged = merged.merge(b, left_on=["race_id", "b_no"], right_on=["race_id", "partner_horse_no"], how="inner")
    if merged.empty:
        return merged

    merged["race_started_flag"] = np.maximum(
        pd.to_numeric(merged.get("anchor_race_started_flag"), errors="coerce").fillna(0.0),
        pd.to_numeric(merged.get("partner_race_started_flag"), errors="coerce").fillna(0.0),
    )
    merged["race_minutes_to_post"] = np.minimum(
        pd.to_numeric(merged.get("anchor_race_minutes_to_post"), errors="coerce").fillna(np.inf),
        pd.to_numeric(merged.get("partner_race_minutes_to_post"), errors="coerce").fillna(np.inf),
    ).replace(np.inf, np.nan)
    merged["race_post_at"] = first_text_series(merged, ["anchor_race_post_at", "partner_race_post_at"], "")
    merged["runtime_decision_generated_at"] = first_text_series(
        merged, ["anchor_runtime_decision_generated_at", "partner_runtime_decision_generated_at"], ""
    )

    # Keep the better AI-ranked horse as anchor for display and scoring consistency.
    swap = merged["partner_ai_rank_num"] < merged["anchor_ai_rank_num"]
    for col in [c.replace("anchor_", "") for c in merged.columns if c.startswith("anchor_")]:
        ac = f"anchor_{col}"
        pc = f"partner_{col}"
        if ac in merged.columns and pc in merged.columns:
            tmp = merged.loc[swap, ac].copy()
            merged.loc[swap, ac] = merged.loc[swap, pc]
            merged.loc[swap, pc] = tmp

    place_joint = np.sqrt((merged["anchor_place_score"] * merged["partner_place_score"]).clip(0.0, 1.0))
    front_bonus = np.maximum(merged["anchor_projected_front5_prob"], merged["partner_projected_front5_prob"])
    overlay = np.maximum(merged["anchor_market_overlay_score"], merged["partner_market_overlay_score"])
    danger = np.maximum(merged["anchor_danger_popular_score"], merged["partner_danger_popular_score"])
    danger_in_pair = np.maximum(
        pd.to_numeric(merged["anchor_danger_popular_inclusion_score"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["partner_danger_popular_inclusion_score"], errors="coerce").fillna(0.0),
    )
    race_difficulty = np.maximum(
        pd.to_numeric(merged["anchor_race_difficulty_score"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["partner_race_difficulty_score"], errors="coerce").fillna(0.0),
    )
    pace_pair = np.maximum(merged["anchor_pace_fit_runtime_score"], merged["partner_pace_fit_runtime_score"])
    anchor_front_prob = pd.to_numeric(merged["anchor_projected_front5_prob"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    partner_front_prob = pd.to_numeric(merged["partner_projected_front5_prob"], errors="coerce").fillna(0.5).clip(0.0, 1.0)
    slow_pace_pair = np.maximum(
        pd.to_numeric(merged["anchor_race_slow_pace_risk"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["partner_race_slow_pace_risk"], errors="coerce").fillna(0.0),
    ).clip(0.0, 1.0)
    collapse_pair = np.maximum(
        pd.to_numeric(merged["anchor_race_pace_collapse_risk"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["partner_race_pace_collapse_risk"], errors="coerce").fillna(0.0),
    ).clip(0.0, 1.0)
    pressure_pair = np.maximum(
        pd.to_numeric(merged["anchor_race_early_pressure_score"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["partner_race_early_pressure_score"], errors="coerce").fillna(0.0),
    ).clip(0.0, 1.0)
    queue_clarity_pair = np.maximum(
        pd.to_numeric(merged.get("anchor_runtime_queue_clarity_score"), errors="coerce").fillna(0.0),
        pd.to_numeric(merged.get("partner_runtime_queue_clarity_score"), errors="coerce").fillna(0.0),
    ).clip(0.0, 1.0)
    front_duel_risk_pair = np.maximum(
        pd.to_numeric(merged.get("anchor_runtime_front_duel_risk_score"), errors="coerce").fillna(0.0),
        pd.to_numeric(merged.get("partner_runtime_front_duel_risk_score"), errors="coerce").fillna(0.0),
    ).clip(0.0, 1.0)
    no_clear_leader_pair = np.maximum(
        pd.to_numeric(merged.get("anchor_runtime_no_clear_leader_score"), errors="coerce").fillna(0.0),
        pd.to_numeric(merged.get("partner_runtime_no_clear_leader_score"), errors="coerce").fillna(0.0),
    ).clip(0.0, 1.0)
    projected_front_load_pair = np.maximum(
        pd.to_numeric(merged.get("anchor_runtime_projected_front_load_score"), errors="coerce").fillna(0.0),
        pd.to_numeric(merged.get("partner_runtime_projected_front_load_score"), errors="coerce").fillna(0.0),
    ).clip(0.0, 1.0)
    front5_top_gap_pair = np.maximum(
        pd.to_numeric(merged.get("anchor_runtime_front5_top_gap"), errors="coerce").fillna(0.0),
        pd.to_numeric(merged.get("partner_runtime_front5_top_gap"), errors="coerce").fillna(0.0),
    ).clip(0.0, 1.0)
    front5_candidate_count_pair = np.maximum(
        pd.to_numeric(merged.get("anchor_runtime_front5_candidate_count"), errors="coerce").fillna(0.0),
        pd.to_numeric(merged.get("partner_runtime_front5_candidate_count"), errors="coerce").fillna(0.0),
    )
    lead_top_score_pair = np.maximum(
        pd.to_numeric(merged.get("anchor_runtime_lead_top_score"), errors="coerce").fillna(0.0),
        pd.to_numeric(merged.get("partner_runtime_lead_top_score"), errors="coerce").fillna(0.0),
    ).clip(0.0, 1.0)
    lead_top_gap_pair = np.maximum(
        pd.to_numeric(merged.get("anchor_runtime_lead_top_gap"), errors="coerce").fillna(0.0),
        pd.to_numeric(merged.get("partner_runtime_lead_top_gap"), errors="coerce").fillna(0.0),
    ).clip(0.0, 1.0)
    lead_candidate_count_pair = np.maximum(
        pd.to_numeric(merged.get("anchor_runtime_lead_candidate_count"), errors="coerce").fillna(0.0),
        pd.to_numeric(merged.get("partner_runtime_lead_candidate_count"), errors="coerce").fillna(0.0),
    )
    pace_shape_pair_label = first_text_series(
        merged, ["anchor_runtime_pace_shape_label", "partner_runtime_pace_shape_label"], "mixed_queue"
    )
    front_adv_pair = np.maximum(
        pd.to_numeric(merged["anchor_front_advantage_score"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["partner_front_advantage_score"], errors="coerce").fillna(0.0),
    )
    draw_pace_pair = np.maximum(
        pd.to_numeric(merged["anchor_draw_pace_fit_runtime_score"], errors="coerce").fillna(0.5),
        pd.to_numeric(merged["partner_draw_pace_fit_runtime_score"], errors="coerce").fillna(0.5),
    )
    same_day_bias_pair = np.maximum(
        pd.to_numeric(merged["anchor_same_day_bias_fit_score"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["partner_same_day_bias_fit_score"], errors="coerce").fillna(0.0),
    )
    merged = add_pair_pace_regime_scores(merged, pressure_pair, front_bonus)
    merged = add_front_survival_context_scores(merged, pressure_pair, front_bonus)
    front_front_slow_fit = (
        anchor_front_prob
        * partner_front_prob
        * (0.55 * slow_pace_pair + 0.45 * norm01(front_adv_pair))
    ).clip(0.0, 1.0)
    front_front_clash = (anchor_front_prob * partner_front_prob * pressure_pair).clip(0.0, 1.0)
    front_collapse_risk = (
        np.maximum(anchor_front_prob, partner_front_prob) * collapse_pair
    ).clip(0.0, 1.0)
    front_value_risk = (0.60 * front_front_clash + 0.40 * front_collapse_risk).clip(0.0, 1.0)
    anchor_closer_rate = pd.to_numeric(merged["anchor_horse_closer_rate_past5"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    partner_closer_rate = pd.to_numeric(merged["partner_horse_closer_rate_past5"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    anchor_closing_tendency = norm01(pd.to_numeric(merged["anchor_closing_tendency"], errors="coerce").fillna(0.0))
    partner_closing_tendency = norm01(pd.to_numeric(merged["partner_closing_tendency"], errors="coerce").fillna(0.0))
    closer_pair_max = np.maximum(
        np.maximum(anchor_closer_rate, partner_closer_rate),
        np.maximum(anchor_closing_tendency, partner_closing_tendency),
    ).clip(0.0, 1.0)
    style_diversity_score = np.maximum(
        anchor_front_prob * partner_closer_rate,
        partner_front_prob * anchor_closer_rate,
    ).clip(0.0, 1.0)
    position_front_value_score = (
        0.34 * pct_rank(front_front_slow_fit)
        + 0.22 * pct_rank(front_adv_pair)
        + 0.16 * pct_rank(draw_pace_pair)
        + 0.12 * pct_rank(same_day_bias_pair)
        + 0.16 * pct_rank(front_value_risk, higher_is_better=False)
    ).clip(0.0, 1.0)
    collapse_pct = pct_rank(collapse_pair)
    closer_pct = pct_rank(closer_pair_max)
    slow_pct = pct_rank(front_front_slow_fit)
    clash_pct = pct_rank(front_front_clash)
    front_value_pct = pct_rank(position_front_value_score)
    overlay_pct = pct_rank(overlay)
    danger_pct = pct_rank(danger)
    position_closer_value_score = (
        0.36 * collapse_pct
        + 0.24 * closer_pct
        + 0.16 * pct_rank(style_diversity_score)
        + 0.12 * pct_rank(draw_pace_pair)
        + 0.12 * pct_rank(front_value_risk)
    ).clip(0.0, 1.0)
    closer_logic_watch_flag = (
        collapse_pct.between(0.33, 0.85)
        & closer_pct.ge(0.33)
        & slow_pct.le(0.75)
        & clash_pct.ge(0.25)
        & front_value_pct.le(0.75)
        & overlay_pct.ge(0.33)
        & danger_pct.le(0.92)
    ).astype(float)
    closer_logic_watch_score = (
        0.34 * position_closer_value_score
        + 0.20 * collapse_pct
        + 0.16 * closer_pct
        + 0.12 * clash_pct
        + 0.10 * overlay_pct
        + 0.08 * pct_rank(danger, higher_is_better=False)
    ).clip(0.0, 1.0)
    workout_pair = np.maximum(merged["anchor_workout_runtime_score"], merged["partner_workout_runtime_score"])
    workout_auto_pair = np.maximum(
        pd.to_numeric(merged.get("anchor_workout_auto_knowledge_score"), errors="coerce").fillna(0.0),
        pd.to_numeric(merged.get("partner_workout_auto_knowledge_score"), errors="coerce").fillna(0.0),
    ).clip(0.0, 1.0)
    anchor_fast_clock = pd.to_numeric(merged["anchor_fast_clock_runtime_score"], errors="coerce").fillna(0.5)
    partner_fast_clock = pd.to_numeric(merged["partner_fast_clock_runtime_score"], errors="coerce").fillna(0.5)
    fast_clock_pair = ((anchor_fast_clock + partner_fast_clock) / 2.0).clip(0.0, 1.0)
    fast_clock_pair_min = np.minimum(anchor_fast_clock, partner_fast_clock).clip(0.0, 1.0)
    anchor_time_refinement = pd.to_numeric(merged["anchor_time_refinement_composite"], errors="coerce").fillna(0.5)
    partner_time_refinement = pd.to_numeric(merged["partner_time_refinement_composite"], errors="coerce").fillna(0.5)
    time_refinement_pair = (
        0.60 * np.maximum(anchor_time_refinement, partner_time_refinement)
        + 0.40 * ((anchor_time_refinement + partner_time_refinement) / 2.0)
    ).clip(0.0, 1.0)
    time_refinement_pair_min = np.minimum(anchor_time_refinement, partner_time_refinement).clip(0.0, 1.0)
    anchor_time_relative = pd.to_numeric(merged["anchor_time_value_relative_rank_score"], errors="coerce").fillna(0.5)
    partner_time_relative = pd.to_numeric(merged["partner_time_value_relative_rank_score"], errors="coerce").fillna(0.5)
    time_relative_pair_avg = ((anchor_time_relative + partner_time_relative) / 2.0).clip(0.0, 1.0)
    time_relative_pair_min = np.minimum(anchor_time_relative, partner_time_relative).clip(0.0, 1.0)
    anchor_fast_clock_today = pd.to_numeric(merged["anchor_fast_clock_x_today_likelihood"], errors="coerce").fillna(0.5)
    partner_fast_clock_today = pd.to_numeric(merged["partner_fast_clock_x_today_likelihood"], errors="coerce").fillna(0.5)
    fast_clock_today_pair_min = np.minimum(anchor_fast_clock_today, partner_fast_clock_today).clip(0.0, 1.0)
    pace_tracking_pair = (
        (
            pd.to_numeric(merged["anchor_pace_tracking_score"], errors="coerce").fillna(0.5)
            + pd.to_numeric(merged["partner_pace_tracking_score"], errors="coerce").fillna(0.5)
        )
        / 2.0
    ).clip(0.0, 1.0)
    late_speed_pair = (
        (
            pd.to_numeric(merged["anchor_late_speed_value"], errors="coerce").fillna(0.5)
            + pd.to_numeric(merged["partner_late_speed_value"], errors="coerce").fillna(0.5)
        )
        / 2.0
    ).clip(0.0, 1.0)
    sustained_speed_pair = (
        (
            pd.to_numeric(merged["anchor_sustained_speed_score"], errors="coerce").fillna(0.5)
            + pd.to_numeric(merged["partner_sustained_speed_score"], errors="coerce").fillna(0.5)
        )
        / 2.0
    ).clip(0.0, 1.0)
    expected_pace_text = first_text_series(merged, ["anchor_expected_pace", "partner_expected_pace"], "").str.lower()
    expected_fast = expected_pace_text.str.contains("fast|ハイ", regex=True).astype(float)
    expected_slow = expected_pace_text.str.contains("slow|スロー", regex=True).astype(float)
    expected_middle = expected_pace_text.str.contains("middle|mid|ミドル", regex=True).astype(float)
    race_pressure_for_lap = np.maximum(
        pd.to_numeric(merged.get("anchor_race_early_pressure_score"), errors="coerce").fillna(0.5),
        pd.to_numeric(merged.get("partner_race_early_pressure_score"), errors="coerce").fillna(0.5),
    ).clip(0.0, 1.0)
    race_collapse_for_lap = np.maximum(
        pd.to_numeric(merged.get("anchor_race_pace_collapse_risk"), errors="coerce").fillna(0.5),
        pd.to_numeric(merged.get("partner_race_pace_collapse_risk"), errors="coerce").fillna(0.5),
    ).clip(0.0, 1.0)
    race_slow_for_lap = np.maximum(
        pd.to_numeric(merged.get("anchor_race_slow_pace_risk"), errors="coerce").fillna(0.5),
        pd.to_numeric(merged.get("partner_race_slow_pace_risk"), errors="coerce").fillna(0.5),
    ).clip(0.0, 1.0)
    front_adv_for_lap = np.maximum(
        pd.to_numeric(merged.get("anchor_front_advantage_score"), errors="coerce").fillna(0.5),
        pd.to_numeric(merged.get("partner_front_advantage_score"), errors="coerce").fillna(0.5),
    ).clip(0.0, 1.0)
    closer_adv_for_lap = np.maximum(
        pd.to_numeric(merged.get("anchor_closer_advantage_score"), errors="coerce").fillna(0.5),
        pd.to_numeric(merged.get("partner_closer_advantage_score"), errors="coerce").fillna(0.5),
    ).clip(0.0, 1.0)
    pos_adv_for_lap = np.maximum(
        pd.to_numeric(merged.get("anchor_positioning_advantage_score"), errors="coerce").fillna(0.5),
        pd.to_numeric(merged.get("partner_positioning_advantage_score"), errors="coerce").fillna(0.5),
    ).clip(0.0, 1.0)
    lap_race_profile = normalize_profile(
        pd.DataFrame(
            {
                "fast": 0.38 * race_pressure_for_lap
                + 0.30 * race_collapse_for_lap
                + 0.20 * expected_fast
                + 0.12 * (1.0 - race_slow_for_lap),
                "slow": 0.42 * race_slow_for_lap
                + 0.20 * expected_slow
                + 0.18 * (1.0 - race_pressure_for_lap)
                + 0.20 * front_adv_for_lap,
                "instant": 0.46 * race_slow_for_lap
                + 0.20 * expected_slow
                + 0.18 * closer_adv_for_lap
                + 0.16 * (1.0 - race_pressure_for_lap),
                "sustain": 0.34 * race_pressure_for_lap
                + 0.26 * race_collapse_for_lap
                + 0.18 * expected_middle
                + 0.12 * pos_adv_for_lap
                + 0.10 * (1.0 - race_slow_for_lap),
                "long_spurt": 0.24 * race_pressure_for_lap
                + 0.22 * race_collapse_for_lap
                + 0.18 * closer_adv_for_lap
                + 0.18 * pos_adv_for_lap
                + 0.18 * expected_middle,
            },
            index=merged.index,
        )
    )
    lap_profile_concentration = lap_race_profile.max(axis=1)
    lap_race_confidence = (
        0.58 * ((lap_profile_concentration - 0.20) / 0.80).clip(0.0, 1.0)
        + 0.22 * (race_pressure_for_lap - race_slow_for_lap).abs().clip(0.0, 1.0)
        + 0.20
        * (
            1.0
            - (1.0 - (race_collapse_for_lap - race_slow_for_lap).abs().clip(0.0, 1.0)).clip(0.0, 1.0)
            * 0.35
        )
    ).clip(0.0, 1.0)

    def side_lap_raw(side: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "fast": pd.to_numeric(merged.get(f"{side}_horse_fast_lap_score_past5"), errors="coerce").fillna(0.0),
                "slow": pd.to_numeric(merged.get(f"{side}_horse_slow_lap_score_past5"), errors="coerce").fillna(0.0),
                "instant": pd.to_numeric(merged.get(f"{side}_horse_instant_lap_score_past5"), errors="coerce").fillna(0.0),
                "sustain": pd.to_numeric(merged.get(f"{side}_horse_sustain_lap_score_past5"), errors="coerce").fillna(0.0),
                "long_spurt": pd.to_numeric(merged.get(f"{side}_horse_long_spurt_lap_score_past5"), errors="coerce").fillna(0.0),
            },
            index=merged.index,
        ).clip(0.0, 1.0)

    anchor_lap_raw = side_lap_raw("anchor")
    partner_lap_raw = side_lap_raw("partner")
    anchor_lap_profile = normalize_profile(anchor_lap_raw)
    partner_lap_profile = normalize_profile(partner_lap_raw)
    anchor_lap_fit = (anchor_lap_raw * lap_race_profile).sum(axis=1).clip(0.0, 1.0)
    partner_lap_fit = (partner_lap_raw * lap_race_profile).sum(axis=1).clip(0.0, 1.0)
    anchor_lap_reliability = pd.to_numeric(
        merged.get("anchor_lap_aptitude_reliability_score"), errors="coerce"
    ).fillna(0.5).clip(0.0, 1.0)
    partner_lap_reliability = pd.to_numeric(
        merged.get("partner_lap_aptitude_reliability_score"), errors="coerce"
    ).fillna(0.5).clip(0.0, 1.0)
    anchor_lap_confident = (
        anchor_lap_fit * (0.55 + 0.45 * anchor_lap_reliability) * (0.60 + 0.40 * lap_race_confidence)
    ).clip(0.0, 1.0)
    partner_lap_confident = (
        partner_lap_fit * (0.55 + 0.45 * partner_lap_reliability) * (0.60 + 0.40 * lap_race_confidence)
    ).clip(0.0, 1.0)
    anchor_lap_sharpness = (anchor_lap_profile.max(axis=1) - anchor_lap_profile.mean(axis=1)).clip(0.0, 1.0)
    partner_lap_sharpness = (partner_lap_profile.max(axis=1) - partner_lap_profile.mean(axis=1)).clip(0.0, 1.0)
    anchor_lap_top_mode = anchor_lap_profile.idxmax(axis=1).astype(str)
    partner_lap_top_mode = partner_lap_profile.idxmax(axis=1).astype(str)
    anchor_lap_axis_candidate = (
        0.48 * anchor_lap_confident
        + 0.28 * pd.to_numeric(merged.get("anchor_lap_pace_versatility_score"), errors="coerce").fillna(0.5).clip(0.0, 1.0)
        + 0.24 * pct_rank(anchor_lap_fit)
    ).clip(0.0, 1.0)
    partner_lap_axis_candidate = (
        0.48 * partner_lap_confident
        + 0.28 * pd.to_numeric(merged.get("partner_lap_pace_versatility_score"), errors="coerce").fillna(0.5).clip(0.0, 1.0)
        + 0.24 * pct_rank(partner_lap_fit)
    ).clip(0.0, 1.0)
    anchor_pop_for_lap = pd.to_numeric(
        first_series(merged, ["anchor_popularity", "anchor_pop", "anchor_pop_rank"], 9.0), errors="coerce"
    ).fillna(9.0)
    partner_pop_for_lap = pd.to_numeric(
        first_series(merged, ["partner_popularity", "partner_pop", "partner_pop_rank"], 9.0), errors="coerce"
    ).fillna(9.0)
    anchor_lap_partner_specialist = (
        anchor_lap_confident
        * (0.55 + 0.45 * anchor_lap_sharpness)
        * (0.65 + 0.35 * (1.0 - norm01(anchor_pop_for_lap, lo=1.0, hi=9.0)))
    ).clip(0.0, 1.0)
    partner_lap_partner_specialist = (
        partner_lap_confident
        * (0.55 + 0.45 * partner_lap_sharpness)
        * (0.65 + 0.35 * (1.0 - norm01(partner_pop_for_lap, lo=1.0, hi=9.0)))
    ).clip(0.0, 1.0)
    lap_mode_same = anchor_lap_top_mode.eq(partner_lap_top_mode).astype(float)
    lap_contradiction = (
        0.48 * (1.0 - lap_mode_same)
        + 0.32 * (anchor_lap_fit - partner_lap_fit).abs().clip(0.0, 1.0)
        + 0.20 * (anchor_lap_sharpness - partner_lap_sharpness).abs().clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    pair_lap_same_fit = (
        0.52 * np.minimum(anchor_lap_confident, partner_lap_confident)
        + 0.28 * ((anchor_lap_confident + partner_lap_confident) / 2.0)
        + 0.20 * (1.0 - lap_contradiction)
    ).clip(0.0, 1.0)
    anchor_live_win = pd.to_numeric(merged.get("anchor_live_win_odds"), errors="coerce")
    partner_live_win = pd.to_numeric(merged.get("partner_live_win_odds"), errors="coerce")
    anchor_popular_lap_risk = (
        anchor_live_win.le(5.0).fillna(False).astype(float)
        * (1.0 - anchor_lap_fit)
        * (0.65 + 0.35 * lap_race_confidence)
    ).clip(0.0, 1.0)
    partner_popular_lap_risk = (
        partner_live_win.le(5.0).fillna(False).astype(float)
        * (1.0 - partner_lap_fit)
        * (0.65 + 0.35 * lap_race_confidence)
    ).clip(0.0, 1.0)
    lap_popular_mismatch = np.maximum(anchor_popular_lap_risk, partner_popular_lap_risk).clip(0.0, 1.0)
    anchor_race_quality_fit = pd.to_numeric(merged["anchor_race_quality_fit_score"], errors="coerce").fillna(0.5)
    partner_race_quality_fit = pd.to_numeric(merged["partner_race_quality_fit_score"], errors="coerce").fillna(0.5)
    race_quality_pair_fit = (
        0.58 * np.maximum(anchor_race_quality_fit, partner_race_quality_fit)
        + 0.42 * ((anchor_race_quality_fit + partner_race_quality_fit) / 2.0)
    ).clip(0.0, 1.0)
    race_quality_pair_min = np.minimum(anchor_race_quality_fit, partner_race_quality_fit).clip(0.0, 1.0)

    pair_fast_need = np.maximum(
        pd.to_numeric(merged.get("anchor_race_quality_fast_need_score"), errors="coerce").fillna(0.0),
        pd.to_numeric(merged.get("partner_race_quality_fast_need_score"), errors="coerce").fillna(0.0),
    ).clip(0.0, 1.0)
    pair_slow_need = np.maximum(
        pd.to_numeric(merged.get("anchor_race_quality_slow_need_score"), errors="coerce").fillna(0.0),
        pd.to_numeric(merged.get("partner_race_quality_slow_need_score"), errors="coerce").fillna(0.0),
    ).clip(0.0, 1.0)
    pair_sustain_need = (1.0 - np.maximum(pair_fast_need, pair_slow_need)).clip(0.0, 1.0)

    def side_past3_lap_fit(side: str) -> tuple[pd.Series, pd.Series]:
        weighted_sum = pd.Series(0.0, index=merged.index, dtype=float)
        evidence_weight = pd.Series(0.0, index=merged.index, dtype=float)
        best = pd.Series(0.0, index=merged.index, dtype=float)
        for lag, weight in [(1, 1.00), (2, 0.86), (3, 0.72)]:
            prefix = f"{side}_past{lag}_lap"
            fast = pd.to_numeric(merged.get(f"{prefix}_fast_success"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
            slow = pd.to_numeric(merged.get(f"{prefix}_slow_success"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
            instant = pd.to_numeric(merged.get(f"{prefix}_instant_success"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
            sustain = pd.to_numeric(merged.get(f"{prefix}_sustain_success"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
            long_spurt = pd.to_numeric(merged.get(f"{prefix}_long_spurt_success"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
            rpci = pd.to_numeric(merged.get(f"{prefix}_rpci"), errors="coerce")
            has_lag = rpci.notna().astype(float)
            lag_fit = (
                pair_fast_need * (0.72 * fast + 0.18 * sustain + 0.10 * long_spurt)
                + pair_slow_need * (0.58 * slow + 0.24 * instant + 0.18 * sustain)
                + pair_sustain_need * (0.42 * sustain + 0.34 * long_spurt + 0.24 * instant)
            ).clip(0.0, 1.0)
            weighted_sum = weighted_sum + weight * lag_fit * has_lag
            evidence_weight = evidence_weight + weight * has_lag
            best = np.maximum(best, lag_fit.where(has_lag.gt(0), 0.0))
        fit = (weighted_sum / evidence_weight.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        ready = evidence_weight.gt(0.0).astype(float)
        return (0.70 * fit.fillna(0.5) + 0.30 * best.fillna(0.5)).where(ready.gt(0), 0.5).clip(0.0, 1.0), ready

    anchor_past3_lap_fit, anchor_past3_lap_ready = side_past3_lap_fit("anchor")
    partner_past3_lap_fit, partner_past3_lap_ready = side_past3_lap_fit("partner")
    past3_lap_pair_fit = (
        0.56 * np.maximum(anchor_past3_lap_fit, partner_past3_lap_fit)
        + 0.44 * ((anchor_past3_lap_fit + partner_past3_lap_fit) / 2.0)
    ).clip(0.0, 1.0)
    past3_lap_pair_min = np.minimum(anchor_past3_lap_fit, partner_past3_lap_fit).clip(0.0, 1.0)
    past3_lap_ready = np.maximum(anchor_past3_lap_ready, partner_past3_lap_ready).clip(0.0, 1.0)
    past3_lap_mid_band = (
        past3_lap_ready.gt(0.0)
        & past3_lap_pair_fit.between(PAST3_LAP_BUY_MIN, PAST3_LAP_BUY_MAX)
    ).astype(float)
    past3_lap_bad_band = (
        past3_lap_ready.gt(0.0)
        & ~past3_lap_pair_fit.between(PAST3_LAP_BUY_MIN, PAST3_LAP_BUY_MAX)
    ).astype(float)
    anchor_corner_shape = pd.to_numeric(merged["anchor_corner_shape_fit_runtime_score"], errors="coerce").fillna(0.5)
    partner_corner_shape = pd.to_numeric(merged["partner_corner_shape_fit_runtime_score"], errors="coerce").fillna(0.5)
    corner_shape_pair = ((anchor_corner_shape + partner_corner_shape) / 2.0).clip(0.0, 1.0)
    corner_shape_pair_max = np.maximum(anchor_corner_shape, partner_corner_shape).clip(0.0, 1.0)
    corner_shape_low_sample_count = (
        pd.to_numeric(merged["anchor_corner_shape_low_sample_flag"], errors="coerce").fillna(0.0)
        + pd.to_numeric(merged["partner_corner_shape_low_sample_flag"], errors="coerce").fillna(0.0)
    ).clip(0.0, 2.0)
    early_start_pair_shadow = np.maximum(
        pd.to_numeric(merged["anchor_early_start_shadow_score"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["partner_early_start_shadow_score"], errors="coerce").fillna(0.0),
    ).clip(0.0, 1.0)
    second_leg_pair_shadow = np.maximum(
        pd.to_numeric(merged["anchor_second_leg_shadow_score"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["partner_second_leg_shadow_score"], errors="coerce").fillna(0.0),
    ).clip(0.0, 1.0)
    gelding_pair_risk = np.maximum(
        pd.to_numeric(merged["anchor_gelding_risk_score"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["partner_gelding_risk_score"], errors="coerce").fillna(0.0),
    )
    gelding_pair_value = np.maximum(
        pd.to_numeric(merged["anchor_gelding_value_score"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["partner_gelding_value_score"], errors="coerce").fillna(0.0),
    )
    local_pair_risk = np.maximum(
        pd.to_numeric(merged["anchor_local_transition_risk_score"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["partner_local_transition_risk_score"], errors="coerce").fillna(0.0),
    )
    local_pair_value = np.maximum(
        pd.to_numeric(merged["anchor_local_transition_value_score"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["partner_local_transition_value_score"], errors="coerce").fillna(0.0),
    )
    first_condition_pair_raw_uncertainty = np.maximum(
        pd.to_numeric(merged["anchor_first_condition_uncertainty_score"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["partner_first_condition_uncertainty_score"], errors="coerce").fillna(0.0),
    )
    first_condition_pair_net_uncertainty = np.maximum(
        pd.to_numeric(merged["anchor_first_condition_net_uncertainty_score"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["partner_first_condition_net_uncertainty_score"], errors="coerce").fillna(0.0),
    )
    first_condition_pair_impressive = np.maximum(
        pd.to_numeric(merged["anchor_first_condition_prev_impressive_score"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["partner_first_condition_prev_impressive_score"], errors="coerce").fillna(0.0),
    )
    anchor_first_condition_supported = (
        pd.to_numeric(merged["anchor_first_condition_supported_uncertain_flag"], errors="coerce").fillna(0.0).ge(1.0)
    ).astype(float)
    partner_first_condition_supported = (
        pd.to_numeric(merged["partner_first_condition_supported_uncertain_flag"], errors="coerce").fillna(0.0).ge(1.0)
    ).astype(float)
    race_first_condition_supported_count = np.maximum(
        pd.to_numeric(merged["anchor_race_first_condition_supported_uncertain_count"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["partner_race_first_condition_supported_uncertain_count"], errors="coerce").fillna(0.0),
    )
    first_condition_ticket_supported_count = anchor_first_condition_supported + partner_first_condition_supported
    first_condition_unmodeled_supported = (
        race_first_condition_supported_count.sub(first_condition_ticket_supported_count).gt(0)
    ).astype(float)
    anchor_member_raw = (
        0.30 * pd.to_numeric(merged.get("anchor_confirmed_member_level_adjusted_score"), errors="coerce").fillna(0.0)
        + 0.22 * pd.to_numeric(merged.get("anchor_prev_race_member_level"), errors="coerce").fillna(0.0)
        + 0.18 * pd.to_numeric(merged.get("anchor_past3_max_race_member_level"), errors="coerce").fillna(0.0)
        + 0.14 * pd.to_numeric(merged.get("anchor_prev_confirmed_opponent_good_run_score"), errors="coerce").fillna(0.0)
        + 0.10 * pd.to_numeric(merged.get("anchor_prev_class_time_value_score"), errors="coerce").fillna(0.0)
        + 0.06 * pd.to_numeric(merged.get("anchor_rotation_class_down_flag"), errors="coerce").fillna(0.0)
        - 0.06 * pd.to_numeric(merged.get("anchor_rotation_class_up_flag"), errors="coerce").fillna(0.0)
    )
    partner_member_raw = (
        0.30 * pd.to_numeric(merged.get("partner_confirmed_member_level_adjusted_score"), errors="coerce").fillna(0.0)
        + 0.22 * pd.to_numeric(merged.get("partner_prev_race_member_level"), errors="coerce").fillna(0.0)
        + 0.18 * pd.to_numeric(merged.get("partner_past3_max_race_member_level"), errors="coerce").fillna(0.0)
        + 0.14 * pd.to_numeric(merged.get("partner_prev_confirmed_opponent_good_run_score"), errors="coerce").fillna(0.0)
        + 0.10 * pd.to_numeric(merged.get("partner_prev_class_time_value_score"), errors="coerce").fillna(0.0)
        + 0.06 * pd.to_numeric(merged.get("partner_rotation_class_down_flag"), errors="coerce").fillna(0.0)
        - 0.06 * pd.to_numeric(merged.get("partner_rotation_class_up_flag"), errors="coerce").fillna(0.0)
    )
    anchor_member_support = norm01(anchor_member_raw)
    partner_member_support = norm01(partner_member_raw)
    member_pair_support = (
        0.52 * np.maximum(anchor_member_support, partner_member_support)
        + 0.48 * ((anchor_member_support + partner_member_support) / 2.0)
    ).clip(0.0, 1.0)
    pair_prob_raw = np.sqrt((merged["anchor_ai_prob"] * merged["partner_ai_prob"]).clip(0.0, 1.0))
    merged["pair_quinella_score"] = (
        0.29 * ((merged["anchor_quinella_score"] + merged["partner_quinella_score"]) / 2.0)
        + 0.20 * norm01(pair_prob_raw, lo=0.015, hi=0.12)
        + 0.16 * overlay
        + 0.13 * place_joint
        + 0.10 * front_bonus
        + 0.07 * pace_pair
        + 0.05 * workout_pair
        + 0.006 * workout_auto_pair
        + 0.03 * gelding_pair_value
        + 0.02 * local_pair_value
        + 0.02 * first_condition_pair_impressive
        + 0.025 * member_pair_support
        + 0.015 * fast_clock_pair
        + 0.015 * race_quality_pair_fit
        + 0.010 * past3_lap_mid_band
        + 0.010 * corner_shape_pair
        - 0.12 * danger
        - 0.07 * gelding_pair_risk
        - 0.04 * local_pair_risk
        - 0.010 * past3_lap_bad_band
        - 0.010 * (corner_shape_low_sample_count / 2.0)
    ).clip(0.0, 1.0)
    merged["pair_score"] = (
        0.42 * merged["pair_quinella_score"]
        + 0.25 * overlay
        + 0.18 * front_bonus
        + 0.15 * (1.0 - danger)
        + 0.03 * gelding_pair_value
        + 0.02 * local_pair_value
        + 0.02 * first_condition_pair_impressive
        + 0.015 * fast_clock_pair
        + 0.012 * race_quality_pair_fit
        + 0.004 * workout_auto_pair
        + 0.010 * past3_lap_mid_band
        + 0.010 * corner_shape_pair
        - 0.08 * gelding_pair_risk
        - 0.04 * local_pair_risk
        - 0.010 * past3_lap_bad_band
        - 0.010 * (corner_shape_low_sample_count / 2.0)
    ).clip(0.0, 1.0)
    partner_place_score = pd.to_numeric(merged["partner_place_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    partner_quinella_score = pd.to_numeric(merged["partner_quinella_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    partner_ai_rank_num = pd.to_numeric(merged["partner_ai_rank_num"], errors="coerce").fillna(99.0)
    merged["partner_ability_floor_score"] = (
        0.34 * partner_place_score
        + 0.30 * partner_quinella_score
        + 0.22 * merged["pair_quinella_score"]
        + 0.14 * merged["pair_score"]
    ).clip(0.0, 1.0)
    merged["partner_ability_floor_fail_flag"] = (
        (
            partner_place_score.lt(0.42)
            & partner_quinella_score.lt(0.46)
        )
        | partner_ai_rank_num.gt(6)
        | (
            merged["partner_ability_floor_score"].lt(0.50)
            & front_bonus.lt(0.72)
        )
    ).astype(float)
    anchor_floor5 = pd.to_numeric(merged.get("anchor_ability_floor_score_5"), errors="coerce")
    partner_floor5 = pd.to_numeric(merged.get("partner_ability_floor_score_5"), errors="coerce")
    floor5_pair = pd.concat([anchor_floor5, partner_floor5], axis=1)
    floor5_available_count = floor5_pair.notna().sum(axis=1)
    merged["pair_min_ability_floor_score_5"] = floor5_pair.min(axis=1, skipna=True)
    merged["pair_avg_ability_floor_score_5"] = floor5_pair.mean(axis=1, skipna=True)
    merged["ability_floor_score_5_available_count"] = floor5_available_count.astype(float)
    merged["ability_floor_score_5_full_available_flag"] = floor5_available_count.ge(2).astype(float)
    merged["ability_floor_score_5_fail_flag"] = (
        floor5_available_count.ge(2)
        & pd.to_numeric(merged["pair_min_ability_floor_score_5"], errors="coerce").lt(0.20)
    ).astype(float)
    merged["basic_ability_floor_runtime_tag"] = np.select(
        [
            floor5_available_count.lt(1),
            floor5_available_count.lt(2),
            merged["ability_floor_score_5_fail_flag"].ge(1.0),
            pd.to_numeric(merged["pair_min_ability_floor_score_5"], errors="coerce").ge(0.20),
        ],
        [
            "ability_floor_no_history",
            "ability_floor_partial_history",
            "ability_floor_low_block",
            "ability_floor_clear",
        ],
        default="ability_floor_unknown",
    )
    merged["top_eval_in_pair_flag"] = (
        pd.to_numeric(merged["anchor_ai_rank_num"], errors="coerce").fillna(99.0).le(3)
        | partner_ai_rank_num.le(3)
    ).astype(float)
    merged["top_eval_missing_risk_flag"] = (
        merged["top_eval_in_pair_flag"].lt(1.0)
        & overlay.lt(0.78)
    ).astype(float)
    race_top_market_no = pd.to_numeric(
        merged.get("anchor_race_top_market_horse_no"), errors="coerce"
    ).fillna(
        pd.to_numeric(merged.get("partner_race_top_market_horse_no"), errors="coerce")
    )
    race_top_market_prob = pd.to_numeric(
        merged.get("anchor_race_top_market_prob"), errors="coerce"
    ).fillna(
        pd.to_numeric(merged.get("partner_race_top_market_prob"), errors="coerce")
    ).fillna(0.0)
    race_top_market_odds = pd.to_numeric(
        merged.get("anchor_race_top_market_odds"), errors="coerce"
    ).fillna(
        pd.to_numeric(merged.get("partner_race_top_market_odds"), errors="coerce")
    ).fillna(999.0)
    merged["top_market_in_pair_flag"] = (
        pd.to_numeric(merged["anchor_horse_no"], errors="coerce").eq(race_top_market_no)
        | pd.to_numeric(merged["partner_horse_no"], errors="coerce").eq(race_top_market_no)
    ).astype(float)
    merged["dominant_favorite_missing_flag"] = (
        merged["top_market_in_pair_flag"].lt(1.0)
        & (
            race_top_market_prob.ge(0.28)
            | race_top_market_odds.le(3.0)
        )
        & overlay.lt(0.86)
    ).astype(float)
    merged["race_top_market_horse_no"] = race_top_market_no
    merged["race_top_market_prob"] = race_top_market_prob
    merged["race_top_market_odds"] = race_top_market_odds
    pressure = pd.to_numeric(merged["anchor_race_early_pressure_score"], errors="coerce").fillna(0.0)
    confidence = pd.to_numeric(merged["anchor_ai_confidence_score"], errors="coerce").fillna(0.20)
    gap = pd.to_numeric(merged["anchor_ai_score_gap_to_second"], errors="coerce").fillna(0.0)
    field = pd.to_numeric(merged["anchor_field_size"], errors="coerce").fillna(12.0)
    merged["skip_risk_score"] = (
        0.30 * (1.0 - confidence)
        + 0.18 * norm01(field, lo=10.0, hi=18.0)
        + 0.16 * (pressure - 0.42).abs().clip(0.0, 0.42) / 0.42
        + 0.18 * (1.0 - norm01(gap, lo=0.0, hi=0.10))
        + 0.18 * danger
        + 0.10 * gelding_pair_risk
        + 0.05 * local_pair_risk
        - 0.04 * first_condition_pair_impressive
    ).clip(0.0, 1.0)
    merged["skip_risk_score"] = (
        merged["skip_risk_score"]
        + 0.020 * (1.0 - fast_clock_pair)
        + 0.025 * (corner_shape_low_sample_count / 2.0)
        + 0.020 * past3_lap_bad_band
    ).clip(0.0, 1.0)
    merged["market_overlay_score"] = overlay
    merged["projected_front5_prob"] = front_bonus
    merged["race_queue_clarity_score"] = queue_clarity_pair
    merged["race_front_duel_risk_score"] = front_duel_risk_pair
    merged["race_no_clear_leader_score"] = no_clear_leader_pair
    merged["race_projected_front_load_score"] = projected_front_load_pair
    merged["race_lead_top_score"] = lead_top_score_pair
    merged["race_lead_top_gap"] = lead_top_gap_pair
    merged["race_lead_candidate_count"] = lead_candidate_count_pair
    merged["race_front5_top_gap"] = front5_top_gap_pair
    merged["race_front5_candidate_count"] = front5_candidate_count_pair
    merged["race_pace_shape_label"] = pace_shape_pair_label
    merged["pace_fit_pair_score"] = pace_pair
    merged["collapse_fit"] = collapse_pair
    merged["front_front_slow_fit"] = front_front_slow_fit
    merged["front_front_clash"] = front_front_clash
    merged["front_collapse_risk_score"] = front_collapse_risk
    merged["position_front_value_score"] = position_front_value_score
    merged["closer_pair_max"] = closer_pair_max
    merged["style_diversity_score"] = style_diversity_score
    merged["position_closer_value_score"] = position_closer_value_score
    merged["closer_logic_watch_score"] = closer_logic_watch_score
    merged["closer_logic_watch_flag"] = closer_logic_watch_flag
    merged["closer_logic_collapse_pct"] = collapse_pct
    merged["closer_logic_closer_pct"] = closer_pct
    merged["closer_logic_slow_pct"] = slow_pct
    merged["closer_logic_clash_pct"] = clash_pct
    merged["closer_logic_front_value_pct"] = front_value_pct
    merged["closer_logic_overlay_pct"] = overlay_pct
    merged["workout_pair_score"] = workout_pair
    merged["workout_auto_pair_score"] = workout_auto_pair
    merged["fast_clock_pair_score"] = fast_clock_pair
    merged["fast_clock_pair_min_score"] = fast_clock_pair_min
    merged["time_refinement_pair_score"] = time_refinement_pair
    merged["time_refinement_pair_min_score"] = time_refinement_pair_min
    merged["time_relative_pair_avg_score"] = time_relative_pair_avg
    merged["time_relative_pair_min_score"] = time_relative_pair_min
    merged["fast_clock_today_pair_min_score"] = fast_clock_today_pair_min
    merged["pace_tracking_pair_score"] = pace_tracking_pair
    merged["late_speed_pair_score"] = late_speed_pair
    merged["sustained_speed_pair_score"] = sustained_speed_pair
    merged["time_refinement_runtime_tag"] = np.select(
        [
            time_refinement_pair.ge(TIME_REFINEMENT_SHADOW_PAIR_MIN)
            & time_relative_pair_min.ge(TIME_REFINEMENT_SHADOW_RELATIVE_MIN),
            time_relative_pair_min.lt(TIME_RELATIVE_LOW_CAUTION_MAX),
        ],
        ["time_refinement_strong_shadow", "time_relative_low_caution"],
        default="time_neutral",
    )
    merged["lap_pair_same_race_fit_score"] = pair_lap_same_fit
    merged["lap_pair_race_confidence_score"] = lap_race_confidence
    merged["lap_pair_contradiction_score"] = lap_contradiction
    merged["lap_pair_popular_mismatch_score"] = lap_popular_mismatch
    merged["anchor_lap_profile_fit_score"] = anchor_lap_fit
    merged["partner_lap_profile_fit_score"] = partner_lap_fit
    merged["anchor_lap_top_mode"] = anchor_lap_top_mode
    merged["partner_lap_top_mode"] = partner_lap_top_mode
    merged["lap_pair_runtime_tag"] = np.select(
        [
            pair_lap_same_fit.ge(LAP_PAIR_FIT_SHADOW_MIN)
            & lap_race_confidence.ge(LAP_PAIR_CONFIDENCE_SHADOW_MIN)
            & lap_contradiction.le(LAP_PAIR_CONTRADICTION_CAUTION_MAX),
            lap_popular_mismatch.ge(LAP_POPULAR_MISMATCH_CAUTION_MIN),
            lap_contradiction.gt(LAP_PAIR_CONTRADICTION_CAUTION_MAX),
        ],
        ["lap_pair_fit_shadow", "lap_popular_mismatch_caution", "lap_pair_contradiction_caution"],
        default="lap_neutral",
    )
    lap_light_safety_caution = (
        0.56 * ((lap_contradiction - LAP_PAIR_CONTRADICTION_CAUTION_MAX) / (0.85 - LAP_PAIR_CONTRADICTION_CAUTION_MAX)).clip(0.0, 1.0)
        + 0.44 * ((lap_popular_mismatch - LAP_POPULAR_MISMATCH_CAUTION_MIN) / (0.75 - LAP_POPULAR_MISMATCH_CAUTION_MIN)).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    merged["lap_light_safety_caution_score"] = lap_light_safety_caution
    merged["lap_light_safety_runtime_tag"] = np.select(
        [
            lap_light_safety_caution.ge(LAP_LIGHT_SAFETY_CAUTION_MAX),
            lap_light_safety_caution.ge(0.35),
            pair_lap_same_fit.ge(LAP_PAIR_FIT_SHADOW_MIN)
            & lap_race_confidence.ge(LAP_PAIR_CONFIDENCE_SHADOW_MIN)
            & lap_contradiction.le(LAP_PAIR_CONTRADICTION_CAUTION_MAX),
        ],
        ["lap_light_safety_block_unless_edge", "lap_light_safety_watch", "lap_light_safety_clear_fit"],
        default="lap_light_safety_neutral",
    )
    merged["race_quality_pair_fit_score"] = race_quality_pair_fit
    merged["race_quality_pair_min_score"] = race_quality_pair_min
    merged["past3_lap_pair_fit_score"] = past3_lap_pair_fit
    merged["past3_lap_pair_min_score"] = past3_lap_pair_min
    merged["past3_lap_evidence_ready"] = past3_lap_ready
    merged["past3_lap_mid_band_flag"] = past3_lap_mid_band
    merged["past3_lap_bad_band_flag"] = past3_lap_bad_band
    merged["past3_lap_band"] = np.select(
        [
            past3_lap_ready.lt(1.0),
            past3_lap_pair_fit.lt(PAST3_LAP_BUY_MIN),
            past3_lap_pair_fit.le(PAST3_LAP_BUY_MAX),
            past3_lap_pair_fit.gt(PAST3_LAP_BUY_MAX),
        ],
        ["no_history", "weak", "middle", "overvisible"],
        default="unknown",
    )
    merged["race_quality_label"] = first_text_series(
        merged, ["anchor_race_quality_label", "partner_race_quality_label"], "unknown"
    )
    merged["race_quality_v2_predicted_lap_mode"] = first_text_series(
        merged,
        ["anchor_race_quality_v2_predicted_lap_mode", "partner_race_quality_v2_predicted_lap_mode"],
        "unknown",
    )
    merged["race_quality_v2_confidence"] = pd.to_numeric(
        merged.get("anchor_race_quality_v2_confidence"), errors="coerce"
    ).fillna(pd.to_numeric(merged.get("partner_race_quality_v2_confidence"), errors="coerce")).fillna(0.0)
    merged["race_quality_v2_margin"] = pd.to_numeric(
        merged.get("anchor_race_quality_v2_margin"), errors="coerce"
    ).fillna(pd.to_numeric(merged.get("partner_race_quality_v2_margin"), errors="coerce")).fillna(0.0)
    for klass in ["fast", "slow", "instant", "sustain"]:
        merged[f"race_quality_v2_prob_{klass}"] = pd.to_numeric(
            merged.get(f"anchor_race_quality_v2_prob_{klass}"), errors="coerce"
        ).fillna(pd.to_numeric(merged.get(f"partner_race_quality_v2_prob_{klass}"), errors="coerce")).fillna(0.0)
    merged["race_quality_v2_runtime_tag"] = first_text_series(
        merged,
        ["anchor_race_quality_v2_runtime_tag", "partner_race_quality_v2_runtime_tag"],
        "v2_unknown",
    )
    merged["race_quality_context_ready"] = np.maximum(
        pd.to_numeric(merged["anchor_race_quality_context_ready"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["partner_race_quality_context_ready"], errors="coerce").fillna(0.0),
    )
    merged["race_quality_fast_need_score"] = np.maximum(
        pd.to_numeric(merged["anchor_race_quality_fast_need_score"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["partner_race_quality_fast_need_score"], errors="coerce").fillna(0.0),
    )
    merged["race_quality_slow_need_score"] = np.maximum(
        pd.to_numeric(merged["anchor_race_quality_slow_need_score"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["partner_race_quality_slow_need_score"], errors="coerce").fillna(0.0),
    )
    merged["race_quality_expected_front3f_sec"] = pd.to_numeric(
        merged["anchor_race_quality_expected_front3f_sec"], errors="coerce"
    ).fillna(pd.to_numeric(merged["partner_race_quality_expected_front3f_sec"], errors="coerce"))
    merged["hist_avg_front3f_sec"] = pd.to_numeric(merged["anchor_hist_avg_front3f_sec"], errors="coerce").fillna(
        pd.to_numeric(merged["partner_hist_avg_front3f_sec"], errors="coerce")
    )
    merged["hist_condition_sample_count"] = pd.to_numeric(
        merged["anchor_hist_condition_sample_count"], errors="coerce"
    ).fillna(pd.to_numeric(merged["partner_hist_condition_sample_count"], errors="coerce"))
    merged["corner_shape_pair_score"] = corner_shape_pair
    merged["corner_shape_pair_max_score"] = corner_shape_pair_max
    merged["corner_shape_low_sample_count"] = corner_shape_low_sample_count
    merged["early_start_pair_shadow_score"] = early_start_pair_shadow
    merged["second_leg_pair_shadow_score"] = second_leg_pair_shadow
    merged["gelding_pair_risk_score"] = gelding_pair_risk
    merged["gelding_pair_value_score"] = gelding_pair_value
    merged["local_pair_risk_score"] = local_pair_risk
    merged["local_pair_value_score"] = local_pair_value
    merged["member_pair_support_score"] = member_pair_support
    merged["first_condition_pair_raw_uncertainty_score"] = first_condition_pair_raw_uncertainty
    merged["first_condition_pair_uncertainty_score"] = first_condition_pair_net_uncertainty
    merged["first_condition_pair_impressive_prev_score"] = first_condition_pair_impressive
    merged["first_condition_supported_uncertain_in_ticket_count"] = first_condition_ticket_supported_count
    merged["first_condition_unmodeled_supported_not_in_ticket_flag"] = first_condition_unmodeled_supported
    merged["first_condition_extra_edge_required_flag"] = first_condition_pair_net_uncertainty.ge(0.45).astype(float)
    merged["ticket_danger_popular_score"] = danger
    merged["ticket_danger_popular_in_pair_score"] = danger_in_pair
    merged["race_difficulty_score"] = race_difficulty
    merged["late_value_survives_score"] = 0.50 + 0.30 * overlay - 0.12 * danger
    merged["late_value_survives_score"] = merged["late_value_survives_score"].clip(0.0, 1.0)

    def _norm_mode_series(series: pd.Series) -> pd.Series:
        mode = series.astype("string").fillna("").astype(str).str.lower().str.strip()
        return mode.replace(
            {
                "long_spurt": "sustain",
                "stamina": "sustain",
                "middle": "sustain",
                "mid": "sustain",
                "": "unknown",
                "nan": "unknown",
                "none": "unknown",
            }
        )

    v2_mode = _norm_mode_series(merged["race_quality_v2_predicted_lap_mode"])
    anchor_mode = _norm_mode_series(merged["anchor_lap_top_mode"])
    partner_mode = _norm_mode_series(merged["partner_lap_top_mode"])
    v2_known = v2_mode.ne("unknown")
    anchor_match = (anchor_mode.eq(v2_mode) & v2_known).astype(float)
    partner_match = (partner_mode.eq(v2_mode) & v2_known).astype(float)
    pair_mode_fit_raw = (
        0.56 * np.maximum(anchor_match, partner_match)
        + 0.24 * ((anchor_match + partner_match) / 2.0)
        + 0.20 * (
            anchor_mode.eq(partner_mode)
            & anchor_mode.ne("unknown")
            & partner_mode.ne("unknown")
        ).astype(float)
    ).where(v2_known, 0.5).clip(0.0, 1.0)
    pace_mode_consensus = (
        v2_known
        & (anchor_mode.eq(v2_mode) | partner_mode.eq(v2_mode))
        & anchor_mode.ne("unknown")
        & partner_mode.ne("unknown")
    ).astype(float)
    merged["continuous_pair_pace_fit_score"] = (
        0.26 * norm01(merged["lap_pair_same_race_fit_score"], lo=0.18, hi=0.55)
        + 0.18 * norm01(merged["lap_pair_race_confidence_score"], lo=0.20, hi=0.58)
        + 0.18 * (1.0 - norm01(merged["lap_pair_contradiction_score"], lo=0.05, hi=0.65))
        + 0.14 * norm01(merged["past3_lap_pair_min_score"], lo=0.12, hi=0.65)
        + 0.10 * pair_mode_fit_raw
        + 0.08 * norm01(merged["race_quality_v2_confidence"], lo=0.26, hi=0.38)
        + 0.04 * norm01(merged["race_quality_v2_margin"], lo=0.02, hi=0.10)
        + 0.02 * pace_mode_consensus
    ).clip(0.0, 1.0)
    merged["continuous_pair_readability_score"] = (
        0.32 * norm01(merged["race_quality_v2_confidence"], lo=0.24, hi=0.42)
        + 0.24 * norm01(merged["race_quality_v2_margin"], lo=0.01, hi=0.12)
        + 0.20 * pace_mode_consensus
        + 0.14 * (1.0 - norm01(merged["race_difficulty_score"], lo=0.38, hi=0.72))
        + 0.10 * norm01(merged["race_queue_clarity_score"], lo=0.28, hi=0.76)
    ).clip(0.0, 1.0)
    merged["continuous_pair_value_score"] = (
        0.24 * merged["pair_quinella_score"]
        + 0.20 * merged["pair_score"]
        + 0.18 * merged["market_overlay_score"]
        + 0.16 * norm01(merged["pair_score"], lo=0.45, hi=0.82)
        + 0.12 * merged["late_value_survives_score"]
        + 0.10 * norm01(merged["pair_quinella_score"], lo=0.48, hi=0.75)
    ).clip(0.0, 1.0)
    merged["continuous_pair_formal_score"] = (
        0.36 * merged["continuous_pair_value_score"]
        + 0.30 * merged["continuous_pair_pace_fit_score"]
        + 0.18 * merged["continuous_pair_readability_score"]
        + 0.10 * merged["member_pair_support_score"]
        + 0.06 * merged["race_quality_pair_fit_score"]
        - 0.10 * merged["ticket_danger_popular_score"]
        - 0.06 * merged["ticket_danger_popular_in_pair_score"]
    ).clip(0.0, 1.0)
    lap_positive_score = (
        0.24 * ((anchor_lap_fit + partner_lap_fit) / 2.0)
        + 0.18 * np.minimum(anchor_lap_confident, partner_lap_confident)
        + 0.16 * ((anchor_lap_axis_candidate + partner_lap_axis_candidate) / 2.0)
        + 0.12 * np.maximum(anchor_lap_partner_specialist, partner_lap_partner_specialist)
        + 0.12 * merged["race_quality_v2_confidence"]
        + 0.08 * merged["race_quality_v2_margin"]
        + 0.06 * np.maximum(anchor_match, partner_match)
        + 0.04 * (anchor_match * partner_match)
        - 0.16 * merged["lap_pair_popular_mismatch_score"]
    ).clip(0.0, 1.0)
    lap_role_a = (0.56 * anchor_lap_axis_candidate + 0.44 * partner_lap_partner_specialist).clip(0.0, 1.0)
    lap_role_b = (0.56 * partner_lap_axis_candidate + 0.44 * anchor_lap_partner_specialist).clip(0.0, 1.0)
    lap_role_score = np.maximum(lap_role_a, lap_role_b).clip(0.0, 1.0)

    lap_modes = ["fast", "slow", "instant", "sustain", "long_spurt"]

    def profile_similarity(profile: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
        aligned_profile = profile.reindex(columns=lap_modes).fillna(0.0)
        aligned_target = target.reindex(columns=lap_modes).fillna(1.0 / len(lap_modes))
        # For two one-hot distributions across five modes, mean absolute gap is 0.40.
        gap = aligned_profile.sub(aligned_target).abs().mean(axis=1)
        return (1.0 - gap / 0.40).fillna(0.5).clip(0.0, 1.0)

    anchor_strict_waveform_score = profile_similarity(anchor_lap_profile, lap_race_profile)
    partner_strict_waveform_score = profile_similarity(partner_lap_profile, lap_race_profile)
    strict_waveform_pair_avg = ((anchor_strict_waveform_score + partner_strict_waveform_score) / 2.0).clip(0.0, 1.0)
    strict_waveform_pair_min = np.minimum(anchor_strict_waveform_score, partner_strict_waveform_score).clip(0.0, 1.0)
    strict_waveform_pair_gap = (anchor_strict_waveform_score - partner_strict_waveform_score).abs().clip(0.0, 1.0)

    def side_goodrun_profile(side: str) -> tuple[pd.DataFrame, pd.Series]:
        weighted = pd.DataFrame(0.0, index=merged.index, columns=lap_modes)
        evidence_weight = pd.Series(0.0, index=merged.index, dtype=float)
        for lag, weight in [(1, 0.50), (2, 0.30), (3, 0.20)]:
            rpci = pd.to_numeric(merged.get(f"{side}_past{lag}_lap_rpci"), errors="coerce")
            has_lag = rpci.notna().astype(float)
            for mode in lap_modes:
                value = pd.to_numeric(
                    merged.get(f"{side}_past{lag}_lap_{mode}_success"),
                    errors="coerce",
                ).fillna(0.0).clip(0.0, 1.0)
                weighted[mode] = weighted[mode] + weight * value * has_lag
            evidence_weight = evidence_weight + weight * has_lag
        averaged = weighted.div(evidence_weight.replace(0.0, np.nan), axis=0).fillna(0.0)
        return normalize_profile(averaged), evidence_weight.gt(0.0).astype(float)

    anchor_goodrun_profile, anchor_goodrun_ready = side_goodrun_profile("anchor")
    partner_goodrun_profile, partner_goodrun_ready = side_goodrun_profile("partner")
    anchor_goodrun_fit = (
        0.56 * (anchor_goodrun_profile * lap_race_profile).sum(axis=1).clip(0.0, 1.0)
        + 0.44 * profile_similarity(anchor_goodrun_profile, lap_race_profile)
    ).where(anchor_goodrun_ready.gt(0.0), 0.5).clip(0.0, 1.0)
    partner_goodrun_fit = (
        0.56 * (partner_goodrun_profile * lap_race_profile).sum(axis=1).clip(0.0, 1.0)
        + 0.44 * profile_similarity(partner_goodrun_profile, lap_race_profile)
    ).where(partner_goodrun_ready.gt(0.0), 0.5).clip(0.0, 1.0)
    goodrun_lap_pair_avg = ((anchor_goodrun_fit + partner_goodrun_fit) / 2.0).clip(0.0, 1.0)
    goodrun_lap_pair_min = np.minimum(anchor_goodrun_fit, partner_goodrun_fit).clip(0.0, 1.0)
    goodrun_lap_pair_ready = np.maximum(anchor_goodrun_ready, partner_goodrun_ready).clip(0.0, 1.0)

    def side_front_role(side: str) -> pd.Series:
        candidates = [
            pd.to_numeric(merged.get(f"{side}_projected_front5_prob"), errors="coerce").fillna(0.0),
            pd.to_numeric(merged.get(f"{side}_front5_model_prob"), errors="coerce").fillna(0.0),
            pd.to_numeric(merged.get(f"{side}_horse_front_run_rate_past5"), errors="coerce").fillna(0.0),
            pd.to_numeric(merged.get(f"{side}_runtime_lead_score"), errors="coerce").fillna(0.0),
        ]
        return pd.concat(candidates, axis=1).max(axis=1).clip(0.0, 1.0)

    def side_receiver_role(side: str) -> pd.Series:
        candidates = [
            pd.to_numeric(merged.get(f"{side}_horse_closer_rate_past5"), errors="coerce").fillna(0.0),
            norm01(pd.to_numeric(merged.get(f"{side}_closing_tendency"), errors="coerce").fillna(0.0)),
            pd.to_numeric(merged.get(f"{side}_closer_advantage_score"), errors="coerce").fillna(0.0),
            pd.to_numeric(merged.get(f"{side}_sustained_speed_score"), errors="coerce").fillna(0.0),
        ]
        return pd.concat(candidates, axis=1).max(axis=1).clip(0.0, 1.0)

    anchor_front_role = side_front_role("anchor")
    partner_front_role = side_front_role("partner")
    anchor_receiver_role = side_receiver_role("anchor")
    partner_receiver_role = side_receiver_role("partner")
    sustain_need = lap_race_profile["sustain"].fillna(0.0).clip(0.0, 1.0)
    long_need = lap_race_profile["long_spurt"].fillna(0.0).clip(0.0, 1.0)
    instant_need = lap_race_profile["instant"].fillna(0.0).clip(0.0, 1.0)
    fast_need = lap_race_profile["fast"].fillna(0.0).clip(0.0, 1.0)
    front_survival_need = (0.36 * sustain_need + 0.28 * long_need + 0.22 * fast_need + 0.14 * strict_waveform_pair_avg).clip(0.0, 1.0)
    receiver_need = (0.36 * instant_need + 0.30 * sustain_need + 0.22 * long_need + 0.12 * strict_waveform_pair_avg).clip(0.0, 1.0)
    anchor_front_survival_role = (anchor_front_role * front_survival_need).clip(0.0, 1.0)
    partner_front_survival_role = (partner_front_role * front_survival_need).clip(0.0, 1.0)
    anchor_receiver_fit_role = (anchor_receiver_role * receiver_need).clip(0.0, 1.0)
    partner_receiver_fit_role = (partner_receiver_role * receiver_need).clip(0.0, 1.0)
    lap_role_front_receiver_pair_score = np.maximum(
        anchor_front_survival_role * partner_receiver_fit_role,
        partner_front_survival_role * anchor_receiver_fit_role,
    ).clip(0.0, 1.0)
    lap_role_front_front_collision_risk = (
        anchor_front_role
        * partner_front_role
        * (0.58 * front_duel_risk_pair + 0.42 * projected_front_load_pair)
    ).clip(0.0, 1.0)
    lap_role_pair_probability_proxy = (
        0.34 * strict_waveform_pair_avg
        + 0.26 * goodrun_lap_pair_avg
        + 0.26 * lap_role_front_receiver_pair_score
        + 0.14 * (1.0 - lap_role_front_front_collision_risk)
    ).clip(0.0, 1.0)
    lap_advanced_combo_score = (
        0.30 * strict_waveform_pair_avg
        + 0.18 * strict_waveform_pair_min
        + 0.24 * goodrun_lap_pair_avg
        + 0.18 * lap_role_pair_probability_proxy
        + 0.10 * merged["continuous_pair_pace_fit_score"]
    ).clip(0.0, 1.0)
    merged["strict_waveform_pair_avg_score"] = strict_waveform_pair_avg
    merged["strict_waveform_pair_min_score"] = strict_waveform_pair_min
    merged["strict_waveform_pair_gap_score"] = strict_waveform_pair_gap
    merged["goodrun_lap_pair_avg_score"] = goodrun_lap_pair_avg
    merged["goodrun_lap_pair_min_score"] = goodrun_lap_pair_min
    merged["goodrun_lap_pair_ready_score"] = goodrun_lap_pair_ready
    merged["lap_role_front_receiver_pair_score"] = lap_role_front_receiver_pair_score
    merged["lap_role_front_front_collision_risk"] = lap_role_front_front_collision_risk
    merged["lap_role_pair_probability_proxy"] = lap_role_pair_probability_proxy
    merged["lap_advanced_combo_score"] = lap_advanced_combo_score
    merged["lap_advanced_shadow_label"] = np.select(
        [
            lap_advanced_combo_score.ge(LAP_ADV_COMBO_STRONG_MIN)
            & lap_role_pair_probability_proxy.ge(LAP_ADV_ROLE_STRONG_MIN)
            & lap_role_front_front_collision_risk.le(LAP_ADV_COLLISION_SAFE_MAX),
            goodrun_lap_pair_avg.ge(LAP_ADV_GOODRUN_STRONG_MIN)
            & goodrun_lap_pair_ready.gt(0.0),
            lap_advanced_combo_score.ge(LAP_ADV_COMBO_STRONG_MIN),
            lap_advanced_combo_score.ge(LAP_ADV_COMBO_WATCH_MIN),
        ],
        [
            "lap_role_goodrun_strong",
            "goodrun_lap_strong",
            "lap_advanced_combo_strong",
            "lap_advanced_combo_watch",
        ],
        default="lap_advanced_neutral",
    )
    merged["lap_advanced_shadow_note"] = np.select(
        [
            merged["lap_advanced_shadow_label"].eq("lap_role_goodrun_strong"),
            merged["lap_advanced_shadow_label"].eq("goodrun_lap_strong"),
            merged["lap_advanced_shadow_label"].eq("lap_advanced_combo_strong"),
            merged["lap_advanced_shadow_label"].eq("lap_advanced_combo_watch"),
        ],
        [
            "ラップロール強: 想定ラップと前受け/差し役割がかみ合う",
            "好走時ラップ適性強: 好走した時のラップ型と今回想定が近い",
            "ラップ総合強: 波形・好走時ラップ・役割が揃う",
            "ラップ総合注視: ラップ面は準候補",
        ],
        default="",
    )
    merged = add_lap_track_shadow_features(merged)
    anchor_class_name = first_text_series(merged, ["anchor_クラス名", "anchor_class_name"], "")
    partner_class_name = first_text_series(merged, ["partner_クラス名", "partner_class_name"], "")
    anchor_distance_now = pd.to_numeric(first_series(merged, ["anchor_距離", "anchor_distance"], np.nan), errors="coerce")
    partner_distance_now = pd.to_numeric(first_series(merged, ["partner_距離", "partner_distance"], np.nan), errors="coerce")
    anchor_distance_prev = pd.to_numeric(first_series(merged, ["anchor_前距離", "anchor_previous_distance"], np.nan), errors="coerce")
    partner_distance_prev = pd.to_numeric(first_series(merged, ["partner_前距離", "partner_previous_distance"], np.nan), errors="coerce")
    anchor_distance_diff = (anchor_distance_now - anchor_distance_prev).where(
        anchor_distance_prev.notna(),
        pd.to_numeric(first_series(merged, ["anchor_distance_diff"], np.nan), errors="coerce"),
    )
    partner_distance_diff = (partner_distance_now - partner_distance_prev).where(
        partner_distance_prev.notna(),
        pd.to_numeric(first_series(merged, ["partner_distance_diff"], np.nan), errors="coerce"),
    )
    lap_1win_fast_same_distance_shadow = (
        anchor_class_name.eq("1勝")
        & partner_class_name.eq("1勝")
        & v2_mode.eq("fast")
        & anchor_distance_diff.abs().le(100).fillna(False)
        & partner_distance_diff.abs().le(100).fillna(False)
        & merged["lap_pair_popular_mismatch_score"].le(0.30)
    )
    merged["lap_positive_expansion_score"] = lap_positive_score
    merged["lap_axis_specialist_role_score"] = lap_role_score
    merged["lap_1win_fast_same_distance_shadow_flag"] = lap_1win_fast_same_distance_shadow.astype(float)
    merged["lap_positive_expansion_label"] = np.select(
        [
            lap_1win_fast_same_distance_shadow,
            lap_positive_score.ge(0.50)
            & merged["lap_pair_popular_mismatch_score"].le(0.28)
            & merged["race_quality_v2_confidence"].ge(0.30),
            lap_positive_score.ge(0.44)
            & merged["lap_pair_popular_mismatch_score"].le(0.34)
            & merged["race_quality_v2_confidence"].ge(0.24),
            lap_role_score.ge(0.50)
            & merged["lap_pair_popular_mismatch_score"].le(0.30)
            & merged["race_quality_v2_confidence"].ge(0.24),
        ],
        ["lap_1win_fast_same_distance_shadow", "lap_promote_strong", "lap_promote_watch", "lap_role_watch"],
        default="lap_neutral",
    )
    merged["lap_positive_expansion_note"] = np.select(
        [
            merged["lap_positive_expansion_label"].eq("lap_1win_fast_same_distance_shadow"),
            merged["lap_positive_expansion_label"].eq("lap_promote_strong"),
            merged["lap_positive_expansion_label"].eq("lap_promote_watch"),
            merged["lap_positive_expansion_label"].eq("lap_role_watch"),
        ],
        [
            "ラップ1勝シャドー強: fast想定と距離継続の条件が揃う",
            "ラップ適合から昇格候補",
            "ラップ面は準候補",
            "軸と相手のラップ役割は合う",
        ],
        default="",
    )
    merged["pace_pair_gate_label"] = np.select(
        [
            merged["continuous_pair_formal_score"].ge(0.72)
            & merged["continuous_pair_pace_fit_score"].ge(0.62)
            & pace_mode_consensus.ge(1.0),
            merged["continuous_pair_formal_score"].ge(0.64)
            & merged["continuous_pair_pace_fit_score"].ge(0.54),
            merged["continuous_pair_pace_fit_score"].lt(0.38)
            | merged["lap_pair_contradiction_score"].ge(0.62),
        ],
        ["pace_pair_strong", "pace_pair_watch", "pace_pair_caution"],
        default="pace_pair_neutral",
    )
    merged["pace_pair_gate_note"] = np.select(
        [
            merged["pace_pair_gate_label"].eq("pace_pair_strong"),
            merged["pace_pair_gate_label"].eq("pace_pair_watch"),
            merged["pace_pair_gate_label"].eq("pace_pair_caution"),
        ],
        [
            "\u5c55\u958b\u8aad\u307f\u3068\u30da\u30a2\u9069\u6027\u304c\u304b\u307f\u5408\u3046",
            "\u5c55\u958b\u9762\u306f\u6ce8\u8996",
            "\u5c55\u958b\u3068\u30da\u30a2\u9069\u6027\u306f\u5f31\u3081",
        ],
        default="\u5c55\u958b\u9762\u306f\u4e2d\u7acb",
    )
    merged["closer_shadow_score"] = (
        0.34 * merged["closer_logic_watch_score"]
        + 0.22 * merged["position_closer_value_score"]
        + 0.16 * merged["collapse_fit"]
        + 0.12 * merged["closer_pair_max"]
        + 0.10 * merged["style_diversity_score"]
        + 0.06 * merged["market_overlay_score"]
        - 0.08 * merged["front_front_slow_fit"]
    ).clip(0.0, 1.0)
    merged["closer_shadow_label"] = np.select(
        [
            merged["closer_shadow_score"].ge(0.68),
            merged["closer_shadow_score"].ge(0.52),
        ],
        ["closer_watch_strong", "closer_watch"],
        default="closer_neutral",
    )
    merged["closer_shadow_note"] = np.select(
        [
            merged["closer_shadow_label"].eq("closer_watch_strong"),
            merged["closer_shadow_label"].eq("closer_watch"),
        ],
        [
            "\u5dee\u3057\u304c\u5c4a\u304f\u5c55\u958b\u3092\u5f37\u3081\u306b\u8b66\u6212",
            "\u5dee\u3057\u6d6e\u4e0a\u306b\u6ce8\u610f",
        ],
        default="",
    )
    merged["late_odds_drop_rate"] = np.maximum(
        pd.to_numeric(merged["anchor_late_odds_drop_rate"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["partner_late_odds_drop_rate"], errors="coerce").fillna(0.0),
    ).clip(0.0, 3.0)
    merged["late_odds_drift_rate"] = np.maximum(
        pd.to_numeric(merged["anchor_late_odds_drift_rate"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["partner_late_odds_drift_rate"], errors="coerce").fillna(0.0),
    ).clip(0.0, 3.0)
    merged["session_odds_drop_rate"] = np.maximum(
        pd.to_numeric(merged["anchor_session_odds_drop_rate"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["partner_session_odds_drop_rate"], errors="coerce").fillna(0.0),
    ).clip(0.0, 5.0)
    merged["session_odds_drift_rate"] = np.maximum(
        pd.to_numeric(merged["anchor_session_odds_drift_rate"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["partner_session_odds_drift_rate"], errors="coerce").fillna(0.0),
    ).clip(0.0, 5.0)
    merged["odds_timeline_ready"] = np.maximum(
        pd.to_numeric(merged["anchor_odds_timeline_ready"], errors="coerce").fillna(0.0),
        pd.to_numeric(merged["partner_odds_timeline_ready"], errors="coerce").fillna(0.0),
    )
    merged["ticket_hit_prob"] = (
        0.020
        + 0.125 * merged["pair_quinella_score"]
        + 0.020 * (merged["anchor_ai_rank_num"].le(3) & merged["partner_ai_rank_num"].le(6)).astype(float)
        + 0.015 * front_bonus
        + 0.008 * pace_pair
        + 0.006 * workout_pair
        + 0.004 * merged["member_pair_support_score"]
        + 0.004 * fast_clock_pair
        + 0.003 * race_quality_pair_fit
        + 0.003 * corner_shape_pair
        + 0.003 * gelding_pair_value
        + 0.002 * local_pair_value
        - 0.020 * danger
        - 0.003 * (corner_shape_low_sample_count / 2.0)
        - 0.010 * gelding_pair_risk
        - 0.006 * local_pair_risk
    ).clip(0.015, 0.22)
    merged["required_target_roi"] = 1.35
    merged["min_acceptable_odds"] = (merged["required_target_roi"] / merged["ticket_hit_prob"]).clip(1.0, 200.0)
    pair_live_odds = pd.to_numeric(merged["live_odds"], errors="coerce")
    merged["live_odds"] = pair_live_odds.where(pair_live_odds.gt(0))
    live_odds = merged["live_odds"].fillna(0.0)
    merged["runtime_expected_roi"] = merged["ticket_hit_prob"] * live_odds
    slippage_haircut = (
        1.0
        - 0.18 * merged["late_odds_drift_rate"]
        - 0.08 * merged["late_odds_drop_rate"].where(overlay.lt(0.45), 0.0)
    ).clip(0.55, 1.03)
    merged["expected_roi_after_slippage"] = merged["runtime_expected_roi"] * slippage_haircut
    merged["min_odds_margin_ratio"] = live_odds / merged["min_acceptable_odds"].replace(0, np.nan)
    merged["strongest_current_score"] = (
        0.30 * norm01(merged["runtime_expected_roi"], lo=1.0, hi=4.0)
        + 0.24 * merged["pair_quinella_score"]
        + 0.18 * merged["market_overlay_score"]
        + 0.14 * merged["projected_front5_prob"]
        + 0.14 * (1.0 - merged["skip_risk_score"])
        + 0.02 * merged["gelding_pair_value_score"]
        + 0.01 * merged["local_pair_value_score"]
        + 0.015 * merged["first_condition_pair_impressive_prev_score"]
        + 0.015 * merged["member_pair_support_score"]
        + 0.020 * merged["fast_clock_pair_score"]
        + 0.018 * merged["race_quality_pair_fit_score"]
        + 0.016 * merged["past3_lap_mid_band_flag"]
        + 0.015 * merged["corner_shape_pair_score"]
        - 0.04 * merged["ticket_danger_popular_in_pair_score"]
        - 0.03 * merged["race_difficulty_score"]
        - 0.020 * (1.0 - merged["fast_clock_pair_score"])
        - 0.018 * merged["past3_lap_bad_band_flag"]
        - 0.020 * (merged["corner_shape_low_sample_count"] / 2.0)
        - 0.08 * merged["gelding_pair_risk_score"]
        - 0.04 * merged["local_pair_risk_score"]
    ).clip(0.0, 1.0)
    merged = add_runtime_umaren_sim_scores(merged)
    return merged


def add_runtime_umaren_sim_scores(merged: pd.DataFrame) -> pd.DataFrame:
    """Estimate each unordered pair's same-race top-2 probability for runtime umaren gating."""
    if merged.empty:
        out = merged.copy()
        out["race_sim_umaren_runtime_prob_raw"] = np.nan
        out["race_sim_umaren_runtime_rank"] = np.nan
        out["race_sim_umaren_runtime_available"] = 0.0
        return out

    out = merged.copy()
    idx = out.index

    def side_num(side: str, name: str, default: float = np.nan) -> pd.Series:
        col = f"{side}_{name}"
        if col not in out.columns:
            return pd.Series(default, index=idx, dtype=float)
        return pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    runner_parts: list[pd.DataFrame] = []
    for side in ("anchor", "partner"):
        front5 = side_num(side, "front5_model_prob").fillna(side_num(side, "projected_front5_prob")).clip(0.0, 1.0)
        pace_fit = side_num(side, "pace_fit_runtime_score").fillna(side_num(side, "pace_fit_score")).clip(0.0, 1.0)
        runner_parts.append(
            pd.DataFrame(
                {
                    "race_id": out["race_id"].astype(str),
                    "horse_no": side_num(side, "horse_no"),
                    "ai_prob": side_num(side, "ai_prob").clip(0.0, 1.0),
                    "ai_rank_score": side_num(side, "ai_rank_score").clip(0.0, 1.0),
                    "quinella": side_num(side, "quinella_score").clip(0.0, 1.0),
                    "place": side_num(side, "place_score").clip(0.0, 1.0),
                    "odds": side_num(side, "live_win_odds"),
                    "front5": front5,
                    "front_tendency": side_num(side, "horse_front_run_rate_past5").clip(0.0, 1.0),
                    "closer_tendency": side_num(side, "horse_closer_rate_past5").clip(0.0, 1.0),
                    "pace_fit": pace_fit,
                    "front_adv": side_num(side, "front_advantage_score").clip(0.0, 1.0),
                    "danger": side_num(side, "danger_popular_score").clip(0.0, 1.0),
                    "risk": side_num(side, "race_difficulty_score").clip(0.0, 1.0),
                    "pressure": side_num(side, "race_early_pressure_score").clip(0.0, 1.0),
                    "slow": side_num(side, "race_slow_pace_risk").clip(0.0, 1.0),
                    "collapse": side_num(side, "race_pace_collapse_risk").clip(0.0, 1.0),
                }
            )
        )

    runners = pd.concat(runner_parts, ignore_index=True).dropna(subset=["race_id", "horse_no"]).copy()
    if runners.empty:
        out["race_sim_umaren_runtime_prob_raw"] = np.nan
        out["race_sim_umaren_runtime_rank"] = np.nan
        out["race_sim_umaren_runtime_available"] = 0.0
        return out

    runners["horse_no"] = pd.to_numeric(runners["horse_no"], errors="coerce").astype("Int64")
    agg_cols = {col: "mean" for col in runners.columns if col not in {"race_id", "horse_no"}}
    runners = runners.groupby(["race_id", "horse_no"], as_index=False).agg(agg_cols)
    runners["inv_odds"] = 1.0 / runners["odds"].replace(0, np.nan)
    runners["market_prior"] = runners["inv_odds"] / runners.groupby("race_id")["inv_odds"].transform("sum")
    ai_norm = runners["ai_prob"] / runners.groupby("race_id")["ai_prob"].transform("sum")
    field_size = runners.groupby("race_id")["horse_no"].transform("count").replace(0, np.nan)
    runners["market_prior"] = runners["market_prior"].fillna(ai_norm).fillna(1.0 / field_size).clip(0.0005, 0.85)
    for col in [
        "ai_prob",
        "ai_rank_score",
        "quinella",
        "place",
        "front5",
        "front_tendency",
        "closer_tendency",
        "pace_fit",
        "front_adv",
        "danger",
        "risk",
        "pressure",
        "slow",
        "collapse",
    ]:
        runners[col] = pd.to_numeric(runners[col], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    runners["base_score"] = (
        0.26 * runners["market_prior"]
        + 0.20 * runners["ai_prob"]
        + 0.18 * runners["quinella"]
        + 0.13 * runners["place"]
        + 0.09 * runners["pace_fit"]
        + 0.07 * runners["front5"]
        + 0.04 * runners["ai_rank_score"]
        - 0.07 * runners["danger"]
        - 0.04 * runners["risk"]
    ).clip(0.0, 1.0)
    runners["front_score"] = (
        runners["base_score"]
        + 0.35 * runners["front5"]
        + 0.12 * runners["front_tendency"]
        + 0.12 * runners["front_adv"]
    ).clip(0.0, 1.5)
    runners["closer_score"] = (
        runners["base_score"]
        + 0.34 * runners["closer_tendency"]
        + 0.14 * runners["collapse"]
        + 0.08 * (1.0 - runners["front5"])
    ).clip(0.0, 1.5)

    def make_weights(group: pd.DataFrame, score_col: str, scale: float) -> pd.Series:
        centered = group[score_col].astype(float) - float(group[score_col].astype(float).mean())
        weights = np.power(group["market_prior"].astype(float).clip(0.0005, 0.85), 0.45) * np.exp(
            np.clip(scale * centered, -4.0, 4.0)
        )
        return pd.Series(weights, index=group.index).clip(1e-6)

    def unordered_top2_prob(weights: pd.Series, a_no: int, b_no: int) -> float:
        if a_no not in weights.index or b_no not in weights.index or a_no == b_no:
            return np.nan
        total = float(weights.sum())
        wa = float(weights.loc[a_no])
        wb = float(weights.loc[b_no])
        if total <= 0 or wa <= 0 or wb <= 0 or total <= max(wa, wb):
            return np.nan
        return float((wa / total) * (wb / max(total - wa, 1e-9)) + (wb / total) * (wa / max(total - wb, 1e-9)))

    def scenario_mix(group: pd.DataFrame) -> tuple[float, float, float]:
        slow = float(group["slow"].mean())
        collapse = float(group["collapse"].mean())
        pressure = float(group["pressure"].mean())
        front_adv = float(group["front_adv"].mean())
        front_share = float(np.clip(0.20 + 0.34 * slow + 0.20 * front_adv - 0.25 * collapse, 0.05, 0.65))
        collapse_share = float(np.clip(0.15 + 0.36 * collapse + 0.08 * pressure - 0.15 * slow, 0.05, 0.65))
        total = front_share + collapse_share
        if total > 0.88:
            front_share *= 0.88 / total
            collapse_share *= 0.88 / total
        return float(1.0 - front_share - collapse_share), front_share, collapse_share

    probabilities: dict[int, float] = {}
    top_rows: list[dict[str, Any]] = []
    runner_groups = {race_id: group.set_index("horse_no", drop=False) for race_id, group in runners.groupby("race_id", sort=False)}
    for race_id, pair_rows in out.groupby("race_id", sort=False):
        group = runner_groups.get(str(race_id))
        if group is None or group.empty:
            continue
        neutral_weights = make_weights(group, "base_score", 3.5)
        front_weights = make_weights(group, "front_score", 3.1)
        closer_weights = make_weights(group, "closer_score", 3.0)
        neutral_share, front_share, collapse_share = scenario_mix(group)
        race_probs: list[tuple[int, float]] = []
        for row_idx, row in pair_rows.iterrows():
            a = num(row.get("a_no"))
            b = num(row.get("b_no"))
            if not np.isfinite(a) or not np.isfinite(b):
                prob = np.nan
            else:
                a_no = int(a)
                b_no = int(b)
                prob = (
                    neutral_share * unordered_top2_prob(neutral_weights, a_no, b_no)
                    + front_share * unordered_top2_prob(front_weights, a_no, b_no)
                    + collapse_share * unordered_top2_prob(closer_weights, a_no, b_no)
                )
            probabilities[row_idx] = prob
            race_probs.append((row_idx, prob))
        valid = [(row_idx, prob) for row_idx, prob in race_probs if np.isfinite(prob)]
        if valid:
            top_idx, top_prob = max(valid, key=lambda item: item[1])
            top_row = out.loc[top_idx]
            a_top = int(num(top_row.get("a_no")))
            b_top = int(num(top_row.get("b_no")))
            top_rows.append(
                {
                    "race_id": str(race_id),
                    "race_sim_umaren_top_pair_key": f"{min(a_top, b_top)}-{max(a_top, b_top)}",
                    "race_sim_umaren_top_prob_raw": float(top_prob),
                    "race_sim_umaren_top_live_odds": num(top_row.get("live_odds")),
                }
            )

    out["race_sim_umaren_runtime_prob_raw"] = pd.Series(probabilities, dtype=float)
    out["race_sim_umaren_runtime_rank"] = out.groupby("race_id")["race_sim_umaren_runtime_prob_raw"].rank(
        method="first", ascending=False
    )
    out["race_sim_umaren_runtime_rank_score"] = (1.0 - (out["race_sim_umaren_runtime_rank"] - 1.0) / 9.0).clip(0.0, 1.0)
    out["race_sim_umaren_runtime_available"] = out["race_sim_umaren_runtime_prob_raw"].notna().astype(float)
    if top_rows:
        top = pd.DataFrame(top_rows).drop_duplicates("race_id", keep="last")
        out = out.merge(top, on="race_id", how="left")
    else:
        out["race_sim_umaren_top_pair_key"] = ""
        out["race_sim_umaren_top_prob_raw"] = np.nan
        out["race_sim_umaren_top_live_odds"] = np.nan
    return out


def gelding_reason_tag(row: pd.Series) -> str:
    notes = [
        text(row.get("anchor_gelding_context_note")),
        text(row.get("partner_gelding_context_note")),
    ]
    notes = [note for note in notes if note]
    if any("強く割引" in note or "人気薄は割引" in note for note in notes):
        return "gelding_risk"
    if any("2戦目" in note for note in notes):
        return "gelding_second_upside"
    if any("3戦目" in note for note in notes):
        return "gelding_third_underneath"
    if any("去勢明け初戦" in note for note in notes):
        return "gelding_debut_watch"
    if any("時期不明" in note for note in notes):
        return "gelding_unknown_timing"
    return "none"


def select_tickets(candidates: pd.DataFrame, *, max_per_day: int, max_per_race: int) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    work = candidates.copy()
    venue_code = work["race_id"].astype(str).str.slice(8, 10)
    venue_allowed = ~venue_code.eq("02")
    going_allowed = (
        pd.to_numeric(work.get("anchor_runtime_soft_heavy_flag", 0), errors="coerce").fillna(0.0).le(0)
        & pd.to_numeric(work.get("partner_runtime_soft_heavy_flag", 0), errors="coerce").fillna(0.0).le(0)
    )
    track_available = (
        pd.to_numeric(work.get("anchor_runtime_track_condition_available", 0), errors="coerce").fillna(0.0).ge(1)
        | pd.to_numeric(work.get("partner_runtime_track_condition_available", 0), errors="coerce").fillna(0.0).ge(1)
    )
    anchor_going_class = first_text_series(work, ["anchor_runtime_going_class"], "")
    partner_going_class = first_text_series(work, ["partner_runtime_going_class"], "")
    tokyo_wet_guard = (
        venue_code.eq("05")
        & track_available
        & (
            anchor_going_class.isin(["Yielding", "Soft", "Heavy"])
            | partner_going_class.isin(["Yielding", "Soft", "Heavy"])
        )
    )
    base_mask = (
        work["anchor_ai_rank_num"].le(3)
        & work["partner_ai_rank_num"].le(8)
        & work["anchor_strongest_feature_parity_ready"].ge(1.0)
        & work["partner_strongest_feature_parity_ready"].ge(1.0)
        & work["min_odds_margin_ratio"].ge(0.95)
        & work["runtime_expected_roi"].ge(1.35)
        & work["skip_risk_score"].le(0.52)
        & work["ticket_danger_popular_score"].le(0.70)
        & work["gelding_pair_risk_score"].le(0.35)
        & work["pair_quinella_score"].ge(0.52)
        & venue_allowed
        & going_allowed
        & ~tokyo_wet_guard
    )
    selected = work[base_mask].copy()
    if selected.empty:
        return selected

    first_condition_uncertainty = pd.to_numeric(
        selected.get("first_condition_pair_uncertainty_score", 0.0), errors="coerce"
    ).fillna(0.0)
    first_condition_extra_edge_ok = (
        first_condition_uncertainty.lt(0.60)
        | (
            selected["min_odds_margin_ratio"].ge(3.00)
            & selected["runtime_expected_roi"].ge(1.60)
        )
    )
    danger_popular_in_pair = pd.to_numeric(
        selected.get("ticket_danger_popular_in_pair_score", 0.0), errors="coerce"
    ).fillna(0.0)
    danger_popular_extra_edge_ok = (
        danger_popular_in_pair.lt(0.42)
        | (
            selected["min_odds_margin_ratio"].ge(3.00)
            & selected["runtime_expected_roi"].ge(1.60)
        )
    )
    race_difficulty = pd.to_numeric(selected.get("race_difficulty_score", 0.0), errors="coerce").fillna(0.0)
    race_readability_extra_edge_ok = (
        race_difficulty.le(0.58)
        | (
            selected["min_odds_margin_ratio"].ge(3.20)
            & selected["runtime_expected_roi"].ge(1.75)
        )
    )
    pace_regime_collapse_warning = pd.to_numeric(
        selected.get("pace_regime_collapse_warning_flag", 0.0), errors="coerce"
    ).fillna(0.0)
    pace_regime_safety_ok = pace_regime_collapse_warning.lt(1.0)
    partner_ability_floor_ok = pd.to_numeric(
        selected.get("partner_ability_floor_fail_flag", 0.0), errors="coerce"
    ).fillna(0.0).lt(1.0)
    basic_ability_floor_ok = pd.to_numeric(
        selected.get("ability_floor_score_5_fail_flag", 0.0), errors="coerce"
    ).fillna(0.0).lt(1.0)
    dominant_favorite_ok = pd.to_numeric(
        selected.get("dominant_favorite_missing_flag", 0.0), errors="coerce"
    ).fillna(0.0).lt(1.0)
    fast_clock_pair = pd.to_numeric(selected.get("fast_clock_pair_score", 0.5), errors="coerce").fillna(0.5)
    fast_clock_buy_ok = (
        fast_clock_pair.ge(FAST_CLOCK_BUY_MIN)
        | (
            selected["min_odds_margin_ratio"].ge(3.20)
            & selected["runtime_expected_roi"].ge(1.75)
        )
    )
    before_post_ok = pd.to_numeric(selected.get("race_started_flag", 0.0), errors="coerce").fillna(0.0).lt(1.0)
    race_sim_rank = pd.to_numeric(selected.get("race_sim_umaren_runtime_rank", np.nan), errors="coerce")
    race_sim_available = pd.to_numeric(
        selected.get("race_sim_umaren_runtime_available", 0.0), errors="coerce"
    ).fillna(0.0)
    race_sim_umaren_gate_ok = race_sim_available.lt(1.0) | race_sim_rank.le(5.0)
    past3_lap_ready = pd.to_numeric(selected.get("past3_lap_evidence_ready", 0.0), errors="coerce").fillna(0.0)
    past3_lap_mid = pd.to_numeric(selected.get("past3_lap_mid_band_flag", 0.0), errors="coerce").fillna(0.0)
    past3_lap_buy_ok = (
        past3_lap_ready.lt(1.0)
        | past3_lap_mid.ge(1.0)
        | (
            selected["min_odds_margin_ratio"].ge(3.20)
            & selected["runtime_expected_roi"].ge(1.75)
        )
    )
    lap_light_safety_caution = pd.to_numeric(
        selected.get("lap_light_safety_caution_score", 0.0), errors="coerce"
    ).fillna(0.0)
    lap_light_safety_ok = (
        lap_light_safety_caution.le(LAP_LIGHT_SAFETY_CAUTION_MAX)
        | (
            selected["min_odds_margin_ratio"].ge(LAP_LIGHT_SAFETY_EDGE_MARGIN)
            & selected["runtime_expected_roi"].ge(LAP_LIGHT_SAFETY_EDGE_ROI)
            & selected["strongest_current_score"].ge(0.90)
        )
    )

    # Keep the dashboard BUY layer sparse. The broad MCS/PBO proxy is useful
    # for watch-listing, but very long-odds pairs are too noisy. Historical
    # guard analysis supports keeping final executable umaren at 120x or less.
    strict_buy_mask = (
        before_post_ok
        & selected["strongest_current_score"].ge(0.86)
        & selected["min_odds_margin_ratio"].ge(2.50)
        & selected["skip_risk_score"].le(0.45)
        & selected["projected_front5_prob"].ge(0.60)
        & selected["pace_fit_pair_score"].ge(0.35)
        & selected["position_front_value_score"].ge(POSITION_FRONT_VALUE_MIN)
        & selected["workout_pair_score"].ge(0.20)
        & selected["live_odds"].le(MAX_FINAL_BUY_UMAREN_ODDS)
        & fast_clock_buy_ok
        & first_condition_extra_edge_ok
        & danger_popular_extra_edge_ok
        & race_readability_extra_edge_ok
        & pace_regime_safety_ok
        & partner_ability_floor_ok
        & basic_ability_floor_ok
        & dominant_favorite_ok
        & race_sim_umaren_gate_ok
        & past3_lap_buy_ok
        & lap_light_safety_ok
    )
    selected = selected[strict_buy_mask].copy()
    if selected.empty:
        return selected
    selected["selection_tier"] = "strongest_mcs_pbo_strict"
    selected["runtime_policy_gate"] = (
        "skip_hakodate|skip_soft_heavy|skip_tokyo_wet_guard|"
        f"umaren_odds_le{MAX_FINAL_BUY_UMAREN_ODDS:g}|"
        f"position_front_value_ge{POSITION_FRONT_VALUE_MIN:.2f}|"
        f"fast_clock_ge{FAST_CLOCK_BUY_MIN:.2f}_or_edge|"
        "first_condition_impressive_rescue_u0.60_m3.0|danger_popular_extra_edge|"
        "race_readability_extra_edge|pace_regime_collapse_warning_excluded|"
        "partner_ability_floor|basic_ability_floor5_ge0.20_if_full_history|"
        "dominant_favorite_missing_excluded|race_sim_umaren_rank_le5|"
        f"past3_lap_middle_{PAST3_LAP_BUY_MIN:.3f}_{PAST3_LAP_BUY_MAX:.3f}_or_edge|"
        f"lap_light_safety_le{LAP_LIGHT_SAFETY_CAUTION_MAX:.2f}_or_edge"
    )

    selected = selected.sort_values(
        ["race_id", "strongest_current_score", "min_odds_margin_ratio", "runtime_expected_roi"],
        ascending=[True, False, False, False],
    )
    selected = selected.groupby("race_id", as_index=False).head(max_per_race).copy()
    selected = selected.sort_values(
        ["strongest_current_score", "min_odds_margin_ratio", "runtime_expected_roi"],
        ascending=[False, False, False],
    ).head(max_per_day)

    base_stake = np.select(
        [
            selected["min_odds_margin_ratio"].ge(1.80) & selected["strongest_current_score"].ge(0.72),
            selected["min_odds_margin_ratio"].ge(1.25),
        ],
        [500.0, 300.0],
        default=100.0,
    )
    selected["runtime_stake_yen"] = base_stake
    selected["stake_yen"] = selected["runtime_stake_yen"]
    selected["runtime_action"] = "BUY"
    selected["runtime_decision_generated_before_post"] = True
    selected["runtime_purchase_valid"] = True
    selected["gelding_runtime_tag"] = selected.apply(gelding_reason_tag, axis=1)
    selected["first_condition_runtime_tag"] = np.where(
        pd.to_numeric(selected.get("first_condition_pair_uncertainty_score", 0.0), errors="coerce").fillna(0.0).ge(0.45),
        "low_sample_edge_ok",
        "clear",
    )
    selected["danger_popular_runtime_tag"] = np.where(
        pd.to_numeric(selected.get("ticket_danger_popular_in_pair_score", 0.0), errors="coerce").fillna(0.0).ge(0.42),
        "included_popular_edge_ok",
        "clear",
    )
    selected["race_readability_runtime_tag"] = np.where(
        pd.to_numeric(selected.get("race_difficulty_score", 0.0), errors="coerce").fillna(0.0).gt(0.58),
        "difficult_edge_ok",
        "clear",
    )
    selected["basic_ability_floor_runtime_tag"] = selected.get(
        "basic_ability_floor_runtime_tag",
        pd.Series("ability_floor_no_history", index=selected.index),
    ).astype(str)
    fast_clock_pair = pd.to_numeric(selected.get("fast_clock_pair_score", 0.5), errors="coerce").fillna(0.5)
    corner_shape_pair = pd.to_numeric(selected.get("corner_shape_pair_score", 0.5), errors="coerce").fillna(0.5)
    corner_shape_low_sample = pd.to_numeric(
        selected.get("corner_shape_low_sample_count", 0.0), errors="coerce"
    ).fillna(0.0)
    selected["speed_corner_runtime_tag"] = np.select(
        [
            fast_clock_pair.lt(FAST_CLOCK_BUY_MIN),
            corner_shape_pair.ge(CORNER_SHAPE_STRONG_SUPPORT),
            fast_clock_pair.ge(FAST_CLOCK_STRONG_SUPPORT),
            corner_shape_low_sample.ge(2.0),
        ],
        [
            "fast_clock_weak_edge_rescue",
            "corner_shape_fit",
            "fast_clock_fit",
            "corner_shape_low_sample",
        ],
        default="neutral",
    )
    selected["past3_lap_runtime_tag"] = np.select(
        [
            pd.to_numeric(selected.get("past3_lap_evidence_ready", 0.0), errors="coerce").fillna(0.0).lt(1.0),
            pd.to_numeric(selected.get("past3_lap_mid_band_flag", 0.0), errors="coerce").fillna(0.0).ge(1.0),
            pd.to_numeric(selected.get("past3_lap_pair_fit_score", 0.5), errors="coerce").fillna(0.5).gt(PAST3_LAP_BUY_MAX),
        ],
        [
            "past3_lap_no_history",
            "past3_lap_middle_fit",
            "past3_lap_overvisible_caution",
        ],
        default="past3_lap_weak_caution",
    )
    selected["workout_auto_pair_runtime_tag"] = np.select(
        [
            pd.to_numeric(selected.get("workout_auto_pair_score", 0.0), errors="coerce").fillna(0.0).ge(0.99),
            pd.to_numeric(selected.get("workout_auto_pair_score", 0.0), errors="coerce").fillna(0.0).ge(0.40),
        ],
        ["auto_workout_candidate", "auto_workout_shadow"],
        default="auto_workout_none",
    )
    selected["runtime_ticket_status"] = "最強版厳選買い"
    selected["runtime_reason"] = (
        "strongest_current_proxy"
        + "|umaren_only"
        + "|margin="
        + selected["min_odds_margin_ratio"].round(2).astype(str)
        + "|skip="
        + selected["skip_risk_score"].round(2).astype(str)
        + "|front5="
        + selected["projected_front5_prob"].round(2).astype(str)
        + "|front_value="
        + selected["position_front_value_score"].round(2).astype(str)
        + f"|umaren_odds_guard<={MAX_FINAL_BUY_UMAREN_ODDS:g}"
        + "|gelding="
        + selected["gelding_runtime_tag"].astype(str)
        + "|local_risk="
        + selected["local_pair_risk_score"].round(2).astype(str)
        + "|local_value="
        + selected["local_pair_value_score"].round(2).astype(str)
        + "|member="
        + selected["member_pair_support_score"].round(2).astype(str)
        + "|first_condition="
        + selected["first_condition_runtime_tag"].astype(str)
        + "|fc_unc="
        + (
            pd.to_numeric(selected.get("first_condition_pair_uncertainty_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|danger_popular="
        + selected["danger_popular_runtime_tag"].astype(str)
        + "|danger_in_pair="
        + (
            pd.to_numeric(selected.get("ticket_danger_popular_in_pair_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|readability="
        + selected["race_readability_runtime_tag"].astype(str)
        + "|difficulty="
        + (
            pd.to_numeric(selected.get("race_difficulty_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|pace_regime_front_survival="
        + (
            pd.to_numeric(selected.get("pace_regime_front_survival_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|pace_regime_collapse="
        + (
            pd.to_numeric(selected.get("pace_regime_collapse_conversion_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|partner_ability="
        + (
            pd.to_numeric(selected.get("partner_ability_floor_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|basic_ability_floor="
        + selected["basic_ability_floor_runtime_tag"].astype(str)
        + "|floor5="
        + (
            pd.to_numeric(selected.get("pair_min_ability_floor_score_5", np.nan), errors="coerce")
            .round(2)
            .astype(str)
        )
        + "|top_market_in_pair="
        + (
            pd.to_numeric(selected.get("top_market_in_pair_flag", 0.0), errors="coerce")
            .fillna(0.0)
            .round(0)
            .astype(str)
        )
        + "|top_market_odds="
        + (
            pd.to_numeric(selected.get("race_top_market_odds", 0.0), errors="coerce")
            .fillna(0.0)
            .round(1)
            .astype(str)
        )
        + "|speed_corner="
        + selected["speed_corner_runtime_tag"].astype(str)
        + "|fast_clock="
        + (
            pd.to_numeric(selected.get("fast_clock_pair_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|time_refinement="
        + selected.get("time_refinement_runtime_tag", pd.Series("time_neutral", index=selected.index)).astype(str)
        + "|time_pair="
        + (
            pd.to_numeric(selected.get("time_refinement_pair_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|time_relative_min="
        + (
            pd.to_numeric(selected.get("time_relative_pair_min_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|race_quality="
        + (
            pd.to_numeric(selected.get("race_quality_pair_fit_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|race_quality_label="
        + selected.get("race_quality_label", pd.Series("unknown", index=selected.index)).astype(str)
        + "|past3_lap="
        + selected["past3_lap_runtime_tag"].astype(str)
        + "|past3_lap_fit="
        + (
            pd.to_numeric(selected.get("past3_lap_pair_fit_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|lap_pair="
        + selected.get("lap_pair_runtime_tag", pd.Series("lap_neutral", index=selected.index)).astype(str)
        + "|lap_pair_fit="
        + (
            pd.to_numeric(selected.get("lap_pair_same_race_fit_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|lap_pair_conf="
        + (
            pd.to_numeric(selected.get("lap_pair_race_confidence_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|lap_pair_contra="
        + (
            pd.to_numeric(selected.get("lap_pair_contradiction_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|lap_light_safety="
        + selected.get("lap_light_safety_runtime_tag", pd.Series("lap_light_safety_neutral", index=selected.index)).astype(str)
        + "|lap_light_caution="
        + (
            pd.to_numeric(selected.get("lap_light_safety_caution_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|pace_pair_gate="
        + selected.get("pace_pair_gate_label", pd.Series("pace_pair_neutral", index=selected.index)).astype(str)
        + "|pace_pair_score="
        + (
            pd.to_numeric(selected.get("continuous_pair_formal_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|pace_pair_fit="
        + (
            pd.to_numeric(selected.get("continuous_pair_pace_fit_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|lap_promote="
        + selected.get("lap_positive_expansion_label", pd.Series("lap_neutral", index=selected.index)).astype(str)
        + "|lap_promote_score="
        + (
            pd.to_numeric(selected.get("lap_positive_expansion_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|lap_role_score="
        + (
            pd.to_numeric(selected.get("lap_axis_specialist_role_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|lap_adv="
        + selected.get("lap_advanced_shadow_label", pd.Series("lap_advanced_neutral", index=selected.index)).astype(str)
        + "|lap_adv_score="
        + (
            pd.to_numeric(selected.get("lap_advanced_combo_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|goodrun_lap="
        + (
            pd.to_numeric(selected.get("goodrun_lap_pair_avg_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|role_proxy="
        + (
            pd.to_numeric(selected.get("lap_role_pair_probability_proxy", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|wave_gap="
        + (
            pd.to_numeric(selected.get("strict_waveform_pair_gap_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|lap_track="
        + selected.get("lap_track_shadow_label", pd.Series("neutral", index=selected.index)).astype(str)
        + "|lap_track_score="
        + (
            pd.to_numeric(selected.get("lap_track_shadow_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|target_ra_lap="
        + selected.get("target_ra_lap_shadow_label", pd.Series("target_ra_lap_no_history", index=selected.index)).astype(str)
        + "|target_ra_fit="
        + (
            pd.to_numeric(selected.get("target_ra_lap_pair_fit_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|target_ra_mismatch="
        + (
            pd.to_numeric(selected.get("target_ra_lap_mismatch_risk_score", 1.0), errors="coerce")
            .fillna(1.0)
            .round(2)
            .astype(str)
        )
        + "|closer_shadow="
        + selected.get("closer_shadow_label", pd.Series("closer_neutral", index=selected.index)).astype(str)
        + "|closer_shadow_score="
        + (
            pd.to_numeric(selected.get("closer_shadow_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|front_context="
        + selected.get("front_context_gate_label", pd.Series("front_context_neutral", index=selected.index)).astype(str)
        + "|front_context_collapse="
        + (
            pd.to_numeric(selected.get("front_context_collapse_risk_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|workout_auto="
        + selected["workout_auto_pair_runtime_tag"].astype(str)
        + "|workout_auto_score="
        + (
            pd.to_numeric(selected.get("workout_auto_pair_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|corner_shape="
        + (
            pd.to_numeric(selected.get("corner_shape_pair_score", 0.0), errors="coerce")
            .fillna(0.0)
            .round(2)
            .astype(str)
        )
        + "|corner_low_sample="
        + (
            pd.to_numeric(selected.get("corner_shape_low_sample_count", 0.0), errors="coerce")
            .fillna(0.0)
            .round(0)
            .astype(str)
        )
    )
    return selected


def export_tickets(selected: pd.DataFrame, all_candidates: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "current_strongest_all_candidates.csv"
    selected_path = out_dir / "selected_after_live_safety.csv"
    dashboard_path = out_dir / "current_strongest_dashboard_tickets.csv"
    all_candidates.to_csv(all_path, index=False, encoding="utf-8-sig")

    if selected.empty:
        selected_out = pd.DataFrame(
            columns=[
                "race_id",
                "ticket_type",
                "anchor_no",
                "partner_no",
                "runtime_stake_yen",
                "runtime_action",
                "runtime_ticket_status",
            ]
        )
    else:
        selected_out = selected.copy()
        selected_out["ticket_type"] = "umaren"
        selected_out["anchor_no"] = pd.to_numeric(selected_out["anchor_horse_no"], errors="coerce").astype("Int64")
        selected_out["partner_no"] = pd.to_numeric(selected_out["partner_horse_no"], errors="coerce").astype("Int64")
        selected_out["horse_a"] = selected_out[["anchor_no", "partner_no"]].min(axis=1)
        selected_out["horse_b"] = selected_out[["anchor_no", "partner_no"]].max(axis=1)
        selected_out["anchor_name"] = selected_out["anchor_馬名"]
        selected_out["partner_name"] = selected_out["partner_馬名"]
        selected_out["quote_pay_proxy_per100"] = selected_out["live_pay_per100"]
        selected_out["quote_odds_proxy"] = selected_out["live_odds"]
        selected_out["runtime_odds"] = selected_out["live_odds"]
        selected_out["runtime_pay_per100"] = selected_out["live_pay_per100"]
        selected_out["dashboard_decision_label"] = selected_out["runtime_ticket_status"]
        selected_out["buy_reason_summary"] = selected_out["runtime_reason"]

    if "runtime_decision_generated_at" not in selected_out.columns:
        selected_out["runtime_decision_generated_at"] = datetime.now().isoformat(timespec="seconds")
    if "runtime_decision_generated_before_post" not in selected_out.columns:
        selected_out["runtime_decision_generated_before_post"] = True
    if "runtime_purchase_valid" not in selected_out.columns:
        selected_out["runtime_purchase_valid"] = True

    # Once a race has started, never let a later refresh create a new BUY for it.
    # Preserve only tickets that were already generated before post time; this
    # keeps actual pre-race bets available for PnL while blocking hindsight buys.
    if selected_path.exists() and not all_candidates.empty and "race_started_flag" in all_candidates.columns:
        try:
            previous = read_csv(selected_path, dtype={"race_id": str})
        except Exception:
            previous = pd.DataFrame()
        if not previous.empty and "race_id" in previous.columns:
            started_races = set(
                all_candidates.loc[
                    pd.to_numeric(all_candidates["race_started_flag"], errors="coerce").fillna(0.0).ge(1.0),
                    "race_id",
                ]
                .astype(str)
                .unique()
            )
            if started_races:
                before_col = previous.get(
                    "runtime_decision_generated_before_post",
                    pd.Series(False, index=previous.index),
                )
                before_post = before_col.astype(str).str.lower().isin({"true", "1", "1.0", "yes", "y"})
                valid_col = previous.get("runtime_purchase_valid", pd.Series(True, index=previous.index))
                valid_purchase = ~valid_col.astype(str).str.lower().isin({"false", "0", "0.0", "no", "n"})
                preserved = previous[
                    previous["race_id"].astype(str).isin(started_races) & before_post & valid_purchase
                ].copy()
                if not preserved.empty:
                    preserved["runtime_preserved_after_post"] = True
                    selected_out["runtime_preserved_after_post"] = False
                    selected_out = pd.concat([preserved, selected_out], ignore_index=True, sort=False)
                    selected_out = selected_out.drop_duplicates(
                        ["race_id", "ticket_type", "anchor_no", "partner_no"],
                        keep="first",
                    )
    for col, default in {
        "past3_lap_pair_fit_score": 0.0,
        "past3_lap_pair_min_score": 0.0,
        "past3_lap_evidence_ready": 0.0,
        "past3_lap_mid_band_flag": 0.0,
        "past3_lap_bad_band_flag": 0.0,
        "past3_lap_band": "no_history",
        "past3_lap_runtime_tag": "past3_lap_no_history",
        "lap_light_safety_caution_score": 0.0,
        "lap_light_safety_runtime_tag": "lap_light_safety_neutral",
        "workout_auto_pair_score": 0.0,
        "workout_auto_pair_runtime_tag": "auto_workout_none",
        "race_quality_v2_predicted_lap_mode": "unknown",
        "race_quality_v2_confidence": 0.0,
        "race_quality_v2_margin": 0.0,
        "race_quality_v2_runtime_tag": "v2_unknown",
        "continuous_pair_pace_fit_score": 0.0,
        "continuous_pair_readability_score": 0.0,
        "continuous_pair_value_score": 0.0,
        "continuous_pair_formal_score": 0.0,
        "pace_pair_gate_label": "pace_pair_neutral",
        "pace_pair_gate_note": "\u5c55\u958b\u9762\u306f\u4e2d\u7acb",
        "lap_positive_expansion_score": 0.0,
        "lap_axis_specialist_role_score": 0.0,
        "lap_1win_fast_same_distance_shadow_flag": 0.0,
        "lap_positive_expansion_label": "lap_neutral",
        "lap_positive_expansion_note": "",
        "strict_waveform_pair_avg_score": 0.0,
        "strict_waveform_pair_min_score": 0.0,
        "strict_waveform_pair_gap_score": 0.0,
        "goodrun_lap_pair_avg_score": 0.0,
        "goodrun_lap_pair_min_score": 0.0,
        "goodrun_lap_pair_ready_score": 0.0,
        "lap_role_front_receiver_pair_score": 0.0,
        "lap_role_front_front_collision_risk": 0.0,
        "lap_role_pair_probability_proxy": 0.0,
        "lap_advanced_combo_score": 0.0,
        "lap_advanced_shadow_label": "lap_advanced_neutral",
        "lap_advanced_shadow_note": "",
        "track_lap_regime": "track_mid_or_unknown",
        "lap_track_shadow_label": "neutral",
        "lap_track_shadow_score": 0.0,
        "lap_track_shadow_note": "",
        "closer_shadow_score": 0.0,
        "closer_shadow_label": "closer_neutral",
        "closer_shadow_note": "",
        "front_context_high_pressure_signal": 0.0,
        "front_context_survival_rate": 0.856,
        "front_context_collapse_rate": 0.035,
        "front_context_survival_support_score": 0.0,
        "front_context_collapse_risk_score": 0.0,
        "front_context_gate_label": "front_context_neutral",
        "front_context_gate_note": "",
        **TARGET_RA_LAP_DEFAULTS,
    }.items():
        if col not in selected_out.columns:
            selected_out[col] = default
    selected_out.to_csv(selected_path, index=False, encoding="utf-8-sig")
    selected_out.to_csv(dashboard_path, index=False, encoding="utf-8-sig")
    return {
        "all_candidates_csv": str(all_path),
        "selected_csv": str(selected_path),
        "dashboard_tickets_csv": str(dashboard_path),
        "candidate_tickets": int(len(all_candidates)),
        "candidate_races": int(all_candidates["race_id"].nunique()) if not all_candidates.empty else 0,
        "selected_tickets": int(len(selected_out)),
        "selected_races": int(selected_out["race_id"].nunique()) if not selected_out.empty and "race_id" in selected_out else 0,
        "stake_yen": float(pd.to_numeric(selected_out.get("runtime_stake_yen"), errors="coerce").fillna(0.0).sum())
        if not selected_out.empty
        else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build current-week strongest-version provisional tickets from pre-race AI and official live odds.")
    parser.add_argument("--prediction-csv", default="")
    parser.add_argument("--entry-csv", default="")
    parser.add_argument("--single-odds-csv", default="data/processed/live_odds/realtime_single_odds_latest.csv")
    parser.add_argument("--pair-odds-csv", default="data/processed/live_odds/realtime_pair_odds_latest.csv")
    parser.add_argument(
        "--odds-timeline-features-csv",
        default="data/processed/live_odds/realtime_single_odds_timeline_features.csv",
    )
    parser.add_argument("--track-condition-csv", default="data/processed/live_track_conditions/current_track_conditions.csv")
    parser.add_argument(
        "--historical-condition-context-csv",
        default="outputs/analysis/historical_condition_lap_context_v1/condition_lap_baselines.csv",
    )
    parser.add_argument("--target-ra-lap-history-csv", default=str(TARGET_RA_OFFICIAL_LAP_HISTORY_DEFAULT))
    parser.add_argument("--workout-auto-candidates-csv", default=WORKOUT_AUTO_CANDIDATES_DEFAULT)
    parser.add_argument("--gelding-history-csv", default="data/processed/gelding_transition/gelding_transition_history.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/current_strongest_runtime_v1")
    parser.add_argument("--max-per-day", type=int, default=4)
    parser.add_argument("--max-per-race", type=int, default=1)
    parser.add_argument("--update-latest-summary", action="store_true")
    args = parser.parse_args()

    prediction_path = project_path(args.prediction_csv) if args.prediction_csv else (
        latest_file("outputs/predictions/preday_target_de_overlay_*/baseline_predictions_*.csv")
        or latest_file("outputs/predictions/preday_strongest_feature_parity/baseline_predictions_*.csv")
        or latest_file("outputs/predictions/preday_netkeiba_enriched_odds_history_context/baseline_predictions_*.csv")
    )
    entry_path = project_path(args.entry_csv) if args.entry_csv else (
        latest_file("data/datasets/inference/weekly/entry_snapshot_*_target_de_overlay_enriched_workout_knowledge.csv")
        or latest_file("data/datasets/inference/weekly/entry_snapshot_*_target_de_overlay_enriched_workout.csv")
        or latest_file("data/datasets/inference/weekly/entry_snapshot_*_target_de_overlay_enriched.csv")
        or latest_file("data/datasets/inference/weekly/entry_snapshot_*_target_de_overlay.csv")
        or latest_file("data/datasets/inference/weekly/entry_snapshot_*_enriched_odds_workout_knowledge.csv")
        or latest_file("data/datasets/inference/weekly/entry_snapshot_*_enriched_odds.csv")
    )
    if prediction_path is None:
        raise FileNotFoundError("No strongest/history-context prediction CSV found.")
    single_path = project_path(args.single_odds_csv)
    pair_path = project_path(args.pair_odds_csv)
    odds_timeline_features_path = project_path(args.odds_timeline_features_csv) if args.odds_timeline_features_csv else None
    track_path = project_path(args.track_condition_csv)
    historical_condition_context_path = (
        project_path(args.historical_condition_context_csv) if args.historical_condition_context_csv else None
    )
    target_ra_lap_history_path = (
        project_path(args.target_ra_lap_history_csv) if args.target_ra_lap_history_csv else None
    )
    workout_auto_candidates_path = (
        project_path(args.workout_auto_candidates_csv) if args.workout_auto_candidates_csv else None
    )
    gelding_history_path = project_path(args.gelding_history_csv) if args.gelding_history_csv else None

    prediction = read_csv(prediction_path)
    entry = read_csv(entry_path) if entry_path is not None and entry_path.exists() else pd.DataFrame()
    gelding_history = read_gelding_history(gelding_history_path)
    if not entry.empty and not gelding_history.empty:
        entry = enrich_current_entries_with_gelding_context(entry, gelding_history)
    single = read_csv(single_path, dtype={"race_id": str})
    pair = read_csv(pair_path, dtype={"race_id": str})
    assert_prediction_odds_overlap(prediction, single, prediction_path, single_path)
    odds_timeline_features = (
        read_csv(odds_timeline_features_path, dtype={"race_id": str})
        if odds_timeline_features_path is not None and odds_timeline_features_path.exists()
        else pd.DataFrame()
    )
    track = read_csv(track_path) if track_path.exists() else pd.DataFrame()
    historical_condition_context = load_historical_condition_context(historical_condition_context_path)
    target_ra_lap_history = load_target_ra_lap_history(target_ra_lap_history_path)
    workout_auto_candidates = (
        read_csv(workout_auto_candidates_path)
        if workout_auto_candidates_path is not None and workout_auto_candidates_path.exists()
        else pd.DataFrame()
    )

    runners = prepare_runners(prediction, single, track, entry, odds_timeline_features)
    runners = apply_historical_race_quality_context(runners, historical_condition_context)
    runners = apply_auto_workout_knowledge(runners, workout_auto_candidates)
    candidates = build_pair_candidates(runners, pair)
    candidates = apply_target_ra_official_lap_overlay(candidates, target_ra_lap_history)
    selected = select_tickets(candidates, max_per_day=args.max_per_day, max_per_race=args.max_per_race)
    out_dir = project_path(args.output_dir)
    export = export_tickets(selected, candidates, out_dir)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "policy": "current_strongest_mcs_pbo_strict",
        "policy_note": "Current-week strict executable layer of the robust strongest policy: umaren-centered, margin/skip-risk/front-position/value gates. Very long-odds pairs stay out of final BUY until body weight and same-day bias are available.",
        "prediction_csv": str(prediction_path),
        "single_odds_csv": str(single_path),
        "pair_odds_csv": str(pair_path),
        "odds_timeline_features_csv": str(odds_timeline_features_path) if odds_timeline_features_path else "",
        "odds_timeline_features_rows": int(len(odds_timeline_features)),
        "historical_condition_context_csv": str(historical_condition_context_path)
        if historical_condition_context_path
        else "",
        "historical_condition_context_rows": int(len(historical_condition_context)),
        "target_ra_lap_history_csv": str(target_ra_lap_history_path) if target_ra_lap_history_path else "",
        "target_ra_lap_history_rows": int(len(target_ra_lap_history)),
        "workout_auto_candidates_csv": str(workout_auto_candidates_path) if workout_auto_candidates_path else "",
        "workout_auto_candidates_rows": int(len(workout_auto_candidates)),
        "gelding_history_csv": str(gelding_history_path) if gelding_history_path else "",
        "gelding_history_rows": int(len(gelding_history)),
        "rows": {
            "runners": int(len(runners)),
            "races": int(runners["race_id"].nunique()) if not runners.empty else 0,
            **export,
        },
        "selection_summary": selected[
            [
                "race_id",
                "anchor_horse_no",
                "anchor_馬名",
                "partner_horse_no",
                "partner_馬名",
                "live_odds",
                "ticket_hit_prob",
                "runtime_expected_roi",
                "min_odds_margin_ratio",
                "skip_risk_score",
                "strongest_current_score",
                "fast_clock_pair_score",
                "time_refinement_pair_score",
                "time_refinement_pair_min_score",
                "time_relative_pair_avg_score",
                "time_relative_pair_min_score",
                "fast_clock_today_pair_min_score",
                "time_refinement_runtime_tag",
                "race_quality_pair_fit_score",
                "race_quality_label",
                "race_quality_v2_predicted_lap_mode",
                "race_quality_v2_confidence",
                "race_quality_v2_margin",
                "race_quality_v2_runtime_tag",
                "past3_lap_pair_fit_score",
                "past3_lap_pair_min_score",
                "past3_lap_evidence_ready",
                "past3_lap_mid_band_flag",
                "past3_lap_bad_band_flag",
                "past3_lap_band",
                "past3_lap_runtime_tag",
                "lap_pair_same_race_fit_score",
                "lap_pair_race_confidence_score",
                "lap_pair_contradiction_score",
                "lap_pair_popular_mismatch_score",
                "lap_pair_runtime_tag",
                "lap_light_safety_caution_score",
                "lap_light_safety_runtime_tag",
                "lap_positive_expansion_score",
                "lap_axis_specialist_role_score",
                "lap_1win_fast_same_distance_shadow_flag",
                "lap_positive_expansion_label",
                "lap_positive_expansion_note",
                "strict_waveform_pair_avg_score",
                "strict_waveform_pair_min_score",
                "strict_waveform_pair_gap_score",
                "goodrun_lap_pair_avg_score",
                "goodrun_lap_pair_min_score",
                "goodrun_lap_pair_ready_score",
                "lap_role_front_receiver_pair_score",
                "lap_role_front_front_collision_risk",
                "lap_role_pair_probability_proxy",
                "lap_advanced_combo_score",
                "lap_advanced_shadow_label",
                "lap_advanced_shadow_note",
                "track_lap_regime",
                "lap_track_shadow_label",
                "lap_track_shadow_score",
                "lap_track_shadow_note",
                "target_ra_lap_pair_fit_score",
                "target_ra_lap_pair_fit_min_score",
                "target_ra_lap_pair_meanfit_score",
                "target_ra_lap_pair_need_match_score",
                "target_ra_lap_pair_strength_score",
                "target_ra_lap_pair_ready_score",
                "target_ra_lap_pair_ready_both",
                "target_ra_lap_mismatch_risk_score",
                "target_ra_lap_shadow_label",
                "target_ra_lap_shadow_note",
                "workout_auto_pair_score",
                "workout_auto_pair_runtime_tag",
                "pair_min_ability_floor_score_5",
                "pair_avg_ability_floor_score_5",
                "ability_floor_score_5_available_count",
                "basic_ability_floor_runtime_tag",
                "corner_shape_pair_score",
                "corner_shape_low_sample_count",
                "early_start_pair_shadow_score",
                "second_leg_pair_shadow_score",
                "runtime_stake_yen",
                "selection_tier",
            ]
        ].to_dict(orient="records")
        if not selected.empty
        else [],
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.update_latest_summary:
        latest_dir = project_path("outputs/analysis/race_day_runtime_operation_latest")
        latest_dir.mkdir(parents=True, exist_ok=True)
        latest_summary = {
            "output_root": str(out_dir),
            "mode": "current_strongest_strict",
            "final_tickets_csv": export["dashboard_tickets_csv"],
            "selected_csv": export["selected_csv"],
            "selected_metrics": {
                "tickets": export["selected_tickets"],
                "races": export["selected_races"],
                "stake_yen": export["stake_yen"],
            },
            "note": summary["policy_note"],
        }
        (latest_dir / "summary.json").write_text(json.dumps(latest_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
