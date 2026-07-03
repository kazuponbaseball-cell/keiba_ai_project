from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIAG = ROOT / "outputs/analysis/race_quality_prediction_v2/race_quality_v2_diagnostics.csv"
DEFAULT_TICKETS = ROOT / "outputs/analysis/lap_pair_refinement_candidates_v1/s_priority_tickets_with_lap_pair_refinement.csv"
DEFAULT_EXPECTED_LAP = ROOT / "outputs/analysis/estimated_front3f_race_quality_v1/race_expected_lap_diagnostics.csv"
DEFAULT_FRONT3F_CACHE = ROOT / "outputs/analysis/estimated_front3f_race_quality_v1/estimated_runner_front3f_cache.csv"
DEFAULT_FRONT3F_DIAG = ROOT / "outputs/analysis/estimated_front3f_race_quality_v1/front3f_race_quality_diagnostics.csv"
DEFAULT_COURSE_TEN = ROOT / "outputs/analysis/course_adjusted_front3f_signal_v1/course_adjusted_front3f_race_diagnostics.csv"
DEFAULT_FEATURE_SOURCES = [
    ROOT / "outputs/analysis/content_bridge_member_features_v1/train_features_with_content_bridge.csv",
    ROOT / "outputs/analysis/content_bridge_member_features_v1/test_features_with_content_bridge.csv",
]
DEFAULT_OUT = ROOT / "outputs/analysis/continuous_race_pace_prediction_v1"

TARGETS = ["front3f_sec", "rpci", "pci3"]
LAP_CLASSES = ["fast", "slow", "instant", "sustain"]


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)


def num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def sigmoid(x: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def normalize_race_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def normalize_surface_value(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    lower = text.lower()
    if "芝" in text or "turf" in lower:
        return "芝"
    if "ダ" in text or "dirt" in lower:
        return "ダート"
    return None


def class_tier(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    if any(k in text for k in ["G1", "Ｇ１"]):
        return "g1"
    if any(k in text for k in ["G2", "Ｇ２"]):
        return "g2"
    if any(k in text for k in ["G3", "Ｇ３"]):
        return "g3"
    if "リステッド" in text or "(L)" in text or "Ｌ" in text:
        return "listed"
    if "OP" in text or "オープン" in text:
        return "open"
    if "3勝" in text or "３勝" in text:
        return "3win"
    if "2勝" in text or "２勝" in text:
        return "2win"
    if "1勝" in text or "１勝" in text:
        return "1win"
    if "未勝利" in text:
        return "maiden"
    if "新馬" in text:
        return "newcomer"
    return "other"


def add_race_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["race_id"] = normalize_race_id(out["race_id"])
    out["year"] = out["race_id"].str[:4]
    out["race_date"] = out["race_id"].str[:8]
    out["venue_code"] = out["race_id"].str[8:10]
    out["race_no"] = out["race_id"].str[-2:]
    return out


def load_race_context(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            raw = read_csv(path, usecols=[0, 3, 5, 6, 12, 13, 18, 19, 20, 34, 40, 41])
        except Exception:
            continue
        if raw.empty:
            continue
        cols = list(raw.columns)
        renamed = raw.rename(
            columns={
                cols[0]: "date",
                cols[1]: "race_no_raw",
                cols[2]: "race_name",
                cols[3]: "class_name",
                cols[4]: "field_size",
                cols[5]: "starter_count",
                cols[6]: "surface_raw",
                cols[7]: "distance_m",
                cols[8]: "going",
                cols[9]: "race_id",
                cols[10]: "track_code",
                cols[11]: "race_type",
            }
        )
        frames.append(renamed)
    if not frames:
        return pd.DataFrame(columns=["race_id"])
    out = pd.concat(frames, ignore_index=True)
    out["race_id"] = normalize_race_id(out["race_id"])
    out["surface"] = out["surface_raw"].map(normalize_surface_value)
    out["class_tier"] = out["class_name"].map(class_tier)
    out["distance_m"] = pd.to_numeric(out["distance_m"], errors="coerce")
    out["field_size"] = pd.to_numeric(out["field_size"], errors="coerce")
    out["starter_count"] = pd.to_numeric(out["starter_count"], errors="coerce")
    keep = [
        "race_id",
        "date",
        "race_no_raw",
        "race_name",
        "class_name",
        "class_tier",
        "surface",
        "distance_m",
        "going",
        "field_size",
        "starter_count",
        "track_code",
        "race_type",
    ]
    return out[keep].dropna(subset=["race_id"]).drop_duplicates("race_id")


def load_front3f_targets(path: Path) -> pd.DataFrame:
    estimates = read_csv(path, usecols=["race_id", "race_first3f_sec", "race_last3f_sec", "race_total_time_sec", "distance_m"])
    estimates["race_id"] = normalize_race_id(estimates["race_id"])
    agg = (
        estimates.groupby("race_id", as_index=False)
        .agg(
            front3f_sec=("race_first3f_sec", "first"),
            last3f_sec=("race_last3f_sec", "first"),
            total_time_sec=("race_total_time_sec", "first"),
            distance_m_from_lap=("distance_m", "first"),
        )
        .reset_index(drop=True)
    )
    for col in ["front3f_sec", "last3f_sec", "total_time_sec", "distance_m_from_lap"]:
        agg[col] = pd.to_numeric(agg[col], errors="coerce")
    return agg


def load_expected_lap(path: Path) -> pd.DataFrame:
    raw = read_csv(path)
    race_col = "レースID(新/馬番無)" if "レースID(新/馬番無)" in raw.columns else raw.columns[0]
    out = raw.rename(
        columns={
            race_col: "race_id",
            "RPCI": "rpci",
            "PCI3": "pci3",
            "course_base_rpci_prior": "course_base_rpci_prior",
            "course_base_pci3_prior": "course_base_pci3_prior",
            "course_base_rpci_count": "course_base_rpci_count",
        }
    )
    out["race_id"] = normalize_race_id(out["race_id"])
    keep = [
        "race_id",
        "rpci",
        "pci3",
        "course_base_rpci_prior",
        "course_base_pci3_prior",
        "course_base_rpci_count",
    ]
    out = out[[c for c in keep if c in out.columns]].drop_duplicates("race_id")
    for col in [c for c in keep if c != "race_id" and c in out.columns]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_race_shape(path: Path, race_col_fallback: str = "race_id") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["race_id"])
    raw = read_csv(path)
    if "レースID(新/馬番無)" in raw.columns:
        raw = raw.rename(columns={"レースID(新/馬番無)": "race_id"})
    elif race_col_fallback in raw.columns:
        raw = raw.rename(columns={race_col_fallback: "race_id"})
    else:
        raw = raw.rename(columns={raw.columns[0]: "race_id"})
    raw["race_id"] = normalize_race_id(raw["race_id"])
    return raw.drop_duplicates("race_id")


def build_base_frame(args: argparse.Namespace) -> pd.DataFrame:
    feature_paths = [Path(p) for p in args.feature_source] if args.feature_source else DEFAULT_FEATURE_SOURCES
    feature_paths = [p if p.is_absolute() else ROOT / p for p in feature_paths]
    diag = add_race_keys(read_csv(Path(args.diag)))
    context = load_race_context(feature_paths)
    front = load_front3f_targets(Path(args.front3f_cache))
    expected = load_expected_lap(Path(args.expected_lap))
    front_diag = load_race_shape(Path(args.front3f_diag))
    course_ten = load_race_shape(Path(args.course_ten))

    out = diag.merge(context, on="race_id", how="left")
    out = out.merge(front, on="race_id", how="left")
    out = out.merge(expected, on="race_id", how="left")
    out = out.merge(front_diag, on="race_id", how="left", suffixes=("", "_frontdiag"))
    out = out.merge(course_ten, on="race_id", how="left", suffixes=("", "_course_ten"))
    if "distance_m" in out and "distance_m_from_lap" in out:
        out["distance_m"] = pd.to_numeric(out["distance_m"], errors="coerce").fillna(
            pd.to_numeric(out["distance_m_from_lap"], errors="coerce")
        )
    return out


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_candidates = [
        c
        for c in frame.columns
        if c.startswith("avg_race_need_")
        or c.startswith("field_mean_horse_lap_")
        or c.startswith("field_max_horse_lap_")
        or c.startswith("field_std_horse_lap_")
        or c.startswith("shape_")
        or c.startswith("v1_prob_")
        or c.startswith("v2_prob_")
        or c.startswith("centroid_prob_")
        or c
        in {
            "field_size_lap_rows",
            "v1_confidence",
            "v1_concentration",
            "v2_confidence",
            "v2_margin",
            "queue_clarity_score",
            "queue_duel_risk_score",
            "queue_front_load_score",
            "queue_top_gap",
            "queue_candidate_count",
            "course_front3f_prior_sec",
            "course_front3f_prior_std",
            "course_front3f_prior_count",
            "race_course_adj_ten_pressure_score",
            "race_course_adj_fast_start_count",
            "race_course_adj_ten_speed_gap_top2",
            "race_course_adj_queue_clarity_score",
            "race_est_ten_pressure_score",
            "race_est_fast_start_count",
            "race_est_ten_speed_gap_top2",
            "race_est_queue_clarity_score",
            "course_base_rpci_prior",
            "course_base_pci3_prior",
            "course_base_rpci_count",
            "distance_m",
            "field_size",
            "starter_count",
        }
    ]
    numeric = []
    for col in numeric_candidates:
        if col not in frame.columns:
            continue
        vals = pd.to_numeric(frame[col], errors="coerce")
        if vals.notna().sum() >= 30:
            numeric.append(col)
    categorical = [c for c in ["venue_code", "surface", "going", "class_tier"] if c in frame.columns]
    return numeric, categorical


def make_design(train: pd.DataFrame, test: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    x_train = train[numeric_cols].apply(pd.to_numeric, errors="coerce").copy()
    x_test = test[numeric_cols].apply(pd.to_numeric, errors="coerce").copy()
    med = x_train.median(axis=0).fillna(0.0)
    x_train = x_train.fillna(med)
    x_test = x_test.fillna(med)

    cats_train = train[categorical_cols].astype("string").fillna("missing") if categorical_cols else pd.DataFrame(index=train.index)
    cats_test = test[categorical_cols].astype("string").fillna("missing") if categorical_cols else pd.DataFrame(index=test.index)
    cats = pd.get_dummies(pd.concat([cats_train, cats_test], axis=0), columns=categorical_cols, prefix=categorical_cols, dtype=float)
    cats_train_d = cats.iloc[: len(train)].set_index(train.index)
    cats_test_d = cats.iloc[len(train) :].set_index(test.index)

    design_train = pd.concat([x_train, cats_train_d], axis=1)
    design_test = pd.concat([x_test, cats_test_d], axis=1)
    mean = design_train.mean(axis=0)
    std = design_train.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    design_train = (design_train - mean) / std
    design_test = (design_test - mean) / std
    meta = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "design_cols": list(design_train.columns),
        "median": {k: float(v) for k, v in med.to_dict().items()},
        "mean": {k: float(v) for k, v in mean.to_dict().items()},
        "std": {k: float(v) for k, v in std.to_dict().items()},
    }
    return design_train, design_test, meta


def fit_ridge_predict(
    x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, alpha: float = 12.0
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mask = pd.to_numeric(y_train, errors="coerce").notna()
    xt = x_train.loc[mask].to_numpy(dtype=float)
    yt = pd.to_numeric(y_train.loc[mask], errors="coerce").to_numpy(dtype=float)
    xt_i = np.column_stack([np.ones(len(xt)), xt])
    reg = np.eye(xt_i.shape[1]) * float(alpha)
    reg[0, 0] = 0.0
    beta = np.linalg.pinv(xt_i.T @ xt_i + reg) @ xt_i.T @ yt
    train_pred = np.column_stack([np.ones(len(x_train)), x_train.to_numpy(dtype=float)]) @ beta
    test_pred = np.column_stack([np.ones(len(x_test)), x_test.to_numpy(dtype=float)]) @ beta
    return train_pred, test_pred, {"alpha": alpha, "n_train": int(mask.sum()), "coef_norm": float(np.linalg.norm(beta[1:]))}


def choose_blend_weight(
    train: pd.DataFrame,
    x_train: pd.DataFrame,
    target: str,
    baseline_col: str | None,
    alpha: float,
) -> tuple[float, dict[str, Any]]:
    if not baseline_col or baseline_col not in train.columns or len(train) < 200:
        return 1.0, {"reason": "no_baseline_or_small_train"}
    order = train["race_id"].astype(str).sort_values().index
    split = max(50, int(len(order) * 0.78))
    fit_idx = order[:split]
    val_idx = order[split:]
    if len(val_idx) < 50:
        return 1.0, {"reason": "small_validation"}
    _fit_pred, val_raw, _meta = fit_ridge_predict(
        x_train.loc[fit_idx],
        train.loc[fit_idx, target],
        x_train.loc[val_idx],
        alpha=alpha,
    )
    actual = pd.to_numeric(train.loc[val_idx, target], errors="coerce")
    baseline = pd.to_numeric(train.loc[val_idx, baseline_col], errors="coerce")
    raw = pd.Series(val_raw, index=val_idx)
    mask = actual.notna() & baseline.notna() & raw.notna()
    if mask.sum() < 50:
        return 1.0, {"reason": "insufficient_validation_baseline"}
    best_weight = 1.0
    best_mae = float("inf")
    rows = []
    for weight in np.linspace(0.0, 1.25, 26):
        pred = baseline[mask] + weight * (raw[mask] - baseline[mask])
        mae = float((pred - actual[mask]).abs().mean())
        rows.append({"weight": float(weight), "mae": mae})
        if mae < best_mae:
            best_mae = mae
            best_weight = float(weight)
    baseline_mae = float((baseline[mask] - actual[mask]).abs().mean())
    raw_mae = float((raw[mask] - actual[mask]).abs().mean())
    return best_weight, {
        "reason": "validation_mae_grid",
        "validation_rows": int(mask.sum()),
        "baseline_mae": baseline_mae,
        "raw_model_mae": raw_mae,
        "blend_mae": best_mae,
        "grid": rows,
    }


def apply_blend(
    frame: pd.DataFrame,
    raw_pred: np.ndarray,
    baseline_col: str | None,
    weight: float,
) -> np.ndarray:
    raw = pd.Series(raw_pred, index=frame.index, dtype=float)
    if not baseline_col or baseline_col not in frame.columns:
        return raw.to_numpy(dtype=float)
    baseline = pd.to_numeric(frame[baseline_col], errors="coerce")
    blended = baseline + weight * (raw - baseline)
    return blended.fillna(raw).to_numpy(dtype=float)


def metric_row(frame: pd.DataFrame, target: str, pred_col: str, baseline_col: str | None) -> dict[str, Any]:
    actual = pd.to_numeric(frame[target], errors="coerce")
    pred = pd.to_numeric(frame[pred_col], errors="coerce")
    mask = actual.notna() & pred.notna()
    row: dict[str, Any] = {
        "target": target,
        "rows": int(mask.sum()),
        "model_mae": float((pred[mask] - actual[mask]).abs().mean()) if mask.any() else np.nan,
        "model_rmse": float(np.sqrt(((pred[mask] - actual[mask]) ** 2).mean())) if mask.any() else np.nan,
        "model_bias": float((pred[mask] - actual[mask]).mean()) if mask.any() else np.nan,
        "model_corr": float(pred[mask].corr(actual[mask])) if mask.sum() >= 3 else np.nan,
    }
    if baseline_col and baseline_col in frame.columns:
        base = pd.to_numeric(frame[baseline_col], errors="coerce")
        bmask = mask & base.notna()
        row.update(
            {
                "baseline_col": baseline_col,
                "baseline_mae": float((base[bmask] - actual[bmask]).abs().mean()) if bmask.any() else np.nan,
                "baseline_rmse": float(np.sqrt(((base[bmask] - actual[bmask]) ** 2).mean())) if bmask.any() else np.nan,
                "mae_improvement": float(
                    ((base[bmask] - actual[bmask]).abs().mean() - (pred[bmask] - actual[bmask]).abs().mean())
                )
                if bmask.any()
                else np.nan,
            }
        )
    return row


def add_continuous_mode(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    idx = out.index
    pred_front = num(out.get("pred_front3f_sec"), idx)
    prior_front = num(out.get("course_front3f_prior_sec"), idx)
    prior_std = num(out.get("course_front3f_prior_std"), idx, 0.8).fillna(0.8).clip(0.35, 2.5)
    pred_rpci = num(out.get("pred_rpci"), idx, 50.0).fillna(50.0)
    pred_pci3 = num(out.get("pred_pci3"), idx, 50.0).fillna(50.0)
    pressure = num(out.get("race_est_ten_pressure_score"), idx, 0.5).fillna(0.5).clip(0.0, 1.5)
    clarity = num(out.get("race_est_queue_clarity_score"), idx, 0.5).fillna(0.5).clip(0.0, 1.0)
    duel = num(out.get("queue_duel_risk_score"), idx, 0.5).fillna(0.5).clip(0.0, 1.0)
    front_delta_z = ((prior_front - pred_front) / prior_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["cont_front_delta_z"] = front_delta_z
    out["cont_rpci_delta"] = pred_rpci - num(out.get("course_base_rpci_prior"), idx, 50.0).fillna(50.0)
    fast_raw = sigmoid(front_delta_z) + sigmoid((50.0 - pred_rpci) / 2.8) + 0.35 * pressure + 0.20 * duel
    slow_raw = sigmoid(-front_delta_z) + sigmoid((pred_rpci - 52.0) / 2.6) + 0.35 * clarity + 0.15 * (1.0 - pressure.clip(0.0, 1.0))
    instant_raw = sigmoid((pred_pci3 - 52.0) / 2.4) + sigmoid((pred_rpci - 52.5) / 2.4) + 0.30 * clarity
    sustain_raw = (
        sigmoid((51.5 - (pred_rpci - 49.0).abs()) / 2.8)
        + 0.45 * pressure
        + 0.35 * duel
        + 0.20 * (1.0 - clarity)
    )
    raw = pd.DataFrame(
        {
            "fast": fast_raw,
            "slow": slow_raw,
            "instant": instant_raw,
            "sustain": sustain_raw,
        },
        index=idx,
    ).clip(lower=0.0)
    probs = raw.div(raw.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.25)
    for cls in LAP_CLASSES:
        out[f"cont_prob_{cls}"] = probs[cls]
    arr = probs[LAP_CLASSES].to_numpy()
    best = arr.argmax(axis=1)
    sorted_probs = np.sort(arr, axis=1)
    out["cont_predicted_lap_mode"] = [LAP_CLASSES[i] for i in best]
    out["cont_confidence"] = arr.max(axis=1)
    out["cont_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    out["cont_agrees_v2"] = out["cont_predicted_lap_mode"].astype(str).eq(out.get("v2_predicted_lap_mode", "").astype(str))
    return out


def ticket_key(frame: pd.DataFrame) -> pd.Series:
    a = num(frame.get("anchor_no"), frame.index).fillna(-1).astype(int).astype(str)
    b = num(frame.get("partner_no"), frame.index).fillna(-1).astype(int).astype(str)
    typ = frame.get("ticket_type", pd.Series("", index=frame.index)).astype(str)
    return frame["race_id"].astype(str) + ":" + a + "-" + b + ":" + typ


def ticket_metrics(frame: pd.DataFrame, segment: str, policy: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "policy": policy,
            "segment": segment,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "roi": np.nan,
            "hit_rate": np.nan,
        }
    stake = num(frame.get("runtime_stake_yen"), frame.index).fillna(num(frame.get("stake_yen"), frame.index)).fillna(0.0)
    ret = num(frame.get("runtime_return_yen"), frame.index).fillna(num(frame.get("return_yen"), frame.index)).fillna(0.0)
    stake_sum = float(stake.sum())
    ret_sum = float(ret.sum())
    if ret_sum > 0 and len(frame) > 1:
        top_idx = int(ret.to_numpy().argmax())
        roi_ex_top = safe_div(ret_sum - float(ret.iloc[top_idx]), stake_sum - float(stake.iloc[top_idx]))
        top_share = float(ret.max() / ret_sum)
    else:
        roi_ex_top = np.nan
        top_share = np.nan
    return {
        "policy": policy,
        "segment": segment,
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()),
        "stake_yen": stake_sum,
        "return_yen": ret_sum,
        "profit_yen": ret_sum - stake_sum,
        "roi": safe_div(ret_sum, stake_sum),
        "hit_rate": float(ret.gt(0).mean()),
        "roi_ex_top1_return": roi_ex_top,
        "top_return_share": top_share,
        "cont_mode_agree_rate": float(frame.get("cont_agrees_v2", pd.Series(False, index=frame.index)).mean()),
        "avg_cont_confidence": float(num(frame.get("cont_confidence"), frame.index).mean()),
        "avg_v2_confidence": float(num(frame.get("v2_confidence"), frame.index).mean()),
    }


def build_ticket_summary(tickets_path: Path, predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tickets = read_csv(tickets_path)
    tickets["race_id"] = normalize_race_id(tickets["race_id"])
    keep = [
        "race_id",
        "year",
        "surface",
        "pred_front3f_sec",
        "pred_rpci",
        "pred_pci3",
        "cont_front_delta_z",
        "cont_rpci_delta",
        "cont_predicted_lap_mode",
        "cont_confidence",
        "cont_margin",
        "cont_agrees_v2",
        "v2_predicted_lap_mode",
        "v2_confidence",
        "v2_margin",
        "actual_lap_mode",
        "front3f_sec",
        "rpci",
        "pci3",
    ]
    merged = tickets.merge(
        predictions[[c for c in keep if c in predictions.columns]].drop_duplicates("race_id"),
        on="race_id",
        how="left",
    ).copy()
    if "year" not in merged.columns:
        left_year = merged.get("year_x", pd.Series(np.nan, index=merged.index))
        right_year = merged.get("year_y", pd.Series(np.nan, index=merged.index))
        merged["year"] = left_year.combine_first(right_year).fillna(merged["race_id"].astype(str).str[:4])
    if "surface" not in merged.columns:
        left_surface = merged.get("surface_x", pd.Series(np.nan, index=merged.index))
        right_surface = merged.get("surface_y", pd.Series(np.nan, index=merged.index))
        merged["surface"] = right_surface.combine_first(left_surface)
    cont_conf = num(merged.get("cont_confidence"), merged.index)
    cont_margin = num(merged.get("cont_margin"), merged.index)
    v2_conf = num(merged.get("v2_confidence"), merged.index)
    v2_margin = num(merged.get("v2_margin"), merged.index)
    front_delta_abs = num(merged.get("cont_front_delta_z"), merged.index).abs()
    thresholds = {
        "cont_conf_q50": float(cont_conf.quantile(0.50)),
        "cont_margin_q50": float(cont_margin.quantile(0.50)),
        "v2_conf_q20": float(v2_conf.quantile(0.20)),
        "v2_margin_q30": float(v2_margin.quantile(0.30)),
        "front_delta_abs_q60": float(front_delta_abs.quantile(0.60)),
    }
    masks = {
        "base_all": pd.Series(True, index=merged.index),
        "continuous_conf_q50": cont_conf.ge(thresholds["cont_conf_q50"]),
        "continuous_agrees_v2": merged.get("cont_agrees_v2", pd.Series(False, index=merged.index)).fillna(False).astype(bool),
        "continuous_agrees_v2_conf": merged.get("cont_agrees_v2", pd.Series(False, index=merged.index)).fillna(False).astype(bool)
        & cont_conf.ge(thresholds["cont_conf_q50"])
        & v2_conf.ge(thresholds["v2_conf_q20"]),
        "continuous_clear_shape": cont_margin.ge(thresholds["cont_margin_q50"])
        & front_delta_abs.ge(thresholds["front_delta_abs_q60"]),
        "continuous_combo_strict": merged.get("cont_agrees_v2", pd.Series(False, index=merged.index)).fillna(False).astype(bool)
        & cont_conf.ge(thresholds["cont_conf_q50"])
        & cont_margin.ge(thresholds["cont_margin_q50"])
        & v2_conf.ge(thresholds["v2_conf_q20"])
        & v2_margin.ge(thresholds["v2_margin_q30"]),
    }
    overall_rows = []
    year_rows = []
    surface_rows = []
    for policy, mask in masks.items():
        sub = merged.loc[mask.fillna(False)].copy()
        overall_rows.append(ticket_metrics(sub, "ALL", policy))
        if "year" in sub.columns:
            for year, g in sub.groupby("year", dropna=False):
                year_rows.append(ticket_metrics(g, str(year), policy))
        if "surface" in sub.columns:
            for surface, g in sub.groupby("surface", dropna=False):
                surface_rows.append(ticket_metrics(g, str(surface), policy))
    return {
        "ticket_policy_overall": pd.DataFrame(overall_rows).sort_values("roi", ascending=False),
        "ticket_policy_by_year": pd.DataFrame(year_rows).sort_values(["policy", "segment"]) if year_rows else pd.DataFrame(),
        "ticket_policy_by_surface": pd.DataFrame(surface_rows).sort_values(["policy", "segment"]) if surface_rows else pd.DataFrame(),
        "ticket_predictions_joined": merged,
        "thresholds": pd.DataFrame([thresholds]),
    }


def to_md_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "(empty)"
    view = frame.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_report(out_dir: Path, outputs: dict[str, pd.DataFrame], summary: dict[str, Any]) -> None:
    lines = [
        "# Continuous Race Pace Prediction v1",
        "",
        "## Target Accuracy",
        to_md_table(outputs["metric_summary"]),
        "",
        "## Ticket Policy Overall",
        to_md_table(outputs["ticket_policy_overall"]),
        "",
        "## Ticket Policy By Year",
        to_md_table(outputs["ticket_policy_by_year"], max_rows=40),
        "",
        "## Ticket Policy By Surface",
        to_md_table(outputs["ticket_policy_by_surface"], max_rows=40),
        "",
        "## Notes",
        "- This is shadow-only. It predicts continuous front3F/RPCI/PCI3 and checks whether the signals can filter existing S-priority tickets.",
        "- Do not use actual target-race RPCI/PCI3/front3F as betting inputs; they are evaluation targets only.",
        "- A higher ROI here means the continuous pace read may help gate or annotate tickets, not that it is ready to replace the current strongest BUY logic.",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/evaluate continuous race pace prediction for front3F/RPCI/PCI3.")
    parser.add_argument("--diag", default=str(DEFAULT_DIAG))
    parser.add_argument("--tickets", default=str(DEFAULT_TICKETS))
    parser.add_argument("--expected-lap", default=str(DEFAULT_EXPECTED_LAP))
    parser.add_argument("--front3f-cache", default=str(DEFAULT_FRONT3F_CACHE))
    parser.add_argument("--front3f-diag", default=str(DEFAULT_FRONT3F_DIAG))
    parser.add_argument("--course-ten", default=str(DEFAULT_COURSE_TEN))
    parser.add_argument("--feature-source", action="append", default=None)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--alpha", type=float, default=12.0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base = build_base_frame(args)
    valid = base[base["source"].isin(["train", "test"])].copy()
    train = valid[valid["source"].eq("train")].copy()
    test = valid[valid["source"].eq("test")].copy()
    numeric_cols, categorical_cols = feature_columns(valid)
    x_train, x_test, design_meta = make_design(train, test, numeric_cols, categorical_cols)

    predictions = valid.copy()
    model_meta: dict[str, Any] = {"design": design_meta, "targets": {}}
    metric_rows = []
    baselines = {
        "front3f_sec": "course_front3f_prior_sec",
        "rpci": "course_base_rpci_prior",
        "pci3": "course_base_pci3_prior",
    }
    for target in TARGETS:
        baseline_col = baselines.get(target)
        blend_weight, blend_meta = choose_blend_weight(train, x_train, target, baseline_col, alpha=args.alpha)
        train_raw, test_raw, meta = fit_ridge_predict(x_train, train[target], x_test, alpha=args.alpha)
        train_pred = apply_blend(train, train_raw, baseline_col, blend_weight)
        test_pred = apply_blend(test, test_raw, baseline_col, blend_weight)
        predictions.loc[train.index, f"raw_pred_{target}"] = train_raw
        predictions.loc[test.index, f"raw_pred_{target}"] = test_raw
        predictions.loc[train.index, f"pred_{target}"] = train_pred
        predictions.loc[test.index, f"pred_{target}"] = test_pred
        model_meta["targets"][target] = meta
        model_meta["targets"][target]["blend_weight"] = blend_weight
        model_meta["targets"][target]["blend_validation"] = blend_meta
        metric = metric_row(predictions.loc[test.index], target, f"pred_{target}", baseline_col)
        metric["blend_weight"] = blend_weight
        metric_rows.append(metric)

    predictions = add_continuous_mode(predictions)
    ticket_outputs = build_ticket_summary(Path(args.tickets), predictions)
    outputs: dict[str, pd.DataFrame] = {
        "metric_summary": pd.DataFrame(metric_rows),
        "continuous_pace_predictions": predictions,
        **{k: v for k, v in ticket_outputs.items() if isinstance(v, pd.DataFrame)},
    }

    for name, frame in outputs.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    summary = {
        "output_dir": str(out_dir),
        "train_races": int(len(train)),
        "test_races": int(len(test)),
        "numeric_features": len(numeric_cols),
        "categorical_features": categorical_cols,
        "target_metrics": outputs["metric_summary"].replace({np.nan: None}).to_dict(orient="records"),
        "best_ticket_policy": outputs["ticket_policy_overall"].head(1).replace({np.nan: None}).to_dict(orient="records"),
    }
    (out_dir / "model_params.json").write_text(json.dumps(model_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir, outputs, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
