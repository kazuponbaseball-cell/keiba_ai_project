from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNNERS = ROOT / "outputs/analysis/lap_pair_refinement_candidates_v1/runner_lap_pair_refinement_features.csv"
DEFAULT_TICKETS = ROOT / "outputs/analysis/lap_pair_refinement_candidates_v1/s_priority_tickets_with_lap_pair_refinement.csv"
DEFAULT_QUEUE = ROOT / "outputs/analysis/queue_shape_race_quality_v1/race_queue_shape_validation.csv"
DEFAULT_COURSE_TEN = ROOT / "outputs/analysis/course_adjusted_front3f_signal_v1/course_adjusted_front3f_race_diagnostics.csv"
DEFAULT_OUT = ROOT / "outputs/analysis/race_quality_prediction_v2"

LAP_CLASSES = ["fast", "slow", "instant", "sustain"]
V1_MODE_COLS = ["race_need_fast", "race_need_slow", "race_need_instant", "race_need_sustain", "race_need_long_spurt"]
HORSE_MODE_COLS = ["horse_lap_fast", "horse_lap_slow", "horse_lap_instant", "horse_lap_sustain", "horse_lap_long_spurt"]


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)


def num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def normalize_rows(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    total = values.sum(axis=1).replace(0.0, np.nan)
    return values.div(total, axis=0).fillna(1.0 / len(frame.columns)).clip(0.0, 1.0)


def mode_or_first(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    values = values[values.ne("")]
    if values.empty:
        return "unknown"
    mode = values.mode()
    return str(mode.iloc[0]) if not mode.empty else str(values.iloc[0])


def map_prediction(label: Any) -> str:
    value = str(label)
    if value == "long_spurt":
        return "sustain"
    if value in LAP_CLASSES:
        return value
    return "unknown"


def softmax_neg_distance(distances: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = -distances / max(float(temperature), 1e-6)
    scaled = scaled - np.nanmax(scaled, axis=1, keepdims=True)
    exp = np.exp(np.clip(scaled, -60, 60))
    denom = exp.sum(axis=1, keepdims=True)
    return exp / np.where(denom == 0, 1.0, denom)


def build_race_frame(runners_path: Path, queue_path: Path, course_ten_path: Path) -> pd.DataFrame:
    runners = read_csv(runners_path)
    runners["race_id"] = runners["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    rows: list[dict[str, Any]] = []
    need_cols = [c for c in V1_MODE_COLS if c in runners.columns]
    horse_cols = [c for c in HORSE_MODE_COLS if c in runners.columns]
    for (source, race_id), group in runners.groupby(["source", "race_id"], sort=False):
        need_mean = group[need_cols].apply(pd.to_numeric, errors="coerce").mean(axis=0) if need_cols else pd.Series(dtype=float)
        horse_mean = group[horse_cols].apply(pd.to_numeric, errors="coerce").mean(axis=0) if horse_cols else pd.Series(dtype=float)
        horse_max = group[horse_cols].apply(pd.to_numeric, errors="coerce").max(axis=0) if horse_cols else pd.Series(dtype=float)
        horse_std = group[horse_cols].apply(pd.to_numeric, errors="coerce").std(axis=0) if horse_cols else pd.Series(dtype=float)
        v1_raw = str(need_mean.idxmax()).replace("race_need_", "") if not need_mean.empty else "unknown"
        rows.append(
            {
                "source": str(source),
                "race_id": str(race_id),
                "actual_lap_mode": mode_or_first(group.get("actual_lap_mode_diagnostic", pd.Series(dtype=object))),
                "v1_predicted_lap_mode_raw": v1_raw,
                "v1_predicted_lap_mode": map_prediction(v1_raw),
                "field_size_lap_rows": int(len(group)),
                "v1_confidence": float(num(group.get("race_lap_prediction_confidence"), group.index).mean()),
                "v1_concentration": float(num(group.get("race_lap_profile_concentration"), group.index).mean()),
                **{f"avg_{c}": float(need_mean.get(c, np.nan)) for c in need_cols},
                **{f"field_mean_{c}": float(horse_mean.get(c, np.nan)) for c in horse_cols},
                **{f"field_max_{c}": float(horse_max.get(c, np.nan)) for c in horse_cols},
                **{f"field_std_{c}": float(horse_std.get(c, np.nan)) for c in horse_cols},
            }
        )
    race = pd.DataFrame(rows)

    if queue_path.exists():
        queue = read_csv(queue_path)
        queue["race_id"] = queue["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        keep = [
            "race_id",
            "queue_shape_label",
            "queue_clarity_score",
            "queue_duel_risk_score",
            "queue_front_load_score",
            "queue_top_gap",
            "queue_candidate_count",
        ]
        race = race.merge(queue[[c for c in keep if c in queue.columns]].drop_duplicates("race_id"), on="race_id", how="left")

    if course_ten_path.exists():
        course = read_csv(course_ten_path)
        if "レースID(新/馬番無)" in course.columns:
            course = course.rename(columns={"レースID(新/馬番無)": "race_id"})
        course["race_id"] = course["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        keep = [
            "race_id",
            "course_front3f_prior_sec",
            "course_front3f_prior_std",
            "course_front3f_prior_count",
            "race_course_adj_ten_pressure_score",
            "race_course_adj_fast_start_count",
            "race_course_adj_ten_speed_gap_top2",
            "race_course_adj_queue_clarity_score",
        ]
        race = race.merge(course[[c for c in keep if c in course.columns]].drop_duplicates("race_id"), on="race_id", how="left")

    # Hand-built pre-race shape signals that are intentionally broad, not final betting rules.
    idx = race.index
    duel = num(race.get("queue_duel_risk_score"), idx, 0.5).fillna(0.5).clip(0.0, 1.0)
    clarity = num(race.get("queue_clarity_score"), idx, 0.5).fillna(0.5).clip(0.0, 1.0)
    front_load = num(race.get("queue_front_load_score"), idx, 0.5).fillna(0.5).clip(0.0, 1.0)
    course_pressure = num(race.get("race_course_adj_ten_pressure_score"), idx, 0.5).fillna(0.5).clip(0.0, 1.0)
    course_clarity = num(race.get("race_course_adj_queue_clarity_score"), idx, 0.5).fillna(0.5).clip(0.0, 1.0)
    fast_count = num(race.get("race_course_adj_fast_start_count"), idx, 0.0).fillna(0.0).clip(0.0, 8.0) / 8.0
    ten_gap = num(race.get("race_course_adj_ten_speed_gap_top2"), idx, 0.0).fillna(0.0).clip(0.0, 1.0)

    race["shape_fast_signal"] = (0.34 * duel + 0.26 * front_load + 0.22 * course_pressure + 0.18 * fast_count).clip(0.0, 1.0)
    race["shape_slow_signal"] = (0.34 * clarity + 0.26 * course_clarity + 0.22 * ten_gap + 0.18 * (1.0 - duel)).clip(0.0, 1.0)
    race["shape_sustain_signal"] = (0.34 * front_load + 0.24 * duel + 0.22 * course_pressure + 0.20 * (1.0 - (clarity - 0.5).abs() * 2.0)).clip(0.0, 1.0)
    race["shape_instant_signal"] = (0.38 * race["shape_slow_signal"] + 0.24 * (1.0 - front_load) + 0.20 * clarity + 0.18 * (1.0 - course_pressure)).clip(0.0, 1.0)
    race["shape_uncertainty_signal"] = (1.0 - (race["shape_fast_signal"] - race["shape_slow_signal"]).abs()).clip(0.0, 1.0)

    return race


def feature_columns(race: pd.DataFrame) -> list[str]:
    candidates = [
        c
        for c in race.columns
        if c.startswith("avg_race_need_")
        or c.startswith("field_mean_horse_lap_")
        or c.startswith("field_max_horse_lap_")
        or c.startswith("field_std_horse_lap_")
        or c
        in {
            "field_size_lap_rows",
            "v1_confidence",
            "v1_concentration",
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
            "shape_fast_signal",
            "shape_slow_signal",
            "shape_sustain_signal",
            "shape_instant_signal",
            "shape_uncertainty_signal",
        }
    ]
    return [c for c in candidates if c in race.columns]


def fit_centroid_classifier(train: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    valid = train[train["actual_lap_mode"].isin(LAP_CLASSES)].copy()
    x = valid[features].apply(pd.to_numeric, errors="coerce")
    med = x.median(axis=0).fillna(0.0)
    x = x.fillna(med)
    mean = x.mean(axis=0)
    std = x.std(axis=0, ddof=0).replace(0.0, 1.0).fillna(1.0)
    z = (x - mean) / std
    centroids = {}
    priors = {}
    for klass in LAP_CLASSES:
        mask = valid["actual_lap_mode"].eq(klass)
        if mask.any():
            centroids[klass] = z.loc[mask].mean(axis=0)
            priors[klass] = float(mask.mean())
        else:
            centroids[klass] = pd.Series(0.0, index=features)
            priors[klass] = 0.0
    return {"features": features, "med": med, "mean": mean, "std": std, "centroids": centroids, "priors": priors}


def serializable_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "classes": LAP_CLASSES,
        "features": list(model["features"]),
        "med": {k: float(v) for k, v in model["med"].to_dict().items()},
        "mean": {k: float(v) for k, v in model["mean"].to_dict().items()},
        "std": {k: float(v) for k, v in model["std"].to_dict().items()},
        "centroids": {
            klass: {k: float(v) for k, v in centroid.to_dict().items()}
            for klass, centroid in model["centroids"].items()
        },
        "priors": {k: float(v) for k, v in model["priors"].items()},
        "long_spurt_mapping": "sustain",
        "version": "race_quality_prediction_v2_centroid_20260629",
    }


def predict_centroid(model: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    features = model["features"]
    x = frame[features].apply(pd.to_numeric, errors="coerce").fillna(model["med"])
    z = (x - model["mean"]) / model["std"]
    dists = []
    for klass in LAP_CLASSES:
        centroid = model["centroids"][klass]
        diff = z - centroid
        dist = (diff * diff).mean(axis=1).to_numpy(dtype=float)
        # Mild prior: enough to avoid impossible classes, small enough to let features move.
        prior = max(float(model["priors"].get(klass, 0.0)), 1e-4)
        dists.append(dist - 0.06 * np.log(prior))
    dist_mat = np.vstack(dists).T
    probs = softmax_neg_distance(dist_mat, temperature=0.85)
    out = pd.DataFrame(probs, columns=[f"centroid_prob_{c}" for c in LAP_CLASSES], index=frame.index)
    best = probs.argmax(axis=1)
    sorted_probs = np.sort(probs, axis=1)
    out["centroid_predicted_lap_mode"] = [LAP_CLASSES[i] for i in best]
    out["centroid_confidence"] = probs.max(axis=1)
    out["centroid_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    return out


def build_v1_probs(frame: pd.DataFrame) -> pd.DataFrame:
    raw = pd.DataFrame(index=frame.index)
    raw["fast"] = num(frame.get("avg_race_need_fast"), frame.index, 0.0).fillna(0.0)
    raw["slow"] = num(frame.get("avg_race_need_slow"), frame.index, 0.0).fillna(0.0)
    raw["instant"] = num(frame.get("avg_race_need_instant"), frame.index, 0.0).fillna(0.0)
    raw["sustain"] = (
        num(frame.get("avg_race_need_sustain"), frame.index, 0.0).fillna(0.0)
        + 0.72 * num(frame.get("avg_race_need_long_spurt"), frame.index, 0.0).fillna(0.0)
    )
    probs = normalize_rows(raw)
    return probs.rename(columns={c: f"v1_prob_{c}" for c in LAP_CLASSES})


def add_predictions(race: pd.DataFrame, model: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = race.copy()
    centroid = predict_centroid(model, out)
    v1_probs = build_v1_probs(out)
    out = pd.concat([out, centroid, v1_probs], axis=1)

    train_mask = out["source"].eq("train") & out["actual_lap_mode"].isin(LAP_CLASSES)
    best_blend = 0.0
    best_hit = -1.0
    blend_rows: list[dict[str, float]] = []
    for blend in np.linspace(0.0, 1.0, 21):
        hybrid = pd.DataFrame(index=out.index)
        for klass in LAP_CLASSES:
            hybrid[klass] = blend * out[f"centroid_prob_{klass}"] + (1.0 - blend) * out[f"v1_prob_{klass}"]
        pred = hybrid.idxmax(axis=1)
        hit = float(pred.loc[train_mask].eq(out.loc[train_mask, "actual_lap_mode"]).mean())
        blend_rows.append({"blend_centroid_weight": float(blend), "train_hit_rate": hit})
        if hit > best_hit:
            best_hit = hit
            best_blend = float(blend)

    hybrid = pd.DataFrame(index=out.index)
    for klass in LAP_CLASSES:
        hybrid[f"v2_prob_{klass}"] = best_blend * out[f"centroid_prob_{klass}"] + (1.0 - best_blend) * out[f"v1_prob_{klass}"]
    out = pd.concat([out, hybrid], axis=1)
    prob_cols = [f"v2_prob_{c}" for c in LAP_CLASSES]
    arr = out[prob_cols].to_numpy(dtype=float)
    best = arr.argmax(axis=1)
    sorted_probs = np.sort(arr, axis=1)
    out["v2_predicted_lap_mode"] = [LAP_CLASSES[i] for i in best]
    out["v2_confidence"] = arr.max(axis=1)
    out["v2_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
    out["v2_v1_agree"] = out["v2_predicted_lap_mode"].eq(out["v1_predicted_lap_mode"])
    out["v1_hit"] = out["v1_predicted_lap_mode"].eq(out["actual_lap_mode"])
    out["centroid_hit"] = out["centroid_predicted_lap_mode"].eq(out["actual_lap_mode"])
    out["v2_hit"] = out["v2_predicted_lap_mode"].eq(out["actual_lap_mode"])
    return out, {"best_blend_centroid_weight": best_blend, "blend_grid": blend_rows}


def summarize_prediction(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid = frame[frame["actual_lap_mode"].isin(LAP_CLASSES)].copy()
    for source in ["train", "test", "ALL"]:
        part = valid if source == "ALL" else valid[valid["source"].eq(source)]
        if part.empty:
            continue
        for name, hit_col, conf_col in [
            ("v1_rule", "v1_hit", "v1_confidence"),
            ("centroid", "centroid_hit", "centroid_confidence"),
            ("v2_hybrid", "v2_hit", "v2_confidence"),
        ]:
            rows.append(
                {
                    "source": source,
                    "model": name,
                    "races": int(len(part)),
                    "hit_rate": float(part[hit_col].mean()),
                    "avg_confidence": float(num(part.get(conf_col), part.index).mean()),
                    "sustain_recall": float(part.loc[part["actual_lap_mode"].eq("sustain"), hit_col].mean())
                    if part["actual_lap_mode"].eq("sustain").any()
                    else np.nan,
                    "fast_recall": float(part.loc[part["actual_lap_mode"].eq("fast"), hit_col].mean())
                    if part["actual_lap_mode"].eq("fast").any()
                    else np.nan,
                    "slow_recall": float(part.loc[part["actual_lap_mode"].eq("slow"), hit_col].mean())
                    if part["actual_lap_mode"].eq("slow").any()
                    else np.nan,
                    "instant_recall": float(part.loc[part["actual_lap_mode"].eq("instant"), hit_col].mean())
                    if part["actual_lap_mode"].eq("instant").any()
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def confusion(frame: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    valid = frame[frame["actual_lap_mode"].isin(LAP_CLASSES)].copy()
    return (
        valid.pivot_table(index="actual_lap_mode", columns=pred_col, values="race_id", aggfunc="count", fill_value=0)
        .reset_index()
    )


def confidence_bins(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[frame["source"].eq("test") & frame["actual_lap_mode"].isin(LAP_CLASSES)].copy()
    valid["confidence_bin"] = pd.qcut(
        valid["v2_confidence"].rank(method="first"), q=5, labels=["q1_low", "q2", "q3", "q4", "q5_high"]
    )
    return (
        valid.groupby("confidence_bin", observed=False)
        .agg(
            races=("race_id", "count"),
            hit_rate=("v2_hit", "mean"),
            avg_confidence=("v2_confidence", "mean"),
            avg_margin=("v2_margin", "mean"),
        )
        .reset_index()
    )


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
        }
    stake = num(frame.get("runtime_stake_yen"), frame.index).fillna(num(frame.get("stake_yen"), frame.index)).fillna(0.0)
    ret = num(frame.get("runtime_return_yen"), frame.index).fillna(num(frame.get("return_yen"), frame.index)).fillna(0.0)
    stake_sum = float(stake.sum())
    ret_sum = float(ret.sum())
    return {
        "policy": policy,
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()),
        "stake_yen": stake_sum,
        "return_yen": ret_sum,
        "profit_yen": ret_sum - stake_sum,
        "roi": safe_div(ret_sum, stake_sum),
        "hit_rate": float(ret.gt(0).mean()),
        "avg_v2_confidence": float(num(frame.get("v2_confidence"), frame.index).mean()),
        "avg_v2_margin": float(num(frame.get("v2_margin"), frame.index).mean()),
    }


def evaluate_ticket_filters(tickets_path: Path, race_diag: pd.DataFrame) -> pd.DataFrame:
    tickets = read_csv(tickets_path)
    if tickets.empty:
        return pd.DataFrame()
    tickets["race_id"] = tickets["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    keep = [
        "race_id",
        "v1_predicted_lap_mode",
        "centroid_predicted_lap_mode",
        "v2_predicted_lap_mode",
        "v2_confidence",
        "v2_margin",
        "v2_v1_agree",
        "shape_fast_signal",
        "shape_slow_signal",
        "shape_sustain_signal",
    ]
    out = tickets.merge(race_diag[[c for c in keep if c in race_diag.columns]].drop_duplicates("race_id"), on="race_id", how="left")

    lap_fit = num(out.get("pair_lap_same_race_fit_score"), out.index)
    lap_conf = num(out.get("pair_lap_race_confidence"), out.index)
    lap_contra = num(out.get("pair_lap_contradiction_score"), out.index)
    v2_conf = num(out.get("v2_confidence"), out.index)
    v2_margin = num(out.get("v2_margin"), out.index)

    thresholds = {
        "lap_fit_q30": float(lap_fit.quantile(0.30)),
        "lap_conf_q40": float(lap_conf.quantile(0.40)),
        "lap_contra_q80": float(lap_contra.quantile(0.80)),
        "v2_conf_q20": float(v2_conf.quantile(0.20)),
        "v2_conf_q40": float(v2_conf.quantile(0.40)),
        "v2_margin_q30": float(v2_margin.quantile(0.30)),
    }
    masks = {
        "base_all": pd.Series(True, index=out.index),
        "v2_conf_q20": v2_conf.ge(thresholds["v2_conf_q20"]),
        "v2_conf_q40": v2_conf.ge(thresholds["v2_conf_q40"]),
        "v2_margin_q30": v2_margin.ge(thresholds["v2_margin_q30"]),
        "v2_agree_v1": out.get("v2_v1_agree", pd.Series(False, index=out.index)).fillna(False).astype(bool),
        "lap_fit_q30": lap_fit.ge(thresholds["lap_fit_q30"]),
        "lap_fit_q30_v2_conf_q20": lap_fit.ge(thresholds["lap_fit_q30"]) & v2_conf.ge(thresholds["v2_conf_q20"]),
        "lap_combo_v2_conf_q20": lap_fit.ge(thresholds["lap_fit_q30"])
        & lap_conf.ge(thresholds["lap_conf_q40"])
        & lap_contra.le(thresholds["lap_contra_q80"])
        & v2_conf.ge(thresholds["v2_conf_q20"]),
        "lap_combo_v2_margin_q30": lap_fit.ge(thresholds["lap_fit_q30"])
        & lap_conf.ge(thresholds["lap_conf_q40"])
        & lap_contra.le(thresholds["lap_contra_q80"])
        & v2_margin.ge(thresholds["v2_margin_q30"]),
    }
    rows = [ticket_metrics(out.loc[mask.fillna(False)], name) for name, mask in masks.items()]
    metrics = pd.DataFrame(rows)
    metrics.attrs["thresholds"] = thresholds
    return metrics


def write_readme(out_dir: Path, summary: pd.DataFrame, ticket_summary: pd.DataFrame, config: dict[str, Any]) -> None:
    def table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "(empty)"
        view = frame.copy()
        for col in view.columns:
            if pd.api.types.is_float_dtype(view[col]):
                view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        cols = list(view.columns)
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in view.iterrows():
            lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
        return "\n".join(lines)

    lines = [
        "# Race Quality Prediction v2",
        "",
        "Purpose: improve pre-race pace/lap-shape diagnosis without touching production BUY gates.",
        "",
        "## Prediction Accuracy",
        table(summary),
        "",
        "## Ticket Filter Backtest",
        table(ticket_summary) if not ticket_summary.empty else "(no ticket summary)",
        "",
        "## Notes",
        f"- Best train-selected blend centroid weight: {config.get('best_blend_centroid_weight')}",
        "- v2 uses only pre-race race-shape, course-adjusted early-speed context, and historical horse lap profiles.",
        "- Actual lap mode is diagnostic only and is not used in ticket filtering.",
        "- Runtime BUY gates are unchanged; candidate filters here are shadow backtests.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate race quality prediction v2 using queue, course-ten, and historical lap profiles.")
    parser.add_argument("--runners", default=str(DEFAULT_RUNNERS))
    parser.add_argument("--tickets", default=str(DEFAULT_TICKETS))
    parser.add_argument("--queue", default=str(DEFAULT_QUEUE))
    parser.add_argument("--course-ten", default=str(DEFAULT_COURSE_TEN))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    race = build_race_frame(Path(args.runners), Path(args.queue), Path(args.course_ten))
    features = feature_columns(race)
    train = race[race["source"].eq("train")].copy()
    model = fit_centroid_classifier(train, features)
    diag, config = add_predictions(race, model)

    summary = summarize_prediction(diag)
    ticket_summary = evaluate_ticket_filters(Path(args.tickets), diag)
    ticket_thresholds = ticket_summary.attrs.get("thresholds", {}) if hasattr(ticket_summary, "attrs") else {}

    diag.to_csv(out_dir / "race_quality_v2_diagnostics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "prediction_summary.csv", index=False, encoding="utf-8-sig")
    confusion(diag[diag["source"].eq("test")], "v1_predicted_lap_mode").to_csv(out_dir / "v1_confusion_test.csv", index=False, encoding="utf-8-sig")
    confusion(diag[diag["source"].eq("test")], "v2_predicted_lap_mode").to_csv(out_dir / "v2_confusion_test.csv", index=False, encoding="utf-8-sig")
    confidence_bins(diag).to_csv(out_dir / "v2_confidence_bins_test.csv", index=False, encoding="utf-8-sig")
    ticket_summary.to_csv(out_dir / "ticket_filter_summary.csv", index=False, encoding="utf-8-sig")
    (out_dir / "model_params.json").write_text(
        json.dumps(serializable_model(model), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = {
        "races": int(len(diag)),
        "test_races": int(diag["source"].eq("test").sum()),
        "features": features,
        "best_blend_centroid_weight": config["best_blend_centroid_weight"],
        "ticket_thresholds": ticket_thresholds,
        "prediction_summary": summary.replace({np.nan: None}).to_dict(orient="records"),
        "ticket_summary": ticket_summary.replace({np.nan: None}).to_dict(orient="records") if not ticket_summary.empty else [],
        "notes": [
            "Shadow diagnostic only. Production BUY gates are unchanged.",
            "v2 maps long_spurt to sustain for broad race-quality comparison.",
            "The next runtime step is to expose v2 confidence/mode in dashboard and LINE as an explanatory field.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(out_dir, summary, ticket_summary, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
