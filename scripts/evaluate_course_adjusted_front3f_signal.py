from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_estimated_front3f_race_quality_features import (
    DEFAULT_MODEL,
    DEFAULT_TEST,
    DEFAULT_TRAIN,
    HORSE_COL,
    RACE_COL,
    build_course_front3f_priors,
    confidence_weight,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ESTIMATES = ROOT / "outputs" / "analysis" / "estimated_front3f_race_quality_v1" / "estimated_runner_front3f_cache.csv"
OUT_DIR = ROOT / "outputs" / "analysis" / "course_adjusted_front3f_signal_v1"


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, **kwargs)


def ncol(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def race_z(values: pd.Series, race_id: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    mean = values.groupby(race_id).transform("mean")
    std = values.groupby(race_id).transform("std").replace(0, np.nan)
    return ((values - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def top_gap(s: pd.Series) -> float:
    vals = np.sort(pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy())[::-1]
    return float(vals[0] - vals[1]) if len(vals) >= 2 else 0.0


def sigmoid(x: pd.Series) -> pd.Series:
    return pd.Series(1.0 / (1.0 + np.exp(-np.clip(x, -30, 30))), index=x.index)


def enrich_course_adjusted_front3f(base: pd.DataFrame, estimates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = base.copy()
    out["_race_key"] = out[RACE_COL].astype(str)
    out["_horse_key"] = out[HORSE_COL].astype(str).str.strip()
    estimates = estimates.copy()
    estimates["race_id"] = estimates["race_id"].astype(str)
    estimates["horse_id"] = estimates["horse_id"].astype(str).str.strip()
    estimates["estimated_front3f_sec"] = pd.to_numeric(estimates["estimated_front3f_sec"], errors="coerce")
    estimates["front3f_confidence_weight"] = confidence_weight(estimates["front3f_confidence"])

    priors = build_course_front3f_priors(out, estimates)
    out = out.merge(priors, on="_race_key", how="left")
    estimates = estimates.merge(priors.rename(columns={"_race_key": "race_id"}), on="race_id", how="left")
    course_prior = pd.to_numeric(estimates["course_front3f_prior_sec"], errors="coerce")
    course_std = pd.to_numeric(estimates["course_front3f_prior_std"], errors="coerce").fillna(0.8).clip(0.35, 2.5)
    estimates["course_adj_ten_z"] = ((course_prior - estimates["estimated_front3f_sec"]) / course_std).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0).clip(-4.0, 4.0)
    estimates["course_adj_ten_z_weighted"] = estimates["course_adj_ten_z"] * estimates["front3f_confidence_weight"]
    estimates["course_adj_fast_start"] = (
        estimates["course_adj_ten_z"].ge(0.55) | estimates["estimated_front3f_sec"].le(course_prior - 0.35)
    ).astype(float)

    out = out.merge(
        estimates[["race_id", "horse_id", "course_adj_ten_z_weighted", "course_adj_fast_start"]].rename(
            columns={"race_id": "_race_key", "horse_id": "_horse_key"}
        ),
        on=["_race_key", "_horse_key"],
        how="left",
    )
    out["_date_order"] = pd.to_numeric(out["_race_key"].str.slice(0, 8), errors="coerce").fillna(0)
    ordered = out.sort_values(["_horse_key", "_date_order", "_race_key"], kind="mergesort")
    values = pd.to_numeric(ordered["course_adj_ten_z_weighted"], errors="coerce")
    fast = pd.to_numeric(ordered["course_adj_fast_start"], errors="coerce")
    out.loc[ordered.index, "horse_course_adj_ten_speed_mean_past5"] = values.groupby(ordered["_horse_key"], sort=False).transform(
        lambda s: s.shift().rolling(5, min_periods=1).mean()
    )
    out.loc[ordered.index, "horse_course_adj_ten_speed_best_past5"] = values.groupby(ordered["_horse_key"], sort=False).transform(
        lambda s: s.shift().rolling(5, min_periods=1).max()
    )
    out.loc[ordered.index, "horse_course_adj_fast_start_rate_past5"] = fast.groupby(ordered["_horse_key"], sort=False).transform(
        lambda s: s.shift().rolling(5, min_periods=1).mean()
    )
    for col in [
        "horse_course_adj_ten_speed_mean_past5",
        "horse_course_adj_ten_speed_best_past5",
        "horse_course_adj_fast_start_rate_past5",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["course_adj_ten_race_z"] = race_z(out["horse_course_adj_ten_speed_mean_past5"], out[RACE_COL])
    course_pos = out["horse_course_adj_ten_speed_mean_past5"].clip(lower=0.0)
    field = course_pos.groupby(out[RACE_COL]).transform("size").replace(0, np.nan)
    out["race_course_adj_ten_pressure_score"] = (course_pos.groupby(out[RACE_COL]).transform("sum") / np.sqrt(field)).fillna(0.0)
    out["race_course_adj_fast_start_count"] = (
        ((out["horse_course_adj_ten_speed_mean_past5"].ge(0.45)) | (out["horse_course_adj_fast_start_rate_past5"].ge(0.34)))
        .astype(float)
        .groupby(out[RACE_COL])
        .transform("sum")
    )
    out["race_course_adj_ten_speed_gap_top2"] = out["horse_course_adj_ten_speed_mean_past5"].groupby(out[RACE_COL]).transform(top_gap)
    out["race_course_adj_queue_clarity_score"] = sigmoid(
        1.5 * out["race_course_adj_ten_speed_gap_top2"] - 0.32 * out["race_course_adj_fast_start_count"] + 0.35
    )
    diag = out[
        [
            RACE_COL,
            "course_front3f_prior_sec",
            "course_front3f_prior_std",
            "course_front3f_prior_count",
            "race_course_adj_ten_pressure_score",
            "race_course_adj_fast_start_count",
            "race_course_adj_ten_speed_gap_top2",
            "race_course_adj_queue_clarity_score",
        ]
    ].drop_duplicates(RACE_COL)
    return out, diag


def bet_metrics(part: pd.DataFrame) -> dict[str, Any]:
    if part.empty:
        return {"bets": 0, "races": 0, "win_rate": 0.0, "top3_rate": 0.0, "win_roi": 0.0, "place_roi": 0.0}
    win_pay = ncol(part, "単勝配当", 0.0).where(part["target_win"].eq(1), 0.0)
    place_pay = ncol(part, "複勝配当", 0.0).where(part["target_top3"].eq(1), 0.0)
    return {
        "bets": int(len(part)),
        "races": int(part[RACE_COL].nunique()),
        "win_rate": float(part["target_win"].mean()),
        "top3_rate": float(part["target_top3"].mean()),
        "win_roi": float(win_pay.sum() / (len(part) * 100.0)),
        "place_roi": float(place_pay.sum() / (len(part) * 100.0)),
        "avg_popularity": float(ncol(part, "人気", np.nan).mean()),
        "avg_odds": float(ncol(part, "単勝オッズ", np.nan).mean()),
    }


def segment_report(enriched: pd.DataFrame) -> pd.DataFrame:
    out = enriched.copy()
    out["rank"] = out.groupby(RACE_COL)["ai_score_eval"].rank(ascending=False, method="first").astype(int)
    train = out[out["_split"].eq("train")]
    test = out[out["_split"].eq("test")]
    q = {
        "course_adj_ten_hi": float(train["horse_course_adj_ten_speed_mean_past5"].quantile(0.75)),
        "course_adj_best_hi": float(train["horse_course_adj_ten_speed_best_past5"].quantile(0.75)),
        "course_adj_clarity_hi": float(train["race_course_adj_queue_clarity_score"].quantile(0.75)),
        "course_adj_pressure_hi": float(train["race_course_adj_ten_pressure_score"].quantile(0.75)),
    }
    pop5 = ncol(test, "人気", np.nan).ge(5)
    checks = [
        ("ai1_all", test["rank"].eq(1)),
        ("ai1_course_adj_ten_hi", test["rank"].eq(1) & test["horse_course_adj_ten_speed_mean_past5"].ge(q["course_adj_ten_hi"])),
        ("ai1_course_adj_best_hi", test["rank"].eq(1) & test["horse_course_adj_ten_speed_best_past5"].ge(q["course_adj_best_hi"])),
        ("ai1_course_adj_clarity_hi", test["rank"].eq(1) & test["race_course_adj_queue_clarity_score"].ge(q["course_adj_clarity_hi"])),
        ("ai1_course_adj_pressure_hi", test["rank"].eq(1) & test["race_course_adj_ten_pressure_score"].ge(q["course_adj_pressure_hi"])),
        (
            "ai3_pop5plus_course_adj_ten_hi",
            test["rank"].le(3) & pop5 & test["horse_course_adj_ten_speed_mean_past5"].ge(q["course_adj_ten_hi"]),
        ),
        (
            "ai3_pop5plus_course_adj_best_hi",
            test["rank"].le(3) & pop5 & test["horse_course_adj_ten_speed_best_past5"].ge(q["course_adj_best_hi"]),
        ),
        (
            "ai3_pop5plus_course_adj_clarity_hi",
            test["rank"].le(3) & pop5 & test["race_course_adj_queue_clarity_score"].ge(q["course_adj_clarity_hi"]),
        ),
    ]
    return pd.DataFrame([{"segment": name, **bet_metrics(test[mask])} for name, mask in checks])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = read_csv(ROOT / DEFAULT_TRAIN)
    test = read_csv(ROOT / DEFAULT_TEST)
    train["_split"] = "train"
    test["_split"] = "test"
    base = pd.concat([train, test], ignore_index=True, sort=False)
    estimates = read_csv(DEFAULT_ESTIMATES, dtype={"race_id": str})
    enriched, diag = enrich_course_adjusted_front3f(base, estimates)
    with (ROOT / DEFAULT_MODEL).open("rb") as fh:
        base_model = pickle.load(fh)
    enriched["ai_score_eval"] = np.nan
    for split_name in ["train", "test"]:
        mask = enriched["_split"].eq(split_name)
        if mask.any():
            enriched.loc[mask, "ai_score_eval"] = base_model.predict(enriched.loc[mask])
    segments = segment_report(enriched)
    keep = [
        RACE_COL,
        HORSE_COL,
        "馬名",
        "人気",
        "単勝オッズ",
        "ai_score_eval",
        "target_win",
        "target_top3",
        "horse_course_adj_ten_speed_mean_past5",
        "horse_course_adj_ten_speed_best_past5",
        "horse_course_adj_fast_start_rate_past5",
        "course_adj_ten_race_z",
        "race_course_adj_ten_pressure_score",
        "race_course_adj_queue_clarity_score",
    ]
    enriched[[c for c in keep if c in enriched.columns]].to_csv(OUT_DIR / "course_adjusted_front3f_enriched_light.csv", index=False, encoding="utf-8-sig")
    diag.to_csv(OUT_DIR / "course_adjusted_front3f_race_diagnostics.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(OUT_DIR / "course_adjusted_front3f_signal_segments.csv", index=False, encoding="utf-8-sig")
    summary = {
        "rows": int(len(enriched)),
        "races": int(enriched[RACE_COL].nunique()),
        "estimate_rows": int(len(estimates)),
        "output_dir": str(OUT_DIR),
        "notes": [
            "This is a lightweight signal test using existing target_score ranks, not a refit of the full strongest model.",
            "Course priors are time-safe: the current race front 3F is excluded from its own prior.",
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    show = segments.copy()
    for col in ["win_rate", "top3_rate", "win_roi", "place_roi"]:
        show[col] = show[col] * 100.0
    print(show.to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
