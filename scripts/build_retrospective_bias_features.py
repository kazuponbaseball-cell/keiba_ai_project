from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CONFIG = "config/baseline_features_workout_optimized_core_same_day_bias.json"


def _num(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(series, errors="coerce")


def _normalize_date(value: Any) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if "." in text and len(text.split(".")) == 3:
        y, m, d = text.split(".")
        y = y if len(y) == 4 else "20" + y.zfill(2)
        return int(f"{int(y):04d}{int(m):02d}{int(d):02d}")
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) == 6:
        return int("20" + digits)
    if len(digits) >= 8:
        return int(digits[:8])
    return None


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _frame_bucket(frame: pd.Series) -> pd.Series:
    value = _num(frame)
    return pd.Series(
        np.select(
            [value <= 3, value.between(4, 6), value >= 7],
            ["inner", "middle", "outer"],
            default="unknown",
        ),
        index=frame.index,
    )


def _style_bucket(corner_rate: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [
                corner_rate.le(0.25),
                corner_rate.gt(0.25) & corner_rate.le(0.45),
                corner_rate.gt(0.45) & corner_rate.lt(0.70),
                corner_rate.ge(0.70),
            ],
            ["front", "stalker", "midpack", "closer"],
            default="unknown",
        ),
        index=corner_rate.index,
    )


def _score01(values: pd.Series) -> pd.Series:
    return values.clip(-1.0, 1.0).fillna(0.0)


def add_retrospective_bias_features(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    data_cfg = config["data"]
    race_col = data_cfg["race_id_column"]
    horse_col = data_cfg["horse_id_column"]
    date_col = data_cfg["date_column"]
    rank_col = data_cfg["rank_column"]

    out = frame.copy()
    out["_retro_date_norm"] = out[date_col].map(_normalize_date)
    field = _num(out.get("出走頭数", out.get("頭数", pd.Series(np.nan, index=out.index)))).replace(0, np.nan)
    field = field.fillna(out.groupby(race_col)[race_col].transform("size").replace(0, np.nan))
    finish = _num(out[rank_col])
    popularity = _num(out.get("人気", pd.Series(np.nan, index=out.index)))
    corner4 = _num(out.get("4角.1", out.get("前4角.1", out.get("4角", pd.Series(np.nan, index=out.index)))))
    corner_rate = (corner4 / field).replace([np.inf, -np.inf], np.nan)

    target_score = _num(out.get("target_score", (field + 1.0 - finish) / field)).fillna(0.0)
    top3 = finish.between(1, 3)
    longshot = popularity.ge(6) | popularity.ge((field * 0.45).fillna(np.inf))
    pop_outperform = ((popularity - finish) / field).replace([np.inf, -np.inf], np.nan)

    out["retro_frame_bucket"] = _frame_bucket(out.get("枠番", pd.Series(np.nan, index=out.index)))
    out["retro_style_bucket"] = _style_bucket(corner_rate)
    out["retro_popularity_outperform"] = pop_outperform.fillna(0.0)
    out["retro_longshot_good_run"] = (
        longshot & (finish.le(4) | finish.le((field * 0.35).fillna(0)))
    ).astype(float)

    race_group = out.groupby(race_col, sort=False)
    style_pop_mean = out.groupby([race_col, "retro_style_bucket"], sort=False)["retro_popularity_outperform"].transform("mean")
    frame_pop_mean = out.groupby([race_col, "retro_frame_bucket"], sort=False)["retro_popularity_outperform"].transform("mean")
    race_pop_mean = race_group["retro_popularity_outperform"].transform("mean")
    style_top3 = top3.astype(float).groupby([out[race_col], out["retro_style_bucket"]], sort=False).transform("mean")
    frame_top3 = top3.astype(float).groupby([out[race_col], out["retro_frame_bucket"]], sort=False).transform("mean")
    race_top3 = top3.astype(float).groupby(out[race_col], sort=False).transform("mean")
    style_longshot = out["retro_longshot_good_run"].groupby([out[race_col], out["retro_style_bucket"]], sort=False).transform("mean")
    frame_longshot = out["retro_longshot_good_run"].groupby([out[race_col], out["retro_frame_bucket"]], sort=False).transform("mean")

    out["retro_style_pop_advantage"] = (style_pop_mean - race_pop_mean).fillna(0.0)
    out["retro_frame_pop_advantage"] = (frame_pop_mean - race_pop_mean).fillna(0.0)
    out["retro_style_top3_advantage"] = (style_top3 - race_top3).fillna(0.0)
    out["retro_frame_top3_advantage"] = (frame_top3 - race_top3).fillna(0.0)
    out["retro_style_longshot_signal"] = style_longshot.fillna(0.0)
    out["retro_frame_longshot_signal"] = frame_longshot.fillna(0.0)
    out["retro_bias_help_score"] = _score01(
        0.55 * out["retro_style_pop_advantage"]
        + 0.25 * out["retro_frame_pop_advantage"]
        + 0.20 * out["retro_style_longshot_signal"]
    ).clip(lower=0.0)
    out["retro_bias_adversity_score"] = _score01(
        -0.65 * out["retro_style_pop_advantage"]
        -0.25 * out["retro_frame_pop_advantage"]
        -0.10 * out["retro_style_top3_advantage"]
    ).clip(lower=0.0)

    performance = (0.65 * target_score + 0.35 * pop_outperform.fillna(0.0)).clip(-1.0, 1.0)
    out["retro_bias_resistant_score"] = (out["retro_bias_adversity_score"] * performance.clip(lower=0.0)).fillna(0.0)
    out["retro_bias_excuse_score"] = (
        out["retro_bias_adversity_score"]
        * (1.0 - target_score).clip(0.0, 1.0)
        * (1.0 - pop_outperform.fillna(0.0)).clip(0.0, 1.5)
    ).fillna(0.0)
    out["retro_bias_overhelped_score"] = (
        out["retro_bias_help_score"]
        * (target_score + pop_outperform.fillna(0.0).clip(lower=0.0))
    ).fillna(0.0)

    ordered = out.sort_values([horse_col, "_retro_date_norm", race_col], kind="mergesort")
    rolling_specs = [
        ("retro_bias_help_score", "prev_retro_bias_help_score", "last"),
        ("retro_bias_adversity_score", "prev_retro_bias_adversity_score", "last"),
        ("retro_bias_resistant_score", "prev_retro_bias_resistant_score", "last"),
        ("retro_bias_excuse_score", "prev_retro_bias_excuse_score", "last"),
        ("retro_bias_overhelped_score", "prev_retro_bias_overhelped_score", "last"),
        ("retro_bias_help_score", "past3_retro_bias_help_score", "mean"),
        ("retro_bias_adversity_score", "past3_retro_bias_adversity_score", "mean"),
        ("retro_bias_resistant_score", "past3_retro_bias_resistant_score", "mean"),
        ("retro_bias_excuse_score", "past3_retro_bias_excuse_score", "mean"),
        ("retro_bias_overhelped_score", "past3_retro_bias_overhelped_score", "mean"),
    ]
    for source, dest, agg in rolling_specs:
        values = _num(ordered[source])
        if agg == "last":
            rolled = values.groupby(ordered[horse_col], sort=False).shift()
        else:
            rolled = values.groupby(ordered[horse_col], sort=False).transform(
                lambda s: s.shift().rolling(3, min_periods=1).mean()
            )
        out.loc[ordered.index, dest] = pd.to_numeric(rolled, errors="coerce").fillna(0.0).astype(float)

    return out.drop(columns=["_retro_date_norm"], errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add retrospective race-bias and prior-bias-history features.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = _load_config(args.config)
    input_path = Path(args.input_csv)
    frame = pd.read_csv(input_path, encoding="utf-8-sig", low_memory=False)
    out = add_retrospective_bias_features(frame, config)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    retro_cols = [col for col in out.columns if col.startswith("retro_") or col.startswith("prev_retro_") or col.startswith("past3_retro_")]
    print(
        json.dumps(
            {
                "input_csv": str(input_path),
                "output_csv": str(output_path),
                "rows": int(len(out)),
                "added_columns": retro_cols,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
