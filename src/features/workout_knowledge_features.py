from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.workout_knowledge import (
    evaluate_workout_knowledge,
    prepare_workouts_for_knowledge,
    select_entry_workouts,
)


KNOWLEDGE_NUMERIC_FEATURES = [
    "workout_knowledge_grade_score",
    "workout_knowledge_registered_flag",
    "workout_knowledge_s_flag",
    "workout_knowledge_a_flag",
    "workout_knowledge_b_flag",
    "workout_knowledge_d_flag",
    "workout_knowledge_high_grade_flag",
    "workout_knowledge_mid_grade_flag",
    "workout_knowledge_plus_count",
    "workout_knowledge_minus_count",
    "workout_knowledge_minus_flag",
    "workout_knowledge_high_x_load_density",
    "workout_knowledge_score_x_load_density",
]

KNOWLEDGE_CATEGORICAL_FEATURES = [
    "workout_knowledge_grade",
    "workout_knowledge_pattern",
]


def add_workout_knowledge_features(
    frame: pd.DataFrame,
    workouts: pd.DataFrame,
    *,
    lookback_days: int = 21,
) -> pd.DataFrame:
    prepared = prepare_workouts_for_knowledge(workouts)
    rows: list[dict[str, Any]] = []
    for _, entry in frame.iterrows():
        selected = select_entry_workouts(entry, prepared, lookback_days=lookback_days)
        result = evaluate_workout_knowledge(entry, selected)
        plus_count = len(result.get("plus_factors") or [])
        minus_count = len(result.get("minus_factors") or [])
        grade = str(result.get("grade", "C"))
        score = float(result.get("grade_score", 2))
        registered = 0.0 if result.get("matched_pattern") == "対象外厩舎" else 1.0
        rows.append(
            {
                "workout_knowledge_grade": grade,
                "workout_knowledge_pattern": str(result.get("matched_pattern", "")),
                "workout_knowledge_grade_score": score,
                "workout_knowledge_registered_flag": registered,
                "workout_knowledge_s_flag": float(grade == "S"),
                "workout_knowledge_a_flag": float(grade == "A"),
                "workout_knowledge_b_flag": float(grade == "B"),
                "workout_knowledge_d_flag": float(grade == "D"),
                "workout_knowledge_high_grade_flag": float(grade in {"S", "A"}),
                "workout_knowledge_mid_grade_flag": float(grade in {"S", "A", "B"}),
                "workout_knowledge_plus_count": float(plus_count),
                "workout_knowledge_minus_count": float(minus_count),
                "workout_knowledge_minus_flag": float(minus_count > 0),
            }
        )

    features = pd.DataFrame(rows, index=frame.index)
    out = frame.copy()
    for col in features.columns:
        out[col] = features[col]

    load_density = _num(out.get("workout_load_density_score", pd.Series(np.nan, index=out.index))).fillna(0.0)
    out["workout_knowledge_high_x_load_density"] = out["workout_knowledge_high_grade_flag"] * load_density
    out["workout_knowledge_score_x_load_density"] = out["workout_knowledge_grade_score"] * load_density
    return out


def _num(values: pd.Series) -> pd.Series:
    if values.dtype == object or str(values.dtype).startswith("string"):
        values = values.astype("string").str.replace(",", "", regex=False).str.replace("+", "", regex=False)
    return pd.to_numeric(values, errors="coerce")
