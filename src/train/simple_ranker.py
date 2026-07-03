from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _to_numeric(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        cleaned = (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("◇", "", regex=False)
            .str.replace("☆", "", regex=False)
            .str.replace("△", "", regex=False)
            .str.replace("▲", "", regex=False)
            .str.replace("★", "", regex=False)
            .str.replace("nan", "", regex=False)
        )
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


@dataclass
class SimpleRaceRanker:
    numeric_features: list[str]
    categorical_features: list[str]
    categorical_top_k: int = 80
    ridge_alpha: float = 10.0

    numeric_medians_: dict[str, float] | None = None
    numeric_means_: dict[str, float] | None = None
    numeric_stds_: dict[str, float] | None = None
    categorical_levels_: dict[str, list[str]] | None = None
    coefficients_: np.ndarray | None = None
    feature_names_: list[str] | None = None

    def fit(self, df: pd.DataFrame, target: str) -> "SimpleRaceRanker":
        self.numeric_medians_ = {}
        self.numeric_means_ = {}
        self.numeric_stds_ = {}
        self.categorical_levels_ = {}

        numeric_parts = []
        feature_names = ["intercept"]
        for col in self.numeric_features:
            values = _to_numeric(df[col]) if col in df else pd.Series(np.nan, index=df.index)
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median).astype(float)
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            if not np.isfinite(std) or std == 0.0:
                std = 1.0
            self.numeric_medians_[col] = median
            self.numeric_means_[col] = mean
            self.numeric_stds_[col] = std
            numeric_parts.append(((filled - mean) / std).to_numpy()[:, None])
            feature_names.append(col)

        categorical_parts = []
        for col in self.categorical_features:
            values = df[col].astype("string").fillna("__MISSING__") if col in df else pd.Series("__MISSING__", index=df.index)
            levels = values.value_counts().head(self.categorical_top_k).index.astype(str).tolist()
            self.categorical_levels_[col] = levels
            for level in levels:
                categorical_parts.append((values.astype(str) == level).astype(float).to_numpy()[:, None])
                feature_names.append(f"{col}={level}")

        intercept = np.ones((len(df), 1), dtype=float)
        x = np.hstack([intercept, *numeric_parts, *categorical_parts])
        y = pd.to_numeric(df[target], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        reg = np.eye(x.shape[1], dtype=float) * self.ridge_alpha
        reg[0, 0] = 0.0
        self.coefficients_ = np.linalg.solve(x.T @ x + reg, x.T @ y)
        self.feature_names_ = feature_names
        return self

    def _design_matrix(self, df: pd.DataFrame) -> np.ndarray:
        if self.numeric_medians_ is None or self.numeric_means_ is None or self.numeric_stds_ is None:
            raise ValueError("Model is not fitted.")
        if self.categorical_levels_ is None:
            raise ValueError("Model is not fitted.")

        parts = [np.ones((len(df), 1), dtype=float)]
        for col in self.numeric_features:
            values = _to_numeric(df[col]) if col in df else pd.Series(np.nan, index=df.index)
            filled = values.fillna(self.numeric_medians_[col]).astype(float)
            parts.append(((filled - self.numeric_means_[col]) / self.numeric_stds_[col]).to_numpy()[:, None])

        for col in self.categorical_features:
            values = df[col].astype("string").fillna("__MISSING__") if col in df else pd.Series("__MISSING__", index=df.index)
            values = values.astype(str)
            for level in self.categorical_levels_[col]:
                parts.append((values == level).astype(float).to_numpy()[:, None])
        return np.hstack(parts)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.coefficients_ is None:
            raise ValueError("Model is not fitted.")
        return self._design_matrix(df) @ self.coefficients_

    def to_metadata(self) -> dict[str, Any]:
        return {
            "model_type": "SimpleRaceRanker",
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "feature_count": len(self.feature_names_ or []),
            "categorical_top_k": self.categorical_top_k,
            "ridge_alpha": self.ridge_alpha,
        }
