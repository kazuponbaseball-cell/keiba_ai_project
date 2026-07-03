from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HORSE_COL = "\u8840\u7d71\u767b\u9332\u756a\u53f7"
RACE_COL = "\u30ec\u30fc\u30b9ID(\u65b0/\u99ac\u756a\u7121)"
DATE_COL = "\u65e5\u4ed8"
AGE_COL = "\u5e74\u9f62"
SURFACE_COL = "\u829d\u30fb\u30c0"
VENUE_COL = "\u5834\u6240"
CLASS_COL = "\u30af\u30e9\u30b9\u540d"

NORTHERN = "\u30ce\u30fc\u30b6\u30f3\u30d5\u30a1\u30fc\u30e0"
SHADAI = "\u793e\u53f0\u30d5\u30a1\u30fc\u30e0"
SHADAI_OLD = "\u793e\u53f0\u30d5\u30a2\u30fc\u30e0"
SHIRAOI = "\u793e\u53f0\u30b3\u30fc\u30dd\u30ec\u30fc\u30b7\u30e7\u30f3\u767d\u8001\u30d5\u30a1\u30fc\u30e0"
SHIRAOI_SHORT = "\u767d\u8001\u30d5\u30a1\u30fc\u30e0"
OIWAKE = "\u8ffd\u5206\u30d5\u30a1\u30fc\u30e0"

BREEDER_NUMERIC_FEATURES = [
    "breeder_northern_farm_flag",
    "breeder_shadai_farm_flag",
    "breeder_shiraoi_farm_flag",
    "breeder_oiwake_farm_flag",
    "breeder_shadai_group_flag",
    "breeder_starts",
    "breeder_win_rate",
    "breeder_top3_rate",
    "breeder_avg_score",
    "breeder_popularity_outperform_rate",
    "breeder_surface_starts",
    "breeder_surface_top3_rate",
    "breeder_distance_starts",
    "breeder_distance_top3_rate",
    "breeder_venue_starts",
    "breeder_venue_top3_rate",
    "breeder_class_starts",
    "breeder_class_top3_rate",
    "breeder_turf_young_flag",
    "breeder_northern_turf_young_flag",
    "breeder_shadai_turf_young_flag",
    "breeder_young_turf_fit_score",
    "breeder_context_fit_score",
]
BREEDER_CATEGORICAL_FEATURES = ["breeder_group_for_model"]


def _num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index)
    if series.dtype == object or str(series.dtype).startswith("string"):
        series = series.astype("string").str.replace(",", "", regex=False).str.replace("+", "", regex=False)
    return pd.to_numeric(series, errors="coerce")


def _breeder_group(name: pd.Series) -> pd.Series:
    text = name.astype("string").fillna("")
    return pd.Series(
        np.select(
            [
                text.eq(NORTHERN),
                text.isin([SHADAI, SHADAI_OLD]),
                text.isin([SHIRAOI, SHIRAOI_SHORT]),
                text.eq(OIWAKE),
                text.str.contains("\u793e\u53f0|\u767d\u8001|\u8ffd\u5206", regex=True, na=False),
            ],
            ["northern_farm", "shadai_farm", "shiraoi_farm", "oiwake_farm", "other_shadai_group"],
            default="other",
        ),
        index=name.index,
    )


def _class_bucket(series: pd.Series) -> pd.Series:
    values = series.astype("string").fillna("")
    return pd.Series(
        np.select(
            [
                values.str.contains("\u65b0\u99ac", na=False),
                values.str.contains("\u672a\u52dd\u5229", na=False),
                values.str.contains("1\u52dd|500\u4e07", regex=True, na=False),
                values.str.contains("2\u52dd|1000\u4e07", regex=True, na=False),
                values.str.contains("3\u52dd|1600\u4e07", regex=True, na=False),
                values.str.contains("\u30aa\u30fc\u30d7\u30f3|OP|L", regex=True, na=False),
                values.str.contains("G", na=False),
            ],
            ["newcomer", "maiden", "class_1win", "class_2win", "class_3win", "open", "graded"],
            default="other",
        ),
        index=series.index,
    )


def _previous_stats(frame: pd.DataFrame, keys: list[str], prefix: str) -> pd.DataFrame:
    work = frame.copy()
    work["target_score"] = _num(work.get("target_score"), work.index, 0.0).fillna(0.0)
    group = work.groupby(keys, dropna=False, sort=False)
    starts = group.cumcount()
    out = pd.DataFrame(index=work.index)
    out[f"{prefix}_starts"] = starts.astype(float)
    out[f"{prefix}_win_rate"] = (group["target_win"].cumsum() - work["target_win"]) / starts.replace(0, np.nan)
    out[f"{prefix}_top3_rate"] = (group["target_top3"].cumsum() - work["target_top3"]) / starts.replace(0, np.nan)
    out[f"{prefix}_avg_score"] = (group["target_score"].cumsum() - work["target_score"]) / starts.replace(0, np.nan)
    if "popularity_outperform" in work.columns:
        out[f"{prefix}_popularity_outperform_rate"] = (
            group["popularity_outperform"].cumsum() - work["popularity_outperform"]
        ) / starts.replace(0, np.nan)
    return out


def _previous_top3(frame: pd.DataFrame, keys: list[str], prefix: str) -> pd.DataFrame:
    group = frame.groupby(keys, dropna=False, sort=False)
    starts = group.cumcount()
    return pd.DataFrame(
        {
            f"{prefix}_starts": starts.astype(float),
            f"{prefix}_top3_rate": (group["target_top3"].cumsum() - frame["target_top3"]) / starts.replace(0, np.nan),
        },
        index=frame.index,
    )


def add_breeder_features(train: pd.DataFrame, test: pd.DataFrame, breeder_master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    breeder = breeder_master[[HORSE_COL, "breeder_name", "breeder_code", "birthplace"]].drop_duplicates(HORSE_COL)
    train_out = train.merge(breeder, on=HORSE_COL, how="left")
    test_out = test.merge(breeder, on=HORSE_COL, how="left")
    train_out["_split"] = "train"
    test_out["_split"] = "test"
    frame = pd.concat([train_out, test_out], ignore_index=True, sort=False)

    frame["_date_num"] = _num(frame.get(DATE_COL), frame.index, 0).fillna(0)
    frame["_race_num"] = _num(frame.get(RACE_COL), frame.index, 0).fillna(0)
    frame["_orig_order"] = np.arange(len(frame))
    frame = frame.sort_values(["_date_num", "_race_num", "_orig_order"], kind="mergesort").reset_index(drop=True)

    frame["breeder_name"] = frame["breeder_name"].astype("string").fillna("__UNKNOWN_BREEDER__")
    frame["breeder_group_for_model"] = _breeder_group(frame["breeder_name"])
    frame["target_win"] = _num(frame["target_win"], frame.index, 0).fillna(0.0)
    frame["target_top3"] = _num(frame["target_top3"], frame.index, 0).fillna(0.0)
    rank = _num(frame.get("\u78ba\u5b9a\u7740\u9806"), frame.index)
    pop = _num(frame.get("\u4eba\u6c17"), frame.index)
    frame["popularity_outperform"] = (rank.notna() & pop.notna() & rank.lt(pop)).astype(float)
    frame["breeder_class_bucket"] = _class_bucket(frame.get(CLASS_COL, pd.Series("", index=frame.index)))

    group = frame["breeder_group_for_model"]
    frame["breeder_northern_farm_flag"] = group.eq("northern_farm").astype(float)
    frame["breeder_shadai_farm_flag"] = group.eq("shadai_farm").astype(float)
    frame["breeder_shiraoi_farm_flag"] = group.eq("shiraoi_farm").astype(float)
    frame["breeder_oiwake_farm_flag"] = group.eq("oiwake_farm").astype(float)
    frame["breeder_shadai_group_flag"] = group.isin(
        ["northern_farm", "shadai_farm", "shiraoi_farm", "oiwake_farm", "other_shadai_group"]
    ).astype(float)
    is_turf = frame[SURFACE_COL].astype("string").str.contains("\u829d", na=False)
    is_young = _num(frame.get(AGE_COL), frame.index).le(3)
    frame["breeder_turf_young_flag"] = (is_turf & is_young).astype(float)
    frame["breeder_northern_turf_young_flag"] = (group.eq("northern_farm") & is_turf & is_young).astype(float)
    frame["breeder_shadai_turf_young_flag"] = (group.eq("shadai_farm") & is_turf & is_young).astype(float)

    for col, values in _previous_stats(frame, ["breeder_name"], "breeder").items():
        frame[col] = values
    for keys, prefix in [
        (["breeder_name", SURFACE_COL], "breeder_surface"),
        (["breeder_name", "distance_category"], "breeder_distance"),
        (["breeder_name", VENUE_COL], "breeder_venue"),
        (["breeder_name", "breeder_class_bucket"], "breeder_class"),
    ]:
        for col, values in _previous_top3(frame, keys, prefix).items():
            frame[col] = values

    frame["breeder_young_turf_fit_score"] = np.where(
        frame["breeder_turf_young_flag"].eq(1),
        frame[["breeder_surface_top3_rate", "breeder_class_top3_rate"]].mean(axis=1) - frame["breeder_top3_rate"],
        0.0,
    )
    context_cols = ["breeder_surface_top3_rate", "breeder_distance_top3_rate", "breeder_venue_top3_rate", "breeder_class_top3_rate"]
    frame["breeder_context_fit_score"] = frame[context_cols].mean(axis=1) - frame["breeder_top3_rate"]

    for col in BREEDER_NUMERIC_FEATURES:
        if col not in frame.columns:
            frame[col] = np.nan

    frame = frame.sort_values("_orig_order", kind="mergesort")
    train_final = frame[frame["_split"].eq("train")].drop(columns=[c for c in frame.columns if c.startswith("_")])
    test_final = frame[frame["_split"].eq("test")].drop(columns=[c for c in frame.columns if c.startswith("_")])
    return train_final.reset_index(drop=True), test_final.reset_index(drop=True)


def update_config(base_config: Path, output_config: Path) -> None:
    config = json.loads(base_config.read_text(encoding="utf-8"))
    numeric = list(config.get("generated_numeric_features", []))
    for col in BREEDER_NUMERIC_FEATURES:
        if col not in numeric:
            numeric.append(col)
    categorical = list(config.get("generated_categorical_features", []))
    for col in BREEDER_CATEGORICAL_FEATURES:
        if col not in categorical:
            categorical.append(col)
    config["generated_numeric_features"] = numeric
    config["generated_categorical_features"] = categorical
    config.setdefault("metadata", {})["breeder_features"] = {
        "source": "TARGET UM_DATA breeder master",
        "policy": "breeder identity/group plus previous-only expanding statistics",
        "numeric_features": BREEDER_NUMERIC_FEATURES,
        "categorical_features": BREEDER_CATEGORICAL_FEATURES,
    }
    output_config.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--breeder-master-csv", default="data/processed/target/breeder_master.csv")
    parser.add_argument("--output-dir", default="data/datasets/cache/breeder_enriched")
    parser.add_argument("--base-config", default="config/baseline_features_body_workout.json")
    parser.add_argument("--output-config", default="config/baseline_features_body_workout_breeder.json")
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv, low_memory=False)
    test = pd.read_csv(args.test_csv, low_memory=False)
    breeder = pd.read_csv(args.breeder_master_csv, low_memory=False)
    train_out, test_out = add_breeder_features(train, test, breeder)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_features_with_same_day_bias_v3_retro_body_breeder.csv"
    test_path = output_dir / "test_features_with_same_day_bias_v3_retro_body_breeder.csv"
    train_out.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_out.to_csv(test_path, index=False, encoding="utf-8-sig")
    update_config(Path(args.base_config), Path(args.output_config))

    summary = {
        "train_rows": int(len(train_out)),
        "test_rows": int(len(test_out)),
        "train_breeder_coverage": float(train_out["breeder_name"].notna().mean()),
        "test_breeder_coverage": float(test_out["breeder_name"].notna().mean()),
        "train_group_counts": train_out["breeder_group_for_model"].value_counts().to_dict(),
        "test_group_counts": test_out["breeder_group_for_model"].value_counts().to_dict(),
        "train_csv": str(train_path),
        "test_csv": str(test_path),
        "config": args.output_config,
    }
    (output_dir / "breeder_feature_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
