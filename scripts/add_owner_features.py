from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


OWNER_NUMERIC_FEATURES = [
    "owner_starts",
    "owner_win_rate",
    "owner_top3_rate",
    "owner_avg_score",
    "owner_popularity_outperform_rate",
    "owner_surface_starts",
    "owner_surface_top3_rate",
    "owner_distance_starts",
    "owner_distance_top3_rate",
    "owner_venue_starts",
    "owner_venue_top3_rate",
    "owner_class_starts",
    "owner_class_top3_rate",
    "owner_trainer_pair_starts",
    "owner_trainer_pair_top3_rate",
    "owner_jockey_pair_starts",
    "owner_jockey_pair_top3_rate",
    "owner_trainer_synergy_score",
    "owner_context_fit_score",
]

OWNER_CATEGORICAL_FEATURES = [
    "owner_name_for_model",
    "owner_type_for_model",
]


def _read_csv(path: Path, encoding: str | None = None) -> pd.DataFrame:
    if encoding:
        return pd.read_csv(path, encoding=encoding, low_memory=False)
    return pd.read_csv(path, low_memory=False)


def _num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index)
    if series.dtype == object or str(series.dtype).startswith("string"):
        series = series.astype("string").str.replace(",", "", regex=False).str.replace("+", "", regex=False)
    return pd.to_numeric(series, errors="coerce")


def _class_bucket(series: pd.Series) -> pd.Series:
    values = series.astype("string").fillna("")
    return pd.Series(
        np.select(
            [
                values.str.contains("新馬", na=False),
                values.str.contains("未勝利", na=False),
                values.str.contains("1勝|500万", regex=True, na=False),
                values.str.contains("2勝|1000万", regex=True, na=False),
                values.str.contains("3勝|1600万", regex=True, na=False),
                values.str.contains("オープン|OP|L", regex=True, na=False),
                values.str.contains("G", na=False),
            ],
            ["newcomer", "maiden", "class_1win", "class_2win", "class_3win", "open", "graded"],
            default="other",
        ),
        index=series.index,
    )


def _previous_group_stats(
    frame: pd.DataFrame,
    keys: list[str],
    prefix: str,
    score_col: str = "target_score",
) -> pd.DataFrame:
    work = frame.copy()
    if score_col in work.columns:
        work[score_col] = _num(work[score_col], work.index, 0.0).fillna(0.0)
    group = frame.groupby(keys, dropna=False, sort=False)
    work_group = work.groupby(keys, dropna=False, sort=False)
    starts = group.cumcount()
    out = pd.DataFrame(index=frame.index)
    out[f"{prefix}_starts"] = starts.astype(float)
    out[f"{prefix}_win_rate"] = (group["target_win"].cumsum() - frame["target_win"]) / starts.replace(0, np.nan)
    out[f"{prefix}_top3_rate"] = (group["target_top3"].cumsum() - frame["target_top3"]) / starts.replace(0, np.nan)
    if score_col in frame.columns:
        out[f"{prefix}_avg_score"] = (work_group[score_col].cumsum() - work[score_col]) / starts.replace(0, np.nan)
    if "popularity_outperform" in frame.columns:
        out[f"{prefix}_popularity_outperform_rate"] = (
            group["popularity_outperform"].cumsum() - frame["popularity_outperform"]
        ) / starts.replace(0, np.nan)
    return out


def _previous_top3_rate(frame: pd.DataFrame, keys: list[str], prefix: str) -> pd.DataFrame:
    group = frame.groupby(keys, dropna=False, sort=False)
    starts = group.cumcount()
    return pd.DataFrame(
        {
            f"{prefix}_starts": starts.astype(float),
            f"{prefix}_top3_rate": (group["target_top3"].cumsum() - frame["target_top3"]) / starts.replace(0, np.nan),
        },
        index=frame.index,
    )


def _resolve_input_path(raw_arg: str, fallback_glob: str | None = None) -> Path:
    path = Path(raw_arg)
    if path.exists():
        return path
    if fallback_glob:
        matches = list(Path(".").glob(fallback_glob))
        if matches:
            return matches[0]
    raise FileNotFoundError(raw_arg)


def add_owner_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    raw: pd.DataFrame | None = None,
    owner_master: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    race_col = "レースID(新/馬番無)"
    horse_col = "血統登録番号"
    owner_latest_col = "馬主(最新/仮想)"
    owner_race_col = "馬主(レース時)"
    owner_type_latest_col = "馬主タイプ"
    owner_type_race_col = "馬主タイプ(レース時)"

    if owner_master is not None:
        owner_lookup = owner_master.copy()
        if "owner_name" in owner_lookup.columns:
            owner_lookup["owner_name_for_model"] = owner_lookup["owner_name"]
        if "owner_code" in owner_lookup.columns:
            owner_lookup["owner_type_for_model"] = owner_lookup["owner_code"]
        owner_lookup = owner_lookup[[horse_col, "owner_name_for_model", "owner_type_for_model"]].drop_duplicates(horse_col)
        train_out = train.merge(owner_lookup, on=horse_col, how="left")
        test_out = test.merge(owner_lookup, on=horse_col, how="left")
    else:
        if raw is None:
            raise ValueError("Either raw or owner_master is required.")
        owner_cols = [
            race_col,
            horse_col,
            owner_latest_col,
            owner_race_col,
            owner_type_latest_col,
            owner_type_race_col,
        ]
        owner_lookup = raw[[col for col in owner_cols if col in raw.columns]].drop_duplicates([race_col, horse_col])
        owner_lookup["owner_name_for_model"] = owner_lookup.get(owner_race_col, pd.Series("", index=owner_lookup.index))
        owner_lookup["owner_name_for_model"] = owner_lookup["owner_name_for_model"].fillna(owner_lookup.get(owner_latest_col))
        owner_lookup["owner_type_for_model"] = owner_lookup.get(owner_type_race_col, pd.Series("", index=owner_lookup.index))
        owner_lookup["owner_type_for_model"] = owner_lookup["owner_type_for_model"].fillna(owner_lookup.get(owner_type_latest_col))
        owner_lookup = owner_lookup[[race_col, horse_col, "owner_name_for_model", "owner_type_for_model"]]
        train_out = train.merge(owner_lookup, on=[race_col, horse_col], how="left")
        test_out = test.merge(owner_lookup, on=[race_col, horse_col], how="left")
    train_out["_split"] = "train"
    test_out["_split"] = "test"
    combined = pd.concat([train_out, test_out], ignore_index=True, sort=False)

    combined["_date_num"] = _num(combined.get("日付"), combined.index, 0).fillna(0)
    combined["_race_num"] = _num(combined.get(race_col), combined.index, 0).fillna(0)
    combined["_horse_num"] = _num(combined.get("馬番"), combined.index, 0).fillna(0)
    combined["_orig_order"] = np.arange(len(combined))
    combined = combined.sort_values(["_date_num", "_race_num", "_horse_num", "_orig_order"], kind="mergesort").reset_index(drop=True)

    combined["owner_name_for_model"] = combined["owner_name_for_model"].astype("string").fillna("__UNKNOWN_OWNER__")
    combined["owner_type_for_model"] = combined["owner_type_for_model"].astype("string").fillna("__UNKNOWN_OWNER_TYPE__")
    combined["target_win"] = _num(combined["target_win"], combined.index, 0).fillna(0.0)
    combined["target_top3"] = _num(combined["target_top3"], combined.index, 0).fillna(0.0)
    rank = _num(combined.get("確定着順"), combined.index)
    pop = _num(combined.get("人気"), combined.index)
    combined["popularity_outperform"] = (rank.notna() & pop.notna() & rank.lt(pop)).astype(float)

    combined["owner_class_bucket"] = _class_bucket(combined.get("クラス名", pd.Series("", index=combined.index)))

    owner_stats = _previous_group_stats(combined, ["owner_name_for_model"], "owner")
    for col in owner_stats.columns:
        combined[col] = owner_stats[col]

    contexts = [
        (["owner_name_for_model", "芝・ダ"], "owner_surface"),
        (["owner_name_for_model", "distance_category"], "owner_distance"),
        (["owner_name_for_model", "場所"], "owner_venue"),
        (["owner_name_for_model", "owner_class_bucket"], "owner_class"),
        (["owner_name_for_model", "調教師コード"], "owner_trainer_pair"),
        (["owner_name_for_model", "騎手コード"], "owner_jockey_pair"),
    ]
    for keys, prefix in contexts:
        stats = _previous_top3_rate(combined, keys, prefix)
        for col in stats.columns:
            combined[col] = stats[col]

    combined["owner_trainer_synergy_score"] = (
        combined["owner_trainer_pair_top3_rate"] - combined["owner_top3_rate"]
    ).where(combined["owner_trainer_pair_starts"].ge(5), 0.0)
    context_cols = [
        "owner_surface_top3_rate",
        "owner_distance_top3_rate",
        "owner_venue_top3_rate",
        "owner_class_top3_rate",
    ]
    combined["owner_context_fit_score"] = combined[context_cols].mean(axis=1) - combined["owner_top3_rate"]

    for col in OWNER_NUMERIC_FEATURES:
        if col not in combined.columns:
            combined[col] = np.nan

    combined = combined.sort_values("_orig_order", kind="mergesort")
    train_final = combined[combined["_split"].eq("train")].drop(columns=[c for c in combined.columns if c.startswith("_")])
    test_final = combined[combined["_split"].eq("test")].drop(columns=[c for c in combined.columns if c.startswith("_")])
    return train_final.reset_index(drop=True), test_final.reset_index(drop=True)


def update_config(base_config: Path, output_config: Path) -> None:
    config = json.loads(base_config.read_text(encoding="utf-8"))
    numeric = list(config.get("generated_numeric_features", []))
    for col in OWNER_NUMERIC_FEATURES:
        if col not in numeric:
            numeric.append(col)
    categorical = list(config.get("generated_categorical_features", []))
    for col in OWNER_CATEGORICAL_FEATURES:
        if col not in categorical:
            categorical.append(col)
    config["generated_numeric_features"] = numeric
    config["generated_categorical_features"] = categorical
    config.setdefault("metadata", {})
    config["metadata"]["owner_features"] = {
        "source": "馬主(レース時), fallback 馬主(最新/仮想)",
        "policy": "previous-only expanding statistics; no same-race result leakage",
        "numeric_features": OWNER_NUMERIC_FEATURES,
        "categorical_features": OWNER_CATEGORICAL_FEATURES,
    }
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--raw-csv", default="date/raw/全競走馬成績.csv")
    parser.add_argument("--raw-glob", default="date/raw/*.csv")
    parser.add_argument("--owner-master-csv", default="data/processed/target/owner_master.csv")
    parser.add_argument("--output-dir", default="data/datasets/cache/owner_enriched")
    parser.add_argument("--base-config", default="config/baseline_features_body_workout.json")
    parser.add_argument("--output-config", default="config/baseline_features_body_workout_owner.json")
    args = parser.parse_args()

    train_path = Path(args.train_csv)
    test_path = Path(args.test_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = _read_csv(train_path)
    test = _read_csv(test_path)
    owner_master_path = Path(args.owner_master_csv)
    if owner_master_path.exists():
        owner_master = _read_csv(owner_master_path)
        raw = None
    else:
        raw_path = _resolve_input_path(args.raw_csv, args.raw_glob)
        raw_cols = [
            "レースID(新/馬番無)",
            "血統登録番号",
            "馬主(最新/仮想)",
            "馬主タイプ",
            "馬主(レース時)",
            "馬主タイプ(レース時)",
        ]
        raw = pd.read_csv(raw_path, encoding="cp932", usecols=lambda c: c in raw_cols, low_memory=False)
        owner_master = None
    train_out, test_out = add_owner_features(train, test, raw=raw, owner_master=owner_master)

    train_out_path = output_dir / "train_features_with_same_day_bias_v3_retro_body_owner.csv"
    test_out_path = output_dir / "test_features_with_same_day_bias_v3_retro_body_owner.csv"
    train_out.to_csv(train_out_path, index=False, encoding="utf-8-sig")
    test_out.to_csv(test_out_path, index=False, encoding="utf-8-sig")
    update_config(Path(args.base_config), Path(args.output_config))

    coverage = {
        "train_rows": int(len(train_out)),
        "test_rows": int(len(test_out)),
        "train_owner_coverage": float(train_out["owner_name_for_model"].ne("__UNKNOWN_OWNER__").mean()),
        "test_owner_coverage": float(test_out["owner_name_for_model"].ne("__UNKNOWN_OWNER__").mean()),
        "unique_train_owners": int(train_out["owner_name_for_model"].nunique(dropna=True)),
        "unique_test_owners": int(test_out["owner_name_for_model"].nunique(dropna=True)),
        "train_csv": str(train_out_path),
        "test_csv": str(test_out_path),
        "config": args.output_config,
    }
    (output_dir / "owner_feature_summary.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(coverage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
