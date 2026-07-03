from __future__ import annotations

import argparse
import pickle
from datetime import datetime

import pandas as pd

from src.data.loaders import (
    inference_optional_columns,
    inference_required_columns,
    load_historical_csv,
    load_json_config,
    required_columns,
)
from src.features.baseline import (
    add_pace_scenario_scores,
    add_bloodline_features,
    add_confirmed_opponent_form_features,
    add_deep_pace_style_features,
    add_draw_bias_features,
    add_general_generated_features,
    add_horse_surface_history,
    add_jockey_trainer_rotation_features,
    add_lap_aptitude_features,
    add_local_racing_context_features,
    add_member_level_features,
    add_race_pace_features,
    add_race_relative_features,
    add_same_day_bias_interaction_features,
    add_time_value_features,
    add_track_condition_features,
    assert_no_leakage,
    contract_from_config,
    prepare_training_frame,
)
from src.features.odds_timeline import build_odds_timeline_features_from_file, merge_odds_timeline_features
from src.utils.paths import ensure_dir, project_path


def _date_numeric(series: pd.Series) -> pd.Series:
    raw = series.astype("string").fillna("").str.strip()
    direct = pd.to_numeric(raw, errors="coerce")
    dotted = pd.to_datetime(raw, errors="coerce")
    dotted_num = pd.to_numeric(dotted.dt.strftime("%y%m%d"), errors="coerce")
    return direct.fillna(dotted_num)


def _add_training_targets(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    race_col = config["data"]["race_id_column"]
    rank_col = config["data"]["rank_column"]
    out = df.copy()
    out[rank_col] = pd.to_numeric(out[rank_col], errors="coerce")
    field_size = pd.Series(pd.NA, index=out.index, dtype="Float64")
    for field_col in ["出走頭数", "頭数"]:
        if field_col in out.columns:
            field_size = pd.to_numeric(out[field_col], errors="coerce")
            break
    if field_size.notna().sum() == 0:
        field_size = out.groupby(race_col)[rank_col].transform("max")
    out = out[field_size >= 2].copy()
    field_size = field_size.loc[out.index]
    out["target_score"] = (field_size + 1.0 - out[rank_col]) / field_size
    out["target_win"] = (out[rank_col] == 1).astype(int)
    out["target_top3"] = (out[rank_col] <= 3).astype(int)
    return out


def _load_historical_context(config: dict, columns: list[str], inference: pd.DataFrame) -> pd.DataFrame:
    data_cfg = config["data"]
    train_cfg = config["training"]
    race_col = data_cfg["race_id_column"]
    date_col = data_cfg["date_column"]
    rank_col = data_cfg["rank_column"]
    abnormal_col = data_cfg["abnormal_column"]

    history_path = project_path(data_cfg["historical_csv"])
    encoding = data_cfg.get("encoding", "cp932")
    history_header = pd.read_csv(history_path, encoding=encoding, nrows=0).columns.tolist()
    history_columns = [col for col in columns if col in history_header]
    if not history_columns:
        raise ValueError(f"No requested historical columns are available in {history_path}")
    horse_ids = set()
    if data_cfg["horse_id_column"] in inference.columns:
        horse_ids = set(inference[data_cfg["horse_id_column"]].astype("string").str.strip().dropna())
    if horse_ids:
        chunks = []
        for chunk in pd.read_csv(
            history_path,
            encoding=encoding,
            usecols=history_columns,
            chunksize=200_000,
            low_memory=False,
        ):
            keys = chunk[data_cfg["horse_id_column"]].astype("string").str.strip()
            matched = chunk[keys.isin(horse_ids)]
            if not matched.empty:
                chunks.append(matched)
        history = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=history_columns)
    else:
        history = pd.read_csv(history_path, encoding=encoding, usecols=history_columns, low_memory=False)
    history[rank_col] = pd.to_numeric(history[rank_col], errors="coerce")
    history[abnormal_col] = pd.to_numeric(history[abnormal_col], errors="coerce")
    history[date_col] = pd.to_numeric(history[date_col], errors="coerce")

    valid_abnormal = set(train_cfg.get("valid_abnormal_values", [0]))
    history = history[history[abnormal_col].isin(valid_abnormal)]
    race_type_col = train_cfg.get("race_type_column")
    exclude_race_types = set(train_cfg.get("exclude_race_type_values", []))
    if race_type_col and race_type_col in history.columns and exclude_race_types:
        race_types = pd.to_numeric(history[race_type_col], errors="coerce")
        history = history[~race_types.isin(exclude_race_types)]

    history = history[history[rank_col].notna()]
    history = history[history[race_col].notna()]
    history = history[history[date_col].notna()]

    inference_dates = _date_numeric(inference[date_col]) if date_col in inference.columns else pd.Series(dtype=float)
    if inference_dates.notna().any():
        history = history[history[date_col] < inference_dates.dropna().min()]

    return _add_training_targets(history, config)


def _add_inference_generated_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = add_general_generated_features(df, config)
    out = add_local_racing_context_features(out, config)
    out = add_jockey_trainer_rotation_features(out, config)
    out = add_time_value_features(out, config)
    out = add_horse_surface_history(out, config)
    out = add_track_condition_features(out, config)
    out = add_race_pace_features(out, config)
    out = add_deep_pace_style_features(out, config)
    out = add_lap_aptitude_features(out, config)
    out = add_bloodline_features(out, config)
    out = add_draw_bias_features(out, config)
    out = add_member_level_features(out, config)
    out = add_confirmed_opponent_form_features(out, config)
    out = add_race_relative_features(out, config)
    out = add_same_day_bias_interaction_features(out)
    return out


def _prediction_feature_passthrough(config: dict) -> list[str]:
    """Expose non-leaky generated features needed by the betting layer."""
    keep: list[str] = []
    for key in [
        "generated_numeric_features",
        "generated_categorical_features",
        "numeric_features",
        "categorical_features",
    ]:
        values = config.get(key, [])
        if isinstance(values, list):
            keep.extend(str(v) for v in values)
    keep.extend(
        [
            "horse_front_run_rate_past5",
            "horse_stalker_rate_past5",
            "horse_closer_rate_past5",
            "race_pace_collapse_risk",
            "race_slow_pace_risk",
            "pace_fit_score",
            "front_advantage_score",
            "draw_pace_fit_score",
            "workout_load_density_score",
            "workout_knowledge_grade_score",
            "workout_knowledge_registered_flag",
            "workout_knowledge_high_grade_flag",
            "workout_knowledge_minus_flag",
            "調教師コード",
            "芝・ダ",
            "場所",
            "距離",
            "馬場状態",
            "日付",
            "workout_latest_course_bucket",
            "workout_latest_lap_group",
            "workout_latest_pattern_bucket",
            "workout_latest_total_vs_trainer_z",
            "workout_latest_final1_vs_trainer_z",
            "workout_best_total_vs_trainer_z",
            "workout_best_final1_vs_trainer_z",
            "workout_latest_days_before_race",
            "workout_count",
            "workout_a1_flag",
            "workout_a2_flag",
            "workout_a3_flag",
            "workout_b1_flag",
            "workout_b2_flag",
            "workout_b3_flag",
            "workout_fast_final_flag",
            "workout_strong_finish_flag",
            "workout_partner_win_flag",
        ]
    )
    for lag in (1, 2, 3):
        keep.extend(
            [
                f"past{lag}_lap_fast_success",
                f"past{lag}_lap_slow_success",
                f"past{lag}_lap_instant_success",
                f"past{lag}_lap_sustain_success",
                f"past{lag}_lap_long_spurt_success",
                f"past{lag}_lap_fast_regime",
                f"past{lag}_lap_slow_regime",
                f"past{lag}_lap_instant_regime",
                f"past{lag}_lap_sustain_regime",
                f"past{lag}_lap_long_spurt_regime",
                f"past{lag}_lap_rpci",
                f"past{lag}_lap_pci",
                f"past{lag}_lap_pci3",
                f"past{lag}_lap_target_score",
            ]
        )
    # Preserve order while avoiding duplicated CSV columns.
    return list(dict.fromkeys(keep))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline race ranking inference.")
    parser.add_argument("--config", default="config/baseline_features.json")
    parser.add_argument("--model", default="models/baseline/baseline_ranker.pkl")
    parser.add_argument("--input-csv", default=None, help="Optional weekly/inference CSV. Defaults to historical CSV.")
    parser.add_argument("--race-id", default=None, help="Optional レースID(新/馬番無) filter.")
    parser.add_argument("--odds-timeline-csv", default=None, help="Optional odds timeline CSV to append market-movement features.")
    parser.add_argument(
        "--no-historical-context",
        action="store_true",
        help="Disable historical context for inference. This is faster but weakens pace/style history features.",
    )
    parser.add_argument("--output-dir", default="outputs/predictions")
    args = parser.parse_args()

    config = load_json_config(args.config)
    assert_no_leakage(contract_from_config(config))

    with project_path(args.model).open("rb") as f:
        model = pickle.load(f)

    columns = required_columns(config, for_prediction=True)
    if args.input_csv:
        input_path = project_path(args.input_csv)
        try:
            df = pd.read_csv(input_path, encoding="utf-8-sig", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(input_path, encoding=config["data"].get("encoding", "cp932"), low_memory=False)
        missing_required = [col for col in inference_required_columns(config) if col not in df.columns]
        if missing_required:
            raise ValueError(f"Inference snapshot is missing required columns: {missing_required}")
        for col in inference_optional_columns(config):
            if col not in df.columns:
                df[col] = pd.NA
        keep_cols = [col for col in columns if col in df.columns]
        df = df[keep_cols].copy()
        df["_is_inference_row"] = True
        if args.no_historical_context:
            df = _add_inference_generated_features(df, config)
        else:
            history = _load_historical_context(config, columns, df)
            history["_is_inference_row"] = False
            combined = pd.concat([history, df], ignore_index=True, sort=False)
            combined = _add_inference_generated_features(combined, config)
            df = combined[combined["_is_inference_row"].fillna(False)].copy()
        df = df.drop(columns=["_is_inference_row"], errors="ignore")
    else:
        df = load_historical_csv(config, columns=columns)
        df = prepare_training_frame(df, config)
        df = add_same_day_bias_interaction_features(df)

    race_col = config["data"]["race_id_column"]
    if args.race_id is not None:
        df = df[df[race_col].astype(str) == str(args.race_id)].copy()
    if df.empty:
        raise ValueError("No rows matched the inference input/filter.")
    if args.odds_timeline_csv:
        odds_features = build_odds_timeline_features_from_file(project_path(args.odds_timeline_csv))
        df = merge_odds_timeline_features(df, odds_features, race_col=race_col)

    df["ai_score"] = model.predict(df)
    df["ai_rank"] = df.groupby(race_col)["ai_score"].rank(ascending=False, method="first").astype(int)
    top_scores = df.groupby(race_col)["ai_score"].transform("max")
    second_scores = df[df["ai_rank"] == 2].set_index(race_col)["ai_score"]
    median_scores = df.groupby(race_col)["ai_score"].transform("median")
    df["_second_score"] = df[race_col].map(second_scores)
    df["ai_score_gap_to_second"] = (top_scores - df["_second_score"]).fillna(0.0)
    df["ai_top_score_vs_median"] = (top_scores - median_scores).fillna(0.0)
    df["ai_confidence_score"] = (
        0.70 * df["ai_score_gap_to_second"].clip(0.0, 0.20) / 0.20
        + 0.30 * df["ai_top_score_vs_median"].clip(0.0, 0.35) / 0.35
    ).clip(0.0, 1.0)
    df["ai_confidence_bucket"] = pd.cut(
        df["ai_confidence_score"],
        bins=[-0.01, 0.35, 0.60, 1.01],
        labels=["low", "medium", "high"],
    ).astype("string")
    df = df.drop(columns=["_second_score"], errors="ignore")
    df = add_pace_scenario_scores(df)

    keep = [
        race_col,
        "ai_rank",
        "ai_score",
        "expected_pace",
        "slow_ai_score",
        "middle_ai_score",
        "fast_ai_score",
        "front_running_tendency",
        "closing_tendency",
        "race_front_runner_count",
        "race_early_pressure_score",
        "ai_score_gap_to_second",
        "ai_top_score_vs_median",
        "ai_confidence_score",
        "ai_confidence_bucket",
        *[col for col in _prediction_feature_passthrough(config) if col in df.columns],
        *[col for col in config.get("passthrough_prediction_columns", []) if col in df.columns],
    ]
    keep = list(dict.fromkeys(keep))
    out = df[keep].sort_values([race_col, "ai_rank"])
    out_dir = ensure_dir(project_path(args.output_dir))
    suffix = args.race_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"baseline_predictions_{suffix}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(out.head(30).to_string(index=False))
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
