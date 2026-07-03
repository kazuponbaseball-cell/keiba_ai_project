from __future__ import annotations

import argparse
import json
from pathlib import Path


WORKOUT_NUMERIC_FEATURES = [
    "workout_latest_days_before_race",
    "workout_latest_total_time_sec",
    "workout_latest_penultimate_1f_sec",
    "workout_latest_final_1f_sec",
    "workout_latest_final_2f_sec",
    "workout_latest_final_3f_sec",
    "workout_latest_finish_gain_sec",
    "workout_latest_total_vs_trainer_z",
    "workout_latest_final1_vs_trainer_z",
    "workout_latest_total_vs_course_z",
    "workout_latest_final1_vs_course_z",
    "workout_count",
    "workout_days_span",
    "workout_best_total_time_sec",
    "workout_best_final_1f_sec",
    "workout_avg_final_1f_sec",
    "workout_fast_final_flag",
    "workout_strong_finish_flag",
    "workout_partner_win_flag",
    "workout_a1_flag",
    "workout_b1_flag",
    "workout_a2_flag",
    "workout_b2_flag",
    "workout_a3_flag",
    "workout_b3_flag",
    "workout_best_total_vs_trainer_z",
    "workout_best_final1_vs_trainer_z",
    "workout_hill_count",
    "workout_wood_count",
    "workout_current_week_count",
    "workout_prev_week_count",
    "workout_prev_day_count",
    "workout_prev_day_hill_flag",
    "workout_prev_day_wood_flag",
    "workout_prev_week_hill_count",
    "workout_prev_week_wood_count",
    "workout_prev_weekend_hill_under60_flag",
    "workout_current_week_wood_after_prev_week_hill_flag",
    "workout_hill_wood_mix_flag",
    "workout_latest_course_switch_flag",
    "workout_latest_from_hill_to_wood_flag",
    "workout_latest_from_wood_to_hill_flag",
    "workout_strong_work_count",
    "workout_strong_work_rate",
    "workout_current_week_strong_count",
    "workout_prev_week_strong_count",
    "workout_prev_week_strong_then_current_light_flag",
    "workout_hill_53_count",
    "workout_hill_51_count",
    "workout_wood_67_count",
    "workout_wood_53_count",
    "workout_final_11_count",
    "workout_final_12_count",
    "workout_strong_finish_count",
    "workout_avg_finish_gain_sec",
    "workout_best_finish_gain_sec",
    "workout_latest_slower_than_best_total_sec",
    "workout_avg_gap_days",
    "workout_max_gap_days",
    "workout_load_density_score",
    "workout_course_pattern_starts",
    "workout_course_pattern_win_rate",
    "workout_course_pattern_top3_rate",
    "workout_course_pattern_avg_score",
    "workout_course_pattern_win_roi",
    "workout_course_lap_starts",
    "workout_course_lap_win_rate",
    "workout_course_lap_top3_rate",
    "workout_course_lap_avg_score",
    "workout_course_lap_win_roi",
    "workout_trainer_pattern_starts",
    "workout_trainer_pattern_win_rate",
    "workout_trainer_pattern_top3_rate",
    "workout_trainer_pattern_avg_score",
    "workout_trainer_pattern_win_roi",
    "workout_trainer_lap_starts",
    "workout_trainer_lap_win_rate",
    "workout_trainer_lap_top3_rate",
    "workout_trainer_lap_avg_score",
    "workout_trainer_lap_win_roi",
    "workout_horse_pattern_starts",
    "workout_horse_pattern_win_rate",
    "workout_horse_pattern_top3_rate",
    "workout_horse_pattern_avg_score",
    "workout_horse_pattern_win_roi",
    "workout_horse_lap_starts",
    "workout_horse_lap_win_rate",
    "workout_horse_lap_top3_rate",
    "workout_horse_lap_avg_score",
    "workout_horse_lap_win_roi",
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

WORKOUT_CATEGORICAL_FEATURES = [
    "workout_latest_course_bucket",
    "workout_latest_lap_group",
    "workout_latest_pattern_bucket",
    "workout_knowledge_grade",
    "workout_knowledge_pattern",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a baseline feature config with workout features.")
    parser.add_argument("--base-config", default="config/baseline_features.json")
    parser.add_argument("--output-config", default="config/baseline_features_workout.json")
    args = parser.parse_args()

    base_path = Path(args.base_config)
    output_path = Path(args.output_config)
    config = json.loads(base_path.read_text(encoding="utf-8"))
    config["generated_numeric_features"] = _append_unique(
        config.get("generated_numeric_features", []),
        WORKOUT_NUMERIC_FEATURES,
    )
    config["generated_categorical_features"] = _append_unique(
        config.get("generated_categorical_features", []),
        WORKOUT_CATEGORICAL_FEATURES,
    )
    config["passthrough_prediction_columns"] = _append_unique(
        config.get("passthrough_prediction_columns", []),
        WORKOUT_NUMERIC_FEATURES + WORKOUT_CATEGORICAL_FEATURES,
    )
    output_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_config": str(output_path)}, ensure_ascii=False, indent=2))


def _append_unique(existing: list[str], additions: list[str]) -> list[str]:
    out = list(existing)
    seen = set(out)
    for item in additions:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


if __name__ == "__main__":
    main()
