from __future__ import annotations

import argparse
import json
from pathlib import Path


FEATURES = [
    "body_prev_weight",
    "body_prev_delta",
    "body_prev_delta_pct",
    "body_prev_abs_delta",
    "body_prev_large_gain_flag",
    "body_prev_large_loss_flag",
    "body_prev_extreme_change_flag",
    "body_layoff_flag",
    "body_layoff_gain_flag",
    "body_layoff_loss_flag",
    "body_young_growth_gain_flag",
    "body_female_large_loss_flag",
    "body_small_horse_flag",
    "body_large_horse_flag",
    "body_very_large_horse_flag",
    "body_weight_z_in_race",
    "body_weight_rank_in_race",
    "body_weight_percentile_in_race",
    "body_race_heavy_top3_flag",
    "body_race_heavy_top5_flag",
    "body_age2_flag",
    "body_age3_flag",
    "body_age2_big500_flag",
    "body_age2_big520_flag",
    "body_age2_small460_flag",
    "body_age2_race_heavy_top3_flag",
    "body_age2_race_heavy_top5_flag",
    "body_age3_big500_flag",
    "body_age3_big520_flag",
    "body_age3_small460_flag",
    "body_age3_race_heavy_top3_flag",
    "body_age3_race_heavy_top5_flag",
    "body_young_maturity_score",
    "body_layoff_workout_count_fit",
    "body_layoff_recent_workout_flag",
    "body_loss_with_strong_workout_flag",
    "horse_goodrun_same_pattern_count_past5",
    "horse_goodrun_same_pattern_rate_past5",
    "horse_goodrun_same_lap_count_past5",
    "horse_goodrun_same_lap_rate_past5",
    "horse_last_goodrun_same_pattern_flag",
    "horse_last_goodrun_same_lap_flag",
    "horse_workout_past_goodrun_match_score",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a workout config with body/workout context features.")
    parser.add_argument("--base-config", default="config/baseline_features_workout.json")
    parser.add_argument("--output-config", default="config/baseline_features_body_workout.json")
    args = parser.parse_args()
    config = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    for key in ["generated_numeric_features", "passthrough_prediction_columns"]:
        existing = list(config.get(key, []))
        seen = set(existing)
        for feature in FEATURES:
            if feature not in seen:
                existing.append(feature)
                seen.add(feature)
        config[key] = existing
    Path(args.output_config).write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_config": args.output_config, "added": FEATURES}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
