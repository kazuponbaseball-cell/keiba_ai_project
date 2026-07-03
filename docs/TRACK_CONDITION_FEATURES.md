# Track Condition Features

クッション値と含水率は、予想時点で公表済みなら使えるレース前情報として扱う。
当該レースの結果、時計、ラップ、着順は使わず、過去走だけから馬ごとの適性を作る。

## Input

`data/raw/track_condition_metrics.csv`

Required columns:

- `date`: `YYYYMMDD`, `YYMMDD`, or date-like text
- `venue`: TARGETの`場所`と同じ表記
- `cushion_value`: 芝クッション値
- `moisture_turf_goal`: 芝ゴール前含水率
- `moisture_turf_back`: 芝4角含水率
- `moisture_dirt_goal`: ダートゴール前含水率
- `moisture_dirt_back`: ダート4角含水率

Optional columns:

- `source_url`
- `measured_at`

## Generated Features

Race-level features:

- `race_cushion_value`
- `race_moisture_goal`
- `race_moisture_back`
- `race_moisture_avg`
- `race_cushion_z_by_venue`
- `race_moisture_z_by_venue_surface`
- `race_high_cushion_flag`
- `race_low_cushion_flag`
- `race_wet_moisture_flag`
- `race_dry_moisture_flag`

Horse history features:

- `horse_high_cushion_*`
- `horse_low_cushion_*`
- `horse_wet_moisture_*`
- `horse_dry_moisture_*`
- `horse_cushion_fit_score`
- `horse_moisture_fit_score`
- `horse_track_condition_fit_score`

The horse history features are shifted by horse, so the current race result is not included.

## Current Thresholds

- High cushion: `cushion_value >= 9.5`
- Low cushion: `0 < cushion_value <= 7.5`
- Wet moisture: `moisture_avg >= 12.0`
- Dry moisture: `0 < moisture_avg <= 8.0`

These thresholds are intentionally simple for the first validation pass. They should be re-tuned after enough JRA track-condition history has been collected.
