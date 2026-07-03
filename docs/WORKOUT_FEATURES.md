# 調教特徴量メモ

調教は「時計が速いほど良い」と固定せず、厩舎ごとの勝ちパターンを学習できる形で入れる。

## 取り込み方針

想定入力は1行1本の追い切り/調教CSV。

最低限ほしい列:

- `race_id` または `レースID(新/馬番無)`
- `horse_id` または `血統登録番号`
- `horse_number` または `馬番`
- `trainer_code` または `調教師コード`
- `workout_date` または `追切日` / `調教日`
- `course` または `調教コース`
- `total_time_sec` または `全体時計`
- `final_1f_sec` または `終い1F`

任意で効く列:

- `race_date`
- `distance_f`
- `final_2f_sec`
- `final_3f_sec`
- `intensity` / `強度`
- `partner_result` / `併せ結果`

## TARGET CK_DATAからの抽出

TARGETの `CK_DATA` から坂路調教 `HC*.DAT` とウッドチップ調教 `WC*.DAT` を抽出する。

```powershell
python scripts/extract_target_workouts.py --start-date 20260101 --end-date 20260613 --output-csv data/processed/target/workouts_20260101_20260613.csv
```

抽出後、出馬表や学習データに対して、レース前の調教を紐づける。

```powershell
python scripts/enrich_entry_with_target_workouts.py --entry-csv data/datasets/inference/weekly/entry_snapshot.csv --workouts-csv data/processed/target/workouts_20260101_20260613.csv --lookback-days 21
```

TARGETの調教レコード自体にはレースIDがないため、`血統登録番号 x レース日` で、レース前 `lookback-days` 日以内の調教を選ぶ。

## 作る特徴量

- 本数: `workout_count`
- 最終追い切りの近さ: `workout_latest_days_before_race`
- 最終時計: `workout_latest_total_time_sec`
- 終い: `workout_latest_final_1f_sec`, `workout_best_final_1f_sec`
- 仕上げ型: `workout_latest_course_bucket`, `workout_latest_pattern_bucket`
- 終い加速: `workout_latest_finish_gain_sec`
- 終いラップ分類: `workout_latest_lap_group`
- 厩舎平均との差: `workout_latest_total_vs_trainer_z`, `workout_latest_final1_vs_trainer_z`
- コース平均との差: `workout_latest_total_vs_course_z`, `workout_latest_final1_vs_course_z`
- 仕上げ強調フラグ: `workout_fast_final_flag`, `workout_strong_finish_flag`
- ラップ型フラグ: `workout_a1_flag`, `workout_b1_flag`, `workout_a2_flag`, `workout_b2_flag`, `workout_a3_flag`, `workout_b3_flag`

## 終いラップ分類

`終い2F` と `終い1F` がある場合、`終い2F - 終い1F` を2F目の1Fとして扱う。

- `A1`: 終い1Fのみ12秒台の加速ラップ。例: `13.1-12.5`
- `B1`: 2F目のみ12秒台の減速ラップ。例: `12.8-13.1`
- `A2`: 終い2Fがどちらも12秒台の加速ラップ。例: `12.5-12.1`
- `B2`: 終い2Fがどちらも12秒台の減速ラップ。例: `12.2-12.6`
- `A3`: 終い1Fが11秒台の加速ラップ。2F目が11秒台でも含める。例: `12.5-11.9`
- `B3`: 2F目が11秒台の減速ラップ。終い1Fが11秒台でも含める。例: `11.8-12.7`

この分類は厩舎別に見る。

- `trainer_code x workout_latest_lap_group`
- `trainer_code x workout_latest_course_bucket x workout_latest_lap_group`
- `trainer_code x rotation_bucket x workout_latest_lap_group`
- `race_course_key x workout_latest_pattern_bucket`
- `race_course_key x workout_latest_lap_group`
- `horse_id x workout_latest_pattern_bucket`
- `horse_id x workout_latest_lap_group`

## 見たいロジック

厩舎ごとに以下を検証する。

- 速い全体時計がプラスに出る厩舎
- 終い重点がプラスに出る厩舎
- 坂路主体がプラスに出る厩舎
- ウッドで長めからやるとプラスに出る厩舎
- 併せ先着が効く厩舎
- 休み明けだけ調教強度が効く厩舎
- 東京芝1600はウッドA2/A3が効く、阪神芝2200は坂路A1よりウッド長めが効く、のようなコース別パターン
- 個別馬が特定の調教型だけ走るパターン

## コース別・個体別の検証

調教特徴量を結合した成績CSVに対して、以下の集計を見る。

```powershell
python scripts/analyze_workout_pattern_segments.py --input-csv data/datasets/train/baseline_train_dataset_with_workout.csv
```

主な出力:

- `workout_segment_race_course_pattern.csv`: レースコース別の調教パターン成績
- `workout_segment_race_course_lap_group.csv`: レースコース別のA1/B1/A2/B2/A3/B3成績
- `workout_segment_trainer_pattern.csv`: 厩舎別の調教パターン成績
- `workout_segment_horse_pattern.csv`: 馬個体別の調教パターン成績

個体別はサンプルが薄くなりやすいので、最低出走数を高めにする。

## モデル投入用の累積特徴

調教特徴量を結合した成績CSVに対して、未来を見ない累積特徴を付与する。

```powershell
python scripts/add_workout_history_features.py --input-csv data/datasets/train/baseline_train_dataset_with_workout.csv
```

追加される主な特徴:

- `workout_course_pattern_starts`
- `workout_course_pattern_win_rate`
- `workout_course_pattern_top3_rate`
- `workout_course_pattern_avg_score`
- `workout_course_pattern_win_roi`
- `workout_course_lap_*`
- `workout_trainer_pattern_*`
- `workout_trainer_lap_*`
- `workout_horse_pattern_*`
- `workout_horse_lap_*`

累積特徴はレース単位で過去分だけを参照する。同一レース内の他馬結果は使わない。

最終的には `trainer_code x workout_pattern_bucket` の過去成績を、未来データを使わない累積特徴として入れる。
