# ロンスパ性能特徴量設計

TARGET由来のラップ・馬柱データから、馬ごとの「ロングスパート戦への適性・対応力」を定量化するための設計。

実装入口:

- `src/features/longspurt.py`
- `scripts/build_longspurt_features.py`

## 目的

PCI単体、後半3F単体、後半5Fの絶対値だけでロンスパ性能を判定しない。レース単位では後半5Fを距離・競馬場・芝ダート別に標準化し、ラップ形状と最速ラップ位置から瞬発戦・消耗戦と切り分ける。馬単位では、速い後半5Fのレースで好走できているか、4角から着順を上げているか、早めに進出しているかを組み合わせて評価する。

## 入力カラム

### レース単位

|論理名|TARGET/既存列例|用途|
|---|---|---|
|`race_id`|`レースID(新/馬番無)`|レースキー|
|`date`|`日付`|時系列集計|
|`venue`|`場所`|標準化グループ|
|`surface`|`芝・ダ`|標準化グループ|
|`distance`|`距離`|標準化グループ|
|`going`|`馬場状態`|将来の馬場別補正|
|`race_laps`|`レースラップ` / `レースラップタイム`|後半5F・形状判定|

### 出走馬単位

|論理名|TARGET/既存列例|用途|
|---|---|---|
|`race_id`|`レースID(新/馬番無)`|レース特徴量との結合|
|`horse_id`|`血統登録番号`|馬キー|
|`date`|`日付`|過去走のみ集計|
|`finish`|`確定着順`|好走・着順改善|
|`popularity`|`人気`|人気以上好走|
|`field_size`|`出走頭数` / `頭数`|着順・上がり順位の正規化|
|`corner1`-`corner4`|`1角`-`4角`|進出・位置取り|
|`final3f_rank`|`上り3F順`|上がり順位補正|

## レース単位特徴量

`build_race_longspurt_features(races)` が以下を作る。

|特徴量|意味|
|---|---|
|`last5f_sum`|後半5F合計|
|`last3f_sum`|後半3F合計|
|`last5f_mean`|同距離・同競馬場・同芝ダート内の後半5F平均|
|`last5f_vs_mean`|後半5F - グループ平均。マイナスほど速い|
|`last5f_z`|後半5Fのzスコア。マイナスほど速い|
|`last5f_fast_percentile`|後半5Fの速さ順位。0に近いほど速い|
|`last5f_top25`|同条件内で後半5F上位25%|
|`fastest_lap_index_last5`|後半5F内の最速区間。1=残り5F、5=ラスト1F|
|`fastest_lap_remaining_f`|最速区間が残り何F地点か|
|`last2_bias`|後半5F前半3F平均 - ラスト2F平均。大きいほどラスト2F偏重|
|`final1_deceleration`|ラスト1F - ラスト2F。大きいほど最後に失速|
|`last5f_range`|後半5F内の最大ラップ差|
|`race_longspurt_type`|`ロンスパ戦` / `瞬発戦` / `持続戦` / `消耗戦` / `標準戦`|

標準化グループはまず `distance x venue x surface`。サンプルが少ない場合は `distance x surface` にフォールバックする。

## レース分類

暫定ルール:

|分類|条件|
|---|---|
|ロンスパ戦|`last5f_top25` かつ最速ラップ位置が後半5Fの1-3区間、かつラスト2F偏重でない|
|瞬発戦|`last5f_top25` かつ最速ラップ位置がラスト2F、かつラスト2F偏重|
|消耗戦|ラスト1Fの減速が大きく、後半5Fも上位50%以内|
|持続戦|後半5F上位50%以内で、後半5F内のラップ差が小さく、ラスト1Fの失速が小さい|
|標準戦|上記以外|

現行デフォルト:

- 上位25%: `last5f_fast_percentile <= 0.25`
- ラスト2F偏重: `last2_bias >= 0.35`
- ラスト1F失速: `final1_deceleration >= 0.50`
- 持続形: `last5f_range <= 0.70` かつ `final1_deceleration <= 0.30`

## 馬単位特徴量

`add_runner_longspurt_outcomes(runners, race_features)` で各レース結果に以下を付与する。

|特徴量|意味|
|---|---|
|`top3`|3着以内|
|`finish_score`|頭数で正規化した着順スコア|
|`popularity_outperform`|人気 - 着順。プラスなら人気以上|
|`corner4_to_finish_gain`|4角位置 - 着順。プラスなら4角後に着順を上げた|
|`early_position_gain_to_4c`|最初の通過順 - 4角位置。プラスなら早め進出|
|`early_move_flag`|4角までに2つ以上押し上げ、4角で射程圏|
|`late_burst_only_flag`|早め進出がなく、4角から大きく差しただけの後方一気|
|`final3f_rank_score`|上がり順位の頭数正規化スコア|

`build_horse_longspurt_features(runner_features)` は、各出走行に対して同一馬の過去走だけから以下を作る。

|特徴量|意味|
|---|---|
|`horse_longspurt_starts_prev`|過去のロンスパ戦出走数|
|`horse_longspurt_top3_rate`|過去ロンスパ戦の複勝率|
|`horse_fast_last5f_top3_rate`|後半5F上位25%レースの複勝率|
|`horse_longspurt_avg_finish_score`|ロンスパ戦の平均着順スコア|
|`horse_longspurt_corner4_gain_avg`|ロンスパ戦の平均4角→着順改善|
|`horse_longspurt_early_move_rate`|ロンスパ戦で早め進出した率|
|`horse_longspurt_late_burst_only_rate`|後方一気寄りの率。高いと減点|
|`horse_longspurt_final3f_rank_score_avg`|ロンスパ戦の平均上がり順位スコア|
|`horse_longspurt_popularity_outperform_avg`|ロンスパ戦の平均人気以上好走度|
|`horse_longspurt_score`|総合ロンスパ性能スコア|
|`horse_longspurt_aptitude_flag`|スコアと経験数で判定した適性フラグ|

## スコア式

現行の `horse_longspurt_score` は0-100にクリップする。

```text
35 * ロンスパ戦複勝率
+20 * 後半5F上位25%レース複勝率
+12 * ロンスパ戦平均着順スコア
+10 * 早め進出率
+ 8 * 上がり順位スコア
+ 8 * 4角→着順改善スコア
+ 7 * 人気以上好走スコア
-10 * 後方一気のみ率
```

後方一気だけで届いた馬をロンスパ型にしすぎないよう、`late_burst_only_rate` を明示的に減点している。

## pandas実装方針

```python
from src.features.longspurt import build_longspurt_feature_set

race_features, runner_features = build_longspurt_feature_set(races, runners)
```

CSVから使う場合:

```powershell
<python> scripts/build_longspurt_features.py `
  --races-csv path/to/races.csv `
  --runners-csv path/to/runners.csv `
  --output-dir outputs/features/longspurt
```

## テーブル設計案

### `race_laps`

|列|型|内容|
|---|---|---|
|`race_id`|TEXT PK|レースID|
|`date`|INTEGER|日付|
|`venue`|TEXT|競馬場|
|`surface`|TEXT|芝/ダ|
|`distance`|INTEGER|距離|
|`going`|TEXT|馬場状態|
|`race_laps`|TEXT|`12.4-11.0-...`|

### `race_longspurt_features`

|列|型|内容|
|---|---|---|
|`race_id`|TEXT PK/FK|レースID|
|`last5f_sum`|REAL|後半5F|
|`last5f_z`|REAL|標準化値|
|`last5f_fast_percentile`|REAL|速さ順位|
|`fastest_lap_index_last5`|INTEGER|後半5F内の最速位置|
|`last2_bias`|REAL|ラスト2F偏重|
|`final1_deceleration`|REAL|ラスト1F減速|
|`race_longspurt_type`|TEXT|分類|
|`is_longspurt_race`|BOOLEAN|ロンスパ戦|

### `runner_results`

|列|型|内容|
|---|---|---|
|`race_id`|TEXT FK|レースID|
|`horse_id`|TEXT FK|馬ID|
|`finish`|INTEGER|着順|
|`popularity`|INTEGER|人気|
|`field_size`|INTEGER|頭数|
|`corner1`-`corner4`|INTEGER|通過順|
|`final3f_rank`|INTEGER|上がり順位|

主キーは `race_id, horse_id`。

### `runner_longspurt_outcomes`

|列|型|内容|
|---|---|---|
|`race_id`|TEXT|レースID|
|`horse_id`|TEXT|馬ID|
|`top3`|BOOLEAN|好走|
|`corner4_to_finish_gain`|REAL|4角からの着順改善|
|`early_move_flag`|BOOLEAN|早め進出|
|`late_burst_only_flag`|BOOLEAN|後方一気寄り|

### `horse_longspurt_features`

|列|型|内容|
|---|---|---|
|`race_id`|TEXT|特徴量を使う対象レース|
|`horse_id`|TEXT|馬ID|
|`horse_longspurt_starts_prev`|REAL|過去ロンスパ経験|
|`horse_longspurt_top3_rate`|REAL|過去ロンスパ複勝率|
|`horse_fast_last5f_top3_rate`|REAL|高速後半5F複勝率|
|`horse_longspurt_score`|REAL|総合スコア|
|`horse_longspurt_aptitude_flag`|BOOLEAN|適性フラグ|

`race_id, horse_id` を主キーにすると、学習・推論用の出走馬テーブルへそのままJOINできる。
