# Automation Pipeline

このプロジェクトでは、データ取得から学習・当週推論までを次の3層で運用します。

## 1. Raw

- `data/inbox/jv/`
  - JV-Dataの取得物を最初に置くローカル受け口
- `data/inbox/target/entries/`
  - TARGETの出馬表CSV受け口
- `data/inbox/target/odds/`
  - TARGET仕様オッズCSV受け口
- `data/inbox/target/timeline_odds/`
  - 時系列オッズCSV受け口
- `data/raw/jv/<run_id>/`
  - 差分アーカイブ
- `data/state/jv_fetch_state.json`
  - 取得済みファイルのサイズ・更新時刻を保持

`config/data_pipeline.json` の `jv_data.provider` で取得方法を切り替える。

- `local_drop`
  - 外部ツールや手作業で `data/inbox/jv/` に置く
- `external_command`
  - 任意のエクスポートコマンドを実行してから `data/inbox/jv/` を回収する

## 2. Processed / Normalized

- `data/processed/jv_parsed/<run_id>/`
  - 生ファイルの棚卸し
- `data/processed/normalized/`
  - `races.csv`
  - `runners.csv`
  - `results.csv`
  - `horses_latest.csv`

## 3. Datasets

- `data/datasets/train/`
  - ベースライン学習用データ
- `data/datasets/inference/weekly/`
  - 当週推論スナップショット
- `data/templates/weekly_entry_template.csv`
  - 当週推論入力テンプレート

## Entry Points

### Raw差分取得

```powershell
<python> -m src.pipelines.update_raw_data
```

### 学習用データ再構築

```powershell
<python> -m src.pipelines.build_train_dataset
```

### ベースライン再学習

```powershell
<python> -m src.pipelines.run_retrain
```

### 当週推論データ構築

```powershell
<python> -m src.pipelines.build_weekly_inference_dataset
```

### TARGETの最新出馬表CSVを取り込み

```powershell
<python> -m src.pipelines.import_latest_target_entry
```

列マッピングの確認:

- [WEEKLY_SNAPSHOT_SOURCE_MAP.md](C:/Users/kazup/OneDrive/デスクトップ/keiba_ai_project/docs/WEEKLY_SNAPSHOT_SOURCE_MAP.md)
- [weekly_snapshot_column_map.csv](C:/Users/kazup/OneDrive/デスクトップ/keiba_ai_project/config/weekly_snapshot_column_map.csv)

### 日次推論更新

```powershell
<python> -m src.pipelines.run_daily_inference
```

## Current MVP Scope

- JV本番APIへの直接接続はまだ未実装
- まずは `data/inbox/jv/` に置かれた取得物を差分アーカイブする
- 正規化は既存の `全競走馬成績.csv` を土台に作る
- 当週推論は `entry_snapshot.csv` を置けば流せる

## Operational Notes

- カラム意味は必ず JV-Data仕様書PDF / Excel定義書で確認する
- 学習用特徴量と当週推論用特徴量は混ぜない
- オッズ、馬体重、馬場傾向は時点スナップショットで管理する
- 生データは必ず残し、変換後だけで運用しない
