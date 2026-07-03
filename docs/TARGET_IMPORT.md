# TARGET Import

`TARGET frontier JV` の出馬表CSV形式を、このプロジェクトの `entry_snapshot.csv` に変換する入口。

## 追加ファイル

- [target_entry_aliases.json](C:/Users/kazup/OneDrive/デスクトップ/keiba_ai_project/config/target_entry_aliases.json)
- [import_target_entry_csv.py](C:/Users/kazup/OneDrive/デスクトップ/keiba_ai_project/src/pipelines/import_target_entry_csv.py)
- [import_latest_target_entry.py](C:/Users/kazup/OneDrive/デスクトップ/keiba_ai_project/src/pipelines/import_latest_target_entry.py)

## 使い方

```powershell
<python> -m src.pipelines.import_target_entry_csv --input-csv "path\\to\\TARGET_export.csv"
```

既定では出力先は以下。

- `data/datasets/inference/weekly/entry_snapshot.csv`

TARGETの最新CSVを自動で拾う場合:

```powershell
<python> -m src.pipelines.import_latest_target_entry
```

この場合は以下のフォルダから最新CSVを拾う。

- `data/inbox/target/entries/`

## 方針

- 同名列はそのまま採用
- 差分は `config/target_entry_aliases.json` の候補列から補完
- 足りない列は空欄で残し、`all_null_required_columns` に出す

## 現時点の前提

- `全競走馬成績.csv` にある TARGET 由来の列構成を橋渡しの基準にしている
- `前走*` の列が TARGET 出馬表CSVに含まれていれば、そのまま流せる
- 実サンプル1本で列名差が分かれば、別名マップをすぐ追加できる
