# TARGET Race Import

TARGET のレース詳細系CSVから、レース単位のスナップショットを作る入口。

## 追加ファイル

- [target_race_aliases.json](C:/Users/kazup/OneDrive/デスクトップ/keiba_ai_project/config/target_race_aliases.json)
- [import_target_race_csv.py](C:/Users/kazup/OneDrive/デスクトップ/keiba_ai_project/src/pipelines/import_target_race_csv.py)
- [import_latest_target_race.py](C:/Users/kazup/OneDrive/デスクトップ/keiba_ai_project/src/pipelines/import_latest_target_race.py)

## 受けたい列

- `レースID(新/馬番無)`
- `日付`
- `場所`
- `レース名`
- `芝・ダ`
- `距離`
- `馬場状態`
- `頭数`
- `出走頭数`
- `走破タイム`
- `Ave-3F`
- `上り3F`
- `PCI`
- `PCI3`
- `RPCI`
- `前3F`
- `前4F`
- `後3F`
- `後4F`
- `1角通過順`-`4角通過順`

## 置き場所

- `data/inbox/target/races/`

## 実行

```powershell
<python> -m src.pipelines.import_latest_target_race
```

## ねらい

このレーススナップショットが入ると、

- 馬の好走/凡走
- 前半の速さ
- 後半の質
- コーナー通過
- 馬場状態

をレース単位で結びつけて分析できる。
