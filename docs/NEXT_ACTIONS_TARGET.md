# Next Actions For TARGET

今の状態なら、次はこの順で進めればよい。

## 1. TARGETから出馬表CSVを1本出力

出力先:

- `C:\Users\kazup\OneDrive\デスクトップ\keiba_ai_project\data\inbox\target\entries`

## 2. 取り込み実行

```powershell
C:\Users\kazup\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m src.pipelines.import_latest_target_entry
```

## 3. 当週推論更新まで一気に実行

```powershell
powershell -ExecutionPolicy Bypass -File "C:\Users\kazup\OneDrive\デスクトップ\keiba_ai_project\scripts\run_target_weekly_update.ps1"
```

## 4. もし列名差分で止まったら

- `config/target_entry_aliases.json`
  - TARGET実CSVの列名に合わせて候補を追加する

## 5. 現在のフェーズ

- フェーズ1: ベースライン学習土台
- フェーズ2: 分析・評価土台
- フェーズ3: TARGET取り込み土台
- フェーズ4: 当週推論の自動更新
- フェーズ5: UI / 買い目 / 期待値

現在は **フェーズ3完了間近、フェーズ4の入口**。
