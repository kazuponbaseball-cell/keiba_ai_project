# Weekly Snapshot Source Map

当週推論スナップショット `entry_snapshot.csv` の列について、JV/PC-KEIBA 定義書で確認できたソースと、派生が必要な列を分けて整理した。

機械可読版:

- [weekly_snapshot_column_map.csv](C:/Users/kazup/OneDrive/デスクトップ/keiba_ai_project/config/weekly_snapshot_column_map.csv)

## 確認済みの主ソース

- `レース詳細`
  - `grade_code`
  - `kyoso_shubetsu_code`
  - `kyoso_joken_meisho`
  - `kyori`
  - `track_code`
  - `hasso_jikoku`
  - `toroku_tosu`
  - `shusso_tosu`
  - `tenko_code`
  - `babajotai_code_shiba`
  - `babajotai_code_dirt`
  - `lap_time`
  - `zenhan_3f`
  - `kohan_3f`

- `馬毎レース情報`
  - `ketto_toroku_bango`
  - `bamei`
  - `barei`
  - `chokyoshi_code`
  - `chokyoshimei_ryakusho`
  - `futan_juryo`
  - `kishu_code`
  - `kishumei_ryakusho`
  - `bataiju`
  - `zogen_fugo`
  - `zogen_sa`
  - `ijo_kubun_code`
  - `kakutei_chakujun`
  - `soha_time`
  - `corner_1`-`corner_4`
  - `tansho_odds`
  - `tansho_ninkijun`
  - `kohan_3f`
  - `time_sa`

- `速報系データ レース詳細`
  - `kyori`
  - `track_code`
  - `hasso_jikoku`
  - `tenko_code`
  - `babajotai_code_shiba`
  - `babajotai_code_dirt`

- `速報系データ 馬毎レース情報`
  - `umaban`
  - `futan_juryo`
  - `kishu_code`
  - `kishumei_ryakusho`
  - `bataiju`
  - `zogen_fugo`
  - `zogen_sa`
  - `tansho_odds`
  - `tansho_ninkijun`

- `天候馬場状態`
  - `tenko_code`
  - `babajotai_code_shiba`
  - `babajotai_code_dirt`

- `馬体重`
  - `happyo_tsukihi_jifun`
  - `bataiju_joho_01`-`bataiju_joho_20`

- `競走馬マスタ`
  - `ketto_toroku_bango`
  - `bamei`
  - `seinengappi`
  - `seibetsu_code`

- `騎手マスタ`
  - `kishu_code`
  - `kishumei`
  - `kishumei_ryakusho`

- `調教師マスタ`
  - `chokyoshi_code`
  - `chokyoshimei`
  - `chokyoshimei_ryakusho`

## 派生が必要な列

以下は JV の単一カラムではなく、このプロジェクト側で作る列。

- `レースID(新/馬番無)`
  - `kaisai_nen + kaisai_tsukihi + keibajo_code + race_bango` から固定規則で生成
- `場所`
  - `keibajo_code` をコード表から復元
- `芝・ダ`
  - `track_code` をコード表から芝/ダートに変換
- `クラス名`
  - `grade_code + kyoso_joken_meisho + juryo_shubetsu_code` を整形
- `前走*` 系
  - 同一 `血統登録番号` の直近過去走から派生

## まだ未確定の列

既存CSVにはあるが、今回確認した範囲では JV の単一フィールドにまだ落ちていないもの。

- `前走Ave-3F`
- `前PCI`
- `前走PCI3`
- `前走RPCI`
- `前走上3F地点差`

ここは **既存CSVの生成ロジックを逆算するか、JV仕様書PDFで追加確認してから実装** する。
