# Race Intelligence Lite+ Sunday Observation View

`2026-08-23` のWIN5対象5レース・全70頭を、公式馬番順で読むための静的観測ビューです。正式モデル、予測、順位、確率、市場評価、購入経路とは独立しています。

## 安全境界

- Observation-only / read-only static view
- No AI rank / no probability claims
- No odds / popularity / market / EV / stake
- No BUY / Champion / ticket / notification / order hooks
- No training / backtest / outer OOS
- No EXP-033 run
- No EXP-034 real-data execution
- `formal_buy=false`, `send_order=false`, `stake=0`

既存のdashboard/scenario-labは、旧`ai_score`、順位、市場、BUY経路と結合しているためimportもデータ再利用もしていません。専用generatorは標準ライブラリだけを使い、入力列をallowlistで検査します。禁止列が追加された入力はfail-closeします。

## Queue proxy v0.2 snapshot

`docs/observations/race_intelligence_lite_plus_20260823/snapshot_20260822T180200JST_queue_v02/`

v0.2は従来のlead/front頭数表示を、target-context role、role range、履歴観測率、lead dependency evidence、初角位置確保cost、canonical pairwise conflict、expected position bandを含むQueue Pressure Mapへ置き換えます。5レース・全70頭を公式馬番順で表示します。

- `all_recent_role`: v1と同じ直近最大5走の絶対角位置帯
- `target_context_role`: 同surface±200m → ±400m → 同surface → all-recentの固定fallback
- `normalized_position_proxy`: `(position-1)/max(field_size-1,1)`。今回のmarket-excluded historyには過去頭数がないため全頭`unobserved`
- `forward_propensity`: 分子・分母付きの単純履歴観測率。今回のハナ・位置取り確率ではない
- `lead_dependency_evidence`: `dependency_possible / can_rate_observed / mixed / insufficient_evidence`。`need_lead=true`へ変換しない
- `launch_speed`: 全頭`unobserved`。最初の観測角位置と発馬加速を分離
- `pairwise_pressure_matrix`: `horse_a.horse_no < horse_b.horse_no`のcanonical upper triangle、self pairなし
- `queue_pressure_level`: `HIGH / MEDIUM / LOW / UNCERTAIN`。pace taxonomyやHuman Scenarioへ自動変換しない

`Queue Pressure HIGH ≠ FAST`です。QueueはHuman Scenarioの観測材料に限られ、HumanがMIDDLEを選ぶことも許容します。v0.2のrace confidenceはrole-known horse coverageとnormalized-position horse coverageの両方が80%以上の場合だけ`medium`、それ以外は`low`とする固定規則です。今回は過去field size未接続による正規化coverage 0を反映し、全5レース`low`です。

## 保存済みv1 snapshot

`docs/observations/race_intelligence_lite_plus_20260823/snapshot_20260822T132226JST/`

- `race_intelligence_lite_plus.html`: 5レースboard、全70頭card、全頭比較表、Human入力欄
- `race_intelligence_lite_plus_data.json`: 表示内容のcanonical JSON
- `official_current_entries.json`: JRA公式5ページのcurrent field部分だけを抽出した70頭snapshot
- `human_scenario_freeze.template.json`: 5レースのHuman Scenario入力雛形
- `source_manifest.json`: allowlist入力、SHA-256、除外source class、安全定数

HTMLは単一ファイルで、外部CSS/JavaScript、API、`fetch`、XHR、WebSocket、form action、localStorageを使いません。Human入力のexportだけをローカルbrowser内で行います。

## Evidence contract

各証拠cellは次を区別します。

| status | 意味 |
|---|---|
| `observed` | 入力に直接存在する事実 |
| `derived` | 表示仕様に明記した透明な変換 |
| `proxy` | 直接測定ではない代替観測 |
| `unobserved` | 未接続・未定義。missing reason必須 |

Routeは `exact / partial / similar / unclassified / unobserved`、confidenceは `high / medium / low / unobserved` です。`unobserved`をneutral、0、有利・不利へ補完しません。

能力帯は直近最大5走の過去着順を帯化した記述であり、今回の出走馬間の能力順位ではありません。v1 roleは直近最大5走の最初に観測された角位置の絶対位置帯proxyです。v0.2ではこれを後方互換表示として残し、target-context層とvariation rangeを別表示します。

v0.2は`need_lead`を出力しません。直近最大5走のlead/nonleadとvalid finish 1–3着を透明に数え、肯定証拠が限られる場合も`dependency_possible`に留めます。固定長データの非完走sentinel（100/300/400、99.9秒）は着順・上がり評価から除外し、「非完走等」としてだけ表示します。

Scenario sensitivityのSLOW/MIDDLE/FASTは、独立taxonomyと個体responseが凍結されていないため全頭を`unobserved`としています。既存の`actual_lap_mode`はpaceとshapeを混在するため、S/M/Fへ変換していません。

Transferの`observed race shape`も、history scopeにはRA joinフラグだけがありlap配列・独立shape labelがないため`unobserved`です。RA join済みという事実と、shape欠損を同じstatusへ混ぜません。馬cardのoverall confidenceはこの重要欠損を反映して既走馬でも`low`を上限とし、履歴本数の充足度は別の`history coverage confidence`で示します。

## Data sources

使用したread-only sourceは`source_manifest.json`にpath、bytes、SHA-256を固定しています。

- WIN5 race-name manifest（5 legs）
- resolved official race IDs
- audited declared runner universe（70頭、ID exact join）
- market-column-excluded pre-target history（826 observed runs + debut sentinel）
- horse intelligence readiness / route coverage
- same-condition evidence coverage
- route requirement cards
- 2026-08-22 18:02:00 JSTに再取得したJRA公式current entry field（v0.2）

JRA HTMLはShift_JISとして読み、最初の`div.record_unit`より前だけを対象にし、各rowを最初の`td.past`より前で切っています。過去走人気・オッズは抽出しません。保存済み8月21日snapshotとのidentityは`race_id+horse_id`で70/70一致し、現在は枠・馬番・性齢・騎手・調教師・斤量が70/70観測済みです。

旧監査snapshotの地方所属馬1頭では、linkなし調教師cellを騎手として保持していた差分がありました。current official fieldの「笹野 博司」を表示し、差分を`official_current_entries.json`へ残しています。identity joinは変更していません。

## Build

生成器は入力の自動探索や「latest」選択を行いません。すべてのpathと時刻を明示します。

```powershell
python scripts/research/build_race_intelligence_lite_plus.py capture-official `
  --targets <resolved_win5_targets_predraw.csv> `
  --entries <declared_without_draw_entry_snapshot.csv> `
  --fetched-at-jst 2026-08-22T18:02:00+09:00 `
  --output <snapshot>/official_current_entries.json

python scripts/research/build_race_intelligence_lite_plus.py build `
  --target-manifest <win5_target_race_names.json> `
  --route-cards <win5_route_requirement_cards.json> `
  --targets <resolved_win5_targets_predraw.csv> `
  --entries <declared_without_draw_entry_snapshot.csv> `
  --history <predraw_history_scope.csv> `
  --coverage <predraw_same_condition_evidence_coverage.csv> `
  --readiness <predraw_horse_intelligence_readiness.csv> `
  --route-coverage <predraw_route_coverage.csv> `
  --official-entries <snapshot>/official_current_entries.json `
  --generated-at-jst 2026-08-22T18:02:00+09:00 `
  --output-dir <snapshot>
```

`capture-official`だけが明示的なJRA GETを行います。`build`、`verify`、`freeze`はnetworkを使いません。既存出力は`--overwrite`なしでは上書きしません。

## Human Scenario freeze

1. HTMLで5レースすべてのMain scenario、confidence、reasonを入力します。
2. 当日馬場、風、取消、day biasはrace-day notesへ記入します。
3. 必要ならmain danger horseを人が指定します。人気・オッズからの自動指定はありません。
4. 「Human Freeze入力JSONをexport」で`human_scenario_freeze.input.json`を保存します。
5. 最初の発走前に次を実行します。

```powershell
python scripts/research/build_race_intelligence_lite_plus.py freeze `
  --observation <snapshot>/race_intelligence_lite_plus_data.json `
  --human-input <human_scenario_freeze.input.json> `
  --output-dir <new-freeze-directory>
```

Freeze timestampはCLI実行時のsystem clockからJSTで取得します。利用者指定時刻によるbackdateはできません。

Freezeは次をfail-closeします。

- 5レースの未入力または重複
- observation SHA-256不一致
- SLOW/MIDDLE/FAST以外
- confidence / reason不足
- post time変更
- いずれかの発走時刻以後
- 既存freeze directoryへの上書き

生成する`human_scenario_freeze.json`と`freeze_manifest.json`には時刻とSHA-256を固定します。開催後レビューはHTMLから別JSONへexportし、pre-race freezeを変更しません。Review exportにはobservation SHA-256、人が入力するfreeze manifest SHA-256、記録時刻を含めます。

Artifact検証はJSONからHTMLを再renderしてbyte一致を確認し、source manifestのoutput hashとも照合します。

```powershell
python scripts/research/build_race_intelligence_lite_plus.py verify `
  --observation <snapshot>/race_intelligence_lite_plus_data.json `
  --html <snapshot>/race_intelligence_lite_plus.html `
  --source-manifest <snapshot>/source_manifest.json `
  --official-entries <snapshot>/official_current_entries.json `
  --freeze-template <snapshot>/human_scenario_freeze.template.json
```

## Known limitations

- Exact routeは全5レース0。renovation/versionが未解決であり、「一致なし」の意味ではありません。
- Similarは全5レース0。similarity metric未凍結であり、「類似なし」の意味ではありません。
- Full curve geometryは未整備。course descriptionは定性表示だけです。
- market-excluded historyに過去`field_size`がないため、normalized positionとfront-quartile observed rateは全頭`unobserved`です。対象頭数や角位置最大値から推測していません。
- launch/個体200m start speedは未観測です。first observed corner positionをstart accelerationとして扱いません。
- WIN4はroute cardに初角までの距離を安全に分類できる記述がなく、first-turn position costは全頭`unobserved`です。
- Queue pairは観測role/rateに基づく定性proxyで、誰が速い・誰がハナを取るかを断定しません。
- target個体調教joinは未実行のためconditionは原則`unobserved`です。
- RA coverageはsame-condition race-levelで100%ですが、個体別sectionalではありません。
- 当日馬場、day bias、風はHuman freeze前の入力待ちです。
- favorite collapseはHuman指定欄とrole構造だけです。市場favoriteは取得しません。
- weighted trait synthesis、正確なroute version、day-bias/wind自動接続は未実装です。
