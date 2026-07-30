# 競馬AI 研究憲章 — Level 3（承認付き開発自律）

- 文書版: `v1.0`
- 発効日: `2026-07-30`
- 基準線: `BASE-20260730`
- 基準コミット: `288dff5e86385908281428d5ed4f077625a43e4b`
- 初期研究ブランチ: `research/autonomy-v1`
- 適用範囲: 研究提案、実験コード、履歴バックテスト、shadow出力、研究artifact
- 適用外: 現行BUYロジック、賭け金、注文送信、`main` への直接反映

## 1. 使命

この研究の目的は、的中率や単一期間のROIを最大化することではなく、購入可能時点の情報だけを使い、将来期間で再現する長期ROIを改善することである。

改善とは、次を同時に満たすことを指す。

1. 確率契約とデータ時点契約を破らない。
2. chronological outer OOSでベースラインより改善する。
3. 一部の高配当、競馬場、期間だけに利益が集中しない。
4. 候補選択はオッズから独立している。
5. 現行BUYを変更せず、承認前はshadowに留まる。

## 2. 研究面と本番面の境界

研究面は次の順序を必須とする。

```text
as-of入力
  -> 非オッズ特徴量
  -> Top3集合確率
  -> 非オッズ候補のfreeze + hash
  -> outer OOS評価
  -> freeze後の価格・払戻結合
  -> ROI/価格保持の診断
  -> 人間レビュー
  -> shadow承認
```

本番面の `scripts/build_current_strongest_tickets.py` と関連する運用scriptはcharacterization baselineとして凍結する。現行実装がオッズ、期待ROI、marginを最終BUYに用いることは本憲章の変更対象ではない。研究候補生成にそのmarket-aware選別を流用してはならない。

## 3. 変更不能の契約

### 3.1 ブランチ・本番保護

- `main` へ直接コミットしない。
- 研究は `research/*` ブランチで行い、ドラフトPRを経由する。
- 現行BUY、賭け金、注文、通知、live credentialを研究PRから変更しない。
- 研究PRの差分に本番BUY関連ファイルが含まれる場合はfail-closeとする。
- 既存の作業ツリー変更を暗黙にstageしない。commit対象は明示パスに限定する。
- 自動merge、自動production promotion、自動購入を行わない。

### 3.2 Top3集合softmax確率契約

出走取消反映後の1レースのrunner universeを `U_r`、3頭の非順序集合全体を `S_r = C(U_r, 3)` とする。canonical Top3確率は集合utility `u(s)` から次で得る。

```text
q_r(s) = exp(u(s) / T) / sum_{t in S_r} exp(u(t) / T)
```

必須invariantは次のとおり。

- `T > 0` で、temperatureはouter testより前のcalibration期間だけで決める。
- `q_r(s)` はfiniteかつ `0 <= q_r(s) <= 1`。
- 各raceで `sum_s q_r(s) = 1`。許容誤差は `1e-10` 以下。
- 正解Top3集合は各raceでちょうど1行。
- canonical wide marginalは `p_wide(a,b) = sum_{s: {a,b} subset s} q_r(s)`。
- 各raceで `sum_{a<b} p_wide(a,b) = 3`。許容誤差は `1e-10` 以下。
- horse Top3 marginalのrace内総和も `3`。
- 取消・除外後に集合universeを再構築し、存在しないrunnerへ確率を残さない。
- 独立pair sigmoidをcanonical確率の代替にしない。challengerは集合確率から導出するか、同じmass契約へ射影する。
- mass、marginal、非負性、runner universeのいずれかが失敗したrunは無効とし、ROIを読まない。

### 3.3 Chronological outer OOS契約

すべてのmodel・feature・threshold研究で、期間を次の順序に固定する。

```text
train < validation < calibration < outer test
```

- raceを分割単位とし、同一raceを複数区間へ入れない。
- `max(train_date) < min(validation_date) < min(calibration_date) < min(test_date)` をassertする。
- purge/embargo期間を事前登録し、同一のfold manifestをbaselineとchallengerで共有する。
- feature選択、欠損処理、標準化、カテゴリ辞書、hyperparameter、temperature、閾値はouter testより前のデータだけで決める。
- outer testの着順、払戻、ROI、誤差分析を見て同じexperiment IDを調整しない。変更時は新しいIDと新しい承認が必要。
- 複数のouter foldを結合して選択してはならない。結合値は全fold確定後の報告専用とする。
- 開発OOSを繰り返し見たデータはprospective OOSと呼ばない。
- outer OOSが空、時系列逆転、race重複、as-of違反のrunはfail-closeとする。

### 3.4 オッズ非依存候補契約

候補の追加、除外、順位、tier、tie-break、coverage、abstentionには、対象raceの次を使わない。

- 単勝・複勝・馬連・ワイド等の現在値、最終値、時系列値
- 人気、market probability、odds margin、expected ROI、steam/drift
- 払戻、的中、利益
- 上記から派生した特徴量、proxy、欠損flag

過去raceの人気・市場由来特徴も既定では禁止する。使用する場合は独立した仮説、lineage、as-of根拠を事前登録し、主任研究者と人間approverの双方が明示承認する。

候補manifestには少なくとも `race_id`、候補key、rank、tier、model/config/data/fold hash、`frozen_at` を含める。価格結合の前後で候補key・rank・tierのdigestが100%一致しなければrunを無効にする。

オッズは候補freeze後に限り、次の目的で使用できる。

- 実現可能な価格の記録
- ROI、slippage、価格保持、liquidityの評価
- 凍結済み現行BUYのcharacterization

価格結合後に研究候補を変更してはならない。

### 3.5 As-of・lineage契約

- 各入力に `event_time`、`source_time`、`received_at`、source ID、content hashを持たせる。
- target raceの候補freeze時刻より後に利用可能になった値を特徴量へ入れない。
- 同日情報は対象raceより前に確定したraceだけを使う。
- 欠損を将来値で埋めない。fallback規則はtrain期間で固定する。
- schema、単位、race/runner universe、取消状態をassertし、必須入力不足をwarningで継続しない。

### 3.6 再現性契約

有効なrunは次を記録する。

- experiment ID、承認者、承認時刻
- base commit、result commit、branch
- code/config/data/fold/candidate/BUY-controlのSHA-256
- Pythonと依存packageのversion
- random seedと実行command
- 入力期間、as-of時刻、race数、candidate数
- stdout/stderr、contract audit、metrics、artifact URI

同じmanifestから同じcandidate digestと許容誤差内のmetricsを再生成できないrunはpromotion対象外とする。

## 4. Level 3の権限

Level 3は「承認された範囲をagentが自律実行できる」ことを意味し、「本番を自律変更できる」ことを意味しない。

| 行為 | Agentの権限 | 人間承認 |
|---|---|---|
| read-only監査、仮説整理、experiment draft | 自律可 | 不要 |
| research branch上のdocs・test・shadow code作成 | 承認済みscope内で可 | 実行前に必要 |
| 登録済みconfigのbacktest・再実行 | 自律可 | `APPROVED_TO_RUN` が必要 |
| 結果確定後の仮説・fold・gate変更 | 不可 | 新IDとして再承認 |
| shadow artifact生成 | 可 | `APPROVED_FOR_SHADOW` が必要 |
| 現行BUY・stake・注文・通知の変更 | 不可 | 別PRと明示承認が必要 |
| `main` merge・production promotion | 不可 | Repository ownerが実行 |
| secret変更・live注文・有償外部操作 | 不可 | 本憲章の範囲外 |

## 5. 承認ライフサイクル

Registryのstatusは次を使用する。

```text
PROPOSED
  -> APPROVED_TO_RUN
  -> RUNNING
  -> REVIEW_REQUIRED
  -> APPROVED_FOR_SHADOW
  -> ARCHIVED
```

任意の段階から `REJECTED`、契約違反時は `INVALID` に遷移できる。

- `APPROVED_TO_RUN` と `APPROVED_FOR_SHADOW` は人間だけが設定する。
- Agentは `RUNNING` と `REVIEW_REQUIRED` を更新できる。
- `APPROVED_FOR_PRODUCTION` はこのRegistryでは設定しない。別のproduction PRと承認記録を必要とする。
- 承認後に入力、feature block、split、primary metric、gateを変えた場合、承認は失効する。

## 6. 実験手順

1. `research/EXPERIMENT_TEMPLATE.md` を複製し、仮説と停止条件を事前登録する。
2. `research/REGISTRY.csv` に1行追加し、statusを `PROPOSED` にする。
3. 人間approverがscope、計算量、データ、gateを確認する。
4. `APPROVED_TO_RUN` 後にbranch上で実装・実行する。
5. contract auditを先に評価し、失敗時はmetrics計算前に停止する。
6. baselineとchallengerを同じouter fold、候補coverage、価格snapshotで比較する。
7. 結果を追記し、statusを `REVIEW_REQUIRED` にする。
8. 人間レビュー後、shadowのみ承認できる。現行BUYへの接続は別案件とする。

## 7. 評価原則

- primary metricは実験前に1つ決める。Top3 modelはset NLL、pair rankingはMRR/recall、価格検証はprospective ROIなど、目的に対応させる。
- ROIと確率品質を同時に報告するが、ROIを見てmodel、feature、thresholdを選ばない。
- fold別、半期別、競馬場別、頭数別、欠損別の分解を添える。
- raceまたは開催日cluster bootstrapで不確実性を報告する。
- 最大drawdown、最大配当除外、上位3配当除外、利益集中率、threshold sensitivityを報告する。
- pooled平均だけの改善、1foldだけの改善、小標本の極端なROIはpromotion根拠にしない。
- negative resultもRegistryへ残し、同じ仮説の無断再探索を防ぐ。

## 8. 即時停止条件

次のいずれかでrunを `INVALID` とする。

- Top3 probability contract違反
- outer OOSの時系列逆転、race重複、outer情報利用
- 候補freeze前のオッズ・人気・払戻利用
- 価格結合前後のcandidate digest不一致
- 現行BUY関連diffまたはlive side effect
- as-of違反、必須lineage/hash欠落
- 未承認のscope変更

## 9. 完了の定義

研究基盤v1は、次が揃ったときに運用可能とする。

- clean checkoutからbaselineを再現できる。
- Top3集合softmax contractがtracked testでfail-closeする。
- 共通chronological outer-fold manifestが固定される。
- オッズ非依存candidate manifestとdigest auditが動く。
- Registryとapproval lifecycleが実際のrunへ接続される。
- 現行BUYのsource/config/fixture hashが変化していないことを検証できる。
- strict T-3のprospective Grade-Oデータが最低基準まで蓄積される。
