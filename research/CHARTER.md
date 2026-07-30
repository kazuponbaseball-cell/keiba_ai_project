# 競馬AI Research OS v1 憲章

- 制度: Level 3（承認付き自律研究）
- 発効日: 2026-07-30
- 基準コミット: `288dff5e86385908281428d5ed4f077625a43e4b`
- 初期ブランチ: `chore/research-os-v1`
- 本PRの範囲: 研究ガバナンス、experiment lifecycle、確率契約検査
- 本PRの範囲外: 予測モデル、候補選択、value判定、正式BUYロジック

## 1. 目的

購入可能時点までの情報だけを使い、chronological outer OOSで再現する長期ROI改善を研究する。単一期間の高ROIや大当たり依存を成果とみなさず、確率品質、安定性、再現性、価格結合前の候補独立性を同時に評価する。

## 2. Level 3の意味

Agentは、研究提案、read-only監査、score作成、experiment draftを自律的に行える。人間がscopeを承認した後は、承認済みresearch branch内でbacktest、contract audit、shadow artifact作成を自律実行できる。

Agentは次を行えない。

- 正式BUYの再開、stake決定、注文送信
- 現行の予測モデル、候補選択、value判定、正式BUYロジックの変更
- `main`への直接commit、PRの自動merge、本番promotion
- 結果閲覧後の同一experiment IDによる仮説・fold・gate変更

本番反映とmergeは、常に人間の明示承認を必要とする。

## 3. 研究境界

```text
as-of input
  -> non-odds features
  -> Top3 set softmax
  -> non-odds candidate freeze + digest
  -> chronological outer OOS evaluation
  -> post-freeze price/result join
  -> diagnostic ROI
  -> human review
  -> shadow only
```

価格と払戻は候補freeze後の評価にだけ利用できる。価格結合後に候補key、rank、tier、coverageを変えてはならない。

## 4. 絶対契約

### 4.1 Top3集合確率

取消反映後のrunner universeを `U_r`、非順序3頭集合を `S_r = C(U_r, 3)` とする。canonical probabilityは集合utilityのsoftmaxである。

```text
q_r(s) = exp(u_r(s) / T) / sum(exp(u_r(t) / T) for t in S_r), T > 0
```

- `q_r(s)`はfiniteかつ `0 <= q_r(s) <= 1`。
- 取消後runner universeの全 `C(|U_r|,3)` 集合を一度ずつ含める。
- runner universeは確率artifactとは独立にfreezeし、runner count/hashを契約検査へ渡す。観測集合のunionだけを完全性証明にしない。
- 各raceで `sum(q_r(s)) = 1`、許容誤差は `1e-10`。
- wide probabilityは `p_r(a,b) = sum(q_r(s) for s containing {a,b})` としてのみ導出する。
- 各raceで `sum(p_r(a,b)) = 3`、許容誤差は `1e-10`。
- 独立pair sigmoidをcanonical probabilityの代替にしない。
- mass、finite、重複、runner universeの検査失敗時はfail-closeし、ROIを読まない。

### 4.2 Odds-free candidate

対象raceの単勝・複勝・馬連・ワイド等のodds、人気、market probability、margin、expected ROI、払戻、的中、利益、およびその派生列を候補選択に使用しない。追加、除外、順位、tier、tie-break、coverage、abstentionの全てが対象である。

### 4.3 Chronological outer OOS

- 順序は `train < validation < calibration < outer test`。
- 分割単位はraceで、区間間のrace重複を0とする。
- feature selection、imputation、scaling、category dictionary、hyperparameter、temperature、thresholdはouter testより前だけで決める。
- purge/embargoとfold manifestを事前登録し、baselineとchallengerで共有する。
- outer testの着順、払戻、ROI、失敗分析を見て同じIDを調整しない。

### 4.4 As-of・leakage

事後情報、未来情報、target race後に利用可能になった値、outer result由来の加工、未来値による欠損補完を禁止する。各inputはevent/source/received timeとcontent hashで追跡し、証明できないlineageは未確認としてfail-closeする。

### 4.5 正式BUY停止

Research OSが作る全artifactは `formal_buy=false`、`send_order=false`、`stake=0` とする。研究コードからproduction BUY pathをimportまたは実行しない。正式BUYの再開は本制度の権限外である。

## 5. Hypothesis gate

仮説は `research/HYPOTHESIS_SCORECARD.yaml` で100点満点評価する。

- 75点未満: `BLOCKED_SCORE`。自動実行禁止。
- 75点以上: `PROPOSED`。人間承認までは実行禁止。
- 人間承認済み: `APPROVED_TO_RUN`。承認scope内のみ実行可能。
- 契約違反: `INVALID`。metricsやROIで救済しない。

高得点は本番採用を意味しない。shadow移行と本番反映には別の人間承認が必要である。

## 6. Lifecycle

```text
BLOCKED_SCORE

PROPOSED -> APPROVED_TO_RUN -> RUNNING -> REVIEW_REQUIRED
                                            |-> REJECTED
                                            |-> APPROVED_FOR_SHADOW -> ARCHIVED
any state -> INVALID
```

Registryはappend-only event logとする。承認時点の仮説、score、data、fold、primary metric、gate、code/config hashを変更した場合は新しいIDを発行する。`APPROVED_FOR_PRODUCTION` はResearch OS v1に存在しない。

## 7. Evidence policy

- Tier A: audited commitに含まれるcode/config/docs。
- Tier B: local-only/ignored artifact。暫定証拠で、再現するまでpromotionに使わない。
- Tier C: 文書化された過去claim。背景情報で、採用根拠に使わない。

positive/negative resultを同じ基準で登録する。pooled平均だけ、1 foldだけ、小標本の極端なROI、threshold後付け、大配当集中は採用根拠にならない。

## 8. 即時停止条件

- Top3 massまたはwide mass契約違反
- chronology、race overlap、as-of、lineage違反
- candidate freeze前のmarket/payoff情報利用
- freeze前後のcandidate digest不一致
- 正式BUY、stake、order、production control pathへの接続
- 75点未満、未承認、または承認scope外の実行

## 9. Research OS v1の完了条件

本PRは、ガバナンス文書、state、scorecard、experiment/PR/issue template、append-only registry CLI、確率contract checkerとtestを提供する。既存baselineの再現、共通outer-fold manifest、odds-free candidate generator、正式BUYのproduction側hard-stop証明は後続の承認済み研究課題であり、本PRでは実装しない。
