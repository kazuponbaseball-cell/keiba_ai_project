# 競馬AI Research OS v1 憲章

- 制度: Level 3（承認付き自律研究）
- 発効日: 2026-07-31
- 基準コミット: `288dff5e86385908281428d5ed4f077625a43e4b`
- 初期ブランチ: `chore/research-os-v1`
- 本PRの範囲: 研究ガバナンス、experiment lifecycle、確率契約検査
- 本PRの範囲外: 予測モデル、候補選択、value判定、正式BUYロジック

## 1. 目的

購入可能時点までの情報だけを使い、chronological outer OOSで再現する長期ROI改善を研究する。単一期間の高ROIや大当たり依存を成果とみなさず、確率品質、安定性、再現性、価格結合前の候補独立性を同時に評価する。

## 2. Level 3の意味

Agentは、研究提案、read-only監査、score作成、experiment draftを自律的に行える。GitHub上でproposal scopeが`APPROVED_TO_PREPARE`された後はresearch branch上の実装とsynthetic fixture testを行える。実装後のexact run scopeが別途`APPROVED_TO_RUN`され、同じ承認commentを`RUNNING`直前に再検証した場合だけ、承認scope内の実データ実験を行える。

Agentは次を行えない。

- 正式BUYの再開、stake決定、注文送信
- 現行の予測モデル、候補選択、value判定、正式BUYロジックの変更
- `main`への直接commit、PRの自動merge、本番promotion
- 結果閲覧後の同一experiment IDによる仮説・fold・gate変更
- 自己申告flag、caller指定actor、Codex、bot、automationによる自己承認

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
- 75点以上: `PROPOSED`。GitHub上の承認証拠までは準備・実行禁止。
- `APPROVED_TO_PREPARE`: 実装準備とsynthetic fixture testのみ可能。
- `APPROVED_TO_RUN`: exact code/config/data/fold/command scopeに対する実行承認。
- `APPROVED_FOR_SHADOW`: run承認とは別のreview digest承認。
- 契約違反: `INVALID`。metricsやROIで救済しない。

高得点は本番採用を意味しない。shadow移行と本番反映には別の人間承認が必要である。

## 6. Lifecycle

```text
BLOCKED_SCORE

PROPOSED
  -> APPROVED_TO_PREPARE
  -> PREPARING
  -> RUN_APPROVAL_REQUIRED
  -> APPROVED_TO_RUN
  -> RUNNING
  -> REVIEW_REQUIRED
       -> REJECTED
       -> APPROVED_FOR_SHADOW
       -> INVALID
```

全状態から契約違反時に`INVALID`へ移行できる。`INVALID`はterminalである。

Registryはappend-only event logとする。`APPROVED_FOR_PRODUCTION`、merge承認、BUY承認はResearch OS v1に存在しない。

### 6.1 GitHub-backed approval

承認commentは次のexact形式とする。

```text
APPROVED_TO_PREPARE <proposal_scope_digest>
APPROVED_TO_RUN <run_scope_digest>
APPROVED_FOR_SHADOW <review_digest>
```

- author loginはproposalのbase commit上のtracked
  `research/APPROVERS.json`に含まれなければならない。
- base commitが`origin/main` history上にあることを検証する。
- experiment branch上で変更されたallowlistを信用しない。
- actor typeが`User`でない、またはCodex/bot/automation loginなら拒否する。
- comment ID、URL、author、created_at、updated_at、body SHA-256をregistryへ保存する。
- `RUNNING`直前に同じcommentをread-only GETし、author、body、digest、
  updated_atを再検証する。
- GitHub確認不能、comment欠落、編集、unauthorized authorはfail-closeする。
- CIは外部通信せずfixtureまたはinjected providerを使う。

### 6.2 Canonical scope digest

Markdownは承認scopeの正本ではない。canonical JSONをUTF-8、key sort、
compact separator、Unicode保持、NaN/Infinity禁止でserializeし、
SHA-256化する。setとして扱うproposal listは重複禁止・Unicode順sortし、
execution command listは順序を保持する。

proposal scopeは最低限、experiment ID/title、仮説/null、作用機序、
対象母集団、in/out scope、expected paths、raw sources、data as-of、
allowed/forbidden columns、lineage/hash、chronological fold、fold path/hash、
purge/embargo、primary metric、required effect、rejection gate、stop、
compute budget、variant/threshold上限、base commit、score、安全flagを含む。

run scopeはproposal scope全体とdigestに加え、exact execution commit、
config hash、data/input manifest hash、fold manifest hash、runner-universe
manifest hash、dependency/environment manifest、seed、exact commandsを含む。

承認後にproposal/run scope、実行commit、manifest hash、commands、安全flagが
変わった場合は`RUNNING`へ進めない。新しいexperiment IDまたは再承認を
必要とする。実行後に生成されるresult、candidate digest、price join artifact
はrun scopeから分離してappendする。
hash-bound lifecycle/manifest以外のuncommitted・untracked pathも拒否し、
worktree codeをexact execution commitの一部とみなさない。

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
- 75点未満、GitHub承認証拠なし、または承認scope外の準備・実行
- approval commentの編集、GitHub確認不能、base-commit allowlist不一致
- proposal/run digest、execution commit、config/data/fold/commandの変更

## 9. Research OS v1の完了条件

本PRは、ガバナンス文書、state、scorecard、experiment/PR/issue template、GitHub-backed二段階承認、canonical proposal/run digest、append-only registry CLI、確率contract checkerとtestを提供する。既存baselineの再現、共通outer-fold manifest、odds-free candidate generator、正式BUYのproduction側hard-stop証明は後続の承認済み研究課題であり、本PRでは実装しない。
