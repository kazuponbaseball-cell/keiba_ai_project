# EXP-XXXX: 実験名

> このファイルを `research/experiments/EXP-XXXX.md` へ複製して使用する。結果を見た後に事前登録欄を書き換えない。変更が必要なら新しいexperiment IDを発行する。

## 0. Metadata

| 項目 | 値 |
|---|---|
| Experiment ID | `EXP-XXXX` |
| Backlog ID | `AUT-XXX` |
| Status | `PROPOSED` |
| Owner |  |
| Reviewer |  |
| Approver |  |
| Approval timestamp |  |
| Branch | `research/...` |
| Base commit |  |
| Baseline ID | `BASE-20260730` |
| Created at |  |
| Data as-of |  |
| Compute budget |  |
| Expected completion |  |

### Approval record

- [ ] `research/REGISTRY.csv` に登録した。
- [ ] 仮説、feature block、fold、primary metric、gateを結果閲覧前に固定した。
- [ ] 人間approverが `APPROVED_TO_RUN` を記録した。
- [ ] 現行BUY関連pathがscope外であることを確認した。

承認コメント:

```text

```

## 1. Research question

1文で、何を識別したいかを書く。

```text

```

## 2. Hypothesis

### Mechanism

なぜこのsignalがTop3集合確率、pair順位、abstention、またはfreeze後の価格保持を改善し、最終的にROIへつながるのかを書く。

### Null hypothesis

```text
同一outer OOS条件で、challengerはbaselineより改善しない。
```

### Falsification

仮説を棄却する観測を事前に書く。

```text

```

## 3. Scope

### In scope

- _記入する_

### Out of scope

- 現行BUYロジックの変更
- stake・注文・LINE通知の変更
- outer testを見た後のfeature・threshold探索
- freeze前のオッズ・人気・払戻利用

### Files expected to change

```text

```

### Production control paths

次のdiffは0でなければならない。

```text
scripts/build_current_strongest_tickets.py
scripts/apply_live_runtime_safety_overlay.py
scripts/run_current_strongest_line_update.ps1
```

## 4. Data and as-of contract

| Dataset | Role | Period | Event time | Available/received time | Content hash | Allowed columns |
|---|---|---|---|---|---|---|
|  | train |  |  |  |  |  |
|  | validation |  |  |  |  |  |  |
|  | calibration |  |  |  |  |  |  |
|  | outer test |  |  |  |  |  |  |

### Runner universe

- 取消反映時点:
- 最低頭数:
- race key:
- runner key:
- 重複・欠落時の処理:

### Forbidden lineage

- [ ] target raceのodds・人気・払戻をmodel/candidate matrixから除外した。
- [ ] `odds|popularity|market|payout|pay|roi` とその派生列をtaint auditした。
- [ ] 過去市場proxyを使わない。使う場合は下記例外を承認した。
- [ ] target race後に利用可能になった値を除外した。
- [ ] outer result由来のfeature、imputation、thresholdを除外した。

承認済み例外:

```text
なし
```

## 5. Fixed chronological split

fold manifest path:

```text

```

fold manifest SHA-256:

```text

```

| Fold | Train | Validation | Calibration | Outer test | Purge | Embargo |
|---|---|---|---|---|---:|---:|
| fold1 |  |  |  |  |  |  |
| fold2 |  |  |  |  |  |  |
| fold3 |  |  |  |  |  |  |
| fold4 |  |  |  |  |  |  |

必須assert:

- [ ] `train_max < validation_min < calibration_min < test_min`
- [ ] 各区間のrace overlapが0
- [ ] outer test window同士のoverlapが0
- [ ] `NaT` / 不明日付が0
- [ ] model fit endとcalibration endが各test raceより前
- [ ] empty foldをskipせずfailする

## 6. Baseline and challenger

### Baseline

- Model/artifact:
- Config:
- Feature set:
- Candidate policy:
- Artifact SHA-256:

### Challenger

- 変更するfeature blockは1つ:
- Model class:
- Hyperparameter search space:
- Random seeds:
- Missing-value policy:
- Calibration method:

### Fair comparison

- [ ] 同じouter foldsを使う。
- [ ] 同じrunner universeを使う。
- [ ] 同じcandidate coverageまたは事前登録したcoverageを使う。
- [ ] 同じfreeze後price snapshotを使う。
- [ ] baseline/challenger以外の差分がない。

## 7. Top3 probability contract

- [ ] 全 `C(n,3)` 集合を構築した。
- [ ] 正解集合がraceごとに1行。
- [ ] `q(set)` がfiniteかつ `[0,1]`。
- [ ] `max |sum q(set)-1| <= 1e-10`。
- [ ] `p_wide` を `q(set)` から導出した。
- [ ] `max |sum p_wide-3| <= 1e-10`。
- [ ] horse Top3 marginal総和が3。
- [ ] 取消後universeに対して再計算した。
- [ ] failure時にcandidate/ROI評価へ進まず停止した。

## 8. Candidate freeze and price firewall

candidate manifest path:

```text

```

candidate manifest SHA-256:

```text

```

- Freeze time:
- Quote time(s):
- Final pre-close time:
- Candidate key columns:

必須assert:

- [ ] `candidate_uses_odds=false`
- [ ] quote join前にmanifestを永続化しread-backした。
- [ ] quote join前後でcandidate key・rank・tierが完全一致した。
- [ ] oddsを変えたnegative-controlでcandidate hashが不変だった。
- [ ] research outputは `formal_buy=false`、`send_order=false`、`stake=0`。

## 9. Metrics and preregistered gates

### Primary metric

| Metric | Direction | Baseline | Required effect | Aggregation |
|---|---|---:|---:|---|
|  |  |  |  | race/day cluster |

### Secondary metrics

| Family | Metrics |
|---|---|
| Probability | set NLL、wide Brier、ECE、observed/predicted |
| Ranking | top1-set hit、true-set rank、wide MRR、recall@3/5 |
| Economic | ROI、profit、hit rate、coverage、max drawdown |
| Stability | fold/half-year/venue、bootstrap interval、threshold sensitivity |
| Concentration | max payout removed、top3 payouts removed、profit share |
| Operations | missingness、lineage violations、freeze/quote latency |

### Promotion gate

すべての条件を数値で記入する。

```text

```

### Rejection gate

```text

```

### Stop conditions

- probability contract違反
- chronology/race overlap/as-of違反
- forbidden lineage検出
- candidate digest変化
- production path差分
- 未承認scope変更

## 10. Commands and environment

```powershell
# exact commands
```

```text
Python:
Packages/lock hash:
OS:
Random seed:
```

## 11. Results — 実行後に追記

### Contract audit

| Contract | Result | Evidence |
|---|---|---|
| Chronology |  |  |
| As-of/lineage |  |  |
| Top3 mass |  |  |
| Odds-free candidate |  |  |
| BUY immutability |  |  |
| Reproducibility |  |  |

### Metrics

| Fold/segment | Baseline | Challenger | Delta | Uncertainty |
|---|---:|---:|---:|---:|
|  |  |  |  |  |

### Robustness

- 最大配当除外:
- 上位3配当除外:
- 最大drawdown:
- 最悪fold:
- threshold sensitivity:
- race/day cluster bootstrap:

### Unexpected findings

```text

```

## 12. Decision

選択肢: `REJECTED` / `INVALID` / `REVIEW_REQUIRED` / `APPROVED_FOR_SHADOW`

```text

```

### Reason

```text

```

### Follow-up

- 新しい仮説は新しいexperiment IDとして登録する。
- 現行BUYへの接続は、この実験の承認に含めない。

## 13. Artifact manifest

| Artifact | SHA-256 | Rows/races | Created at | Notes |
|---|---|---:|---|---|
|  |  |  |  |  |

Result commit:

```text

```
