# {{EXPERIMENT_ID}} — {{TITLE}}

> 事前登録欄は結果閲覧後に変更しない。変更が必要なら新しいexperiment IDを発行する。

## 0. Metadata and authority

| Field | Value |
|---|---|
| Experiment ID | `{{EXPERIMENT_ID}}` |
| Status | `{{STATUS}}` |
| Owner | {{OWNER}} |
| Created at | {{CREATED_AT}} |
| Base commit |  |
| Branch |  |
| Data as-of |  |
| Human approver | pending |
| Approval time | pending |

- [ ] Registryへ提案eventをappendした。
- [ ] scoreが75点以上である。
- [ ] 人間が `APPROVED_TO_RUN` を記録した。
- [ ] 本番反映、merge、正式BUY再開は承認scope外である。

## 1. Hypothesis scorecard

| Criterion | Score | Max | Evidence |
|---|---:|---:|---|
| 既存特徴では代替できない独立情報 | {{SCORE_INDEPENDENT_INFORMATION}} | 25 |  |
| 競馬上の作用機序 | {{SCORE_RACING_MECHANISM}} | 20 |  |
| outer OOSの失敗分析による根拠 | {{SCORE_OUTER_OOS_FAILURE_EVIDENCE}} | 20 |  |
| リーク安全性 | {{SCORE_LEAKAGE_SAFETY}} | 15 |  |
| 最小実験による反証可能性 | {{SCORE_MINIMAL_FALSIFIABILITY}} | 10 |  |
| 取得・実装コスト | {{SCORE_ACQUISITION_IMPLEMENTATION_COST}} | 10 |  |
| **Total** | **{{SCORE_TOTAL}}** | **100** |  |

Auto-execution eligibility: `{{AUTO_EXECUTION_ELIGIBLE}}`。75点以上でも人間承認までは実行しない。

## 2. Research question and mechanism

### Research question

{{HYPOTHESIS}}

### Racing mechanism

_なぜ、どの条件で、どちら向きに作用するか。既存特徴との違いを書く。_

### Null hypothesis

同じchronological outer OOS条件で、challengerは凍結baselineよりprimary metricを改善しない。

### Minimal falsification

_1つの最小変更、primary metric、棄却条件、停止条件を書く。_

## 3. Scope

### In scope

- _記入する_

### Always out of scope

- 予測モデルの本番差し替え
- 現行候補選択・value判定・正式BUYロジックの変更
- stake、注文、通知、secretの変更
- outer test閲覧後の同一IDでの再調整

### Expected files and artifacts

```text

```

## 4. Data and leakage contract

| Dataset | Role | Period | Event time | Available time | Hash | Allowed columns |
|---|---|---|---|---|---|---|
|  | train |  |  |  |  |  |
|  | validation |  |  |  |  |  |
|  | calibration |  |  |  |  |  |
|  | outer test |  |  |  |  |  |

- [ ] 事後・未来情報がない。
- [ ] target raceのodds、人気、market、payoff、ROIと派生列をcandidate matrixから除外した。
- [ ] feature生成、欠損処理、calibrationが各outer testより前だけでfitされる。
- [ ] source/received timeとcontent hashを確認した。

## 5. Fixed chronological outer OOS

Fold manifest path / SHA-256:

```text

```

| Fold | Train | Validation | Calibration | Outer test | Purge/embargo |
|---|---|---|---|---|---|
| 1 |  |  |  |  |  |

- [ ] `train_max < validation_min < calibration_min < test_min`。
- [ ] race overlap、unknown date、empty foldが0。
- [ ] baselineとchallengerが同じfold manifestを使う。
- [ ] outer resultをfeature、threshold、variant選択に使わない。

## 6. Top3 and wide probability contract

- Frozen runner universe / count column / SHA-256:

```text

```

- [ ] Top3は非順序集合softmaxである。
- [ ] probabilityはfiniteかつ `[0,1]`、全 `C(n,3)` 集合が一度ずつ存在する。
- [ ] 各raceのTop3集合確率合計が `1 ± 1e-10`。
- [ ] wide pair probabilityをTop3集合から周辺化した。
- [ ] 各raceの全wide pair確率合計が `3 ± 1e-10`。
- [ ] failure時はcandidate/ROI評価へ進まない。

## 7. Odds-free candidate freeze

Candidate manifest / SHA-256:

```text

```

- [ ] `candidate_uses_odds=false`。
- [ ] price join前にkey、rank、tier、coverageをfreezeした。
- [ ] odds perturbationでもcandidate digestが不変である。
- [ ] research outputは `formal_buy=false`、`send_order=false`、`stake=0`。

## 8. Preregistered metrics and gates

Primary metric:

| Metric | Direction | Baseline | Required effect | Aggregation |
|---|---|---:|---:|---|
|  |  |  |  | race/day cluster |

Promotion gate, rejection gate, stop conditions:

```text

```

必須報告: fold別結果、uncertainty、最大配当/上位3配当除外、profit concentration、max drawdown、threshold sensitivity、missingness。

## 9. Commands and reproducibility

```powershell
# exact commands
```

Record: code/config/data/fold/candidate hash、Python/dependency version、seed、stdout/stderr、artifact URI。

## 10. Results — execution後にappend

| Contract | Result | Evidence |
|---|---|---|
| Chronology |  |  |
| As-of / leakage |  |  |
| Top3 mass = 1 |  |  |
| Wide mass = 3 |  |  |
| Odds-free candidate |  |  |
| Formal BUY stopped |  |  |
| Reproducibility |  |  |

| Fold/segment | Baseline | Challenger | Delta | Uncertainty |
|---|---:|---:|---:|---:|
|  |  |  |  |  |

## 11. Decision — human review

Decision: `REVIEW_REQUIRED` / `REJECTED` / `INVALID` / `APPROVED_FOR_SHADOW`

Reason and unconfirmed items:

```text

```

本実験の承認は、本番反映、merge、正式BUY再開を含まない。
