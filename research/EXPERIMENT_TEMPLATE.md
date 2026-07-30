# {{EXPERIMENT_ID}} — {{TITLE}}

> このMarkdownはレビュー表示用であり、承認scopeの正本ではない。正本は下記canonical JSONである。

## 0. Canonical authority

| Field | Value |
|---|---|
| Experiment ID | `{{EXPERIMENT_ID}}` |
| Status | `{{STATUS}}` |
| Owner | {{OWNER}} |
| Created at | `{{CREATED_AT}}` |
| Base commit | `{{BASE_COMMIT}}` |
| Data as-of | `{{DATA_AS_OF}}` |
| Proposal scope JSON | `{{PROPOSAL_SCOPE_PATH}}` |
| Proposal scope digest | `{{PROPOSAL_SCOPE_DIGEST}}` |
| Run scope JSON | pending |
| Run scope digest | pending |

Canonical serialization:

- UTF-8
- object key sort
- compact separators `,` and `:`
- Unicodeをescapeせず保持
- NaN / Infinity禁止
- setとして扱うproposal配列は重複禁止・Unicode順sort
- `exact_execution_commands`など順序に意味がある配列は入力順を保持

## 1. Two-stage human approval

### Preparation approval

- Required Issue comment: `APPROVED_TO_PREPARE {{PROPOSAL_SCOPE_DIGEST}}`
- [ ] base commit上の`research/APPROVERS.json`でauthorを検証した。
- [ ] comment ID、URL、author、created/updated time、body SHA-256をregistryへ保存した。
- [ ] `PREPARING`ではresearch branch上の実装とsynthetic fixture unit testだけを行う。
- [ ] 実データ学習、backtest、outer OOS、ROI計算を行わない。

### Run approval

- Required Issue comment: `APPROVED_TO_RUN <run_scope_digest>`
- [ ] 実装後のcommit、config/data/fold/runner/environment hash、seed、commandsを固定した。
- [ ] `RUNNING`直前に同じcommentをGitHubから再取得した。
- [ ] author、body、digest、updated_atが承認時点と一致した。
- [ ] GitHub確認不能、証拠欠落、scope変更時はfail-closeした。

### Shadow approval

- Required Issue comment: `APPROVED_FOR_SHADOW <review_digest>`
- [ ] run承認とは別コメントで承認した。
- [ ] production、merge、正式BUY再開は承認scope外である。

## 2. Hypothesis scorecard

| Criterion | Score | Max |
|---|---:|---:|
| 既存特徴では代替できない独立情報 | {{SCORE_INDEPENDENT_INFORMATION}} | 25 |
| 競馬上の作用機序 | {{SCORE_RACING_MECHANISM}} | 20 |
| outer OOSの失敗分析による根拠 | {{SCORE_OUTER_OOS_FAILURE_EVIDENCE}} | 20 |
| リーク安全性 | {{SCORE_LEAKAGE_SAFETY}} | 15 |
| 最小実験による反証可能性 | {{SCORE_MINIMAL_FALSIFIABILITY}} | 10 |
| 取得・実装コスト | {{SCORE_ACQUISITION_IMPLEMENTATION_COST}} | 10 |
| **Total** | **{{SCORE_TOTAL}}** | **100** |

Auto-execution eligibility: `{{AUTO_EXECUTION_ELIGIBLE}}`。75点以上でもGitHub上の承認証拠がなければ準備・実行しない。

## 3. Research question

### Hypothesis

{{HYPOTHESIS}}

### Null hypothesis

{{NULL_HYPOTHESIS}}

### Racing mechanism

{{RACING_MECHANISM}}

詳細なtarget population、in/out scope、expected paths、raw sources、
allowed/forbidden columns、lineage、fold、purge/embargo、metric、gate、
stop、budget、variant/threshold上限はcanonical proposal JSONを参照する。

## 4. Immutable run scope

`research/scopes/{{EXPERIMENT_ID}}.run.json`はproposal scope全体に加えて、
次を含む。

- proposal scope digest
- exact execution commit SHA
- config hashes
- data/input manifest hashes
- fold manifest path/hash
- runner-universe manifest path/hash
- dependency/environment manifest path/hash
- seed
- exact execution commands
- `formal_buy=false`
- `send_order=false`
- `stake=0`

結果、candidate digest、price join後artifactなど実行後にのみ生成できる値は
run scopeへ追記せず、result artifactとしてregistryへappendする。

## 5. Absolute research contracts

- [ ] Top3は非順序集合softmaxである。
- [ ] 取消後runner universeの全 `C(n,3)` 集合を含む。
- [ ] 各raceのTop3集合確率合計は `1 ± 1e-10`。
- [ ] Top3集合から導出したwide pair確率合計は `3 ± 1e-10`。
- [ ] 候補選択にodds、人気、market、払戻、ROIを使用しない。
- [ ] `train < validation < calibration < outer test` を維持する。
- [ ] 事後情報、未来情報、リークを使用しない。
- [ ] research outputは `formal_buy=false`、`send_order=false`、`stake=0`。
- [ ] production pathをimport・変更・実行しない。

## 6. Local verification

```text
# synthetic fixture unit tests and exact commands
```

## 7. Results — execution後にappend

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

実行結果、candidate digest、price join後artifactはappend-only evidenceとし、
承認済みproposal/run scopeを上書きしない。

## 8. Decision — human review

Decision: `REVIEW_REQUIRED` / `REJECTED` / `INVALID` / `APPROVED_FOR_SHADOW`

```text
# reason, review_digest, unconfirmed items
```

本実験の承認は、production反映、merge、正式BUY再開、注文送信を含まない。
