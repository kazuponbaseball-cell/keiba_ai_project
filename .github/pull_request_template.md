## Purpose and lineage

- Experiment / governance ID:
- Issue:
- This PR succeeds PR #:
- Draft retained: yes / no

## Scope

### Changed

-

### Explicitly unchanged

- [ ] 予測モデル
- [ ] 候補選択
- [ ] value判定
- [ ] 正式BUYロジック、stake、注文、通知
- [ ] credential / secret
- [ ] production path

## Approval evidence model

- Base commit:
- Base-commit approver allowlist:
- APPROVED_TO_PREPARE comment ID / URL:
- APPROVED_TO_RUN comment ID / URL:
- APPROVED_FOR_SHADOW comment ID / URL:
- [ ] caller `--actor`と`--human-approved`を承認証拠にしていない。
- [ ] comment author、body、digest、updated_atをGitHubからread-only検証した。
- [ ] comment ID、URL、author、created_at、updated_at、body SHA-256をregistryへ保存した。

## Canonical digests

- Proposal scope JSON:
- `proposal_scope_digest`:
- Run scope JSON:
- `run_scope_digest`:
- Review/result manifest:
- `review_digest`:

- [ ] Markdownを承認scopeの正本にしていない。
- [ ] exact commit、config/data/fold/runner/environment hash、seed、commandsをrun scopeで固定した。
- [ ] 実行後にだけ生成されるresult/candidate/price artifactをrun scopeから分離した。

## Research contracts

- [ ] Top3は非順序集合softmax。
- [ ] 取消後runner universeの全`C(n,3)`、Top3 mass=`1 ± 1e-10`。
- [ ] 導出wide mass=`3 ± 1e-10`。
- [ ] candidateはodds、人気、market、払戻、ROIから独立。
- [ ] `train < validation < calibration < outer test`。
- [ ] future/post-event/leakageなし。
- [ ] `formal_buy=false`、`send_order=false`、`stake=0`。

## Fail-close conditions

- [ ] GitHub取得不能またはapproval evidence欠落。
- [ ] allowlist外、Codex、bot、automation author。
- [ ] approval commentのauthor、body、digest、updated_at変更。
- [ ] proposal/run scope digest変更。
- [ ] execution commit、config/data/fold/runner/environment hash変更。
- [ ] exact execution commandまたは安全flag変更。
- [ ] preparation中のreal-data execution。

## Local verification

```text
# exact commands and results
```

## GitHub Actions

- Workflow run:
- Python 3.11:
- Python 3.12:
- Research tests:

## Results and artifacts

- Result artifacts:
- Contract audit:
- Negative/rejection evidence:

## Unconfirmed items

-

## Human-only gates

- [ ] Draftのまま維持する。
- [ ] 新しいPRを作成していない。
- [ ] このPRはformal BUYを再開せず、注文を送らない。
- [ ] Production promotionは別の承認済みPRを必要とする。
- [ ] Mergeは明示的な人間承認を必要とし、auto-mergeしない。
