## Purpose

- Experiment / governance ID:
- Human approver:
- Approval record:

## Scope

### Changed

-

### Explicitly unchanged

- [ ] 予測モデル
- [ ] 候補選択
- [ ] value判定
- [ ] 正式BUYロジック、stake、注文、通知

## Research contracts

- [ ] Top3は非順序集合softmaxである。
- [ ] 取消後runner universeの全 `C(n,3)` 集合を含み、確率はfiniteかつ `[0,1]`。
- [ ] 各raceのTop3集合確率合計は `1 ± 1e-10`。
- [ ] Top3集合から導出したwide pair確率合計は `3 ± 1e-10`。
- [ ] 候補選択にodds、人気、market、払戻、ROIを使用しない。
- [ ] `train < validation < calibration < outer test` を維持する。
- [ ] 事後情報、未来情報、リークを使用しない。
- [ ] research outputは `formal_buy=false`、`send_order=false`、`stake=0`。

## Score and approval

- Hypothesis score: `/100`
- [ ] 75点以上である。75点未満の場合は実行していない。
- [ ] 人間が `APPROVED_TO_RUN` を記録してから実行した。
- [ ] 承認後にscope、fold、primary metric、gateを変更していない。

## OOS and leakage evidence

- Fold manifest / hash:
- Data as-of / hash:
- Frozen runner universe / hash:
- Candidate manifest / hash:
- Contract audit artifact:

## Verification

```text
# exact commands and results
```

## Results and robustness

- Primary metric:
- Fold-level result:
- Uncertainty:
- Payout concentration / max drawdown:
- Negative result or rejection evidence:

## Unconfirmed items

-

## Human-only gates

- [ ] This PR does not resume formal BUY or send orders.
- [ ] Production promotion requires a separate approved PR.
- [ ] Merge requires explicit human approval; no auto-merge.
