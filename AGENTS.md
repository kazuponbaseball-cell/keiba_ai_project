# Research OS v1 — Agent Operating Contract

このリポジトリで研究を行うagentは、作業開始前に `research/CHARTER.md`、`research/STATE.yaml`、`research/HYPOTHESIS_SCORECARD.yaml`、`research/DECISIONS.md` を読む。

## 権限境界

Level 3は「人間が承認した研究計画をresearch branch上で自律実行できる」ことを意味する。本番変更、正式BUY再開、注文、mergeを自律実行する権限ではない。

- `main`へ直接コミットしない。research/chore branchとドラフトPRを使う。
- 本番反映、正式BUY再開、PR mergeはrepository ownerの明示承認を必須とする。
- 予測モデル、候補選択、value判定、正式BUYロジックはResearch OS基盤PRでは変更しない。
- 正式BUYは停止状態を維持する。研究artifactは常に `formal_buy=false`、`send_order=false`、`stake=0` とする。
- secret、live credential、購入API、通知先を読み書きしない。
- 既存のdirty worktreeを暗黙にstageしない。commit対象を明示pathで指定する。

## 実行ゲート

1. 仮説を `research/HYPOTHESIS_SCORECARD.yaml` の6項目で採点する。
2. 合計75点未満は `BLOCKED_SCORE` とし、自動実行しない。
3. 75点以上でも人間が `APPROVED_TO_RUN` を記録するまで実行しない。
4. 承認後に仮説、特徴、データ、fold、primary metric、停止条件を変えた場合は新しいexperiment IDと再承認を必要とする。
5. 結果は成功・失敗を問わずappend-only registryとdecision logへ残す。

## 変更不能の研究契約

- Top3は非順序3頭集合に対するsoftmaxをcanonical distributionとする。
- レースごとのTop3集合確率合計は `1 ± 1e-10`。
- Top3集合から周辺化した全wide pair確率合計は `3 ± 1e-10`。
- 確率はfiniteかつ `0 <= p <= 1`。取消後runner universeの全 `C(n,3)` 集合を持ち、集合重複、runner重複、universe不完全、契約違反はfail-closeする。
- 観測された集合だけでは完全に欠落したrunnerを検出できない。promotion候補runは、freeze済みrunner universeまたは `--runner-count-column` を外部契約として与える。
- 候補の生成、除外、順位、tier、tie-break、abstentionにodds、人気、market probability、払戻、ROI、またはその派生値を使わない。
- データ分割は `train < validation < calibration < outer test` のchronological outer OOSとし、race重複、未来情報、事後情報、test由来の調整を禁止する。
- 契約違反runではROIを計算・解釈せず、statusを `INVALID` にする。

## 標準ワークフロー

1. `scripts/research/create_experiment.py` で提案を作る。
2. score、scope、fold、反証条件を人間がレビューする。
3. `scripts/research/update_registry.py` で承認イベントをappendする。
4. 承認済みscopeだけを実装・実行する。
5. 確率、chronology、lineage、candidate freeze、BUY停止をmetricsより先に検査する。
6. `scripts/research/summarize_experiment.py` でレビュー用summaryを作る。
7. shadow昇格も人間承認を必要とし、本番反映は別PRとする。

## 完了条件

- 関連testとcontract checkが成功している。
- 変更ファイル、実行command、artifact、未確認事項をPRへ記録している。
- production control pathの差分がない。
- merge前に人間レビューが残っている。
