# Research OS v1 — Agent Operating Contract

作業開始前に必ず次を読む。

1. `AGENTS.md`
2. `research/CHARTER.md`
3. `research/STATE.yaml`
4. `research/DECISIONS.md`
5. `research/HYPOTHESIS_SCORECARD.yaml`
6. `research/REGISTRY.jsonl`
7. `research/EXPERIMENT_TEMPLATE.md`
8. `research/experiments/`
9. open中のIssueとPull Request

## 権限境界

Level 3は、GitHub上の人間承認証拠に結び付いたresearch scopeを、
research branch上で準備・実行できる制度である。本番変更、正式BUY再開、
注文、通知、mergeを自律実行する権限ではない。

- `main`へ直接commitしない。research/chore branchとDraft PRを使う。
- 自動mergeしない。
- 予測モデル、候補選択、value判定、正式BUYロジック、production pathを
  Research OS基盤PRで変更しない。
- secret、credential、購入API、通知先を読み書きしない。
- research artifactは常に`formal_buy=false`、`send_order=false`、`stake=0`。
- 既存dirty worktreeを暗黙にstageしない。commit対象を明示する。
- 承認に任意の`--actor`や`--human-approved`を使用しない。

## 二段階実行承認

```text
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

`BLOCKED_SCORE`は75点未満の実行不可状態。`INVALID`はterminalである。

### APPROVED_TO_PREPARE

- GitHub Issue comment
  `APPROVED_TO_PREPARE <proposal_scope_digest>`が必要。
- 実装準備とsynthetic fixture unit testだけを許可する。
- 実データ学習、backtest、outer OOS、ROI計算を禁止する。

### APPROVED_TO_RUN

- 実装後のcanonical run scopeに対するGitHub Issue comment
  `APPROVED_TO_RUN <run_scope_digest>`が必要。
- `RUNNING`直前に同じcommentをGitHubから再取得し、author、body、
  digest、updated_atを再検証する。
- GitHub確認不能、証拠欠落、unauthorized author、comment編集、
  scope/hash/command変更はfail-closeする。

### APPROVED_FOR_SHADOW

- `APPROVED_TO_RUN`とは別の人間承認である。
- `APPROVED_FOR_SHADOW <review_digest>`形式の別commentを必要とする。
- production、merge、正式BUY再開を承認できない。

## GitHub-backed approval evidence

- approver allowlistの正本はtrackedな`research/APPROVERS.json`。
- experiment branch上のallowlistを信用しない。
- proposalに記録したbase commitが`origin/main` history上にあることを検証し、
  そのcommit上のallowlistを`git show`で読む。
- author loginがallowlistにあり、GitHub actor typeが`User`であり、
  Codex/bot/automationでないことを検証する。
- registryへcomment ID、URL、author、created_at、updated_at、
  body SHA-256、approval type、digestを保存する。
- GitHub取得はread-only GETのみ。CIはfixture/injected providerを使い、
  外部通信しない。

## Canonical scope

Markdownは承認scopeの正本ではない。正本は
`research/scopes/<experiment_id>.proposal.json`と
`research/scopes/<experiment_id>.run.json`である。

SerializationはUTF-8、key sort、compact separator、Unicode保持、
NaN/Infinity禁止。setとして扱うproposal listは重複禁止・sortし、
execution command listは順序を保持する。

proposal scopeには仮説、null、作用機序、対象母集団、in/out scope、
expected paths、raw data、as-of、allowed/forbidden columns、lineage、
chronological fold、fold path/hash、purge/embargo、primary metric、
required effect、rejection gate、stop、budget、variant/threshold上限、
base commit、score、安全flagを含める。

run scopeはproposal scope全体とdigestに加え、exact execution commit、
config/data/fold/runner/environment hash、seed、exact commandsを固定する。
hash-bound lifecycle/manifest以外のuncommitted・untracked pathを拒否し、
worktree codeをcommit SHAの代用にしない。
実行後の結果やcandidate/price artifactはrun scopeへ混ぜずappendする。

## 変更不能の研究契約

- Top3は非順序3頭集合softmax。各raceのmassは`1 ± 1e-10`。
- 導出wide pairのmassは各race`3 ± 1e-10`。
- 取消後runner universeの全`C(n,3)`、finite、range、重複を検査する。
- 候補生成・除外・順位・tier・tie-break・coverage・abstentionに
  odds、人気、market、払戻、ROI、その派生値を使わない。
- `train < validation < calibration < outer test`を維持し、
  race overlap、未来・事後情報、test由来調整を禁止する。
- 契約違反runは`INVALID`とし、ROIを計算・解釈しない。

## 完了条件

- Python 3.11/3.12のCIと関連testが成功している。
- 変更file、exact command、artifact、未確認事項をDraft PRへ記録する。
- production、BUY、注文、通知、credential関連差分がない。
- mergeとDraft解除を人間へ残す。
