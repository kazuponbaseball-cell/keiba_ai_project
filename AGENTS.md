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

## インフラ／安全性gate

競馬仮説でないResearch OS control-plane、contract、schema、adapterの変更へ、
競馬上の作用機序やouter OOS根拠を仮装して`HYPOTHESIS_SCORECARD`を適用しては
ならない。明示的な`infrastructure_safety_v1` contractだけを別gateとして扱う。

- ROI仮説の75点gate、scope、digest、既存queue/event schema v2を変更しない。
- 数値scoreの相互補償を行わない。machine-readableな全hard checkを満たさない
  scopeはartifactやregistry eventを作る前にfail-closeする。
- GitHubのprepare/run二段階承認、base ancestry、base-commit allowlist、
  registry-wide comment ID単一用途、承認再検証を同じ強度で要求する。
- 実行種別はhash-boundな`synthetic`だけとし、real data、学習、backtest、
  outer OOS、ROI計算、外部network/API、credential、実Codex dispatchを禁止する。
- model、feature、candidate、value、production、BUY、order、notificationのpathや
  capabilityを含むscopeはインフラ扱いできず、即時拒否する。
- infra lifecycleから`APPROVED_FOR_SHADOW`へ遷移できない。merge、production、
  正式BUY再開の承認状態も追加しない。
- gate policy、approval verifier、approver allowlist、憲章、scorecardなど
  root-of-trustの自己変更をinfra gate経由で行わない。
- ledgerはsymlink/junctionでないexact `research/REGISTRY.jsonl`だけを使う。base→execution commitと
  execution commit→worktreeの既存内容を改変・削除せず、process lockとCASの下で
  1 eventずつ追記する。全transition前にGitHub current mainのledgerを取得し、local ledgerが
  そのbytesと完全一致しない場合は停止する。append直前にmain headを再取得して不変を要求する。
  追記eventは人間merge前のpending evidenceであり権限を持たない。次transitionにはそのeventの
  main mergeとbranch refreshが必要である。alternate registryは承認namespaceにならない。
- run materialはASCII path、`.example.json` config、bounded synthetic envelopeだけに
  制限し、execution commit上のblobへ固定する。dirty/untracked material、symlink/junction、
  credential-like value、row-level real-data shapeを拒否する。
- execution commitはproposal baseの子孫でなければならない。変更Pythonはexact Git blobを
  AST検査し、pure import allowlist外、network、credential、subprocess、production module、
  dynamic call、forbidden capability symbolを拒否する。commandはhash-boundなcurrent
  interpreterを`-B -I -S`で使い、repository-root cwd、空の継承environment、timeout、
  write path 0をrun scopeへ固定する。infra eventの`automatic_execution_allowed`、
  `preparation_authorized`、`execution_authorized`は常にfalseとする。
- 後続infra gateのtest sourceは`research/infra_tests/`だけに置く。通常PR CIが自動探索する
  `tests/research/`はroot-of-trustであり、infra diffから変更できない。専用testをPR作成時に
  自動実行しない。v3はevidence compilerでありexecutorではないため、structured command実行には
  別の人間review済みexecutor/authority verifierを必要とする。

初回gate導入と将来のroot-of-trust変更は自己承認できないgovernance-core変更で
あり、Draft PRの人間reviewと人間mergeを必要とする。mainへmergeされたpolicyだけを
後続infra proposalの信頼根拠にする。
`infrastructure_safety_v1`のpolicy fileはmain merge後にin-place変更しない。policy変更は
新しいgate kind/versionと別pathで導入し、既存v3 scopeを引き続き読めるようにする。

### APPROVED_TO_PREPARE

- GitHub Issue comment
  `APPROVED_TO_PREPARE <proposal_scope_digest>`が必要。
- 新規grantのcomment IDは、同じ`REGISTRY.jsonl`内の過去のprepare/run/shadow
  grantで未使用でなければならない。
- 実装準備とsynthetic fixture unit testだけを許可する。
- 実データ学習、backtest、outer OOS、ROI計算を禁止する。

### APPROVED_TO_RUN

- 実装後のcanonical run scopeに対するGitHub Issue comment
  `APPROVED_TO_RUN <run_scope_digest>`が必要。
- run承認にはprepare承認と異なる未使用comment IDが必要である。
- `APPROVED_TO_RUN`直前にprepare承認commentを再検証する。
- `RUNNING`直前にprepare承認commentとrun承認commentを再検証する。
- GitHub確認不能、証拠欠落、unauthorized author、comment編集・削除・再利用、
  scope/hash/command変更はfail-closeする。
- legacy ROI run scope v1は`execution_kind`をcanonical digestへ含めないため、
  real-data `RUNNING`をfail-closeする。real-data再開にはkindとcapabilityをdigestへ
  固定するversioned ROI run contractの別governance変更が必要である。

### APPROVED_FOR_SHADOW

- `APPROVED_TO_RUN`とは別の人間承認である。
- `APPROVED_FOR_SHADOW <review_digest>`形式で、prepare/run承認のいずれとも
  異なる未使用comment IDを必要とする。
- `APPROVED_FOR_SHADOW`直前にprepare承認commentとrun承認commentを
  GitHubから再取得して再検証する。
- production、merge、正式BUY再開を承認できない。

## GitHub-backed approval evidence

- 承認対象repositoryは`kazuponbaseball-cell/keiba_ai_project`、base branchは
  `main`に実装定数として固定する。proposalはbase commitを固定する。
- GitHubのread-only APIから`refs/heads/main`のcurrent head SHAを取得する。
- proposalのbase commitからcurrent main SHAへのGitHub compareが`ahead`
  または`identical`で、merge-base SHAがproposalのbase commitと一致する場合だけ
  main ancestorとして認める。
- approver allowlistの正本は、GitHub上のproposal base commitにある
  `research/APPROVERS.json`である。ローカルworktree、ローカルobject、
  `refs/remotes/origin/main`、experiment branch上のcopyを承認根拠にしない。
- author loginがallowlistにあり、GitHub actor typeが`User`であり、
  Codex/bot/automationでないことを検証する。
- registryへrepository、base branch、verified current main SHA、verified base
  commit、compare URL/status、merge-base SHA、APPROVERS blob SHA、APPROVERS
  content SHA-256、verification timeを保存する。
- 全transition時にverified current mainの`research/REGISTRY.jsonl`を取得し、path/ref、
  blob SHA、content SHA-256を保存する。local snapshotがremote bytesと完全一致しない場合は
  stale/pending ledgerとしてfail-closeする。append直前のmain ref再検証も必須とする。
- registryへcomment ID、Issue番号、URL、author login/type、body、approval
  keyword/digest、body SHA-256、created_at、updated_atを保存する。
- prepare/run/shadowの新規grantはそれぞれ異なるcomment IDを使う。
  後続transitionで同じ証拠を再検証することは新規grantまたは再利用ではない。
- comment ID、Issue、URL、author、body、keyword、digest、body SHA-256、
  created_at、updated_atの変更、comment削除、grant ID再利用はfail-closeする。
- GitHub取得はread-only GETのみ。CIはfixture/injected providerを使い、
  外部通信しない。
- `research/STATE.yaml`のreconciliation snapshotで観測した`main` commit
  `1eaf364571bd8b9fd27f7de657ce295b563b3f1f`には
  `research/APPROVERS.json`が存在し、GitHub Contents APIからblob SHAと
  content SHA-256を取得済みである。これは固定された観測証拠であり、動的な
  current mainを表さない。ただし、実運用GitHub providerを用いた、実Issue commentによる
  approval transition E2Eは未確認である。
- モデル監査基準commit `288dff5e86385908281428d5ed4f077625a43e4b`
  には`research/APPROVERS.json`がないため、そのcommitをproposal baseとする
  実承認は引き続きfail-closeする。

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
