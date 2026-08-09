## Purpose and lineage

- Experiment / governance ID:
- Gate profile: `roi_research_v1` / `infrastructure_safety_v1` / `governance_core`
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

- Repository: `kazuponbaseball-cell/keiba_ai_project`
- Base branch: `main`
- Verified current main SHA:
- Verified base commit:
- GitHub compare URL / status:
- Merge-base SHA:
- Base-commit `research/APPROVERS.json` blob SHA:
- Base-commit `research/APPROVERS.json` content SHA-256:
- GitHub verification time:
- APPROVED_TO_PREPARE comment ID / URL:
- APPROVED_TO_RUN comment ID / URL:
- APPROVED_FOR_SHADOW comment ID / URL:
- [ ] caller `--actor`と`--human-approved`を承認証拠にしていない。
- [ ] GitHub read-only APIから`refs/heads/main`を取得した。
- [ ] base commitからcurrent mainへのcompareは`ahead`または`identical`で、
  merge-base SHAがbase commitと一致した。
- [ ] `research/APPROVERS.json`をGitHub上のbase commitから取得し、ローカル
  `origin/main`、`git show`、worktree版を承認根拠にしていない。
- [ ] repository、base branch、base commit、compare response、contents refの
  不一致をfail-closeした。
- [ ] prepare/run/shadowのcomment IDは、registry全体で未使用かつ相互に異なる。
- [ ] `PREPARING`前にprepare commentを再検証した。
- [ ] `APPROVED_TO_RUN`前にprepare commentを再検証した。
- [ ] `RUNNING`前にprepare/run commentsを再検証した。
- [ ] `APPROVED_FOR_SHADOW`前にprepare/run commentsを再検証した。
- [ ] comment ID、Issue、URL、author/type、body、keyword/digest、body SHA-256、
  created_at、updated_atをGitHubからread-only検証してregistryへ保存した。

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

## Infrastructure safety contract — applicable when selected above

- Gate policy path / SHA-256:
- Gate bootstrap source: merged `main` / governance-core human review
- [ ] 数値ROI scoreを代用せず、全hard checkの論理ANDで判定した。
- [ ] `execution_kind=synthetic`をrun scopeへ固定し、CLI値と一致させた。
- [ ] execution commitがproposal baseの子孫であり、base-to-execution変更path/blobをexpected pathと一致させた。
- [ ] config/synthetic/environmentはexecution commit blobへ固定し、dirty/untracked materialは0。
- [ ] commandはcode-owned unittest templateから生成した`-B -I -S` argvで、repository-root cwd、継承environmentなし、timeoutを固定した。
- [ ] infra testは`research/infra_tests/`に置き、`tests/research/`を変更せず、PR作成だけで自動実行されない。
- [ ] network/API、credential、real data、training、backtest、outer OOS、ROI、actual Codex dispatchは0。
- [ ] model、feature、candidate、value、production、BUY、order、notification差分は0。
- [ ] gate policy、approval verifier、APPROVERS、憲章、scorecardなどroot-of-trustをinfra gateで自己変更していない。
- [ ] verified current mainのregistry blobとlocal ledgerが完全一致し、append直前のmain head再検証後に1 pending eventだけを作成した。
- [ ] pending infra eventのpreparation/execution authorityはfalseで、人間merge前のeventを実行根拠にしていない。
- [ ] infra lifecycleから`APPROVED_FOR_SHADOW`へ遷移していない。

## Research contracts

- [ ] Top3は非順序集合softmax。
- [ ] 取消後runner universeの全`C(n,3)`、Top3 mass=`1 ± 1e-10`。
- [ ] 導出wide mass=`3 ± 1e-10`。
- [ ] candidateはodds、人気、market、払戻、ROIから独立。
- [ ] `train < validation < calibration < outer test`。
- [ ] future/post-event/leakageなし。
- [ ] `formal_buy=false`、`send_order=false`、`stake=0`。

## Fail-close conditions

- [ ] GitHub current main、compare、base-commit contents取得不能。
- [ ] base commitがcurrent main ancestorでない、compare statusまたはmerge-base不一致。
- [ ] GitHub上のbase-commit `APPROVERS.json`が取得不能・不正JSON。
- [ ] allowlist外、Codex、bot、automation author。
- [ ] approval commentの編集・削除、immutable field変更、grant ID再利用。
- [ ] 必須transition前のprepare/run comment再検証失敗。
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

- [ ] `research/STATE.yaml`のreconciliation snapshotで観測したmain commit上の
  `research/APPROVERS.json`存在確認は、動的なcurrent mainの表明ではない。
- [ ] 実運用GitHub providerを用いた、実Issue commentによるapproval transition
  E2Eは未確認。
- [ ] モデル監査基準commit `288dff5e86385908281428d5ed4f077625a43e4b`
  には同fileがなく、そのcommitをproposal baseとする承認はfail-closeする。

## Human-only gates

- [ ] Draftのまま維持する。
- [ ] 新しいPRを作成していない。
- [ ] このPRはformal BUYを再開せず、注文を送らない。
- [ ] Production promotionは別の承認済みPRを必要とする。
- [ ] Mergeは明示的な人間承認を必要とし、auto-mergeしない。
