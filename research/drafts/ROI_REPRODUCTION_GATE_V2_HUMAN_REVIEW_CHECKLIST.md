# ROI Reproduction Gate v2 — human review checklist

作成日: 2026-08-09

対象design:
`docs/ROI_REPRODUCTION_GATE_V2_DESIGN_DRAFT.md`

machine-readable map:
`research/drafts/ROI_REPRODUCTION_GATE_V2_CONTRACT_MAP.design.json`

状態: `G1_IMPLEMENTED_AND_HUMAN_MERGED / CONTRACT_COMPILER_ONLY_NO_AUTHORITY /
G2_NOT_IMPLEMENTED / EXECUTION_FORBIDDEN`

> このchecklistへの記入はGitHub approval、proposal scope、run scope、execution lease、
> merge approvalではない。PR #36は`fbbebc804c7a2393aff26a6de9ad7c55caa5bc92`として
> mainへmerge済みである。PR #38のG1 implementationも人間ユーザーにより
> `811ffd11bd80447f013c643b96c3eb8145916061`としてmainへmerge済みだが、
> compiler-only/no-authorityである。G2は別human-owned Draftでreview・mergeする。
> G2 activation後の通常experiment transitionには
> Ready、merge、rebase、branch refreshを要求しない。

現行G1はpure compiler、policy、schema、synthetic-only governance testを実装するが、provider、
writer、authority verifier、executorを実装しない。legacy v2 / infra v3 writer semanticsは不変、
schema v4のlive dispatchと通常3-action flowは未activation、G2は未実装である。

- [ ] JSON Schema単独やtrusted policy/schema digestなしのnormalizer結果をauthorityに使わず、
  `NON_AUTHORITY_FIXTURE_VALIDATION`として破棄する。
- [ ] G2 activation前にcatalog source/provider allowlist、全catalog-contract↔run-manifest conformance、
  cwd/env/timeout/read-only input/single output root、expected output/numeric reference、current-mainの
  policy/schema hashを実効検証する。1件でも未実装なら`EXECUTION_FORBIDDEN`を維持する。
- [ ] `catalog publication scope v1`をG2 authorityへ流用しない。expected object count、canonical
  output、finite provider/command/budgetを別version/pathへfreezeし、人間review・main mergeする。

## A. Review metadata

- Reviewer:
- Review date/time:
- Reviewed commit:
- GitHub main observed:
- PR #36 decision/status:
- Overall decision: `ACCEPT_DESIGN / REQUEST_CHANGES / REJECT_DESIGN`
- Notes:

## B. Scope and score boundary — P0

- [ ] `roi_reproduction_audit_v2`を通常strategy gateのoverrideではなく、
  `nonpromotion_reproduction_audit`という別classとして認める。
- [ ] M0のstrategy score 23は記録するが、threshold overrideは不可、score creditは0とする。
- [ ] challenger、新model/feature/threshold、prospective outer、price、payoff、ROI、
  candidate/ticket/value、shadow、production、BUYを1つでも含むscopeは拒否し、
  通常の75点strategy gateへ送る。
- [ ] `REPRODUCED`でもstrategy score、Stage B approval、shadow、model採用、BUYへ
  自動継承しない。
- [ ] Stage Bは別experiment ID、別scope、別score、全て新しいapprovalを必要とする。

Reviewer decision:

## C. Bootstrap and self-approval — P0

- [x] G1はPR #38として人間ユーザーによりmainへmerge済みで、schema/policy/compiler/
  synthetic fixtureだけを含み、全authorityをfalseに維持している。
- [ ] G2はG1と別のhuman-owned Draftにし、attester、authority verifier、
  reusable reference catalog、supervised executor、durable runtime/one-shot lease ledgerをreviewする。
- [ ] G1/G2/Aは自身や`infrastructure_safety_v1`で自己承認しない。
- [ ] PR #36の過去branchやPR #37 branchをroot of trustまたはapproval baseにしない。
- [ ] implementationはmerged main `fbbebc8...`の子孫から開始する。
- [ ] governance DraftのCIが変更されたexecutor、model、real-data testを自動実行しない。
- [ ] G2がhuman merge・durability検証されるまでは`EXECUTION_FORBIDDEN`を維持する。

Reviewer decision:

## D. State machine and terminal meaning — P0

- [ ] proposalはprepublished catalog release/entryをexact bindしてから作る。
- [ ] prepare → internal `CATALOG_BOUND`再検証 → run approval/run → result reviewの順序を固定する。
- [ ] catalog release/entry digestが不一致または未解決のrun scopeを拒否する。
- [ ] terminalを`REPRODUCED`、`RECONSTRUCTED_NOT_REPRODUCED`、
  `REPRODUCTION_FAILED`、`REJECTED`、`INVALID`に分ける。
- [ ] scientific negativeは`REPRODUCTION_FAILED`、human evidence rejectionは
  `REJECTED`、contract violationは`INVALID`とする。
- [ ] 全nonterminalから`INVALID`を許し、terminal resurrectionを拒否する。
- [ ] shadow/production transitionをこのgateへ定義しない。
- [ ] retryは新IDまたは新run scopeと、新しいgrantを必要とする。

Reviewer decision:

## E. Three routine GitHub grants and operator budget — P0

- [ ] 通常success pathの人間操作を次の3 keyword/digest bindingだけにする。
  - `APPROVED_TO_PREPARE <proposal_scope_digest>`
  - `APPROVED_TO_RUN <run_scope_digest>`
  - `ACKNOWLEDGED_REPRODUCTION_RESULT <review_digest>`
- [ ] 通常experimentのReady/merge/rebase/branch refreshを0回とする。
- [ ] 承認以外を含む人間のvisible action合計を3回以下にする。
- [ ] `APPROVED_TO_PREPARE`をworkflowの開始/再開操作とし、別のexperiment開始操作を要求しない。
- [ ] `APPROVED_TO_RUN`をsupervised run開始操作とし、別の「実行開始」確認を要求しない。
- [ ] `clean_a`/`clean_b`のreplicaごとの人間確認を要求しない。
- [ ] 3 comment IDは相互に異なり、schema/gate kindを跨いでglobal grant index全体で未使用とする。
- [ ] `APPROVERS.json`はproposal base commit上のGitHub blobを正本とする。
- [ ] authorはallowlist済み`User`のみとし、bot/Codex/automationを拒否する。
- [ ] comment編集、削除、ID再利用、body/digest/timestamp不一致をfail-closeする。
- [ ] 各phase直前に必要な全grant、current main、ancestryをGitHubから再検証する。
- [ ] catalog更新、scope drift、retryを追加clickで救済せず、別maintenance/new scopeへfail-closeする。
- [ ] catalog publisher/attesterとrun approverのrole separationを必須にするか決定する。

Role-separation decision: `REQUIRED / RECOMMENDED / NOT_REQUIRED`

## F. Durable runtime ledger and authority boundary — P0

- [ ] G2 activation時にmixed v2/v3のfull event chain、全comment ID、各experimentのlatest
  sequence/head/statusをterminal・nonterminalともbyte/hash-bound importする。
- [ ] 既存nonterminal headの欠落、fork、再初期化を拒否する。
- [ ] cutoverはactivation epoch、exact main/registry blob/content digest、old-writer freeze、
  import後のsecond compare、atomic active-backend switchを要求する。
- [ ] snapshotからactivationまでにevent/grantが変化したらabort/reimportする。
- [ ] durable ledgerはglobal sequenceとexperiment/catalog-scope/release-statusごとのsubject sequence、
  previous subject hash、terminalityを検証する。
- [ ] expected head CAS、event append、comment-ID reservation、lease issue/consumeを原子的に行う。
- [ ] activation後の全new v2/v3/v4 grantを同じglobal indexへrouteし、dual writerを拒否する。
- [ ] `research/REGISTRY.jsonl`はperiodic audit checkpointで、次transitionのmerge前提にしない。
- [ ] local JSON、worktree、branch file、alternate registryをauthority fallbackにしない。
- [ ] backend outage、split brain、stale head、concurrent writer、restore rollbackをfail-closeする。
- [ ] event、queue、scope、commentをexecution tokenとして扱わない。
- [ ] registry event safetyは常にauthority/automatic execution/production/merge/BUY=false、
  `formal_buy=false`、`send_order=false`、`stake=0`とする。

Reviewer decision:

## G. One-shot lease and supervised executor — P0

- [ ] verifierはcurrent main root-of-trust、durable ledger head、全grant、scope、capability、command、
  execution commitをreplica直前に再検証する。
- [ ] leaseへphase、replica、proposal/catalog/run digest、expiry、human supervisor、retry budgetを固定し、
  consumptionは別のimmutable receiptで追跡する。
- [ ] preparation leaseへ未生成のrun/replica fieldをnullで先取りしない。
- [ ] M0は`clean_a`と`clean_b`の2 replicaに別leaseを発行する。
- [ ] durable storeで同じlease/run/replicaの再利用を拒否する。
- [ ] required lifecycle transitionまたはstate不変operation append + lease issueを1 transaction、
  lease consume + dispatch reservationを別の1 transactionとしてそれぞれ原子的にし、部分成功と
  二重dispatchを拒否する。state不変issue receiptは既存authorizing lifecycle-event digestをbindする。
- [ ] immutable issue receiptとconsume receiptを分け、`consumed_at`で発行済みleaseを更新しない。
- [ ] lease operation sequence/headをlifecycle event sequence/headと分離し、replica/provider operationが
  `RUNNING`または`CATALOG_PUBLISHING`のself-transitionを作らない。
- [ ] receiptと別のimmutable lease-operation recordを正本にし、subject/kind/sequence/previous digest、
  authorizing event、lease、kind別scope/grant/replica/dispatch、policy/schema、安全定数をexact bindする。
- [ ] operation recordはself digestを含めず、canonical recordからnew digestを計算してreceiptへattestし、
  expected previous operation headとglobal headを同一transactionでCASする。
- [ ] preparation consume recordはproposal digest + dispatch reservationをbindしrun/replica fieldを拒否し、
  historical run consume recordはrun digest + replica IDを必須にする。
- [ ] consumption receiptもphase-conditioned exact shapeとし、preparationはproposal digestのみ、
  historical run replicaはrun digest + replica IDのみを許す。
- [ ] receiptはbackend/writer identity、transaction/idempotency key、previous/new head、
  global/operation sequence、event/grant/lease/run/replica digest、policy/schema digest、timestamp、signer evidence、
  安全定数を固定する。
- [ ] durable consumptionを証明できない間はno-authorityのままにする。
- [ ] crash後のautomatic retryとshared mutable cacheを拒否する。
- [ ] G2初版はhuman-invokedに限定するか、Codex dispatchを別capability・別reviewへ
  送るかを明示する。現在のdesign defaultは`actual_codex_dispatch=false`。
- [ ] exact cwd、sanitized env、interpreter、argv、timeout、read-only inputs、
  isolated writable outputをOS/runtimeで強制する。
- [ ] network、external API、credential、production、BUY、order、notificationを隔離する。
- [ ] GitHub read-only control planeはverifierだけに限定し、provider/model workloadへ
  network capabilityを渡さない。

Reviewer decision:

Executor invocation decision: `HUMAN_ONLY / FUTURE_SEPARATE_CODEX_CAPABILITY_REVIEW`

## H. Reusable reference catalog — P0

- [ ] per-experiment manifest grant/provider readを廃止し、prepublished catalogだけを使う。
- [ ] catalog release publicationをroutine experiment外のseparate maintenance reviewにする。
- [ ] data read前にcatalog publication scope digestを作り、
  `APPROVED_TO_PUBLISH_REFERENCE_CATALOG <catalog_publication_scope_digest>`を検証する。
- [ ] release digestをread前grantへ循環bindせず、releaseはscope/approval/provider receiptをbindする。
- [ ] catalog provider leaseはpublication scope ID/digestをsubjectにし、`experiment_id`、run scope、
  replica IDを禁止してprovider/code/env/command/capability/ledger head/expiryを固定する。
- [ ] catalog provider issue receiptもexperiment execution receiptと別shapeにする。
- [ ] catalog provider consume receiptはrun/replica receiptを流用せず、scope、provider lease/
  identity、dispatch reservation、previous/new ledger headをexact bindし、future output digestを
  先取りしない。
- [ ] catalog grantもbase-commit allowlist、User actor、global unused ID、edit/delete/reuse
  fail-closeを継承し、provider lease直前にGitHubから再取得する。
- [ ] published releaseをexperimentへbindするときもcatalog grant evidenceをproposal binding、
  run-scope freeze、`RUNNING`直前、result sealing、terminal classificationで再取得し、
  mutation/revocation時はproposal前だけ
  `BLOCKED_CATALOG`、`PROPOSED`以後はterminal `INVALID`で停止する。
- [ ] provider retry budgetを0とし、scope digest + grant IDごとにlease/publicationを1回だけ
  reserveする。crash/failure retryは新scope/new grantを要求する。
- [ ] catalog maintenance terminalを`CATALOG_PUBLISHED`、`CATALOG_PUBLICATION_FAILED`、
  `INVALID`に固定し、terminal resurrectionを拒否する。
- [ ] catalog maintenance専用event schemaでscope、global/subject sequence、previous event/status、
  grant reservation、provider lease issue/consume receipt、publication receipt、release、policy/schema、
  安全定数をbindし、experiment event fieldとの混在を拒否する。
- [ ] catalog eventはscope proposed→approved→publishing→published/failedと各nonterminal→`INVALID`
  だけを許し、grant/lease/publication receipt条件のないskipを拒否する。
- [ ] publication eventを後更新せず、別のappend-only release-status eventでpublish時に`ACTIVE`、
  その後は`ACTIVE -> REVOKED`だけを許し、release/reason/evidence/effective time/signerをbindする。
- [ ] caller absolute pathではなくopaque logical source IDを使用する。
- [ ] code-owned versioned providerだけを許可する。
- [ ] catalog entryはhash、size、schema、count、times、lineage、snapshot IDだけに制限する。
- [ ] raw row/value/path、username、secret、credential、arbitrary globをcatalogへ含めない。
- [ ] catalog entryとentry refにも`formal_buy=false`、`send_order=false`、`stake=0`を要求する。
- [ ] catalog publication receiptはscope/approval/provider lease/code/env/command、entry-set/
  snapshot-set、provider consume-receipt digest、signer evidenceをexact bindする。
- [ ] release/entry digestをproposalとrun scopeへ埋め、RUNNING直前に再検証する。
- [ ] proposalの有限domainからexactly one entryを決定論的に解決し、proposal後の選択/更新を拒否する。
- [ ] missing/expired/revoked/ambiguous entryはproposal前だけ`BLOCKED_CATALOG`とし、
  `PROPOSED`以後のcatalog再検証失敗はterminal `INVALID`とする。
- [ ] content-addressed immutable snapshotを必須にし、TOCTOUを拒否する。
- [ ] catalog publication時のmetadata hash readを`real_data_execution=true`として正直に記録しつつ、
  model access/training/replay/price/payoff/ROIをfalseにする。

Reviewer decision:

## I. Exact schema and capability binding — P0

- [ ] v2をlegacy proposal/run v1へのfield追加ではなく、別schema/normalizerにする。
- [ ] proposal/run digestへ`gate_kind`、contract version、`execution_kind`を固定する。
- [ ] all schemaで`additionalProperties=false`、duplicate JSON key拒否を採用する。
- [ ] 全phaseで同一exact capability field集合を使い、欠落・未知field・別名を拒否する。
- [ ] `mode`と`effective_after`をboolean capabilityから分離する。
- [ ] `real_data_execution=true`だけでexecutionを許可しない。
- [ ] resultをstrict schema検証し、review digestをvalidated resultへ機械的にbindする。
- [ ] acknowledgementでcomputed outcomeを`REPRODUCED`へupgradeできない。
- [ ] acknowledgementはcomputed outcomeと同じterminalか`REJECTED`だけを選べる。
  computed `INVALID`は`INVALID`以外へ変更できない。
- [ ] proposal、catalog release/entry ref、run、result、review、event/receiptの全artifactで
  `formal_buy=false`、`send_order=false`、`stake=0`をexact fieldとして要求する。
- [ ] 手動OpenAI受け渡しはprovider/model、prompt、sanitized context、responseのhashを
  proposalへbindし、API keyやraw secretを保存しない。

Reviewer decision:

## J. M0 identity freeze — P1

- [ ] proposalは1つのlegacy reference familyと有限のidentity domainだけを固定し、
  未読referenceのexact identityを推測しない。
- [ ] prepublished catalogはproposal domain外を選べず、proposal作成前にexactly one
  recipe/reference/input universeへ解決する。
- [ ] `legacy_run_mode`を`core`または`full`に確定し、UNKNOWNを拒否する。
- [ ] canonical probability stageを`M0_raw`または`M0_temperature_scaled`に確定する。
- [ ] canonical model name、exact command、comparison artifact IDsを固定する。
- [ ] B0/B1/B2/M0/ablationをparity scopeへ含めるか、明示excludeする。
- [ ] candidate preprocessing=train only、L2 grid/validation、train+validation refit、
  temperature grid/calibration、tie-breakをmanifest化する。
- [ ] reference digest、metric unit、fold-mean/race-weighted NLL mappingを確定する。
- [ ] `ai_score`/`ai_rank`のrecursive OOF/as-of/price-blind lineageを証明する。
- [ ] いずれか未確定ならrun scopeと`APPROVED_TO_RUN`を生成しない。

Selected identity:

- legacy run mode:
- canonical probability stage:
- canonical model name:
- parity included artifact IDs:
- explicitly excluded artifact IDs:

## K. Data, label, fold, and probability contracts — P1

- [ ] source/data/runner/fold/feature/target/model/canonicalization/environment/referenceの
  versioned manifestを全て要求する。
- [ ] target、price、resultをfeature tableから物理分離する。
- [ ] train < validation < calibration < reused historical testをrace単位で固定し、
  overlap=0、purge/embargoを検証する。
- [ ] runner IDsとrunner_countを独立manifestへ固定する。
- [ ] 全unordered `C(n,3)`、finite/range/duplicate、Top3 mass 1、wide mass 3を
  tolerance `1e-10`で検証する。
- [ ] label、取消、同着、失格、降着、不完全resultを別validatorで検証する。
- [ ] price、popularity、payoff、ROIをmodel inputとして開けない。

Reviewer decision:

## L. Determinism and outcome — P1

- [ ] canonical artifactはUTF-8/LF、fixed header/row sort、binary64 digest、
  relative pathsのみとし、NaN/Inf/duplicate/absolute pathを拒否する。
- [ ] 2 clean checkoutでinputs、environment、commands、model state、full probability、
  resultのcanonical digestを比較する。
- [ ] bitwise determinismとlegacy numeric toleranceを別checkとして保存する。
- [ ] toleranceを未読reference exposure前にcommitする。
- [ ] trusted canonical legacy digestがない場合の上限を
  `RECONSTRUCTED_NOT_REPRODUCED`とする。
- [ ] structural/probability/safety違反ではequivalenceやROIを解釈しない。
- [ ] artifactへraw row、absolute path、secret、credentialを出さない。

Reviewer decision:

## M. Compatibility and tests — P1

- [ ] legacy v2 proposal/run/event digestsを変更しない。
- [ ] PR #36 infra queue/event v3を変更しない。
- [ ] merged `research/INFRASTRUCTURE_GATE.json`をread-only compatibility inputとし、
  in-place変更やfuture changed-path候補に含めない。
- [ ] new event schema v4を明示dispatchする。
- [ ] legacy unbound real-data RUNNINGはreconciliation/INVALID以外をfail-closeする。
- [ ] activation後にnew grantを発行するlegacy/infra writerもdurable global indexへrouteする。
- [ ] negative testsにkind spoof、score laundering、comment reuse、ledger split-brain/rollback、
  catalog poisoning/revocation、capability flip、duplicate dispatch/lease、terminal resurrectionを含める。
- [ ] Python 3.11/3.12とmixed-schema regressionを確認する。
- [ ] docs、policy、schema、normalizer、compiler、event field集合をcross-checkする。

Reviewer decision:

## N. Explicitly excluded from this design

- [ ] このdesignはreal data、model、training、historical replay、outer OOS、ROIを実行しない。
- [ ] このdesignはproduction/model/candidate/value/BUYを変更しない。
- [ ] future paired ROIは別`roi_prospective_model_validation_v2`へ送り、
  新ID、75点以上、新approvals、price-blind freeze chainを必須にする。
- [ ] user stake、order、notification、credential、purchase APIへ接続しない。
- [ ] governance root変更のmergeとDraft解除を人間に残す。
- [ ] 通常experiment transitionにはPR Ready/mergeを要求しない。

Reviewer decision:

## O. Final review disposition

Choose exactly one:

- [ ] `ACCEPT_DESIGN` — 3-action/no-routine-mergeを含むG1/G2 human-owned Draft設計へ
  進めてよい。実装・実行承認ではない。
- [ ] `REQUEST_CHANGES` — 下記変更後に再reviewする。
- [ ] `REJECT_DESIGN` — reproduction-only audit classを導入しない。

Required changes or rejection reasons:

Human signature or GitHub review reference:
