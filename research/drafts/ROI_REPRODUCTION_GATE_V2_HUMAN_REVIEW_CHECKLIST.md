# ROI Reproduction Gate v2 — human review checklist

作成日: 2026-08-09

対象design:
`docs/ROI_REPRODUCTION_GATE_V2_DESIGN_DRAFT.md`

machine-readable map:
`research/drafts/ROI_REPRODUCTION_GATE_V2_CONTRACT_MAP.design.json`

状態: `REVIEW_ONLY / NON_AUTHORITY / NOT_IMPLEMENTED`

> このchecklistへの記入はGitHub approval、proposal scope、run scope、execution lease、
> merge approvalではない。実装を進める場合も、PR #36の人間判断後のnew mainから
> 別human-owned Draftを作り、G1とG2を順にreview・mergeする。

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

- [ ] G1はschema/policy/compiler/synthetic fixtureだけで、全authorityをfalseにする。
- [ ] G2はG1がhuman mergeされた後の別Draftにし、attester、authority verifier、
  supervised executor、durable one-shot lease ledgerをreviewする。
- [ ] G1/G2/Aは自身や`infrastructure_safety_v1`で自己承認しない。
- [ ] PR #36の未merge branchをroot of trustまたはapproval baseにしない。
- [ ] PR #36の人間判断後、new mainへrebaseしてからimplementationを開始する。
- [ ] governance DraftのCIが変更されたexecutor、model、real-data testを自動実行しない。

Reviewer decision:

## D. State machine and terminal meaning — P0

- [ ] prepare → manifest approval/attest → run approval/run → result reviewの順序を固定する。
- [ ] `MANIFEST_ATTESTED`が存在しないrun scopeを拒否する。
- [ ] terminalを`REPRODUCED`、`RECONSTRUCTED_NOT_REPRODUCED`、
  `REPRODUCTION_FAILED`、`REJECTED`、`INVALID`に分ける。
- [ ] scientific negativeは`REPRODUCTION_FAILED`、human evidence rejectionは
  `REJECTED`、contract violationは`INVALID`とする。
- [ ] 全nonterminalから`INVALID`を許し、terminal resurrectionを拒否する。
- [ ] shadow/production transitionをこのgateへ定義しない。
- [ ] retryは新IDまたは新run scopeと、新しいgrantを必要とする。

Reviewer decision:

## E. Four GitHub grants — P0

- [ ] 次の4 keywordとdigest bindingを承認する。
  - `APPROVED_TO_PREPARE <proposal_scope_digest>`
  - `APPROVED_TO_ATTEST_MANIFEST <manifest_scope_digest>`
  - `APPROVED_TO_RUN <run_scope_digest>`
  - `ACKNOWLEDGED_REPRODUCTION_RESULT <review_digest>`
- [ ] 4 comment IDは相互に異なり、schema/gate kindを跨いでregistry全体で未使用とする。
- [ ] `APPROVERS.json`はproposal base commit上のGitHub blobを正本とする。
- [ ] authorはallowlist済み`User`のみとし、bot/Codex/automationを拒否する。
- [ ] comment編集、削除、ID再利用、body/digest/timestamp不一致をfail-closeする。
- [ ] 各phase直前に必要な全grant、current main、ancestryをGitHubから再検証する。
- [ ] manifest attester/providerとrun approverのrole separationを必須にするか決定する。

Role-separation decision: `REQUIRED / RECOMMENDED / NOT_REQUIRED`

## F. Registry and authority boundary — P0

- [ ] mixed v2/v3/v4 historyをbyte-preservingで読み、legacy eventを再serializeしない。
- [ ] event ID、sequence、previous chain、全historical transitionを検証する。
- [ ] current-main `REGISTRY.jsonl`とのexact equalityを全statusで要求する。
- [ ] main headをlock内でappend直前に再取得し、一度に1 eventだけappendする。
- [ ] alternate registry、symlink/junction、stale main、concurrent writerを拒否する。
- [ ] pending branch event、queue、scope、commentをexecution authorityとして扱わない。
- [ ] registry event safetyは常にauthority/automatic execution/production/merge/BUY=false、
  `formal_buy=false`、`send_order=false`、`stake=0`とする。

Reviewer decision:

## G. One-shot lease and supervised executor — P0

- [ ] verifierはcurrent main、registry bytes、全grant、scope、capability、command、
  execution commitをreplica直前に再検証する。
- [ ] leaseへphase、replica、全digest、expiry、consumption、human supervisor、retry budgetを固定する。
- [ ] leaseのrequired fieldをphase別にし、manifest leaseへ未生成のattestation/run digest、
  preparation/manifest leaseへ未生成のreplica IDをnullで先取りしない。
- [ ] M0は`clean_a`と`clean_b`の2 replicaに別leaseを発行する。
- [ ] durable storeで同じlease/run/replicaの再利用を拒否する。
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

## H. Manifest attestation — P0

- [ ] manifest scopeをprepare/runと別digest・別grantにする。
- [ ] caller absolute pathではなくopaque logical source IDを使用する。
- [ ] code-owned versioned providerだけを許可する。
- [ ] attestation outputはhash、size、schema、count、times、lineage、snapshot IDだけに制限する。
- [ ] raw row/value/path、username、secret、credential、arbitrary globを返さない。
- [ ] output digestをrun scopeへ埋め、RUNNING直前に再検証する。
- [ ] content-addressed immutable snapshotを必須にし、TOCTOUを拒否する。
- [ ] metadata hash readを`real_data_execution=true`として正直に記録しつつ、
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
- [ ] proposal、manifest scope/attestation、run、result、review、event/receiptの全artifactで
  `formal_buy=false`、`send_order=false`、`stake=0`をexact fieldとして要求する。
- [ ] 手動OpenAI受け渡しはprovider/model、prompt、sanitized context、responseのhashを
  proposalへbindし、API keyやraw secretを保存しない。

Reviewer decision:

## J. M0 identity freeze — P1

- [ ] proposalは1つのlegacy reference familyと有限のidentity domainだけを固定し、
  未読referenceのexact identityを推測しない。
- [ ] manifest attestationはproposal domain外を選べず、run scope前にexactly one
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
- [ ] new event schema v4を明示dispatchする。
- [ ] legacy unbound real-data RUNNINGはreconciliation/INVALID以外をfail-closeする。
- [ ] negative testsにkind spoof、score laundering、comment reuse、stale registry、
  capability flip、duplicate lease、terminal resurrectionを含める。
- [ ] Python 3.11/3.12とmixed-schema regressionを確認する。
- [ ] docs、policy、schema、normalizer、compiler、event field集合をcross-checkする。

Reviewer decision:

## N. Explicitly excluded from this design

- [ ] このdesignはreal data、model、training、historical replay、outer OOS、ROIを実行しない。
- [ ] このdesignはproduction/model/candidate/value/BUYを変更しない。
- [ ] future paired ROIは別`roi_prospective_model_validation_v2`へ送り、
  新ID、75点以上、新approvals、price-blind freeze chainを必須にする。
- [ ] user stake、order、notification、credential、purchase APIへ接続しない。
- [ ] mergeとDraft解除を人間に残す。

Reviewer decision:

## O. Final review disposition

Choose exactly one:

- [ ] `ACCEPT_DESIGN` — G1 human-owned Draft設計へ進めてよい。実装・実行承認ではない。
- [ ] `REQUEST_CHANGES` — 下記変更後に再reviewする。
- [ ] `REJECT_DESIGN` — reproduction-only audit classを導入しない。

Required changes or rejection reasons:

Human signature or GitHub review reference:
