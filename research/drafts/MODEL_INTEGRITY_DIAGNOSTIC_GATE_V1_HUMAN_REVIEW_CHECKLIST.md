# Historical AI duplicate-gate impact v1 — human review checklist

対象:

- `docs/MODEL_INTEGRITY_DIAGNOSTIC_GATE_V1_G2_DESIGN_DRAFT.md`
- `research/drafts/MODEL_INTEGRITY_DIAGNOSTIC_GATE_V1_CONTRACT_MAP.design.json`

状態: `HUMAN_REVIEW_REQUIRED / NON_AUTHORITY / NOT_IMPLEMENTED / EXECUTION_FORBIDDEN`

このchecklistへの回答、PR review、Ready化、CI成功、mergeは、G2 activation、canonical scope、
prepare/run/ack grant、実装、実データ、payoff/ROI計算、shadow、production、BUYを許可しない。

## A. Design-only authority boundary

- [ ] baseはGitHub `main` `b103c68dc2418973fda79fddfc1e0f9aac19813a`である。
- [ ] このPRは3つのdesign fileだけを追加し、policy/schema/compiler/runner/testを作らない。
- [ ] `AGENTS.md`、CHARTER、STATE、DECISIONS、REGISTRY、scope、queue、eventを変更しない。
- [ ] model、config、candidate、value、BUY、order、notification、production pathを変更しない。
- [ ] `authority=false`、`canonical_policy_created=false`、`canonical_proposal_created=false`である。
- [ ] `g2_implemented=false`、`lane_implemented=false`、`EXECUTION_FORBIDDEN`である。
- [ ] comparison/ROIを実行せず、`comparison_executed=false`、`roi_delta_measured=false`である。
- [ ] `formal_buy=false`、`send_order=false`、`stake=0`である。

## B. One-recipe gate identity

- [ ] exact gate kindは`historical_ai_duplicate_gate_impact_v1`である。
- [ ] execution kindも`historical_ai_duplicate_gate_impact_v1`である。
- [ ] contract versionは1である。
- [ ] initial classは`AI_DETERMINISTIC_DESCENDANT_GATE_REMOVAL_V1`である。
- [ ] genericな`model_integrity` catch-all kindを作らない。
- [ ] proposalの`model_integrity=true`自己申告だけではrouteできない。
- [ ] trusted-main policyのkind/class/recipe fingerprint一致を全て要求する。
- [ ] design recipe projection SHA-256は`67b0d3e5b92166a10b3077bff03e107c9db071b310f60f43b8758ce316eda878`である。
- [ ] class/recipe追加はnew kind/version/pathとhuman-reviewed mergeを要求する。

## C. Score laundering and ordinary-gate routing

- [ ] ordinary strategy scoreは46/100のまま記録する。
- [ ] `ordinary_strategy_score_record_present=true`でrecord省略を拒否する。
- [ ] `ordinary_strategy_gate_applicable=false`とlane eligibilityを別fieldにする。
- [ ] ordinary statusは`BLOCKED_SCORE`、thresholdは75、`threshold_met=false`である。
- [ ] score creditは0、threshold overrideはfalseである。
- [ ] comment、CI、Ready、merge、G1、結果、ACKでscoreを変更できない。
- [ ] exactly 2 arm、variant 2、threshold search 0、refit/recalibration 0を要求する。
- [ ] 新feature/data/model/target/loss/estimator/context/calibratorを拒否する。
- [ ] candidate/rank/coverage/ticket/stake/market式の変更を拒否する。
- [ ] 追加変更、曖昧さ、performance optimizationがあればordinary 75-point gateへrouteする。
- [ ] diagnostic結果はscore、shadow、promotion、adoption、production権限を生成しない。
- [ ] 現行rootの75点未満実行禁止を黙って迂回しない。
- [ ] exact one-recipe routeを認める別root-governance amendmentをlane activation前にhuman mergeする。
- [ ] amendment未mergeなら`BLOCKED_GOVERNANCE / EXECUTION_FORBIDDEN`を維持する。

## D. Shared G2 authority and cutover

- [ ] current-main ROI Reproduction Gate v2 design 3 fileのexact SHA-256をdependencyとしてbindする。
- [ ] dependency hash domainはcurrent-mainのGit blob bytesであり、checkout後のCRLF bytesではない。
- [ ] G2 core/catalog/ledger/cutoverはこのPRでは実装しない。
- [x] PR #38 G1は`811ffd11bd80447f013c643b96c3eb8145916061`でmainへmerge済みで、compiler-only/no-authorityである。
- [ ] current-mainに残るG1のpre-merge status表示を、G2/lane activation前に別root-governance changeでreconcileする。
- [ ] separate local ledger、SQLite、worktree、branch、file fallbackを禁止する。
- [ ] shared durable runtime ledgerをsole live authorityにする。
- [ ] global head + subject head CAS、authenticated immutable receiptを要求する。
- [ ] comment IDを全gate/version横断でglobal single-useにする。
- [ ] full legacy registry chain、全historical grant ID、terminal/nonterminal headをimportする。
- [ ] old writer freeze、exact snapshot、second compare、atomic activateを要求する。
- [ ] dual writer、rollback、split brain、automatic retryを拒否する。
- [ ] root-governance amendment、G2 core/cutover、lane activationを別human-reviewed PRにする。
- [ ] lane activation changed-path firewallはAGENTS/CHARTER/DECISIONS/scorecard/STATEを拒否する。
- [ ] G2 implementationで確定するG2 core pathsもlane activation PRから拒否する。

## E. GitHub grants and lifecycle

- [ ] routine human-visible actionsは最大3回である。
- [ ] `APPROVED_TO_PREPARE <proposal_digest>`が唯一のworkflow startを兼ねる。
- [ ] `APPROVED_TO_RUN <run_digest>`はprepareと別のunused comment IDを使う。
- [ ] `ACKNOWLEDGED_MODEL_INTEGRITY_RESULT <review_packet_digest>`はさらに別IDを使う。
- [ ] ACK comment直前状態は必ず`MID_REVIEW_REQUIRED`である。
- [ ] Ready、merge、rebase、branch refresh、replica startをroutine actionに数えない/要求しない。
- [ ] `clean_a`/`clean_b`を同一run/catalog/seed、別phase lease、別clean output rootで実行する。
- [ ] 両replica完了後にsemantic outputだけのcanonical comparison projectionを作る。
- [ ] replica ID、lease/receipt、output root/path、executor identity、時刻は比較projectionから除外し、別authority envelopeへbindする。
- [ ] comparison projection bytesのbitwise一致を要求し、不一致は`INVALID`にする。
- [ ] projectionへ`contract_assertions.csv` semantic digestと`contract_status=VALID`をbindする。
- [ ] trusted comparison coordinatorだけがdistinct clean_a/clean_b、same run、各projection/authority/lease completionをbindしたreceiptを発行する。
- [ ] same-replica二重提出、wrong-run、replay、self-asserted match、片側INVALIDを拒否する。
- [ ] comparison coordinatorとresult sealerに別々のdomain-separated one-shot G2 lease/operation receiptを要求する。
- [ ] operation capability flagsをshared 31-field all-false/zero profileに固定し、result後phase timingは継承せずoperation固有effective-afterを使う。
- [ ] 各leaseがrun、operation capability digest、exact input、service commit/environment/verifierをbindし、self-issue/cross-use/local fallbackを拒否する。
- [ ] 両replica完了・一致・comparison receipt seal前の`result.json` sealを拒否する。
- [ ] actorはallowlisted GitHub `User`で、bot/Codex/automationを拒否する。
- [ ] main ancestry、APPROVERS blob、comment body/ID/author/type/timestamps/hashを再検証する。
- [ ] edit/delete/reuse/unavailable/stale-mainをfail-closeする。
- [ ] prepare/run/decision lease/settlement lease/result seal/ACK直前に必要grantを再取得する。
- [ ] lifecycleはMID namespaceを使い、legacy/infra/ROI eventへ偽装しない。
- [ ] self/skip/terminal resurrectionを拒否し、any nonterminal -> INVALIDを許可する。
- [ ] ACKはcomputed outcomeを変更せず、`APPROVED_FOR_SHADOW` transitionを定義しない。
- [ ] run scopeは既存prepare evidenceとfuture authority type/schema/policy digestだけをbindする。
- [ ] future run grant/lease/receipt IDやdigestをrun scopeへplaceholder予約しない。
- [ ] post-freeze run grant/lease/operation receiptが既存run digestを逆向きにbindする。

## F. Exact one-change recipe

- [x] PR #39 source designは`b103c68dc2418973fda79fddfc1e0f9aac19813a`でmainへmerge済みだが、non-canonical/non-authorityである。
- [ ] `p=top1_wide_prob`、`a=p_action_C0_offset`を固定する。
- [ ] lineage式`a=sigmoid(logit(p)+0.130654047367905)`を固定する。
- [ ] decisionはhash-bound stored `a` bytesを使う。
- [ ] materializerはdiagnostic masterとp-action artifactのp/a/keyを全3,746 rowで照合する。
- [ ] candidate entry provenanceへcross-source equality vector digest/attestationをbindする。
- [ ] decision verifierはattestationを検証し、p-action artifact自体をmountしない。
- [ ] trusted verifierはa/decision-base/D0/D1を独立再計算・照合する。
- [ ] `DECISION_BASE`は`eligible_race=true AND candidate_generated=true`である。
- [ ] `RAW_P_GATE_FAMILY`は`p>=0.225`、`p<0.325`、`p>=0.21275851149504352`を内包する。
- [ ] `ACTION_GATE`は`a>=0.25 AND a<0.4`である。
- [ ] D0は`DECISION_BASE AND RAW_P_GATE_FAMILY AND ACTION_GATE`である。
- [ ] D1は`DECISION_BASE AND ACTION_GATE`である。
- [ ] 唯一のAST diffは`RAW_P_GATE_FAMILY` node削除で、内部thresholdは変更しない。
- [ ] D0 subset D1を全rowで要求し、D0-only rowは`INVALID`にする。
- [ ] shared decision-base vector/producer/schema/lineage digestを固定する。
- [ ] candidate key、p、a、calibrator、fit-data、label bytesをarm間で完全共有する。
- [ ] runner自己申告maskをauthorityにせず、trusted verifierが全3,746 rowを再計算する。
- [ ] R0/R1 parityとD0/D1 impactを同じkind/ID/scope/grant/resultへ混ぜない。

## G. Catalog and physical decision/settlement separation

- [ ] `B_LOCAL_HASHED` inventoryはdata identity候補でありauthorityではない。
- [ ] metadata-only G2 catalog publisherにrow projection生成権限を暗黙追加しない。
- [ ] separate projection-materialization kind/path/scope/grantを事前にhuman reviewする。
- [ ] candidate/settlement materializerを別one-shot providerにする。
- [ ] candidate materializerは2 sourceのexplicit allowlisted columnsだけを読む。
- [ ] materializerまたはsnapshot欠落時は`BLOCKED_CATALOG / EXECUTION_FORBIDDEN`にする。
- [ ] candidate-onlyとsettlementをproposal前に別immutable catalog entryへpublishする。
- [ ] proposalはexactly one ACTIVE releaseをbindする。
- [ ] release内にcandidate-only 1 entry + settlement 1 entryのexactly 2 role entriesを要求する。
- [ ] allowed role setをその2つに固定し、extra role entryを拒否する。
- [ ] releaseまたは各roleが0/複数なら`BLOCKED_CATALOG`にする。
- [ ] catalog grant/statusをproposal、run、lease、result、ACK前に再検証する。
- [ ] revoked/mutated/unavailable entry、local path、caller assertion、symlink repointを拒否する。
- [ ] candidate projectionはexplicit allowlist columnsだけを読む。
- [ ] source `top1_pair_key`をcanonical `candidate_key`へexact mappingする。
- [ ] settlementはofficial payoffの`race_id,horse_a,horse_b,wide_pay`だけを読む。
- [ ] settlementの`wide_pay`をprojectionの`official_wide_pay`へexact mappingする。
- [ ] candidate keyはunordered `min-max` decimal/no-zero-paddingで再構成する。
- [ ] settled race内0 matchはmiss、1 matchはhit、複数matchは`INVALID`にする。
- [ ] full co-resident fileを読んでからoutcome/payoff列をdropする実装を禁止する。
- [ ] decision phaseからresult/payoff/odds/market/ROI列とsettlement pathを物理的に見えなくする。
- [ ] D0/D1 decision digest/receiptをfreezeした後だけ別settlement leaseをconsumeする。
- [ ] settlement join後もcandidate/decision digestが一致することを要求する。
- [ ] hit=trueはunique finite positive payoff、hit=falseはnull可/return 0である。
- [ ] missing/duplicate/key mismatch/candidate driftはrow dropせずrun全体を`INVALID`にする。

## H. Cohort and diagnostic metrics

- [ ] primary candidate cohortはfold2–4の3,746 racesである。
- [ ] fold countsは1,661 / 1,653 / 432である。
- [ ] date rangeは2025-01-05..2026-02-15である。
- [ ] ordered race-ID set digestとfold manifestはcanonical proposal前のblockerとしてnullのまま明記する。
- [ ] full 5,336 cohortや144-race Grade-R holdoutを3,746 cohortへ混ぜない。
- [ ] all enrolled racesを1 race 1 rowで保持し、both-no-betもdropしない。
- [ ] offline evaluation notional 100円とartifact `stake=0`を別概念にする。
- [ ] primaryは`sum(delta_profit)/3746`で、ROIはarm固有stake denominatorのsecondaryである。
- [ ] D0/D1ごとにbets/hits/stake/return/profit/ROI denominatorを報告する。
- [ ] final official payoff、reused development OOS、strict T-3 rows 0を明記する。
- [ ] price retention、ex-ante EV、executable ROI、confirmatory OOS、promotionを主張しない。

## I. Sensitivity and bootstrap exactness

- [ ] candidate returnは`official_wide_pay if hit else 0`をarm eligibility前に作る。
- [ ] top1/top3 payout exclusion setを全3,746 candidateから一度だけ作る。
- [ ] exclusion setは両arm共通で、row/stakeを維持しreturnだけ0にする。
- [ ] race-ID set canonical JSON serializationとSHA-256を保存する。
- [ ] winsorはcandidate returnを先に2,000円capし、その後arm eligibilityを適用する。
- [ ] cluster keyは`race_date × venue_code`である。
- [ ] 100,000 replicates、seed 20260814、`Generator(PCG64)`を固定する。
- [ ] cluster/row ordering、exact one draw call、multiplicity込みdenominatorを固定する。
- [ ] lower boundは`numpy.quantile(...,0.05,method="linear")`である。
- [ ] runner/environment hashをrun scopeへbindする。

## J. Capability and safety matrix

- [ ] 31 capability keysを全phaseでexactly同じkey setにする。
- [ ] This Draftは全flag false、stake 0である。
- [ ] prepareはsynthetic fixture testだけtrueで、real data/payoff/ROIはfalseである。
- [ ] catalog maintenanceは`real_data_execution=true`とsupervised metadata hash read=trueを正直に記録する。
- [ ] catalog maintenanceのmodel/raw-row/training/replay/price/payoff/ROIはfalseである。
- [ ] decision freezeはcandidate-only real-data replayで、label/payoff/ROI/priceはfalseである。
- [ ] `historical_replay=true`はdecision freezeとsettlement diagnosticだけに限定する。
- [ ] settlement phaseだけlabel/payoff/offline unit-notional ROIをtrueにできる。
- [ ] unrestricted `raw_row_output=false`を維持する。
- [ ] decision/settlementだけlane-local allowlisted projected rows/counterfactual guardをtrueにする。
- [ ] generic G1 historical reproduction capabilityを変更しない。
- [ ] model fit/inference、training、recalibration、prospective outerを全phaseでfalseにする。
- [ ] research/production candidate mutationを全phaseでfalseにする。
- [ ] workload network/API、credential、purchase、notification、order、mergeをfalseにする。
- [ ] all artifact/receiptで`formal_buy=false`、`send_order=false`、`stake=0`を要求する。
- [ ] capabilityを別scope/grant/catalog/ACKからunionして作れない。

## K. Result and output contract

- [ ] 全future schemaでrecursive `additionalProperties=false`とexact status shapeを要求する。
- [ ] duplicate/unknown/missing key、bool-as-int、NaN/Infinityを拒否する。
- [ ] manual normalizerとschemaのfield parityをtestする。
- [ ] required future outputsのexact filenamesをmachine mapに固定する。
- [ ] pre-ACK canonical `review_packet.json`を必須outputにする。
- [ ] canonical result authorityを`result.json`に固定し、`summary.json`はrender-onlyにする。
- [ ] `replica_comparison_receipt -> result.json -> review_packet.json -> ACK event/receipt`のdigest DAGを固定する。
- [ ] review packetがexact `result.json` digestをbindする。
- [ ] replica roots、comparison root、canonical seal rootのexact layoutとwrite authorityを分離する。
- [ ] cross-root/cross-replica writeを拒否し、後段readerは前段rootをread-onlyにする。
- [ ] `deterministic_digest.json`をchecksum-only non-authorityとし、authority DAGの代用にしない。
- [ ] result sealerはauthenticated comparison receipt/run/policy-schema-verifier/sealer leaseだけをreadする。
- [ ] result sealerはreceipt内のVALID/failure count/disagreement countだけからfixed outcome ruleを適用し、replica claimを直接読まない。
- [ ] `review.md`はrender-onlyでdigest authorityにしない。
- [ ] fresh output dirだけを使い、overwriteを禁止する。
- [ ] CSV/JSON serialization、column order、race ID、bool/float/missing表現を固定する。
- [ ] invalid runはperformance/ROI outputを生成せずfailure/assertion/digestだけを出す。
- [ ] resultはcandidate freeze、settlement audit、3,746 paired rows、sensitivity、bootstrap receiptをbindする。
- [ ] result limitationは`B_LOCAL_HASHED`、nonconfirmatory、reused development、final-payoff onlyである。
- [ ] `strict_t3_rows=0`、`promotion_eligible=false`、`score_credit=0`を固定する。
- [ ] contract failure -> INVALID、disagreement 0 -> no effect、1以上 -> directional effectをmachine ruleにする。
- [ ] pre-ACK review packetにhuman disposition/comment evidenceを含めない。
- [ ] comment後のappend-only ACK eventがreview-packet digest + GitHub evidence digestをbindする。
- [ ] ROI符号で`APPROVED`、`CHAMPION`、`PROFITABLE`へrenameしない。
- [ ] result/ACKからpolicy/config/model/BUY/order/production artifactを作らない。

## L. Future fail-close tests

- [ ] N01 unknown/self-asserted kind/class/recipeを拒否する。
- [ ] N02 score省略/75化/credit/overrideを拒否する。
- [ ] N03 third arm、second change、search/fit/new feature/model/dataを拒否する。
- [ ] N04 parityとimpact scope混在を拒否する。
- [ ] N05 shared decision-base/candidate/p/a/calibrator/lineage driftを拒否する。
- [ ] N06 verifier mask mismatch、D0 not subset、wrong difference setを拒否する。
- [ ] N07 commit/code/env/cwd/argv/formula/threshold driftを拒否する。
- [ ] N08 decisionのresult/payoff/odds/market/co-resident file accessを拒否する。
- [ ] N09 freeze receiptなしsettlementとpost-join redecisionを拒否する。
- [ ] N10 non-3,746、fold混同、duplicate、row/no-bet dropを拒否する。
- [ ] N11 invalid conditional payoff、missing、duplicate、key mismatchを拒否する。
- [ ] N12 miss payoff nullを許可しreturn 0を要求する。
- [ ] N13 arm-specific exclusion、winsor/bootstrap driftを拒否する。
- [ ] N14 catalog repoint/hash/schema/revocation/late bindingを拒否する。
- [ ] N15 catalogをexecution token/grantとして使うことを拒否する。
- [ ] N16 grant reuse/edit/delete/bot/wrong digest/author/stale mainを拒否する。
- [ ] N17 stale/fork/rollback/dual/local ledgerを拒否する。
- [ ] N18 lease replay/cross-phase/cross-kind/double consumeを拒否する。
- [ ] N19 capability unionを拒否する。
- [ ] N20 automatic retry/idempotency replayを拒否する。
- [ ] N21 ACKのresult rewrite/INVALID upgrade/shadow/adoption/merge/BUYを拒否する。
- [ ] N22 positive ROIからpolicy/config/model/BUY/stake/notification/production出力を拒否する。
- [ ] N23 formal BUY/send order/nonzero stake/network/API/credentialを拒否する。
- [ ] N24 strict T-3/confirmatory/executable ROI/Tier-A/promotion虚偽claimを拒否する。
- [ ] N25 chat/Ready/CI/mergeをexperiment grantとして使うことを拒否する。
- [ ] N26 stale ACKをresult/review digest変更後に再利用することを拒否する。
- [ ] N27 duplicate/unknown/missing key、bool-as-int、nonfinite、status-shape mismatchを拒否する。
- [ ] N28 root-governance / G2 core / lane activation scope混在を拒否する。
- [ ] N29 wrong review-packet digestまたは`MID_REVIEW_REQUIRED`以外からのACKを拒否する。
- [ ] N30 one replica未完了/INVALID、same-replica、wrong-run/replay、自称match、comparison mismatch/receipt欠落、比較前result sealを拒否する。
- [ ] N31 ambiguous result authority、summary/deterministic digest authority化、broken digest DAGを拒否する。
- [ ] N32 unauthenticated/self-issued/local/stale/replayed/non-CAS lane activation receiptを拒否する。
- [ ] N33 output root writer/topology/cross-replica write違反を拒否する。
- [ ] N34 unleased/self-issued/cross-used/replayed/local comparison coordinator/result sealer operationを拒否する。
- [ ] N35 result sealerのdirect replica read、自称outcome、receipt input欠落/不一致を拒否する。

## M. Activation order

- [x] PR #38 G1は`811ffd11bd80447f013c643b96c3eb8145916061`で人間merge済みで、authorityはfalseのままである。
- [ ] current-mainに残るG1のpre-merge status表示を別のhuman-reviewed root-governance changeでreconcileする。
- [ ] AGENTS/CHARTER/DECISIONS/scorecardのexact one-recipe root amendmentを別PRでreview・mergeする。
- [ ] G2 core/catalog/ledger/executorをseparate Draft PRで実装・review・mergeする。
- [ ] full legacy/grant migrationとold-writer fence後、human-owned cutover receiptを作る。
- [ ] projection-materialization maintenanceを別PRでreview・mergeし、別grantで2 snapshotを作る。
- [ ] exact one-recipe lane policy/schema/compilerをさらに別PRでreview・mergeする。
- [ ] lane implementation merge時点は`MERGED_NOT_ACTIVE`で、自分をactivateできない。
- [ ] allowlisted GitHub `User`の`ACTIVATE_HISTORICAL_AI_DUPLICATE_GATE_IMPACT_V1 <activation_scope_digest>`を別に取得する。
- [ ] shared G2 durable ledger authority writerだけがremote evidence再検証後にone-shot subject-head CASでactivation receiptを発行する。
- [ ] receiptはactivation scope/human evidence/current main/root/G2/recipe/policy-schema-compiler-verifier/heads/writer-signer/time/safetyをbindする。
- [ ] caller/lane executor/local backendによる発行、stale/replay/duplicate/self-issued/non-CAS receiptを拒否する。
- [ ] activation receiptをexperiment prepare/run grantとして数えない。
- [ ] post-activation mainから新experiment ID/canonical proposalを作る。
- [ ] exact catalog binding後、unused prepare grantを取得する。
- [ ] research-only runner/synthetic fixturesを作り、run scopeをfreezeする。
- [ ] distinct unused run grantを取得し、全authority evidenceを再検証する。
- [ ] supervised decision lease、settlement lease、result sealを順に行う。
- [ ] distinct result acknowledgementを取得する。
- [ ] production cleanup/adoptionは別ordinary strategy proposal/PR/approvalへ残す。

## Human review decision

- [ ] この一件限定gateを将来実装する設計として承認する。
- [ ] 修正を要求する。
- [ ] 設計をrejectし、ordinary 75-point gateだけを維持する。

Reviewer note:

```text

```
