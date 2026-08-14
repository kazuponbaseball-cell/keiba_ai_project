# Model Integrity Diagnostic Gate v1 / G2 activation extension — human-review design draft

- Status: `HUMAN_REVIEW_REQUIRED / NON_AUTHORITY / NOT_IMPLEMENTED`
- Authority: `false`
- Execution: `EXECUTION_FORBIDDEN`
- Base repository: `kazuponbaseball-cell/keiba_ai_project`
- Base branch: `main`
- Design base commit: `b103c68dc2418973fda79fddfc1e0f9aac19813a`
- Date: `2026-08-14`

> このDraftは設計レビューだけを求める。G2、catalog、ledger、schema、runner、test、
> canonical scope、grant、registry eventを作らず、実データ、backtest、outer OOS、payoff、
> ROIを読まない。`formal_buy=false`、`send_order=false`、`stake=0`を維持する。

Machine-readable companion:
`research/drafts/MODEL_INTEGRITY_DIAGNOSTIC_GATE_V1_CONTRACT_MAP.design.json`

Human checklist:
`research/drafts/MODEL_INTEGRITY_DIAGNOSTIC_GATE_V1_HUMAN_REVIEW_CHECKLIST.md`

## 1. Outcome

独立情報を増やさない構造整理を、通常strategy scoreを水増しせず検証できる、一件限定の
model-integrity laneを設計する。

最初に想定する診断classは
`AI_DETERMINISTIC_DESCENDANT_GATE_REMOVAL_V1`である。これは、同じ`ai_score`由来の
決定論的な値が複数gateへ再注入されている構造について、凍結済みreferenceからexactly one
typed gate-family nodeを除いた場合の判断差を測るnegative-controlである。

このlaneは、通常の75点strategy gateの例外承認ではない。

- ordinary strategy scoreは46/100、`BLOCKED_SCORE`、credit 0のまま保存する
- laneのhard classifierを満たしたnon-promotion診断だけを別namespaceへrouteする
- model、feature、target、calibrator、threshold、candidate、market式を改善する試験はrouteしない
- 結果が良くてもscore、shadow、promotion、production、BUYの権限を一切付与しない
- adoptionを検討する場合は、新しい通常strategy proposalと75点以上の審査へ戻す

## 2. Gate identity and anti-bypass boundary

将来のversioned identityは次とする。

```text
gate_kind       = historical_ai_duplicate_gate_impact_v1
contract_version = 1
execution_kind  = historical_ai_duplicate_gate_impact_v1
initial_diagnostic_class = AI_DETERMINISTIC_DESCENDANT_GATE_REMOVAL_V1
```

proposal側の`model_integrity=true`という自己申告ではlaneへ入れない。trusted policyに埋め込んだ
gate kind、diagnostic class、recipe fingerprintが全て一致する場合だけrouteする。初版は上記1 classと
D0/D1 recipeだけを認識し、classまたはrecipe追加は新しいkind/version/pathのhuman-reviewed
governance PRを必要とする。genericな`model_integrity` catch-all kindは定義しない。

このdesign mapのcanonical `recipe_contract` projection SHA-256は
`67b0d3e5b92166a10b3077bff03e107c9db071b310f60f43b8758ce316eda878`である。
future policyはhuman-reviewed exact recipe bytesを再hashし、この値または新versionで承認された値だけを認識する。

次のどれかを含む場合は、このlaneを拒否し、通常strategy gateへrouteする。

- 3 arm以上、複数のone-change、variant探索
- 新feature、raw data、model、target、loss、estimator、calibrator、context signal
- refit、retraining、recalibration、parameter search、threshold search
- candidate pair、candidate universe、coverage、ticket budget、stake ruleの変更
- market、odds、payoff、resultをdecision freeze前に利用
- prospective outer、shadow、promotion、production、BUY、注文、通知
- performanceを見てcohort、cutoff、formula、除外条件を変更
- resultをstrategy scoreへ加点する目的

ambiguousな場合もfail-closeし、laneへ推測routeしない。

## 3. Honest score record

全proposal、queue、event、run、result、reviewは次の意味を保持する。

```json
{
  "ordinary_strategy_score_record_present": true,
  "ordinary_strategy_gate_applicable": false,
  "diagnostic_lane_eligibility_is_separate": true,
  "recorded_total": 46,
  "ordinary_strategy_status": "BLOCKED_SCORE",
  "score_threshold_met": false,
  "credit": 0,
  "threshold_override": false
}
```

`ordinary_strategy_gate_applicable=false`はscoreを削除する意味ではない。record自体の存在を必須とし、
46点というordinary strategy評価を残した上で、
non-promotion structural diagnosticのhard classifierだけを使う。人間comment、CI成功、PR Ready、merge、
G1成功のいずれも、このrecordを75点へ変更できない。

現行`AGENTS.md`、CHARTER、DECISIONSは75点未満を実行不可としており、このDraftだけでは例外を
作れない。G2やlane codeより先に、一件限定non-promotion routeを明記する別のhuman-reviewed
root-governance amendmentをmainへmergeしなければならない。そのamendmentは46点/credit 0を維持し、
上記exact kind/class/recipe以外を拒否する。amendmentが未mergeなら、全実装が存在しても
`EXECUTION_FORBIDDEN`を維持する。

## 4. G2 is a shared prerequisite, not a new local backend

このlaneは、mainにあるROI Reproduction Gate v2設計のG2 control planeを継承する。
別ledger、別grant index、local fallbackを作らない。design dependencyはbase commit
`b103c68dc2418973fda79fddfc1e0f9aac19813a`のGit blob bytesである。checkout時の
CRLF変換後bytesはhash正本に使わない。

- `docs/ROI_REPRODUCTION_GATE_V2_DESIGN_DRAFT.md`
  SHA-256 `4ed7e54c87f638f37e23c866d2030b431f9115328d9ccd002befeefcd68c2039`
- `research/drafts/ROI_REPRODUCTION_GATE_V2_CONTRACT_MAP.design.json`
  SHA-256 `44b7a8921e3cbdcfcfdb1887b52292b00a025129d7791f828e03d06060919c7d`
- `research/drafts/ROI_REPRODUCTION_GATE_V2_HUMAN_REVIEW_CHECKLIST.md`
  SHA-256 `e4aaadfac87168a9fe947caaec1302dfb0bf24bccabacdf01530de61822f4af9`

継承するnormative boundaryは次である。

- authenticated durable runtime ledgerをsole live authorityにする
- global headとsubject headをserializable transaction/CASで更新する
- approval comment IDを全gate、全schema、全experimentでglobal single-use reserveする
- current registryのfull byte-exact chain、全grant ID、terminal/nonterminal headをcutover前にimportする
- exact current-main/registry bytesをfreezeし、legacy writerを停止、second compare後にatomic activateする
- activate後の`REGISTRY.jsonl`はaudit checkpointでありlive authorityではない
- immutable prepublished reference catalog、release status、revocationを使用する
- GitHub commentのactor、body、digest、timestamp、edit/delete/reuseを毎gateで再検証する
- one-shot leaseをatomic consumeしてからだけsupervised executorを起動する
- no local file/SQLite/worktree/branch fallback、no dual writer、no automatic retry

本DraftはG2を実装・activateしない。PR #38のG1は
`811ffd11bd80447f013c643b96c3eb8145916061`でmainへmerge済みだが、
compiler-only/no-authorityである。PR #39も
`b103c68dc2418973fda79fddfc1e0f9aac19813a`でmainへmerge済みだが、source designであって
scope/grant/authorityではない。G2 implementationとhuman cutoverが完了するまで、このlaneは常に
`EXECUTION_FORBIDDEN`である。

なおcurrent-mainの`research/STATE.yaml`と一部G1 design/checklistには、PR #38 merge前の
`PENDING_HUMAN_REVIEW_AND_MAIN_MERGE`または`NOT_IMPLEMENTED`表示が残る。本PRはroot文書を変更せず、
G2またはlane activationより前に、別のhuman-reviewed root-governance changeでmerge事実と
no-authority状態へreconcileする。reconciliation未完了時はfail-closeする。

## 5. Eligibility — all checks are non-compensating

laneへ入れるproposalは、全項目を満たす。

1. exactly two arms: frozen reference D0とsingle-change D1
2. D1はpredeclared deterministic-descendant gate-familyのexactly one typed node deletion
3. canonical AST diffはその1 gate-family node以外byte-equivalent
4. arm間でinput rows、candidate key、calibrator bytes、non-AI clauses、market formula、stake ruleが同一
5. `variant_count=2`、`threshold_search_count=0`、`refit_count=0`
6. no new feature/data/model/target/loss/estimator/context/calibrator
7. no training、inference、recalibration、parameter fitting
8. candidate-only projectionをoutcome/payoffなしで先にfreezeする
9. decision freeze後だけsettlement projectionを別leaseでjoinする
10. all enrolled racesを1 race 1 rowで保持し、no-betもdropしない
11. historical label/payoffはdiagnostic settlementだけに使い、selectionやtuningへ戻さない
12. resultはnonconfirmatory、reused-development、final-payoff-onlyのlimitationを保持する
13. outcomeはmodel adoption、shadow、production、score creditを決めない
14. workload network、credential、notification、purchase、order、merge pathを持たない
15. `formal_buy=false`、`send_order=false`、`stake=0`

one-change proof、lineage、catalog、candidate freeze、settlement completenessのどれかが不明なら
`INVALID`とし、ROIを計算・解釈しない。

## 6. Canonical objects and domain separation

将来G2 implementationは新しいpath/versionで少なくとも次を定義する。

| Object | Purpose |
|---|---|
| duplicate-gate impact policy v1 | initial diagnostic-class/recipe fingerprint、capability、forbidden route |
| proposal v1 | honest score、D0/D1 AST、cohort、catalog binding、one-change proof |
| run v1 | proposal全体、execution commit、catalog、code/config/env/seed/command |
| result v1 | candidate freeze、settlement join、paired rows、metrics、limitations |
| review packet v1 | pre-ACK result digestとmachine-derived classification。human dispositionなし |
| acknowledgement event/receipt v1 | post-comment GitHub evidenceとreview-packet digestをappend |
| queue envelope v5 | exact kind/version dispatch。legacy v2、infra v3、ROI v4と混同しない |
| model-integrity event v1 | lane専用lifecycle。ROI registry eventへ偽装しない |
| G2 lease/operation/receipt refs | shared durable ledgerのdomain-separated authority evidence |

全top-level canonical artifact/receiptはpolicy/schema SHA-256、current main、safety constantsをbindする。
embedded value objectはenclosing artifact bindingを継承する。kind、version、phase、experiment、scope、
grant、catalog、lease、operation digestをdomain-separatedにし、cross-kind substitutionを拒否する。

Markdownはauthorityではない。canonical JSONはUTF-8、BOMなし、key sort、compact separator、
Unicode保持、NaN/Infinity禁止、set listの重複禁止/sort、command listの順序保持とする。
全future schemaはrecursive `additionalProperties=false`、duplicate/unknown/missing key reject、
bool-as-int reject、status-conditioned exact shape、manual normalizerとのfield parityを要求する。

## 7. Reference catalog and physical decision/settlement separation

`B_LOCAL_HASHED` pathはinventoryであってauthorityではない。proposal前に、別のcatalog maintenance
approvalでcontent-addressed immutable snapshotへ公開する。

ただし、継承元G2 catalog publisherはmetadata/hash publicationだけを行い、結果列が同居するmasterから
row projectionを新規生成する権限を持たない。candidate-only/settlement snapshotが未materializeなら、
別kind/version/pathのprojection-materialization maintenanceをhuman review・mergeし、別scope/grant、
別one-shot providerで2 snapshotを先に作る必要がある。このDraftはそのproviderを実装・承認しない。
candidate materializerはdiagnostic masterとp-action artifactから
`race_id,top1_pair_key,top1_wide_prob,p_action_C0_offset`等のexplicit allowlisted source columnsだけを読み、
全3,746 rowのcross-source equality vector digest/attestationをcandidate entry provenanceへbindする。
decision/metricは計算しない。
settlement materializerはofficial payoff sourceだけを読み、candidate decisionを読まない。未実装または
snapshot欠落なら`BLOCKED_CATALOG / EXECUTION_FORBIDDEN`である。

初版診断はexactly one ACTIVE releaseに、role別でexactly two entriesを必要とする。allowed role setは
`candidate_only_projection`と`settlement_projection`だけで、extra role entryを禁止する。

1. `candidate_only_projection`
   - race/candidate/calibrated probability/eligibilityに必要なallowlisted列だけ
   - result、hit、payoff、odds、ROI列を物理的に含まない
   - exact schema、row/race count、unique key、date/fold、content digestをbind
   - source `top1_pair_key`をcanonical `candidate_key`へ写像し、explicit source-usecolsだけを読む
2. `settlement_projection`
   - decision freeze後にだけ使うofficial result/payoffとjoin key
   - candidateを選び直す列やmodel inputを含まない
   - source/provenance/as-of class、schema、content digestをbind
   - official payoff sourceの`race_id,horse_a,horse_b,wide_pay`だけを読み、`wide_pay`を
     `official_wide_pay`へ写像し、unordered pair keyを再構成する

candidate keyは`min(horse_no)-max(horse_no)`の10進数・zero paddingなしで固定する。settled raceが存在し、
candidate pairがexactly one rowに一致すればhit、0 rowならmiss、複数一致なら`INVALID`とする。

catalog publisherはraw valuesをreleaseへ埋め込まずmetadata/content digestだけを公開する。
proposalはexactly one release、one candidate role entry、one settlement role entryをbindし、
releaseまたは各roleが0件/複数件なら`BLOCKED_CATALOG`とする。release grantと
latest ACTIVE statusをproposal、run freeze、lease issue/consume、result seal、acknowledgement直前に
再検証する。revoked/mutated/unavailableなら`INVALID`である。

executionは2 phaseに分ける。

```text
DECISION_FREEZE:
  candidate-only entry only
  settlement/result/payoff/price access = false
  seal D0/D1 eligibility + candidate digest

SETTLEMENT_DIAGNOSTIC:
  requires sealed decision digest
  settlement entry only
  candidate digest must remain unchanged
  official payoff + offline unit-notional paired metrics only
```

同じprocess mountや同じunfiltered CSVで両phaseを済ませることを禁止する。

## 8. Lifecycle and three human-visible actions

pre-registry states:

```text
BLOCKED_G2
BLOCKED_CATALOG
BLOCKED_ELIGIBILITY
ROUTE_TO_STRATEGY_GATE
```

registered lifecycle:

```text
MID_PROPOSED
  -> MID_APPROVED_TO_PREPARE
  -> MID_PREPARING
  -> MID_RUN_APPROVAL_REQUIRED
  -> MID_APPROVED_TO_RUN
  -> MID_RUNNING
  -> MID_REVIEW_REQUIRED
       -> MID_ACKNOWLEDGED_NO_DECISION_EFFECT
       -> MID_ACKNOWLEDGED_DIRECTIONAL_EFFECT
       -> INVALID
```

any nonterminal state may transition to`INVALID`; self-transition、skip、terminal resurrectionは禁止する。
terminal classはsealed result/review packetからmachine-derivedし、人間acknowledgementは符号や分類を変更しない。
contract failureがあれば`INVALID`、それ以外でdecision disagreement countが0なら
`MID_ACKNOWLEDGED_NO_DECISION_EFFECT`、1以上なら`MID_ACKNOWLEDGED_DIRECTIONAL_EFFECT`とする。
R0/R1 parity refactorはこのimpact experimentへ混ぜず、別kind/ID/scope/grantで扱う。

success pathのhuman-visible action上限は3である。

1. `APPROVED_TO_PREPARE <proposal_digest>`
2. `APPROVED_TO_RUN <run_digest>`
3. `ACKNOWLEDGED_MODEL_INTEGRITY_RESULT <review_packet_digest>`

prepare commentはwaiting workflowの唯一のstart actionを兼ねる。別start、replica start、Ready、merge、
rebase、branch refresh操作をroutine experimentへ要求しない。3 commentは相互に異なり、global grant
indexで未使用でなければならない。acknowledgementはshadow、promotion、production、BUY、mergeを
許可しない。

policy/schema/catalog/ledger/executorを変えるgovernance PRの人間mergeは、この3 actionとは別である。
それらがmerged/activated済みでなければexperiment actionを開始できない。

## 9. Prepare and run authority

prepare grantが許可するのは、承認scope内のresearch-only runner/config/schemaとsynthetic fixture
unit testの準備だけである。prepare中はreal data、historical replay、payoff、ROIを禁止する。

run scopeはproposal全体とdigestに加え、次をexact bindする。

- execution commit、clean worktree、changed-path allowlist
- policy/schema/compiler/verifier/executor digest
- D0/D1 canonical AST、shared decision-base vector/producer/schema/lineageとone-change proof
- catalog release、candidate entry、settlement entry
- candidate projection/fold/enrollment/settlement manifest
- interpreter、dependency lock、environment、timezone、locale
- exact structured argv/cwd/env/timeout、seed/RNG、compute budget
- output root/schemaと全capability/safety constants
- prepare grant evidence digest
- 後続run grant、decision/settlement lease、operation receiptに要求するtype/schema/policy digest

genericな`real_data_execution=true`はauthorityではない。exact gate kind、phase、run digest、grants、
catalog、current main、one-shot leaseが一致した場合だけ、supervised executorがphase-specific accessを得る。
retry budgetは0。再実行には新scopeと新grantを必要とする。

run grant evidence、decision/settlement lease、operation/consumption receiptはrun scope freeze後に生じるため、
run scope digestへ含めない。これらappend-only authority objectが既存run digestを逆向きにbindし、result/reviewが
各evidence digestをbindする。placeholder、self-reference、future receipt IDの事前予約をrun digestへ混ぜない。

runは`clean_a`と`clean_b`の2 replicaを、同じrun/catalog/seedと別々のone-shot phase lease、
clean isolated output rootで実行する。人間のreplica start操作は0である。両replicaが完了した後、
candidate/decision/settlement/paired rows、metrics、sensitivity、bootstrap distributionと共有contract digestだけを
含むcanonical `replica_comparison_projection.json`を各replicaから作る。projectionは
`contract_assertions.csv`のsemantic digestと`contract_status=VALID`もbindし、そのbytesを比較する。replica ID、
lease/operation/receipt ID・digest、output root/path、executor invocation、時刻は比較projectionから除外するが、
各replica固有のauthority envelopeへ別々にbindする。比較projectionがbitwise一致しなければ`INVALID`とし、
`replica_comparison_receipt.json`もcanonical resultもsealしない。片方だけの完了や比較receiptなしでresultを
sealしてはならない。

comparison receiptはhash-bound trusted comparison coordinatorだけが発行する。receiptはrun scope、distinctな
`clean_a`/`clean_b` IDと各projection digest、各replicaのauthority envelope・completed phase lease receipt、
両contract status `VALID`、contract failure count 0、disagreement count、bitwise equality結果、
comparison verifier/schema/policy/execution commit、issued_atをbindする。
replica executorのself-assertion、同じreplicaの二重提出、別run、replay、片側INVALIDは`INVALID_NO_RESULT_SEAL`とする。

comparison coordinatorはshared G2 durable ledgerが発行・consumeしたdomain-separated one-shot leaseの下だけで動く。
leaseはrun、両projection/authority envelope、coordinator commit/environment/verifier、operation capability digestを
bindし、coordinatorは自分のleaseを発行・承認できない。comparisonはsealed-artifact-onlyなので、capability flagsは
`review_acknowledgement.flags`と同じexact 31-field all-false/zero profileにするが、同phaseのresult後という
`effective_after`は継承しない。comparison operation自身のeffective-afterを両replica完了・VALIDに固定し、settlement
payoff/ROI capabilityを継承しない。receiptはcoordinator authority envelopeとlease/operation receiptも
bindする。canonical result sealerも別のone-shot G2 leaseを必要とし、そのleaseはrun、comparison receipt、
result policy/schema/verifier、sealer commit/environment、同じall-false/zero operation capability digestをbindする。2 operationのlease/receiptを
cross-use、replay、local fallbackしてはならない。

output topologyはfresh run rootの下で分離する。`replicas/clean_a/`と`replicas/clean_b/`は対応するreplica executorだけ、
`comparison/`はcomparison coordinatorだけ、`canonical/`はresult sealerだけがwriteできる。cross-root/cross-replica writeは
禁止し、後段は前段rootをread-onlyで読む。各replica rootはscientific outputs、assertions、comparison projectionを持ち、
comparison rootはreceiptだけ、canonical rootは`result.json`、`review_packet.json`、render-only summary/reviewを持つ。
各replica rootとcanonical rootの`deterministic_digest.json`はchecksum-only non-authority indexであり、projection、receipt、result、review packetの
代用にもcanonical authority DAGの一部にもならない。

## 10. Result classification is non-promotion

canonical result authorityは`result.json`だけである。`summary.json`と`review.md`はrender-onlyでdigest authorityではない。
`result.json`は両replica完了後の`replica_comparison_receipt.json` digestをbindし、その後のpre-ACK canonical
`review_packet.json`は次を必ずbindする。

result sealerのread allowlistはauthenticated comparison receipt、run scope、result policy/schema/verifier、sealer lease/operation
receiptだけである。replica rootやreplicaのoutcome claimを直接読まず、receiptにseal済みの
`both_contract_status_valid`、`contract_failure_count`、`disagreement_count`だけから固定ruleでcomputed outcomeを作る。
missing/extra/mismatchがあればresultをsealしない。

- full 1-row-per-enrolled-race candidate digest before settlement
- settlement join audit、duplicate/missing/conditional payoff checks
- D0/D1 eligible、stake、return、profit、deltaのpaired rows
- arm別bet count、stake denominator、hit、return、profit、ROI
- race当たりpaired mean delta profit
- common high-payout exclusion/winsor sensitivity
- exact cluster bootstrap receipt
- `nonconfirmatory=true`
- `reused_development_oos=true`
- `strict_t3=false`
- `final_official_payoff_only=true`
- `promotion_eligible=false`
- `score_credit=0`
- safety constants

`review_packet.json`はexact `result.json` digest、computed outcome/rule、limitations、policy/schema digest、run digestだけを
bindし、人間dispositionやcomment evidenceを含まない。`review.md`はrender-onlyでdigest authorityではない。
ACK commentを受け付ける直前状態は必ず`MID_REVIEW_REQUIRED`であり、人間commentは
`ACKNOWLEDGED_MODEL_INTEGRITY_RESULT <review_packet_digest>`としてreview-packet digestをbindする。
その後のappend-only acknowledgement event/receiptが
review-packet digestとGitHub evidence digestをbindする。ACK eventはresult、packet、computed outcomeを変更しない。
canonical digest DAGは
`replica_comparison_receipt -> result.json -> review_packet.json -> acknowledgement event/receipt`で一方向に固定する。

terminal classificationは判断差の有無とcontract validityだけを表す。positive ROI、negative ROI、
bootstrap intervalはterminal名を`APPROVED`、`CHAMPION`、`PROFITABLE`へ変更しない。

resultを見て構造変更を採用したい場合も、このlaneからproductionへ進まない。別のordinary strategy
proposalで、untouched prospective evidence、75点以上、別approvalを満たす。

## 11. D0/D1 example — non-authoritative binding

PR #39 head `f30e9fcd7e07f7645da542998f1b343d05ae5b68`からmain merge commit
`b103c68dc2418973fda79fddfc1e0f9aac19813a`へ入ったsingle-source designを、
初版classのsource exampleとする。source JSONのGit blob bytes SHA-256は
`2ef0acec15c34e959993d83eab590650f3be3bd00c6b72065c72085bca3c9672`である。
ただしPR #39はcanonical scopeでもgrantでもなく、このDraftへformulaを載せても実行権限にならない。

```text
p = top1_wide_prob
a = p_action_C0_offset
a = sigmoid(logit(p) + 0.130654047367905)
decision_base = eligible_race AND candidate_generated

D0_REFERENCE = decision_base
               AND 0.225 <= p < 0.325
               AND 0.25 <= a < 0.4
               AND p >= 0.21275851149504352

D1_REMOVE_RAW_GATES = decision_base
                      AND 0.25 <= a < 0.4
```

only permitted AST changeはD0の`RAW_P_GATE_FAMILY` node deletionである。このnodeは3つの既存raw-`p`
clausesを内包し、内部thresholdを変えない。calibrator、threshold、candidate、
non-AI clause、market formula、stake rule、cohortを変えない。

trusted verifierはcandidate entry provenanceにあるcross-source equality attestationを検証し、
hash-bound runtimeで`a`、shared decision-base、D0、D1を全3,746 rowについて再計算する。
p-action artifact自体はdecision phaseへmountしない。upstream Top3 posteriorがないのに
`p`を再推論したと自己申告することや、runner-provided maskをauthorityとして使うことを禁止する。

proposed diagnostic cohortはfold2–4の3,746 races、2025-01-05から2026-02-15である。
全raceをpaired denominatorに残し、100円unit notionalを評価用に使うが、artifactの`stake`は0のままにする。

```text
evaluation_stake_arm = 100 if eligible_arm else 0
evaluation_return_arm = official_wide_pay if eligible_arm and candidate_hit else 0
evaluation_profit_arm = evaluation_return_arm - evaluation_stake_arm
delta_profit = profit_D1 - profit_D0
```

`hit=true`だけ一意・finite・positive payoffを必須とし、`hit=false`のpayoff nullはreturn 0とする。
missing result、duplicate、candidate drift、conditional payoff不正はrow dropせずrun全体を`INVALID`にする。

primaryは`mean(delta_profit) per enrolled race`、ROIはarm固有stake denominatorのsecondaryである。
high-payout top1/top3集合と2,000円winsorは全3,746 candidate returnからarm-independentに一度だけ作る。
uncertaintyは`race_date × venue_code` cluster bootstrap、100,000回、seed `20260814`、PCG64を固定する。

このcohortはreused development OOSかつfinal official payoffだけで、strict T-3 overlapは0である。
したがって価格残存、ex-ante EV、実行可能ROI、confirmatory outer OOSを主張しない。

## 12. Capability phases

31-field capability setはG2 shared policyと同じ名称を使い、全phaseでexact key equalityを要求する。

| Phase | Synthetic | Real candidate | Label/payoff | Historical diagnostic ROI | Production/BUY/order |
|---|---:|---:|---:|---:|---:|
| This Draft | false | false | false | false | false |
| Catalog maintenance | false | supervised source hash read | false | false | false |
| Prepare | true | false | false | false | false |
| Decision freeze | false | true | false | false | false |
| Settlement diagnostic | false | frozen projection only | true | true | false |
| Review/ack | false | sealed artifacts only | sealed artifacts only | no recomputation | false |

catalog maintenanceのsource hash readはmetadata-onlyでも`real_data_execution=true`として正直に記録し、
model/raw-row/training/replay/price/payoff/ROIはfalseにする。

`historical_replay=true`はhash-bound historical candidate rowsへD0/D1を適用するdecision freezeと、
post-freeze settlement diagnosticの2 phaseだけで許可する。
`historical_result_label_access`、`backtest_interpretation`、`payoff_access`、`roi_calculation`、
`offline_unit_notional_evaluation`はsettlement diagnosticだけtrueにできる。
それ以外のphaseではfalseである。全phaseで次はfalse/0のまま固定する。

- actual Codex dispatch、automatic execution/approval
- training、model fit/inference、recalibration、prospective outer
- price access、research/production candidate mutation
- workload network/external API、credential、purchase path
- production change、shadow approval、merge、notification、order
- `formal_buy=false`、`send_order=false`、`stake=0`

`raw_row_output=false`はunrestricted source row output禁止を意味する。lane-local guardとして、decision/
settlement phaseだけ`allowlisted_projected_research_rows=true`と
`historical_value_eligibility_counterfactual=true`をexact policyに持たせる。他scopeのflagとのunionは禁止する。

## 13. Future implementation boundary

このdesignが人間review・mergeされても実行は開始しない。順序は次で固定する。

1. **完了済み**: PR #38 G1を`811ffd11bd80447f013c643b96c3eb8145916061`でhuman mergeした。G1はno-authorityのまま。
2. current-mainに残るG1のpre-merge status表示を、別のhuman-reviewed root-governance changeでreconcileする。
3. AGENTS/CHARTER/DECISIONS/scorecardへ一件限定non-promotion routeを追加するroot-governance amendmentを別PRでhuman mergeする。
4. G2 core/catalog/ledger/cutoverを別のhuman-reviewed implementation PRで作る。
5. G2をhuman mergeし、complete legacy/grant migration後にhuman-owned cutoverを行う。
6. projection-materialization maintenanceを別PRでreview・mergeし、別grantで2 snapshotをmaterialize/publishする。
7. `historical_ai_duplicate_gate_impact_v1` policy/schema/compilerを別のhuman-reviewed implementation PRで作る。
8. merge時点では`MERGED_NOT_ACTIVE`とし、allowlisted GitHub `User`が
   `ACTIVATE_HISTORICAL_AI_DUPLICATE_GATE_IMPACT_V1 <activation_scope_digest>`を別に承認する。
   shared G2 durable ledger authority writerだけが、remote evidence再検証後にlane subject headを
   `MERGED_NOT_ACTIVE -> ACTIVE`へone-shot CASし、activation receiptを発行する。
9. post-activation current mainから新experiment IDのcanonical D0/D1 proposalを作る。
10. Issueでexact prepare grantを取得し、research-only implementation/synthetic testsを行う。
11. run scopeをfreezeし、別comment IDのrun grantを取得して再検証する。
12. supervised two-stage diagnosticを実行し、sealed reviewを人間がacknowledgeする。

root-governance amendment、G2 core、lane implementation/activationを1つのself-activating PRへまとめない。
lane implementation PRは自分をactivateできず、activation receiptはexperiment grantではない。production builder、existing model、
candidate/value/BUY path、current registry bytes、current gatesをこのdesign PRで変更しない。

activation receiptの正本kindは`historical_ai_duplicate_gate_lane_activation_receipt_v1`である。receiptは
activation scope、allowlisted human GitHub evidence、current main、root amendment、G2 backend/head、recipe、
policy/schema/compiler/verifier、previous/new lane subject head、G2 writer/signer、issued_at、安全flagをbindする。
caller、lane compiler/verifier/executor、local backendはreceiptを発行できない。stale、replay、duplicate、self-issued、
local、non-CAS receiptはfail-closeし、laneを`MERGED_NOT_ACTIVE`のままにする。このactivation actionはroutine
experimentの3 comment外であり、prepare/run grantの代用にはならない。

## 14. Required future negative matrix

future implementationは少なくとも次をfail-close testする。

- wrong/missing/duplicate kind、version、JSON key、bool-as-int、NaN/Infinity
- recursive schemaのunknown/missing field、status shape、normalizer-field drift
- ordinary score 46を75へ変更、credit非0、threshold override
- proposalの自己申告だけでlane route、unknown diagnostic class
- third arm、second AST change、threshold/refit/recalibration/feature/model/target追加
- candidate、calibrator、market式、non-AI clause、stake rule drift
- outcome/payoff/odds column in decision phase、same unfiltered file mounted to both phases
- candidate digest changes after settlement join
- missing/duplicate race、row drop、no-bet drop、miss payoff nullをerror扱い、hit payoff不正
- arm-specific high-payout exclusion、bootstrap seed/cluster/draw drift
- catalog selected after proposal、revoked release、local path/caller assertion fallback
- edited/deleted/reused grant、bot actor、wrong digest、stale main、unauthorized approver
- incomplete legacy import、dual writer、ledger rollback/split brain、local fallback
- expired/reused lease、automatic retry、executor self-issuing lease
- result/ackによるscore、shadow、promotion、production、BUY、merge upgrade
- any network、credential、notification、order side effect、nonzero stake
- root-governance、G2 core、lane activation changed pathsの同一PR混在/self-activation
- wrong review-packet digestまたは`MID_REVIEW_REQUIRED`以外からのACK
- one replica未完了/INVALID、same-replica、wrong-run/replay、自称match、comparison mismatch/receipt欠落のままcanonical result seal
- `summary.json`/`deterministic_digest.json`をresult authorityとして扱うこと、またはcomparison→result→packet→ACK digest DAGの破壊
- unauthenticated/self-issued/local/stale/replayed/non-CAS lane activation receipt
- replica/comparison/canonical rootのwriter・topology・cross-replica write違反
- unleased/self-issued/cross-used/replayed/local comparison coordinatorまたはresult sealer operation
- result sealerによるdirect replica read、自称outcome、receipt input欠落/不一致

## 15. Human decisions requested

人間reviewでは次を判断する。

1. `historical_ai_duplicate_gate_impact_v1`を通常75点gateとは別の一件限定non-promotion laneとして認めるか
2. 初版classを`AI_DETERMINISTIC_DESCENDANT_GATE_REMOVAL_V1`だけに限定するか
3. score 46/`BLOCKED_SCORE`/credit 0を全artifactへ保持するか
4. D0/D1 exact one-changeと3,746-race final-payoff diagnosticを初期use caseにするか
5. G2 shared catalog/ledger/grant indexを唯一のauthority backendにするか
6. root-governance amendment、G2 core、lane activationを別PR/cutoverにするか
7. result acknowledgementが一切のpromotion権限を持たないことを承認するか

このDraftの承認・mergeは、G2 activation、canonical proposal、prepare/run grant、実装、test実行、
実データ読込、payoff/ROI計算、shadow、production、BUY、注文、通知、merge自動化を承認しない。
