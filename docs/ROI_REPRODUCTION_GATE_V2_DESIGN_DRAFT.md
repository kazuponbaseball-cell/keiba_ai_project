# ROI Reproduction Gate v2 — human-review design draft

作成日: 2026-08-09

状態: `HUMAN_REVIEW_REQUIRED / NON_AUTHORITY / NOT_IMPLEMENTED`

設計base: `b282cafb39435d15098bc409e76b4efaa6690f08`

seed commit: `e218c9dbd372f3e1f58531bcc71bbe9723652ed7`

想定gate kind: `roi_reproduction_audit_v2`

> この文書と同時作成するdesign JSONはレビュー資料であり、Research OSの
> policy、schema、proposal scope、queue、registry event、承認tokenではない。
> このDraftだけではコード準備、synthetic test、manifest取得、実データ読取、
> 学習、historical replay、outer OOS、ROI計算を一切許可しない。

Machine-readable design:
`research/drafts/ROI_REPRODUCTION_GATE_V2_CONTRACT_MAP.design.json`

Human review checklist:
`research/drafts/ROI_REPRODUCTION_GATE_V2_HUMAN_REVIEW_CHECKLIST.md`

## 1. Outcome

`roi_reproduction_audit_v2`は、既存モデルをclean checkoutから再現できるかだけを
判定する、非promotion・非strategyのversioned gateとする。

名称にROIを含むが、このgate自身はROIを計算しない。後続の回収率比較で必要になる
M0 counterfactualを再現可能にするための前提gateであり、price/ROI評価は新ID・75点以上・
別approvalを要する将来の`roi_prospective_model_validation_v2`へ分離する。

このgateが許せる最終結論は次のいずれかだけである。

- `REPRODUCED`: canonical legacy referenceとexact identityを確認した。
- `RECONSTRUCTED_NOT_REPRODUCED`: clean replayは決定的だが、canonical legacy
  referenceが存在しない、またはexact identityを証明できない。
- `REPRODUCTION_FAILED`: scopeと安全契約は満たしたが、2 clean replayの決定性または
  事前登録したequivalence条件を満たさない。
- `REJECTED`: 人間reviewがevidence不十分として受理しない。
- `INVALID`: scope、lineage、probability、approval、安全契約に違反した。

いずれの結論も、モデル採用、shadow、production、BUY、注文、通知、merge、
将来experimentのscore加点を自動承認しない。

## 2. なぜ通常の75点gateと分けるか

M0 clean reproductionは、新しい競馬情報や作用機序を追加しない。seedでの正直な
scoreは23/100であり、通常strategyとしては`BLOCKED_SCORE`である。

一方、後続の性能仮説を正しく比較するには、同一counterfactualの再現が必要である。
そこでv2はscoreを代替するのではなく、次のhard classificationで対象を狭める。

### Reproduction gateに入れられるもの

- 既知の単一legacy recipeの再現だけ
- 1 pipeline、1 reference、1 input universe、1 command plan
- frozen feature、target、fold、L2 grid、temperature grid、tie-break
- historical reference population上のidentity/equivalence検査
- no challenger、no promotion decision、no ROI

### 通常strategy gateへ戻すもの

- 新規feature、target、loss、estimator、calibrator、candidate、ticket、value変更
- 複数variant、追加threshold、結果を見たrecipe変更
- baseline対challenger性能比較
- prospective outer OOS
- price、payoff、ROI、shadow、promotionの評価

分類に迷う場合は通常strategy gateへfail-closeする。人間commentでこの分類や
75点条件を上書きしてはならない。

## 3. Bootstrapを3層に分離する

### G1 — contract compiler only

最初のhuman-owned governance PRは、schema、policy、canonical serializer、
state validator、registry event compiler、synthetic fixturesだけを追加する。

- authority flagsは全てfalse
- manifest providerやexecutorを起動しない
- subprocess、network、real data、training、ROIを扱わない
- gateは自分自身を承認できない
- human reviewとhuman mergeだけがroot of trustを変更できる

### G2 — attester and supervised executor

G1がmainへmergeされた後、別のhuman-owned governance PRで追加する。

- GitHub-backed manifest attestation verifier
- content-addressed immutable data snapshot provider
- structured command executor
- no-workload-network、empty/sanitized environment、bounded timeout、read-only input、
  isolated writable outputのruntime enforcement
- GitHub read-only control planeはverifierだけに限定し、provider/model workloadへ渡さない
- code/environment/provider hashとcapability binding
- Windows/Linuxのlink、junction、ADS、casefold、path escape、TOCTOU検査

PR #36が明記する通り、static AST/evidence compilerはOS sandboxではない。
G2はruntime isolationを別に証明しなければならず、G1成功から推論しない。

### A — M0 reproduction proposal

G1/G2がhuman mergeされた後、初めてhost-assigned proposal Aを作る。

- A0: `APPROVED_TO_PREPARE`後のimplementation preparationとsynthetic tests
- AM: distinct manifest approval後のmetadata-only attestation
- A1: distinct `APPROVED_TO_RUN`後のhistorical reproduction
- AR: distinct review approval後の非promotion classification

この設計DraftではG1、G2、Aのどれも実装・実行しない。

## 4. Lifecycle

```text
DESIGN_DRAFT                    # registry外、権限なし

BLOCKED_ELIGIBILITY             # queue作成なし、新IDで再提案

PROPOSED
  -> APPROVED_TO_PREPARE
  -> PREPARING                  # code + synthetic only
  -> MANIFEST_APPROVAL_REQUIRED
  -> APPROVED_TO_ATTEST_MANIFEST
  -> MANIFEST_ATTESTING         # metadata-only provider
  -> MANIFEST_ATTESTED
  -> RUN_APPROVAL_REQUIRED
  -> APPROVED_TO_RUN
  -> RUNNING                    # historical reproduction only
  -> REVIEW_REQUIRED
  -> ACKNOWLEDGED_REPRODUCTION_RESULT
       -> REPRODUCED
       -> RECONSTRUCTED_NOT_REPRODUCED
       -> REPRODUCTION_FAILED
       -> REJECTED

Any nonterminal state -> INVALID # contract violation; terminal
```

`REPRODUCED`、`RECONSTRUCTED_NOT_REPRODUCED`、`REPRODUCTION_FAILED`、
`REJECTED`、`INVALID`はterminal。
`APPROVED_FOR_SHADOW`とproduction状態はこのgateに存在しない。

## 5. Four distinct GitHub grants

```text
APPROVED_TO_PREPARE <proposal_scope_digest>
APPROVED_TO_ATTEST_MANIFEST <manifest_scope_digest>
APPROVED_TO_RUN <run_scope_digest>
ACKNOWLEDGED_REPRODUCTION_RESULT <review_digest>
```

4つのcomment IDはregistry全体で未使用かつ相互に異なる必要がある。全grantで
repository、base branch、verified current-main、compare URL/status、merge-base、
base-commit `APPROVERS.json` blob/content hash、Issue/comment identity、author
login/type、body/body SHA-256、keyword/digest、created_at、updated_atを保存する。

再検証点:

- `PREPARING`直前: prepare
- manifest attestation直前: prepare + manifest
- run-scope freeze直前: prepare + manifest
- `APPROVED_TO_RUN`直前: prepare + manifest
- `RUNNING`直前: prepare + manifest + run
- terminal classification直前: prepare + manifest + run + result acknowledgement

comment欠落、編集、削除、再利用、unauthorized author、GitHub確認不能、main
ancestry不一致はfail-closeする。任意の`--actor`、`--human-approved`、local
allowlist、branch上のallowlistは拒否する。

## 6. Merged-registry serialization

registryは全schema versionで1つのcomment-ID namespaceを共有する。

- transition前にverified current-main上の`research/REGISTRY.jsonl` blob bytesとlocal
  committed Git blob bytesをbyte-exact一致させる。worktreeのnewline変換をauthorityにしない。
- registry pathはsymlink/junctionでないことを確認する。
- main refをappend直前に再取得し、変化していれば失敗する。
- appendはlock + compare-and-swapで行う。
- local appendはpending evidenceであり、authorityではない。
- exact event bytesがhuman mergeされ、branchをnew mainへrefreshした後だけ、別の
  verifierが次phaseのone-shot leaseを検討できる。

queue、registry event、scope、approval commentのいずれもexecution tokenではない。
verifierはGitHub current main、registry全bytes、全grant、scope、capability、command、
execution commitを再検証し、期限・replica ID・消費状態を持つone-shot leaseだけを
supervised executorへ渡す。信頼できるdurable lease ledgerがない限り、v2は
no-authorityのままfail-closeする。

これにより、同じmainから作られた2 worktreeのgrant再利用、過去eventの削除・再構築、
terminal historyの巻き戻しを防ぐ。

## 7. Eligibility contract

全項目hard checkとし、1つでもfalseなら`BLOCKED_ELIGIBILITY`でqueue/eventを作らない。

1. `purpose=reproduction_only`
2. `gate_kind=roi_reproduction_audit_v2`
3. proposalはexactly one legacy reference familyと有限のpredeclared identity domainを固定する
4. challenger、prospective outer、ROI、candidate/value/ticket policyを含まない
5. feature/target/fold/grid/temperature/tie-breakがreferenceから固定される
6. variant count 1、threshold search count 0
7. L2/temperature gridはfixed internal selectionで、outer-derived searchではない
8. upstream scoreのOOF/as-of/price-blind lineageがhash-bound
9. current/final/historical odds、人気、payoff、ROI派生列がmodel inputにない
10. full `C(n,3)`、runner manifest、finite/range/duplicate、mass契約を検査する
11. dirty/untracked code/config/environmentを実行しない
12. referenceを見た後のtolerance、command、recipe変更を禁止する
13. no production/BUY/order/notification/credential/network side effect
14. successful resultがpromotion、shadow、strategy score権限を持たない

proposal-entry eligibilityとrun eligibilityは分ける。proposalでは、未読referenceを
推測で決めないため、`core|full`、`M0_raw|M0_temperature_scaled`、parity scopeの有限domainを
固定してよい。manifest attestationはそのdomain外を選べず、metadata evidenceからexactly
one identityを確定する。`MANIFEST_ATTESTED`後のhard gateで、recipe、reference、input
universeが各1つに確定しない限りrun scopeと`APPROVED_TO_RUN`を生成しない。

## 8. Canonical object set

v2は次のobjectを別digestで固定する。

| Object | Purpose |
|---|---|
| proposal | reproduction eligibilityとA0 scope |
| manifest scope | metadata providerが読めるlogical source IDsとexact command |
| manifest attestation | content/source/lineage hashes。raw値・absolute pathなし |
| run scope | exact execution commit、attestation、recipe、environment、commands |
| result | deterministic/equivalence evidence。proposal/runを上書きしない |
| review | proposed terminal classificationとreview digest |
| queue/event v4 | lifecycle evidence。legacy v2、infra v3と明示dispatch |

Canonical serializationはUTF-8、sorted object keys、compact separators、Unicode保持、
NaN/Infinity禁止。set-like listはunique + Unicode sort、command listはorder保持。
全objectへpolicy SHA-256とschema SHA-256を保存する。

## 9. Manifest attestation boundary

run scopeは実データhashを必要とするため、prepareとrunの間にmetadata-only grantを置く。

Manifest scopeで固定するもの:

- repository/base/current-main evidence
- provider kind/version、execution commit、code/environment hash
- opaque logical source IDs。caller指定absolute pathは禁止
- expected object count、allowed metadata fields、maximum bytes/runtime
- output schemaとcanonical digest rule
- zero network/credentials/raw rows/path/secret return
- exact read-only command template

Provider outputで許可するもの:

- logical object ID
- content SHA-256、byte size、schema fingerprint、row count
- source/event/received/data-as-of times
- upstream object/recipe lineage hashes
- attester/provider identityとcreated_at
- snapshot IDとoutput digest

禁止するもの:

- raw row/sample/value
- local absolute path、username、secret、credential
- arbitrary file glob、caller-selected root、symlink/junction/ADS
- model deserializationやtraining

TOCTOUを避けるため、runはattestation時に作ったcontent-addressed immutable snapshotを
digestで開く。単に「hash後に元fileを再度読む」方式は採用しない。snapshot機構とOS
isolationをG2で証明できなければ、このgateは実装せず`BLOCKED_CAPABILITY`とする。

## 10. Run scope and structured command

Run scopeは最低限、次を含む。

- complete proposal object + digest
- complete manifest attestation + digest
- exact execution commit and base ancestry evidence
- recipe/config/fold/runner/feature/target hashes
- canonical legacy reference class/path-independent digest
- dependency lock、Python executable/hash/version、numpy/pandas versions
- OS/platform/numeric environment
- exact seed semantics
- one code-owned command template ID + ordered typed arguments
- repository-root cwd、empty/sanitized env、timeout
- read-only snapshot IDs、single writable output root
- exact expected output set and canonicalization

free-form shell、PowerShell/cmd/bash、`python -c`、pipe、redirect、URL、package install、
caller-selected executable/pathは拒否する。

## 11. M0-specific frozen blockers

M0 seedから次は未確定のため、proposalでは有限の選択domainとして固定し、manifest
attestation後・run scope作成前にexactly one identityへ決定する。

- legacy `--mode`: static default `core`か`full`か
- canonical probability stage: `M0_raw`か`M0_temperature_scaled`か
- B0/B1/B2/comparison outputsをparity scopeへ含めるか
- canonical full per-set reference digestの有無
- STATE hit-rate値のprovenance/unit mapping
- STATE NLLがfold meanかrace-weightedか
- upstream `ai_score`/`ai_rank`のOOF/as-of/price-blind lineage
- legacy input artifacts、dependency environment、runner universeの正本

これらを推測で埋めない。canonical reference digestがない場合、成功上限は
`RECONSTRUCTED_NOT_REPRODUCED`である。

## 12. Reproduction decision rule

### Clean replay determinism

同一run scope、同一environment、別の2 clean checkoutで次をexact一致させる。

- discrete manifestsとrace/runner/triple universe
- per-fold model state: mean/std/weights/L2/temperature
- full per-set probabilitiesのcanonical bytes/digest
- canonical metrics/result object

### Legacy equivalence

- canonical legacy digestが存在し、exact artifact identityを満たす:
  `REPRODUCED`
- canonical digestがなく、2 clean replayがexact一致し、事前固定numeric toleranceを
  満たす: `RECONSTRUCTED_NOT_REPRODUCED`
- scopeと安全契約はpassしたがdeterminismまたはtolerance不一致:
  `REPRODUCTION_FAILED`
- 人間reviewがevidence不十分として受理しない: `REJECTED`
- scope/lineage/mass/safety違反: `INVALID`

validated resultが計算したoutcomeをhuman acknowledgementが上位へ変更してはならない。
`REPRODUCED`、`RECONSTRUCTED_NOT_REPRODUCED`、`REPRODUCTION_FAILED`は同じstatusか
`REJECTED`だけを選べ、computed `INVALID`は`INVALID`のままとする。

legacy numeric toleranceは、既にSTATEへ記録された値と静的script以外の未読reference
artifactをproviderが公開する前にcommitする。

## 13. Probability and data contracts

- canonical full per-set columns:
  `race_id, horse_id_1, horse_id_2, horse_id_3, runner_count, top3_probability`
- frozen starter-ID manifestを確率artifactと独立に持つ
- all unordered `C(n,3)` exactly once
- Top3 set mass `1 ± 1e-10`
- derived wide mass `3 ± 1e-10`
- finite、range、duplicate、runner countをfail-close検査
- label/取消/同着/失格/降着は別validatorで検査
- historical resultはpartition/as-of固定後のtarget builderだけが利用
- price/payoff/ROIはこのreproduction gateでは一切読まない

## 14. Capability matrix

| Phase | Synthetic prep | Metadata read | Historical training/replay | Prospective outer | ROI | Shadow/promotion |
|---|---:|---:|---:|---:|---:|---:|
| Design/G1 | false | false | false | false | false | false |
| A0 PREPARING | true | false | false | false | false | false |
| MANIFEST_ATTESTING | false | true | false | false | false | false |
| A1 RUNNING | false | false | true | false | false | false |
| Review/terminal | false | false | false | false | false | false |

全phaseで次はfalse/0:

- automatic execution/approval
- workload external API/network（GitHub read-only control planeはverifierだけに分離）
- credential/purchase path access
- production/candidate/value policy change
- notification/order side effect
- formal BUY、send order、stake
- merge authority

上表の`Metadata read`はproviderがhistorical sourceへアクセスする実処理を含むため、
machine-readable capabilityでは`real_data_execution=true`かつ
`supervised_manifest_hash_read=true`と明示する。一方、model access、raw-row output、
training、replay、price、payoff、ROIはfalseのままとする。

全phaseは同じexact capability field集合を持つ。field欠落、未知field、別名、
boolean以外の値を拒否し、`mode`と`effective_after`はcapability booleanから分離する。
`real_data_execution=true`だけで権限を導出してはならない。

## 15. One-shot lease and supervised executor

各leaseは共通fieldとしてlease ID、experiment ID、gate/execution kind、phase、
capability/command digest、execution commit、verified current-main、verifier/executor/
policy/schema digest、GitHub evidence digest、issued/expires/consumed time、human supervisor、
retry budget、安全定数を固定する。未来にしか存在しないdigestをnullで先取りしない。

phase別binding:

- preparation: proposal digestだけをrequiredとし、manifest/run/replica fieldを禁止する。
- manifest attestation: proposal + manifest-scope digestをrequiredとし、まだ存在しない
  attestation output/run/replica fieldを禁止する。
- historical reproduction: proposal + manifest scope + attestation output + run digest +
  replica IDを全てrequiredとする。

M0 replayは`replica_count=2`、replica IDs=`clean_a,clean_b`をrun scopeへ固定し、
各replicaが別leaseを一度だけ消費する。同じlease、run digest、replica IDの再利用、
crash後の自動retry、共有mutable cacheを拒否する。retry budget外は新run scopeと
新しいrun grantを必要とする。

各replica直前にGitHub main不変、current-main registry exact bytes、全grant、commit
ancestry、attestation、clean worktree、exact cwd/env/interpreter/argv/timeout、read-only
input mount、isolated writable output、network/credential/production isolationを再検証する。

## 16. Compatibility and PR #36

2026-08-09時点のDraft PR #36は`infrastructure_safety_v1`のevidence compilerで、
全authorityをfalseにする。v2 designはPR #36を実行権限として利用しない。

- mainは`b282caf...`で、PR #36は未merge
- legacy ROI proposal/run/event schema v2とdigestを変更しない
- PR #36のinfra queue/event schema v3を変更しない
- v2はqueue/event schema v4として明示dispatchする
- comment ID namespaceは全versionで共有する
- PR #36がmergeされた場合はexact-main registry/CAS/ancestry helperを再利用できる
- mergeされない場合も同等invariantをv2側で独立に要求する

将来のG1 implementationは、PR #36と同じroot-of-trust fileを多数変更するため、PR #36の
human decision後にnew mainへrebaseしてから開始することを推奨する。Draft同士を暗黙merge
したり、PR #36の未merge branchをapproval rootにしない。

## 17. Future root-of-trust implementation paths

候補path。人間reviewで確定するまでexpected changed pathsではない。

```text
AGENTS.md
.github/workflows/research-os.yml
research/CHARTER.md
research/DECISIONS.md
research/STATE.yaml
research/HYPOTHESIS_SCORECARD.yaml
research/APPROVERS.json
research/REGISTRY.jsonl
research/ROI_REPRODUCTION_GATE_V2.json
research/INFRASTRUCTURE_GATE.json
research/schemas/roi_reproduction_proposal_v2.schema.json
research/schemas/roi_reproduction_manifest_scope_v1.schema.json
research/schemas/roi_reproduction_manifest_attestation_v1.schema.json
research/schemas/roi_reproduction_run_v2.schema.json
research/schemas/roi_reproduction_result_v1.schema.json
research/schemas/roi_reproduction_review_v1.schema.json
research/schemas/roi_reproduction_queue_v4.schema.json
research/schemas/roi_reproduction_execution_lease_v1.schema.json
research/schemas/research_registry_event_v4.schema.json
scripts/research/github_approval.py
scripts/research/roi_reproduction_contract_v2.py
scripts/research/roi_reproduction_authority_verifier.py
scripts/research/prepare_roi_reproduction_run_scope_v2.py
scripts/research/attest_roi_reproduction_manifest_v1.py
scripts/research/roi_reproduction_supervised_executor_v2.py
scripts/research/update_registry.py
tests/research/test_roi_reproduction_contract_v2.py
tests/research/test_roi_reproduction_lifecycle_v2.py
tests/research/test_registry_jsonl.py
```

これらはgovernance rootであり、新gateで自己承認しない。G1/G2ともnormal Draft governance
PR、human review、human mergeを必要とする。

## 18. Threat model

| Threat | Required defense |
|---|---|
| score laundering | challenger/ROI/performanceを検出したら通常75点gateへroute |
| reference laundering | local-only referenceでは`REPRODUCED`禁止 |
| approval replay | registry-wide unused IDs + GitHub refetch |
| local ledger rewrite | exact current-main equality + merged-event serialization |
| manifest caller assertion | code-owned provider + dedicated GitHub grant |
| source TOCTOU | content-addressed immutable snapshot |
| dirty code/config | execution-commit blob/type/path exact match |
| command drift | structured template + ordered typed args |
| path escape | root containment + symlink/junction/ADS/casefold checks |
| result peeking | tolerance/recipe committed before reference exposure |
| market leakage | odds/popularity/payoff/ROI hard firewall |
| incomplete Top3 universe | independent starter manifest + all `C(n,3)` check |
| nondeterministic output | canonical encoding + two-clean-checkout exact digest |
| provider exfiltration | no network/credential/raw/path output + exact schema |
| executor side effect | runtime sandbox + read-only inputs + bounded output |
| duplicate or crash replay | durable one-shot lease consumption + no automatic retry |

## 19. Human decisions required

1. reproduction-only専用eligibilityをstrategy scoreとは別に認めるか
2. manifest用の第3grantとreview用の第4grantを追加するか
3. merged current-main eventを再検証した別verifierのone-shot leaseだけをauthorityにするか
4. content-addressed immutable snapshotをrunの必須条件にするか
5. canonical referenceなしの上限を`RECONSTRUCTED_NOT_REPRODUCED`とするか
6. G1とG2を別governance PRへ分離するか
7. implementation開始をPR #36のhuman decision後まで待つか
8. M0 referenceの`core/full`、`raw/temperature_scaled`、parity output scopeを何にするか
9. future root-of-trust path listを承認するか
10. scientific negativeを`REPRODUCTION_FAILED`、evidence拒否を`REJECTED`へ分けるか
11. manifest attester/providerとrun approverのrole separationを必須にするか
12. G2初版をhuman-invoked supervised executorに限定するか、将来Codex dispatchを
    one-shot leaseの別capabilityとして追加reviewするか

推奨current decision:

```text
DESIGN_REVIEWABLE
IMPLEMENTATION_BLOCKED
EXECUTION_FORBIDDEN
```

## 20. Safety declaration for this Draft

次はcommitted 3 artifact自体のcapabilityを表す。設計時のopen Issue/PR確認では
GitHub read-only repository inspectionを行ったが、外部model API、model/data、
training、ROIは利用していない。

- `actual_codex_dispatch=false`
- `automatic_github_approval=false`
- `credential_access=false`
- `github_read_only_repository_inspection=true`
- `external_model_api_calls=false`
- `artifact_runtime_network_calls=0`
- `real_data_execution=false`
- `historical_training_execution=false`
- `prospective_outer_execution=false`
- `roi_calculation=false`
- `candidate_policy_change=false`
- `production_change=false`
- `notification_side_effects=false`
- `order_side_effects=false`
- `purchase_path_access=false`
- `formal_buy=false`
- `send_order=false`
- `stake=0`
- `merge=false`
