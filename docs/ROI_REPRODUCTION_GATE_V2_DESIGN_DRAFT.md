# ROI Reproduction Gate v2 — human-review design draft

作成日: 2026-08-09

状態: `HUMAN_REVIEW_REQUIRED / NON_AUTHORITY / NOT_IMPLEMENTED`

設計origin base: `b282cafb39435d15098bc409e76b4efaa6690f08`

操作簡略化review base: `fbbebc804c7a2393aff26a6de9ad7c55caa5bc92`

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

現行G1 implementationのboundaryは`CONTRACT_COMPILER_ONLY_NO_AUTHORITY`である。G1は
pure compiler、policy、schema、synthetic-only governance testを実装するが、provider、writer、
authority verifier、executorを実装しない。artifact contractは
`execution_kind=historical_reproduction_v2`をbindする一方、現在のruntime execution kindはnone、
全authorityはfalse/0、`EXECUTION_FORBIDDEN`である。G2は未実装であり、現行legacy v2と
infra v3 writer semantics、approval keyword、digestを変更しない。schema v4 dispatchと
prepare/run/resultの3 action workflowも現在activationされていない。

G1のJSON Schema単独、またはtrusted policy/schema digestを渡さないnormalizer呼出しは
`NON_AUTHORITY_FIXTURE_VALIDATION`に限る。artifactを有効と判定するには、code-owned Python
normalizerを必須とし、G2 verifierがcurrent-main上のpolicy/schema bytes、catalog release、全manifest
contract、expected output/numeric reference、execution contextを再取得してhash照合しなければならない。
Schemaだけのvalidation、caller提供のdigest/path/ACTIVE表明、G1の`structural`表示をgrant・lease・runへ
変換してはならない。

G2 activation前の未実装blockerは、有限catalog source/provider allowlist、9種のcatalog contractと
run manifestのconformance検証、repository-root cwd・sanitized env・timeout・read-only snapshot・single
output rootの実効強制、exact policy/schema digest context、publisher/attester、durable ledger、one-shot
lease、supervised executorである。いずれかが未実装なら`EXECUTION_FORBIDDEN`を維持する。
現G1の`catalog publication scope v1`はexpected object count、canonical output rule、有限provider/
command/budget domainをまだ完全には表現しないため、そのままG2 authorityへ昇格できない。G2は
code-owned finite semanticsを別versionのscope/schema/policyへfreezeし、人間review・main mergeする。
`NEW_KIND_VERSION_AND_PATH_ONLY`のため、現v1 schemaをin-place拡張してactivationしてはならない。

### G2 — attester and supervised executor

G1がmainへmergeされた後、別のhuman-owned governance PRで追加する。

- GitHub-backed reference catalog release verifier
- content-addressed immutable data snapshot provider
- reusableなreference metadata catalog publisher
- serializable transaction、global grant reservation、hash chain、terminalityを持つ
  durable runtime ledger
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
- A1: distinct `APPROVED_TO_RUN`後のhistorical reproduction
- AR: distinct result acknowledgement後の非promotion classification

proposal Aは、G2が事前に公開したimmutable reference catalog releaseからexactly one
identityを解決できる場合だけ作る。実験ごとのmanifest取得・manifest承認は行わない。
catalogにentryがない、複数候補が残る、snapshot/hash/lineageが不足する場合は
`BLOCKED_CATALOG`として、catalog releaseを別のmaintenance reviewで更新する。

### Routine operator interaction budget

通常の1 experimentで人間に求める操作は次の3回を上限とする。

1. `APPROVED_TO_PREPARE <proposal_scope_digest>`
2. `APPROVED_TO_RUN <run_scope_digest>`
3. `ACKNOWLEDGED_REPRODUCTION_RESULT <review_digest>`

1の承認はsupervised workflowの開始または再開も兼ね、別のexperiment開始操作を要求しない。
2の承認は同じworkflowをrunへ再開する操作も兼ねる。別の「実行開始」clickや
`clean_a`/`clean_b`ごとの確認は要求しない。承認コメント以外を含む人間のvisible actionも
合計3回を上限とする。これは自動承認や無人scheduleを許す意味ではなく、scope外変更、
retry、catalog不一致では再開せず停止する。

通常experimentのstate transitionにPRのReady化、merge、rebase、branch refreshを要求しない。
これらは全て0回とする。Ready/mergeは
schema、policy、executor、durable ledger、catalog publisherなどgovernance rootを
変更するときだけ必要である。reference catalog releaseの更新もexperiment本体とは
別のmaintenance actionであり、同じcatalog releaseはdigest不変の間、複数experimentから
再利用できる。catalogやdurable ledgerが利用不能なら、per-transition mergeへfallbackせず
fail-closeする。

PR #37の設計Draft自体はG1、G2、Aを実装・実行しなかった。後続の現G1変更は
contract compilerだけを実装し、G2とAは引き続き未実装・実行禁止である。

## 4. Lifecycle

```text
DESIGN_DRAFT                    # registry外、権限なし

BLOCKED_ELIGIBILITY             # queue作成なし、新IDで再提案
BLOCKED_CATALOG                 # reusable catalog不足、queue作成なし

PROPOSED
  -> APPROVED_TO_PREPARE
  -> PREPARING                  # code + synthetic only
  -> CATALOG_BOUND              # pre-bound entryの機械的再検証のみ
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

## 5. Three routine GitHub grants

```text
APPROVED_TO_PREPARE <proposal_scope_digest>
APPROVED_TO_RUN <run_scope_digest>
ACKNOWLEDGED_REPRODUCTION_RESULT <review_digest>
```

3つのcomment IDはglobal grant index全体で未使用かつ相互に異なる必要がある。全grantで
repository、base branch、verified current-main、compare URL/status、merge-base、
base-commit `APPROVERS.json` blob/content hash、Issue/comment identity、author
login/type、body/body SHA-256、keyword/digest、created_at、updated_atを保存する。

再検証点:

- `PREPARING`直前: prepare
- run-scope freeze直前: prepare + reference catalog release/entry + durable ledger head
- `APPROVED_TO_RUN`直前: prepare + catalog binding
- `RUNNING`直前: prepare + run + catalog binding
- result manifest sealing直前: prepare + run + catalog publication grant + latest `ACTIVE` release status
- terminal classification直前: prepare + run + result acknowledgement + catalog publication grant +
  latest `ACTIVE` release status

comment欠落、編集、削除、再利用、unauthorized author、GitHub確認不能、main
ancestry不一致はfail-closeする。任意の`--actor`、`--human-approved`、local
allowlist、branch上のallowlistは拒否する。

## 6. Durable runtime ledger; no per-transition merge

`research/REGISTRY.jsonl`を毎transitionでmainへmergeする方式は、通常experimentの
serialization pointにしない。v2のlive stateはG2で人間reviewされたdurable runtime
ledgerだけが保持し、次を1つのserializable transactionで行う。

- expected previous event/sequence/head digestのcompare-and-swap
- new event appendとhash-chain更新
- approval comment IDのglobal single-use reservation
- terminality、run/replica、lease consumptionの一意性検査
- immutable receiptの生成

writer/verifierの認証、global/experiment sequence、idempotency key、lease発行・state append・
consumeのatomicity、split-brain/restore rollback拒否、backup/recoveryも必須とする。

G2 activation時にcurrent-main `research/REGISTRY.jsonl`のfull byte-exact event chain、全comment
ID、各experimentのlatest sequence/head/statusをterminal・nonterminalともhash-bound importする。
既存nonterminal headを欠落・fork・再初期化できない。activation後の新規grantはlegacy v2、
infra v3、ROI v4を問わず同じglobal grant indexへ原子的にreserveする。global indexへreserve
できないlegacy writerは新規grantをfail-closeし、read/監査互換だけを維持する。

cutoverはactivation epoch、exact current-main SHA、registry blob SHA/content SHA-256を固定し、
old writerのnew-grant発行を先にfreezeする。import後・active-backend切替直前にmain refとsource
registryを再比較し、1 event/grantでも増減していればabort/reimportする。active backend pointerは
原子的に切り替え、activation receiptを保存する。

repositoryの`REGISTRY.jsonl`はperiodic audit export/checkpointとして残せるが、そのPR mergeを
次transitionや次experimentの前提にしない。checkpoint遅延時もdurable ledgerがsole live
serialization pointであり、branch-local/local-file/alternate registryへfallbackしない。

queue、event、scope、approval commentのいずれもexecution tokenではない。verifierは各phase
直前にGitHub current mainのroot-of-trust、ancestry、全grant、durable ledger head、scope、
catalog binding、capability、command、execution commitを再検証し、期限・replica ID・消費状態を
持つone-shot leaseだけをsupervised executorへ渡す。backend、atomicity、durability、backup/
restore、audit exportをG2で証明できない限りv2はno-authorityのままfail-closeする。

これにより、人間に毎回mergeを求めず、同じmainから作られた2 worktreeのgrant再利用、
過去eventの削除・再構築、terminal historyの巻き戻しを防ぐ。

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

proposal-entry eligibilityとrun eligibilityは分ける。proposal compilerは、未読referenceを
推測せず、事前公開済みreference catalog releaseのmetadataだけを使う。`core|full`、
`M0_raw|M0_temperature_scaled`、parity scopeの有限domainからexactly one identityを解決し、
catalog release/entry digestをproposalへ固定する。0件または複数件なら`BLOCKED_CATALOG`とし、
proposal、queue、grantを作らない。run hard gateでも同じcatalog release/entry、recipe、reference、
input universeが各1つに確定していることを再検証する。`PROPOSED`作成後の再検証失敗は
pre-registryの`BLOCKED_CATALOG`へ戻さず、registered terminal `INVALID`とする。

## 8. Canonical object set

v2は次のobjectを別digestで固定する。

| Object | Purpose |
|---|---|
| reference catalog release | reusableなcontent/source/lineage/snapshot metadata。raw値・absolute pathなし |
| proposal | reproduction eligibility、exact catalog entry binding、A0 scope |
| run scope | exact execution commit、catalog entry、recipe、environment、commands |
| result | deterministic/equivalence evidence。proposal/runを上書きしない |
| review | proposed terminal classificationとreview digest |
| durable event/lease receipt v4 | live lifecycle evidence。legacy v2、infra v3と明示dispatch |
| repository registry checkpoint | durable ledgerの監査export。live authorityではない |

Canonical serializationはUTF-8、sorted object keys、compact separators、Unicode保持、
NaN/Infinity禁止。set-like listはunique + Unicode sort、command listはorder保持。
全top-level canonical artifact/receiptへpolicy SHA-256とschema SHA-256を保存する。
embedded catalog entry/refはenclosing release/proposal/runのpolicy/schema bindingを継承し、
自身のstrict embedded schemaで検証する。

## 9. Reusable reference catalog boundary

run scopeが必要とする実データhashは、experiment開始前にG2のcatalog maintenance flowで
公開したimmutable reference catalog releaseから取得する。maintenance flowは次の順序を
固定する。

1. raw dataを読まずにcatalog publication scopeを作り、provider/code/environment、opaque
   source IDs、metadata output schema、budget、commandをdigestへ固定する。
2. 人間が`APPROVED_TO_PUBLISH_REFERENCE_CATALOG <catalog_publication_scope_digest>`を承認する。
3. verifierがscope grantを再検証し、one-shot provider leaseでmetadata hash readを行う。
4. releaseはpublication-scope digest、approval-evidence digest、provider receipt、entry digestを
   bindしてimmutable publishする。

output後にしか分からないrelease digestをread前のgrant対象にしてはならない。同じreleaseを使う
各experimentではcatalog publicationを再承認しない。

catalog publication grantもroutine 3 grantと同じGitHub evidence contractを継承する。
base-commit `APPROVERS.json`、actor type `User`、global unused comment ID、body/digest/timestamp、
edit/delete/reuse fail-closeを要求し、provider lease直前にcommentを再取得する。Codex、bot、
automationはcatalogを承認できない。

published releaseを後続experimentへbindするときも、catalog publication grant evidenceを
proposal binding、run-scope freeze、`RUNNING`直前、result sealing、terminal classificationで
GitHubから再取得して検証する。commentの
編集・削除・evidence不一致またはappend-only revocationが見つかった場合、proposal作成前は
`BLOCKED_CATALOG`、`PROPOSED`作成後は実行前後を問わずterminal `INVALID`とし、release/entry
digestの一致だけで続行しない。

maintenance lifecycleは`CATALOG_PUBLICATION_SCOPE_PROPOSED ->
APPROVED_TO_PUBLISH_REFERENCE_CATALOG -> CATALOG_PUBLISHING -> CATALOG_PUBLISHED |
CATALOG_PUBLICATION_FAILED | INVALID`とし、後3つをterminalにする。provider leaseの
retry budgetは0、`(publication_scope_digest, approval_comment_id)`ごとにlease/publicationを
1回だけreserveする。crash/failure後のretryは新しいscope digestと未使用grantを必要とする。
この履歴はexperiment向けregistry event v4へ混ぜず、catalog publication専用event v1で
scope ID/digest、global/subject sequence、previous event/status、grant reservation、provider
lease issue/consume receipt、publication receipt、release ID/digest、policy/schema、安全定数を
strict bindする。遷移はscope proposed→approved→publishing→published/failedの順序だけを許し、
各nonterminalから`INVALID`へのfail-closeだけを追加で許す。grant/lease/publication receipt条件を
満たさないskipを拒否する。専用eventなしでcatalog stateやterminalityを台帳へ記録してはならない。

publication lifecycleの`CATALOG_PUBLISHED`を後から書き換えず、release利用可否は別のappend-only
catalog release status event v1で管理する。publish時にexactly one `ACTIVE` eventを作り、以後は
`ACTIVE -> REVOKED`だけを許す。release ID/digest、global/release-status sequence、previous event、
reason/evidence、effective time、signer、policy/schema、安全定数をbindする。proposal binding、
run-scope freeze、`RUNNING`直前はlatest statusが`ACTIVE`であることも検証する。

Catalog publication scopeで固定するもの:

- repository/base/current-main evidence
- provider kind/version、execution commit、code/environment hash
- opaque logical source IDs。caller指定absolute pathは禁止
- expected object count、allowed metadata fields、maximum bytes/runtime
- output schemaとcanonical digest rule
- zero network/credentials/raw rows/path/secret return
- exact read-only command template

Catalog entry payloadで許可するもの:

- entry ID、logical object ID
- content SHA-256、byte size、schema fingerprint、row count
- source/event/received/data-as-of times
- upstream object/recipe lineage hashes
- resolved model/input-universe identity、snapshot ID
- `formal_buy=false`、`send_order=false`、`stake=0`

entry自身へself digestを入れない。release indexがentry IDとexternal canonical entry digest、
entry-set digest、provider/attester/created-at、content-addressed publication receipt digestを持つ。entry refは
release ID/external release digestとentry ID/external entry digestを持つ。release自身のdigestも
release payloadへ埋めず、proposal/runのref側で固定する。

禁止するもの:

- raw row/sample/value
- local absolute path、username、secret、credential
- arbitrary file glob、caller-selected root、symlink/junction/ADS
- model deserializationやtraining

TOCTOUを避けるため、runはcatalog publication時に作ったcontent-addressed immutable snapshotを
digestで開く。単に「hash後に元fileを再度読む」方式は採用しない。experiment compilerは
catalog metadata以外のsourceを読めず、caller assertionやad-hoc pathでentryを補完できない。
snapshot機構、catalog release署名/receipt、OS isolationをG2で証明できなければ、このgateは
実装せず`BLOCKED_CAPABILITY`とする。

## 10. Run scope and structured command

Run scopeは最低限、次を含む。

- complete proposal object + digest
- exact reference catalog release/entry refs + digests
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

M0 seedから次は未確定のため、reference catalog releaseで有限の選択domainから
exactly one identityへ決定した後にだけproposalを作る。

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
| Catalog release maintenance | false | true | false | false | false | false |
| A0 PREPARING | true | false | false | false | false | false |
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

上表の`Metadata read`はcatalog maintenance providerがhistorical sourceへアクセスする
実処理を含むため、machine-readable capabilityでは`real_data_execution=true`かつ
`supervised_manifest_hash_read=true`と明示する。一方、通常experimentのproposal/A0では
catalog metadataだけを読み、underlying sourceへアクセスしない。catalog maintenanceでも
model access、raw-row output、training、replay、price、payoff、ROIはfalseのままとする。

全phaseは同じexact capability field集合を持つ。field欠落、未知field、別名、
boolean以外の値を拒否し、`mode`と`effective_after`はcapability booleanから分離する。
`real_data_execution=true`だけで権限を導出してはならない。

## 15. One-shot lease and supervised executor

experiment execution leaseは共通fieldとしてlease ID、experiment ID、gate/execution kind、phase、
capability/command digest、execution commit、verified current-main、verifier/executor/
policy/schema digest、durable ledger backend/head/grant-reservation receipt digest、GitHub
evidence digest、issued/expires time、human supervisor、retry budget、安全定数を
固定する。未来にしか存在しないdigestをnullで先取りしない。

leaseは発行後に書き換えない。issue transactionはprevious/new ledger head、transaction/
idempotency key、authorizing lifecycle event、grant reservation、lease ID/digestをimmutable issue
receiptへ保存する。必要なstate変更が同時にある場合だけ新lifecycle eventを同じtransactionでappendし、
既存`RUNNING`内のreplica lease発行などstate不変の場合は既存のauthorizing lifecycle-event digestと
新しいoperation recordをbindする。
実行直前のconsume transactionは別のimmutable consumption receiptへprevious/new global headと
previous/new experiment operation head、lease/run/replica digest、`consumed_at`を保存する。catalog providerは
run/replica receiptを流用せず、
publication scope、provider lease/identity、dispatch reservation、previous/new headを持つ専用の
pre-dispatch consume receiptを使う。まだ存在しないentry/snapshot digestをそこへ先取りしない。
実行後のcatalog publication receiptがscope/approval/provider lease/code/environment/command、
entry-set/snapshot-set、専用consume-receipt digestを固定する。
全receiptはauthenticated writer/signer evidenceと安全定数を持ち、strict canonical digestでbindする。

lease issue/consumeのoperation sequenceはlifecycle event sequenceと別である。replicaごとのconsumeや
catalog providerのissue/consumeはoperation headだけを進め、`RUNNING -> RUNNING`や
`CATALOG_PUBLISHING -> CATALOG_PUBLISHING`という未定義self-transitionを作らない。state変更時だけ
registry/catalog lifecycle eventをappendし、そのevent sequenceとprevious-event chainを検証する。

operation headの正本はreceipt自身ではなく、別のimmutable lease operation record v1とする。
recordはsubject kind/ID、operation kind/sequence、previous operation digest、authorizing lifecycle-event
digest、lease/capability、kind別scope/grant/replica/dispatch binding、writer、policy/schema、安全定数を
持ち、自分自身のdigestは含めない。`new_operation_digest = SHA256(canonical operation record)`を
計算してからreceiptへattestし、expected previous operation headのCASとglobal-head updateを同じ
transactionで行う。receiptを自己参照してoperation digestを作る実装は禁止する。
preparation consumptionはproposal digestとdispatch reservationだけをbindしてrun/replica fieldを
禁止し、historical run consumptionはrun digestとreplica IDを必須にする。
consumption receiptも同じphase dispatchを行い、common receipt + exact `PREPARATION` bindingでは
proposal digestだけ、exact `HISTORICAL_RUN_REPLICA` bindingではrun digest + replica IDだけを許す。

phase別binding:

- preparation: proposal digestとcatalog entry refをrequiredとし、run/replica fieldを禁止する。
- historical reproduction: proposal + catalog release/entry + run digest + replica IDを
  全てrequiredとする。

catalog publicationはexperiment lease/receiptへ混ぜない。別のprovider leaseがpublication scope ID/digest、
catalog/gate kind、provider/code/environment/command/capability、current main、durable ledger head、
catalog grant evidence、expiry、安全定数を固定する。`experiment_id`、run scope、replica IDは
禁止し、catalog provider専用のseparate atomic issue/consume receiptを使う。

M0 replayは`replica_count=2`、replica IDs=`clean_a,clean_b`をrun scopeへ固定し、
各replicaが別leaseを一度だけ消費する。同じlease、run digest、replica IDの再利用、
crash後の自動retry、共有mutable cacheを拒否する。retry budget外は新run scopeと
新しいrun grantを必要とする。

各replica直前にGitHub mainのroot-of-trust不変、durable ledger head、全grant、commit
ancestry、catalog binding、clean worktree、exact cwd/env/interpreter/argv/timeout、read-only
input mount、isolated writable output、network/credential/production isolationを再検証する。

## 16. Compatibility and PR #36

PR #36は2026-08-09に人間判断でmainへmergeされ、merge commitは
`fbbebc804c7a2393aff26a6de9ad7c55caa5bc92`である。`infrastructure_safety_v1`は
evidence compilerで全authorityをfalseにする。v2 designはPR #36を実行権限として利用しない。

- 操作簡略化review baseは`fbbebc8...`
- legacy ROI proposal/run/event schema v2とdigestを変更しない
- PR #36のinfra queue/event schema v3を変更しない
- v2はqueue/event schema v4として明示dispatchする
- historical comment IDをdurable global grant indexへimportし、全versionで共有する
- PR #36のGitHub trust、ancestry、strict JSON helperは再利用候補だが、merged-event
  serializationをv2 live authorityとして再利用しない
- v2 durable ledger/catalog/lease providerは別のhuman review済みG2 root-of-trustとする
- merged `research/INFRASTRUCTURE_GATE.json`はread-only compatibility inputであり、
  future changed-path候補に含めずin-place変更しない

G1 compiler implementationはPR #37を含むmerged mainの子孫で行い、containing commitがhuman
mergeされた後だけcode-owned G1 rootとして扱う。branch copyをapproval rootにせず、G2の
governance変更も別にhuman review/mergeする。
G2 activation後の通常experiment transitionはPR mergeを要求しない。

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
research/schemas/roi_reproduction_proposal_v2.schema.json
research/schemas/roi_reproduction_reference_catalog_publication_scope_v1.schema.json
research/schemas/roi_reproduction_catalog_publication_provider_lease_v1.schema.json
research/schemas/roi_reproduction_reference_catalog_release_v1.schema.json
research/schemas/roi_reproduction_reference_catalog_entry_v1.schema.json
research/schemas/roi_reproduction_reference_catalog_entry_ref_v1.schema.json
research/schemas/roi_reproduction_run_v2.schema.json
research/schemas/roi_reproduction_result_v1.schema.json
research/schemas/roi_reproduction_review_v1.schema.json
research/schemas/roi_reproduction_queue_v4.schema.json
research/schemas/roi_reproduction_execution_lease_v1.schema.json
research/schemas/roi_reproduction_lease_operation_record_v1.schema.json
research/schemas/roi_reproduction_durable_ledger_receipt_v1.schema.json
research/schemas/roi_reproduction_catalog_publication_receipt_v1.schema.json
research/schemas/roi_reproduction_catalog_publication_event_v1.schema.json
research/schemas/roi_reproduction_catalog_release_status_event_v1.schema.json
research/schemas/roi_reproduction_registry_event_v4.schema.json
scripts/research/github_approval.py
scripts/research/roi_reproduction_contract_v2.py
scripts/research/roi_reproduction_authority_verifier.py
scripts/research/prepare_roi_reproduction_run_scope_v2.py
scripts/research/publish_roi_reproduction_reference_catalog_v1.py
scripts/research/bind_roi_reproduction_reference_catalog_v1.py
scripts/research/roi_reproduction_durable_ledger.py
scripts/research/export_roi_reproduction_audit_checkpoint_v1.py
scripts/research/roi_reproduction_supervised_executor_v2.py
scripts/research/update_registry.py
research/infra_tests/test_roi_reproduction_contract_v2.py
research/infra_tests/test_roi_reproduction_lifecycle_v2.py
research/infra_tests/test_roi_reproduction_reference_catalog_v1.py
research/infra_tests/test_roi_reproduction_durable_ledger_v1.py
tests/research/test_registry_jsonl.py
```

これらはgovernance rootであり、新gateで自己承認しない。G1/G2ともnormal Draft governance
PR、human review、human mergeを必要とする。

## 18. Threat model

| Threat | Required defense |
|---|---|
| score laundering | challenger/ROI/performanceを検出したら通常75点gateへroute |
| reference laundering | local-only referenceでは`REPRODUCED`禁止 |
| approval replay | durable global grant index + GitHub refetch |
| local ledger rewrite | durable transactional CAS/hash chain; local files never authoritative |
| per-transition merge fatigue | live durable ledger; repository registry is periodic audit checkpoint |
| catalog caller assertion | reusable code-owned catalog release + separately reviewed publication |
| catalog poisoning/revocation | binding-before-proposal + immutable release/entry digest + fail-close revocation policy |
| ledger outage/split brain | no fallback + authenticated serializable backend + restore rollback rejection |
| double dispatch | atomic state/lease issue/consume + idempotency key |
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
2. 通常experimentをprepare/run/resultの3操作、Ready/merge 0回に固定するか
3. durable runtime ledgerとglobal grant indexをsole live serialization pointにするか
4. reusable reference catalog releaseとcontent-addressed immutable snapshotを必須にするか
5. canonical referenceなしの上限を`RECONSTRUCTED_NOT_REPRODUCED`とするか
6. G1とG2を別governance PRへ分離するか
7. M0 referenceの`core/full`、`raw/temperature_scaled`、parity output scopeを何にするか
8. future root-of-trust path listを承認するか
9. scientific negativeを`REPRODUCTION_FAILED`、evidence拒否を`REJECTED`へ分けるか
10. catalog publisher/attesterとrun approverのrole separationを必須にするか
11. G2初版をhuman-invoked supervised executorに限定するか、将来Codex dispatchを
    one-shot leaseの別capabilityとして追加reviewするか

owner directionとして2は本Draftへ反映済みである。これはG2実装や実行承認ではなく、
G2 reviewで上記durability/security invariantsを満たせなければ`EXECUTION_FORBIDDEN`を維持する。

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
