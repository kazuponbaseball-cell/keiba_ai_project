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

## 通常strategyの二段階実行承認

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

通常strategy routeでは`BLOCKED_SCORE`は75点未満の実行不可状態である。
`INVALID`は全routeでterminalである。

## 登録済み非昇格診断 fast lane

`registered_nonpromotion_diagnostic_v1`は、通常strategyの実行・採用ではなく、
凍結済みartifactに対する再現可能な構造監査とhistorical impact診断だけを行う
別routeである。通常の`HYPOTHESIS_SCORECARD`を無効化せず、strategy由来の診断は元のscore、
`BLOCKED_SCORE`、`threshold_met=false`、`score_credit=0`を全artifactへ保持する。純粋な
parity監査には架空のordinary scoreを作らず`NO_ORDINARY_STRATEGY_CLAIM`と記録する。
診断結果をscore、shadow、adoption、production、BUYの根拠へ変換してはならない。

trusted dispatcherは、`ordinary_strategy_v1`を既存75点gateへ、
`registered_nonpromotion_diagnostic_v1`を下記strict hard classifierへ、
`registered_nonpromotion_offline_diagnostic_v1`を後述のexact one-recipe local classifierへ完全分岐し、その他を拒否する。
proposalの自己申告だけでrouteを選択できない。

- `A_PARITY_NO_DECISION_CHANGE`: lineage、hash、schema、alias、確率mass、bitwise parityを
  監査する。candidate key、rank、tier、coverage、eligibility、decision vectorの差は0で
  なければならず、result、odds、payoff、settlement、ROIを読まない。1行でも差があれば
  settlementを開かずoutcome `INVALID`、reason code `INVALID_NO_SETTLEMENT`とする。
  ordinary score recordは既存値を保持するか、
  strategy claimがない場合は作成しない。
- `B_REGISTERED_HISTORICAL_IMPACT`: current main上の有限・immutableなrecipe registryに
  登録されたexact digestだけを実行する。exactly 2 arms、exactly 1 registered transform、
  threshold/refit/recalibration/search 0を要求する。live policy/config、candidate identity、
  rank/tier、model、calibrator、market式、cohort、stake ruleは変更しない。recipeが明示した
  counterfactual eligibility mask差だけを許し、decision freeze後にのみsettlement/payoffを読む。
  odds、price、popularity、market dataは全phaseで読まず、settlementはrace ID、candidate key、hit、
  outcome completeness、official payoffのstrict allowlistに限る。source strategyのordinary score recordを
  exact digestでbindしなければならない。

fast laneは一度だけ、別のhuman-owned Draft PRでpolicy、strict schema、finite recipe registry、
bounded generic runner、GitHub verifier、content-addressed candidate/settlement catalog、durable
single-use ledger、one-shot lease、supervised executorを実装し、人間mergeとcutover receiptを
完了してから有効になる。このroot amendmentのmergeだけでは実装・activation・executionを
許可しない。

有効化後の既存recipe実行ではper-run code変更、prepare phase、PR、mergeを行わない。
recipe、input、cohort、metric、sensitivity、payoff、bootstrap、budget、seed/RNG、environment、phase plan、
replica/attempt/retry topology、argvはproposal/caller値から作らず、active current-main registry/policyから
resolverがexact bytesを取得する。attempt countは登録値そのものとし、automatic/manual retryとreplica結果の
選り好みを禁止する。run digestはrepository、base branch、run-scope base commit、
expected pre-grant global/subject heads、APPROVERS blob/content digest、lane activation receipt、policy、
schema、compiler、verifier、executor、capability profile、recipe entry blob、ACTIVE candidate/settlement
releaseとrevocation state、ordinary score scope、candidate identity、reference/counterfactual AST、
allowed difference、semantic subject、race-set、metric/sensitivity/payoff/bootstrap、phase plan、lease
schema/issuer/capability/counts、replica/attempt/retry topology、runner commit/blob、interpreter/dependency/
locale/timezone/seed、cwd、empty environment、
timeout、read/write allowlist、exact argv、fresh output root、approval evidence schemaを固定する。allowlist済みGitHub
`User`による未使用comment
`APPROVED_NONPROMOTION_DIAGNOSTIC_RUN <run_scope_digest>`を1回だけ要求する。commentとcurrent
main、APPROVERS、recipe、catalog、authenticated ledger receipt chainはdispatch直前に再検証する。chat指示、CI成功、
PR Ready、merge、過去commentはgrantではない。canonical semantic/exact subjectの不可逆実行を全generation横断で
最大1とし、rename、alias、別experiment IDによる
performance再探索を拒否する。認証済みpre-access abort後だけ、新scope、新comment、新generationで再予約できる。

run scope seal後に作成されたcommentだけを受理する。durable ledgerの単一transaction/CASでcomment ID、
semantic/exact subjectのprovisional reservation、run approval receiptを原子的に記録し、receiptへexpected/actualの
old/new global/subject headsをbindする。comment IDはこの時点で永久consumeするが、phase leaseは発行しない。
各phaseの直前にshared G2だけが、predecessor receiptとremote再検証を条件にrun/recipe/replica/phase/attempt別の
one-shot leaseを発行・consumeする。cross-phase、cross-replica、cross-run、replay、self-issue、capability unionを
拒否する。GitHub evidence、current main、APPROVERS、lane、recipe、catalog、authenticated receipt chainはdispatch、
decision lease、settlement lease、result sealの各直前に再検証する。unrelatedな後続global appendだけでは、
認証済みsubject receipt chainを無効にしない。

run scopeへbindしたlane/policy/recipe/catalog/APPROVERS digestはbit-identical、ACTIVE、unrevokedでなければならない。
driftをdecision lease consume前かつcandidate/result/odds/payoff/settlementへ一度もaccessする前に検出した場合だけ、
authenticated atomic abort CASで旧runを`INVALID` terminal化し、global/subject headsを進め、発行済み未consume leaseを
全てrevoke/tombstoneし、旧approval/predecessor receiptを永久に不適格化した上でsemantic/exact subject reservationを
解除できる。comment IDは解除せず、置換実行には新scopeと新commentを要求する。decision leaseをcandidate mount直前に
consumeする処理、semantic/exact subjectの永久consume、question-family execution countの加算、authenticated
irreversible receipt発行を1つのatomic global/subject-head CASで行う。candidate mountはそのreceiptを再検証しなければ
ならない。それ以降のdrift、crash、contract failureは`INVALID` terminalであり、新scope、retry、置換実行を認めない。
automatic retryは禁止する。
future approval receiptまたはlease ID/digestをrun scopeへ入れない。post-approvalで発行する全receiptと
leaseが、逆向きにfrozen run digestをbindしなければならない。

pre-grant resolverが読めるのは署名済みcatalog metadata/manifestだけで、candidate/settlement blobのmountやrow
readを禁止する。ordered race-setはprepublished signed manifest digestから取得し、candidate/settlement contentは
authenticated irreversible receiptの再検証後にのみmount/readできる。

diagnostic lifecycleは通常strategy lifecycleを流用しない。

```text
RND_RUN_SCOPE_FROZEN
  -> RND_RUN_APPROVAL_REQUIRED
  -> RND_APPROVED
  -> RND_LEASED
  -> RND_RUNNING
  -> RND_RESULT_SEALED
       -> RND_COMPLETED
       -> INVALID
```

上記の隣接遷移だけを許し、各nonterminalから`INVALID`へのfail-closeを許す。self transition、state skip、
terminalからの復活を禁止する。scope seal、approval、lease issue、irreversible start、result seal、completion、
fail-closeの各遷移は、run state、global/subject heads、対応receiptを1つのatomic CASで更新する。特に不可逆CASは
`RND_LEASED -> RND_RUNNING`も同時に行い、state/head/receiptのsplit-brainを拒否する。

`semantic_subject_digest`はcanonical IDへ正規化したgate/tier、recipe class、recipe ID/version/entry digest、
source model/policy/calibrator、reference AST、registered transform、candidate identity/rank contract、population/cohort rule、
metric/sensitivity、ordinary score scopeから作る。`exact_run_subject_digest`はこれにcandidate/settlement
releaseとordered race-setを加える。ledgerは両方をreserveし、同じsemantic subjectを別release、別名、
別experiment IDでroutine再実行しない。release/cohortを変える場合はfast laneを拒否し、通常strategyへ戻す。
subject stateは`PROVISIONALLY_RESERVED`、`RELEASED_PREACCESS_ABORT`、`IRREVERSIBLY_CONSUMED`を区別する。
global single-use blockerは新規実行かつ`IRREVERSIBLY_CONSUMED`にだけ適用し、pre-access abortで正しくreleaseされた
subjectは新generationのscope/commentでのみ再予約できる。旧generationはterminal/tombstoneのまま、aggregate
subject headをCASしてgeneration+1を`PROVISIONALLY_RESERVED`として開始する。同時active generationは最大1、
全generation横断のirreversible executionは最大1とし、旧run、approval、receipt、leaseは永久tombstoneのままとする。
exact digestのreplayは、trusted sealerのauthenticated receipt、semantic/exact subject digest、result digest、
original approval evidence/receipt chain、lane/recipe/catalog/resultのunrevoked状態を再検証したsealed resultの
read-only retrievalだけとする。不一致またはcache miss時は状態遷移せず停止し、新comment、lease、executor、
再計算を起動しない。

さらに`question_family_digest`はrecipe suppliedのID、名前、digestを信用せず、trusted verifierがimmutable source
model/policy/calibratorとreference AST node、canonical target-decision/population/metric registry digestから導出する。
recipe表示名、transform、threshold、cohort、releaseはdigestへ影響させず、同じlineageのaliasは同一digestへ解決する。
既登録canonical familyへのrecipe追加はmerge eligibility判定前に拒否する。
B-tierは1 familyにつきlifetime recipe 1件、new execution 1件を上限とする。隣接thresholdや別transformを
新recipe名で追加して探索することを拒否し、追加検証は通常75点gateまたは新gate versionへ戻す。

PR #40の一件限定designは初期recipeのsource evidenceであり、generic fast laneの実装契約または
authorityではない。本root amendmentは同designの`generic catch-all=false`、recipe追加ごとのnew kind、
prepare+run+ACKという将来案を、別のhuman-reviewed generic implementation contractにより置き換える。
PR #40 bytesは変更せず、generic implementationがmainへmerge・cutoverされるまで旧designから権限を推論しない。

shared G2 durable ledgerをsole live authorityとし、lane専用local/separate backendを作らない。activation前に
legacy event、全gate共通grant comment ID、全terminal/nonterminal subject headを完全移行する。global headと
subject headの両方をatomic CASし、old writerをfenceした後にsecond remote compareを通し、atomic cutoverと
authenticated receiptを要求する。authenticated external monotonic checkpoint/witnessでbackup restore、rollback、
fork、stale headを検出し、検出時はlaneを停止して新規leaseを発行しない。dual writerやlocal file、SQLite、
worktree、branch、process-memory authorityへのfallbackを許可しない。

run approval evidenceは既存`github_backed_approval_evidence_v1`をそのまま継承する。このrouteに限り、proposal base
commitの役割をrun-scope base commitが担い、APPROVERSはGitHub上のそのcommitから取得する。それ以外のtrust
semanticsは変更しない。repository、base branch、
run-scope base commit、verified current main、compare URL/status、merge-base、APPROVERS blob SHAとcontent SHA-256、
verification time、comment ID、Issue、URL、author login/type、body、keyword、run digest、body SHA-256、created_at、
updated_atを個別に保存・再検証し、いずれかの変更、削除、再利用、非allowlistまたはbot actorをfail-closeする。

新しいrecipeは既存entryを編集・削除せず、append-onlyの新規recipe JSONだけを小さなhuman-reviewed
PRで追加する。そのPRはrunner、schema、policy、verifier、workflow、root governance、model/config/data/
production path、execution scopeを同時変更せず、実データ、result、payoff、ROIを読まないCIで
recipe schema/fixtureだけを検証する。
新operator、新capability、free-form AST/SQL/Python/argv、新data source、training、
fit、model inference、calibration、threshold/cohort/variant search、3 arms以上、candidate/rank/tier、
market/value式、stake変更、prospective outer、shadow、production、network、credential、BUY、order、
notificationを含む場合はfast laneを拒否し、通常75点gateまたは新gate versionへrouteする。

全resultは`evidence_purpose_class=DIAGNOSTIC_NONPROMOTION`とし、source authority classを保持する。
初期recipeは`source_authority_class=B_LOCAL_HASHED`である。全resultは`confirmatory=false`、
`promotion_eligible=false`、`score_credit=0`、`formal_buy=false`、`send_order=false`、`stake=0`
をconstとする。A-tier outcomeは`NO_DECISION_EFFECT|INVALID`、B-tierは
`NO_DECISION_EFFECT|DIRECTIONAL_EFFECT|INVALID`だけで、`APPROVED_FOR_SHADOW`への遷移を定義しない。

## 一件限定の軽量offline非昇格診断

`registered_nonpromotion_offline_diagnostic_v1`は、上記strict G2 routeを緩和しない。
`historical_ai_duplicate_gate_impact_v1@1`だけを対象にする別gate kind/version/pathであり、通常strategyの
75点gateと`registered_nonpromotion_diagnostic_v1`は不変である。source strategyは
`46 / BLOCKED_SCORE / threshold_met=false / score_credit=0`を保持する。

root契約と固定実装は同じhuman-owned Draft PRでbootstrapできるが、含有commitが人間により`main`へmergeされる
までは利用不能である。merge、CI、PR Ready、review、chat指示だけではrunを許可しない。merge後にcleanな
current-main checkoutでscopeをsealし、その後に作成されたallowlist済みGitHub `User`のexact comment

```text
APPROVED_OFFLINE_NONPROMOTION_DIAGNOSTIC_RUN <run_scope_digest>
```

をread-onlyで検証した場合だけ、そのscopeのlocal offline runを受理する。commentのrepository、Issue、URL、ID、
author login/type、body、body SHA-256、created_at、updated_atを固定し、編集、削除、再利用、bot、allowlist外、
seal以前、別digest、GitHub取得不能をfail-closeする。candidate contentを開く直前とresult公開直前にcomment、
current main ancestry、APPROVERS、policy、recipe、schema、runner、input hashを再検証する。

human merge後、scope seal前に許可する実データ操作は、policy固定の3 source hashからcandidate-only projectionと
settlement-only projectionを別fileへ作る決定論的materializationだけである。materializerはD0/D1、hit率、profit、
ROI、bootstrap、threshold、variantを計算・出力しない。raw sourcesに同居するoutcome、label、payoff、
popularity列は固定hash検証の対象bytesに含まれるが、materializerはroleごとのexplicit allowlist列だけを投影する。
candidate projectionへoutcome/label/payoff/market値を使用・出力・永続化せず、decision/run phaseへraw sourceをmountしない。
承認再検証後かつexclusive start receipt作成前に限り、control-plane provenance verifierが固定3 sourceを再読し、
projectionとmanifestを決定論的に再生成してbyte一致を検査する。このpreaccess検査はdecision、metric、ROIを計算せず、
完了後にraw bytesとrunner内部のraw root参照を破棄し、fixed runner codeはexclusive start receipt後にraw sourceを
openしない。OS/ACLによるfilesystem capability isolationを保証したとは主張せず、workloadはprojectionだけを読む。
runはcandidate projectionだけを先にopenし、exact
3,746 races / folds 2–4 / one row per race / p-action cross-source equalityを検査してD0/D1 decisionをfreezeし、
freeze receiptを保存した後だけsettlement projectionをopenする。settlementはrace ID、candidate key、hit、
outcome completeness、official payoffだけを含み、odds、price、popularity、market列をすべてのdecision/run phaseで禁止する。

runはexactly 2 arms、1 registered transform、固定cohort、固定metric/sensitivity、100,000回の固定bootstrap、
logical replicas `clean_a`/`clean_b`各1回の意味論的一致、retry 0である。別process/OS隔離は保証したとは主張しない。callerの
formula、threshold、cohort、metric、seed、argv、output変更を拒否する。candidate identity、
rank、tier、model、calibrator、live policy、stake ruleを変えない。候補を開く前にfixed question-family pathへ
exclusive local start receiptを作成し、それ以後のcrash、drift、contract failureは
`INVALID_AFTER_START_NO_RETRY`とする。fresh output rootだけを許し、overwriteとpartial performance outputを禁止する。

この軽量routeにはdurable remote ledger、global CAS、rollback/fork detection、OS sandboxがない。全artifactは
`single_use_policy=ONE_ACCEPTED_EXECUTION`、
`single_use_enforcement=BEST_EFFORT_LOCAL_EXCLUSIVE_RECEIPT`、`global_replay_proof=false`、
`rollback_resistant=false`、`durable_remote_ledger=false`、
`network_isolation=APPLICATION_LEVEL_NOT_OS_SANDBOX`を記録する。削除・rollback・別cloneによる再計算を技術的に
完全阻止したとは主張しない。

resultは常に`B_LOCAL_HASHED / DIAGNOSTIC_NONPROMOTION / confirmatory=false / promotion_eligible=false /
score_credit=0 / strict_t3_rows=0 / reused_development_oos=true`であり、ROI差をscore、shadow、adoption、production、
BUYへ利用しない。`formal_buy=false`、`send_order=false`、`stake=0`を固定する。同question familyの将来のstrict-v1
routeは独立かつ不変であり、この軽量routeはcross-route重複実行を技術的に防止しない。offline resultはglobalな
question-family consumptionではなく、将来のstrict resultを独立証拠として扱う場合はprior offline exposureを明示的に
bindしなければならない。cross-route single-use、独立replica、global replay protectionを主張しない。
将来の別question-family診断はこの実装パターンを再利用できるが、このv1に自動適格はしない。同じ固定operator/capabilityの
add-only recipe/versionを小さなhuman-reviewed PRで追加し、distinct question-familyであることを検査する。隣接threshold、同familyの
別transform、追加arm、searchは通常strategy gateへ戻す。shared G2/root bootstrapを診断ごとに作り直さない。

軽量routeのlifecycleは次に限定する。各nonterminalから`INVALID`を許すが、skip、self transition、terminal復活、
retryを禁止する。protected content access前の検証失敗は`BLOCKED_PREACCESS`で、まだrunを消費しない。
control-planeのclean HEAD/status検証に限り、固定argvのread-only `git` subprocessを許可する。free-formまたはworkload subprocessは
禁止する。

```text
RNOD_RUN_SCOPE_FROZEN
  -> RNOD_RUN_APPROVAL_REQUIRED | INVALID
  -> RNOD_APPROVED | INVALID
  -> RNOD_RUNNING | INVALID
  -> RNOD_RESULT_SEALED | INVALID
  -> RNOD_COMPLETED | INVALID
```

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

## ROI reproduction v2 — G1権限境界

`roi_reproduction_audit_v2`のG1は
`CONTRACT_COMPILER_ONLY_NO_AUTHORITY`である。G1はschema、別pathのpolicy、canonical
serializer、state validator、非権限object compiler、synthetic-only governance testだけを
実装する。provider、writer、authority verifier、executorは実装せず、proposal、queue、eventの
正規形候補をrepositoryやledgerへ書かない。catalog未検証のproposal候補は
`BLOCKED_CATALOG`、run候補は`BLOCKED_CAPABILITY`で停止する。

- canonical artifact identityは将来用途を`execution_kind=historical_reproduction_v2`へbindするが、
  G1の有効なruntime execution kindはnone、全authority flagはfalse/0、結論は
  `EXECUTION_FORBIDDEN`である。人間comment、score、design artifactでも上書きできない。
- G2のcatalog publisher/attester、authority verifier、durable runtime ledger、one-shot lease、
  supervised executor、real-data/model accessは未実装であり、G1から存在を推論しない。
- 現行legacy ROI queue/event schema v2とinfra queue/event schema v3のwriter、digest、transition、
  approval semanticsは変更しない。現行writerはv4をdispatchせず、G1のv4 schemaとcanonical
  line compilerはvalidation用であってlive event writerまたはauthority sourceではない。
- prepare/run/resultの3 action workflowは現在activationされていない。既存のapproval keywordや
  commentはG1/G2/Aのgrant、開始、再開、実行権限にならない。
- G1/G2 root-of-trustは自身、`infrastructure_safety_v1`、branch copyで承認できない。
  各bootstrapは別のhuman-owned Draft PRとしてreview・mergeし、merge済みmainだけを根拠にする。
- G1 bootstrapのexact changed pathsはpolicyへ監査用に固定する。後続proposalからgovernance
  root、workflow、既存approval verifier、registry writer、scorecard、infra v1 policyを変更できない。

## 通常strategyの承認詳細

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

## 通常strategyのGitHub-backed approval evidence

この節の`REGISTRY.jsonl` event、prepare/run/shadow comment、branch merge前提は通常strategyに適用する。
fast laneが`github_backed_approval_evidence_v1`から継承するのはrepository identity、main ancestry、APPROVERS、
human User、comment immutable fieldsのtrust semanticsだけである。fast laneのevidence保存、single run comment、
CAS、leaseは前節のshared G2専用契約に従い、この節のordinary registry/event mechanicsを継承しない。

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

## 通常strategyのCanonical scope

この節のproposal scope前提は通常strategyに適用する。fast laneはproposal free-form値を持たず、前節の
current-main registry/policy resolverから作る専用canonical run scope契約に従う。

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
