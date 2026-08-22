# 競馬AI Research OS v1 憲章

- 制度: Level 3（承認付き自律研究）
- 発効日: 2026-07-31
- Research OS foundation status: `RESEARCH_OS_V1_FOUNDATION_MERGED`
- reconciled-through main commit: `1eaf364571bd8b9fd27f7de657ce295b563b3f1f`
- モデル監査基準コミット: `288dff5e86385908281428d5ed4f077625a43e4b`
- Foundation branch（履歴）: `chore/research-os-v1`
- Reconciliation snapshot（2026-08-01T15:21:25+09:00）: PR #3、branch
  `chore/research-os-post-merge-reconciliation`、observed open・Draft・未merge
- Snapshot scope: PR #2 merge後のstate・governance文書・template整合
- Snapshot scope外: 実験、予測モデル、候補選択、value判定、正式BUYロジック

上記PR #3のbranchとreview状態は観測時点のreconciliation snapshotであり、
Research OS本体のlive stateではない。

## 1. 目的

購入可能時点までの情報だけを使い、chronological outer OOSで再現する長期ROI改善を研究する。単一期間の高ROIや大当たり依存を成果とみなさず、確率品質、安定性、再現性、価格結合前の候補独立性を同時に評価する。

## 2. Level 3の意味

Agentは、研究提案、read-only監査、score作成、experiment draftを自律的に行える。通常strategyでは、GitHub上でproposal scopeが`APPROVED_TO_PREPARE`された後はresearch branch上の実装とsynthetic fixture testを行える。実装後のexact run scopeがexecution kindとcapabilityをcanonical digestへ拘束するversioned contractであり、別の未使用comment IDで`APPROVED_TO_RUN`され、prepareとrunの両承認commentを`RUNNING`直前に再検証した場合だけ、承認scope内の実データ実験を行える。登録済み非昇格診断だけは§5.3の専用lifecycleを使う。現行legacy ROI contractはこの拘束を欠くためsynthetic-onlyである。

Agentは次を行えない。

- 正式BUYの再開、stake決定、注文送信
- 現行の予測モデル、候補選択、value判定、正式BUYロジックの変更
- `main`への直接commit、PRの自動merge、本番promotion
- 結果閲覧後の同一experiment IDによる仮説・fold・gate変更
- 自己申告flag、caller指定actor、Codex、bot、automationによる自己承認

本番反映とmergeは、常に人間の明示承認を必要とする。

## 3. 研究境界

```text
as-of input
  -> non-odds features
  -> Top3 set softmax
  -> non-odds candidate freeze + digest
  -> chronological outer OOS evaluation
  -> post-freeze price/result join
  -> diagnostic ROI
  -> human review
  -> shadow only
```

価格と払戻は候補freeze後の評価にだけ利用できる。価格結合後に候補key、rank、tier、coverageを変えてはならない。

## 4. 絶対契約

### 4.1 Top3集合確率

取消反映後のrunner universeを `U_r`、非順序3頭集合を `S_r = C(U_r, 3)` とする。canonical probabilityは集合utilityのsoftmaxである。

```text
q_r(s) = exp(u_r(s) / T) / sum(exp(u_r(t) / T) for t in S_r), T > 0
```

- `q_r(s)`はfiniteかつ `0 <= q_r(s) <= 1`。
- 取消後runner universeの全 `C(|U_r|,3)` 集合を一度ずつ含める。
- runner universeは確率artifactとは独立にfreezeし、runner count/hashを契約検査へ渡す。観測集合のunionだけを完全性証明にしない。
- 各raceで `sum(q_r(s)) = 1`、許容誤差は `1e-10`。
- wide probabilityは `p_r(a,b) = sum(q_r(s) for s containing {a,b})` としてのみ導出する。
- 各raceで `sum(p_r(a,b)) = 3`、許容誤差は `1e-10`。
- 独立pair sigmoidをcanonical probabilityの代替にしない。
- mass、finite、重複、runner universeの検査失敗時はfail-closeし、ROIを読まない。

### 4.2 Odds-free candidate

対象raceの単勝・複勝・馬連・ワイド等のodds、人気、market probability、margin、expected ROI、払戻、的中、利益、およびその派生列を候補選択に使用しない。追加、除外、順位、tier、tie-break、coverage、abstentionの全てが対象である。

### 4.3 Chronological outer OOS

- 順序は `train < validation < calibration < outer test`。
- 分割単位はraceで、区間間のrace重複を0とする。
- feature selection、imputation、scaling、category dictionary、hyperparameter、temperature、thresholdはouter testより前だけで決める。
- purge/embargoとfold manifestを事前登録し、baselineとchallengerで共有する。
- outer testの着順、払戻、ROI、失敗分析を見て同じIDを調整しない。

### 4.4 As-of・leakage

事後情報、未来情報、target race後に利用可能になった値、outer result由来の加工、未来値による欠損補完を禁止する。各inputはevent/source/received timeとcontent hashで追跡し、証明できないlineageは未確認としてfail-closeする。

### 4.5 正式BUY停止

Research OSが作る全artifactは `formal_buy=false`、`send_order=false`、`stake=0` とする。研究コードからproduction BUY pathをimportまたは実行しない。正式BUYの再開は本制度の権限外である。

## 5. Hypothesis gate

仮説は `research/HYPOTHESIS_SCORECARD.yaml` で100点満点評価する。

- 通常strategy routeで75点未満: `BLOCKED_SCORE`。実行禁止。
- 75点以上: `PROPOSED`。GitHub上の承認証拠までは準備・実行禁止。
- `APPROVED_TO_PREPARE`: 実装準備とsynthetic fixture testのみ可能。
- `APPROVED_TO_RUN`: exact code/config/data/fold/command scopeに対する実行承認。
- `APPROVED_FOR_SHADOW`: prepare/runと異なる未使用comment IDによるreview digest承認。
- 契約違反: `INVALID`。metricsやROIで救済しない。

高得点は本番採用を意味しない。shadow移行と本番反映には別の人間承認が必要である。

### 5.1 Infrastructure safety gate

Research OSのcontrol-plane、contract、schema、synthetic adapterなど競馬仮説でない
変更は、`research/INFRASTRUCTURE_GATE.json`の
`infrastructure_safety_v1`を使う。これはHypothesis score 75点の代替得点ではなく、
全項目必須の非補償型gateである。

- 競馬仮説をinfraへ再分類して75点gateを迂回できない。
- proposal/run digestはgate kind、policy hash、capability lock、synthetic fixture、
  exact commit、変更path/blob、構造化command、environmentを固定する。
- prepare/runのGitHub Issue承認と直前再検証はROI lifecycleと共通である。
- `RUNNING`はsynthetic-onlyで、real data、training、backtest、outer OOS、ROI、
  external API/network、credential、actual Codex dispatchを一切許可しない。
- model、feature、candidate、value、production、BUY、order、notification、shadowは
  gateの権限外である。
- gate自身、approval verifier、allowlist、憲章、scorecardなどroot-of-trustは
  infra gateで変更できない。
- registryはsymlink/junctionでないexact code-owned ledgerだけをprocess lock/CASで追記し、base commit、
  execution commit、worktree間で過去bytesのappend-only prefixを検証する。さらに権限遷移前に
  GitHub current mainのledgerをremote取得し、localがそのbytesと完全一致しなければ拒否する。
  append直前にmain headを再検証する。追記eventはmainへ人間mergeされるまでpendingであり、
  次transitionはmerge後のexact ledgerへbranchをrefreshしなければ作成できない。
- synthetic materialは専用prefix、bounded provenance envelope、hash、secret/row-shape
  sentinelを要求し、execution commit blobへ固定してdirty/untrackedとsymlink/junctionを拒否する。
  execution commitはproposal baseの子孫だけを認める。変更Python blobはpure import allowlist、
  AST import/call/symbol firewallを通し、isolated `-B -I -S` interpreter、repository-root cwd、
  空の継承environment、proposal budget由来timeoutだけをcommand scopeへ固定する。
- このgateは自動executorを提供しない。infra eventの`automatic_execution_allowed`、
  `preparation_authorized`、`execution_authorized`は常にfalseである。v3はevidence compilerであり、
  exact synthetic commandの実行には別の人間review済みexecutor/authority verifierを必要とする。
- 後続infra testは通常PR CIの`tests/research/test_*.py`探索対象へ置かず、専用の
  `research/infra_tests/test_*.py`へ置く。`tests/research/`はroot-of-trustとして変更を拒否し、
  専用testは別のexecutor/authority verifierが導入されるまで実行しない。

このgateの初回bootstrapは既存gateで自己承認しない。Draft governance PRを人間が
review・mergeしたmain commitからのみ有効となる。merge前のbranch copyを後続proposalの
信頼根拠にしてはならない。
v1 policyはmain merge後にin-place変更しない。将来のpolicy変更は新しいgate
kind/versionと別pathを追加し、既存v3 scopeを旧policyのまま監査・無効化できるようにする。

既存のunversioned ROI run scopeは`execution_kind`をcanonical digestへ含めていない。
したがって現行contractからのreal-data `RUNNING`はfail-closeし、synthetic実行だけを
許可する。real-dataを再開するには、execution kindとcapabilityをproposal/run digestへ
固定し、legacy digestを変更しないversioned ROI contractを別のgovernance PRで導入する。

`ordinary_real_data_run_v3`はこのversioned境界を別schemaとして実装する。version fieldのない
legacy scopeは従来どおりv2へdispatchし、そのbytes、digest、validation、writer semanticsを
変更しない。exact v3以外のversionはfail-closeする。v3は有限capability profile、execution kind、
execution commit、input/environment/command/access/output sealをcanonical digestへ束縛するが、
scopeまたはgovernance PRの存在だけではrow access authorityを持たない。actual row mountには、
別の未使用`APPROVED_TO_RUN`、Prepare/Run comment再検証、human-merged current-main `RUNNING`
event、execution receipt、metadata-only preflight、fresh output root、phase read/write allowlistの全成立を
必要とする。EXP-034 canonicalizationとEXP-033 research model runは別scope・別root・別sealとする。
production、Champion、candidate/value policy、BUY、notification、order、stake、merge/promotion能力は
常にfalseである。
receipt自体はnon-executingとし、exact phase argvを再観測したbrokerだけが一時的な実効authorityを
導出する。EXP-033によるEXP-034 output消費には、receipt/result/artifact digestを束縛した
digest-addressed post-run attestationのGitHub mainへの人間mergeを追加で必要とする。

### 5.2 ROI reproduction v2 G1 declaration

`roi_reproduction_audit_v2`の現行G1境界は
`CONTRACT_COMPILER_ONLY_NO_AUTHORITY`である。これはreproduction専用gateの実行承認ではなく、
G1 implementationをschema、別version policy、canonical serialization、state validation、
非権限object compilation、synthetic-only governance testに限定する契約である。この変更は
それらのpure compiler、policy、schemaを実装するが、provider、writer、authority verifier、
executorを実装せず、repository/ledgerへproposal、queue、run scope、eventを生成・追記しない。

G1はcatalog/source provider、attester、authority verifier、durable ledger、lease、executor、
subprocess、network、real data、model、training、historical replay、outer OOS、ROIを扱わない。
artifact contractは`execution_kind=historical_reproduction_v2`をdigestへbindする一方、現在の
有効runtime execution kindはnone、全authorityはfalse/0、`EXECUTION_FORBIDDEN`とする。
G2は未実装であり、別のhuman-owned governance Draftとhuman mergeなしに権限を追加できない。

現行legacy ROI schema v2とinfra schema v3のwriter、digest、transition、approval semanticsは
不変とする。現行writerはschema v4をdispatchしない。
`research/schemas/roi_reproduction_registry_event_v4.schema.json`とcanonical line compilerは
validation用であり、現時点のlive writerまたはauthority sourceではない。prepare/run/resultの3 action flowも
未activationであり、既存comment、design、score、branch artifactは開始・再開・実行tokenにならない。
G1 schema単独またはtrusted policy/schema digest contextなしのnormalizer結果はfixture検証に限り、
proposal、queue、event、grant、lease、execution authorityとして永続化・利用しない。G2はcatalog
source/providerの有限allowlist、全manifest contract conformance、execution context、expected outputと
numeric referenceをcurrent-mainのpolicy/schema bytesへhash-bindしてからでなければactivationできない。

### 5.3 Registered non-promotion diagnostic fast lane

頻発する構造監査を通常strategy開発と同じ準備工程で毎回作り直さないため、
`registered_nonpromotion_diagnostic_v1`を別routeとして定義する。これは75点gateの
点数免除ではない。strategy由来の診断はordinary score recordを必ず保持し、75点未満なら
strategy statusは`BLOCKED_SCORE`、threshold未達、credit 0のままである。strategy claimを
持たないpure parity監査に架空のscoreを作ってはならない。diagnostic authorizationは
strategy authorizationとして利用できない。

trusted dispatcherは`ordinary_strategy_v1`を既存75点gateへ、
`registered_nonpromotion_diagnostic_v1`を下記hard classifierへ分岐し、unknown/ambiguous routeを
拒否する。routeは二層とする。

1. `A_PARITY_NO_DECISION_CHANGE`は、凍結済みbytesのlineage、schema、hash、alias、mass、
   deterministic parityだけを扱う。candidate、rank、tier、coverage、eligibility、decisionの
   semantic digestは完全一致しなければならず、settlement、result、odds、payoff、ROIへ
   accessしない。既存scoreがあれば保持し、なければ`NO_ORDINARY_STRATEGY_CLAIM`とする。
2. `B_REGISTERED_HISTORICAL_IMPACT`は、current main上の有限・immutableなrecipe registryに
   登録されたexact recipeだけを扱う。exactly 2 arms / 1 registered transform / 1 frozen cohort
   とし、training、fit、inference、recalibration、threshold・variant・cohort searchを行わない。
   live policyを変更せず、candidate identityとrank/tierを固定し、登録recipeが明示した
   counterfactual eligibility mask差だけを許す。source strategyのordinary score recordを
   exact digestでbindする。decision freeze後にのみ、物理分離された
   settlement snapshotをjoinしてhistorical unit-notional impactを計算できる。odds、price、popularity、market
   dataは全phaseで禁止し、settlementはrace ID、candidate key、hit、outcome completeness、official payoffの
   strict allowlistだけをmount/readする。

proposalはfree-form AST、SQL、Python、argv、threshold、cohort、metricを定義できず、
current-main recipe ID/version/digestとprepublished content-addressed input releaseだけを参照する。
recipe registry entryの追加はappend-onlyとし、既存entryの編集・削除を禁止する。registration
PRは実データ、result、payoff、ROIを読まず、schemaとsynthetic fixtureだけを検証する。新operator、
capability、data sourceまたは自由度の追加は新gate kind/version/pathとhuman reviewを必要とする。

このlaneは、別のhuman-owned implementation PRでstrict schema、finite registry、bounded generic
runner、GitHub verifier、candidate/settlement catalog、durable global single-use ledger、one-shot
lease、supervised executorを実装し、人間merge後のcutover receiptがcurrent mainと全digestを
bindするまでは`EXECUTION_FORBIDDEN`である。この憲章変更、design、CI、chat、PR Ready、merge
だけから実行権限を推論しない。

activation後の登録済みrecipeはpre-merged runnerだけを使い、per-run code preparation、PR、mergeを
禁止する。recipe、input、cohort、metric、sensitivity、payoff、bootstrap、budget、seed/RNG、environment、phase plan、
replica/attempt/retry topology、argvはproposal/callerで作らず、active current-main registry/policyからresolverが
取得する。attempt countは登録値そのものとし、automatic/manual retryとreplica結果の選り好みを禁止する。
canonical run scopeはrepository、base branch、run-scope base commit、expected pre-grant
global/subject heads、APPROVERS blob/content digest、lane activation receipt、policy/schema/compiler/
verifier/executor、capability profile、recipe blob、ACTIVE candidate/settlement releases、ordinary score
scope、candidate/reference/counterfactual AST、allowed difference、semantic subject、race-set、
metric/sensitivity/payoff/bootstrap、runner commit/blob、phase plan、lease schema/issuer/capability/counts、
replica/attempt/retry topology、interpreter/
dependencies/locale/timezone/seed、cwd、empty environment、timeout、read/write allowlist、argv、output、
approval evidence schemaをfreezeする。その後、allowlist済みGitHub `User`の未使用comment
`APPROVED_NONPROMOTION_DIAGNOSTIC_RUN <run_scope_digest>`を1回だけ取得し、dispatch直前にremote
evidence、current main、APPROVERS、recipe、catalog、authenticated ledger receipt chainを再検証する。run approvalは
そのdigestの診断実行だけを許し、code変更、再探索、result acknowledgement、shadow、adoption、
promotion、production、BUYまたはmergeを許可しない。canonical semantic/exact subjectの不可逆実行は全generation
横断で最大1とし、同じ問いをrename、alias、別experiment IDで再実行することを拒否する。認証済みpre-access
abort後だけ、新scope、新comment、新generationで再予約できる。

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

このlifecycleに`PREPARING`はない。run approval前後でcode、recipe、inputまたはcommandが変われば
new digestとなり、旧approvalは無効である。`semantic_subject_digest`はcanonical IDへ正規化した
gate/tier、recipe class、recipe ID/version/entry digest、source model/policy/calibrator、reference AST、registered transform、candidate
contract、population/cohort rule、metric/sensitivity、ordinary score scopeから作る。
`exact_run_subject_digest`はcandidate/settlement releaseとordered race-setも含める。両digestをprovisional reserveし、
認証済みpre-access abortによるrelease以外で同一semantic subjectを別releaseで実行するfast-lane requestは拒否する。
subject stateは`PROVISIONALLY_RESERVED`、`RELEASED_PREACCESS_ABORT`、`IRREVERSIBLY_CONSUMED`を区別する。
global single-use blockerは新規実行かつ`IRREVERSIBLY_CONSUMED`にだけ適用する。pre-access abortでreleaseされたsubjectは
新generationのscope/commentでだけ再予約できる。旧generationはterminal/tombstoneのままaggregate subject headを
CASしてgeneration+1を開始する。同時active generationは最大1、全generation横断のirreversible executionは最大1とし、
旧run、approval、receipt、leaseは永久tombstoneのままとする。
exact replayはtrusted sealerのauthenticated receipt、semantic/exact subject digest、result digest、original approval
evidence/receipt chain、lane/recipe/catalog/resultのunrevoked状態を再検証したsealed resultのread-only retrievalだけである。
不一致またはcache missでは状態遷移せず停止し、新comment、lease、executor、再計算を起動しない。

`question_family_digest`はrecipe suppliedのID、名前、digestを信用せず、trusted verifierがimmutable source
model/policy/calibratorとreference AST node、canonical target-decision/population/metric registry digestから導出する。
recipe表示名、transform、threshold、cohort、releaseは影響させず、同じlineageのaliasは同一digestへ解決する。
既登録canonical familyへのrecipe追加はmerge eligibility判定前に拒否する。B-tierは1 familyにつき
lifetime recipe 1件、new execution 1件とし、隣接thresholdや別transformを別recipeとして反復しない。

PR #40の一件限定designは初期recipeのsource evidenceに限る。generic catch-all禁止、recipeごとのnew
kind、prepare+run+ACKという同designの将来案は、このroot amendmentに基づく別のhuman-reviewed
generic implementation contractが置き換える。PR #40 bytesは不変・non-authorityであり、新contractの
mergeとcutover前にgeneric lane authorityとして扱わない。

run scope seal後に作成されたcommentだけを受理し、durable ledgerの単一transaction/CASでcomment ID、
semantic/exact subjectのprovisional reservation、approval receiptを記録する。receiptはexpected/actual old/new
global/subject headsをbindし、comment IDは永久consumeするが、phase leaseはこのCASで発行しない。各phaseの
直前にshared G2だけがpredecessor receiptとremote再検証を条件に、run/recipe/replica/phase/attempt別の
domain-separated one-shot leaseを発行・consumeする。cross-use、replay、self-issue、capability unionを拒否する。

run scopeへbindしたlane、recipe、catalog、APPROVERS digestはbit-identical、ACTIVE、unrevokedでなければならない。
driftをdecision lease consume前かつcandidate/result/odds/payoff/settlementへ一度もaccessする前に検出した場合だけ、
authenticated atomic abort CASで旧runを`INVALID` terminal化し、global/subject headsを進め、発行済み未consume
leaseを全てrevoke/tombstoneし、旧approval/predecessor receiptを永久に不適格化してからsubject reservationを
解除し、新scopeと新commentへ置換できる。comment IDは解除しない。decision lease consume、semantic/exact subjectの
永久consume、question-family execution count加算、authenticated irreversible receipt発行をcandidate mount直前の
1つのatomic global/subject-head CASで行い、candidate mountはそのreceiptを再検証する。それ以降のdrift、crash、
contract failureは`INVALID` terminalで、新scope、retry、置換実行を認めない。future approval receiptまたはlease ID/digestをrun scopeへ入れず、post-approvalの全
receipt/leaseが逆向きにfrozen run digestをbindする。unrelatedな後続global appendだけでは、認証済みsubject
receipt chainを無効にしない。

pre-grant resolverは署名済みcatalog metadata/manifestだけを読み、candidate/settlement blobのmountやrow readを
行わない。ordered race-setはprepublished signed manifest digestから取得し、candidate/settlement contentは
authenticated irreversible receiptの再検証後にだけmount/readする。

lifecycleは上記の隣接遷移だけを許し、各nonterminalから`INVALID`へのfail-closeを許す。self transition、state
skip、terminalからの復活を禁止する。scope seal、approval、lease issue、irreversible start、result seal、completion、
fail-closeの各遷移はrun state、global/subject heads、対応receiptを1つのatomic CASで更新する。不可逆CASは
`RND_LEASED -> RND_RUNNING`も含み、state/head/receiptのsplit-brainを拒否する。

shared G2 durable ledgerをsole live authorityとする。lane専用local/separate backendを作らず、legacy
event、global grant-ID、全terminal/nonterminal subject headの完全移行、global/subject head両方のatomic CAS、
old-writer fence後のsecond remote compare、atomic cutover、authenticated receiptを要求し、dual writerと
local file、SQLite、worktree、branch、process-memory authority fallbackを禁止する。authenticated external
monotonic checkpoint/witnessでbackup restore、rollback、fork、stale headを検出し、検出時はlaneを停止して
新規leaseを発行しない。
run approvalは既存`github_backed_approval_evidence_v1`のidentity、ancestry、approver、comment evidence trust
semanticsだけを継承し、ordinary registry/prepare/run/shadow event mechanicsは継承しない。このrouteに限りproposal base commitの役割を
run-scope base commitが担い、APPROVERSをGitHub上のそのcommitから取得する。それ以外のtrust semanticsは
変更しない。repository、base branch、run-scope base commit、
verified current main、compare URL/status、merge-base、APPROVERS blob/content digest、verification time、および
comment ID、Issue、URL、author login/type、body、keyword、run digest、body SHA-256、created_at、updated_atを
個別に保存・再検証する。

resultは`evidence_purpose_class=DIAGNOSTIC_NONPROMOTION`とsource authority classを分離して保持する。
初期recipeのsource authority classは`B_LOCAL_HASHED`である。resultは`confirmatory=false`、`promotion_eligible=false`、
`score_credit=0`、`formal_buy=false`、`send_order=false`、`stake=0`に固定する。許可するoutcomeは
A-tierが`NO_DECISION_EFFECT|INVALID`、B-tierが`NO_DECISION_EFFECT|DIRECTIONAL_EFFECT|INVALID`
のみである。positive ROIや方向差を
ordinary scorecardへ入力したり、`APPROVED_FOR_SHADOW`、production、BUYへ遷移させたりしない。
採用を検討する場合は、新しい通常strategy proposal、75点以上、untouched evidence、通常の
prepare/run/shadow承認へ戻る。

## 6. Ordinary strategy lifecycle

```text
BLOCKED_SCORE

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

全状態から契約違反時に`INVALID`へ移行できる。`INVALID`はterminalである。

Registryはappend-only event logとする。`APPROVED_FOR_PRODUCTION`、merge承認、BUY承認はResearch OS v1に存在しない。

### 6.1 GitHub-backed approval

承認commentは次のexact形式とする。

```text
APPROVED_TO_PREPARE <proposal_scope_digest>
APPROVED_TO_RUN <run_scope_digest>
APPROVED_FOR_SHADOW <review_digest>
```

- 対象repositoryを`kazuponbaseball-cell/keiba_ai_project`、base branchを`main`
  に実装定数として固定し、proposalに固定したbase commitとの一致を検証する。
- GitHub read-only APIから`refs/heads/main`のcurrent head SHAを取得する。
- proposalのbase commitからcurrent main SHAへのGitHub compareが`ahead`
  または`identical`で、merge-base SHAがproposalのbase commitと一致する場合だけ
  main ancestorとして認める。
- author loginはGitHub上のproposal base commitから取得した
  `research/APPROVERS.json`に含まれなければならない。
- ローカルworktree、ローカルgit object、`refs/remotes/origin/main`、
  experiment branch上のallowlistを承認根拠として信用しない。
- actor typeが`User`でない、またはCodex/bot/automation loginなら拒否する。
- registryへrepository、base branch、verified current main SHA、verified base
  commit、compare URL/status、merge-base SHA、APPROVERS blob SHA、APPROVERS
  content SHA-256、verification timeを保存する。
- 全transition前にverified current mainの`research/REGISTRY.jsonl`を取得し、local snapshotが
  remote contentとbyte-for-byteで完全一致することを要求する。path/ref、blob SHA、content SHA-256を
  eventへ保存し、append直前にmain headが不変であることも再検証する。1 eventを人間mergeしてbranchを
  refreshするまで次eventを作れず、stale worktreeがterminal/consumed grantを無視できないようにする。
- registryへcomment ID、Issue番号、URL、author login/type、body、approval
  keyword/digest、body SHA-256、created_at、updated_atを保存する。
- prepare/run/shadowの新規grantは、同じregistry全体で未使用の異なるcomment
  IDを使う。後続transitionで同じ証拠を再検証することは新規grantではない。
- `PREPARING`前はprepare、`APPROVED_TO_RUN`前はprepare、`RUNNING`前は
  prepareとrun、`APPROVED_FOR_SHADOW`前はprepareとrunをread-only GETで
  再検証する。
- GitHub確認不能、base ancestry/allowlist検証不能、comment欠落・編集・削除・
  再利用、unauthorized authorはfail-closeする。
- CIは外部通信せずfixtureまたはinjected providerを使う。
- `research/STATE.yaml`のreconciliation snapshotで観測した`main` commit
  `1eaf364571bd8b9fd27f7de657ce295b563b3f1f`には
  `research/APPROVERS.json`が存在し、GitHub Contents APIからblob SHAと
  content SHA-256を取得済みである。これは固定された観測証拠であり、動的な
  current mainを表さない。ただし、実運用GitHub providerを用いた、実Issue commentによる
  approval transition E2Eは未確認である。
- モデル監査基準commit `288dff5e86385908281428d5ed4f077625a43e4b`
  には`research/APPROVERS.json`がないため、そのcommitをproposal baseとする
  承認は引き続きfail-closeする。

### 6.2 Canonical scope digest

Markdownは承認scopeの正本ではない。canonical JSONをUTF-8、key sort、
compact separator、Unicode保持、NaN/Infinity禁止でserializeし、
SHA-256化する。setとして扱うproposal listは重複禁止・Unicode順sortし、
execution command listは順序を保持する。

proposal scopeは最低限、experiment ID/title、仮説/null、作用機序、
対象母集団、in/out scope、expected paths、raw sources、data as-of、
allowed/forbidden columns、lineage/hash、chronological fold、fold path/hash、
purge/embargo、primary metric、required effect、rejection gate、stop、
compute budget、variant/threshold上限、base commit、score、安全flagを含む。

run scopeはproposal scope全体とdigestに加え、exact execution commit、
config hash、data/input manifest hash、fold manifest hash、runner-universe
manifest hash、dependency/environment manifest、seed、exact commandsを含む。

承認後にproposal/run scope、実行commit、manifest hash、commands、安全flagが
変わった場合は`RUNNING`へ進めない。新しいexperiment IDまたは再承認を
必要とする。実行後に生成されるresult、candidate digest、price join artifact
はrun scopeから分離してappendする。
hash-bound lifecycle/manifest以外のuncommitted・untracked pathも拒否し、
worktree codeをexact execution commitの一部とみなさない。

## 7. Evidence policy

- Tier A: audited commitに含まれるcode/config/docs。
- Tier B: local-only/ignored artifact。暫定証拠で、再現するまでpromotionに使わない。
- Tier C: 文書化された過去claim。背景情報で、採用根拠に使わない。

positive/negative resultを同じ基準で登録する。pooled平均だけ、1 foldだけ、小標本の極端なROI、threshold後付け、大配当集中は採用根拠にならない。

## 8. 即時停止条件

- Top3 massまたはwide mass契約違反
- chronology、race overlap、as-of、lineage違反
- candidate freeze前のmarket/payoff情報利用
- freeze前後のcandidate digest不一致
- 正式BUY、stake、order、production control pathへの接続
- 通常strategyの75点未満、または各routeで必要なGitHub承認証拠なし・承認scope外の実行
- GitHub main/compare/base-commit allowlist検証不能または不一致
- approval commentの編集・削除・grant ID再利用、先行承認の再検証失敗
- proposal/run digest、execution commit、config/data/fold/commandの変更

## 9. Research OS v1の完了条件

PR #2は、ガバナンス文書、state、scorecard、experiment/PR/issue template、GitHub-backed二段階承認、canonical proposal/run digest、append-only registry CLI、確率contract checkerとtestを提供して`main`へmergeされた。PR #3として記録するpost-merge reconciliationはstateと文書上のimmutableなGitHub snapshotだけを整合させる。既存baselineの再現、共通outer-fold manifest、odds-free candidate generator、正式BUYのproduction側hard-stop証明は後続の承認済み研究課題であり、このreconciliationでは実装しない。
