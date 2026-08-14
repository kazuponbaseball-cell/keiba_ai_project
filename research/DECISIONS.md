# Research OS v1 Decision Log

- 基準日: `2026-07-30`
- 基準コミット: `288dff5e86385908281428d5ed4f077625a43e4b`
- 形式: append-only。過去の決定を上書きせず、変更時は新しいDecision IDを追加する。
- 証拠階層: `research/STATE.yaml` の `A_HEAD` / `B_LOCAL_HASHED` / `C_WORKTREE` / `D_DOC_CLAIM`

## D-001 — 監査対象をGit基準線とローカル成果に分離する

- Decision: `ACCEPTED`
- Evidence: `A_HEAD`, `B_LOCAL_HASHED`, `C_WORKTREE`, `D_DOC_CLAIM`

Git管理されたcode/configだけを再現可能な基準線とする。ignored output、未追跡script、dirty差分の数値は暫定証拠であり、clean checkoutで再現するまでpromotionに使わない。tracked docs上の過去ROIは背景情報に限定する。

## D-002 — 現行baselineは監査基準線であり、再現済みchampionではない

- Decision: `DOCUMENTED_NOT_REPRODUCIBLE`
- Evidence: `A_HEAD`

`SimpleRaceRanker`のridge baseline、`config/baseline_features.json`、単一recent-20% holdoutを現状のcommitted baselineとして記録する。ただし次の理由でLevel 3の比較championとは認定しない。

- standard splitはnested chronological outer OOSではない。
- `data/` ignore規則により `src/data/loaders.py` が未追跡だが、48 tracked Python filesがimportする。
- data/model/output、dependency lock、test、CI、immutable manifestがbase commitにない。

後続実験は、clean clone replayと共通fold manifestを先に承認・実装する。

## D-003 — Top3集合softmaxをcanonical probabilityとする

- Decision: `LOCKED_CONTRACT`
- Evidence: governance decision + `B_LOCAL_HASHED`

Top3は非順序集合softmaxを維持し、各raceの集合確率合計を1、そこから周辺化する全wide pair確率合計を3とする。許容誤差は `1e-10`。違反runは `INVALID` とし、ROIを読まない。

ローカルのordered Top3 artifactは5,336 races / 2,117,093 setsで契約を満たしたが、主要実装とartifactが未追跡なので、HEADの保証には昇格させない。

## D-004 — Odds-free candidateとchronological outer OOSを採用する

- Decision: `LOCKED_CONTRACT`
- Evidence: governance decision

候補の生成、除外、順位、tier、tie-break、coverage、abstentionにodds、人気、market probability、払戻、ROI、その派生値を使用しない。分割は `train < validation < calibration < outer test` とし、未来・事後情報、race overlap、outer result由来の調整を禁止する。

committed production builderはmarket-awareであるため、odds-free research candidate generatorとして流用しない。本決定はproduction builderを変更する承認ではない。

## D-005 — Hypothesis score 75未満を自動実行しない

- Decision: `LOCKED_GATE`
- Evidence: governance decision

100点scorecardで75点未満は `BLOCKED_SCORE` とする。75点以上でも人間の `APPROVED_TO_RUN` がなければ実行しない。scoreは実行優先度であり、shadowまたはproduction採用承認ではない。

## D-006 — Local Top3 comparatorはresearch-onlyを維持する

- Decision: `NO_PROMOTION`
- Evidence: `B_LOCAL_HASHED`
- Sources:
  - `outputs/analysis/umaren_wide_rebuild_v1/top3_set_m0_v1/summary.json`
  - `outputs/analysis/umaren_wide_rebuild_v1/top3_set_m1c_m1a1_combined_v1/summary.json`
  - `outputs/analysis/umaren_wide_rebuild_v1/ordered_top3_contract_v1/summary.json`

M0の平均set NLLは約4.701877、M1C+M1A1 varianceのweighted NLLは約4.667477だった。確率契約もlocal artifact上はpassした。ただしrepeated development OOS、未追跡実装、upstream `ai_score` lineage未証明のため、モデル差し替え、候補変更、formal BUY再開を認めない。

## D-007 — 固定wide policyに安定したROI優位は未確認

- Decision: `NO_STABLE_PROFITABILITY`
- Evidence: `B_LOCAL_HASHED`
- Source: `outputs/analysis/umaren_wide_rebuild_v1/wide_fixed_policy_stability_audit_v1/summary.json`

| Policy | Overall ROI | Decision |
|---|---:|---|
| candidate_all | 94.65% | NO_STABLE_PROFITABILITY |
| primary_confidence | 97.22% | NO_STABLE_PROFITABILITY |
| sensitivity_confidence | 98.34% | NO_STABLE_PROFITABILITY |
| abc_guard | 104.08% | POSITIVE_BUT_UNSTABLE |

`abc_guard`はthreshold sensitivityとfold calibrationを通過していない。strict T-3 + final quoteは0/150、Grade-O featureは0/1,348である。value model fit、ROI threshold tuning、formal BUY再開を行わない。

## D-008 — Pair rerankerを棄却する

- Decision: `REJECTED`
- Evidence: `B_LOCAL_HASHED`
- Source: `outputs/analysis/umaren_wide_rebuild_v1/wide_pair_reranker_v1/summary.json`

posterior-only、interaction、floor/stabilityを含む3 variantはいずれもadoption gateを満たさなかった。全variantでcluster bootstrapの下側安定性を満たさず、pooled top1 hitも改善しなかった。同じdataと同じfeature定義の名称変更による再探索を禁止する。

## D-009 — Axis-conditioned partner modelを棄却する

- Decision: `REJECTED`
- Evidence: `B_LOCAL_HASHED`
- Source: `outputs/analysis/umaren_wide_rebuild_v1/axis_conditioned_partner_model_v1/decision.json`

一部foldの改善はあったがfold間で安定せず、cluster bootstrap gateを通過しなかった。candidate/action calibratorへ接続しない。新データまたは事前登録した別機序がなければ再実行しない。

## D-010 — Existing horse-condition proxy incrementを棄却する

- Decision: `REJECTED_FOR_ADOPTION`
- Evidence: `B_LOCAL_HASHED`
- Source: `outputs/analysis/umaren_wide_rebuild_v1/horse_condition_factor_oos_v1/summary.json`

combined minus baselineのlogloss差は `+0.000216`で、安定したincrementを示さなかった。外厩historyは候補期間とのoverlapが0で未検証である。既存proxyの再探索は閉じるが、新しいas-of sourceとoverlapを持つ外厩情報は別仮説としてのみ提案できる。

## D-011 — Layoff/return pair asymmetryを棄却する

- Decision: `REJECTED`
- Evidence: `B_LOCAL_HASHED`
- Source: `outputs/analysis/umaren_wide_rebuild_v1/layoff_return_pair_asymmetry_oos_v1/summary.json`

delta loglossは `+0.002504`、delta Brierは `+0.000688`、改善期間は1/3だった。既存feature blockを候補または確率modelへ接続しない。

## D-012 — State/growth/connections blocksをresearch-onlyに留める

- Decision: `REJECTED_FOR_ADOPTION`
- Evidence: `B_LOCAL_HASHED`
- Source: `outputs/analysis/umaren_wide_rebuild_v1/state_growth_connections_screen_v1/summary.json`

performance content、body/workout state、jockey/trainer contextの全blockがadoption gateに失敗した。fold改善数はそれぞれ2/4、1/4、1/4で、pooled Brierも悪化した。現行model・candidateへ接続しない。

## D-013 — Combined race-mechanics action residualを棄却する

- Decision: `REJECTED`
- Evidence: `B_LOCAL_HASHED`
- Source: `outputs/analysis/umaren_wide_rebuild_v1/race_mechanics_action_residual_v1/summary.json`

pooled delta loglossは `-0.000328`だったが、改善は1 outer foldだけで、cluster bootstrap q90は `+0.002159`だった。C4 probability shapeは観察用、pair fragilityはwatch-onlyとし、combined action incrementをpromotionしない。

## D-014 — Sparse-history shrinkageを現行modelへ接続しない

- Decision: `NOT_ADOPTED`
- Evidence: `B_LOCAL_HASHED`
- Source: `outputs/analysis/umaren_wide_rebuild_v1/sparse_history_feature_shrinkage_v1/summary.json`

`ability_floor_score_5`、`ability_stability_score_3`、`recent_weighted_score_3`、`condition_adjusted_recent_ability_score`のshrinkageは欠損を一部回収したが、4項目すべてrawより単純分離が弱かった。true source lookback countも未証明であり、formal eligibleではない。

## D-015 — Recent regimeは診断に留める

- Decision: `DEFERRED_INSUFFICIENT_SAMPLE`
- Evidence: `B_LOCAL_HASHED`
- Source: `outputs/analysis/umaren_wide_rebuild_v1/recent_regime_prequential_v1/summary.json`

recent repaired racesは36、venue-date clusterは3、strict regime evidenceは0で、watch thresholdの100 races / 8 clustersに未達である。再学習、候補変更、threshold変更を認めない。

## D-016 — Fixed-candidate value policyをblockする

- Decision: `BLOCKED_DATA`
- Evidence: `B_LOCAL_HASHED`
- Source: `outputs/analysis/umaren_wide_rebuild_v1/fixed_candidate_value_policy_v1/summary.json`

strict T-3 rowsは0、minimumは150である。model fit、threshold tuning、ROI optimizationは実行しない。候補freeze後のstrict prospective dataが承認済み手順で蓄積されるまで再開しない。

## D-017 — Formal BUY停止はResearch OS内で強制し、本番状態は未認証とする

- Decision: `RESEARCH_STOP_LOCKED_PRODUCTION_UNCONFIRMED`
- Evidence: `A_HEAD`, `C_WORKTREE`

Research OS artifactは常に `formal_buy=false`、`send_order=false`、`stake=0` とし、production BUY pathを実行しない。本PRでは予測、候補、value、BUY codeを変更しない。

一方、audited HEADのproduction builderはmarket-aware BUYを生成できる。dirty worktreeでは既定停止が観測され、local auto-purchase設定もpaper-onlyだが、未コミットで再有効化可能である。したがって「最後に観測されたlocal運用は停止」だが「HEADおよびlive processの停止は未認証」と決定する。version-controlled hard-stopとruntime確認は別の人間承認案件とする。

## D-018 — Historical high-ROI claimsを採用根拠にしない

- Decision: `CONTEXT_ONLY`
- Evidence: `D_DOC_CLAIM`
- Source: `docs/EXTERNAL_AI_PROJECT_BRIEF.md`

記載されたROI 375.9%〜716.4%は、小標本、上振れ、過学習リスクを文書自身が警告している。凍結manifestと再現手順がないため、現在baseline、hypothesis score、promotion gateに利用しない。

## D-019 — 人間承認をGitHub Issue commentへ結び付ける

- Decision: `LOCKED_GATE`
- Evidence: governance decision + tracked implementation

`--human-approved`やcaller指定`--actor`は人間身份の証明として扱わない。
承認はGitHub Issue上のexact commentだけを証拠とし、proposalのbase commitが
`origin/main` history上にあることを検証したうえで、そのcommit上の
`research/APPROVERS.json`でauthor loginを検証する。experiment branch上の
allowlist変更、allowlist外user、Codex、bot、automation actorを拒否する。

registryへcomment ID、URL、author、created_at、updated_at、body SHA-256、
approval type、承認digestを保存する。`APPROVED_TO_RUN`から`RUNNING`へ
進む直前に同じcommentをread-only再取得し、author、body、digest、
updated_atの変更またはGitHub取得不能時はfail-closeする。

`APPROVED_TO_PREPARE`は実装とsynthetic fixture testだけを許可する。
実データ学習、backtest、outer OOS、ROI計算には、実装後scopeに対する
別の`APPROVED_TO_RUN`を必要とする。shadowも`review_digest`に対する
別の`APPROVED_FOR_SHADOW`を必要とする。

## D-020 — 承認scopeの正本をcanonical JSON digestとする

- Decision: `LOCKED_CONTRACT`
- Evidence: governance decision + tracked implementation

Markdownを承認scopeの正本にしない。proposal/run scopeはcanonical JSONを
UTF-8、key sort、compact separator、Unicode保持でserializeし、SHA-256化する。
setとして扱うproposal listはsorted uniqueとし、execution command listは
順序を保持する。

proposal scopeは仮説、null、作用機序、対象母集団、in/out scope、expected
paths、data/as-of、column/lineage、fold、metric/gate/stop、budget、
variant/threshold上限、base commit、score、安全flagを固定する。

run scopeはproposal scope全体とdigestに加え、exact execution commit、
config/data/fold/runner/environment manifest hash、seed、exact commandsを
固定する。`RUNNING`直前にdigest、current commit、実ファイルhashを再検証する。
hash-bound lifecycle/manifest以外のuncommitted・untracked pathも拒否する。
承認後の変更は新しいexperiment IDまたは再承認を必要とする。

実験結果、candidate digest、price join後artifactは実行前承認scopeへ混ぜず、
append-only result evidenceとして保存する。

## D-021 — D-019の承認信頼境界をGitHub remote検証と単一用途commentへ置換する

- Decision: `LOCKED_GATE`
- Evidence: governance decision + tracked implementation and fixture tests
- Supersedes: D-019の`origin/main`、local `git show`、run承認だけの再検証に関する部分

D-019は履歴として保持するが、承認可否にローカルremote-tracking ref、ローカル
git object、worktree上の`research/APPROVERS.json`を使わない。対象repositoryを
`kazuponbaseball-cell/keiba_ai_project`、base branchを`main`に固定する。
GitHub read-only APIから`refs/heads/main`のcurrent head SHAを取得し、proposalの
base commitからcurrent main SHAへのcompareが`ahead`または`identical`で、
merge-base SHAがproposalのbase commitと一致する場合だけancestor性を認める。

allowlistはGitHub上のproposal base commitから
`research/APPROVERS.json`を取得して構築する。registryにはrepository、base
branch、verified current main SHA、verified base commit、compare URL/status、
merge-base SHA、APPROVERS blob SHA、APPROVERS content SHA-256、verification
timeを監査証拠として保存する。GitHub取得不能、responseのrepository/base
branch/base commit不一致、compare不一致、contents欠落・不正JSONはfail-closeする。

prepare、run、shadowの新規grantは、同じ`REGISTRY.jsonl`全体で未使用の異なる
comment IDを必要とする。後続transitionで同じ証拠を再検証することは新規grant
ではない。`PREPARING`前はprepare、`APPROVED_TO_RUN`前はprepare、`RUNNING`前は
prepareとrun、`APPROVED_FOR_SHADOW`前はprepareとrunをGitHubから再取得する。
comment ID、Issue番号、URL、author login/type、body、approval keyword/digest、
body SHA-256、created_at、updated_atの変更、comment削除、grant ID再利用は
fail-closeする。

監査時点のpre-merge `main` `288dff5e86385908281428d5ed4f077625a43e4b`
には`research/APPROVERS.json`がないため、そのcommitをbaseとする承認は
fail-closeする。PR #2 merge後のmain commitに対するend-to-end取得・検証は
未確認事項として残す。本決定はproduction、merge、BUY、注文の承認を追加しない。

## D-022 — PR #2 merge後のrepository stateをreconcileする

- Decision: `POST_MERGE_STATE_RECONCILED`
- Evidence: GitHub read-only API snapshot + merged PR #2 metadata
- Observed at: `2026-08-01T13:43:58+09:00`

PR #2は人間判断によりmergeされ、確認時点のGitHub `refs/heads/main`は
`1eaf364571bd8b9fd27f7de657ce295b563b3f1f`を指している。このobserved main
snapshotと、既存モデル・production状態を監査した基準commit
`288dff5e86385908281428d5ed4f077625a43e4b`は役割が異なるため、
`research/STATE.yaml`で`reconciled_through_main_commit`と
`model_audit_baseline_commit`に分けて管理する。top-level statusはPR #3の
review状態ではなく、永続的な`RESEARCH_OS_V1_FOUNDATION_MERGED`とする。

モデル監査基準commitからobserved main commitへのGitHub compareは`ahead`、
merge-baseはモデル監査基準commitと一致した。observed main commit上の
`research/APPROVERS.json`はGitHub Contents APIで存在を確認し、blob SHA
`c973f7d83de78cc0eea09ef6a240e99e4512937e`、decoded content SHA-256
`ac14971c8d8f6f4502c30c6b6434da9ce061009955461248c95363b36ee137b5`を記録した。

これはallowlist fileのremote存在確認であり、実運用GitHub providerを用いた、実Issue commentによる
approval transition E2Eは未確認である。`APPROVED_TO_PREPARE`、
`APPROVED_TO_RUN`、`APPROVED_FOR_SHADOW`の全transitionでfail-close境界を維持する。
モデル監査基準commitには同fileが存在しないため、そのcommitをproposal baseと
する承認も引き続きfail-closeする。

PR #3のbranchとopen・Draft・未merge状態は、観測日時を伴うreconciliation
snapshotへ分離する。これらをResearch OS本体のlive stateとして扱わず、PR #3の
未確定な将来のmerge commit SHAも記録しない。

このreconciliationはstateとgovernance/template文言だけを更新する。実験、ROI
仮説、production、予測モデル、候補選択、value、BUY、stake、注文、通知、
credentialを変更せず、merge判断はDraft PR上の人間に残す。

## D-023 — 競馬仮説とインフラ／安全性変更のgateを分離する

- Decision: `LOCKED_GATE_ON_MAIN_MERGE`
- Evidence: governance design + synthetic fixture tests

`HYPOTHESIS_SCORECARD`は競馬上の独立情報、作用機序、outer OOS失敗根拠を評価する
ROI仮説専用gateとして凍結する。Research OS control-plane、contract、schema、
synthetic adapterへ架空の競馬根拠を割り当てず、明示的な
`infrastructure_safety_v1` contractを並立させる。

infra gateは数値scoreを合算しない。machine-readable policyの全hard check、
path/capability firewall、synthetic-only run、canonical proposal/run digest、
base-to-execution commit diff、構造化command、GitHub-backed prepare/run承認をすべて
満たした場合だけ既存lifecycleの`PROPOSED`から`REVIEW_REQUIRED`までを使える。
real-dataと`APPROVED_FOR_SHADOW`は常に拒否する。

既存ROI proposal/run、queue/event schema v2、canonical digestはmigrationも再serializeも
しない。infra queue/eventはschema v3とし、approval comment IDの単一用途namespaceと
先行承認再検証はschemaをまたいで共有する。profile名だけの偽装、ROI/model/data/pathの
混入、自由形式shell、credential/network/production/BUY/order/notification capability、
root-of-trustの自己変更はhard failureであり、scoreや人間run commentで救済できない。

この決定を含む初回gate導入はgate自身によるbootstrapを行わない。Draft PR上の人間reviewと
人間mergeを唯一のactivation境界とし、mainへmergeされるまで後続infra proposalの承認根拠に
使わない。本決定はproduction、merge、正式BUY、注文、通知の承認を追加しない。

## D-024 — legacy ROI runの実行種別をfail-closeする

- Decision: `LOCKED_SECURITY`
- Evidence: canonical run scope inspection + legacy digest golden tests

既存のunversioned ROI run scopeは`execution_kind`をcanonical JSONへ含めず、registry CLIの
呼出時引数だけで`synthetic`と`real-data`を選べる。このため、synthetic想定で承認された
scopeを同じdigestのままreal dataへ切り替えられる境界不備として扱う。

legacy proposal/runのbyte列とdigestを変更せず、現行contractではreal-data `RUNNING`を
fail-closeする。synthetic lifecycleと既存artifactのread/監査互換性は維持する。
real-dataを再び許可するには、execution kindとcapability matrixをproposal/run scopeへ
hash-bindするversioned ROI contract、migration境界、negative testを別のDraft governance
PRで導入し、人間がreview・mergeしなければならない。

## D-025 — infra ledger・material・commandを非迂回境界へ固定する

- Decision: `LOCKED_SECURITY_ON_MAIN_MERGE`
- Evidence: adversarial registry/path/source/fixture tests

infra approvalの単一用途namespaceはsymlink/junctionでないexact `research/REGISTRY.jsonl`だけとし、alternate
ledgerを拒否する。base→execution commitとexecution commit→worktreeでregistryの
append-only prefixを検証し、append時はprocess lock、snapshot CAS、flush、fsyncを行う。
全transition前にはverified current mainのregistry blobをGitHubから取得し、local snapshotとの
byte-for-byte完全一致を要求する。append直前にmain headを再取得して不変を確認する。1回のappendは
pending candidateだけを作り、人間mergeとbranch refresh前には次transitionも権限も許可しない。
同時writer、stale main、過去eventのrewrite/deleteを検知した場合はeventを書かずfail-closeする。

run materialはASCII canonical path、`.example.json` config、`research/synthetic/`の
1 MiB以下のprovenance envelopeへ限定する。symlink/junction、秘密らしい値、row-level
real-data shapeを拒否し、全materialをexecution commitのblobへ固定してdirty/untrackedを
許可しない。execution commitはproposal baseの子孫だけを認める。current interpreterの
path/hash/versionをenvironment manifestへ固定する。変更Pythonのexact Git blobはASTで検査し、
pure allowlist外import、network、credential、subprocess、production系import、dynamic call、
forbidden symbolを拒否する。command templateはそのinterpreterを`-B -I -S`で使い、repository-root
cwd、継承environmentなし、空environment、proposal budget由来timeout、write path 0をscopeへ固定し、
自由shell、PATH選択、site customizationを承認scopeへ入れない。

このgateは自動executorではないため、infra eventの`automatic_execution_allowed`、
`preparation_authorized`、`execution_authorized`は全状態でfalseとする。v1 policyはmain merge後に
in-place変更せず、将来のpolicyは新gate kind/versionと
別pathで追加して既存v3 scopeのread/`INVALID`互換性を維持する。

通常PR CIは`tests/research/test_*.py`を自動実行するため、このdirectoryをinfra gateの
変更対象にすると`APPROVED_TO_RUN`前の実行になる。よって`tests/research/`とworkflowを
root-of-trustとして固定し、後続infra testはCI自動探索外の`research/infra_tests/`だけに置く。
そのtestはhash-bound run scopeとstructured commandへ固定するが、別の人間review済み
executor/authority verifierが導入されるまで実行しない。初回bootstrapのgovernance testは
gate activation前の人間依頼に基づく別境界である。

## D-026 — ROI reproduction v2 G1を非権限compiler境界として宣言する

- Decision: `CONTRACT_COMPILER_ONLY_NO_AUTHORITY_ON_HUMAN_MERGE`
- Evidence: merged PR #37 design artifacts + current legacy v2 / infra v3 compatibility inspection

`roi_reproduction_audit_v2`のG1は、schema、別pathのpolicy、canonical serializer、state
validator、非権限event compiler、CI自動探索外synthetic fixtureだけを将来実装可能な上限とする。
このgovernance変更はpure compiler、policy、schema、synthetic-only governance testを実装するが、
provider、writer、authority verifier、executorを実装せず、repository/ledgerへproposal、queue、
run scope、registry eventを生成・追記しない。G1のmodeは
`CONTRACT_COMPILER_ONLY_NO_AUTHORITY`である。artifact contractは
`execution_kind=historical_reproduction_v2`をbindするが、現在の有効runtime execution kindはnone、
全authorityはfalse/0、結論は`EXECUTION_FORBIDDEN`である。

catalog publisher/attester、authority verifier、durable runtime ledger、one-shot lease、supervised
executor、real-data/model/training/historical replay capabilityはG2の責務であり、現在未実装である。
G2はG1 declarationのhuman merge後に別のhuman-owned Draftでreview・mergeしなければならず、
G1の成功、design、score、人間commentからG2 authorityを推論しない。

既存legacy ROI queue/event schema v2とinfra queue/event schema v3のwriter、canonical digest、
transition、approval semanticsを変更しない。現行`update_registry.py`はschema v4をdispatchせず、
`research/schemas/roi_reproduction_registry_event_v4.schema.json`とcanonical line compilerは
validation用であり、live writerまたはauthority sourceではない。
prepare/run/resultの3 action flowは現在activationされておらず、既存approval commentは
G1/G2/Aの開始、再開、grant、実行tokenにならない。

G1/G2のbootstrapは自身または`infrastructure_safety_v1`で承認できない。workflow、既存test、
approval verifier、registry writer、approver allowlist、scorecard、infra v1 policyをG1 compilerの
changed-path候補へ含めず、root-of-trust変更は人間reviewと人間mergeだけで有効にする。

## D-027 — ROI reproduction v2 G1のhuman merge事実を非権限状態へ整合する

- Decision: `G1_IMPLEMENTED_AND_HUMAN_MERGED_NO_AUTHORITY_RECONCILED`
- Evidence: GitHub PR #38 merge metadata + current-main ancestry + merged G1 policy/contracts

PR #38のG1 implementationは、GitHub actor typeが非botの人間ユーザー
`kazuponbaseball-cell`により2026-08-14T12:21:27Zにmainへmergeされた。merge commitは
`811ffd11bd80447f013c643b96c3eb8145916061`であり、観測時のcurrent main
`69a95a25a618e04fa73620f21a9010e78143f1eb`はその子孫である。このdecisionはmerge済みという
記述状態だけを整合し、PR #38のformal reviewDecisionが存在したとは主張しない。

G1の境界は引き続き`CONTRACT_COMPILER_ONLY_NO_AUTHORITY`である。G2は未実装、effective runtime
execution kindはnone、結論は`EXECUTION_FORBIDDEN`であり、provider、writer、authority verifier、
durable ledger、lease、executor、real-data、model、training、historical replay、ROI、shadow、
production、BUYのauthority/capabilityを追加しない。PR merge、CI成功、chat指示、このdecisionは
proposal、prepare、run、result、activation、grant、execution tokenではない。

`research/drafts/ROI_REPRODUCTION_GATE_V2_CONTRACT_MAP.design.json`はG1 testでself-amendment拒否対象と
され、policyのbootstrap exact-path inventoryにも含まれるnon-authoritative historical design snapshot
である。このdecisionは現行policyのsource-ref digestとの一致を主張せず、そのpre-merge status文字列を
current live statusまたはauthorityとして扱わない。frozen policy/schema/compiler/test、legacy writer、
registry、approver allowlist、scorecard、workflowは変更しない。G2と一件限定model-integrity laneの
root amendment/activationは、それぞれ別のhuman-owned PR、scope、grant、durable receiptを必要とする。
