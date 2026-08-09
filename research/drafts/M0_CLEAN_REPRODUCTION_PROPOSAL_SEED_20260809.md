# M0 clean reproduction gate — proposal seed

作成日: 2026-08-09

固定base commit: `b282cafb39435d15098bc409e76b4efaa6690f08`

状態: `DRAFT_NOT_CANONICAL / BLOCKED_SCORE / BLOCKED_PREREQUISITES`

この文書はM0再現基盤の設計seedであり、canonical proposal scope、experiment、registry event、GitHub承認証拠ではない。コード準備、synthetic test、実データ読取、学習、backtest、outer OOS、ROI計算を許可しない。

Machine-readable seed: `research/drafts/M0_CLEAN_REPRODUCTION_PROPOSAL_SEED_20260809.json`

## 結論

M0再現基盤は回収率を直接上げない。しかし、後続の改善がデータ差、fold差、実装差、環境差ではなく、事前登録したモデル変更によるものだと判定するための必須条件である。

回収率への因果経路は次の順序に限定する。

1. M0をclean checkoutから再現し、同一counterfactualを固定する。
2. M0とchallengerを同一の未使用prospective outer、runner universe、fold、price-blind入力でpaired比較する。
3. calibration期間までで候補規則を固定し、候補artifactをhash固定する。
4. その後に限り、評価器が価格・結果・払戻をjoinする。
5. 確率品質のgateを通過した場合だけROIを副次評価する。

M0再現だけで「ROI向上」「モデル昇格」「BUY再開」を主張してはならない。

## 現時点で確認できたreference

- `research/STATE.yaml`はM0を`B_LOCAL_HASHED / RESEARCH_ONLY_NOT_PROMOTED`として記録している。
- 記録値は4 chronological development folds、再利用済みOOS 5,336 races、平均set NLL 4.701877、top1 set hit rate 0.0827575、wide top1 hit rate 0.275505、reported mass error 0.0である。
- legacy scriptのhit-rate出力はpercentage pointsである。一方、STATE上の0.0827575/0.275505がどのartifact・単位から転記されたかは`PROVENANCE_UNIT_UNCONFIRMED`とする。`reported mass error=0.0`はlegacy codeで丸めたderived-wide mass errorであり、Top3 massやfull-universe完全性の独立証明ではない。legacy `avg_nll`はfold単純平均で、race-weighted NLLは別列であるため、STATE値の対応列をmanifestで確定する。
- 5,336 racesは既に繰り返し研究に使われているため、再現照合には使えるが、新しい性能証拠や採用証拠には使えない。
- local rootには`train_top3_set_m0_v1.py`があるがbase commitではuntrackedである。2026-08-09のread-only観測SHA-256は`5b91fb586c5d418f834081d8f64c4f81a46e9f3d20f041934f30d083d784f4b8`であり、authorityではない。
- local rootのuniverse builder `build_full_unordered_pair_universe_v1.py`もuntrackedで、観測SHA-256は`ac003fd12fdce76ffca5f4167cd7f562bea96de495722f8d2281ae709b6f0a74`である。このbuilderはmarket oddsとpayoffを同じpair artifactへ同梱するため、strict allowlistによる物理的分離が必要である。
- legacy builderは`is_refund=false`をhard-codeし、starter snapshotを`unknown_reference_only`、universeを`reference_derived`として扱う。`finish<=3`が3頭超または`finish<=2`が2頭超ならdead-heat/nonstandardとし、pair-count mismatch、result incomplete、label-mass mismatchもtraining対象外にする。取消・失格・降着は独自解決せずupstream状態を信用する。この挙動を再現照合と将来の安全なlabel policyで混同しない。
- legacy input、output、dependency lock、upstream `ai_score`の完全なOOF/as-of lineageはbase commitから再構成できない。
- legacy trainerは与えられたtriple rows上でrace-grouped softmaxを実装する。全`C(n,3)`になるのはbuilderまたはfallback生成が完全な場合だけで、外部Top3 CSVの完全性、重複、runner欠落をtrainer自身は検証しない。入力artifactの`ai_score`と`ai_rank`の由来がprice-blindであることも未証明である。
- legacy implementationはfold、L2 grid、temperature grid、feature listをscriptへ直書きし、Top3 inputの存在有無で処理を変える。専用config、dependency lock、model parameter、preprocessing parameter、full per-set probability、input/code/environment hashを保存しない。
- legacy trainerのdefault inputは`outputs/analysis/umaren_wide_rebuild_v1/full_unordered_pair_universe_v1/full_unordered_pair_universe.csv`と`full_top3_set_universe.csv`、default outputは`outputs/analysis/umaren_wide_rebuild_v1/top3_set_m0_v1/`である。いずれもclean cloneに存在しない。universe builderのheavy Top3生成は`--write-top3`と`--rewrite-top3-rows`の両方を必要とする。
- universe builderのdefault upstreamは`outputs/analysis/historical_reference_2023plus_prediction_detail_v1/prediction_detail_reference_2023plus.csv`と`data/processed/target/wide_payoffs.csv`である。trainer/builderの直接dependencyはstdlib、numpy、pandasだが、version lockはない。
- base commitでtrackedの`scripts/research/check_probability_contracts.py`（git blob `d90c87135789c6a76d16120211c0a772ed49933f`）は再利用候補だが、legacy M0はvalidatorが必要とするfull per-set probability artifactを出していない。

## 仮説とnull

仮説:

> commit、source/data lineage、runner universe、target、feature schema、fold、environment、seed、command、reference artifactをhash固定したM0 packageを作れば、clean checkoutからM0の構造契約と事前固定した数値許容差を再現でき、後続challengerとの差をモデル変更へ帰属できる。

Null:

> referenceの完全なlineageを固定できない、race/runner/target/foldが一致しない、または再現値が事前固定許容差を超える。この場合、M0再現は失敗であり、challenger比較とROI計算へ進まない。

競馬上の新規作用機序はない。これは戦略仮説ではなくreproduction gateである。

## Step A0 — preparation package

必要権限: 将来のhost-assigned experimentについて、GitHub上の有効な`APPROVED_TO_PREPARE <proposal_scope_digest>`。

許可する準備:

- clean checkoutでimport可能なtracked M0 implementationを作る。
- Python dependency/environmentをversion固定する。
- canonical JSON manifest schemaとfail-close validatorを作る。
- synthetic runner universeだけでset mass、wide mass、finite/range/duplicate、取消、同着、3頭未満、異常着順を検査する。
- price-blind allowlistとforbidden-column firewallをsynthetic fixtureで検査する。
- 既に記録済みのSTATE値と静的script以外の、未読canonical reference artifactを比較する前にreproduction acceptance toleranceを固定する。
- input fileの存在有無によるfallbackを廃止し、canonical input pathを1つに固定する。
- external triple universeを学習前にstrict validationし、全starter ID、`runner_count`、全`C(n,3)`、重複なしを証明する。
- probability validatorへ接続できる全`C(n,3)`のper-set artifactを、`race_id, horse_id_1, horse_id_2, horse_id_3, runner_count, top3_probability`のcanonical schemaで出力する。
- target cardinality、finish label、取消、同着、失格、降着、pair input uniquenessはprobability validatorと別のrunner/label validatorで検査する。

禁止する準備:

- real race/data/model/output artifactの読取またはhash計算
- 学習、backtest、historical replay、outer OOS、ROI計算
- odds、人気、market rank、払戻、ROIをcandidate/model inputへ入れること
- BUY、stake、order、通知、本番path、credential、external API
- 既存local scriptをそのままauthorityとして扱うこと

Step A0 deliverables:

1. tracked implementation and dependency lock
2. source/data/feature/target/fold/environment/runner/command/reference manifestのschemaとtemplate
3. price-blind feature allowlist and normalized forbidden-field rules
4. exact runner-universe and label contract
5. deterministic serialization contract
6. synthetic-only contract tests
7. numeric reproduction tolerance committed before any unread canonical legacy artifact is exposed

Stage Aのrun scopeへ実データhashを入れる方法もA0で契約化する。任意のlocal manifest、任意の`human-approved` flag、caller指定attesterは拒否する。許容候補は、GitHub-backedの専用`APPROVED_TO_ATTEST_MANIFEST <manifest_scope_digest>`に結び付いたversioned supervised providerだけである。manifest scopeはprovider kind/version、execution commit、code/environment hash、provider identity、attester identity、source object hash、created_at、exact返却schema、output digestに加え、repository/base branch/current-main compare/merge-base、APPROVERS blob/content hash、Issue番号/URL、comment ID/URL、author login/type、body/body SHA-256、keyword/digest、created_at/updated_atを固定し、raw row/path/secretを返さない。attestation comment IDはregistry全体で未使用かつprepare/run IDと異なり、run-scope freeze、`APPROVED_TO_RUN`、`RUNNING`直前に再取得・再検証する。attestation/hash不一致はfail-closeする。このkindとkeywordはまだ存在しないため`UNRESOLVED_BLOCKER`であり、A0の通常権限でreal artifactを読んだりhash計算したりしてはならない。

## Stage A — historical reproduction run

必要権限: Step A0をcommitし、canonical run scopeを固定した後の、別comment IDによる`APPROVED_TO_RUN <run_scope_digest>`。

Stage Aは旧5,336-race populationだけを使い、既存M0を再現する。性能探索ではない。variantは1、threshold searchは0、hyperparameter gridとtemperature gridはlegacy referenceどおりに固定する。

実行前に次をすべてhash固定する。

- exact execution commit
- input source snapshot and data lineage
- pair/top3 input artifact
- race and runner universe
- feature schema and upstream `ai_score` provenance
- target construction and abnormal-result handling
- four-fold race assignment manifest
- config, dependency lock, interpreter, platform, numeric libraries
- seed and exact ordered commands
- legacy reference artifacts and metrics

Primary判定は2つに分ける。

1. `clean_replay_determinism_digest_match`: 同一environment/run scopeを使う2つのclean checkoutで、canonical discrete manifest/universeとcanonical floating artifactのbitwise digestが完全一致すること。
2. `legacy_numeric_equivalence`: 未読canonical legacy artifactとの比較前にcommitした許容差で照合すること。これはreference digestがない限り`REPRODUCED` statusを付与しない。

受入条件:

- 4 folds、5,336 reference races、各raceのrunner集合、全`C(n,3)`、target set、canonical orderがreferenceと一致する。
- `train < validation < calibration < test`を各foldで満たし、partition間race overlapが0である。
- 各raceのTop3 set probability mass errorとderived wide mass errorがともに`<= 1e-10`である。
- NaN、Infinity、範囲外、重複set/pair、target cardinality不正が0である。
- 事前固定したlegacy numeric toleranceをすべて満たす。
- 同一run scopeを2つのclean checkoutで実行し、canonical prediction/set artifact digestが完全一致する。
- `formal_buy=false`、`send_order=false`、`stake=0`であり、production/BUY/order/notification差分が0である。

棄却・停止条件:

- reference hash、lineage、fold、environment、許容差のいずれかが未固定
- upstream `ai_score`がOOF/as-ofまたはprice-blindであると証明できない
- race/runner/target/fold不一致
- mass contract、finite/range/duplicate、label cardinality違反
- dirty/untracked input、code、config、environment、artifactへの依存
- numeric tolerance超過または非決定性
- price、market、payoff、ROIが時点を問わずmodel featureまたはcandidate/ticket processへ入り、選択・除外・順位・tier・weight・no-bet判断を変える
- current/outer resultがprediction、candidate、final-ticket、unit-notional、no-bet digest固定前にmodel/candidate/ticket processへ入る
- scope外path、network、credential、BUY、order、notification、本番side effect

legacy M0の信頼できるcanonical reference digestを確定できない場合、到達状態は`RECONSTRUCTED_NOT_REPRODUCED`であり、「M0再現成功」とは呼ばない。

停止時は`INVALID`または該当するfail-closed状態とし、ROIを計算・解釈しない。

## Legacy fold reference

以下はlocal untracked implementationから静的に読んだreferenceであり、canonical fold manifestではない。

| Fold | Train | Validation | Calibration | Reused test |
|---|---|---|---|---|
| fold1 | 2023-01-01..2023-12-31 | 2024-01-01..2024-03-31 | 2024-04-01..2024-06-30 | 2024-07-01..2024-12-31 |
| fold2 | 2023-01-01..2024-06-30 | 2024-07-01..2024-09-30 | 2024-10-01..2024-12-31 | 2025-01-01..2025-06-30 |
| fold3 | 2023-01-01..2024-12-31 | 2025-01-01..2025-03-31 | 2025-04-01..2025-06-30 | 2025-07-01..2025-12-31 |
| fold4 | 2023-01-01..2025-06-30 | 2025-07-01..2025-09-30 | 2025-10-01..2025-12-31 | 2026-01-01..2026-06-30 |

これらのtest期間は既に開封・再利用済みであり、将来のouter testではない。

## Legacy model recipe reference

以下はuntracked scriptから静的に読んだrecipe候補であり、authorityではない。Stage Aでは専用configへ移し、hash固定する。

- target: unordered tripleが`{runner | finish_num <= 3}`と完全一致する場合だけ`y_top3_set=1`
- runner transforms: race内`ai_score.rank(pct=True, ascending=True, method=average)`でmissingは0.5、missing `ai_rank`は`field_size`へfillしてstrength 0相当、分母不成立時は0.5、critical missing flag
- exact five features: `sum_primary_strength`, `min_primary_strength`, `sum_rank_strength`, `min_rank_strength`, `critical_missing_count`
- estimator: interceptなしのlinear race-group softmax、zero initialization、learning rate 0.18、train+validationの`np.std(ddof=0)`、zero stdは1、softmaxのutility差clipは[-50,50]
- L2 selection: `[0.0, 0.005, 0.02, 0.08]`を各trainで100 epochs fitし、validation NLL最小を選ぶ。同値ならlist先頭を選ぶ
- refit: selected L2でtrain+validationを140 epochs fitし、mean/std/weightsもtrain+validationから作る
- temperature: calibrationだけで`[0.20, 0.25, ..., 0.95, 1.00, 1.10, ..., 3.00]`からNLL最小を選ぶ。同値ならlist先頭を選ぶ
- learning RNG: なし。`20260713`は2,500件超のKendall近似samplingだけに使われ、model fit seedではない

`allowed_variant_count=1`はM0 reproduction pipelineの数を意味する。4個のL2と37個のtemperatureはlegacy再現用に事前固定したpipeline内部選択であり、新規variant/threshold探索ではない。selection partition、metric、tie-breakは上記どおり固定し、outer由来の再選択は0、exact compute budgetはformal化前のblockerとする。

legacy outputsはfold summary、model comparison、OOS race detail、pooled summary、segment summary、aggregate summary、`summary.json`、`review.md`である。model mean/std/weights、full per-set probabilities、canonical input/code/environment hashは保存していない。Stage A packageではper-fold model stateとfull probabilitiesを追加し、race IDとsorted tripleでcanonical sortし、UTF-8、line ending、float encoding、relative pathを固定する。absolute checkout pathはdigest対象artifactへ含めない。

external Top3 merge不一致をlegacy codeは0でfillする。canonical packageではsilent fillを禁止してfail-closeし、この意図的な安全修正をlegacy numeric parityから分離したmanifest差分として記録する。

残るrecipe blockerとして、legacy実行の`--mode`は未確認である（static default候補は`core`、別選択肢は`full`）。またcanonical probabilityを`M0_raw`と`M0_temperature_scaled`のどちらにするか、B0/B1/B2とcomparison outputをparity digestへ含めるかも未確認である。canonical summary/evidenceをattestするまで`UNKNOWN_PENDING_CANONICAL_REFERENCE`とし、exact command、model name、probability stage、output scopeをrun scopeへ固定しない限りStage Aをblockする。

## Price firewall

将来の候補生成側で許可できるのは、締切時点で既知の非市場情報と、その情報だけから生成し、lineageを証明したmodel scoreに限る。

M0再構築のfinal model feature allowlist候補は、`sum_primary_strength`, `min_primary_strength`, `sum_rank_strength`, `min_rank_strength`, `critical_missing_count`の5列だけである。ただし、これらの親である`ai_score`と`ai_rank`の再帰lineageがprice-blind、OOF、as-ofであると証明できるまでallowlistは未確定とし、runをblockする。formal化前にexact raw/final column名、Unicode/case/区切り正規化、alias registry、全派生列の親hashを機械可読policyへ固定する。

禁止family:

- current/final/historical odds
- popularity、market rank
- payout、payoff、return、profit、ROI
- BUY、stake、order情報
- future/outer result information
- 上記の別名、camelCase、略称、集計、比率、残差、埋込み、派生値

払戻・価格はprice-blind candidate artifactをcanonical hash固定した後、評価器だけがjoinする。

historical finish/result labelはfeature tableと物理分離し、partitionとas-ofを固定した後にtarget builderだけがtrain/validation/calibrationへjoinする。outer resultはmodel、prediction、candidate artifactを結果blindでhash固定した後に評価器だけがjoinする。price/payoffは全stageでmodel/candidate processから隔離し、offline evaluatorだけが参照する。

## Stage B — prospective handoff

M0再現成功後も、H1または別challengerを自動採用しない。別の正式proposalで、実行前に次を固定する。

- M0とchallengerの同一prospective outer start/end rule
- 同一race/runner universeとpaired comparison
- outerを一度だけ開封する手順
- primary: race-weighted unordered Top3 set NLL
- secondary: calibration/Brier/ranking、field-size stability、mass contracts
- candidate mapping freeze point
- probability gate通過後だけ行うdiagnostic ROI

ROI改善はこのprospective experimentの副次結果であり、M0 reproductionの結果ではない。

時系列は、calibration完了 → M0/challengerとprice-blind ticket policy/manifest固定 → result-blind outer予測 → price/result公開前に各armのfinal ticket universe、candidate-to-ticket mapping、unit-notional vector、no-bet/abstention decision、prediction/candidate/ticket digestを固定 → sealed T-3 price snapshotを評価器へappend → official result/payout/取消/返還/同着settlementをappend → probability評価 → frozen artifactのままpaired ROI評価、とする。評価器はjoinとreturn計算だけを行い、quoteによる選択・除外・重み変更・abstentionを禁止する。outer開始点はproposal digest、承認、mapping、manifestがすべて固定された後に初めてeligibleになる未観測raceとする。150件到達はlabel/priceを開かないblind enrollment countで確認する。

Stage Aの人間review後に作る別proposalの判定案は次のとおりである。

- primary: `ΔNLL = NLL_challenger - NLL_frozen_M0`
- required effect候補: `ΔNLL <= -0.005`
- race-block bootstrap one-sided 90% upper bound `< 0`
- 全predeclared outer subperiodで`ΔNLL <= 0`
- outerでfeature、regularization、temperature、thresholdを再選択しない
- probability gate未達は、ROIが高くても棄却する

ROIを成功と呼ぶ場合の副次gate候補:

- point ROI `> 100%`
- race-block one-sided 90% lower bound `>= 100%`
- 全outer blockでROI `>= 100%`
- 単一raceの利益寄与 `< 20%`
- strict T-3 Grade-Oを事前登録した150件以上で評価する

現状のstrict T-3 Grade-Oは0/150であり、必要件数に達するまでROI判定は`BLOCKED_DATA`である。

ROIはM0とchallengerで同じrace、quote、price-blind candidate-to-ticket policy、race当たり総unit exposure、no-bet ruleを使う。Primary paired ROI endpoint候補を`Δnet_return = net_return_challenger - net_return_M0`、required point effectを`> 0`、race-block one-sided 90% lower boundを`>= 0`とし、absolute profitability gateはchallenger armへ適用する。quote source/received timestamp、missing odds、取消、返還、同着、settlement、ROI分子/分母、unit notional、no-bet、outer block、power/effective race countを先に固定する。`stake=0`は実購入が0という意味であり、offline unit notionalとは別である。

## Honest score

現行strategy scorecardへ無理に当てはめたStage Aの暫定値は23/100で、`BLOCKED_SCORE`である。

- independent information: 0/25
- racing mechanism: 0/20
- reproducible outer-OOS failure evidence: 0/20
- leakage safety: 8/15
- minimal falsifiability: 10/10
- acquisition/implementation cost: 5/10

理由: M0は新情報・新作用機序を追加せず、旧5,336 racesは新しいouter証拠ではない。`ai_score` lineageも未証明である。これを75点以上に採点することはscore inflationである。Prospective Stage Bはchallenger未定義の現在、scoreを`null / BLOCKED_DEFINITION_AND_SCORE`とする。53/100は、具体的な作用機序、非同一性/ablation、完全lineageを別途証明した場合だけの仮上限で、実行権限ではない。Stage Bは通常strategy gateの75点未満なら実行せず、専用reproduction eligibilityを流用しない。

## Formal化前のblocker

1. legacy reference artifactのcanonical path/hash
2. pair/top3 input artifactのcanonical path/hashとsource lineage
3. upstream `ai_score`/`ai_rank`のOOF、as-of、price-blind lineage
4. race/runner universe、取消、同着、失格、降着のlabel policy
5. exact dependency/environment manifest
6. deterministic artifact contractと数値許容差
7. exact expected changed paths
8. host-assigned experiment IDとfresh main base commit
9. model/real-data executionを扱えるResearch OS routeとGitHub二段階承認
10. strategy/proposal ID、brain model/prompt/context provenance（外部AI由来の場合）
11. target population、in/out scope、raw sources、data as-of、exact allowed/forbidden columns
12. fold manifest path/hash、purge/embargo、compute budget
13. primary/effect/rejection/stopとcanonical名の`allowed_variant_count=1`、`allowed_threshold_search_count=0`
14. future v2の`schema_version`、`title`、`lineage_hash_requirements`、`chronological_fold_design`、exact phase-specific safety/capability field map

これらが埋まるまではcanonical proposal、registry append、implementation、runへ進まない。

現行`infrastructure_safety_v1`はmodel、feature、candidate、real data、ROIを扱えないため、このproposalの実行routeには使わない。human-reviewed governance Draftで、例えば`roi_reproduction_audit_v2`と`roi_prospective_model_validation_v2`のkind-bound contract、schema、supervised executorを先に導入し、そのmerge後mainを新しいbaseにする必要がある。infra gate自身にこのroot-of-trust変更を承認させてはならない。

`roi_reproduction_audit_v2`はstrategy scoreを75点へ水増ししたり、75点gateを迂回したりしてはならない。非promotionの再現監査に専用のeligibility contractを人間が定義し、成功してもmodel採用・shadow・BUY権限を一切付与しない設計が必要である。prospective Stage Bは別scope、別digest、別prepare/run comment IDで申請し、Stage Aの承認IDを再利用しない。

Proposal Aのprepare/run comment IDはregistry全体で未使用かつ相互に異なる必要がある。`APPROVED_TO_RUN`直前にprepare evidenceを再検証し、`RUNNING`直前にprepare/run両evidenceを再検証する。Proposal Bも新しいprepare/run IDを必要とし、人間commentはscore、kind、scope、hashのineligibilityを上書きしない。

## Safety declaration

- applies_to=`current_document_only`
- `actual_codex_dispatch=false`
- `automatic_github_approval=false`
- `candidate_policy_change=false`（production candidate policyを意味する）
- `credential_access=false`
- `external_api_calls=false`
- `formal_buy=false`
- `merge=false`
- `notification_side_effects=false`
- `production_change=false`
- `purchase_path_access=false`
- `real_data_execution=false`
- `send_order=false`
- `stake=0`

上記はこのseed作成turnの事実である。将来のProposal A historical replayは、versioned kind-bound contractと有効なrun承認がある場合だけ`real_data_execution=true`（mode=`historical_reproduction_only`）を要求するが、ROI、production candidate policy、BUY、stake、orderは許可しない。Proposal Bの`real_data_execution=true`は、通常strategy score gateを含む全eligibilityと別run承認を満たす場合だけ要求できる。
