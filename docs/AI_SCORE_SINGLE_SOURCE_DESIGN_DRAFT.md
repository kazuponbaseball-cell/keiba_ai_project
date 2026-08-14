# AI score single-source architecture — human-review Draft

- Status: `DESIGN_REVIEWABLE`
- Authority: `NON_CANONICAL_DESIGN_ONLY`
- Implementation: `BLOCKED`
- Real-data / backtest / outer OOS / ROI execution: `FORBIDDEN`
- Production / BUY / order / notification change: `FORBIDDEN`
- Base: GitHub `main` `f24e6a0e394dd376d100502d33872d47dad0ec9a`
- Date: `2026-08-14`

## 結論

`ai_score`由来の信号を複数の別票として再加算する構造は、ROIが変わらなくても
解消する方向が妥当である。ただし、現行championを直接書き換えない。

最初に、AI確率の入口を1つに限定したversioned research sidecarを作る。
現行は凍結した比較対象として残し、構造契約、確率非劣後、将来の経済的非劣後を
順に満たした場合だけshadow候補とする。ROI向上は構造改善の必須条件にしないが、
ROI悪化を無視してproductionへ昇格することも認めない。

このDraftは設計だけを固定する。実装、synthetic test、実データ評価、ROI計算、
production切替の権限を付与しない。

## 1. なぜ直すのか

現在のtracked codeには、区別すべき3つのscore・probability objectがある。

1. predictorのhorse-level raw `ai_score`と、その決定論的なrank、gap、confidence
2. production builderが`ai_score`をrace内softmaxして作るhorse-level `ai_prob`
3. local/unmerged Grade-R pipelineのTop3 set posterior `q_ai(S)`とwide marginal `p_wide`

1の派生列は`src/predict/predict_baseline.py:281-315`で生成される。2は
`scripts/build_current_strongest_tickets.py:2146-2150`で生成される。market-awareな
現行builderでは、horse-level `ai_prob`、rank、confidence、gapがoverlay、danger、runner score、pair score、
`ticket_hit_prob`、expected ROI、strongest score、simulation、複数gateへ再流入する
（`scripts/build_current_strongest_tickets.py:2146-2336,3123-3270,3829-4075`）。

これらは独立した観測票ではない。同じ誤差を複数回増幅し、個々の係数やgateの意味を
説明しにくくする。一方で、rankやgapという関数形自体が役立つ可能性はあるため、
派生値を一律削除するのではなく、1つの明示的なAI component内へ閉じ込める。

既存のlocal dedup再学習比較では、496特徴から重複名34個を除いて462特徴にしても、
同じ6,749レースのTop1勝率は28.34494%から28.35976%への微差で、Top1 Top3率と
Top3内勝馬率は同値だった。ただしROIは保存されていない。この証拠は
`B_LOCAL_HASHED`であり、単純な入力列重複が大きな順位劣化要因ではなさそうだと示すだけで、
下流の自己増幅によるROI影響を識別しない。

なお、read-only追跡ではactive Grade-R Top3 bundleのutilityへ直接入るcurrent
`ai_score` leafは`sum_primary_strength`の1本だった。M1Cの履歴能力やM1A1の
pace/style特徴を、根拠なく`ai_score` aliasとして削除してはならない。明確な重複は、
posteriorをraw probability、calibrated probability、rank、margin、confidence、gateへ
分岐させて再利用する判断層と、production builderの多段compositeにある。したがって
初手は予測coreの削除ではなく、候補freeze後のdecision-bearing signalを1本にするsidecarとする。

## 2. あるべき依存構造

```text
as-of / price-blind runner data
              |
              v
      AI probability model
              |
              v
 canonical q_ai(Top3 set), mass = 1
              |
              +----> p_wide(pair), mass = 3
              |              |
              |              v
              |       odds-free candidate freeze
              |
              +----> observability only
                     rank / gap / confidence / entropy

 candidate freeze ----> 1-D action calibration -> p_action_calibrated
                                                   |
                         T-3 quote append ----------+
                                                   v
                                      offline value eligibility
                                                   |
                                                   v
                                          result / ROI evaluation
```

このchallengerでのAI正本はTop3非順序集合分布`q_ai(S)`とする。候補決定にはその周辺確率
`p_wide(i,j)`だけを使う。候補freeze後のcandidate abstentionではなく、post-freezeの
value eligibilityへ、事前較正した`p_action_calibrated`だけをAI-bearing scalarとして渡す。rank、gap、
raw probability、margin、confidence、entropyは表示・監査用として保存できるが、
post-freeze value-eligibility utilityへ別々に再注入しない。

### Single-source invariant

対象境界はpost-freezeのvalue-eligibility utilityとする。frozen candidate keyとfreeze hashは
routing identityであり、数値AI入力として数えない。このutilityに許可する数値AI descendantは
`p_action_calibrated`の1列だけとする。

- 許可: `q_ai -> p_wide -> candidate rank`
- 許可: frozen candidateの`p_action_calibrated`をvalue layerへ1回渡す
- 許可: `q_ai -> confidence`を表示列として出力
- 禁止: `p_wide + ai_rank + ai_prob + gap + confidence`を別係数で再加算
- 禁止: AI派生値をdanger、pair score、hit proxy、ROI proxyへ入れ子で再注入
- 禁止: posterior-derived scoreを「独立モデル票」と数える
- 禁止: market値をcandidate freeze前のrank、coverage、abstentionへ使う

`p_action_calibrated`はfrozen candidateの`p_wide`だけを入力にする1次元calibrator
`c(p_wide)`とする。calibrator method、fit partition、fit cutoff、fit-data manifest、label manifest、
parameters、code、input/output schemaをartifact hashへ固定する。rank、gap、confidence、margin、raw horse `ai_score`、
horse-level production `ai_prob`をcalibrator入力にすることも禁止する。

将来、非AI context residualを追加する場合は、AI descendant、market、price、result、payoffを
親列として持たないことをlineageで証明し、別のincrementality検定を通す。初版challengerには
context residualを追加しない。

## 3. Versioned challenger

### Frozen control

現行championと既存artifactを変更しない。production builderをResearch OSからimport、実行、
または候補生成器として流用しない。現行挙動の比較仕様を作る場合もread-onlyなversioned
reference contractとして再記述する。R0はdecision式、threshold、calibration artifact、
training cutoff、fit-data/label manifest、input lineage、code/schema digestを固定できるまで比較armとして扱わない。

### Challenger `SINGLE_SOURCE_AI_V1`

- AI入口: frozen/as-of/OOFで生成した`q_ai(S)`だけ
- wide probability: `p_wide(i,j)=sum(q_ai(S) for S containing {i,j})`
- odds-free candidate: `p_wide`降順、事前固定した非market tie-break
- frozen candidateの判断AI値: `p_action_calibrated=c(p_wide)`の1本だけ
- raw probability/margin/confidence/rank/gap/entropy: output可、同じ判断へのinput不可
- market: candidate digest確定後のoffline value eligibilityだけ
- result/payoff: candidateとquoteの両方をfreezeした後の評価だけ
- `formal_buy=false`
- `send_order=false`
- `stake=0`

Phase 1の純粋なcleanup armでは、R1はR0と同じ`p_action_calibrated` bytesと同じcalibrator artifactを
再利用し、calibratorを再fitしない。R1のvalue-eligibility式とeffective thresholdも事前にhash固定し、
R0と同じ非AI条件・market式・effective thresholdを保ったまま、余分なAI descendant gateだけを除く。
calibratorまたはthresholdを再fitするarmは、このcleanupと混ぜず、別scope・別variantとして数える。

この初版は新情報追加モデルではなく、責務分離とnegative controlである。既存Top3
probability artifactやcandidate keyは上書きせず、additive sidecarとして並走させる。

## 4. 採否をROI向上だけにしない

構造改善の採用判断は次の三段階に分ける。

### Gate A — structural correctness

全項目必須とする。

- post-freeze value eligibilityへの数値AI descendantが`p_action_calibrated`の1経路だけ
- AI派生列は`observability_only`または単一component内部に閉じる
- Top3全`C(n,3)`、finite、range、duplicate検査
- raceごとのTop3 mass `1 ± 1e-10`
- 導出wide mass `3 ± 1e-10`
- candidate freeze前のodds、market、payoff、result利用が0
- production、BUY、order、notification、credential importが0

1項目でも失敗したら`INVALID`で、性能やROIを読まない。

### Gate B — Phase 1 probability equivalence / Phase 2 predictive non-inferiority

Phase 1のdecision-layer sidecarでは、frozen `q_ai`、`p_wide`、candidate digestに加えて、
`p_action_calibrated` bytesとcalibrator artifactの完全一致を要求する。candidate rank、tier、coverage、
abstentionは変更しない。唯一の介入は、hash固定したR1 value-eligibility式で余分なAI descendant gateを
除くこととする。R1式、effective threshold、非AI条件を事前固定し、再較正を禁止する。

このためPhase 1のaction log loss、Brier、calibrationは両armで同一になる共有diagnosticであり、
差分metricや採否gateにはしない。Phase 1のprimary gateはprobability/calibratorのbyte-level parityと
single-source構造検査である。value-eligibility disagreement、stake差、return差、ROI差は全て0を要求し、
非zeroなら性能差ではなくcontract failureとする。

probability core自体のalias cleanupをPhase 2として行う場合だけ、同一runner universe、
同一fold、同一fit/calibration budgetで、current referenceとsingle-source probability coreの
Top3 set NLLを比較する。Phase 1はq/pが完全一致するためTop3 NLLを判定指標にしない。

- Phase 1 primary gate: q/p/candidate/p_action/calibratorのexact parityとsingle-source構造検査
- Phase 1 shared diagnostics: action log loss、Brier、calibration（arm差は定義しない）
- Phase 1 parity diagnostics: value-eligibility disagreement count `0`、同一eligibility rate
- Phase 2 primary: race-paired Top3 set NLL差
  `delta_nll = NLL(single_source) - NLL(reference)`
- proposed non-inferiority margin: `+0.005`
- date × venue_code cluster bootstrapのone-sided 95%上限が`+0.005`未満
- foldごとの符号、wide Brier、calibration、候補変更数を併記
- outer testを見たthreshold、coverage、margin変更は禁止

marginはcanonical proposalを作る前に人間reviewで確定し、その後変更しない。

### Gate C — economic checksを3つに分ける

#### C1. Phase 1 parity economics

Phase 1の`R0_REFERENCE`と`R1_SINGLE_SOURCE`はparity refactorである。post-freeze value eligibility、
stake、return、ROIの全てを完全一致させる。primary economic metric、margin、bootstrapは
`NOT_APPLICABLE_PARITY`で、mismatchは性能差ではなくcontract failureとする。

#### C2. Historical D0/D1 impact diagnostic

value eligibilityがarm間で異なり得るのは、別scopeの`D0_REFERENCE`対
`D1_REMOVE_RAW_GATES`だけである。これはfinal official payoffを用いるmechanistic diagnosticであり、
economic non-inferiorityやshadow採用の判定には使わない。全enrolled raceをno-bet込みで残し、
primaryをrace当たりpaired profit差、ROIをarm固有stake denominatorのsecondaryとする。

#### C3. Prospective economic non-inferiority

非parityの将来challengerをshadow候補にする場合だけ、候補freeze後の同一T-3 quote、同一candidate
enrollment、同一unit notionalでpaired net returnを比較する。candidate coverage、最大ticket budget、
quote source/snapshot、settlement ruleを同一化し、value eligibility rateとarm別stake denominator、
high-payout感度、profit concentration、`date × venue_code` cluster bootstrapを併記する。
economic marginはprospective cohortを開く前に固定し、strict T-3証拠が不足する間は`BLOCKED_DATA`とする。
ROI向上は必須にしないが、事前固定した実用的悪化幅を超えないことを要求する。通過しても
「利益改善」とは表現しない。

## 5. 最小反証比較

最初の比較は2 armに限定する。

```text
R0_REFERENCE:
  versioned frozen research candidate and decision reference
  exact formula / threshold / calibrator / cutoff / lineage / digests required

R1_SINGLE_SOURCE:
  same frozen q_ai / p_wide and candidate key
  exact same p_action_calibrated bytes and calibrator artifact as R0
  exact hash-bound value-eligibility formula and effective threshold
  p_action_calibrated = c(p_wide) is the only numeric AI input
  no reinjected raw p, rank, gap, confidence, danger, pair composite, hit proxy
```

R0/R1は同じcalibration fit-data manifestとlabel manifestを参照する。Phase 1でのcalibrator refitは
禁止し、再較正を試す場合は別scope・別variant・別承認対象にする。

Phase 1では全diagnostic列を個別に摂動してもdecisionが不変で、`p_action_calibrated`だけを
変えた場合に限りdecisionが変わることをsynthetic contractにする。

補助negative controlとして、`q_ai`の決定論的変換だけを追加したarmを使う場合も、
variant countへ含め、outer test前に固定する。これが改善しても「新しい情報」とは呼ばず、
再校正または関数形の差と解釈する。

### 5.1 Read-only historical inventory: parityと影響診断を分離する

2026-08-14のread-only inventoryで、local research ABC guardの式と既存artifact候補を確認した。
これは`B_LOCAL_HASHED`の非権限証拠であり、本PRではR0/R1 replay、ROI再計算、再較正を実行していない。

記号を次のように置く。

```text
p = top1_wide_prob
a = p_action_C0_offset
a = sigmoid(logit(p) + 0.130654047367905)
decision_base = eligible_race AND candidate_generated
observed_audit_base = decision_base AND official_outcome_joined
```

hash固定した既存audit scriptが観測用artifactへ適用したmaskは次である。

```text
R0_OBSERVED_AUDIT = observed_audit_base
                    AND 0.225 <= p < 0.325
                    AND 0.25 <= a < 0.4
                    AND p >= 0.21275851149504352

R0_DECISION = decision_base
              AND 0.225 <= p < 0.325
              AND 0.25 <= a < 0.4
              AND p >= 0.21275851149504352
```

`official_outcome_joined`は既存evaluation scriptのsettlement availability検査であり、出走前decisionの
入力ではない。将来runnerでは`R0_DECISION`を結果列なしでfreezeし、その後に全3,746 raceのsettlement
completenessを検査する。inventory対象ではmissing outcomeが0なので、両maskの選択結果は一致し得るが、
契約上は別objectとして保持する。

calibratorは単調なので、R0のeffective raw intervalは
`0.22630985987893942 <= p < 0.325`である。したがって、純粋なparity refactorは
次の1本へ代数的に畳み込める。

```text
R1_PARITY = decision_base AND 0.25 <= a < 0.35429028335310075
```

`R1_PARITY`は`R0_DECISION`とcandidate、判定、stake、return、ROIが全raceで完全一致するはずである。
ROI差`0`は実測成果ではなく代数的帰結であり、非zero mismatchは改善ではなく実装・contract failureとする。

一方、ユーザーが確認したい「重複gateが過去ROIを悪化させたか」は、別の影響診断である。
raw probability gateだけを除くone-change armは次となる。

```text
D0_REFERENCE = R0_DECISION
D1_REMOVE_RAW_GATES = decision_base AND 0.25 <= a < 0.4
```

`D1_REMOVE_RAW_GATES`は選択raceを増やし得るため、parity refactorではなくvalue-policy counterfactualである。
Phase 1の2 armへ混ぜず、別experiment ID、別variant budget、別canonical scope、別承認でのみ実行する。

read-only inventoryで確認した候補cohortは次の3つで、混同しない。

- full development OOS: 5,336 races、2024-07-06から2026-02-15
- 既報の比較対象: fold2–4の3,746 races、2025-01-05から2026-02-15
- policy-freeze後の補助holdout: 144 races、2026-07-18/19/25/26、Grade-R

3,746 race診断を将来実行する場合は、全enrolled raceを1 race 1 rowで保持する。

```text
stake_arm = 100 if eligible else 0
return_arm = official_wide_pay if eligible and hit else 0
profit_arm = return_arm - stake_arm
delta_profit = profit_D1 - profit_D0
```

decision maskをfreezeした後にだけofficial outcome/payoffをsettlementとしてjoinする。全raceでofficial
outcomeとcandidate hit判定を必須とする。`hit=true`のときだけ、一意・finite・positiveな
`official_wide_pay`を必須とし、`hit=false`ではpayoff空欄を許可してreturnを0とする。必要な結果・払戻の
欠測、duplicate、candidate mismatchはraceを黙って落とさずrun全体を`INVALID`にする。

primaryは`mean(delta_profit) per enrolled race`とし、ROIはarm固有stake denominatorのsecondaryとする。
both-no-bet、D0-only、D1-only、both-betをすべて残し、no-betはstake/return/profitを0とする。
高配当除外はarm別に選ばない。`settled_candidate_return_yen`は
`official_wide_pay if candidate_hit else 0`と定義する。ranking universeはdecision mask確定後の
全3,746 enrolled raceとし、settled candidate returnを降順、同額時は`race_date`、`race_id`昇順で
並べ、top1/top3 race ID集合を作る。
その共通集合を両armへ適用し、rowとstakeは残したままreturnだけを0にする。生成したrace-ID setの
正本はrace IDをUnicode code-point昇順に並べた重複なしJSON arrayとし、UTF-8、BOMなし、compact
separator、末尾改行なしのbytesをSHA-256化してrun evidenceへ固定する。別のwinsor感度は各armで
共通の`capped_candidate_return_yen = min(settled_candidate_return_yen, 2000)`を先に作り、
`return_arm_winsorized_yen = capped_candidate_return_yen if eligible_arm else 0`とする。rowとstakeは維持する。

uncertaintyは`race_date × venue_code` cluster bootstrapとし、100,000回、seed `20260814`、
`numpy.random.Generator(PCG64)`を固定する。`race_date`は`YYYY-MM-DD`、`venue_code`は2桁文字列と検証し、
unique tupleを両列のlexicographic昇順で並べる。各replicateでexactly 1回、
`rng.integers(0, n_clusters, size=n_clusters, dtype=np.int64, endpoint=False)`を呼び、返却index順に、各clusterの
`race_id`昇順rowを全て追加する。各replicateで
`sum(delta_profit) / resampled race-row count including multiplicity`を計算する。one-sided 95% lower boundは
`numpy.quantile(statistics, 0.05, method="linear")`とし、実行時はrunner/environment hashも固定する。

ただし、このcohortはreview済みdevelopment OOSかつfinal official payoff診断である。historical候補期間と
利用可能なT-3 odds期間のoverlapは0 race、strict T-3適格rowも0である。したがって将来実行できても、
価格残存、ex-ante EV、実行可能ROI、confirmatory outer OOSの証拠にはならない。

主要なlocal source candidateは次である。全てuntracked/ignoredで、authorityではない。

- diagnostic master SHA-256: `697142b64e8052b212731dc0319ccafb7f61ac29dbc46f67385f9ae050129de9`
- R0 policy SHA-256: `74a2175a0490c3b998fbd539d280671f6857508392d995dcb65d4666f4bad67f`
- R0 script SHA-256: `06132d923f10587664fe4ed19ce9dc25c3b152d34a1461acd034f928e6e1e465`
- R0 audit config SHA-256: `fae9c66c77574f09f0b5cede6c74e837cd6afb33e0b26f54809ef643e567837e`
- p-action artifact SHA-256: `34f56b5a61261bd9b6cfd38797b65bd88415d0778d98cef29eebfbe2f09e513c`
- calibrator fit CSV SHA-256: `a4dd2a5d82792b0a4f952797253286555ca2b39449f2d685596c188ef4270546`
- calibrator summary SHA-256: `fd53aca115ec7055e3fadc6aa2ef01566d9e4f91df35d2f4b82de9b89fbc38f6`
- calibrator producer SHA-256: `b36b77fe92b250ac8056a8743c6f3f73f87a04ace1de921d5527c7bb21a69f2c`
- official payoff source SHA-256: `b94b0c0ea2ce4424d70432f7d070a9083d01850876a710432ec5b98538070d83`
- 144-race joined holdout SHA-256: `076824fbd189ad0c5bf0884b614ead3a3ff159422aa58ef3eefd3792927300a5`

decision runnerへ結果・払戻同居CSVを直接渡してはならない。将来の承認済みproviderはcandidate-only
projectionを先にfreezeし、そのdigest確定後にsettlement snapshotをjoinする。

## 6. 変更の順序

1. このdesign Draftを人間reviewする。
2. PR #38とは別に、model-integrity / negative-controlを扱えるgovernance laneを
   人間review・mergeするか、G2の正式contractへ明示的に組み込む。
3. post-G2のcurrent `main`からcanonical proposalを新しいexperiment IDで作る。
4. `APPROVED_TO_PREPARE <proposal_digest>`後に、research-only implementationと
   synthetic fixture testだけを作る。
5. versioned run scopeと別の`APPROVED_TO_RUN <run_digest>`後にだけ、承認済み実データを評価する。
6. review後もproduction切替は別の人間承認・別PRとする。

## 7. 現時点のblocker

- cleanup仮説は新しい独立情報ではなく、現行scorecardの`independent_information=0`
- clean/hash-boundなupstream OOF/as-of `ai_score` lineageが未確定
- common fold / runner-universe manifestが未確定
- current `main`にtracked Grade-R candidate packet producerがなく、producer commit/schemaが未確定
- R0 reference formula/threshold/calibrator/cutoff/fit-data/label/input lineage/digestが未確定
- R1 value-eligibility formula/effective thresholdと、共有1-D calibratorのmethod/fit cutoff/fit-data/label/parameters/code/schema hashが未確定
- strict T-3 prospective evidenceが不足
- PR #38のG1は`EXECUTION_FORBIDDEN`で、G2は未実装
- 現行legacy ROI contractはexecution kindをdigestへbindしないためreal-data実行不可
- production replacementはResearch OSの権限外

暫定scoreは46/100で`BLOCKED_SCORE`とする。scoreを水増しして実装へ進めない。

| Criterion | Score | Max | Reason |
|---|---:|---:|---|
| Independent information | 0 | 25 | 新規観測ではなく重複整理 |
| Racing mechanism | 10 | 20 | 誤差自己増幅の方向と限界は説明可能 |
| Outer OOS failure evidence | 10 | 20 | local/reused evidenceのみ、clean再現なし |
| Leakage safety | 8 | 15 | 禁止規則は定義、lineage機械証明は未完成 |
| Minimal falsifiability | 10 | 10 | 2 armの非劣後で反証可能 |
| Cost | 8 | 10 | 小さく可逆だがG2とmanifestが必要 |

## 8. 人間reviewで決める事項

- `q_ai(S)`を唯一のAI正本とするか
- rank/gap/confidenceをobservability-onlyにするか
- ROI向上なしでも非劣後ならshadow候補にするか
- NLL非劣後margin `+0.005`を採用するか
- economic non-inferiority marginをどの単位で事前固定するか
- candidate packet producer/schema、R0/R1共有calibrator、fit-data/label manifest、R1式/thresholdをどのcommit/hashで固定するか
- G2へmodel-integrity laneを追加するか、別versioned governance PRにするか

本Draftの承認は、実装、テスト実行、実データ読込、ROI計算、shadow、production、
BUY、注文、通知、mergeを承認しない。
