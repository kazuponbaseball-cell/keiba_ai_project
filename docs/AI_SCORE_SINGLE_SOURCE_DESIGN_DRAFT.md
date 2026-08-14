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
single-source構造検査である。介入の挙動差はvalue-eligibility disagreement/rate、経済差はGate Cで測る。

probability core自体のalias cleanupをPhase 2として行う場合だけ、同一runner universe、
同一fold、同一fit/calibration budgetで、current referenceとsingle-source probability coreの
Top3 set NLLを比較する。Phase 1はq/pが完全一致するためTop3 NLLを判定指標にしない。

- Phase 1 primary gate: q/p/candidate/p_action/calibratorのexact parityとsingle-source構造検査
- Phase 1 shared diagnostics: action log loss、Brier、calibration（arm差は定義しない）
- Phase 1 intervention diagnostics: value-eligibility disagreement/rate
- Phase 2 primary: race-paired Top3 set NLL差
  `delta_nll = NLL(single_source) - NLL(reference)`
- proposed non-inferiority margin: `+0.005`
- date × venue cluster bootstrapのone-sided 95%上限が`+0.005`未満
- foldごとの符号、wide Brier、calibration、候補変更数を併記
- outer testを見たthreshold、coverage、margin変更は禁止

marginはcanonical proposalを作る前に人間reviewで確定し、その後変更しない。

### Gate C — economic non-inferiority

ROI向上は必須にしない。候補freeze後の同一T-3 quote、同一candidate enrollment、同一unit
notionalでpaired net returnを比較する。candidate enrollmentは同一だが、post-freezeの
value eligibilityはarm間で異なり得る。no-bet raceはstake=0、return=0としてpaired rowに残す。
単純なROI比だけをprimaryにしない。

- primary: race単位の`single_source - reference` paired net return
- candidate coverage、最大ticket budget、quote source/snapshot、settlement ruleを同一化
- value eligibility rateとarm別stake denominatorを必ず併記
- high-payout除外、profit concentration、date × venue cluster bootstrapを併記
- economic marginはprospective cohortを開く前に固定
- strict T-3証拠が不足する間は`BLOCKED_DATA`

ROIが同等でもGate AとBを通り、Gate Cで事前固定した実用的悪化幅を超えなければ、
構造改善としてshadow候補にできる。ただし「利益改善」とは表現しない。

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
