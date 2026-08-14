# AI score single-source architecture — human review checklist

Status: `DESIGN_REVIEWABLE / IMPLEMENTATION_BLOCKED / EXECUTION_FORBIDDEN`

## A. Architecture decision

- [ ] Top3集合確率`q_ai(S)`を唯一のAI正本とする。
- [ ] wide確率は`q_ai(S)`の周辺化だけで作る。
- [ ] rank、gap、confidence、entropyは表示・監査用に残せる。
- [ ] 上記派生値を同じcandidate/value utilityへ別票として再注入しない。
- [ ] 初版challengerへ新しいcontext、market、pair rerankerを追加しない。
- [ ] 初版はfrozen candidateを読むadditive sidecarとし、candidate keyを変えない。
- [ ] post-freeze value eligibilityで使うAI値を`p_action_calibrated`の1本にする。
- [ ] post-freezeの用語を`value eligibility`とし、candidate abstentionと混同しない。
- [ ] `p_action_calibrated`の唯一の入力をfrozen candidateの`p_wide`にする。
- [ ] calibrator method、fit cutoff、fit-data/label manifest、parameters、code/schema hashを固定する。
- [ ] 現行championを直接変更せず、versioned challengerにする。

## B. Interpretation

- [ ] ROI向上を構造改善の必須条件にしない。
- [ ] ROIが同等でも、構造契約と予測非劣後を通ればshadow候補にできる。
- [ ] この場合に「ROI改善」または「利益優位」と表現しない。
- [ ] economic non-inferiority未確認のままproductionへ切り替えない。
- [ ] posterior派生値を独立モデルの合議票として数えない。

## C. Structural gate

- [ ] post-freeze value eligibilityへの数値AI descendantは`p_action_calibrated`の1経路だけである。
- [ ] counted boundaryをpost-freeze value-eligibility utilityに限定する。
- [ ] candidate key/freeze hashはrouting identityで、数値AI票として数えない。
- [ ] numeric AI allowlistは`p_action_calibrated`の1列だけである。
- [ ] raw p/rank/gap/margin/confidenceの各列を摂動してもdecisionが不変である。
- [ ] 全`C(n,3)`、Top3 mass 1、wide mass 3を`1e-10`で検査する。
- [ ] candidate rank、coverage、abstentionにodds・人気・marketを使わない。
- [ ] result/payoffはcandidate/quote freeze後だけにjoinする。
- [ ] production、BUY、order、notification、credential pathをimportしない。
- [ ] structural違反時は`INVALID`とし、NLLやROIで救済しない。

## D. Non-inferiority protocol

- [ ] 比較armは`R0_REFERENCE`と`R1_SINGLE_SOURCE`の2つに固定する。
- [ ] Phase 1ではq/p/candidateに加え、`p_action_calibrated` bytesとcalibrator artifactの完全一致を要求する。
- [ ] Phase 1でcalibratorを再fitせず、再較正armは別scope・別variantにする。
- [ ] R1のvalue-eligibility式とeffective thresholdをhash固定する。
- [ ] R0/R1で非AI条件、market式、effective threshold contractを同一にし、余分なAI descendant gateだけを除く。
- [ ] Phase 1 primaryをexact parityとsingle-source構造検査にする。
- [ ] Phase 1のR0/R1はvalue eligibility、stake、return、ROIまで完全一致させる。
- [ ] Phase 1のeconomic primary、margin、bootstrapを`NOT_APPLICABLE_PARITY`にする。
- [ ] action log loss/Brier/calibrationは共有diagnosticとし、arm差を採否に使わない。
- [ ] Phase 1でTop3 NLL差を採否に使わない。
- [ ] probability coreのalias cleanupはPhase 2の別scopeに分ける。
- [ ] runner universe、fold、fit/calibration budgetを共有する。
- [ ] Phase 2 primaryをrace-paired Top3 set NLL差にする。
- [ ] proposed NLL margin `+0.005`をcanonical proposal前に確定する。
- [ ] date × venue_code cluster bootstrapを事前固定する。
- [ ] outer testを見たthreshold、coverage、margin変更を禁止する。
- [ ] ROIは同一candidate enrollment、同一T-3 snapshot、同一unit notionalのpaired net returnで診断する。
- [ ] no-bet raceをstake=0/return=0でpaired rowに残す。
- [ ] arm別value eligibility rateとstake denominatorを報告する。
- [ ] economic marginをouter結果を見る前に確定する。

## E. Historical ROI counterfactual boundary

- [ ] parity refactorはR0と判定・ROIが代数的に完全一致し、差が非zeroならcontract failureとする。
- [ ] 重複gateの影響診断はparity refactorへ混ぜず、別experiment ID・別scope・別variantにする。
- [ ] historical D0/D1診断をeconomic non-inferiorityやshadow採否へ使用しない。
- [ ] prospective economic non-inferiorityは別scopeのnon-parity challengerにだけ適用する。
- [ ] decision baseは`eligible_race AND candidate_generated`に固定し、結果・払戻をdecision maskへ入れない。
- [ ] 既存auditの`official_outcome_joined`込みmaskと、将来の結果非依存R0 decision maskを別objectとして記録する。
- [ ] 全3,746 raceのsettlement completenessはdecision freeze後のpreconditionとして検査する。
- [ ] impact armは`D1 = decision_base AND 0.25 <= p_action_C0_offset < 0.4`に固定し、再較正・threshold探索を行わない。
- [ ] 主診断cohortはfold2–4の3,746 racesに固定し、5,336 full cohortや144 holdoutと混同しない。
- [ ] 全3,746 racesをno-bet込みでpaired rowに残し、primaryをenrolled race当たりprofit差にする。
- [ ] decision maskをfreezeしてからsettlementをjoinし、全raceのoutcome/candidate-hitを必須にする。
- [ ] hit時だけ一意・finite・positiveな払戻を必須とし、miss時の払戻空欄はreturn 0として許可する。
- [ ] 必要な結果・条件付き払戻の欠測や重複はrow dropでなくrun全体をINVALIDにする。
- [ ] ROI denominatorをarmごとに報告し、ROIだけをprimaryにしない。
- [ ] 高配当除外は全3,746 raceのsettled candidate return降順、同額時race_date/race_id昇順でtop1/top3共通race集合を作る。
- [ ] settled candidate returnを`official_wide_pay if candidate_hit else 0`に固定する。
- [ ] 共通除外raceでは両armのreturnだけを0にし、row/stakeを残してrace-ID set SHA-256を固定する。
- [ ] race-ID set正本をUnicode code-point昇順のunique JSON array、UTF-8/BOMなし/compact/末尾改行なしに固定する。
- [ ] winsor感度は共通candidate returnを2,000円でcapしてからarm eligibilityを適用し、row/stakeを残す。
- [ ] bootstrapはrace_date×venue_code cluster、100,000回、seed 20260814、PCG64に固定する。
- [ ] bootstrap分母を重複cluster込みのreplicate内race-row数、下限を`numpy.quantile(...,0.05,method="linear")`に固定する。
- [ ] cluster tupleのlexicographic順、race_id順、replicateごとのexact `rng.integers` callとrunner/environment hashを固定する。
- [ ] local sourceは`B_LOCAL_HASHED`で、tracked/canonical authorityではないと表示する。
- [ ] R0 audit config、calibrator summary、calibrator producerをpathとSHA-256で固定する。
- [ ] final official payoff診断と明記し、strict T-3・価格残存・ex-ante EVを主張しない。
- [ ] candidate-only projectionをsettlement/resultから物理分離してから将来runnerへ渡す。
- [ ] 本PRではcomparison/ROI replayを実行していないことを維持する。

## F. Governance boundary

- [ ] 暫定46/100の`BLOCKED_SCORE`を75点へ水増ししない。
- [ ] 本件を`infrastructure_safety_v1`へ偽装しない。
- [ ] PR #38 G1はno-authority / execution-forbiddenのまま扱う。
- [ ] G2または別のhuman-reviewed model-integrity laneを先にmergeする。
- [ ] tracked candidate packet producer commit/schemaを先に固定する。
- [ ] R0のformula/threshold/calibrator/cutoff/fit-data/label/lineage/digestsを固定する。
- [ ] R1のformula/effective thresholdとR0とのcalibrator/p_action parityを固定する。
- [ ] post-G2 `main`から新しいcanonical proposal/digestを作る。
- [ ] `APPROVED_TO_PREPARE`前に実装・synthetic testを行わない。
- [ ] 別の`APPROVED_TO_RUN`前に実データ、OOS、ROIを実行しない。
- [ ] shadow、production、BUY、mergeを別の人間判断に残す。

## G. This Draft PR

- [ ] 変更は3つのdesign fileだけである。
- [ ] `src/`、`config/`、model、candidate、value、BUY codeを変更していない。
- [ ] `research/REGISTRY.jsonl`を変更していない。
- [ ] canonical proposal、queue、run scope、approval eventを作っていない。
- [ ] 実データ・モデル・ROI testを実行していない。
- [ ] Draft解除とmergeは人間に残している。
