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
