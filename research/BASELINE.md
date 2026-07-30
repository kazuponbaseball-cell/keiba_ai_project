# BASE-20260730 — 競馬AI 研究基準線

- Audit date: `2026-07-30` (`Asia/Tokyo`)
- Repository: `kazuponbaseball-cell/keiba_ai_project`
- Audited commit: `288dff5e86385908281428d5ed4f077625a43e4b`
- Commit date: `2026-07-03T18:21:52+09:00`
- Branch created for this work: `research/autonomy-v1`
- Baseline status: `DOCUMENTED_NOT_YET_REPRODUCIBLE`
- Production decision: `NO_CHANGE`

## 1. Executive conclusion

version-controlledな基準線は、race-relative順位を学習するridge baseline、当週推論、market-awareな最強版ticket/BUY、複数のstandalone評価scriptから成る。良いas-of部品とpurged walk-forward実装はあるが、Level 3研究基盤としては次が未達である。

- clean cloneから主要pipelineを実行できない。
- 標準baselineは単一recent-20% holdoutで、strict chronological outer OOSではない。
- Top3集合softmaxと確率契約の主要実装・testはローカルにあるが未追跡である。
- production builderは候補構築からBUYまでmarket情報を使うため、非オッズ研究候補生成器として使用できない。
- 最新のローカル開発OOSは確率整合性を示す一方、安定したROI優位をまだ示していない。
- strict T-3 + final priceのprospective Grade-O証拠は0行である。

したがって、当面のchampionは「採用model」ではなく、この文書で凍結する監査基準線である。現行BUYは変更せず、次の作業は再現性、outer OOS、candidate firewall、prospective captureとする。

## 2. 証拠の階層

| Tier | 定義 | 本文での扱い |
|---|---|---|
| A | audited commitに含まれるcode/config/docs | review可能な事実 |
| B | SHA-256を記録したlocal ignored artifact | provisional evidence。PRだけでは再現不能 |
| C | tracked docsに記載された過去の数値 | historical claim。promotion証拠ではない |

監査時のworktreeには多数の既存modified/untracked fileがあった。本PRはそれらを所有・stageせず、Tier Aの基準線とTier Bのローカル研究証拠を分離する。

## 3. 現状の構成

### 3.1 End-to-end map

| Stage | Main entrypoint | Input / output | Audit observation |
|---|---|---|---|
| JV intake | `src.pipelines.update_raw_data` | `data/inbox/jv` -> raw archive/state | 現在のparseは主にfile inventory |
| Normalization | `src/jv/build_normalized_tables.py` | historical CSV -> races/runners/results | JV archiveではなく既存historical CSVが土台 |
| TARGET import | `src/pipelines/import_target_entry_csv.py` | TARGET CSV -> weekly entry snapshot | 必須列all-nullでもwarning継続し得る |
| Feature build | `src/features/baseline.py:40-84` | pre-race/history -> model frame | shifted history部品はあるがlineageは列名中心 |
| Train dataset | `src.pipelines.build_train_dataset` | frame -> train/recent test CSV | recent 20%の単一split |
| Baseline train | `src.train.train_baseline` | rawまたはprepared CSV -> pickle/metadata | ridge ranker。固定pathへ上書き |
| Weekly inference | `src.pipelines.run_daily_inference` | entry snapshot -> prediction CSV | validated weekly datasetとpredict inputが一意でない |
| Runner prediction | `src.predict.predict_baseline` | as-of history + entry -> AI score/rank | 現在oddsはbaseline featureでなくpassthrough |
| Strongest tickets | `scripts/build_current_strongest_tickets.py` | prediction + live odds + context -> candidates/BUY | market-awareなproduction layer |
| Race-day wrapper | `scripts/run_current_strongest_line_update.ps1` | odds/body/track -> dashboard/LINE | 多数のartifactを生成する運用入口 |
| Robustness diagnostics | `scripts/validate_purged_walkforward_mcs_pbo.py` | historical tickets -> WF/PBO/MCS | 良い部品だが全研究へ強制されない |

### 3.2 Version-controlled baseline model

`config/data_pipeline.json:3-6` はactive feature contractを `config/baseline_features.json`、modelを `models/baseline/baseline_ranker.pkl` とする。

学習の主要仕様:

- `src/features/baseline.py:64-69` が確定着順からrace-relative `target_score`、win、Top3 targetを作る。
- `src/train/simple_ranker.py:27-77` は数値標準化、category one-hot、ridge回帰を実装する。
- `src/train/train_baseline.py:69-94` はmodelとmetadataを固定名で保存する。
- `config/baseline_features.json` のactive contractは数値27、category 11、generated numeric 226、generated category 2を宣言する。
- `config/baseline_features.json:333-367,393-396` は現在odds・人気をprediction passthroughと明記する。
- `src/features/baseline.py:28-37` のleak guardはfeature名の禁止keyword検査であり、source-time/descendant lineage検査ではない。

### 3.3 Split and evaluation

- `config/baseline_features.json` は `test_recent_fraction=0.2`。
- `src/features/baseline.py:2233-2242` は日付を昇順にし、直近20%を1つのtestへ置く。
- `src/evaluate/evaluate_baseline.py:50-77` は単勝/複勝ROI、win/Top3、人気等を診断する。
- `docs/EXTERNAL_AI_PROJECT_BRIEF.md:186-196` はT-5/T-3、年度walk-forward、purged WF、MCS/PBO、drawdown、payout concentration、threshold sensitivityを要求する。
- ただし標準train entrypointはそのouter protocolを強制しない。

### 3.4 Production ticket and BUY boundary

audited commitの `scripts/build_current_strongest_tickets.py` は次を行う。

- live単勝oddsとrunnerを結合し、market probability/overlayを作る: `:2120-2170`。
- live pair oddsの存在するpairを使い、pair probability、margin、expected ROIを作る。
- `select_tickets` のbase/strict gateでodds margin、expected ROI、live oddsを使う: `:4204-4349`。
- race/day上限を適用し、`runtime_action=BUY` を出す: `:4366-4388`。
- 発走後のrefreshによる新規BUYを防ぐ: `:4820-4853`。

この設計は現行production baselineとして凍結する。一方、新しい研究候補は別artifactでodds-freeにfreezeしなければならない。

監査時のdirty worktreeには、同scriptのformal BUY挙動をaudited commitから変える既存変更があった。この変更は本PRのscope外であり、commit・PRへ含めない。production controlは上記commitの内容とする。

## 4. Quantitative baseline

### 4.1 Local ridge model metadata — Tier B

Source: `models/baseline/baseline_metadata.json`

| Metric | Train | Recent temporal test |
|---|---:|---:|
| Races | 26,974 | 6,749 |
| Rows | 376,647 | 92,986 |
| Top1 win rate | 23.94% | 23.32% |
| Top1 Top3 rate | 53.67% | 54.78% |
| Top3 contains winner | 53.00% | 52.70% |
| Mean winner predicted rank | 4.336 | 4.316 |

- Run ID: `20260419_234804`
- Trained at: `2026-04-19T23:48:04`
- Temporal cutoff: `240210`（local metadata表記）
- Feature count: `271`
- SHA-256: `89fa609f5fa313082342cc9cca43424fed97d47a667e48641e94983feaca912a`

注意: このlocal metadataはmojibakeしたpath/fieldを含み、strict JSON parserで読めない。数値は行単位で監査したprovisional値である。

### 4.2 Local coherent Top3 research — Tier B

4つのchronological development fold、合計5,336 outer-test raceのlocal artifactを監査した。

| Model | Weighted/average set NLL | Top1 set hit | Wide top1 hit | Mass error |
|---|---:|---:|---:|---:|
| M0 linear set softmax | 4.701877 | 8.276% | 27.551% | 0.0 |
| M0 strength temperature-scaled | 4.696995 | 8.264% | 27.661% | 0.0 |
| M1C + M1A1 core | 4.669367 | 8.321% | 27.942% | 0.0 |
| M1C + M1A1 variance | 4.667477 | 8.414% | 27.961% | 0.0 |

M1C + M1A1 varianceはNLL上の最良local comparatorだが、artifact自身のdecisionはresearch onlyであり、formal BUY再開を許可しない。

Artifact hashes:

- M0 aggregate: `0f589fb6f7e153c90a75116d6f47b23e6b8a10d51f52e931a601c55910073672`
- M1C/M1A1 aggregate: `38602cc3162f897e05fb616e8eef94389727f138c733e27d1622e8ed98bb087c`

ordered Top3 contractのlocal audit:

- 5,336 races / 2,117,093 Top3 sets / 4 folds。
- `max_probability_mass_error = 1.7763568394002505e-14`。
- `max_marginal_reference_error = 1.1102230246251565e-16`。
- tolerance `1e-10`、all contracts pass。
- `candidate_uses_odds=false`、`formal_buy_changed=false`。

これは強いdevelopment evidenceだが、関連code/test/configはaudited commitに含まれず、version-controlled guaranteeではない。

### 4.3 ROI robustness — Tier B

Legacy purged walk-forward artifact:

| Metric | Value |
|---|---:|
| Period | 2025-01-05〜2026-02-15 |
| Tickets / races | 2,657 / 1,417 |
| Stake / return | 492,700円 / 442,070円 |
| ROI | 89.72% |
| Max drawdown | -79,160円 |
| Top5-return removed ROI | 82.52% |
| PBO | 88.57% |
| Average selected test ROI | 84.82% |

Source SHA-256: `f77b4173e1bd2b3e1517e3af8622fc5ed43adbc37c911f2b8044ccf5d8974f23`

最新local Top3-wide fixed-policy stability audit:

| Policy | Overall ROI | Latest fold ROI | Bootstrap p20 ROI | Decision |
|---|---:|---:|---:|---|
| candidate_all | 94.65% | 95.74% | 92.22% | NO_STABLE_PROFITABILITY |
| primary_confidence | 97.22% | 93.86% | 94.46% | NO_STABLE_PROFITABILITY |
| sensitivity_confidence | 98.34% | 96.43% | 95.39% | NO_STABLE_PROFITABILITY |
| abc_guard | 104.08% | 105.56% | 100.21% | POSITIVE_BUT_UNSTABLE |

`abc_guard` はthreshold sensitivityとfold calibrationを通過していない。全5,336 raceは既にmodel/policy reviewへ使ったdevelopment OOSであり、strict T-3 value replayでもない。formal W/umaren BUYは停止状態と記録されている。

prospective readiness:

- Strict T-3 event structure: pass。
- Strict prospective T-3 + final-price: `0 / 150`。
- Feature registry: 1,348 features。
- Grade-O approved: `0 / 1,348`。
- Value modelは最小行数までfitしない方針。

### 4.4 Historical headline ROI — Tier C

`docs/EXTERNAL_AI_PROJECT_BRIEF.md:34-45` は次を報告する。

| Variant | Tickets / races | Reported ROI |
|---|---:|---:|
| Base strongest | 217 / 191 | 375.9% |
| Ability-floor filter | 180 / 157 | 426.8% |
| Time-relative rank + floor | 143 / 126 | 459.4% |
| Top time-refined + floor | 35 / 31 | 716.4% |

同文書自身が小標本、過学習、上振れ、2026不足を警告する。これらはhistorical contextであり、Level 3のpromotion baselineには使用しない。

## 5. 重大な監査gap

| Severity | Gap | Evidence / consequence |
|---|---|---|
| P0 | Clean cloneが主要pipelineを実行できない | `.gitignore:13` の `data/` が `src/data/loaders.py` もignoreし、48 tracked moduleが未追跡loaderをimport |
| P0 | Top3 contractがHEADにない | Top3/ordered contract/testの主要fileがuntracked |
| P0 | Standard outer OOS不在 | baselineは単一recent-20% split。outer result再利用をRegistryで防げない |
| P0 | Candidate/price layer混在 | production builderはodds join、market overlay、expected ROI、marginで選別 |
| P0 | BUY regression test不在 | 約5,000行のbuilderにhard-coded gateが集中し、tracked golden testなし |
| P0 | Artifact再現性不足 | `data/`、`models/`、`outputs/`、test、dependency lockがtrackedでない |
| P1 | Retrain lineage不一致 | dataset build後、prepared CSVを指定せずrawから再生成してtrain |
| P1 | Inference inputの曖昧さ | validated weekly outputではなく元entry snapshotをpredictへ渡す経路 |
| P1 | Validationがfail-open | required inputやChampion/snapshot生成の一部がwarning継続 |
| P1 | As-of proof不足 | leakage guardは名前中心でsource/received timeの共通契約がない |
| P1 | Production source of truthが複数 | strongest line update系と別race-day runtime系が併存 |

## 6. 既存の良い土台

- target raceより前のhorse historyだけを推論へ結合する処理がある。
- 多数の履歴集計がshift/expanding priorを用いる。
- 発走後に新規BUYを作らない保護がある。
- purged walk-forward、PBO、簡略MCS、prior-year calibrationの実装部品がある。
- pre-race decision ledgerとChampion hash manifestの概念がある。
- tracked docsは長期OOS、T-5/T-3、drawdown、大当たり依存、threshold sensitivityを要求する。

本基盤はこれらを捨てず、共通contract、Registry、approval lifecycleへ接続する。

## 7. Frozen control declaration

このbaselineに対する研究runは次を守る。

- Production code control: audited commit `288dff5e...`。
- Baseline ID: `BASE-20260730`。
- 現行BUYのsource、threshold、stake、output pathを変更しない。
- research candidateは別manifestへodds-freeでfreezeする。
- Top3 contractとchronologyが失敗したrunはROIを評価しない。
- Tier B artifactはAUT-001/AUT-006で再現されるまでpromotion証拠にしない。
- 次の承認済み作業は `research/BACKLOG.md` と `research/REGISTRY.csv` から開始する。
