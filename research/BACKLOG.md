# ROI改善 研究バックログ

- 版: `v1.0`
- 基準線: `BASE-20260730`
- 状態: 初期提案。すべて `PROPOSED` であり、実行承認ではない。

## 優先順位の根拠

監査の結論は、次のとおりである。

1. Gitの基準線はclean checkoutから主要pipelineを再現できない。
2. committed baselineは単一recent-20% holdoutであり、strict chronological outer OOSではない。
3. coherent Top3集合softmax、4-fold研究、確率契約testはローカルに存在するが未追跡である。
4. production ticket builderはmarket-awareであり、オッズ非依存の研究候補生成器としては使用できない。
5. 最新のローカル開発OOSでは、固定wide policyの主要系はROI 100%未満、正のpolicyも不安定である。
6. strict T-3 + final-priceのprospective Grade-O行はまだ0であり、価値modelを選べる段階ではない。

したがって、最初のROI改善はmodel探索ではなく、再現可能なlineage、odds-free候補freeze、strict outer OOS、prospective価格証拠を作ることである。

## 一覧

| ID | Priority | Theme | ROIへの経路 | Status | 依存 |
|---|---|---|---|---|---|
| AUT-001 | P0 | Reproducible baseline & lineage freeze | 偽の改善と再実行不能を排除 | PROPOSED | なし |
| AUT-002 | P0 | Chronological outer-OOS harness | test再利用による過大評価を抑制 | PROPOSED | AUT-001 |
| AUT-003 | P0 | Odds-free candidate firewall | model edgeと価格edgeを分離 | PROPOSED | AUT-001 |
| AUT-004 | P0 | Top3 contract & BUY immutability | 確率破綻と本番driftをfail-close | PROPOSED | AUT-001 |
| AUT-005 | P0 | Strict T-3 Grade-O capture | 実購入可能価格で価値を測定 | PROPOSED | AUT-003, AUT-004 |
| AUT-006 | P1 | Frozen Top3 baseline replay | 比較可能なchampion基準を確立 | PROPOSED | AUT-001〜004 |
| AUT-007 | P1 | Non-odds conditional pair challenger | axis/partner同時失敗を減らす | PROPOSED | AUT-006 |
| AUT-008 | P1 | Fixed-candidate action calibration | 過大確率を抑え選択の質を安定化 | PROPOSED | AUT-006 |
| AUT-009 | P1 | Prospective value confirmation | freeze後価格で正のROIを確認 | PROPOSED | AUT-005, AUT-008 |
| AUT-010 | P2 | Negative-result registry | hypothesis churnと多重探索を削減 | PROPOSED | AUT-001 |
| AUT-011 | P2 | Drift & coverage monitor | edge消失時の損失を早期停止 | PROPOSED | AUT-005, AUT-006 |

## P0 — 研究可能性を作る

### AUT-001: Reproducible baseline and lineage freeze

仮説: code、data、config、fold、candidate、modelをimmutable manifestへ固定すれば、見かけ上のROI改善と実行経路の取り違えを除去できる。

主な作業:

- `.gitignore` の `data/` が `src/data/` まで除外する問題を別PRで修正する。
- `src.data.loaders`、依存manifest/lock、最小smoke testをversion controlへ入れる。
- retrainが生成したtrain/test artifactを実際にmodelへ渡す。
- daily inferenceのvalidated inputとpredict inputを一意にする。
- model/data/config/git/environment/fold hashを持つrun manifestを作る。
- `latest` 上書きではなくrun ID付きartifactを正とする。

提案gate:

- fresh clean checkoutでimportと最小runが成功する。
- 必須hash、timestamp、schema versionの欠損が0。
- 同じmanifestの2回実行でcandidate digestが一致する。
- 同じ4 outer folds、5,336 raceのbaseline replayが許容誤差内で一致する。

変更禁止: 現行BUY、stake、order、通知。

### AUT-002: Chronological outer-OOS harness

仮説: 全実験を共通fold manifestとnested time splitへ強制すれば、開発OOSの再利用による過大評価を抑えられる。

主な作業:

- expanding `train -> validation -> calibration -> outer test` の4foldを中央manifest化する。
- purge/embargo、race grouping、fit/calibration cutoffをassertする。
- feature selection、L2、temperature、thresholdの選択履歴を記録する。
- outer testを見たvariantは同じtestで再選択できないようRegistryへ記録する。

提案gate:

- 全foldで `train_max < val_min < cal_min < test_min`。
- 区間間race overlap、test間overlap、`NaT` がすべて0。
- artifact fit/calibration endが各test race dateより前。
- empty foldはskipではなくfailure。
- baselineとchallengerのfold manifest hashが同一。

### AUT-003: Odds-free candidate lineage firewall

仮説: candidateを価格情報から切り離せば、予測edgeと後付け価格filterを識別でき、prospective ROIを正しく評価できる。

主な作業:

- candidate feature whitelistとlineage graphを作る。
- 対象raceのodds、人気、market、payoff、ROIと全descendantをtaint扱いする。
- 過去市場proxyも既定禁止とし、例外は個別承認する。
- price join前にcandidate manifestをpersistし、read-back hashを固定する。
- oddsを大きく変えたnegative-controlを追加する。

提案gate:

- forbidden lineage検出が0。
- `candidate_uses_odds=false`。
- price join前後でcandidate key、rank、tier、digestが100%一致。
- odds perturbationでもcandidate digestが不変。
- research artifactは `formal_buy=false`、`send_order=false`、`stake=0`。

### AUT-004: Top3 probability contract and BUY immutability

仮説: canonical contractとproduction characterizationをtracked testにすれば、研究追加によるsilent driftを防げる。

主な作業:

- 全 `C(n,3)`、label一意性、finite/nonnegative、`sum q=1` をassertする。
- `q(set)` からwide/horse marginalを導出し、race massをassertする。
- 取消・同着・欠損・3頭未満のfixtureを追加する。
- すべてのconsumerが共通assertを通るようにする。
- 現行BUYのsource/config hashとgolden fixtureを作り、research PRでの差分を0要求する。

提案gate:

- `max |sum q-1| <= 1e-10`。
- `max |sum p_wide-3| <= 1e-10`。
- probability/marginal reference error `<= 1e-10`。
- candidate生成前にcontract failureがraiseされる。
- production control path diff、BUY fixture diffが0。

### AUT-005: Strict T-3 Grade-O prospective capture

仮説: candidate freeze後の購入可能時点価格を蓄積すれば、final payoffを代用したdevelopment ROIから、実運用可能なvalue評価へ移行できる。

主な作業:

- candidate freeze、persist acknowledgement、T-3 quote、final pre-close quote、resultを別eventとして記録する。
- source timeとreceived timeを両方保存する。
- missing quote、取消、発売停止、runner/pair key不一致をfail-closeで分類する。
- 150行未満ではvalue modelをfitしない。

提案gate:

- 初期150 joint rows、望ましくは300以上。
- 12 race day以上、2 venue以上。
- as-of違反、candidate-by-odds変更、pair mismatchが0。
- strict T-3 + final pre-close + resultのlineageが全行で追跡可能。

## P1 — ROIを改善する仮説

### AUT-006: Frozen Top3 baseline replay

仮説: ローカルのcoherent Top3研究基準をversioned manifestで再現できれば、以後のchallengerを同じ尺度で比較できる。

固定対象:

- 4 outer folds / 5,336 races。
- M0 non-odds strength baseline。
- M1C + M1A1 variantはresearch comparatorであり、即championではない。
- ordered Top3 / wide mass contract。

提案gate:

- race、set、candidate digestがmanifestと一致。
- 4foldすべて再現し、mass toleranceに合格。
- probability/ranking metricsが保存済みartifactと許容誤差内。
- ROIをmodel/feature選択に使わない。

### AUT-007: One preregistered non-odds conditional pair challenger

仮説: `min_ability_floor`、`pair_scenario_variance`、`pair_clash_score`、`pair_front_closer_complement` の1 feature blockは、Top3集合softmaxの確率整合性を保ちながらaxis/partner同時失敗を減らす。

制約:

- 1実験で1 blockだけを追加する。
- 現行candidate/BUYへ接続しない。
- odds、人気、払戻をfeatureやgateに使わない。
- baselineと同じfold、coverage、temperature手順を使う。

screen gate案:

- non-missing raceが1,000以上。
- 効果方向が4fold中3fold以上で一致。
- conditional residual contrast `>= 0.01`。
- PSI `<= 0.25`。

selector gate案:

- wide MRR改善が評価foldの3分の2以上。
- recall@5がnon-inferior。
- top1 hit `+0.5pp` またはchanged-pair net gainが2fold以上で正。
- binary logloss/Brierが悪化しない。
- race-cluster bootstrap changed-gainの10%点が0以上。

### AUT-008: Fixed-candidate action calibration

仮説: candidateを固定したままprior-fold-onlyでaction probabilityを較正すれば、pairを変えずに過大確率と不安定なconfidence gateを改善できる。

制約:

- pair、rank、tierは固定。
- calibrationは各testより前のfoldだけでfitする。
- odds、ROIでcalibratorやthresholdを選ばない。
- outputはshadowのみ。

提案gate:

- pooled loglossとBrierがnon-inferior。
- fold別observed/predictedがすべて `0.85〜1.15`。
- ECEが悪化しない。
- candidate digestの変化が0。

### AUT-009: Frozen prospective value and price-retention confirmation

仮説: AUT-005で凍結した候補は、T-3から最終価格までedgeを保持し、prospective ROIを残す。

オッズはcandidate freeze後の評価にだけ使用する。model、candidate、rank、tierの変更は禁止する。

limited gate案:

- 400 bets以上、80 hits以上。
- ROI `>= 125%`、latest period `>= 100%`。
- 最大配当除外ROI `>= 115%`、上位3配当除外ROI `>= 105%`。
- race/day cluster bootstrapの20%点 `>= 100%`。
- threshold sensitivityの最小ROI `>= 100%`。
- 最大1配当/上位3配当の利益share `<= 25% / 50%`。

stretch gate案:

- Grade-O 200行以上。
- ROI `>= 150%`、上位3配当除外ROI `>= 120%`。
- bootstrapの10%点 `>= 100%`。
- profitable periodが2以上。
- contract違反とcandidate-by-odds変更が0。

どちらを満たしても自動promotionしない。人間のshadow承認を必要とする。

## P2 — 研究速度と損失制御

### AUT-010: Negative-result and hypothesis budget

仮説: negative resultをhash付きで登録すれば、条件、休養、状態、関係者、race mechanicsの同型探索を減らし、研究budgetを有望な識別問題へ集中できる。

提案gate:

- 過去runの仮説、period、feature、result、artifact hashをRegistryへ記録する。
- `REJECTED` 仮説の再実行は、新データ、新lineage、または事前登録した識別可能な変更がある場合だけ許可する。
- outcomeを見て名称だけ変えた再探索を禁止する。

### AUT-011: Drift and coverage monitor

仮説: probability、missingness、lineage、coverage、price retentionのdriftを検出すれば、edge消失時の実運用損失を抑えられる。

監視対象:

- fold/半期/venue/field size別NLL・Brier・MRR・recall。
- source-time、missingness、unseen category、runner universe差。
- candidate coverage、abstention、価格欠損、slippage。
- payout concentration、drawdown、observed/predicted。

提案gate:

- thresholdは過去期間で事前登録する。
- 違反時はshadowをfail-closeし、BUYを変更しない。
- drift警告を新しいfeature/thresholdの自動採用に使わない。

## 明示的に後回しにする研究

- 新しい馬券種への拡張
- stake最適化・Kelly最適化
- oddsを使うcandidate reranking
- 現行BUY thresholdの探索
- 追加featureを大量投入するmodel zoo
- strict Grade-O未達のまま行うvalue model fit

これらは、AUT-001〜005が完了し、承認済みprospective証拠が得られるまで開始しない。
