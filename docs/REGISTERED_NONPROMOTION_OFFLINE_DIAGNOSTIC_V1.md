# 登録済み軽量 offline 非昇格診断 v1

## 目的と適用範囲

`registered_nonpromotion_offline_diagnostic_v1` は、固定 recipe
`historical_ai_duplicate_gate_impact_v1@1` の過去影響だけをローカルで確認する、一件限定の診断 route です。対象は folds 2–4 の 3,746 races（fold2: 1,661、fold3: 1,653、fold4: 432、期間 2025-01-05〜2026-02-15）、固定 2 arms の D0/D1、固定 100,000 回 bootstrap です。threshold、cohort、variant、seed、metric、入力、出力先を実行者が変更する機能はありません。

この route は診断専用です。結果は `DIAGNOSTIC_NONPROMOTION / B_LOCAL_HASHED` であり、`confirmatory=false`、`promotion_eligible=false`、`score_credit=0`、`formal_buy=false`、`send_order=false`、`stake=0` です。shadow、production、正式 BUY、注文、通知への遷移はありません。実装・synthetic fixture 検証では実データの ROI 計算を実行していません。

## 実行前提

次の条件をすべて満たす必要があります。

- この実装を含む commit が人間により `main` へ merge 済みであること。CI 成功、review、PR Ready、chat 指示だけでは有効化にも実行承認にもなりません。
- checkout が GitHub で検証した current `main` と同一の clean worktree であること。
- Python 3.11 または 3.12、および固定された NumPy 2.4.3 を使用すること。
- canonical run では fail-close の GitHub remote 再検証を完走できるよう、non-empty `GH_TOKEN` または `GITHUB_TOKEN` を environment に設定すること。rate-limit 回避のために remote 検証回数や検証対象を弱めることはしません。token 値は artifact や error 出力へ保存しません。
- `--source-root` の下に、次の固定 path・SHA-256 の raw source が存在すること。

| role | `--source-root` からの固定 path | SHA-256 |
| --- | --- | --- |
| `diagnostic_master` | `outputs/analysis/umaren_wide_rebuild_v1/wide_diagnostic_master_v1/wide_diagnostic_master_v1.csv` | `697142b64e8052b212731dc0319ccafb7f61ac29dbc46f67385f9ae050129de9` |
| `p_action_artifact` | `outputs/analysis/umaren_wide_rebuild_v1/m1c_action_calibration_offset_v1/m1c_action_calibration_offset_oos_predictions.csv` | `34f56b5a61261bd9b6cfd38797b65bd88415d0778d98cef29eebfbe2f09e513c` |
| `official_payoff_source` | `data/processed/target/wide_payoffs.csv` | `b94b0c0ea2ce4424d70432f7d070a9083d01850876a710432ec5b98538070d83` |

`--source-root` は raw source を置く root であり、repository root と同じでも別でも構いません。ただし、絶対解決した local path、安全な通常 file、固定 hash のすべてを満たす必要があります。

## 固定 workflow

以下は repository root から実行します。runner が出力する JSON を保存し、値を手入力で作らないでください。
canonical public API / CLI は GitHub provider や clock の caller 注入を受け付けず、read-only GitHub provider と実 UTC 時刻を内部生成します。fixture provider / clock は非公開 synthetic worker にだけ閉じています。

### 1. candidate / settlement projection を materialize する

```powershell
python scripts/research/registered_nonpromotion_offline_runner_v1.py --root . materialize --source-root "D:\path\to\source-root"
```

materializer は raw file を開く前に、実装を含む commit が検証済み current GitHub `main` であることと clean worktree を確認します。その後、固定された 3 source の exact bytes と SHA-256 を検査し、次の固定 artifact だけを新規作成します。

- candidate: `outputs/research/registered_nonpromotion_offline_materialized/historical_ai_duplicate_gate_impact_v1/candidate_projection.jsonl`
- settlement: `outputs/research/registered_nonpromotion_offline_materialized/historical_ai_duplicate_gate_impact_v1/settlement_projection.jsonl`
- manifest: `outputs/research/registered_nonpromotion_offline_materialized/historical_ai_duplicate_gate_impact_v1/materialization_manifest.json`

candidate projection は settlement source を開く前に作成・fsync・seal されます。materializer は D0/D1 の判定、hit 率、profit、ROI、bootstrap、threshold、variant を計算または出力しません。

official wide settlement は、enrolled race ごとに重複のない 3〜7 payout pairs があることを structural fail-close 条件とします。1〜2 rows しかない部分的な race を miss とみなして ROI に混ぜることはありません。一方、固定 SHA-256 は登録済み local artifact の exact bytes を認証するもので、同着等における対外的な official pair-set の完全性を独立に証明するものではありません。この限界も `B_LOCAL_HASHED` / nonpromotion 扱いの理由です。

raw CSV には outcome、label、payoff、popularity、odds、price、market 等の非許可列が同居していても構いません。materializer が扱うのは role ごとの固定 `usecols` だけです。CSV の header / record を許可列抽出のために token 化することはありますが、非許可列を semantic な型へ変換したり、判断へ使用したり、projection や manifest に永続化したりしません。official payoff は settlement projection のために `official_payoff_source` の固定許可列からのみ取得します。

### 2. run scope を seal する

```powershell
python scripts/research/registered_nonpromotion_offline_runner_v1.py --root . compile-scope --source-root "D:\path\to\source-root"
```

scope seal も、post-merge の clean verified current GitHub `main` と同じ raw source root を要求します。seal 中に 3 raw source の exact bytes をもう一度読み、candidate projection、settlement projection、manifest を決定論的に再生成します。保存済み artifact と byte-for-byte で一致しない場合は scope を作りません。

scope の保存先は caller が指定できません。canonical scope digest が `RUN_SCOPE_DIGEST` の場合、固定 path は次です。

```text
outputs/research/registered_nonpromotion_offline_scopes/RUN_SCOPE_DIGEST.run.json
```

成功時の JSON には `run_scope_digest`、digest 由来の `run_scope_path`、必要な approval comment が出力されます。scope file は overwrite できません。

### 3. scope seal 後に人間の run approval を得る

allowlist に含まれる GitHub `User` が、対象 Issue に次の exact body を新規 comment します。

```text
APPROVED_OFFLINE_NONPROMOTION_DIAGNOSTIC_RUN <run_scope_digest>
```

comment は scope seal より後に作成されている必要があります。Issue 番号と GitHub comment ID を控えてください。merge、過去 comment、編集済み comment、bot、別 digest の comment は承認になりません。

### 4. 固定 run を一度だけ実行する

```powershell
python scripts/research/registered_nonpromotion_offline_runner_v1.py --root . run `
  --run-scope-digest <run_scope_digest> `
  --source-root "D:\path\to\source-root" `
  --issue-number <issue_number> `
  --comment-id <comment_id>
```

run は digest から上記固定 scope path を導出し、任意の scope file や output root を受け付けません。固定 output root は次で、実行前に存在していてはいけません。

```text
outputs/research/registered_nonpromotion_offline/historical_ai_duplicate_gate_impact_v1
```

protected data の access 順序は固定です。

1. GitHub approval、current `main` ancestry、APPROVERS、policy、recipe、schema、runner、input binding を検証する。
2. exclusive start より前の control plane で固定 3 raw source を読み、decision/metric/ROI を計算せず projection と manifest を再生成し、保存済み bytes と完全一致することを再検証する。この preaccess provenance 検証に失敗した場合、run は未開始のまま fail-close する。
3. raw bytes を破棄した後、candidate open 直前の approval checkpoint を remote 再検証する。
4. `approval_evidence_initial.json`、`approval_evidence_before_candidate.json` と、candidate checkpoint の evidence digest に bind した local start receipt を private staging directory で fsync し、これらを含む directory を fixed output root へ atomic rename する。この rename が exclusive start であり、それ以前の staging failure は run を消費しない。
5. candidate projection だけを open・検証し、同じ sealed bytes を使う logical replicas `clean_a` / `clean_b` で D0/D1 decision を各 1 回 freeze する。
6. 両 replica の decision projection が一致したことを確認し、`decision_freeze_receipt.json` を保存する。
7. その後に限り settlement projection を openし、固定 100,000 回 bootstrap を含む診断を行う。
8. 両 replica の scientific projection の意味論的一致、artifact hash、result 公開直前の approval checkpoint を再検証する。`approval_evidence_before_result.json` を canonical 保存し、result と `result_seal_receipt.json` をその evidence digest へ bind した後、`RNOD_RESULT_SEALED` を永続化してから `result.json` を atomic publish する。

canonical run は start receipt より前の provenance control plane でだけ raw source を開き、再生成後に明示的な raw root / byte reference を破棄します。exclusive start 後、固定 runner code は raw source を再 open せず、candidate-only projection の decision freeze と、その後の settlement-only projection の診断だけを行います。ただし same process / filesystem の軽量 route であり、raw source 自体を unmount したり、ACL / namespace / filesystem capability で不可視化したりする保証はありません。

## artifact と失敗時の扱い

正常完了時の主な artifact は固定 output root 内の `approval_evidence_initial.json`、`approval_evidence_before_candidate.json`、`start_receipt.json`、`decision_freeze_receipt.json`、`approval_evidence_before_result.json`、`result_seal_receipt.json`、`result.json` です。approval evidence 本体には verified Issue / comment ID と remote trust metadata が残り、各 self-digest と receipt/result binding により事後再監査できます。

exclusive start receipt より前の contract failure は `BLOCKED_PREACCESS` です。start 後の crash、drift、contract failure は terminal な `INVALID_AFTER_START_NO_RETRY` となり、`INVALID.json` には安定した status / reason code だけを保存します。exception 文字列、race row、metric、profit、ROI 等の数値は INVALID artifact に保存しません。replica は各 1 attempt、retry は 0 で、片方の結果を選ぶことも output を overwrite することもできません。

process hard-crash 後に fixed output root / start tombstone だけが残った場合、次回 invocation は candidate、settlement、raw source、GitHub API を開く前に local chain を分類します。completed chain は再実行せず `ALREADY_COMPLETED_NO_RERUN`、未完了または壊れた chain は exact `INVALID_AFTER_START_NO_RETRY` として terminal 化します。filesystem 上の `INVALID.json` の有無から例外種別を推測せず、runner 内部の専用 terminal status を使用します。run 全体は fixed output の sibling lock file に対する OS-backed nonblocking lock を保持し、同時 invocation は output を変更せず `LOCAL_RUN_LOCK_HELD` で停止します。result の pre-commit temp は output root 外の固定-prefix sibling に限定し、lock 取得後の crash recovery で安全な通常 file だけを除去します。

## 分離と再現性の保証範囲

`clean_a` と `clean_b` は、同一 process 内で同じ sealed input bytes を用いる logical replicas です。独立 process、独立 machine、OS-level isolation を意味しません。

environment binding は Python executable hash、Python minor、NumPy version等の固定による軽量保証です。NumPy wheel / `RECORD`、native binary、BLAS build 全体の supply-chain digest を独立に証明するものではありません。same-process replicas も共通 dependency drift を独立に検知するものではなく、これも lightweight / nonpromotion の限界です。

workload 区間では、固定 workload code が利用し得る common socket / URL / subprocess API と environment view に application-level deny guard を置き、固定コード自身も network、free-form subprocess、credential を使用しません。一方、control plane では current commit と worktree cleanliness の確認に固定された read-only `git rev-parse HEAD` / `git status` subprocess を使用し、GitHub approval の read-only 検証も行います。この guard は pre-imported alias、native call、file credential access 等まで網羅する capability sandbox ではなく、subprocess や network を OS 全体で遮断したという保証もありません。

この軽量 route が提供する single-use enforcement は、固定 path の exclusive local receipt による best effort です。durable remote ledger、global CAS、remote witness、global replay proof、rollback / fork resistance はありません。file 削除、repository rollback、別 clone、別 machine、strict route を横断した再実行を技術的に完全防止するものではありません。結果をそれらの保証がある証拠として扱わないでください。

path traversal、UNC、ADS、既存 symlink / junction / reparse point は拒否します。ただし pathname 検証と file open の間に別の local process が path component を交換する concurrent filesystem race を、descriptor-based OS sandbox と同等には防止しません。固定 SHA-256 と再読検証は異なる scientific bytes を拒否しますが、local adversary に対する完全な path-provenance 保証ではありません。

この offline result は研究 family 全体での global consumption を構成しません。後続の strict result を独立な証拠として主張するには、この offline exposure を後続 scope と評価解釈へ明示的に binding する必要があります。
