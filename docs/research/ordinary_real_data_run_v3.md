# Versioned Ordinary Real-Data Run Contract v3

## Governance route

`ordinary_real_data_run_v3` is a Research OS control-plane contract introduced by this
human-owned Draft governance pull request. Existing D-024 is its prerequisite and
evidence; D-029 is introduced by the same PR and is neither pre-existing nor
self-authorizing. It is not an ordinary racing hypothesis, does not receive a
Hypothesis Scorecard score, and is not self-authorized by
`infrastructure_safety_v1`. The containing commit must be reviewed and merged to
`main` by a human before the contract can become a trust root.

Merging this contract adds validation capability only. It does not create an experiment,
Registry event, run scope, approval, RUNNING receipt, real-data mount, model fit,
prediction, promotion, BUY, notification, order, or merge authority.

## Backward compatibility

The legacy v2 implementation remains in `scope_contract.py` and
`prepare_run_scope.py`. This change does not modify its `RUN_FIELDS`, normalizer,
canonical serialization, digests, material verifier, writer semantics, or historical
Registry events.

The ordinary dispatcher has exactly three outcomes:

1. no `run_scope_schema_version` field: dispatch to unchanged legacy v2;
2. exact `ordinary_real_data_run_v3`: dispatch to this contract;
3. null, misspelled, case-changed, or unknown version: fail closed.

Legacy v2 real-data RUNNING remains forbidden. Historical v2 bytes are never upgraded or
reinterpreted as v3.

## Canonical scope

The schema is
`research/schemas/ordinary_real_data_run_v3.schema.json`. The trusted normalizer is
`scripts/research/ordinary_real_data_run_contract_v3.py`, and the separate compiler is
`scripts/research/prepare_ordinary_real_data_run_scope_v3.py`.
Input metadata and the dependency lock use the separate exact schemas
`ordinary_real_data_input_manifest_v1.schema.json` and
`ordinary_real_data_environment_lock_v1.schema.json`. A successful EXP-034 release is
not consumable by EXP-033 until a separate, human-merged
`ordinary_real_data_output_attestation_v1` binds the exact receipt, result manifest,
artifact digest, and RUNNING event on GitHub main.

The canonical digest binds:

- schema version and `synthetic` or `real_data` execution kind;
- the complete proposal and proposal digest;
- an exact finite capability profile;
- execution commit plus code/config hashes;
- data/catalog, runner, training, target, lineage, label, fold, and environment bindings;
- interpreter path/version, the complete installed-distribution lock, locale, and timezone;
- random seed and exact repository working directory;
- ordered structured argv (`python -I -B <hash-bound-runner> --phase ... --config ...`;
  no shell, module dispatch, or free-form command);
- network disabled, path-and-phase read/write allowlists, and access class;
- fresh experiment-specific output root;
- timeout/CPU/memory/disk/model-fit/OOS/inference budgets;
- source cutoff, as-of, ordered phase plan, and append-only output seal;
- `formal_buy=false`, `send_order=false`, and `stake=0`.

All JSON is UTF-8, sorted-key compact canonical JSON, with LF termination and no
NaN/Infinity or duplicate keys. Set-like lists are sorted and duplicate-free. Command
and phase lists preserve their declared order.

## Finite capability profiles

The profile ID is not a caller label. The normalizer reconstructs the complete boolean
matrix and rejects omissions, unions, unknown fields, or mismatches.

### `synthetic_governance_v1`

This is a schema/compiler/preflight validation profile. Every row-read, write/seal,
canonicalization, training, validation, calibration, OOS, and target-inference
capability is false. Synthetic fixture unit tests run outside the real-data broker and
it can never obtain `real_data_execution_allowed=true`.

### `exp034_input_canonicalization_v1`

Allows only:

- `read_real_input_manifests`
- `read_real_runner_rows`
- `canonicalize_input_release`
- `read_historical_training_rows`
- `write_research_outputs`
- `seal_research_outputs`

Training, validation, calibration, untouched outer OOS, and target inference are false.
This profile is accepted only for `EXP-20260821-034`.

### `exp033_leakfree_research_v1`

Consumes an exact sealed EXP-034 release and allows one frozen fit, validation,
calibration, one untouched outer OOS evaluation, and one target research inference.
Input canonicalization remains false. This profile is accepted only for
`EXP-20260821-033`.

All profiles permanently deny production model writes, Champion changes,
candidate/value policy changes, formal BUY, notification, order, nonzero stake, merge,
and production promotion.

## Planned output versus sealed input

EXP-034 creates the feature input release and feature-lineage release. Their future
content hashes cannot honestly exist before the run. Its v3 scope therefore binds a
`planned_output_contract` (schema/declaration hash) and records the actual content hash
only in the immutable result manifest.

EXP-033 is a separate scope. It must use `sealed_input_artifact` and bind the actual
EXP-034 producer scope, durable receipt, successful result manifest, content hashes,
and one digest-addressed post-run output attestation fetched byte-exact from GitHub
current main. Local, unsigned receipt/result bytes alone are never producer authority.
The runner, training input, target input, feature-lineage, and label-reconciliation roles
must all appear in that single result seal and in EXP-033's typed read inventory. A
pending, placeholder, mutable `latest`, cross-run composition, or unsealed release is
rejected.

## Metadata-only preflight and access boundary

Before any row/blob mount, the verifier reads only canonical single-object metadata
payloads and checks their hash-bound wrapper. A data manifest must have the exact
`ordinary_real_data_input_manifest_v1` shape and enumerate the exact path, SHA-256,
access class, required capability, and phase of every row blob. The catalog repeats and
hash-binds that complete `metadata_manifest_refs` / `row_blob_refs` inventory:

- catalog/source release identity and manifest digest;
- metadata row/race/runner counts and race set;
- complete `source_event_time`, `received_at`, and `available_as_of` coverage;
- active/non-revoked state;
- runner-universe digest and target date;
- phase-specific read capability.

The catalog payload is hashed separately from its wrapper, avoiding a self-referential
manifest digest. Arbitrary JSONL or a row payload relabeled as metadata is rejected by
the finite manifest parser. Preflight opens only the catalog, typed manifests, committed
contracts, dependency lock, and (for EXP-033) the EXP-034 seal envelopes. It never opens
a catalog `row_blob_refs` path and returns `real_data_rows_opened=0`.

Read allowlist entries declare `metadata_manifest` or exactly one of
`runner_row_blob`, `canonicalization_source_row_blob`,
`historical_training_row_blob`, and `sealed_input_row_blob`. Each class maps to one
finite capability and phase; a row path cannot be opened by relabeling it as metadata.
Row/write access uses the code-owned broker, re-hashes the opened bytes, and requires all
of the following:

- exact v3 `real_data` scope and finite profile;
- exact, distinct, unused Prepare and Run human approval evidence;
- both comments unedited (`created_at == updated_at`) and revalidated;
- the exact RUNNING event human-merged into GitHub current main;
- execution-commit ancestry evidence;
- exact scope, capability, input, complete environment, command, and allowlist digests;
- exact metadata preflight receipt with zero rows opened;
- a fresh output-root reservation bound into the execution receipt;
- phase/path/access-class allowlist membership.

Receipt issuance and every broker access also re-observe the exact Git HEAD, actual cwd,
complete installed distributions, locale/timezone, `-I` interpreter isolation, and exact
process argv. Dirty or untracked worktree paths outside the frozen read/write set, and
ignored executable/importable `.py`, bytecode, native library, zip/egg/wheel, or shell
paths, fail closed. A pristine execution worktree is therefore mandatory.

A branch-local pending event, boolean argument, empty receipt, stale comment, or scope
digest alone is not authority.

## Approval lifecycle

The existing ordinary lifecycle remains:

```text
PREPARING
  -> RUN_APPROVAL_REQUIRED
  -> APPROVED_TO_RUN
  -> RUNNING
```

The Run grant is an unused GitHub User comment with exact body:

```text
APPROVED_TO_RUN <v3_run_scope_digest>
```

Its comment ID must differ from the Prepare grant and every prior Registry grant. The
proposal base and execution commit must both be ancestors of GitHub current main.
Immediately before RUNNING, both Prepare and Run comments are fetched again and every
immutable field is compared. Any deletion, edit, author/type change, digest drift,
scope-version drift, capability drift, input/environment/argv drift, or current-main
drift fails closed.

The branch-local v3 real-data RUNNING event deliberately records
`real_data_execution_allowed=false`, `execution_authorized=false`, and
`automatic_execution_allowed=false`. After that exact event is human-merged, the
verifier re-fetches live GitHub main, the complete Registry chain/grant history, proposal,
run scope, and both comments. The atomically persisted execution receipt also retains
both authority booleans as false. Effective `real_data_execution_allowed=true` exists
only as the broker's ephemeral decision for one exact phase after it re-observes the
approved argv, worktree, environment, live authority, receipt, and access path.
Synthetic and legacy scopes can never obtain that decision.

## Output sealing

Each scope uses a fresh `outputs/research/<experiment_id>/<run-id>` root. Existing,
linked, mutable-alias, or cross-experiment roots are rejected. Receipt, success/failure
manifest, and every role-to-file path are fixed before approval. The receipt and outputs
are written with create-exclusive semantics; writes are limited to the exact phase
allowlist and never overwrite a prior root.

The result manifest binds the run/receipt, code, config, input, and environment digests,
generated-at/as-of, artifact hashes, and row/race/runner counts. Success requires every
profile-specific artifact to be complete. The sealer reopens the exact on-disk bytes,
checks canonical JSON/JSONL or the finite opaque model role, and derives counts and
identities instead of trusting caller counts. Failure/crash must seal the partial artifact
list in an immutable failure manifest; partial output is never consumer-eligible.
At seal time the complete output-root tree is enumerated: only the durable receipt,
the exact declared artifacts, and the selected result/failure manifest may exist;
symlinks, transient directories, and unmanifested files fail closed. EXP-034 output
consumption additionally requires a later human-reviewed governance change that places
the digest-addressed output attestation on GitHub main.
EXP-034 canonical releases and EXP-033 model/OOS/prediction artifacts use separate roots
and manifest roles.

This PR binds `network_policy=disabled` and all compute/OOS/inference budgets but does
not itself install an OS-level sandbox or adapt the current EXP-033/034 runners to the
new `--phase` broker interface. Before any real run, an exact later execution commit must
provide that broker-only runner and a trusted supervisor that enforces network isolation,
timeouts/resources, one-shot counters, and crash-time failure sealing. Until then receipt
issuance/execution remains an explicitly disclosed hard blocker; callers must not bypass
the broker with direct file or socket access.
The live GitHub GET verification belongs to the control-plane receipt issuer before the
data-plane process starts; it is not permission for the data-plane runner to use network.

## Compiler example

After the governance PR is human-merged and a future experiment execution commit has
all required metadata contracts, compile first with a dry run:

```text
python scripts/research/prepare_ordinary_real_data_run_scope_v3.py \
  EXP-20260821-034 \
  --input research/drafts/EXP-20260821-034.run_scope_v3_input.json \
  --queue-file research/queue/EXP-20260821-034.json \
  --output research/scopes/EXP-20260821-034.run.json \
  --dry-run
```

The compiler verifies execution-commit blobs and metadata manifests only and refuses
to overwrite an existing scope or output root. It never opens a real-data row.

## Required next steps for EXP-034

After human merge of this governance PR:

1. refresh the EXP-034 research branch from that main;
2. resolve the remaining PR #52 provenance/status/post-time/training-feature/semantic
   replay blockers without opening rows outside approved scope;
3. implement and review the exact `-I` broker-only EXP-034 phase runner plus trusted
   network/compute/crash supervisor in its execution commit;
4. freeze exact source catalog metadata and typed manifest-to-row inventory, execution
   commit, complete environment, argv,
   EXP-034 capability profile, read/write allowlists, and planned-output contracts;
5. compile and review the v3 run scope;
6. register `RUN_APPROVAL_REQUIRED`;
7. obtain a new, unused exact `APPROVED_TO_RUN <digest>` human comment;
8. register and human-merge `APPROVED_TO_RUN`;
9. refresh from main, revalidate both approvals, register and human-merge RUNNING;
10. from a pristine isolated worktree, revalidate live main and metadata, atomically
    persist the receipt, and only then permit each allowlisted row mount.
11. after a successful immutable seal, create a digest-addressed output attestation,
    submit it through a separate human-reviewed governance PR, and merge it to main
    before any EXP-033 scope may bind the release.

Until every step passes, EXP-034 materialization and all EXP-033 training/OOS/inference
remain fail closed.
