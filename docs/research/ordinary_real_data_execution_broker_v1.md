# Ordinary real-data execution broker / supervisor v1

## Status

This document describes the operational boundary of:

- `scripts/research/ordinary_real_data_execution_broker_v1.py`
- `scripts/research/ordinary_real_data_supervisor_v1.py`

It is an implementation and operations note under the already merged D-029 / ordinary
real-data run contract v3. It does not extend D-029, create a new governance decision,
change the v3 contract or schemas, or grant execution authority. D-029 and the canonical
v3 scope, receipt, access, and output-sealing rules remain authoritative if this document
and an implementation detail ever differ.

At this version, real-data execution is deliberately unavailable. No supported OS
isolation backend exists, no real-data child can be started, and there is no fallback to
the synthetic backend. The current operational result is therefore:

```text
real_data_ready=false
real_data_rows_opened=0
real_data_execution_allowed=false
execution_authorized=false
automatic_execution_allowed=false
formal_buy=false
send_order=false
stake=0
```

This work does not read a real-data row, make a lifecycle transition, add a Registry
event, run EXP-033/EXP-034, fit a model, perform OOS evaluation or inference, calculate
BUY, notify, or send an order.

## Authority and non-goals

The broker and supervisor are consumers of the frozen v3 contract. They do not replace
or reinterpret:

- D-029;
- `ordinary_real_data_run_v3` canonical normalization and digest semantics;
- live GitHub and Registry authority verification;
- metadata preflight, execution-receipt, read/write allowlist, or output-seal rules;
- EXP-033/EXP-034 separation and later output-attestation requirements; or
- the invariant that durable governance artifacts remain non-executing.

The v1 modules add typed boundaries and fail-closed scaffolding. A Python constructor
token or naming a method “broker” is not an OS security boundary. Until the real process
is isolated from direct files, sockets, alternate imports, and unapproved child
processes, the modules must not be described as a non-bypass real execution system.

## Component responsibilities

| Component | Responsibility | Explicit limitation |
|---|---|---|
| `OrdinaryRealDataExecutionBrokerV1` | Validate exact v3 real scope and typed access claims; delegate receipt, metadata, row, sealed-input, output, and seal operations to the frozen v3 API | Every real row/output/receipt/seal operation first requires all three host enforcement planes; none are currently available |
| `OrdinaryRealDataExecutionSupervisorV1` | Validate the exact phase command and require a ready real broker | It never launches a real child and ends with `REAL_EXECUTION_BACKEND_NOT_IMPLEMENTED` even if a future probe reports all planes enforced |
| `SyntheticFixtureBrokerV1` | Read hash-bound bytes from an explicitly marked synthetic fixture root and count only synthetic access | It has no v3 receipt, live-authority, real-row, output-authority, or real-seal API |
| `SyntheticExecutionSupervisorV1` | Exercise deterministic subprocess, timeout, output-manifest, bounded log, and synthetic seal behavior | It provides no OS network isolation, broker-only filesystem, CPU/memory/disk enforcement, or real authority |
| `SyntheticImmutableSealPort` | Create one exclusive synthetic success or failure manifest | Its manifest is not a v3 receipt, result manifest, failure manifest, or consumer attestation |

`open_metadata` is a metadata-only v3 preflight operation and does not require the three
real host enforcement planes. It must never be relabelled as row access. It also does
not grant row authority or prove that real execution is ready.

## Current host enforcement status

`observe_real_host_enforcement()` currently returns
`UNSUPPORTED_FAIL_CLOSED` for all three planes on every platform. In particular, the
current Windows status is not “partially enforced” and must be reported exactly as:

| Platform | Network isolation | Broker-only filesystem isolation | Resource supervision | Real child permitted |
|---|---|---|---|---|
| Windows (current v1) | `UNSUPPORTED_FAIL_CLOSED` | `UNSUPPORTED_FAIL_CLOSED` | `UNSUPPORTED_FAIL_CLOSED` | No |
| Linux (current v1) | `UNSUPPORTED_FAIL_CLOSED` | `UNSUPPORTED_FAIL_CLOSED` | `UNSUPPORTED_FAIL_CLOSED` | No |
| Other / unknown (current v1) | `UNSUPPORTED_FAIL_CLOSED` | `UNSUPPORTED_FAIL_CLOSED` | `UNSUPPORTED_FAIL_CLOSED` | No |

Readiness is conjunctive: all three statuses must be `ENFORCED`. Missing any one plane
prevents receipt issuance, row or sealed-input delivery, output opening, sealing, and
child launch. The stable fail-close order is:

1. `NETWORK_ISOLATION_UNAVAILABLE`
2. `BROKER_ONLY_FILESYSTEM_ISOLATION_UNAVAILABLE`
3. `RESOURCE_ENFORCEMENT_UNAVAILABLE`

On the current Windows implementation, the first surfaced readiness failure is therefore
`NETWORK_ISOLATION_UNAVAILABLE`; child processes started and real-data rows opened both
remain zero. This ordering is diagnostic only and does not imply that the later planes
are available.

## Target real-mode operation

The following is the required architecture for a later execution implementation. It is
not implemented and does not authorize a run.

1. The control plane canonicalizes an exact `ordinary_real_data_run_v3` scope and selects
   one frozen phase. Versionless, legacy v2, synthetic, and non-canonical scopes fail.
2. A code-owned host probe obtains non-self-asserted evidence for network isolation,
   broker-only filesystem isolation, and resource supervision. If any plane is not
   `ENFORCED`, no receipt or child is created.
3. With zero row opens, the control plane performs the v3 metadata preflight and the
   exact live GitHub, Registry, ancestry, approval, worktree, environment, and output-root
   checks required by D-029.
4. The exact v3 issuer persists the durable execution receipt with create-exclusive
   semantics. The receipt remains non-executing; it is evidence bound to the scope, not a
   bearer token.
5. A trusted OS backend enters the sandbox before launch and binds the resulting child
   process handle, sandbox identity, and broker channel to that proof.
6. The supervisor starts exactly one approved phase with the frozen argv, cwd,
   environment, timeout, and hashes. It uses no shell and exposes no general filesystem
   or network path to the data-plane process.
7. The child obtains row bytes and output writes only through typed broker IPC. For every
   access, the host broker rechecks phase, path, access class, capability, content hash,
   live v3 authority, and the phase-bound receipt context before delivering bytes or
   accepting output.
8. The supervisor enforces the frozen wall-time and resource/counter budgets over the
   entire process tree. Broker-side byte, disk, row, model-fit, OOS, and inference counts
   must be authoritative; child-reported counts are not sufficient.
9. Success or failure is sealed through the exact v3 sealer. Partial or unexpected
   output is never consumer-eligible. A later, separate human-reviewed attestation is
   still required where D-029 requires it.

There is no permitted path from an unsupported real backend to a fake backend, from a
synthetic plan to a real scope, or from metadata access to row access.

## Exact process binding

`frozen_phase_command()` is validation-only. It accepts one canonical v3 `real_data`
scope and requires the exact eight-element structured argv:

```text
<interpreter> -I -B <runner.py> --phase <phase_id> --config <config_path>
```

The following are fixed by the v3 scope and must be observed, not reconstructed:

| Binding | Required behavior |
|---|---|
| executable | Exact `environment.interpreter_path` |
| argv | Exact ordered array; no string command line |
| dispatch | `-c`, `--command`, `-m`, and shell executables forbidden |
| runner | Repository-relative `.py` path present in `code_hashes` |
| config | Repository-relative path present in `config_hashes` |
| cwd | Exact `repository_working_directory` |
| timeout | Positive phase timeout no greater than the frozen compute budget |
| environment | A future real backend must construct and verify the exact frozen environment, scrub unapproved inherited variables, and prevent import/path injection |
| process creation | `shell=False`, no stdin, bounded stdout/stderr, and a sandbox-bound child/process tree |

The synthetic supervisor binds the executable to its own exact `sys.executable`, checks
an exact sorted environment, and rejects `PYTHONPATH`, `PYTHONHOME`, NUL values, and
malformed environment keys. That test behavior is not evidence of real environment or OS isolation.
Likewise, `process.kill()` in the synthetic timeout path is not proof of process-tree
containment or crash-safe real sealing.

## Broker access boundary

Real read and write calls use typed requests and reconcile all caller claims with the
canonical v3 inventory:

- phase ID;
- repository-relative path;
- access class;
- capability ID and capability profile;
- expected input hash; and
- exact read or write allowlist entry.

Row and sealed-input reads increment `real_data_rows_opened` only after v3 authorization
and a successful delivery. Synthetic reads increment `synthetic_blobs_opened`, never the
real counter. Metadata reads do not increment the real row counter. Row-as-metadata and
metadata-as-row relabelling are rejected.

A future real child must not receive a mount containing row files or a writable output
root. It receives only a narrowly typed IPC endpoint. The host-side broker owns file
handles, no-follow/reparse protection, create-exclusive writes, hash verification, live
per-access revalidation, and counters.

## Receipt and durable-authority constraints

The new facade must call the existing v3 `issue_execution_receipt`; it must not define a
replacement receipt or change its fields. The canonical
`ordinary_real_data_execution_receipt_v1` remains an exact-field object containing:

- scope and lifecycle bindings: experiment, RUNNING event, run-scope digest, execution
  kind, and capability profile;
- commit and live-main ancestry bindings;
- Registry and Prepare/Run approval evidence;
- metadata-preflight, capability, input, environment, exact-command, and read/write
  allowlist digests;
- fresh output-root reservation evidence and issuance time; and
- the fixed safety fields below.

The durable field values must remain:

```text
execution_kind=real_data
output_root_was_fresh=true
real_data_execution_allowed=false
execution_authorized=false
formal_buy=false
send_order=false
stake=0
```

The human-merged branch-local RUNNING event also keeps
`automatic_execution_allowed=false`. Effective permission is only the broker's ephemeral
decision for one exact access after all v3 and OS conditions have just been revalidated.
It is not written as `true` into the receipt or lifecycle event.

Runtime `reason_code` values in these v1 modules are diagnostics. They are not new v3
receipt fields and must not be injected into, or used to overload, the frozen v3 failure
manifest schema. Any future need for a durable new field requires a separate governance
decision and schema version, not an ad-hoc extension here.

## Synthetic is not real

Synthetic execution is intentionally a separate type and authority domain:

- construction requires `SYNTHETIC_TEST_ONLY` and `execution_kind=synthetic`;
- fixture files are constrained to a dedicated root and verified by declared hash,
  access class, and capability;
- the synthetic supervisor may launch a fixture child with `shell=False`, an exact
  argv/cwd/environment, a wall timeout, bounded logs, and a declared-output check;
- cooperative-fixture timeout, crash, and unexpected-output outcomes are sealed in
  `synthetic.success.manifest.json` or `synthetic.failure.manifest.json` with exclusive
  creation; and
- every synthetic seal records `consumer_eligible=false`,
  `real_data_rows_opened=0`, `formal_buy=false`, `send_order=false`, and `stake=0`.

The synthetic seal has `schema_version=synthetic_supervisor_seal_v1` and
`assurance=SYNTHETIC_TEST_ONLY`. It is not a v3 execution receipt, result seal, failure
seal, output attestation, live authority proof, or evidence that a platform is supported.
Synthetic success must never switch a real enforcement status to `ENFORCED`.

The synthetic child is not OS-confined. If a deliberately hostile fixture pre-creates
or replaces the reserved `fixture-seal/` namespace, the supervisor raises
`SYNTHETIC_SEAL_UNAVAILABLE` and claims no valid seal; the forged bytes are never parsed
or accepted as a manifest. This is fail-closed detection, not crash-safe sealing. Only a
future real broker-only filesystem backend can make the seal namespace non-bypassable.

## Platform backend requirements

These are target requirements only. They do not change the current all-unsupported
status.

### Windows target

A Windows backend would need independently testable evidence for at least:

- an AppContainer or equivalently restricted token with no network capability, explicit
  inherited-handle allowlisting, and no ambient credential access;
- a broker-only filesystem design enforced by ACLs and handle inheritance, with reparse
  point and path-escape resistance and no direct real-row/output-root access;
- a Job Object containing the complete process tree and enforcing termination, CPU,
  memory, and process-count bounds;
- a trusted wall-time watchdog; and
- host-broker enforcement of byte, row, output-disk, model-fit, OOS, and inference
  counters.

Until these properties and their negative tests exist, Windows remains three-way
`UNSUPPORTED_FAIL_CLOSED`.

### Linux target

A Linux backend would need independently testable evidence for at least:

- user, mount, network, and PID namespaces with `no_new_privs` and an allowlisted seccomp
  profile;
- no network interface or socket escape and no credential-bearing inherited descriptor;
- an exact read-only runtime/code view with real row and writable output paths absent;
- broker-only pipes or equivalent typed IPC, plus host-side `openat2`/no-follow path
  controls;
- cgroup v2 process-tree CPU/memory/process limits, pidfd-based supervision, affinity
  enforcement where frozen, and a wall-time watchdog; and
- host-broker byte, row, disk, model-fit, OOS, and inference counters.

Until these properties and their negative tests exist, Linux also remains three-way
`UNSUPPORTED_FAIL_CLOSED`. An unknown platform has no permissive default.

## Hard blockers before any real mode

### 1. Direct v3 API and Python-level bypass

The existing v3 module deliberately exposes receipt/read/write/seal functions as public
Python APIs. Adding a facade does not make those imports inaccessible, and code-owned
factory tokens do not prevent a process with ordinary filesystem or socket access from
bypassing the facade. Real mode therefore remains blocked until the runner has no direct
row/output access, the OS sandbox exposes only broker IPC, and tests prove alternate
import, file, socket, symlink/reparse, inherited-handle, and child-process routes fail.

### 2. EXP-033/EXP-034 runner adaptation

Current EXP-033/EXP-034 runners are not adapted to the exact `-I -B ... --phase ...
--config ...` broker-only interface. A later execution commit must supply and hash-bind
the phase runner and config, remove direct data/output access, use only typed broker IPC,
and freeze the exact argv/cwd/environment before approval. This v1 documentation does
not perform or authorize that adaptation.

### 3. Real OS backend

No Windows, Linux, or other real isolation backend is implemented. The supervisor has no
path that launches a real child. All three enforcement planes must be independently
implemented and proved; a timeout-only wrapper is insufficient.

### 4. Crash and failure sealing

There is no real process-tree watchdog or automatic crash-time seal path.
`OrdinaryRealDataExecutionBrokerV1.seal_failure()` only delegates to the exact v3 sealer
after the caller already supplies the required v3 failure manifest bytes, artifact bytes,
path, receipt, and authority context. It does not observe a crash, kill descendants,
inventory partial output, or construct the failure manifest.

A future trusted supervisor must own the child and output-root handles, terminate the
entire process tree on timeout/crash/resource breach, close broker capabilities, enumerate
and hash partial output, and attempt exactly one create-exclusive v3 failure seal. If the
failure seal itself cannot be completed, the root must remain non-consumer-eligible and
the condition must be reported outside the frozen manifest without fabricating success.
The synthetic failure seal is not transferable to this real path.

## Machine-readable diagnostics

The broker exposes fail-closed diagnostics for scope/version/kind errors, enforcement
unavailability, request relabelling, allowlist/hash/capability/phase mismatches, v3
delegation failures, freshness/seal mismatches, and attempted synthetic promotion. The
supervisor adds exact process, environment, timeout, path, backend, and synthetic seal
errors. Important operational codes include:

| Boundary | Reason codes |
|---|---|
| Host readiness | `NETWORK_ISOLATION_UNAVAILABLE`, `BROKER_ONLY_FILESYSTEM_ISOLATION_UNAVAILABLE`, `RESOURCE_ENFORCEMENT_UNAVAILABLE` |
| Scope and authority | `RUN_SCOPE_NOT_EXACT_V3`, `RUN_SCOPE_VERSION_UNKNOWN`, `LEGACY_V2_REAL_DATA_FORBIDDEN`, `EXECUTION_KIND_MISMATCH`, `V3_LIVE_AUTHORITY_REVALIDATION_FAILED`, `V3_RECEIPT_ISSUANCE_REJECTED` |
| Broker access | `PHASE_MISMATCH`, `PATH_NOT_ALLOWLISTED`, `ACCESS_CLASS_MISMATCH`, `CAPABILITY_MISMATCH`, `CAPABILITY_PROFILE_MISMATCH`, `INPUT_HASH_BINDING_MISMATCH`, `INPUT_HASH_MISMATCH`, `ROW_AS_METADATA_RELABEL_FORBIDDEN`, `METADATA_AS_ROW_RELABEL_FORBIDDEN` |
| Process binding | `EXACT_ARGV_MISMATCH`, `SHELL_OR_FREE_FORM_COMMAND_FORBIDDEN`, `EXACT_PROCESS_BINDING_MISMATCH`, `INTERPRETER_MISMATCH`, `CWD_MISMATCH`, `EXACT_ENVIRONMENT_MISMATCH`, `HASH_BOUND_EXECUTABLE_MISMATCH`, `RESOURCE_BUDGET_MISMATCH` |
| Real implementation blockers | `REAL_BROKER_FACTORY_REQUIRED`, `REAL_SUPERVISOR_FACTORY_REQUIRED`, `REAL_EXECUTION_BACKEND_NOT_IMPLEMENTED`, `SYNTHETIC_BACKEND_NOT_REAL_AUTHORITY` |
| Output and sealing | `OUTPUT_ROOT_NOT_FRESH`, `OUTPUT_PATH_NOT_MANIFESTED`, `OUTPUT_SEAL_STATUS_MISMATCH`, `V3_OUTPUT_ACCESS_REJECTED`, `V3_OUTPUT_SEAL_REJECTED`, `SYMLINK_OR_REPARSE_ESCAPE`, `SYNTHETIC_SEAL_UNAVAILABLE` |
| Synthetic-only outcome | `CHILD_TIMEOUT`, `CHILD_CRASH`, `UNMANIFESTED_OUTPUT`, `SYNTHETIC_SUPERVISED_RUN_OK` |

These values support deterministic tests and operator diagnosis; they do not relax any
v3 condition and do not become lifecycle decisions.

## Required test evidence for a future backend

Before changing one platform plane from `UNSUPPORTED_FAIL_CLOSED` to `ENFORCED`, tests
must demonstrate both the positive path and escape attempts against the actual OS
backend. At minimum:

- no real child, receipt, row, output, or seal when any one enforcement plane is absent;
- exact v3 scope and exact phase/argv/cwd/environment/hash binding;
- no shell, free-form command, environment/import injection, alternate v3 API bypass,
  network connection, direct file open, path traversal, symlink/reparse escape, inherited
  handle, undeclared output, or uncontained descendant;
- per-access live authority revalidation and typed broker reconciliation;
- authoritative row/byte/output/resource counters and budget-stop behavior;
- timeout, crash, resource breach, broker loss, and supervisor loss behavior over the
  whole process tree;
- immutable success/failure sealing and zero consumer eligibility for partial output;
- synthetic types and seals rejected wherever real authority is required; and
- repeated identical inputs produce the same validation and reason-code result.

Passing unit tests for the synthetic fake backend is useful for interface behavior only.
It cannot satisfy a real platform acceptance item.

## Operational conclusion

The two v1 modules are safe scaffolding only: exact v3 normalization and typed broker
claims are present, while all real execution paths fail closed before receipt issuance or
row access. The current Windows, Linux, and unsupported-platform status remains three-way
`UNSUPPORTED_FAIL_CLOSED`. Real execution cannot proceed until the direct-v3-API,
broker-only runner, OS isolation/resource, and crash-time v3 failure-seal blockers are
resolved in a later reviewed execution change.
