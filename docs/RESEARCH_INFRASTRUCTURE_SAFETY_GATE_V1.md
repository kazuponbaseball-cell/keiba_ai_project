# Research OS Infrastructure Safety Gate v1

## Outcome

`infrastructure_safety_v1` separates non-racing Research OS infrastructure from
ROI hypothesis research. It does not award substitute racing points. A proposal
passes only when every machine-readable safety check passes; one failure rejects
creation before a canonical scope, queue record, or registry event is written.

This gate cannot bootstrap itself. The first policy and every later root-of-trust
change require a normal Draft governance PR, human review, and human merge. Only
the policy present on the proposal base commit on GitHub `main` may become approval
evidence.

## Classification boundary

Use the ROI hypothesis gate when work touches racing data, a model or feature,
candidate selection, value logic, odds, results, payouts, ROI, shadow behavior, or
production behavior. Use the infrastructure gate only for bounded Research OS
contracts, schemas, deterministic compilers, synthetic adapters, and fixture tests.

The gate profile is stored in canonical JSON. A CLI option, GPT response, branch
copy, environment variable, or caller assertion cannot change it after creation.
Legacy ROI proposal and run JSON are not reserialized and their existing digests
remain unchanged.

## Lifecycle and authority

```text
PROPOSED
  -> APPROVED_TO_PREPARE
  -> PREPARING
  -> RUN_APPROVAL_REQUIRED
  -> APPROVED_TO_RUN
  -> RUNNING (synthetic only)
  -> REVIEW_REQUIRED
       -> REJECTED
       -> INVALID
```

The existing GitHub trust boundary is reused:

- `APPROVED_TO_PREPARE <proposal_scope_digest>` and
  `APPROVED_TO_RUN <run_scope_digest>` require different registry-wide unused
  comment IDs.
- Current `main`, base ancestry, base-commit `research/APPROVERS.json`, author type,
  immutable comment fields, and prior grants are fetched and revalidated through
  read-only GitHub evidence.
- `APPROVED_FOR_SHADOW` is forbidden for this profile.
- Production, merge, formal BUY, order, and notification approval do not exist.

## Hard locks

The policy fixes all of the following:

- execution kind `synthetic`;
- zero real-data rows, training, backtest, outer OOS, and ROI calculations;
- zero external network/API calls, credential access, and actual Codex dispatches;
- no model, feature, candidate, value, production, BUY, order, or notification path;
- `formal_buy=false`, `send_order=false`, `stake=0`;
- one variant and zero threshold searches;
- code-owned structured command templates, never a free-form shell string;
- an exact base-to-execution commit diff with path and blob hashes, where the
  execution commit is a descendant of the proposal base;
- fixture, config, dependency/environment, command, and policy material bound to
  exact execution-commit blobs, never dirty or untracked copies;
- a clean execution worktree outside explicitly hash-bound lifecycle artifacts;
- one non-linked code-owned registry path, exact current-main serialization,
  append-only commit/worktree prefixes, and a locked compare-and-swap append;
- bounded synthetic provenance envelopes under `research/synthetic/`, with
  symlink/junction, populated secret field, and row-level real-data sentinels;
- a pure exact-module import allowlist plus an AST import/call/symbol firewall over
  every changed Python Git blob. Non-static call targets, dangerous dunder
  symbols/strings, wildcard imports, skip/load-test hooks, and callable lookup
  helpers fail closed; the conventional `__name__ == "__main__"` guard remains
  permitted.

Constitutional and approval root-of-trust paths are never eligible for an
infrastructure experiment. This includes the operating contract, charter, both
gate policies, approver allowlist, approval verifier, registry classifier, and
their schemas. Those paths require a governance-core PR.

The committed envelope, size limit, content sentinels, and AST checks make
materials reviewable and reject known unsafe forms; they do not mathematically
prove that neutral-looking bytes were never copied from real data, nor do they
form an operating-system sandbox. That residual provenance/semantic question is
why v1 emits no preparation or execution authority. A future executor must add
an independently reviewed sandbox and provenance mechanism rather than treating
these static checks as runtime isolation.

## Command boundary

Run scope commands are structured objects selected from code-owned templates.
Shell strings, `python -c`, PowerShell, cmd, bash, package installation, URLs,
pipes, redirects, response files, globbing, and caller-selected executables are
rejected. The exact current Python executable and its hash/version are frozen in
the environment evidence, and argv uses isolated `-B -I -S` mode. The run scope
also fixes repository-root cwd, no inherited environment, an empty explicit
environment, zero writable paths, and a timeout derived from the proposal budget. The run-scope
compiler records a deterministic plan; it does not execute the command, start a
subprocess, contact GitHub, or call an external provider. Accordingly,
`automatic_execution_allowed`, `preparation_authorized`, and
`execution_authorized` remain false in every infrastructure event.
Future gate-scoped tests live only under `research/infra_tests/`, outside the
ordinary pull-request discovery path `tests/research/test_*.py`. The latter and
the workflow are roots of trust. A gate-scoped test is therefore not executed
merely because a PR is opened. Each pre-registered `test_id` maps by lowercase
and hyphen/underscore normalization to exactly one
`test_<normalized_id>` method on a committed changed `unittest.TestCase` class.
The structured command must select that changed module, or the fixed
infrastructure discovery template; missing modules, ambiguous mappings, and
zero discoverable test methods fail closed. Schema v3 is an evidence compiler, not an
executor or an executable authorization token. A separately reviewed future
executor/authority verifier is required before any structured command can run.

## Registry authority

Only the non-symlink/non-junction `research/REGISTRY.jsonl` is accepted. Before run-scope transitions, the
base registry must be an exact prefix of the execution-commit registry, and that
blob must be a prefix of the current worktree ledger after newline normalization.
Appending an event takes a process lock and compares the current bytes with the
pre-verification snapshot; a concurrent write fails closed and must be retried.
Before every transition, including `PROPOSED` and `INVALID`, the verifier fetches
the registry blob at verified current `main`; after newline normalization the
local snapshot must be byte-for-byte equal, not merely a prefix. The main ref is
re-read immediately before append and must be unchanged. Consequently, one local
append produces only a pending, non-authoritative candidate. A human must merge
that exact event and the branch must refresh to the new current-main ledger before
another transition can be created. Concurrent candidates from the same main do
not grant authority; after one is merged, stale candidates fail exact-ledger or
already-consumed-comment validation. This is the durable serialization point for
terminal history and the registry-wide comment-ID namespace.

## Policy versioning

The v1 policy file is immutable after this bootstrap reaches `main`. A future
policy change must add a new gate kind/version and a different policy path rather
than rewriting v1. This preserves the ability to read and invalidate old schema-v3
experiments against their original policy hash.

## Compatibility

- Existing ROI proposal/run objects, queue schema v2, registry event schema v2,
  score threshold 75, and GPT strategy schema v1 remain ROI-only and unchanged.
- Current-main ledger verification for legacy transitions is transient; no new
  field is serialized into schema-v2 events.
- Infrastructure queue/event schema v3 is explicit and does not add defaults to
  legacy canonical objects.
- The committed-ledger validator dispatches explicitly between event schema v2
  and v3; neither profile is interpreted as the other.
- Approval comment IDs share one registry-wide namespace across schema versions.
- Existing committed proposal and run digests are golden compatibility evidence.
- Legacy unversioned ROI run scopes do not bind `execution_kind` in their
  canonical digest. They remain readable and byte-for-byte unchanged, but
  real-data `RUNNING` now fails closed. Re-enabling it requires a separate
  versioned ROI contract that hash-binds execution kind and capabilities.

## Bootstrap and follow-up

This governance change may be reviewed and merged by a human, but it may not use
the new gate to approve itself. After merge, a new infrastructure proposal may be
created from that exact `main` commit for the synthetic GPT-provider/dry-run Codex
dispatcher successor. It may compile proposal/run evidence but may not execute a
structured test until a separate executor/authority-verifier governance change
is reviewed and merged. `EXP-20260809-032` remains `INVALID` and must not be reused.

The gate does not authorize an actual external GPT API or actual Codex CLI call.
Those require a separate future governance decision and cannot be inferred from a
synthetic success.
