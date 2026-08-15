# Registered Nonpromotion Diagnostic v1 — implementation boundary

Status: **IMPLEMENTED CLIENT/CONTRACT, NOT ACTIVATED, EXECUTION FORBIDDEN**

This package implements the bounded, reusable diagnostic route declared by the
root Research OS contract.  It does not activate the route and it does not give
the repository, a branch, a GitHub comment, or a local process execution
authority.

## Why this exists

Repeated structural checks such as “does a deterministic `ai_score` descendant
gate change historical unit-return?” should not require a new runner or a new
multi-PR lifecycle every time.  Once this lane is human-merged and separately
cut over to the shared external G2 authority, an already-registered recipe will
use one frozen run scope and one exact human comment:

```text
APPROVED_NONPROMOTION_DIAGNOSTIC_RUN <run_scope_digest>
```

There is no per-run code change, `PREPARING` phase, pull request, merge, result
acknowledgement, shadow transition, or production transition.

## Implemented here

- an append-only, finite recipe registry;
- the initial two-arm historical `ai_score` descendant-gate recipe sourced from
  PR #40, with its original formula fingerprint preserved;
- strict recipe, run-scope, catalog, authority, lease, result, and cutover
  schemas;
- a declarative typed-AST evaluator (`and`, `boolean_eq`, `ge`, `lt` only);
- a run compiler that resolves metrics, cohort, sensitivity, bootstrap,
  environment, phase, replica, retry, and output contracts from the registered
  recipe rather than caller-supplied formulas;
- a lane-specific GitHub verifier.  It does not expand the existing ordinary
  prepare/run/shadow keywords;
- role-separated candidate and settlement catalog validation.  Pregrant
  validation exposes metadata only, never rows or blob readers;
- a bounded decision/settlement evaluator whose public/canonical execution path is the
  `RUN_SCOPE_BOUND_PHASE_PLAN` callable with no subprocess, shell, arbitrary
  argv, network, credential, production, BUY, order, or notification path;
- two-replica semantic comparison and a receipt-only canonical result sealer;
- a concrete typed shared-G2 client boundary, strict authenticated
  receipt/head/witness validation, and rejection of local or duck-typed
  authority before any protected content read;
- synthetic positive and negative tests.

## Deliberately not implemented as local authority

The following cannot be substituted with a local file, `REGISTRY.jsonl`,
SQLite, a worktree, a Git branch, or process memory:

- the remote transactional shared-G2 backend;
- the independent monotonic checkpoint/witness;
- legacy grant and subject-head migration;
- the old-writer fence and second remote comparison;
- authenticated cutover and activation receipts;
- live content-addressed candidate/settlement publication;
- a runtime phase-lease issuer.
- the external approval-reservation service that consumes the run-bound lane
  activation receipt, pregrant global/subject heads, and latest witness
  checkpoint;
- the process-isolated `clean_a` / `clean_b` dispatcher and its authenticated
  failure-to-`INVALID` transition;
- server-side operation-grammar and state-digest recomputation.

Until those external components exist and the human-owned cutover is complete,
the policy remains `authority=false` and `EXECUTION_FORBIDDEN`.  CI success,
branch push, PR readiness, review, merge, or chat approval cannot change that.
The deterministic calculation helpers are private implementation details, are
not exported by the module, and are not an execution or authority API.
The client-side evaluator can label replica work, but this repository package
alone does not prove process or checkout isolation.  That proof belongs to the
external supervisor and is a hard activation prerequisite.

## Initial recipe boundary

The registered comparison is not algebraic parity.  It freezes the candidate
pair, rank, score bytes, calibrator, cohort, metric, and all non-AI clauses, then
removes one named raw-probability gate family:

```text
D0 = decision_base AND 0.225 <= p < 0.325
     AND 0.25 <= p_action < 0.4
     AND p >= 0.21275851149504352

D1 = decision_base AND 0.25 <= p_action < 0.4
```

Consequently, D1 may add the registered region
`0.325 <= p < 0.36909082652451414`.  This is a historical counterfactual
eligibility change, not a live policy mutation.  It is exactly why settlement
and unit-return access require Tier B and the irreversible shared-G2 receipt.

The source evidence remains:

- ordinary strategy score: `46 / BLOCKED_SCORE`;
- score credit: `0`;
- source authority class: `B_LOCAL_HASHED`;
- reused development OOS: `true`;
- strict T-3 rows: `0`;
- confirmatory/promotion/shadow/production eligibility: `false`.

## Expected post-cutover run flow

1. Resolve the immutable registered recipe and authenticated catalog metadata.
2. Seal a complete run scope; read no candidate or settlement rows.
3. Verify one exact GitHub approval comment and atomically reserve the global
   comment ID, semantic subject, exact subject, and run in shared G2.
4. Issue the exact `clean_a` + `clean_b` decision-lease batch.  In one atomic
   transaction, consume both leases, transition `RND_LEASED -> RND_RUNNING`,
   permanently consume the semantic and exact subjects, increment the
   question-family count, and emit one irreversible batch receipt.
5. Only after receipt revalidation, mount the candidate-only object and freeze
   D0/D1 decisions independently in `clean_a` and `clean_b`.
6. Issue separate settlement leases; mount only the five settlement columns
   after each decision receipt.
7. Require both replica projections to be valid and bitwise-semantically equal.
8. Seal the result from the authenticated comparison receipt.  The canonical
   sealer cannot read replica roots directly.

Any odds, price, popularity, market, free-form threshold, third arm, refit,
recalibration, retry, row drop, candidate change, or subject replay fails closed.

## Verification commands for this implementation PR

These commands use only repository code and synthetic fixtures.  They do not
read model, race, result, payoff, or ROI data.

```text
python -m unittest discover -s tests/research -p "test_registered_nonpromotion_*.py" -v
python -m py_compile scripts/research/registered_nonpromotion_*.py scripts/research/shared_g2_*.py
git diff --check
```

The first real-data run remains a separate, authenticated post-cutover action.
It will report only the 3,746-race final-official-payoff diagnostic and cannot
establish executable T-3 ROI or future profitability.
