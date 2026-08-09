# {{EXPERIMENT_ID}} — {{TITLE}}

> This Markdown is review-only. The canonical approval authority is
> `{{PROPOSAL_SCOPE_PATH}}` with SHA-256 digest `{{PROPOSAL_SCOPE_DIGEST}}`.

## Canonical authority

| Field | Value |
|---|---|
| Status | `{{STATUS}}` |
| Gate kind | `infrastructure_safety_v1` |
| Gate contract | `1` |
| Queue / event schema | `3` / `3` |
| Owner | {{OWNER}} |
| Created at | `{{CREATED_AT}}` |
| Base commit | `{{BASE_COMMIT}}` |
| Source as-of | `{{SOURCE_AS_OF}}` |
| Gate policy | `research/INFRASTRUCTURE_GATE.json` |
| Gate policy SHA-256 | `{{GATE_POLICY_SHA256}}` |
| Gate result | `PASS_ALL_HARD_CHECKS` |

No numeric ROI hypothesis score is assigned. This gate cannot lower, replace,
or reinterpret `research/HYPOTHESIS_SCORECARD.yaml`.

## Change hypothesis

{{CHANGE_HYPOTHESIS}}

## Null hypothesis

{{NULL_HYPOTHESIS}}

## Safety objective

{{SAFETY_OBJECTIVE}}

## Expected changed paths

{{EXPECTED_CHANGED_PATHS}}

## Pre-registered synthetic test matrix

Gate-scoped test source must be under `research/infra_tests/`; do not place it
under the ordinary PR CI discovery path `tests/research/`.
Each `test_id` must have exactly one committed changed `unittest.TestCase`
method named `test_<normalized_id>`, where normalization lowercases the ID and
collapses hyphens/underscores to one underscore. The command plan must select
that changed test module or the fixed infrastructure discovery template.

{{TEST_MATRIX}}

## Approval boundary

- Preparation requires `APPROVED_TO_PREPARE {{PROPOSAL_SCOPE_DIGEST}}` from a
  verified human approver.
- A frozen synthetic run requires a distinct unused
  `APPROVED_TO_RUN <run_scope_digest>` comment.
- `RUNNING` is valid only with `execution_kind=synthetic`.
- Every locally appended event is a pending, non-authoritative candidate until
  its exact bytes are human-merged to current main; the next transition requires
  a refreshed branch whose ledger exactly equals that main blob.
- Schema v3 does not itself authorize preparation or execution. A separate,
  human-reviewed executor/authority verifier is required before commands run.
- Real data, ROI, production, BUY, order, notification, credential access,
  external API calls, Codex dispatch, shadow approval, and merge approval are
  outside this gate.
- Every artifact keeps `formal_buy=false`, `send_order=false`, and `stake=0`.

## Immutable run scope

The run scope freezes the proposal and digest, exact execution commit, gate
policy/config/synthetic input/environment hashes, base-to-execution changed
path/hash manifest, seed, structured command-template plan, exact argv, and the
repository-root/no-inherited-environment/timeout execution context. All material
must resolve to execution-commit blobs; dirty or untracked copies are invalid.
The preparation CLI validates and writes JSON only; it does not execute the
argv or grant execution authority.

## Result — append after separately approved synthetic execution

Decision: `REVIEW_REQUIRED` / `REJECTED` / `INVALID`

Production promotion, merge, shadow use, and formal BUY are not represented by
this lifecycle.
