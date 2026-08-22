# Decision Summary Layer V0

## Status and purpose

Decision Summary Layer V0 is a read-only consumer View Model for the Horse Intelligence / WIN5 observation UI. It changes the reading order without replacing the existing Horsecard, Queue, or Route evidence generators:

1. race-level Decision Summary,
2. compact summaries for every runner,
3. expandable Horsecard / Queue / Route evidence.

`PRIMARY_REVIEW`, `CONDITIONAL_REVIEW`, `FRAGILE`, and `INSUFFICIENT` are **review priorities, not an ability ordering**. The layer does not produce a result probability, place probability, weighted total, price/value judgment, or purchase instruction.

Weekend V0 is intentionally synthetic-only. The committed sample contains fictional races, horses, riders, trainers, and evidence. It neither reads nor materializes the 2026-08-23 real observation snapshot.

## Architecture

```text
strict synthetic Horsecard projection
  -> canonical safe projection
  -> condition / Queue / role derivation
  -> fixed boolean classification precedence
  -> race-level aggregation
  -> cross-object validation
  -> canonical JSON + static no-JavaScript HTML
```

The implementation is the standalone module `scripts/research/build_decision_summary_layer_v0.py`. It does not import or change `build_race_intelligence_lite_plus.py`; it has no real-data, training, OOS, prediction, notification, or transaction command.

The source boundary accepts only these structured blocks:

- runner identity and basic Horsecard fields;
- target-context role, role range, expected position, flexibility, forward propensity, lead dependency, first-turn cost, rating evidence, and pressure-rival conflicts;
- directly observed current-condition fit;
- evidence IDs and the original synthetic evidence rows retained for display.

Role and Queue use separate typed evidence-ID lists. Role IDs must resolve to `ROLE` or `ROUTE` details, Queue IDs to `QUEUE` details, and current-condition IDs to `CONDITION` details. The output carries these IDs by dimension and on every trigger; runtime validation reconciles each ID with the retained evidence row and its required dimension. A single evidence item cannot be counted as both the role and Queue material needed by `PRIMARY_REVIEW`. Any structured reference to a missing, mistyped, or `UNOBSERVED` detail fails closed.

Unknown fields fail closed at every nesting level. Forbidden legacy/scenario/market vocabulary is also rejected from the free-text evidence rows. Therefore legacy ability, `ai_score`, model rank/gap/confidence, scenario-sensitivity branches, market fields, and purchase fields cannot enter the builder. The only purchase-related names in an artifact are false/zero safety attestations under the top-level `safety` object.

## Clean ability boundary

The existing Lite+ `ability` block is a past-finish-position band, not a versioned clean-ability contract. Weekend V0 never projects it. Every horse is therefore emitted as:

```json
{
  "ability_status": {
    "value": "UNKNOWN",
    "availability": "NOT_AVAILABLE",
    "evidence_count": 0,
    "reason_codes": ["ABILITY_CLEAN_NOT_AVAILABLE"]
  }
}
```

An available clean-ability value cannot be added to this V0 input. A future clean source requires a new schema version and explicit lineage; it must not silently fill this field.

## Classification V0

Classification uses ordered boolean gates, not a numeric total:

1. `FRAGILE` when at least one explicit failure trigger is derived.
2. `PRIMARY_REVIEW` when role core is available, Queue fit is `SUPPORTIVE`, and there is no fragility trigger.
3. `CONDITIONAL_REVIEW` when role core is available, Queue fit is `CONDITIONAL`, at least one explicit positive world-state trigger exists, and there is no fragility trigger.
4. `INSUFFICIENT` otherwise.

Role core is available only when target role, role range, expected position, and their evidence IDs are present.

Queue fit is derived by a fixed predicate:

- `ADVERSE`: high pressure rival, high first-turn cost, or possible lead dependency;
- `CONDITIONAL`: role core and required Queue context are known, with a medium pressure rival or medium first-turn cost;
- `SUPPORTIVE`: role core, forward propensity, first-turn cost, and dependency status are known; cost is low; no medium/high rival exists; dependency is absent;
- `UNKNOWN`: required structured context is missing.

Fragility is derived from one or more of:

- narrow role range together with limited flexibility;
- possible lead dependency;
- high pair conflict;
- high first-turn cost;
- directly observed adverse current condition.

`FRAGILE` has precedence even when positive role evidence also exists. Positive evidence remains visible; it is not erased by the classification.

## Current condition and confidence

Current-condition fit is available only when its source status is `OBSERVED`. `DERIVED`, `PROXY`, and `UNOBSERVED` inputs remain `UNKNOWN / NOT_AVAILABLE`; rest patterns or other proxies do not fill the value.

Summary `confidence` is newly derived from the availability set `{ROLE, QUEUE, CONDITION}`. It does not copy any upstream confidence:

- `MEDIUM`: non-fragile classified horse with ROLE, QUEUE, and directly observed CONDITION;
- `LOW`: a non-insufficient horse below that availability set, or any `FRAGILE` horse;
- `NOT_AVAILABLE`: `INSUFFICIENT`.

Weekend V0 is capped at `MEDIUM`. This is evidence-coverage confidence, not a calibrated probability.

## Output schema

The output contract is [decision_summary_layer_v0.schema.json](schemas/decision_summary_layer_v0.schema.json). `additionalProperties` is false throughout the contract. Runtime validation additionally checks runner-universe identity, official horse-number order, classification partition, bucket reconciliation, reason-code membership, trigger requirements, and evidence-detail identity.

Top-level semantics are fixed:

```json
{
  "layer_type": "READ_ONLY_CONSUMER_VIEW_MODEL",
  "review_priority_semantics": "REVIEW_PRIORITY_NOT_ABILITY_ORDERING",
  "aggregation_rule": "ORDERED_BOOLEAN_PRECEDENCE_V0",
  "source_mode": "SYNTHETIC_FIXTURE_ONLY",
  "synthetic_only": true,
  "weighted_total_generated": false
}
```

Each race contains:

- `decision_summary.primary_review`
- `decision_summary.conditional_review`
- `decision_summary.fragile_or_downgrade`
- `decision_summary.insufficient_review`
- `decision_summary.key_world_states`
- `horse_summaries` for every runner
- `evidence_details` for every runner

Each horse summary contains:

- classification and explicit review-priority semantics;
- clean-ability status;
- current-condition fit;
- Queue fit;
- role and expected position;
- upside and fragility triggers;
- winning/in-the-money and failure world-state descriptions;
- Decision Summary confidence;
- distinct evidence counts and their typed evidence IDs;
- allowlisted machine-readable reason codes;
- an anchor to the expandable evidence detail.

All race buckets and horse rows remain in official horse-number order. No list order represents an ability or preference rank.

## Static UI structure

For every race the DOM order is fixed:

```text
Race header
Decision Summary
  - まず見る馬
  - 展開次第で見る馬
  - Fragile / 割引・注意
  - Key world states
All-runner compact summary table
Expandable Horsecard / Queue / Route evidence
```

All runner summaries are visible outside closed `<details>` elements. Every runner has exactly one matching expandable evidence section. The page has inline CSS and no JavaScript, form, external asset, storage, network, notification, or transaction hook.

## Synthetic sample

The fixture is `tests/research/fixtures/decision_summary_layer_v0.synthetic.json`. The committed deterministic artifacts are:

- `docs/observations/decision_summary_layer_v0/synthetic_sample/decision_summary_v0.json`
- `docs/observations/decision_summary_layer_v0/synthetic_sample/decision_summary_v0.html`

The four fictional runners exercise every classification:

| Horse no. | Horse | Review priority | Structural reason |
|---:|---|---|---|
| 1 | Synthetic Alpha | `PRIMARY_REVIEW` | role core + low Queue burden; no explicit fragility |
| 2 | Synthetic Beta | `CONDITIONAL_REVIEW` | medium Queue/corner cost; early-pressure relief is explicit |
| 3 | Synthetic Gamma | `FRAGILE` | narrow role, dependency, high conflict/cost, adverse observed condition |
| 4 | Synthetic Delta | `INSUFFICIENT` | role and Queue evidence unavailable |

All four have clean ability `UNKNOWN / NOT_AVAILABLE`.

## Build and verification

```powershell
python scripts/research/build_decision_summary_layer_v0.py build `
  --input tests/research/fixtures/decision_summary_layer_v0.synthetic.json `
  --output-dir docs/observations/decision_summary_layer_v0/synthetic_sample

python scripts/research/build_decision_summary_layer_v0.py verify `
  --input tests/research/fixtures/decision_summary_layer_v0.synthetic.json `
  --output-dir docs/observations/decision_summary_layer_v0/synthetic_sample

python -m unittest tests.research.test_decision_summary_layer_v0 -v
```

Canonical JSON uses UTF-8, sorted keys, compact separators, Unicode preservation, finite values only, and a trailing newline. The projection digest is based only on the validated canonical projection. No clock, UUID, random value, or input runner ordering affects output bytes.

## Not implemented in V0

- no real Lite+ snapshot adapter or real-data materialization;
- no live integration into the current Sunday page;
- no clean-ability source or inferred replacement;
- no day bias, wind, result review, training, OOS evaluation, or target prediction;
- no probability, market, price/value, purchase, stake, notification, or order path;
- no EXP-033 / EXP-034 lifecycle, scope, or governance changes;
- no deployment or merge automation.

Connecting a real Horsecard projection is a separate governed follow-up. This PR proves the consumer contract, classification semantics, deterministic output, and UI reading order with synthetic evidence only.
