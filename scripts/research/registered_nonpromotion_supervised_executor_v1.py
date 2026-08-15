from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from registered_nonpromotion_contract_v1 import (
    ContractError,
    RegisteredRecipe,
    canonical_digest,
    evaluate_registered_decisions,
    verify_canonical_run_scope,
)
from shared_g2_durable_ledger_v1 import GlobalHead
from shared_g2_lease_authority_v1 import (
    ConsumedDecisionLeaseBatch,
    ConsumedPhaseLease,
    IrreversibleLifecycleView,
    IssuedDecisionLeaseBatch,
    IssuedPhaseLease,
    REPLICA_COMPARE_ACTOR,
    RevalidatedPhaseOutputSeal,
    SealedPhaseOutput,
    SharedG2LeaseAuthorityClient,
    TrustedPhaseOutputAttestation,
    canonical_predecessor_output_digest,
)


FORBIDDEN_ANY_PHASE_TOKENS = (
    "odds",
    "price",
    "popularity",
    "market",
)


RUNNER_TEMPLATE_ID = "REGISTERED_NONPROMOTION_BOUNDED_EXECUTOR_V1"
STRUCTURED_ARGV = (
    "registered_nonpromotion_supervised_executor_v1",
    "RUN_SCOPE_BOUND_PHASE_PLAN",
)
REPLICA_IDS = ("clean_a", "clean_b")
SETTLEMENT_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "gate_kind",
        "run_scope_digest",
        "recipe_id",
        "recipe_version",
        "recipe_digest",
        "replica_id",
        "decision_freeze_receipt_digest",
        "settlement_operation_receipt_digest",
        "scientific_projection",
        "scientific_projection_digest",
        "computed_outcome",
        "authority",
        "authenticated_phase_output_seal_required",
        "evidence_purpose_class",
        "source_authority_class",
        "confirmatory",
        "promotion_eligible",
        "score_credit",
        "strict_t3_rows",
        "formal_buy",
        "send_order",
        "stake",
        "result_digest",
    }
)
SCIENTIFIC_PROJECTION_FIELDS = frozenset(
    {
        "primary",
        "sensitivity",
        "bootstrap",
        "candidate_projection_digest",
        "decision_vector_digest",
        "settlement_projection_digest",
        "paired_rows_digest",
        "contract_status",
    }
)


class ProtectedContentTransport(Protocol):
    """External content transport; this interface is never an authority source."""

    def fetch_candidate_rows(
        self,
        *,
        catalog_release_id: str,
        candidate_entry_sha256: str,
        run_scope_digest: str,
        replica_id: str,
        irreversible_receipt_digest: str,
    ) -> Sequence[Mapping[str, Any]]:
        ...

    def fetch_settlement_rows(
        self,
        *,
        catalog_release_id: str,
        settlement_entry_sha256: str,
        run_scope_digest: str,
        replica_id: str,
        decision_output_seal_receipt_digest: str,
        settlement_consumption_receipt_digest: str,
    ) -> Sequence[Mapping[str, Any]]:
        ...


def _ensure_no_market_columns(columns: Sequence[str]) -> None:
    for column in columns:
        lowered = column.lower()
        if any(token in lowered for token in FORBIDDEN_ANY_PHASE_TOKENS):
            raise ContractError(f"forbidden market/price column mounted: {column}")


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value == "0" * 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{label} must be a non-zero lowercase full SHA-256")
    return value


@dataclass(frozen=True)
class AuthenticatedProtectedContentProvider:
    """Receipt-gated wrapper around an external, content-addressed transport."""

    transport: ProtectedContentTransport
    run_scope_digest: str
    catalog_release_id: str
    candidate_entry_sha256: str
    settlement_entry_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.run_scope_digest, "content provider run_scope_digest")
        _require_sha256(
            self.candidate_entry_sha256,
            "content provider candidate_entry_sha256",
        )
        _require_sha256(
            self.settlement_entry_sha256,
            "content provider settlement_entry_sha256",
        )
        if (
            not isinstance(self.catalog_release_id, str)
            or not self.catalog_release_id
        ):
            raise ContractError("content provider catalog_release_id is required")

    def read_candidate_rows(
        self,
        *,
        consumed_batch: ConsumedDecisionLeaseBatch,
        replica_id: str,
    ) -> Sequence[Mapping[str, Any]]:
        if not isinstance(consumed_batch, ConsumedDecisionLeaseBatch):
            raise ContractError(
                "candidate access requires a typed consumed decision lease batch"
            )
        if replica_id not in REPLICA_IDS:
            raise ContractError("candidate access replica is not registered")
        consumptions = {
            item.lease.binding.replica_id: item
            for item in consumed_batch.replica_consumptions
        }
        if tuple(sorted(consumptions)) != REPLICA_IDS:
            raise ContractError(
                "candidate access requires exact clean_a and clean_b consumptions"
            )
        consumed = consumptions[replica_id]
        binding = consumed.lease.binding
        if (
            binding.phase != "DECISION_FREEZE"
            or binding.attempt != 1
            or binding.run_scope_digest != self.run_scope_digest
            or consumed.decision_issued_batch != consumed_batch.issued
            or consumed.decision_consumption_batch != consumed_batch.receipt
            or consumed.transaction != consumed_batch.transaction
        ):
            raise ContractError(
                "candidate access decision batch chain does not match the frozen run"
            )
        return self.transport.fetch_candidate_rows(
            catalog_release_id=self.catalog_release_id,
            candidate_entry_sha256=self.candidate_entry_sha256,
            run_scope_digest=self.run_scope_digest,
            replica_id=replica_id,
            irreversible_receipt_digest=consumed_batch.receipt.payload_digest,
        )

    def read_settlement_rows(
        self,
        *,
        consumed: ConsumedPhaseLease,
        decision_output_seal: SealedPhaseOutput,
    ) -> Sequence[Mapping[str, Any]]:
        if not isinstance(consumed, ConsumedPhaseLease) or not isinstance(
            decision_output_seal, SealedPhaseOutput
        ):
            raise ContractError(
                "settlement access requires typed consumption and predecessor seal"
            )
        binding = consumed.lease.binding
        predecessor = decision_output_seal.receipt
        if (
            binding.phase != "SETTLEMENT_DIAGNOSTIC"
            or binding.attempt != 1
            or binding.run_scope_digest != self.run_scope_digest
            or predecessor.phase != "DECISION_FREEZE"
            or predecessor.replica_id != binding.replica_id
            or predecessor.run_scope_digest != self.run_scope_digest
            or canonical_predecessor_output_digest(
                successor_binding=binding,
                predecessor_output_seals=(decision_output_seal,),
            )
            != binding.predecessor_receipt_digest
        ):
            raise ContractError(
                "settlement access predecessor or lease binding is invalid"
            )
        return self.transport.fetch_settlement_rows(
            catalog_release_id=self.catalog_release_id,
            settlement_entry_sha256=self.settlement_entry_sha256,
            run_scope_digest=self.run_scope_digest,
            replica_id=binding.replica_id,
            decision_output_seal_receipt_digest=predecessor.payload_digest,
            settlement_consumption_receipt_digest=consumed.receipt.payload_digest,
        )


def validate_candidate_rows(
    recipe: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    requirements = recipe["catalog_requirements"]["candidate_role"]
    required = set(requirements["required_columns"])
    forbidden = set(requirements["forbidden_columns"])
    expected_count = recipe["cohort"]["race_count"]
    expected_folds = Counter(recipe["cohort"]["fold_counts"])
    if len(rows) != expected_count:
        raise ContractError(
            f"candidate cohort must contain exactly {expected_count} rows; got {len(rows)}"
        )
    output: list[dict[str, Any]] = []
    race_ids: set[str] = set()
    fold_counts: Counter[str] = Counter()
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ContractError(f"candidate row {index} is not an object")
        columns = set(raw)
        _ensure_no_market_columns(list(columns))
        missing = required - columns
        prohibited = forbidden & columns
        extra = columns - required
        if missing or prohibited or extra:
            raise ContractError(
                f"candidate row schema mismatch; missing={sorted(missing)}, "
                f"forbidden={sorted(prohibited)}, extra={sorted(extra)}"
            )
        row = dict(raw)
        race_id = row.get("race_id")
        if not isinstance(race_id, str) or not race_id.isdigit() or len(race_id) != 16:
            raise ContractError("race_id must be a 16-digit string")
        if race_id in race_ids:
            raise ContractError(f"duplicate candidate race_id {race_id}")
        race_ids.add(race_id)
        candidate_key = row.get("candidate_key")
        if not isinstance(candidate_key, str) or not candidate_key:
            raise ContractError("candidate_key must be a non-empty string")
        for field in ("candidate_generated", "eligible_race"):
            if type(row.get(field)) is not bool:
                raise ContractError(f"{field} must be boolean")
        for field in ("top1_wide_prob", "p_action_C0_offset"):
            try:
                value = float(row.get(field))
            except (TypeError, ValueError) as exc:
                raise ContractError(f"{field} must be numeric") from exc
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ContractError(f"{field} must be finite and in [0,1]")
        fold = row.get("fold")
        if fold not in recipe["cohort"]["fold_values"]:
            raise ContractError(f"unexpected fold {fold!r}")
        fold_counts[str(fold)] += 1
        output.append(row)
    if fold_counts != expected_folds:
        raise ContractError(
            f"fold counts differ from recipe; expected={dict(expected_folds)}, "
            f"observed={dict(fold_counts)}"
        )
    return sorted(output, key=lambda row: row["race_id"])


def _freeze_decisions_after_authenticated_mount(
    registered: RegisteredRecipe,
    rows: Sequence[Mapping[str, Any]],
    *,
    run_scope_digest: str,
    replica_id: str,
    irreversible_receipt_digest: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validated = validate_candidate_rows(registered.recipe, rows)
    masks = evaluate_registered_decisions(registered.recipe, validated)
    by_key = {(row["race_id"], row["candidate_key"]): row for row in masks}
    decision_rows: list[dict[str, Any]] = []
    for row in validated:
        mask = by_key[(row["race_id"], row["candidate_key"])]
        decision_rows.append(
            {
                "race_id": row["race_id"],
                "race_date": row["race_date"],
                "venue_code": row["venue_code"],
                "fold": row["fold"],
                "candidate_key": row["candidate_key"],
                "horse_a": row["horse_a"],
                "horse_b": row["horse_b"],
                "top1_wide_prob": float(row["top1_wide_prob"]),
                "p_action_C0_offset": float(row["p_action_C0_offset"]),
                "d0_eligible": mask["d0_eligible"],
                "d1_eligible": mask["d1_eligible"],
            }
        )
    vector_projection = [
        {
            "race_id": row["race_id"],
            "candidate_key": row["candidate_key"],
            "d0_eligible": row["d0_eligible"],
            "d1_eligible": row["d1_eligible"],
        }
        for row in decision_rows
    ]
    receipt = {
        "schema_version": 1,
        "receipt_kind": "DECISION_FREEZE",
        "gate_kind": registered.policy["gate_kind"],
        "run_scope_digest": run_scope_digest,
        "recipe_digest": registered.recipe_digest,
        "replica_id": replica_id,
        "irreversible_receipt_digest": irreversible_receipt_digest,
        "candidate_row_count": len(decision_rows),
        "candidate_projection_digest": canonical_digest(validated),
        "decision_rows_digest": canonical_digest(decision_rows),
        "decision_vector_digest": canonical_digest(vector_projection),
        "settlement_accessed": False,
        "odds_price_popularity_or_market_accessed": False,
        "authority": False,
        "authenticated_phase_output_seal_required": True,
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    receipt["receipt_digest"] = canonical_digest(receipt)
    return decision_rows, receipt


def _settled_candidate_return(row: Mapping[str, Any]) -> float:
    hit = row["candidate_hit"]
    payoff = row["official_wide_pay"]
    if type(hit) is not bool:
        raise ContractError("candidate_hit must be boolean")
    if hit:
        try:
            value = float(payoff)
        except (TypeError, ValueError) as exc:
            raise ContractError("hit=true requires one finite positive payoff") from exc
        if not math.isfinite(value) or value <= 0:
            raise ContractError("hit=true requires one finite positive payoff")
        return value
    if payoff not in (None, "", 0, 0.0):
        raise ContractError("hit=false cannot carry a positive candidate payoff")
    return 0.0


def validate_settlement_rows(
    recipe: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[tuple[str, str], dict[str, Any]]:
    expected_columns = set(
        recipe["catalog_requirements"]["settlement_role"]["required_columns"]
    )
    if len(rows) != recipe["cohort"]["race_count"]:
        raise ContractError("settlement row count differs from enrolled cohort")
    output: dict[tuple[str, str], dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            raise ContractError(f"settlement row {index} is not an object")
        columns = set(raw)
        _ensure_no_market_columns(list(columns))
        if columns != expected_columns:
            raise ContractError(
                f"settlement schema must be exact; observed={sorted(columns)}"
            )
        row = dict(raw)
        if row.get("official_outcome_completeness") is not True:
            raise ContractError("official settlement completeness must be true for every race")
        race_id = row.get("race_id")
        candidate_key = row.get("candidate_key")
        if not isinstance(race_id, str) or not isinstance(candidate_key, str):
            raise ContractError("settlement join keys must be strings")
        key = (race_id, candidate_key)
        if key in output:
            raise ContractError(f"duplicate settlement key {key}")
        row["settled_candidate_return_yen"] = _settled_candidate_return(row)
        output[key] = row
    return output


def _arm_summary(rows: Sequence[Mapping[str, Any]], prefix: str) -> dict[str, Any]:
    eligible_key = f"{prefix}_eligible"
    stake_key = f"{prefix}_stake_yen"
    return_key = f"{prefix}_return_yen"
    bets = sum(bool(row[eligible_key]) for row in rows)
    hits = sum(bool(row[eligible_key] and row["candidate_hit"]) for row in rows)
    stake = sum(float(row[stake_key]) for row in rows)
    returns = sum(float(row[return_key]) for row in rows)
    return {
        "arm": prefix.upper(),
        "bet_count": bets,
        "hit_count": hits,
        "stake_denominator_yen": stake,
        "return_yen": returns,
        "profit_yen": returns - stake,
        "roi_percent": None if stake == 0 else returns / stake * 100.0,
    }


def _apply_returns(
    decision_rows: Sequence[Mapping[str, Any]],
    settlements: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    notional_yen: float,
    return_cap_yen: float | None = None,
    zero_return_race_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    zero_ids = zero_return_race_ids or set()
    paired: list[dict[str, Any]] = []
    decision_keys = {(row["race_id"], row["candidate_key"]) for row in decision_rows}
    if decision_keys != set(settlements):
        missing = sorted(decision_keys - set(settlements))[:5]
        extra = sorted(set(settlements) - decision_keys)[:5]
        raise ContractError(f"candidate/settlement key mismatch; missing={missing}, extra={extra}")
    for decision in decision_rows:
        key = (decision["race_id"], decision["candidate_key"])
        settlement = settlements[key]
        base_return = float(settlement["settled_candidate_return_yen"])
        if return_cap_yen is not None:
            base_return = min(base_return, return_cap_yen)
        if decision["race_id"] in zero_ids:
            base_return = 0.0
        row = dict(decision)
        row["candidate_hit"] = settlement["candidate_hit"]
        row["settled_candidate_return_yen"] = float(
            settlement["settled_candidate_return_yen"]
        )
        for arm in ("d0", "d1"):
            eligible = bool(row[f"{arm}_eligible"])
            row[f"{arm}_stake_yen"] = notional_yen if eligible else 0.0
            row[f"{arm}_return_yen"] = base_return if eligible else 0.0
            row[f"{arm}_profit_yen"] = (
                row[f"{arm}_return_yen"] - row[f"{arm}_stake_yen"]
            )
        row["delta_profit_yen"] = row["d1_profit_yen"] - row["d0_profit_yen"]
        paired.append(row)
    return paired


def _common_high_payout_set(
    settlements: Mapping[tuple[str, str], Mapping[str, Any]], count: int
) -> list[str]:
    ranked = sorted(
        settlements.values(),
        key=lambda row: (-float(row["settled_candidate_return_yen"]), row["race_id"]),
    )
    return [str(row["race_id"]) for row in ranked[:count]]


def _metric_projection(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    d0 = _arm_summary(rows, "d0")
    d1 = _arm_summary(rows, "d1")
    delta_sum = sum(float(row["delta_profit_yen"]) for row in rows)
    disagreement_count = sum(
        bool(row["d0_eligible"] != row["d1_eligible"]) for row in rows
    )
    return {
        "enrolled_race_count": len(rows),
        "d0": d0,
        "d1": d1,
        "sum_delta_profit_yen": delta_sum,
        "mean_delta_profit_yen_per_enrolled_race": delta_sum / len(rows),
        "decision_disagreement_count": disagreement_count,
    }


def _cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]], *, replicates: int, seed: int
) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - runtime dependency contract
        raise ContractError("numpy is required by the registered bootstrap contract") from exc
    clusters: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        clusters[(str(row["race_date"]), str(row["venue_code"]))].append(
            float(row["delta_profit_yen"])
        )
    keys = sorted(clusters)
    if not keys:
        raise ContractError("bootstrap requires at least one cluster")
    arrays = [np.asarray(clusters[key], dtype=np.float64) for key in keys]
    rng = np.random.Generator(np.random.PCG64(seed))
    statistics = np.empty(replicates, dtype=np.float64)
    n_clusters = len(keys)
    for index in range(replicates):
        draw = rng.integers(
            0, n_clusters, size=n_clusters, dtype=np.int64, endpoint=False
        )
        total = 0.0
        count = 0
        for selected in draw:
            values = arrays[int(selected)]
            total += float(values.sum())
            count += int(values.size)
        statistics[index] = total / count
    return {
        "cluster_count": n_clusters,
        "replicates": replicates,
        "seed": seed,
        "rng": "numpy.random.Generator(PCG64)",
        "mean": float(statistics.mean()),
        "one_sided_95_lower_bound": float(
            np.quantile(statistics, 0.05, method="linear")
        ),
        "distribution_digest": canonical_digest(statistics.tolist()),
    }


def _settle_diagnostic_after_authenticated_mount(
    registered: RegisteredRecipe,
    decision_rows: Sequence[Mapping[str, Any]],
    settlement_rows: Sequence[Mapping[str, Any]],
    *,
    run_scope_digest: str,
    replica_id: str,
    decision_freeze_receipt: Mapping[str, Any],
    settlement_operation_receipt_digest: str,
    bootstrap_replicates_override_for_synthetic_test: int | None = None,
) -> dict[str, Any]:
    receipt_digest = decision_freeze_receipt.get("receipt_digest")
    unsigned_receipt = dict(decision_freeze_receipt)
    unsigned_receipt.pop("receipt_digest", None)
    if receipt_digest != canonical_digest(unsigned_receipt):
        raise ContractError("decision freeze receipt was changed")
    if decision_freeze_receipt.get("run_scope_digest") != run_scope_digest:
        raise ContractError("decision receipt belongs to another run")
    if decision_freeze_receipt.get("replica_id") != replica_id:
        raise ContractError("decision receipt belongs to another replica")
    settlements = validate_settlement_rows(registered.recipe, settlement_rows)
    notional = float(registered.recipe["metric"]["offline_evaluation_notional_yen"])
    paired = _apply_returns(decision_rows, settlements, notional_yen=notional)
    primary = _metric_projection(paired)
    top1 = set(_common_high_payout_set(settlements, 1))
    top3 = set(_common_high_payout_set(settlements, 3))
    sensitivities = {
        "common_top1_return_zeroed": _metric_projection(
            _apply_returns(
                decision_rows,
                settlements,
                notional_yen=notional,
                zero_return_race_ids=top1,
            )
        ),
        "common_top3_return_zeroed": _metric_projection(
            _apply_returns(
                decision_rows,
                settlements,
                notional_yen=notional,
                zero_return_race_ids=top3,
            )
        ),
        "common_2000_yen_winsor": _metric_projection(
            _apply_returns(
                decision_rows,
                settlements,
                notional_yen=notional,
                return_cap_yen=float(registered.recipe["sensitivity"]["winsor_cap_yen"]),
            )
        ),
        "top1_race_ids": sorted(top1),
        "top3_race_ids": sorted(top3),
    }
    replicates = registered.recipe["bootstrap"]["replicates"]
    if bootstrap_replicates_override_for_synthetic_test is not None:
        if bootstrap_replicates_override_for_synthetic_test <= 0:
            raise ContractError("synthetic bootstrap override must be positive")
        replicates = bootstrap_replicates_override_for_synthetic_test
    bootstrap = _cluster_bootstrap(
        paired,
        replicates=replicates,
        seed=registered.recipe["bootstrap"]["seed"],
    )
    outcome = (
        "NO_DECISION_EFFECT"
        if primary["decision_disagreement_count"] == 0
        else "DIRECTIONAL_EFFECT"
    )
    scientific_projection = {
        "primary": primary,
        "sensitivity": sensitivities,
        "bootstrap": bootstrap,
        "candidate_projection_digest": decision_freeze_receipt[
            "candidate_projection_digest"
        ],
        "decision_vector_digest": decision_freeze_receipt["decision_vector_digest"],
        "settlement_projection_digest": canonical_digest(
            sorted(settlements.values(), key=lambda row: row["race_id"])
        ),
        "paired_rows_digest": canonical_digest(paired),
        "contract_status": "VALID",
    }
    result = {
        "schema_version": 1,
        "gate_kind": registered.policy["gate_kind"],
        "run_scope_digest": run_scope_digest,
        "recipe_id": registered.recipe["recipe_id"],
        "recipe_version": registered.recipe["recipe_version"],
        "recipe_digest": registered.recipe_digest,
        "replica_id": replica_id,
        "decision_freeze_receipt_digest": receipt_digest,
        "settlement_operation_receipt_digest": settlement_operation_receipt_digest,
        "scientific_projection": scientific_projection,
        "scientific_projection_digest": canonical_digest(scientific_projection),
        "computed_outcome": outcome,
        "authority": False,
        "authenticated_phase_output_seal_required": True,
        "evidence_purpose_class": "DIAGNOSTIC_NONPROMOTION",
        "source_authority_class": registered.recipe["source_authority_class"],
        "confirmatory": False,
        "promotion_eligible": False,
        "score_credit": 0,
        "strict_t3_rows": registered.recipe["cohort"]["strict_t3_rows"],
        "formal_buy": False,
        "send_order": False,
        "stake": 0,
    }
    result["result_digest"] = canonical_digest(result)
    return result


def settlement_result_output_digest(result: Mapping[str, Any]) -> str:
    """Return the authenticated-seal output digest for one settlement result.

    A settlement result is already a canonical, self-digesting projection, so its
    phase-output digest is the verified ``result_digest`` itself.  This helper is
    intentionally stricter than a generic self-hash check: it also verifies the
    finite registered result envelope and its embedded scientific projection.
    """

    if not isinstance(result, Mapping):
        raise ContractError("settlement result must be an object")
    payload = dict(result)
    missing = sorted(SETTLEMENT_RESULT_FIELDS - set(payload))
    extra = sorted(set(payload) - SETTLEMENT_RESULT_FIELDS)
    if missing or extra:
        raise ContractError(
            "settlement result fields must be exact; "
            f"missing={missing}, extra={extra}"
        )
    result_digest = _require_sha256(
        payload.get("result_digest"), "settlement result result_digest"
    )
    unsigned = dict(payload)
    unsigned.pop("result_digest")
    if canonical_digest(unsigned) != result_digest:
        raise ContractError("settlement result self-digest mismatch")

    scientific = payload.get("scientific_projection")
    if not isinstance(scientific, Mapping):
        raise ContractError("settlement scientific projection must be an object")
    scientific_payload = dict(scientific)
    if set(scientific_payload) != SCIENTIFIC_PROJECTION_FIELDS:
        raise ContractError("settlement scientific projection fields are not exact")
    scientific_digest = _require_sha256(
        payload.get("scientific_projection_digest"),
        "settlement scientific_projection_digest",
    )
    if canonical_digest(scientific_payload) != scientific_digest:
        raise ContractError("settlement scientific projection digest mismatch")
    for key in (
        "candidate_projection_digest",
        "decision_vector_digest",
        "settlement_projection_digest",
        "paired_rows_digest",
    ):
        _require_sha256(
            scientific_payload.get(key), f"settlement scientific projection {key}"
        )
    primary = scientific_payload.get("primary")
    if not isinstance(primary, Mapping):
        raise ContractError("settlement primary projection must be an object")
    disagreement_count = primary.get("decision_disagreement_count")
    if (
        not isinstance(disagreement_count, int)
        or isinstance(disagreement_count, bool)
        or disagreement_count < 0
    ):
        raise ContractError("decision_disagreement_count must be non-negative")
    expected_outcome = (
        "NO_DECISION_EFFECT" if disagreement_count == 0 else "DIRECTIONAL_EFFECT"
    )
    if payload.get("computed_outcome") != expected_outcome:
        raise ContractError("settlement computed outcome is inconsistent")
    if scientific_payload.get("contract_status") != "VALID":
        raise ContractError("settlement result contract status is not VALID")
    if (
        payload.get("schema_version") != 1
        or isinstance(payload.get("schema_version"), bool)
        or payload.get("replica_id") not in REPLICA_IDS
        or payload.get("authority") is not False
        or payload.get("authenticated_phase_output_seal_required") is not True
        or payload.get("evidence_purpose_class") != "DIAGNOSTIC_NONPROMOTION"
        or payload.get("confirmatory") is not False
        or payload.get("promotion_eligible") is not False
        or payload.get("score_credit") != 0
        or payload.get("formal_buy") is not False
        or payload.get("send_order") is not False
        or payload.get("stake") != 0
    ):
        raise ContractError("settlement result safety or purpose contract is invalid")
    for key in (
        "run_scope_digest",
        "recipe_digest",
        "decision_freeze_receipt_digest",
        "settlement_operation_receipt_digest",
    ):
        _require_sha256(payload.get(key), f"settlement result {key}")
    return result_digest


def _compare_replica_results(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    if left.get("replica_id") == right.get("replica_id"):
        raise ContractError("replica comparison requires distinct replica IDs")
    for key in ("run_scope_digest", "recipe_digest", "scientific_projection_digest"):
        if left.get(key) != right.get(key):
            raise ContractError(f"replica mismatch in {key}")
    if left.get("computed_outcome") != right.get("computed_outcome"):
        raise ContractError("replica computed outcomes differ")
    if left.get("result_digest") == right.get("result_digest"):
        raise ContractError("replica result digests must be distinct")
    receipt = {
        "schema_version": 1,
        "receipt_kind": "REPLICA_COMPARISON",
        "run_scope_digest": left["run_scope_digest"],
        "recipe_digest": left["recipe_digest"],
        "replica_ids": sorted([left["replica_id"], right["replica_id"]]),
        "replica_result_digests": sorted(
            [left["result_digest"], right["result_digest"]]
        ),
        "scientific_projection_digest": left["scientific_projection_digest"],
        "computed_outcome": left["computed_outcome"],
        "both_contract_status_valid": True,
        "bitwise_semantic_equality": True,
        "authority": False,
        "authenticated_phase_output_seal_required": True,
    }
    receipt["receipt_digest"] = canonical_digest(receipt)
    return receipt


def resolve_run_scope_bound_phase_plan(
    registered: RegisteredRecipe,
    run_scope: Mapping[str, Any],
) -> tuple[str, ...]:
    verify_canonical_run_scope(registered, run_scope)
    resolved = run_scope.get("resolved_contracts")
    if not isinstance(resolved, Mapping):
        raise ContractError("run scope resolved_contracts are missing")
    execution = resolved.get("execution_contract")
    if not isinstance(execution, Mapping):
        raise ContractError("run scope execution contract is missing")
    if (
        execution.get("runner_template_id") != RUNNER_TEMPLATE_ID
        or tuple(execution.get("structured_argv", ())) != STRUCTURED_ARGV
        or execution.get("shell_interpretation") is not False
        or execution.get("free_form_argv_allowed") is not False
        or execution.get("retry_count") != 0
    ):
        raise ContractError("run scope does not bind the canonical executor argv")
    phase_plan = resolved.get("phase_plan")
    expected = (
        "VERIFY_CUTOVER_AND_RUN_SCOPE",
        "DECISION_FREEZE",
        "SETTLEMENT_DIAGNOSTIC",
        "REPLICA_COMPARE",
        "RESULT_SEAL",
    )
    if tuple(phase_plan or ()) != expected:
        raise ContractError("run scope phase plan is not canonical")
    return expected


def decision_output_projection_digest(
    decision_rows: Sequence[Mapping[str, Any]],
    decision_freeze_receipt: Mapping[str, Any],
) -> str:
    rows_digest = canonical_digest([dict(row) for row in decision_rows])
    if decision_freeze_receipt.get("decision_rows_digest") != rows_digest:
        raise ContractError("decision rows do not match their freeze receipt")
    receipt_digest = decision_freeze_receipt.get("receipt_digest")
    _require_sha256(receipt_digest, "decision freeze receipt_digest")
    return canonical_digest(
        {
            "schema_version": 1,
            "projection_kind": "REGISTERED_NONPROMOTION_DECISION_OUTPUT_V1",
            "run_scope_digest": decision_freeze_receipt.get("run_scope_digest"),
            "recipe_digest": decision_freeze_receipt.get("recipe_digest"),
            "replica_id": decision_freeze_receipt.get("replica_id"),
            "decision_rows_digest": rows_digest,
            "decision_freeze_receipt_digest": receipt_digest,
        }
    )


@dataclass(frozen=True)
class SupervisedDiagnosticExecutor:
    """Typed shared-G2 coordinator with no local or duck-typed authority path."""

    registered: RegisteredRecipe
    run_scope: Mapping[str, Any]
    authority: SharedG2LeaseAuthorityClient
    content: AuthenticatedProtectedContentProvider

    def __post_init__(self) -> None:
        if not isinstance(self.authority, SharedG2LeaseAuthorityClient):
            raise ContractError(
                "live executor requires the concrete authenticated shared-G2 client"
            )
        if not isinstance(self.content, AuthenticatedProtectedContentProvider):
            raise ContractError(
                "live executor requires the receipt-gated content provider"
            )
        run_digest = verify_canonical_run_scope(self.registered, self.run_scope)
        resolve_run_scope_bound_phase_plan(self.registered, self.run_scope)
        bindings = self.run_scope["runtime_bindings"]
        if (
            self.run_scope.get("recipe_digest") != self.registered.recipe_digest
            or self.content.run_scope_digest != run_digest
            or self.content.catalog_release_id != bindings["catalog_release_id"]
            or self.content.candidate_entry_sha256
            != bindings["candidate_entry_sha256"]
            or self.content.settlement_entry_sha256
            != bindings["settlement_entry_sha256"]
        ):
            raise ContractError("executor run, recipe, or catalog binding mismatch")
        context = self.authority.context
        expectations = context.expectations
        if (
            expectations.repository != bindings["repository"]
            or expectations.base_branch != bindings["base_branch"]
            or expectations.current_main_sha
            != bindings["verified_current_main_sha"]
            or context.cutover_receipt_digest
            != bindings["cutover_receipt_sha256"]
        ):
            raise ContractError("shared-G2 cutover context does not match the run scope")
        runtime_digests = set(expectations.runtime_blob_digests.values())
        if not set(self.registered.runtime_material_digests.values()).issubset(
            runtime_digests
        ):
            raise ContractError(
                "shared-G2 cutover does not bind every registered runtime material"
            )

    @property
    def run_scope_digest(self) -> str:
        return verify_canonical_run_scope(self.registered, self.run_scope)

    def _validate_consumed_decision_batch(
        self, consumed: ConsumedDecisionLeaseBatch
    ) -> tuple[ConsumedPhaseLease, ConsumedPhaseLease]:
        if not isinstance(consumed, ConsumedDecisionLeaseBatch):
            raise ContractError("decision freeze requires a typed consumed lease batch")
        replica_consumptions = tuple(consumed.replica_consumptions)
        if (
            len(replica_consumptions) != 2
            or tuple(
                item.lease.binding.replica_id for item in replica_consumptions
            )
            != REPLICA_IDS
        ):
            raise ContractError(
                "decision batch must contain clean_a and clean_b exactly once"
            )
        if any(
            item.lease.binding.phase != "DECISION_FREEZE"
            or item.lease.binding.attempt != 1
            or item.lease.binding.run_scope_digest != self.run_scope_digest
            or item.lease.binding.recipe_digest != self.registered.recipe_digest
            or item.transaction != consumed.transaction
            or item.decision_issued_batch != consumed.issued
            or item.decision_consumption_batch != consumed.receipt
            for item in replica_consumptions
        ):
            raise ContractError("decision batch chain is not bound to this executor run")
        return replica_consumptions

    def consume_decision_freeze_batch(
        self,
        *,
        issued_batch: IssuedDecisionLeaseBatch,
        expected_global_head: GlobalHead,
        dispatch_digests: Mapping[str, str],
        consume_revalidation_receipt_digest: str,
        operation_id: str,
        idempotency_key: str,
        requested_at: str,
        irreversible_lifecycle: IrreversibleLifecycleView,
    ) -> ConsumedDecisionLeaseBatch:
        """Atomically consume both decision leases without reading either replica."""

        if not isinstance(issued_batch, IssuedDecisionLeaseBatch):
            raise ContractError("decision freeze requires a typed two-replica lease batch")
        consumed = self.authority.consume_decision_lease_batch(
            issued=issued_batch,
            expected_global_head=expected_global_head,
            dispatch_digests=dispatch_digests,
            consume_revalidation_receipt_digest=(
                consume_revalidation_receipt_digest
            ),
            operation_id=operation_id,
            idempotency_key=idempotency_key,
            requested_at=requested_at,
            irreversible_lifecycle=irreversible_lifecycle,
        )
        if not isinstance(consumed, ConsumedDecisionLeaseBatch):
            raise ContractError("shared-G2 returned an invalid decision batch receipt")
        self._validate_consumed_decision_batch(consumed)
        return consumed

    def decision_freeze_replica(
        self,
        *,
        replica_id: str,
        consumed_batch: ConsumedDecisionLeaseBatch,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Run exactly one named replica after the supervisor's atomic consume."""

        if replica_id not in REPLICA_IDS:
            raise ContractError("decision replica is not registered")
        replica_consumptions = self._validate_consumed_decision_batch(consumed_batch)
        matching = [
            item
            for item in replica_consumptions
            if item.lease.binding.replica_id == replica_id
        ]
        if len(matching) != 1:
            raise ContractError("decision replica label is missing or duplicated")
        rows = self.content.read_candidate_rows(
            consumed_batch=consumed_batch,
            replica_id=replica_id,
        )
        return _freeze_decisions_after_authenticated_mount(
            self.registered,
            rows,
            run_scope_digest=self.run_scope_digest,
            replica_id=replica_id,
            irreversible_receipt_digest=consumed_batch.receipt.payload_digest,
        )

    def settlement_diagnostic(
        self,
        *,
        issued_lease: IssuedPhaseLease,
        expected_global_head: GlobalHead,
        dispatch_digest: str,
        consume_revalidation_receipt_digest: str,
        operation_id: str,
        idempotency_key: str,
        requested_at: str,
        decision_rows: Sequence[Mapping[str, Any]],
        decision_freeze_receipt: Mapping[str, Any],
        decision_output_seal: SealedPhaseOutput,
    ) -> tuple[dict[str, Any], ConsumedPhaseLease]:
        if not isinstance(issued_lease, IssuedPhaseLease) or not isinstance(
            decision_output_seal, SealedPhaseOutput
        ):
            raise ContractError("settlement requires typed lease and decision seal")
        binding = issued_lease.lease.binding
        replica_id = binding.replica_id
        revalidated_predecessor = (
            self.authority.fetch_and_revalidate_unrevoked_phase_output_seal(
                decision_output_seal.receipt.payload_digest
            )
        )
        if (
            binding.phase != "SETTLEMENT_DIAGNOSTIC"
            or replica_id not in REPLICA_IDS
            or binding.attempt != 1
            or binding.run_scope_digest != self.run_scope_digest
            or binding.recipe_digest != self.registered.recipe_digest
            or revalidated_predecessor.receipt != decision_output_seal.receipt
            or canonical_predecessor_output_digest(
                successor_binding=binding,
                predecessor_output_seals=(decision_output_seal,),
            )
            != binding.predecessor_receipt_digest
        ):
            raise ContractError("settlement lease or decision predecessor is invalid")
        consumed = self.authority.consume_phase_lease(
            issued=issued_lease,
            expected_global_head=expected_global_head,
            dispatch_digest=dispatch_digest,
            consume_revalidation_receipt_digest=(
                consume_revalidation_receipt_digest
            ),
            operation_id=operation_id,
            idempotency_key=idempotency_key,
            requested_at=requested_at,
        )
        expected_decision_digest = decision_output_projection_digest(
            decision_rows, decision_freeze_receipt
        )
        if (
            decision_freeze_receipt.get("replica_id") != replica_id
            or decision_freeze_receipt.get("run_scope_digest")
            != self.run_scope_digest
            or decision_output_seal.receipt.output_digest
            != expected_decision_digest
            or decision_output_seal.attestation.output_digest
            != expected_decision_digest
        ):
            raise ContractError(
                "decision output does not match its authenticated settlement predecessor"
            )
        rows = self.content.read_settlement_rows(
            consumed=consumed,
            decision_output_seal=decision_output_seal,
        )
        result = _settle_diagnostic_after_authenticated_mount(
            self.registered,
            decision_rows,
            rows,
            run_scope_digest=self.run_scope_digest,
            replica_id=replica_id,
            decision_freeze_receipt=decision_freeze_receipt,
            settlement_operation_receipt_digest=consumed.receipt.payload_digest,
        )
        return result, consumed

    def replica_compare(
        self,
        *,
        issued_lease: IssuedPhaseLease,
        expected_global_head: GlobalHead,
        dispatch_digest: str,
        consume_revalidation_receipt_digest: str,
        operation_id: str,
        idempotency_key: str,
        requested_at: str,
        clean_a_result: Mapping[str, Any],
        clean_b_result: Mapping[str, Any],
        settlement_output_seals: Sequence[SealedPhaseOutput],
    ) -> tuple[dict[str, Any], ConsumedPhaseLease]:
        """Compare only two remotely current, result-bound settlement outputs."""

        if not isinstance(issued_lease, IssuedPhaseLease):
            raise ContractError("replica comparison requires a typed phase lease")
        seals = tuple(settlement_output_seals)
        if len(seals) != 2 or any(
            not isinstance(seal, SealedPhaseOutput) for seal in seals
        ):
            raise ContractError(
                "replica comparison requires two typed settlement output seals"
            )
        binding = issued_lease.lease.binding
        if (
            binding.phase != "REPLICA_COMPARE"
            or binding.replica_id != REPLICA_COMPARE_ACTOR
            or binding.attempt != 1
            or binding.run_scope_digest != self.run_scope_digest
            or binding.recipe_digest != self.registered.recipe_digest
        ):
            raise ContractError("replica comparison lease is not bound to this run")

        # Results are deliberately not inspected until both predecessor receipts
        # have been re-fetched and proven current by the remote authority/witness.
        revalidated: list[RevalidatedPhaseOutputSeal] = []
        for seal in seals:
            evidence = (
                self.authority.fetch_and_revalidate_unrevoked_phase_output_seal(
                    seal.receipt.payload_digest
                )
            )
            if not isinstance(evidence, RevalidatedPhaseOutputSeal):
                raise ContractError(
                    "shared-G2 returned an untyped settlement seal revalidation"
                )
            revalidated.append(evidence)

        expected_replicas = REPLICA_IDS
        for replica_id, seal, evidence in zip(expected_replicas, seals, revalidated):
            receipt = seal.receipt
            attestation = seal.attestation
            consumed = seal.consumed
            if not isinstance(attestation, TrustedPhaseOutputAttestation) or not isinstance(
                consumed, ConsumedPhaseLease
            ):
                raise ContractError("settlement seal chain is not fully typed")
            if (
                evidence.receipt != receipt
                or receipt.phase != "SETTLEMENT_DIAGNOSTIC"
                or receipt.replica_id != replica_id
                or receipt.run_scope_digest != self.run_scope_digest
                or receipt.recipe_digest != self.registered.recipe_digest
                or receipt.attempt != 1
                or consumed.lease.binding.phase != "SETTLEMENT_DIAGNOSTIC"
                or consumed.lease.binding.replica_id != replica_id
                or consumed.lease.binding.run_scope_digest != self.run_scope_digest
                or consumed.lease.binding.recipe_digest
                != self.registered.recipe_digest
                or receipt.binding_digest != consumed.lease.binding.digest
                or receipt.lease_payload_digest != consumed.lease.lease_digest
                or receipt.lease_consumption_receipt_digest
                != consumed.receipt.payload_digest
                or receipt.output_attestation_digest != attestation.payload_digest
                or receipt.output_digest != attestation.output_digest
            ):
                raise ContractError(
                    "settlement output seal is not the authenticated replica result"
                )

        predecessor_digest = canonical_predecessor_output_digest(
            successor_binding=binding,
            predecessor_output_seals=seals,
        )
        if predecessor_digest != binding.predecessor_receipt_digest:
            raise ContractError(
                "replica comparison predecessor digest does not match the two seals"
            )

        # The phase lease is consumed before any protected settlement result is
        # inspected.  A failed/stale consume therefore cannot leak result fields.
        consumed_compare = self.authority.consume_phase_lease(
            issued=issued_lease,
            expected_global_head=expected_global_head,
            dispatch_digest=dispatch_digest,
            consume_revalidation_receipt_digest=(
                consume_revalidation_receipt_digest
            ),
            operation_id=operation_id,
            idempotency_key=idempotency_key,
            requested_at=requested_at,
        )
        if (
            not isinstance(consumed_compare, ConsumedPhaseLease)
            or consumed_compare.lease != issued_lease.lease
        ):
            raise ContractError(
                "shared-G2 returned an invalid replica comparison consumption"
            )

        # Snapshot caller mappings exactly once after predecessor authentication
        # and irreversible comparison-phase consumption.
        result_by_replica = {
            "clean_a": dict(clean_a_result),
            "clean_b": dict(clean_b_result),
        }
        for replica_id, seal in zip(expected_replicas, seals):
            result = result_by_replica[replica_id]
            output_digest = settlement_result_output_digest(result)
            if (
                result.get("replica_id") != replica_id
                or result.get("run_scope_digest") != self.run_scope_digest
                or result.get("recipe_digest") != self.registered.recipe_digest
                or result.get("recipe_id")
                != self.registered.recipe.get("recipe_id")
                or result.get("recipe_version")
                != self.registered.recipe.get("recipe_version")
                or result.get("gate_kind")
                != self.registered.policy.get("gate_kind")
                or result.get("source_authority_class")
                != self.registered.recipe.get("source_authority_class")
                or result.get("strict_t3_rows")
                != self.registered.recipe["cohort"].get("strict_t3_rows")
                or result.get("settlement_operation_receipt_digest")
                != seal.consumed.receipt.payload_digest
                or seal.receipt.output_digest != output_digest
                or seal.attestation.output_digest != output_digest
            ):
                raise ContractError(
                    "settlement result does not match its authenticated phase-output seal"
                )

        comparison = _compare_replica_results(
            result_by_replica["clean_a"], result_by_replica["clean_b"]
        )
        return comparison, consumed_compare


def RUN_SCOPE_BOUND_PHASE_PLAN(
    *,
    registered: RegisteredRecipe,
    run_scope: Mapping[str, Any],
    authority: SharedG2LeaseAuthorityClient,
    content_transport: ProtectedContentTransport,
) -> SupervisedDiagnosticExecutor:
    """Canonical structured-argv target; it only assembles typed runtime clients."""

    run_digest = verify_canonical_run_scope(registered, run_scope)
    resolve_run_scope_bound_phase_plan(registered, run_scope)
    if not isinstance(authority, SharedG2LeaseAuthorityClient):
        raise ContractError(
            "RUN_SCOPE_BOUND_PHASE_PLAN rejects local or duck-typed authority"
        )
    bindings = run_scope["runtime_bindings"]
    content = AuthenticatedProtectedContentProvider(
        transport=content_transport,
        run_scope_digest=run_digest,
        catalog_release_id=bindings["catalog_release_id"],
        candidate_entry_sha256=bindings["candidate_entry_sha256"],
        settlement_entry_sha256=bindings["settlement_entry_sha256"],
    )
    return SupervisedDiagnosticExecutor(
        registered=registered,
        run_scope=run_scope,
        authority=authority,
        content=content,
    )


__all__ = [
    "AuthenticatedProtectedContentProvider",
    "ProtectedContentTransport",
    "RUNNER_TEMPLATE_ID",
    "RUN_SCOPE_BOUND_PHASE_PLAN",
    "STRUCTURED_ARGV",
    "SupervisedDiagnosticExecutor",
    "decision_output_projection_digest",
    "resolve_run_scope_bound_phase_plan",
    "settlement_result_output_digest",
    "validate_candidate_rows",
    "validate_settlement_rows",
]
