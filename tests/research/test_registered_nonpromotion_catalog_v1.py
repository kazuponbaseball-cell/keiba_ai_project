from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "research"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import registered_nonpromotion_catalog_v1 as catalog


H = {character: character * 64 for character in "123456789abcdef"}
CANDIDATE_COLUMNS = [
    "candidate_generated",
    "candidate_key",
    "eligible_race",
    "fold",
    "horse_a",
    "horse_b",
    "p_action_C0_offset",
    "race_date",
    "race_id",
    "top1_wide_prob",
    "venue_code",
]
CANDIDATE_FORBIDDEN_COLUMNS = [
    "candidate_hit",
    "hit",
    "official_outcome_joined",
    "official_wide_pay",
    "return_yen",
    "roi",
    "wide_odds_high",
    "wide_odds_low",
    "wide_pay",
    "wide_popularity",
]
SETTLEMENT_COLUMNS = [
    "race_id",
    "candidate_key",
    "candidate_hit",
    "official_outcome_completeness",
    "official_wide_pay",
]


def sample_requirements() -> dict[str, Any]:
    return {
        "exact_active_release_count": 1,
        "exact_entry_count": 2,
        "allowed_roles": [
            catalog.CANDIDATE_ROLE,
            catalog.SETTLEMENT_ROLE,
        ],
        "candidate_role": {
            "required_columns": list(CANDIDATE_COLUMNS),
            "forbidden_columns": list(CANDIDATE_FORBIDDEN_COLUMNS),
            "key": ["race_id", "candidate_key"],
        },
        "settlement_role": {
            "required_columns": list(SETTLEMENT_COLUMNS),
            "key": ["race_id", "candidate_key"],
        },
        "candidate_and_settlement_must_be_distinct_content_objects": True,
    }


def sample_entry(role: str) -> dict[str, Any]:
    if role == catalog.CANDIDATE_ROLE:
        content_digest = H["1"]
        entry_id = "candidate-entry-v1"
        columns = sorted(CANDIDATE_COLUMNS)
        schema_digest = H["3"]
        provenance_digest = H["4"]
        role_attestations = {
            "p_action_cross_source_equality_attestation_sha256": H["a"],
            "candidate_materializer_usecols_sha256": H["b"],
            "decision_base_lineage_sha256": H["c"],
        }
    elif role == catalog.SETTLEMENT_ROLE:
        content_digest = H["2"]
        entry_id = "settlement-entry-v1"
        columns = sorted(SETTLEMENT_COLUMNS)
        schema_digest = H["5"]
        provenance_digest = H["6"]
        role_attestations = {
            "official_settlement_provenance_sha256": H["d"],
        }
    else:
        raise AssertionError("test fixture role is unknown")
    return {
        "entry_id": entry_id,
        "role": role,
        "object_id": f"sha256:{content_digest}",
        "content_sha256": content_digest,
        "schema_sha256": schema_digest,
        "provenance_sha256": provenance_digest,
        "role_attestations": role_attestations,
        "columns": columns,
        "key_columns": ["race_id", "candidate_key"],
        "row_count": 3746,
        "race_count": 3746,
        "ordered_race_id_set_sha256": H["7"],
    }


def sample_release(
    release_id: str = "rnd-release-v1",
    *,
    status: str = "ACTIVE",
    revoked: bool = False,
) -> dict[str, Any]:
    entries = [
        sample_entry(catalog.CANDIDATE_ROLE),
        sample_entry(catalog.SETTLEMENT_ROLE),
    ]
    return {
        "schema_version": 1,
        "release_id": release_id,
        "status": status,
        "revoked": revoked,
        "manifest_sha256": catalog.release_manifest_sha256(
            release_id=release_id,
            normalized_entries=entries,
        ),
        "authentication_receipt_sha256": H["8"],
        "status_receipt_sha256": H["9"],
        "entries": entries,
    }


def reseal(release: dict[str, Any]) -> None:
    entries = sorted(release["entries"], key=lambda entry: entry["role"])
    release["manifest_sha256"] = catalog.release_manifest_sha256(
        release_id=release["release_id"],
        normalized_entries=entries,
    )


class PoisonedContentProvider:
    def __init__(self, releases: list[dict[str, Any]]) -> None:
        self._releases = copy.deepcopy(releases)
        self.metadata_calls = 0
        self.row_calls = 0
        self.content_calls = 0

    def list_authenticated_release_metadata(self) -> list[dict[str, Any]]:
        self.metadata_calls += 1
        return copy.deepcopy(self._releases)

    def read_rows(self, *_args: Any, **_kwargs: Any) -> None:
        self.row_calls += 1
        raise AssertionError("pregrant validation attempted to read rows")

    def read_content(self, *_args: Any, **_kwargs: Any) -> None:
        self.content_calls += 1
        raise AssertionError("pregrant validation attempted to read content")


class FailingMetadataProvider:
    def list_authenticated_release_metadata(self) -> list[dict[str, Any]]:
        raise RuntimeError("synthetic remote failure")


class RegisteredNonpromotionCatalogV1Tests(unittest.TestCase):
    def validate(
        self,
        releases: list[dict[str, Any]],
        *,
        requirements: dict[str, Any] | None = None,
        expected_race_count: int = 3746,
    ) -> tuple[dict[str, Any], PoisonedContentProvider]:
        provider = PoisonedContentProvider(releases)
        result = catalog.validate_pregrant_catalog_metadata(
            provider,
            catalog_requirements=requirements or sample_requirements(),
            expected_race_count=expected_race_count,
        )
        return result, provider

    def assert_invalid(
        self,
        releases: list[dict[str, Any]],
        *,
        requirements: dict[str, Any] | None = None,
    ) -> PoisonedContentProvider:
        provider = PoisonedContentProvider(releases)
        with self.assertRaises(catalog.CatalogValidationError):
            catalog.validate_pregrant_catalog_metadata(
                provider,
                catalog_requirements=requirements or sample_requirements(),
                expected_race_count=3746,
            )
        return provider

    def test_metadata_only_resolution_selects_exactly_one_active_release(self) -> None:
        inactive = sample_release("rnd-release-old", status="INACTIVE")
        active = sample_release()
        result, provider = self.validate([inactive, active])

        self.assertEqual(result["release_id"], "rnd-release-v1")
        self.assertEqual(result["race_count"], 3746)
        self.assertTrue(result["metadata_only"])
        self.assertEqual(result["candidate_entry"]["role"], catalog.CANDIDATE_ROLE)
        self.assertEqual(
            result["settlement_entry"]["role"], catalog.SETTLEMENT_ROLE
        )
        self.assertEqual(provider.metadata_calls, 1)
        self.assertEqual(provider.row_calls, 0)
        self.assertEqual(provider.content_calls, 0)

    def test_zero_or_multiple_active_releases_fail_closed(self) -> None:
        with self.subTest("zero"):
            self.assert_invalid([sample_release(status="INACTIVE")])
        with self.subTest("multiple"):
            self.assert_invalid(
                [sample_release("rnd-release-v1"), sample_release("rnd-release-v2")]
            )

    def test_release_requires_exact_candidate_and_settlement_roles(self) -> None:
        release = sample_release()
        release["entries"][1] = copy.deepcopy(release["entries"][0])
        release["entries"][1]["entry_id"] = "candidate-entry-v2"
        release["entries"][1]["content_sha256"] = H["a"]
        release["entries"][1]["object_id"] = f"sha256:{H['a']}"
        self.assert_invalid([release])

        release = sample_release()
        release["entries"].append(sample_entry(catalog.CANDIDATE_ROLE))
        self.assert_invalid([release])

    def test_candidate_and_settlement_objects_must_be_distinct(self) -> None:
        release = sample_release()
        release["entries"][1]["content_sha256"] = H["1"]
        release["entries"][1]["object_id"] = f"sha256:{H['1']}"
        self.assert_invalid([release])

    def test_candidate_role_attestations_are_exact_full_sha256_values(self) -> None:
        required_fields = (
            "p_action_cross_source_equality_attestation_sha256",
            "candidate_materializer_usecols_sha256",
            "decision_base_lineage_sha256",
        )
        for field in required_fields:
            with self.subTest(missing=field):
                release = sample_release()
                del release["entries"][0]["role_attestations"][field]
                self.assert_invalid([release])

        release = sample_release()
        release["entries"][0]["role_attestations"]["extra_attestation_sha256"] = H[
            "e"
        ]
        self.assert_invalid([release])

        release = sample_release()
        release["entries"][0]["role_attestations"][required_fields[0]] = "not-a-sha"
        self.assert_invalid([release])

    def test_settlement_role_attestation_is_exact_full_sha256_value(self) -> None:
        release = sample_release()
        del release["entries"][1]["role_attestations"][
            "official_settlement_provenance_sha256"
        ]
        self.assert_invalid([release])

        release = sample_release()
        release["entries"][1]["role_attestations"][
            "candidate_materializer_usecols_sha256"
        ] = H["e"]
        self.assert_invalid([release])

        release = sample_release()
        release["entries"][1]["role_attestations"][
            "official_settlement_provenance_sha256"
        ] = "f" * 63
        self.assert_invalid([release])

    def test_role_attestations_are_bound_by_release_manifest_digest(self) -> None:
        release = sample_release()
        release["entries"][0]["role_attestations"][
            "decision_base_lineage_sha256"
        ] = H["e"]
        # The attestation remains structurally valid, but changing it without
        # resealing the release must invalidate the content-addressed manifest.
        self.assert_invalid([release])

    def test_candidate_entry_uses_exact_explicit_allowlist(self) -> None:
        release = sample_release()
        release["entries"][0]["columns"].append("debug_score")
        self.assert_invalid([release])

        release = sample_release()
        release["entries"][0]["columns"].remove("eligible_race")
        self.assert_invalid([release])

    def test_candidate_result_payoff_odds_and_market_fields_are_hard_forbidden(self) -> None:
        for forbidden_column in (
            "candidate_hit",
            "official_outcome_completeness",
            "official_wide_pay",
            "closing_odds",
            "market_price",
            "roi",
        ):
            with self.subTest(forbidden_column):
                requirements = sample_requirements()
                requirements["candidate_role"]["required_columns"].append(
                    forbidden_column
                )
                release = sample_release()
                release["entries"][0]["columns"].append(forbidden_column)
                provider = self.assert_invalid(
                    [release], requirements=requirements
                )
                # Requirements are rejected before even authenticated metadata is fetched.
                self.assertEqual(provider.metadata_calls, 0)
                self.assertEqual(provider.row_calls, 0)
                self.assertEqual(provider.content_calls, 0)

    def test_settlement_entry_uses_exact_fixed_allowlist(self) -> None:
        release = sample_release()
        release["entries"][1]["columns"].append("wide_odds_low")
        self.assert_invalid([release])

        requirements = sample_requirements()
        requirements["settlement_role"]["required_columns"].remove(
            "official_outcome_completeness"
        )
        provider = self.assert_invalid(
            [sample_release()], requirements=requirements
        )
        self.assertEqual(provider.metadata_calls, 0)

    def test_content_address_and_manifest_are_verified(self) -> None:
        release = sample_release()
        release["entries"][0]["object_id"] = f"sha256:{H['a']}"
        self.assert_invalid([release])

        release = sample_release()
        release["manifest_sha256"] = H["a"]
        self.assert_invalid([release])

    def test_candidate_and_settlement_must_bind_same_ordered_race_set(self) -> None:
        release = sample_release()
        release["entries"][1]["ordered_race_id_set_sha256"] = H["a"]
        reseal(release)
        self.assert_invalid([release])

    def test_frozen_cohort_count_and_one_row_per_race_are_enforced(self) -> None:
        release = sample_release()
        release["entries"][0]["row_count"] = 3745
        self.assert_invalid([release])

        release = sample_release()
        release["entries"][0]["row_count"] = 3747
        release["entries"][0]["race_count"] = 3747
        self.assert_invalid([release])

    def test_unknown_fields_cannot_smuggle_rows_or_content_into_metadata(self) -> None:
        release = sample_release()
        release["entries"][0]["rows"] = [{"race_id": "synthetic"}]
        provider = self.assert_invalid([release])
        self.assertEqual(provider.row_calls, 0)
        self.assertEqual(provider.content_calls, 0)

        release = sample_release()
        release["content"] = "synthetic raw payload"
        self.assert_invalid([release])

    def test_revocation_state_and_metadata_provider_failure_fail_closed(self) -> None:
        release = sample_release(status="ACTIVE", revoked=True)
        self.assert_invalid([release])

        with self.assertRaises(catalog.CatalogValidationError):
            catalog.validate_pregrant_catalog_metadata(
                FailingMetadataProvider(),
                catalog_requirements=sample_requirements(),
                expected_race_count=3746,
            )

    def test_boolean_cannot_launder_integer_counts(self) -> None:
        release = sample_release()
        release["entries"][0]["row_count"] = True
        self.assert_invalid([release])

        requirements = sample_requirements()
        requirements["exact_active_release_count"] = True
        provider = self.assert_invalid(
            [sample_release()], requirements=requirements
        )
        self.assertEqual(provider.metadata_calls, 0)


if __name__ == "__main__":
    unittest.main()
