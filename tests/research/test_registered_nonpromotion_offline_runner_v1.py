from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "research"
TEST_DIR = ROOT / "tests" / "research"
for directory in (SCRIPT_DIR, TEST_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import registered_nonpromotion_offline_runner_v1 as runner
from registered_nonpromotion_contract_v1 import ContractError, canonical_digest
from registered_nonpromotion_offline_contract_v1 import resolve_offline_registered_recipe
from test_registered_nonpromotion_schema_v1 import _validate


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _approval_evidence(
    checkpoint: str,
    *,
    marker: str,
    original_evidence_digest: str | None = None,
) -> dict[str, object]:
    evidence: dict[str, object] = {
        "run_scope_digest": "d" * 64,
        "verification_checkpoint": checkpoint,
        "original_evidence_digest": original_evidence_digest,
        "comment": {"issue_number": 7, "comment_id": 11},
        "marker": marker,
    }
    evidence["evidence_digest"] = canonical_digest(evidence)
    return evidence


class OfflineRunnerFixtureMixin:
    def build_sources(self, root: Path) -> tuple[dict[str, dict[str, str]], list[dict[str, object]]]:
        master_path = root / "master.csv"
        p_action_path = root / "p_action.csv"
        payoff_path = root / "payoff.csv"
        fold_sequence = ["fold2"] * 1661 + ["fold3"] * 1653 + ["fold4"] * 432
        master_header = list(
            json.loads((ROOT / runner.POLICY_RELATIVE_PATH).read_text(encoding="utf-8"))[
                "projection_contract"
            ]["raw_source_allowed_projection_columns"]["diagnostic_master"]
        ) + ["hit", "wide_pay", "wide_popularity", "roi"]
        p_action_header = list(
            json.loads((ROOT / runner.POLICY_RELATIVE_PATH).read_text(encoding="utf-8"))[
                "projection_contract"
            ]["raw_source_allowed_projection_columns"]["p_action_artifact"]
        ) + ["target_label", "wide_pay"]
        payoff_header = ["race_id", "horse_a", "horse_b", "wide_pay", "wide_popularity"]
        candidates: list[dict[str, object]] = []
        with (
            master_path.open("w", encoding="utf-8", newline="") as master_handle,
            p_action_path.open("w", encoding="utf-8", newline="") as p_action_handle,
            payoff_path.open("w", encoding="utf-8", newline="") as payoff_handle,
        ):
            master_writer = csv.DictWriter(master_handle, fieldnames=master_header)
            p_action_writer = csv.DictWriter(p_action_handle, fieldnames=p_action_header)
            payoff_writer = csv.DictWriter(payoff_handle, fieldnames=payoff_header)
            master_writer.writeheader()
            p_action_writer.writeheader()
            payoff_writer.writeheader()
            for index, fold in enumerate(fold_sequence):
                race_id = f"{2025000000000000 + index:016d}"
                horse_a = index % 15 + 1
                horse_b = horse_a + 1
                candidate_key = f"{horse_a}-{horse_b}"
                p = 0.34 if index == 0 else 0.25
                p_text = format(p, ".17g")
                a_text = format(runner._expected_p_action(p), ".17g")
                race_date = (
                    runner.DATE_MIN
                    if index == 0
                    else runner.DATE_MAX
                    if index == len(fold_sequence) - 1
                    else "2025-06-01"
                )
                candidate = {
                    "candidate_generated": "true",
                    "eligible_race": "true",
                    "fold": fold,
                    "horse_a": str(horse_a),
                    "horse_b": str(horse_b),
                    "p_action_C0_offset": a_text,
                    "race_date": race_date,
                    "race_id": race_id,
                    "top1_pair_key": candidate_key,
                    "top1_wide_prob": p_text,
                    "venue_code": f"{index % 10 + 1:02d}",
                }
                master_writer.writerow(
                    {
                        **candidate,
                        "hit": "DO_NOT_SELECT",
                        "wide_pay": "DO_NOT_SELECT",
                        "wide_popularity": "DO_NOT_SELECT",
                        "roi": "DO_NOT_SELECT",
                    }
                )
                p_action_writer.writerow(
                    {
                        **{key: candidate[key] for key in p_action_header if key in candidate},
                        "target_label": "DO_NOT_SELECT",
                        "wide_pay": "DO_NOT_SELECT",
                    }
                )
                hit = index % 2 == 0
                payoff_writer.writerow(
                    {
                        "race_id": race_id,
                        "horse_a": str(horse_a if hit else 17),
                        "horse_b": str(horse_b if hit else 18),
                        "wide_pay": str(120 + index % 10),
                        "wide_popularity": "999",
                    }
                )
                payoff_writer.writerow(
                    {
                        "race_id": race_id,
                        "horse_a": "19",
                        "horse_b": "20",
                        "wide_pay": "130",
                        "wide_popularity": "998",
                    }
                )
                payoff_writer.writerow(
                    {
                        "race_id": race_id,
                        "horse_a": "21",
                        "horse_b": "22",
                        "wide_pay": "140",
                        "wide_popularity": "997",
                    }
                )
                candidates.append(
                    {
                        "race_id": race_id,
                        "candidate_key": candidate_key,
                        "hit": hit,
                    }
                )
        specs = {
            "diagnostic_master": {"path": "master.csv", "expected_sha256": _sha(master_path)},
            "p_action_artifact": {"path": "p_action.csv", "expected_sha256": _sha(p_action_path)},
            "official_payoff_source": {"path": "payoff.csv", "expected_sha256": _sha(payoff_path)},
        }
        return specs, candidates

    def policy(self) -> dict[str, object]:
        return json.loads((ROOT / runner.POLICY_RELATIVE_PATH).read_text(encoding="utf-8"))

    def materialize(self, repo: Path, source: Path, specs: dict[str, dict[str, str]]) -> dict[str, object]:
        return runner._materialize_core(
            repo_root=repo,
            source_root=source,
            policy=self.policy(),
            implementation_binding={
                "implementation_commit": "a" * 40,
                "runtime_material_bundle_sha256": canonical_digest(
                    {"runner": "b" * 64}
                ),
            },
            expected_source_inputs=specs,
            expected_race_count=runner.RACE_COUNT,
            expected_fold_counts=runner.FOLD_COUNTS,
        )


class RegisteredNonpromotionOfflineRunnerV1Tests(
    OfflineRunnerFixtureMixin, unittest.TestCase
):
    def test_p_action_binary64_nextafter_boundary_and_mismatch(self) -> None:
        p = 0.325
        expected = runner._expected_p_action(p)
        runner._verify_p_action_formula(format(p, ".17g"), format(expected, ".17g"))
        adjacent = math.nextafter(expected, math.inf)
        runner._verify_p_action_formula(format(p, ".17g"), format(adjacent, ".17g"))
        with self.assertRaisesRegex(ContractError, "fixed calibrator formula"):
            runner._verify_p_action_formula(
                format(p, ".17g"), format(expected + 2e-12, ".17g")
            )

    def test_live_bootstrap_is_fixed_and_override_requires_private_capability(self) -> None:
        self.assertEqual(runner.BOOTSTRAP_REPLICATES, 100000)
        parameters = inspect.signature(
            runner.execute_offline_registered_diagnostic
        ).parameters
        self.assertIn("source_root", parameters)
        for forbidden in (
            "provider",
            "now",
            "_bootstrap_replicates_override_for_synthetic_test",
        ):
            self.assertNotIn(forbidden, parameters)
        self.assertNotIn(
            "_execute_offline_registered_diagnostic_worker", runner.__all__
        )
        for public_api in (
            runner.materialize_fixed_projections,
            runner.compile_fixed_run_scope_artifact,
        ):
            public_parameters = inspect.signature(public_api).parameters
            self.assertNotIn("provider", public_parameters)
            self.assertNotIn("now", public_parameters)
        self.assertNotIn("_materialize_fixed_projections_worker", runner.__all__)
        self.assertNotIn("_compile_fixed_run_scope_artifact_worker", runner.__all__)
        with self.assertRaises(TypeError):
            runner.compile_fixed_run_scope_artifact(  # type: ignore[call-arg]
                root=ROOT,
                source_root=ROOT,
                provider=object(),
                now="1970-01-01T00:00:00Z",
            )

    def test_materializer_is_exact_usecols_decision_free_and_ordered(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            source = workspace / "source"
            repo = workspace / "repo"
            source.mkdir()
            repo.mkdir()
            specs, _ = self.build_sources(source)
            open_order: list[str] = []
            original = runner._read_verified_source_bytes

            def observed(path: Path, *, expected_sha256: str, role: str):
                if role == "official_payoff_source":
                    candidates = list(
                        repo.glob(
                            "outputs/research/registered_nonpromotion_offline_materialized/"
                            ".offline-materialize-*/candidate_projection.jsonl"
                        )
                    )
                    self.assertEqual(len(candidates), 1)
                    self.assertGreater(candidates[0].stat().st_size, 0)
                open_order.append(role)
                return original(path, expected_sha256=expected_sha256, role=role)

            with mock.patch.object(runner, "_read_verified_source_bytes", side_effect=observed):
                evidence = self.materialize(repo, source, specs)
            self.assertEqual(
                open_order,
                ["diagnostic_master", "p_action_artifact", "official_payoff_source"],
            )
            self.assertFalse(evidence["decisions_computed"])
            self.assertFalse(evidence["metrics_computed"])
            self.assertFalse(evidence["roi_computed"])
            candidate_path = repo / runner.PROJECTION_INPUTS["candidate_projection"]["path"]
            settlement_path = repo / runner.PROJECTION_INPUTS["settlement_projection"]["path"]
            manifest_path = repo / runner.MATERIALIZATION_MANIFEST_PATH
            candidate_text = candidate_path.read_text(encoding="utf-8")
            settlement_text = settlement_path.read_text(encoding="utf-8")
            self.assertNotIn("wide_popularity", candidate_text)
            self.assertNotIn("wide_pay", candidate_text)
            self.assertNotIn("candidate_hit", candidate_text)
            self.assertNotIn("DO_NOT_SELECT", candidate_text)
            self.assertNotIn("wide_popularity", settlement_text)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(manifest["raw_forbidden_semantic_values_selected"])
            self.assertEqual(manifest["projection_rows"]["candidate_projection"], 3746)
            self.assertRegex(manifest["ordered_race_id_sha256"], r"^[0-9a-f]{64}$")

    def test_cross_source_or_formula_mismatch_stops_before_payoff_open(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            source = workspace / "source"
            repo = workspace / "repo"
            source.mkdir()
            repo.mkdir()
            specs, _ = self.build_sources(source)
            p_path = source / "p_action.csv"
            with p_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            a_index = rows[0].index("p_action_C0_offset")
            rows[1][a_index] = "0.9"
            with p_path.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerows(rows)
            specs["p_action_artifact"]["expected_sha256"] = _sha(p_path)
            opened: list[str] = []
            original = runner._read_verified_source_bytes

            def observed(path: Path, *, expected_sha256: str, role: str):
                opened.append(role)
                return original(path, expected_sha256=expected_sha256, role=role)

            with (
                mock.patch.object(runner, "_read_verified_source_bytes", side_effect=observed),
                self.assertRaises(ContractError),
            ):
                self.materialize(repo, source, specs)
            self.assertNotIn("official_payoff_source", opened)

    def test_exact_unique_race_and_date_cohort_fail_before_payoff_open(self) -> None:
        for mutation in ("duplicate_race", "date_endpoint"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as raw:
                workspace = Path(raw)
                source = workspace / "source"
                repo = workspace / "repo"
                source.mkdir()
                repo.mkdir()
                specs, _ = self.build_sources(source)
                master_path = source / "master.csv"
                with master_path.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.reader(handle))
                if mutation == "duplicate_race":
                    race_index = rows[0].index("race_id")
                    rows[2][race_index] = rows[1][race_index]
                else:
                    date_index = rows[0].index("race_date")
                    rows[1][date_index] = "2025-01-06"
                with master_path.open("w", encoding="utf-8", newline="") as handle:
                    csv.writer(handle).writerows(rows)
                specs["diagnostic_master"]["expected_sha256"] = _sha(master_path)
                opened: list[str] = []
                original = runner._read_verified_source_bytes

                def observed(path: Path, *, expected_sha256: str, role: str):
                    opened.append(role)
                    return original(
                        path, expected_sha256=expected_sha256, role=role
                    )

                with (
                    mock.patch.object(
                        runner, "_read_verified_source_bytes", side_effect=observed
                    ),
                    self.assertRaises(ContractError),
                ):
                    self.materialize(repo, source, specs)
                self.assertNotIn("official_payoff_source", opened)

    def test_candidate_projection_drift_after_pre_settlement_seal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            source = workspace / "source"
            repo = workspace / "repo"
            source.mkdir()
            repo.mkdir()
            specs, _ = self.build_sources(source)
            original = runner._read_verified_source_bytes

            def mutate_after_seal(path: Path, *, expected_sha256: str, role: str):
                if role == "official_payoff_source":
                    candidate = next(
                        repo.glob(
                            "outputs/research/registered_nonpromotion_offline_materialized/"
                            ".offline-materialize-*/candidate_projection.jsonl"
                        )
                    )
                    with candidate.open("ab") as handle:
                        handle.write(b" ")
                return original(path, expected_sha256=expected_sha256, role=role)

            with (
                mock.patch.object(
                    runner,
                    "_read_verified_source_bytes",
                    side_effect=mutate_after_seal,
                ),
                self.assertRaisesRegex(ContractError, "changed after its pre-settlement seal"),
            ):
                self.materialize(repo, source, specs)

    def test_incomplete_official_wide_payout_set_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            source = workspace / "source"
            repo = workspace / "repo"
            source.mkdir()
            repo.mkdir()
            specs, _ = self.build_sources(source)
            payoff_path = source / "payoff.csv"
            with payoff_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            first_race = rows[1][rows[0].index("race_id")]
            retained = [
                rows[0],
                *[
                    row
                    for index, row in enumerate(rows[1:])
                    if row[rows[0].index("race_id")] != first_race or index < 2
                ],
            ]
            with payoff_path.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerows(retained)
            specs["official_payoff_source"]["expected_sha256"] = _sha(payoff_path)
            with self.assertRaisesRegex(ContractError, "complete 3-to-7-row"):
                self.materialize(repo, source, specs)

    def test_scope_seal_reproduction_rejects_self_consistent_forged_projection(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            source = workspace / "source"
            repo = workspace / "repo"
            source.mkdir()
            repo.mkdir()
            specs, _ = self.build_sources(source)
            self.materialize(repo, source, specs)
            candidate_path = repo / runner.PROJECTION_INPUTS["candidate_projection"]["path"]
            lines = candidate_path.read_text(encoding="utf-8").splitlines()
            forged = json.loads(lines[0])
            forged["eligible_race"] = False
            lines[0] = json.dumps(forged, sort_keys=True, separators=(",", ":"))
            candidate_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
            manifest_path = repo / runner.MATERIALIZATION_MANIFEST_PATH
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["projection_bindings"]["candidate_projection"].update(
                {
                    "sha256": _sha(candidate_path),
                    "byte_size": candidate_path.stat().st_size,
                }
            )
            manifest.pop("manifest_digest")
            manifest["manifest_digest"] = canonical_digest(manifest)
            manifest_path.write_bytes(runner._canonical_bytes(manifest))
            registered = SimpleNamespace(
                policy=self.policy(), runtime_material_digests={"runner": "b" * 64}
            )
            with (
                mock.patch.object(runner, "SOURCE_INPUTS", specs),
                self.assertRaisesRegex(ContractError, "not the exact deterministic projection"),
            ):
                runner._verify_deterministic_materialization_against_raw(
                    repo_root=repo,
                    source_root=source,
                    registered=registered,
                    availability={"implementation_commit": "a" * 40},
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            scope = {
                "verified_current_main_sha": "a" * 40,
                "run_scope_base_commit": "a" * 40,
                "run_scope_digest": "d" * 64,
                "semantic_subject_digest": "e" * 64,
                "exact_subject_digest": "f" * 64,
                "output_root": runner.FIXED_OUTPUT_ROOT,
                "runtime_bindings": {
                    "environment_manifest_sha256": "3" * 64,
                    "materialization_manifest": {
                        "path": runner.MATERIALIZATION_MANIFEST_PATH,
                        "sha256": _sha(manifest_path),
                        "byte_size": manifest_path.stat().st_size,
                    },
                    "source_bindings": manifest["source_bindings"],
                    "projection_bindings": manifest["projection_bindings"],
                },
            }
            approval = SimpleNamespace(
                verify_offline_run_approval=mock.Mock(
                    return_value=_approval_evidence(
                        "INITIAL_APPROVAL", marker="initial"
                    )
                ),
                reverify_offline_run_approval=mock.Mock(
                    return_value=_approval_evidence(
                        "BEFORE_CANDIDATE_OPEN", marker="candidate"
                    )
                ),
            )
            with (
                mock.patch.object(runner, "SOURCE_INPUTS", specs),
                mock.patch.object(
                    runner, "resolve_offline_registered_recipe", return_value=registered
                ),
                mock.patch.object(runner, "verify_canonical_offline_run_scope"),
                mock.patch.object(runner, "_verify_clean_git_head"),
                mock.patch.object(runner, "_verify_runtime_environment"),
                mock.patch.object(runner, "_approval_module", return_value=approval),
                self.assertRaisesRegex(
                    ContractError, "not the exact deterministic projection"
                ),
            ):
                runner._execute_offline_registered_diagnostic_worker(
                    root=repo,
                    source_root=source,
                    run_scope=scope,
                    issue_number=7,
                    comment_id=11,
                    provider=object(),
                )
            self.assertFalse((repo / runner.FIXED_OUTPUT_ROOT).exists())

    def test_scope_compiler_binds_single_read_compared_buffers_not_later_swaps(self) -> None:
        registered = resolve_offline_registered_recipe(ROOT)
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            source = workspace / "source"
            repo = workspace / "repo"
            source.mkdir()
            repo.mkdir()
            specs, _ = self.build_sources(source)
            runner._materialize_core(
                repo_root=repo,
                source_root=source,
                policy=registered.policy,
                implementation_binding={
                    "implementation_commit": "a" * 40,
                    "runtime_material_bundle_sha256": canonical_digest(
                        dict(registered.runtime_material_digests)
                    ),
                },
                expected_source_inputs=specs,
                expected_race_count=runner.RACE_COUNT,
                expected_fold_counts=runner.FOLD_COUNTS,
            )
            candidate_path = repo / runner.PROJECTION_INPUTS["candidate_projection"]["path"]
            manifest_path = repo / runner.MATERIALIZATION_MANIFEST_PATH
            original_candidate_sha = _sha(candidate_path)
            original_manifest_sha = _sha(manifest_path)
            captured: dict[str, object] = {}
            original_recheck = runner._verify_deterministic_materialization_against_raw

            def recheck_then_swap(**kwargs: object):
                sealed = original_recheck(**kwargs)  # type: ignore[arg-type]
                candidate_path.write_bytes(candidate_path.read_bytes() + b" ")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["projection_bindings"]["candidate_projection"].update(
                    {
                        "sha256": _sha(candidate_path),
                        "byte_size": candidate_path.stat().st_size,
                    }
                )
                manifest.pop("manifest_digest")
                manifest["manifest_digest"] = canonical_digest(manifest)
                manifest_path.write_bytes(runner._canonical_bytes(manifest))
                return sealed

            def compile_scope(_registered: object, bindings: object):
                captured.update(bindings)  # type: ignore[arg-type]
                return {"run_scope_digest": "d" * 64}

            availability = {
                "verified_current_main_sha": "a" * 40,
                "implementation_commit": "a" * 40,
                "github_trust": {
                    "approvers_blob_sha": "b" * 40,
                    "approvers_content_sha256": "c" * 64,
                },
            }
            approval = SimpleNamespace(
                verify_offline_gate_availability=mock.Mock(return_value=availability)
            )
            with (
                mock.patch.object(runner, "SOURCE_INPUTS", specs),
                mock.patch.object(runner, "_approval_module", return_value=approval),
                mock.patch.object(runner, "_verify_clean_git_head"),
                mock.patch.object(
                    runner, "resolve_offline_registered_recipe", return_value=registered
                ),
                mock.patch.object(
                    runner,
                    "_verify_runtime_environment",
                    return_value=(
                        {"python_minor_version": "3.11", "numpy_version": "2.4.3"},
                        "9" * 64,
                    ),
                ),
                mock.patch.object(
                    runner,
                    "_verify_deterministic_materialization_against_raw",
                    side_effect=recheck_then_swap,
                ),
                mock.patch.object(
                    runner, "compile_offline_run_scope", side_effect=compile_scope
                ),
            ):
                runner._compile_fixed_run_scope_artifact_worker(
                    root=repo,
                    source_root=source,
                    provider=object(),
                    now="2026-08-15T00:00:00Z",
                )
            self.assertEqual(
                captured["projection_bindings"]["candidate_projection"]["sha256"],  # type: ignore[index]
                original_candidate_sha,
            )
            self.assertEqual(
                captured["materialization_manifest"]["sha256"],  # type: ignore[index]
                original_manifest_sha,
            )
            self.assertNotEqual(_sha(candidate_path), original_candidate_sha)
            self.assertNotEqual(_sha(manifest_path), original_manifest_sha)

    def test_scientific_projection_matches_strict_v1_pure_helpers(self) -> None:
        import registered_nonpromotion_supervised_executor_v1 as strict

        registered = resolve_offline_registered_recipe(ROOT)
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            source = workspace / "source"
            repo = workspace / "repo"
            source.mkdir()
            repo.mkdir()
            specs, _ = self.build_sources(source)
            self.materialize(repo, source, specs)
            candidate_bytes = (repo / runner.PROJECTION_INPUTS["candidate_projection"]["path"]).read_bytes()
            settlement_bytes = (repo / runner.PROJECTION_INPUTS["settlement_projection"]["path"]).read_bytes()
            candidates = runner._load_jsonl_exact_bytes(candidate_bytes)
            settlements = runner._load_jsonl_exact_bytes(settlement_bytes)
            decision_rows, decision_projection = runner._freeze_decisions(
                recipe=registered.recipe, candidate_rows=candidates
            )
            observed, _outcome = runner._scientific_projection(
                recipe=registered.recipe,
                decision_rows=decision_rows,
                decision_projection=decision_projection,
                settlement_rows=settlements,
                bootstrap_replicates=7,
            )
            strict_rows, strict_receipt = strict._freeze_decisions_after_authenticated_mount(
                registered.source_registered,
                candidates,
                run_scope_digest="a" * 64,
                replica_id="clean_a",
                irreversible_receipt_digest="b" * 64,
            )
            strict_result = strict._settle_diagnostic_after_authenticated_mount(
                registered.source_registered,
                strict_rows,
                settlements,
                run_scope_digest="a" * 64,
                replica_id="clean_a",
                decision_freeze_receipt=strict_receipt,
                settlement_operation_receipt_digest="c" * 64,
                bootstrap_replicates_override_for_synthetic_test=7,
            )
            self.assertEqual(observed, strict_result["scientific_projection"])
            self.assertEqual(canonical_digest(observed), strict_result["scientific_projection_digest"])

    def test_complete_synthetic_execution_orders_preaccess_raw_and_projection_workload(self) -> None:
        registered = resolve_offline_registered_recipe(ROOT)
        with tempfile.TemporaryDirectory() as raw:
            workspace = Path(raw)
            source = workspace / "source"
            repo = workspace / "repo"
            source.mkdir()
            repo.mkdir()
            specs, _ = self.build_sources(source)
            runner._materialize_core(
                repo_root=repo,
                source_root=source,
                policy=registered.policy,
                implementation_binding={
                    "implementation_commit": "a" * 40,
                    "runtime_material_bundle_sha256": canonical_digest(
                        dict(registered.runtime_material_digests)
                    ),
                },
                expected_source_inputs=specs,
                expected_race_count=runner.RACE_COUNT,
                expected_fold_counts=runner.FOLD_COUNTS,
            )
            manifest_path = repo / runner.MATERIALIZATION_MANIFEST_PATH
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest_binding = {
                "path": runner.MATERIALIZATION_MANIFEST_PATH,
                "sha256": _sha(manifest_path),
                "byte_size": manifest_path.stat().st_size,
            }
            scope = {
                "verified_current_main_sha": "a" * 40,
                "run_scope_base_commit": "a" * 40,
                "run_scope_digest": "d" * 64,
                "semantic_subject_digest": "e" * 64,
                "exact_subject_digest": "f" * 64,
                "output_root": runner.FIXED_OUTPUT_ROOT,
                "runtime_bindings": {
                    "environment_manifest_sha256": "3" * 64,
                    "materialization_manifest": manifest_binding,
                    "source_bindings": manifest["source_bindings"],
                    "projection_bindings": manifest["projection_bindings"],
                },
            }
            checkpoints: list[str] = []
            events: list[str] = []
            initial_evidence = _approval_evidence(
                "INITIAL_APPROVAL", marker="initial"
            )

            def initial_approval(**_kwargs: object) -> dict[str, object]:
                events.append("INITIAL_APPROVAL")
                return dict(initial_evidence)

            def reverify(**kwargs: object) -> dict[str, object]:
                checkpoint = str(kwargs["checkpoint"])
                checkpoints.append(checkpoint)
                events.append(checkpoint)
                return _approval_evidence(
                    checkpoint,
                    marker="reverified",
                    original_evidence_digest=str(initial_evidence["evidence_digest"]),
                )

            approval = SimpleNamespace(
                verify_offline_run_approval=mock.Mock(
                    side_effect=initial_approval
                ),
                reverify_offline_run_approval=mock.Mock(side_effect=reverify),
            )
            raw_roles: list[str] = []
            original_raw_reader = runner._read_verified_source_bytes

            def observe_preaccess_raw(
                path: Path, *, expected_sha256: str, role: str
            ):
                self.assertFalse((repo / runner.FIXED_OUTPUT_ROOT).exists())
                raw_roles.append(role)
                events.append(f"raw:{role}")
                return original_raw_reader(
                    path, expected_sha256=expected_sha256, role=role
                )
            bootstrap_calls: list[tuple[int, int]] = []
            original_write = runner._write_json_exclusive
            original_bound_read = runner._read_bound_bytes
            original_publish = runner._atomic_publish_json
            bound_read_counts: dict[str, int] = {}

            def observe_write(path: Path, value: object) -> None:
                if path.name in {
                    "approval_evidence_initial.json",
                    "approval_evidence_before_candidate.json",
                    "start_receipt.json",
                    "decision_freeze_receipt.json",
                    "approval_evidence_before_result.json",
                    "result_seal_receipt.json",
                }:
                    events.append(f"write:{path.name}")
                original_write(path, value)  # type: ignore[arg-type]

            def observe_bound_read(root: Path, binding: object, *, label: str):
                if (
                    Path(root).resolve() == repo.resolve()
                    and label in {"candidate projection", "settlement projection"}
                ):
                    count = bound_read_counts.get(label, 0) + 1
                    bound_read_counts[label] = count
                    events.append(f"read:{label}:{count}")
                return original_bound_read(root, binding, label=label)  # type: ignore[arg-type]

            def observe_publish(path: Path, value: object) -> None:
                events.append("publish:result.json")
                original_publish(path, value)  # type: ignore[arg-type]

            def fast_bootstrap(
                rows: object, *, replicates: int, seed: int
            ) -> dict[str, object]:
                bootstrap_calls.append((replicates, seed))
                return {
                    "cluster_count": 1,
                    "replicates": replicates,
                    "seed": seed,
                    "rng": "numpy.random.Generator(PCG64)",
                    "mean": 0.0,
                    "one_sided_95_lower_bound": 0.0,
                    "distribution_digest": "7" * 64,
                }

            with (
                mock.patch.object(
                    runner, "resolve_offline_registered_recipe", return_value=registered
                ),
                mock.patch.object(runner, "verify_canonical_offline_run_scope"),
                mock.patch.object(runner, "_verify_clean_git_head"),
                mock.patch.object(runner, "_verify_runtime_environment"),
                mock.patch.object(runner, "_approval_module", return_value=approval),
                mock.patch.object(
                    runner,
                    "_read_verified_source_bytes",
                    side_effect=observe_preaccess_raw,
                ),
                mock.patch.object(runner, "SOURCE_INPUTS", specs),
                mock.patch.object(
                    runner, "_cluster_bootstrap", side_effect=fast_bootstrap
                ),
                mock.patch.object(
                    runner, "_write_json_exclusive", side_effect=observe_write
                ),
                mock.patch.object(
                    runner, "_read_bound_bytes", side_effect=observe_bound_read
                ),
                mock.patch.object(
                    runner, "_atomic_publish_json", side_effect=observe_publish
                ),
            ):
                result = runner._execute_offline_registered_diagnostic_worker(
                    root=repo,
                    source_root=source,
                    run_scope=scope,
                    issue_number=7,
                    comment_id=11,
                    provider=object(),
                )
            self.assertEqual(
                checkpoints,
                ["BEFORE_CANDIDATE_OPEN", "BEFORE_RESULT_PUBLISH"],
            )
            self.assertEqual(
                events,
                [
                    "INITIAL_APPROVAL",
                    "raw:diagnostic_master",
                    "raw:p_action_artifact",
                    "raw:official_payoff_source",
                    "BEFORE_CANDIDATE_OPEN",
                    "write:approval_evidence_initial.json",
                    "write:approval_evidence_before_candidate.json",
                    "write:start_receipt.json",
                    "read:candidate projection:1",
                    "write:decision_freeze_receipt.json",
                    "read:settlement projection:1",
                    "read:candidate projection:2",
                    "read:settlement projection:2",
                    "BEFORE_RESULT_PUBLISH",
                    "write:approval_evidence_before_result.json",
                    "write:result_seal_receipt.json",
                    "publish:result.json",
                ],
            )
            self.assertEqual(
                raw_roles,
                ["diagnostic_master", "p_action_artifact", "official_payoff_source"],
            )
            self.assertEqual(result["lifecycle_state"], "RNOD_COMPLETED")
            result_schema = json.loads(
                (
                    ROOT
                    / "research/schemas/registered_nonpromotion_offline_result_v1.schema.json"
                ).read_text(encoding="utf-8")
            )
            _validate(result, result_schema, result_schema)
            self.assertEqual(
                bootstrap_calls,
                [(100000, runner.BOOTSTRAP_SEED), (100000, runner.BOOTSTRAP_SEED)],
            )
            self.assertEqual(
                result["scientific_projection"]["bootstrap"]["replicates"], 100000
            )
            self.assertTrue(result["replica_semantic_equality"])
            self.assertEqual(
                len(set(result["replica_scientific_projection_digests"].values())), 1
            )
            output = repo / runner.FIXED_OUTPUT_ROOT
            self.assertTrue((output / "approval_evidence_initial.json").is_file())
            self.assertTrue((output / "start_receipt.json").is_file())
            self.assertTrue((output / "decision_freeze_receipt.json").is_file())
            self.assertTrue((output / "result_seal_receipt.json").is_file())
            self.assertTrue((output / "result.json").is_file())
            self.assertFalse((output / "INVALID.json").exists())
            result_seal = json.loads(
                (output / "result_seal_receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result_seal["lifecycle_state"], "RNOD_RESULT_SEALED")
            self.assertEqual(
                result_seal["completed_result_digest"], result["result_digest"]
            )
            with self.assertRaises(runner.OfflineRunAlreadyCompleted):
                runner._raise_poststart_terminal_after_exception(output, scope)
            # The same recovery path is used when an asynchronous exception
            # arrives immediately after the atomic result replace.  A sealed
            # completion must remain uniquely COMPLETED, never COMPLETED plus
            # an INVALID tombstone.
            self.assertFalse((output / "INVALID.json").exists())
            candidate_evidence_path = (
                output / "approval_evidence_before_candidate.json"
            )
            candidate_evidence_path.write_bytes(
                candidate_evidence_path.read_bytes() + b" "
            )
            with self.assertRaises(runner.OfflineInvalidAfterStart):
                runner._recover_or_reject_existing_output(output, scope)
            self.assertTrue((output / "INVALID.json").is_file())


if __name__ == "__main__":
    unittest.main()
