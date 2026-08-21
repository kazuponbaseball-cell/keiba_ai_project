from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import pytest
except ModuleNotFoundError:
    # The repository's existing minimal Research OS CI intentionally installs
    # only numpy.  Stdlib static contract tests below still run there; the full
    # synthetic suite runs in the hash-bound Prepare environment.
    np = None
    pd = None
    pytest = None


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts/research/run_leakfree_predraw_baseline_v0.py"
CONFIG_PATH = ROOT / "research/configs/EXP-20260821-033.leakfree_predraw_baseline_v0.json"
SPEC = importlib.util.spec_from_file_location("exp033_leakfree_runner", RUNNER_PATH)
RUNTIME_DEPS_AVAILABLE = np is not None and pd is not None and pytest is not None
if RUNTIME_DEPS_AVAILABLE:
    assert SPEC is not None and SPEC.loader is not None
    runner = importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name] = runner
    SPEC.loader.exec_module(runner)
else:
    runner = None


UTC = timezone.utc


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


if RUNTIME_DEPS_AVAILABLE:
    @pytest.fixture(scope="module")
    def bundle():
        return runner.load_and_verify_contract(CONFIG_PATH)


class StaticPrepareContractTests(unittest.TestCase):
    """Runs in the repository's existing numpy-only Python 3.11/3.12 CI."""

    def test_frozen_feature_and_proposal_contracts(self):
        allow_path = ROOT / "research/drafts/EXP-20260821-033.feature_allowlist.json"
        deny_path = ROOT / "research/drafts/EXP-20260821-033.feature_denylist.json"
        fold_path = ROOT / "research/drafts/EXP-20260821-033.fold_manifest.json"
        proposal_path = ROOT / "research/scopes/EXP-20260821-033.proposal.json"
        expected_hashes = {
            allow_path: "9d1108ea13d39b74535c5bdd4cdb87afb8f21be37686cfe645332949359db7f2",
            deny_path: "84387f57e1696263f72121c6e6eebe544281d8cbc7a82568512d7cfddfc3e8e9",
            fold_path: "622b01ea20241858d7692d7cf169c4c46082d53620066e6ad7be5e517ea592fb",
        }
        for path, expected in expected_hashes.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
        allow = json.loads(allow_path.read_text(encoding="utf-8"))
        deny = json.loads(deny_path.read_text(encoding="utf-8"))
        self.assertEqual((len(allow["numeric_features"]), len(allow["categorical_features"])), (77, 11))
        self.assertEqual((len(deny["numeric_features"]), len(deny["categorical_features"])), (279, 7))
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
        canonical = json.dumps(
            proposal, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        self.assertEqual(
            hashlib.sha256(canonical).hexdigest(),
            "6993e9f5a6e0d6b2ef726bbc65fd047479a7b1bf79e948689a588f26f034ff6d",
        )

    def test_runner_has_only_the_approved_model_import(self):
        tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        self.assertEqual([name for name in imported if name.startswith("src.")], ["src.train.simple_ranker"])

    def test_config_safety_and_environment_are_frozen(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        environment = json.loads(
            (ROOT / "research/drafts/EXP-20260821-033.environment_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(config["variant"], "leakfree_predraw_baseline_v0")
        self.assertEqual(config["model"]["seed"], 20260823)
        self.assertFalse(config["formal_buy"])
        self.assertFalse(config["send_order"])
        self.assertEqual(config["stake"], 0)
        self.assertEqual(environment["supported_python"], ["3.11", "3.12"])
        self.assertEqual(
            {item["name"]: item["version"] for item in environment["dependencies"]},
            {"numpy": "2.2.6", "pandas": "2.3.2", "pytest": "8.4.1"},
        )
        self.assertIn("tzdata==2026.3", environment["prepare_environment_lock"]["packages"])
        self.assertIn(
            "require hash-bound dependency lock",
            environment["prepare_environment_lock"]["wheel_hashes_status"],
        )


def _card_row(race_id: str, horse_id: str, prediction: datetime, ordinal: int, **updates):
    row = {
        "record_kind": "declared_card",
        "race_id": race_id,
        "horse_id": horse_id,
        "prediction_event_time": _iso(prediction),
        "source_event_time": _iso(prediction - timedelta(days=2)),
        "received_at": _iso(prediction - timedelta(hours=36)),
        "available_as_of": _iso(prediction - timedelta(hours=24)),
        "source_version": "synthetic-card-v1",
        "source_content_sha256": "0" * 64,
        "missing_reason": "not_missing",
        "年齢": 3 + ordinal % 4,
        "斤量": 54.0 + ordinal % 4,
        "距離": 1200 + 200 * (ordinal % 5),
        "場所": ["東京", "中山", "京都", "阪神", "新潟"][ordinal % 5],
        "性別": ["牡", "牝", "セ"][ordinal % 3],
        "騎手コード": f"J{ordinal % 7:03d}",
        "調教師コード": f"T{ordinal % 5:03d}",
        "芝・ダ": "芝" if ordinal % 2 == 0 else "ダ",
        "クラス名": ["未勝利", "1勝", "2勝", "3勝", "G3"][ordinal % 5],
        "トラックコード": f"TR{ordinal % 6:02d}",
        "draw_status": "scheduled_pending_draw",
        "entry_stage": "declared_without_draw",
        "枠番": pd.NA,
        "馬番": pd.NA,
    }
    row.update(updates)
    return row


def _cards(schedule):
    rows = []
    ordinal = 0
    for race_ordinal, (race_id, prediction, horses) in enumerate(schedule):
        race_values = {
            "距離": 1200 + 200 * (race_ordinal % 5),
            "場所": ["東京", "中山", "京都", "阪神", "新潟"][race_ordinal % 5],
            "芝・ダ": "芝" if race_ordinal % 2 == 0 else "ダ",
            "クラス名": ["未勝利", "1勝", "2勝", "3勝", "G3"][race_ordinal % 5],
            "トラックコード": f"TR{race_ordinal % 6:02d}",
        }
        for horse_id in horses:
            rows.append(_card_row(race_id, horse_id, prediction, ordinal, **race_values))
            ordinal += 1
    return runner.bind_source_content_hashes(pd.DataFrame(rows))


def _results(cards: pd.DataFrame, race_ids=None, corner_overrides=None) -> pd.DataFrame:
    selected = cards if race_ids is None else cards[cards["race_id"].isin(set(race_ids))]
    overrides = corner_overrides or {}
    rows = []
    for race_id, group in selected.groupby("race_id", sort=False):
        ordered = group.sort_values("horse_id", kind="mergesort").reset_index(drop=True)
        for ordinal, card in ordered.iterrows():
            prediction = datetime.fromisoformat(str(card["prediction_event_time"]).replace("Z", "+00:00"))
            values = {
                "1角": min(ordinal + 2, len(ordered)),
                "2角": min(ordinal + 2, len(ordered)),
                "4角": ordinal + 1,
            }
            values.update(overrides.get((str(race_id), str(card["horse_id"])), {}))
            missing_reason = (
                "source_not_recorded"
                if any(pd.isna(values[column]) for column in ["1角", "2角", "4角"])
                else "not_missing"
            )
            rows.append(
                {
                    "record_kind": "completed_result",
                    "race_id": str(race_id),
                    "horse_id": str(card["horse_id"]),
                    "prediction_event_time": card["prediction_event_time"],
                    "source_event_time": _iso(prediction + timedelta(minutes=2)),
                    "received_at": _iso(prediction + timedelta(minutes=3)),
                    "available_as_of": _iso(prediction + timedelta(minutes=4)),
                    "source_version": "synthetic-result-v1",
                    "source_content_sha256": "0" * 64,
                    "missing_reason": missing_reason,
                    "確定着順": ordinal + 1,
                    **values,
                }
            )
    return runner.bind_source_content_hashes(pd.DataFrame(rows))


def _three_race_fixture():
    horses = [f"SYN-H{i:02d}" for i in range(1, 7)]
    base = datetime(2020, 1, 5, 6, 0, tzinfo=UTC)
    schedule = [
        ("SYN-R1", base, horses),
        ("SYN-R2", base + timedelta(days=14), horses),
        ("SYN-R3", base + timedelta(days=28), horses),
    ]
    cards = _cards(schedule)
    results = _results(cards)
    return cards, results


def _fit_on_first_two_races(materialized, cards, results, bundle):
    targets = runner.targets_from_results(cards, results, bundle)
    target_map = targets.set_index(["race_id", "horse_id"])["target_score"].to_dict()
    train_mask = materialized.features["race_id"].isin(["SYN-R1", "SYN-R2"])
    train_features = materialized.features.loc[train_mask].reset_index(drop=True)
    y = [target_map[(str(row.race_id), str(row.horse_id))] for row in train_features.itertuples()]
    card_times = cards.set_index(["race_id", "horse_id"])["prediction_event_time"].to_dict()
    event_times = [card_times[(str(row.race_id), str(row.horse_id))] for row in train_features.itertuples()]
    budget = runner.FitBudget()
    model, schema = runner.fit_clean_ranker(
        train_features,
        y,
        event_times,
        bundle,
        execution_kind="synthetic_fixture",
        budget=budget,
    )
    return model, schema, budget


def test_static_contract_is_exact_88_default_deny_and_leakage_closed(bundle):
    assert len(bundle.numeric_features) == 77
    assert len(bundle.categorical_features) == 11
    assert len(bundle.ordered_features) == 88
    assert set(runner.DIRECT_LEAKAGE_ROOTS).isdisjoint(bundle.ordered_features)
    assert set(runner.DIRECT_LEAKAGE_ROOTS).issubset(bundle.denylist["numeric_features"])
    assert bundle.leakage_manifest["prefit_feature_to_feature_descendants"] == []
    descendants = set(bundle.leakage_manifest["contaminated_model_and_output_descendants"])
    assert {"ai_score", "ai_rank", "slow_ai_score", "middle_ai_score", "fast_ai_score"}.issubset(descendants)
    assert bundle.config["formal_buy"] is False
    assert bundle.config["send_order"] is False
    assert bundle.config["stake"] == 0


def test_current_result_draw_and_market_mutation_cannot_change_features_or_scores(bundle):
    cards, results = _three_race_fixture()
    base = runner.materialize_predraw_features(cards, results, bundle)
    model, schema, _ = _fit_on_first_two_races(base, cards, results, bundle)
    base_prediction = runner.predict_clean_ranker(model, base.features, schema, bundle)

    mutated_cards = cards.copy(deep=True)
    for column, first, second in [
        ("確定着順", 1, 99),
        ("target_score", 1.0, -9.0),
        ("target_win", 1, 0),
        ("target_top3", 1, 0),
        ("枠番", 1, 8),
        ("馬番", 1, 18),
        ("odds", 1.1, 999.0),
        ("人気", 1, 18),
        ("market", "A", "B"),
        ("payout", 100, 999999),
        ("ROI", 0.1, 9.9),
        ("BUY", False, True),
        ("stake", 0, 100000),
    ]:
        mutated_cards[column] = [first if i % 2 == 0 else second for i in range(len(mutated_cards))]
    mutated_cards = runner.bind_source_content_hashes(mutated_cards)
    mutated_results = results.copy(deep=True)
    target = mutated_results["race_id"] == "SYN-R3"
    mutated_results.loc[target, "確定着順"] = list(reversed(range(1, int(target.sum()) + 1)))
    mutated_results = runner.bind_source_content_hashes(mutated_results)
    changed = runner.materialize_predraw_features(mutated_cards, mutated_results, bundle)
    pd.testing.assert_frame_equal(base.features, changed.features)
    changed_prediction = runner.predict_clean_ranker(model, changed.features, schema, bundle)
    pd.testing.assert_frame_equal(base_prediction, changed_prediction)
    assert {"枠番", "馬番", "odds", "人気", "ROI", "BUY", "stake"}.issubset(
        set(changed.qa["raw_forbidden_columns_ignored"])
    )


def test_future_append_is_invariant_for_all_prior_rows(bundle):
    cards, results = _three_race_fixture()
    base = runner.materialize_predraw_features(cards, results, bundle)
    horses = sorted(cards["horse_id"].unique())
    future_time = datetime(2020, 3, 1, 6, 0, tzinfo=UTC)
    future_cards = _cards([("SYN-R4", future_time, horses)])
    future_results = _results(future_cards)
    expanded = runner.materialize_predraw_features(
        pd.concat([cards, future_cards], ignore_index=True),
        pd.concat([results, future_results], ignore_index=True),
        bundle,
    )
    earlier = expanded.features[expanded.features["race_id"].isin(set(cards["race_id"]))].reset_index(drop=True)
    pd.testing.assert_frame_equal(base.features.reset_index(drop=True), earlier)


def test_same_event_time_batches_cannot_leak_and_lineage_is_complete(bundle):
    when = datetime(2021, 1, 1, 5, 0, tzinfo=UTC)
    a = [f"A{i}" for i in range(4)]
    b = [f"B{i}" for i in range(4)]
    cards = _cards([("SAME-A", when, a), ("SAME-B", when, b)])
    results = _results(cards)
    materialized = runner.materialize_predraw_features(cards, results, bundle)
    debut_columns = [
        "past3_avg_score",
        "same_distance_category_starts",
        "same_venue_starts",
        "horse_turf_starts",
        "horse_dirt_starts",
    ]
    assert (materialized.features[debut_columns] == 0.0).all().all()
    assert materialized.qa["results_applied"] == 0
    assert len(materialized.lineage) == len(cards) * 88
    assert materialized.lineage["as_of_safe"].all()
    available = pd.to_datetime(materialized.lineage["available_as_of"], utc=True)
    prediction = pd.to_datetime(materialized.lineage["prediction_event_time"], utc=True)
    assert (available < prediction).all()


def test_completed_result_race_is_applied_atomically_not_runner_by_runner(bundle):
    horses = [f"ATOMIC-{i}" for i in range(4)]
    first = datetime(2021, 1, 1, 5, 0, tzinfo=UTC)
    second = first + timedelta(days=14)
    cards = _cards([("ATOMIC-1", first, horses), ("ATOMIC-2", second, horses)])
    results = _results(cards, race_ids=["ATOMIC-1"])
    delayed = results["horse_id"] == horses[-1]
    results.loc[delayed, "available_as_of"] = _iso(first + timedelta(days=20))
    results = runner.bind_source_content_hashes(results)
    materialized = runner.materialize_predraw_features(cards, results, bundle)
    later = materialized.features[materialized.features["race_id"] == "ATOMIC-2"]
    assert (later["past3_avg_score"] == 0.0).all()
    assert (later["same_distance_category_starts"] == 0.0).all()
    assert materialized.qa["results_applied"] == 0


def test_race_aggregate_lineage_includes_every_runner_dependency(bundle):
    horses = [f"LIN-{i}" for i in range(4)]
    when = datetime(2021, 1, 1, 5, 0, tzinfo=UTC)
    cards = _cards([("LINEAGE", when, horses)])
    base = runner.materialize_predraw_features(cards, pd.DataFrame(), bundle)
    focal = horses[0]
    base_weight = base.features.loc[base.features["horse_id"] == focal, "race_weight_light_rank_score"].iloc[0]
    base_lineage = base.lineage[
        (base.lineage["horse_id"] == focal)
        & (base.lineage["feature_name"] == "race_weight_light_rank_score")
    ].iloc[0]
    direct_lineage = base.lineage[
        (base.lineage["horse_id"] == focal) & (base.lineage["feature_name"] == "年齢")
    ].iloc[0]
    assert base_lineage["dependency_count"] == 4
    assert direct_lineage["dependency_count"] == 1

    changed_cards = cards.copy()
    changed_cards.loc[changed_cards["horse_id"] == horses[-1], "斤量"] = 40.0
    changed_cards = runner.bind_source_content_hashes(changed_cards)
    changed = runner.materialize_predraw_features(changed_cards, pd.DataFrame(), bundle)
    changed_weight = changed.features.loc[
        changed.features["horse_id"] == focal, "race_weight_light_rank_score"
    ].iloc[0]
    changed_hash = changed.lineage.loc[
        (changed.lineage["horse_id"] == focal)
        & (changed.lineage["feature_name"] == "race_weight_light_rank_score"),
        "content_hash",
    ].iloc[0]
    assert changed_weight != base_weight
    assert changed_hash != base_lineage["content_hash"]


def test_default_deny_missing_core_and_nonfinite_are_hard_failures(bundle):
    cards, results = _three_race_fixture()
    materialized = runner.materialize_predraw_features(cards, results, bundle)
    unknown = materialized.features.copy()
    unknown["unknown_generated_feature"] = 1.0
    with pytest.raises(runner.ContractError):
        runner.validate_feature_frame(unknown, bundle)
    with pytest.raises(runner.ContractError):
        runner.materialize_predraw_features(cards.drop(columns=["年齢"]), results, bundle)
    nonfinite = materialized.features.copy()
    nonfinite.loc[0, "斤量"] = np.inf
    with pytest.raises(runner.ContractError):
        runner.validate_feature_frame(nonfinite, bundle)

    generated_blank_identity = materialized.features.copy()
    generated_blank_identity.loc[0, "horse_id"] = "__MISSING__"
    with pytest.raises(runner.ContractError, match="feature frame identity is blank"):
        runner.validate_feature_frame(generated_blank_identity, bundle)

    payload_tamper = cards.copy()
    payload_tamper.loc[0, "年齢"] = 9
    with pytest.raises(runner.ContractError, match="source payload hash mismatch"):
        runner.materialize_predraw_features(payload_tamper, results, bundle)

    blank_identity = cards.copy()
    blank_identity.loc[0, "horse_id"] = ""
    blank_identity = runner.bind_source_content_hashes(blank_identity)
    with pytest.raises(runner.ContractError, match="horse_id is blank"):
        runner.materialize_predraw_features(blank_identity, results, bundle)


def test_debut_and_missing_previous_corner_semantics(bundle):
    horses = [f"D{i}" for i in range(4)]
    t0 = datetime(2021, 2, 1, 4, 0, tzinfo=UTC)
    t1 = t0 + timedelta(days=14)
    cards = _cards([("DEBUT", t0, horses), ("NEXT", t1, horses)])
    overrides = {("DEBUT", horse): {"1角": pd.NA, "2角": pd.NA, "4角": pd.NA} for horse in horses}
    results = _results(cards, race_ids=["DEBUT"], corner_overrides=overrides)
    materialized = runner.materialize_predraw_features(cards, results, bundle)
    debut = materialized.features[materialized.features["race_id"] == "DEBUT"]
    next_race = materialized.features[materialized.features["race_id"] == "NEXT"]
    assert (debut["past3_avg_score"] == 0.0).all()
    assert (debut["prev_corner4_position_rate"] == 0.5).all()
    assert (debut["distance_diff"] == 0.0).all()
    assert (debut["class_changed"] == 0.0).all()
    assert (debut["jockey_changed"] == 0.0).all()
    assert (debut["previous_distance_category"] == "__MISSING__").all()
    assert (next_race["prev_corner4_position_rate"] == 0.5).all()
    assert materialized.qa["debut_rows"] == 4
    assert materialized.qa["missing_previous_corner4_rows"] == 8


def test_early_move_fallback_is_rowwise_1c_then_2c_then_4c(bundle):
    horses = [f"E{i}" for i in range(6)]
    base = datetime(2021, 3, 1, 4, 0, tzinfo=UTC)
    schedule = [(f"EARLY-{i}", base + timedelta(days=14 * i), horses) for i in range(4)]
    cards = _cards(schedule)
    focal = horses[0]
    overrides = {
        ("EARLY-0", focal): {"1角": 5, "2角": 4, "4角": 3},
        ("EARLY-1", focal): {"1角": pd.NA, "2角": 6, "4角": 4},
        ("EARLY-2", focal): {"1角": pd.NA, "2角": pd.NA, "4角": 2},
    }
    results = _results(cards, race_ids=["EARLY-0", "EARLY-1", "EARLY-2"], corner_overrides=overrides)
    materialized = runner.materialize_predraw_features(cards, results, bundle)
    target = materialized.features[
        (materialized.features["race_id"] == "EARLY-3") & (materialized.features["horse_id"] == focal)
    ].iloc[0]
    assert target["horse_early_move_avg_past5"] == pytest.approx((2.0 + 2.0 + 0.0) / 3.0)
    assert target["prev_early_move"] == pytest.approx(0.0)


def test_train_live_schema_category_dictionary_and_unknown_policy_match(bundle):
    cards, results = _three_race_fixture()
    cards.loc[cards["race_id"] == "SYN-R3", "調教師コード"] = "UNSEEN-TRAINER"
    cards = runner.bind_source_content_hashes(cards)
    materialized = runner.materialize_predraw_features(cards, results, bundle)
    model, schema, budget = _fit_on_first_two_races(materialized, cards, results, bundle)
    levels_before = copy.deepcopy(model.categorical_levels_)
    parity = runner.validate_schema_parity(materialized.features, model, schema, bundle)
    prediction = runner.predict_clean_ranker(model, materialized.features, schema, bundle)
    assert parity["ordered_feature_parity_fraction"] == 1.0
    assert parity["dtype_parity_fraction"] == 1.0
    assert parity["category_dictionary_parity_fraction"] == 1.0
    assert parity["unknown_category_policy"] == "all_zero_reference_encoding"
    assert model.categorical_levels_ == levels_before
    assert np.isfinite(prediction["clean_ai_score"]).all()
    assert budget.model_fits == 1
    with pytest.raises(runner.ContractError):
        budget.consume_model_fit()
    changed_model = copy.deepcopy(model)
    changed_model.numeric_means_[bundle.numeric_features[0]] += 1.0
    with pytest.raises(runner.ContractError, match="numeric mean contract changed"):
        runner.validate_schema_parity(materialized.features, changed_model, schema, bundle)


def test_current_surface_only_history_matches_audited_base_semantics(bundle):
    cards, results = _three_race_fixture()
    materialized = runner.materialize_predraw_features(cards, results, bundle)
    target = materialized.features[materialized.features["race_id"] == "SYN-R3"]
    assert (target["芝・ダ"] == "芝").all()
    assert (target["horse_turf_starts"] == 1.0).all()
    assert (target[["horse_dirt_starts", "horse_dirt_win_rate", "horse_dirt_top3_rate", "horse_dirt_avg_score"]] == 0.0).all().all()


def test_chronology_overlap_outer_seal_and_asof_fail_closed(bundle):
    bad_fold = copy.deepcopy(bundle.fold_manifest)
    bad_fold["ordered_blocks"][2]["start"] = "2023-01-01"
    with pytest.raises(runner.ContractError):
        runner.validate_fold_contract(bad_fold)
    with pytest.raises(runner.ContractError):
        runner.validate_partition_race_disjointness({"train": ["R1"], "outer": ["R1"]})

    called = {"count": 0}
    seal = runner.OuterSeal()
    with pytest.raises(runner.ContractError):
        seal.open(lambda: called.update(count=1))
    frozen = _hash("frozen")
    seal.freeze(model=frozen, config=frozen, calibrator=frozen, schema=frozen)
    assert seal.open(lambda: "outer") == "outer"
    with pytest.raises(runner.ContractError):
        seal.open(lambda: "again")

    cards, _ = _three_race_fixture()
    bad_cards = cards.copy()
    bad_cards.loc[0, "received_at"] = "2020-01-01T00:00:00"
    bad_cards = runner.bind_source_content_hashes(bad_cards)
    with pytest.raises(runner.ContractError):
        runner.materialize_predraw_features(bad_cards, pd.DataFrame(), bundle)

    missing_version = cards.copy()
    missing_version.loc[0, "source_version"] = pd.NA
    missing_version = runner.bind_source_content_hashes(missing_version)
    with pytest.raises(runner.ContractError, match="source_version is empty"):
        runner.materialize_predraw_features(missing_version, pd.DataFrame(), bundle)
    assert called["count"] == 0


def test_synthetic_predraw_target_materializes_5_races_70_runners_without_draw(bundle):
    when = datetime(2026, 8, 23, 3, 0, tzinfo=UTC)
    schedule = []
    for race in range(5):
        horses = [f"P{race}-{horse:02d}" for horse in range(14)]
        schedule.append((f"PREDRAW-{race}", when + timedelta(minutes=40 * race), horses))
    cards = _cards(schedule)
    quality = runner.validate_target_universe(cards)
    materialized = runner.materialize_predraw_features(cards, pd.DataFrame(), bundle)
    assert quality == {
        "race_count": 5,
        "runner_count": 70,
        "duplicate_count": 0,
        "scheduled_pending_draw_count": 70,
        "confirmed_count": 0,
        "scratched_count": 0,
        "horse_id_join_only": True,
    }
    assert len(materialized.features) == 70
    assert len(materialized.lineage) == 70 * 88
    assert materialized.qa["feature_count"] == 88
    assert materialized.qa["lineage_pass_fraction"] == 1.0
    assert cards["枠番"].isna().all() and cards["馬番"].isna().all()
    scratched = cards.copy()
    scratched.loc[0, "draw_status"] = "scratched"
    scratched = runner.bind_source_content_hashes(scratched)
    scratched_quality = runner.validate_target_universe(scratched)
    assert scratched_quality["scratched_count"] == 1
    with pytest.raises(runner.ContractError, match="scratched runners must be removed"):
        runner.materialize_predraw_features(scratched, pd.DataFrame(), bundle)

    invalid_confirmed = cards.copy()
    invalid_confirmed.loc[0, "draw_status"] = "confirmed"
    with pytest.raises(runner.ContractError, match="confirmed draw has invalid"):
        runner.validate_target_universe(invalid_confirmed)
    valid_confirmed = invalid_confirmed.copy()
    valid_confirmed.loc[0, ["枠番", "馬番"]] = [1, 1]
    confirmed_quality = runner.validate_target_universe(valid_confirmed)
    assert confirmed_quality["confirmed_count"] == 1


def test_clean_ranking_uses_score_desc_then_horse_id_and_exact_ranks(bundle):
    cards, results = _three_race_fixture()
    materialized = runner.materialize_predraw_features(cards, results, bundle)
    train = materialized.features[materialized.features["race_id"] == "SYN-R1"].reset_index(drop=True)
    constant_target = np.full(len(train), 0.5)
    event_times = cards[cards["race_id"] == "SYN-R1"]["prediction_event_time"].tolist()
    model, schema = runner.fit_clean_ranker(
        train,
        constant_target,
        event_times,
        bundle,
        execution_kind="synthetic_fixture",
        budget=runner.FitBudget(),
    )
    prediction = runner.predict_clean_ranker(model, train, schema, bundle)
    ordered = prediction.sort_values("clean_baseline_rank")
    assert ordered["horse_id"].tolist() == sorted(ordered["horse_id"].tolist())
    assert ordered["clean_baseline_rank"].tolist() == list(range(1, len(ordered) + 1))
    assert "horse_id" not in model.numeric_features + model.categorical_features

    # DataFrame index labels are not identities and cannot alter the fit order.
    duplicate_index = train.copy()
    duplicate_index.index = [0] * len(duplicate_index)
    duplicate_model, duplicate_schema = runner.fit_clean_ranker(
        duplicate_index,
        constant_target,
        event_times,
        bundle,
        execution_kind="synthetic_fixture",
        budget=runner.FitBudget(),
    )
    duplicate_prediction = runner.predict_clean_ranker(
        duplicate_model, duplicate_index, duplicate_schema, bundle
    )
    assert duplicate_prediction.sort_values("clean_baseline_rank")["horse_id"].tolist() == sorted(
        train["horse_id"].tolist()
    )


def test_caller_supplied_boolean_cannot_authorize_real_data_fit(bundle):
    cards, results = _three_race_fixture()
    materialized = runner.materialize_predraw_features(cards, results, bundle)
    train = materialized.features[materialized.features["race_id"] == "SYN-R1"].reset_index(drop=True)
    target = np.full(len(train), 0.5)
    event_times = cards[cards["race_id"] == "SYN-R1"]["prediction_event_time"].tolist()
    budget = runner.FitBudget()
    with pytest.raises(runner.ContractError, match="caller-supplied flag"):
        runner.fit_clean_ranker(
            train,
            target,
            event_times,
            bundle,
            execution_kind="real-data",
            real_data_authorized=True,
            budget=budget,
        )
    assert budget.model_fits == 0

    relabeled_real_identity = train.copy()
    relabeled_real_identity.loc[0, "horse_id"] = "2024103226"
    with pytest.raises(runner.ContractError, match="reserved synthetic identities"):
        runner.fit_clean_ranker(
            relabeled_real_identity,
            target,
            event_times,
            bundle,
            execution_kind="synthetic_fixture",
            budget=runner.FitBudget(),
        )


def test_production_imports_outputs_and_top3_probability_are_absent(bundle):
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    repo_imports = [name for name in imported if name.startswith("src.")]
    assert repo_imports == ["src.train.simple_ranker"]
    assert bundle.config["clean_outputs"]["score_column"] == "clean_ai_score"
    assert bundle.config["clean_outputs"]["rank_column"] == "clean_baseline_rank"
    assert bundle.config["clean_outputs"]["top3_probability_output"] is False
    assert bundle.config["production_change_allowed"] is False
    assert bundle.config["champion_change_allowed"] is False
    assert bundle.config["notification_allowed"] is False
    assert bundle.config["order_allowed"] is False


def test_single_variant_seed_family_and_hyperparameters_cannot_drift(bundle, tmp_path):
    assert bundle.config["variant"] == "leakfree_predraw_baseline_v0"
    assert bundle.config["model"]["seed"] == 20260823
    assert bundle.config["model"]["class"] == "SimpleRaceRanker"
    assert bundle.config["model"]["ridge_alpha"] == 10.0
    assert bundle.config["model"]["categorical_top_k"] == 80
    changed = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    changed["variant"] = "second_variant"
    changed_path = tmp_path / "changed.json"
    changed_path.write_text(json.dumps(changed, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(runner.ContractError):
        runner.load_and_verify_contract(changed_path)

    changed_runtime = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    changed_runtime["runtime_authorization"]["synthetic_fixture_execution_kind"] = "real-data"
    changed_runtime_path = tmp_path / "changed-runtime.json"
    changed_runtime_path.write_text(json.dumps(changed_runtime, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(runner.ContractError, match="runtime authorization contract changed"):
        runner.load_and_verify_contract(changed_runtime_path)


def test_real_commands_fail_before_any_data_access_without_canonical_scope(bundle, tmp_path):
    missing_scope = tmp_path / "missing.run.json"
    registry = tmp_path / "registry.jsonl"
    registry.write_text("", encoding="utf-8")
    with pytest.raises(runner.ContractError, match="canonical run scope is missing"):
        runner.verify_real_data_authorization(missing_scope, registry)
