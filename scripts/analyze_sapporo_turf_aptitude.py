from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_sapporo_special_logic import add_sapporo_components, score_with_weights  # noqa: E402
from scripts.analyze_sapporo_stay_transport_factors import add_prev_venue_by_history  # noqa: E402
from scripts.evaluate_expected_lap_rpci_features import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_TEST,
    DEFAULT_TRAIN,
    NEW_CATEGORICAL_FEATURES,
    NEW_NUMERIC_FEATURES,
    RACE_COL,
    add_expected_lap_features,
    fit_ranker,
    num,
)
from src.train.simple_ranker import SimpleRaceRanker  # noqa: E402


DEFAULT_OUT = Path("outputs/analysis/sapporo_turf_aptitude")

SAPPORO_TURF_WEIGHT = {"bias_fit": 0.02}


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def z(values: pd.Series, race: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0)
    mean = values.groupby(race).transform("mean")
    std = values.groupby(race).transform("std").replace(0, np.nan)
    return ((values - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def distance_group(df: pd.DataFrame) -> pd.Series:
    dist = num(df, "距離")
    return pd.Series(
        np.select(
            [dist.le(1200), dist.eq(1500), dist.between(1700, 1800), dist.eq(2000), dist.ge(2400)],
            ["<=1200", "1500", "1700-1800", "2000", "2400+"],
            default="other",
        ),
        index=df.index,
    )


def add_aptitude_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    race = out[RACE_COL]
    sustain = num(out, "horse_sustain_lap_score_past5", 0.0)
    long_spurt = num(out, "horse_long_spurt_lap_score_past5", 0.0)
    slow = num(out, "horse_slow_lap_score_past5", 0.0)
    fast = num(out, "horse_fast_lap_score_past5", 0.0)
    instant = num(out, "horse_instant_lap_score_past5", 0.0)
    body = num(out, "body_prev_weight", np.nan).fillna(num(out, "前走馬体重", np.nan)).fillna(470.0)
    body_power = ((body - 450.0) / 80.0).clip(-0.8, 1.2)
    venue = (
        0.16 * num(out, "same_venue_avg_score", 0.0)
        + 0.14 * num(out, "same_venue_top3_rate", 0.0)
        + 0.12 * num(out, "sire_venue_avg_score", 0.0)
        + 0.10 * num(out, "bms_venue_avg_score", 0.0)
        + 0.10 * num(out, "jockey_venue_avg_score", 0.0)
        + 0.10 * num(out, "trainer_venue_avg_score", 0.0)
        + 0.08 * num(out, "owner_venue_top3_rate", 0.0)
        + 0.08 * num(out, "breeder_venue_top3_rate", 0.0)
        + 0.06 * num(out, "sire_going_top3_rate", 0.0)
        + 0.06 * num(out, "bms_going_top3_rate", 0.0)
    )
    hokkaido = (
        0.22 * num(out, "same_venue_avg_score", 0.0)
        + 0.18 * num(out, "sire_venue_lift", 0.0)
        + 0.16 * num(out, "bms_venue_lift", 0.0)
        + 0.16 * num(out, "bloodline_lift_fit_score", 0.0)
        + 0.14 * sustain
        + 0.14 * long_spurt
    )
    power = 0.30 * body_power + 0.22 * sustain + 0.20 * long_spurt + 0.16 * slow - 0.12 * instant
    speed_mismatch = instant + fast - sustain - long_spurt - body_power
    out["sapporo_turf_venue_fit_z"] = z(venue, race)
    out["sapporo_turf_hokkaido_fit_z"] = z(hokkaido, race)
    out["sapporo_turf_power_fit_z"] = z(power, race)
    out["sapporo_turf_speed_mismatch_z"] = z(speed_mismatch, race)
    out["sapporo_turf_composite_fit_z"] = z(0.38 * hokkaido + 0.32 * power + 0.20 * venue + 0.10 * num(out, "expected_lap_total_fit_score", 0.0), race)
    return out


def prepare_scored(train: pd.DataFrame, test: pd.DataFrame, base_model: SimpleRaceRanker) -> pd.DataFrame:
    train_x, test_x, _ = add_expected_lap_features(train, test)
    plus_numeric = list(base_model.numeric_features) + [
        col for col in NEW_NUMERIC_FEATURES if col not in base_model.numeric_features
    ]
    plus_categorical = list(base_model.categorical_features) + [
        col for col in NEW_CATEGORICAL_FEATURES if col not in base_model.categorical_features
    ]
    model = fit_ranker(train_x, plus_numeric, plus_categorical, float(base_model.ridge_alpha), int(base_model.categorical_top_k))
    test_x["expected_lap_score"] = model.predict(test_x)
    test_x = add_prev_venue_by_history(train_x, test_x)
    sapporo = test_x[
        test_x["場所"].astype(str).eq("札幌") & test_x["芝・ダ"].astype(str).str.contains("芝", regex=False, na=False)
    ].copy()
    sapporo = add_sapporo_components(sapporo, "expected_lap_score")
    sapporo["sapporo_turf_score"] = score_with_weights(sapporo, "expected_lap_score", SAPPORO_TURF_WEIGHT)
    sapporo["distance_group"] = distance_group(sapporo)
    sapporo = add_aptitude_scores(sapporo)
    return sapporo


def metrics(part: pd.DataFrame) -> dict[str, Any]:
    if len(part) == 0:
        return {}
    win_pay = num(part, "単勝配当", 0.0).where(part["target_win"].eq(1), 0.0)
    place_pay = num(part, "複勝配当", 0.0).where(part["target_top3"].eq(1), 0.0)
    return {
        "bets": int(len(part)),
        "races": int(part[RACE_COL].nunique()),
        "win_rate": float(part["target_win"].mean()),
        "top3_rate": float(part["target_top3"].mean()),
        "win_roi": float(win_pay.sum() / (len(part) * 100.0)),
        "place_roi": float(place_pay.sum() / (len(part) * 100.0)),
        "avg_popularity": float(num(part, "人気").mean()),
        "avg_odds": float(num(part, "単勝オッズ").mean()),
        "avg_body": float(num(part, "body_prev_weight", np.nan).fillna(num(part, "前走馬体重", np.nan)).mean()),
        "avg_4c": float(num(part, "4角.1", np.nan).mean()),
    }


def segment_report(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    out["rank"] = out.groupby(RACE_COL)["sapporo_turf_score"].rank(ascending=False, method="first").astype(int)
    top1 = out["rank"].eq(1)
    top3 = out["rank"].le(3)
    q = {
        "venue_hi": num(out, "sapporo_turf_venue_fit_z").quantile(0.70),
        "hokkaido_hi": num(out, "sapporo_turf_hokkaido_fit_z").quantile(0.70),
        "power_hi": num(out, "sapporo_turf_power_fit_z").quantile(0.70),
        "composite_hi": num(out, "sapporo_turf_composite_fit_z").quantile(0.70),
        "mismatch_hi": num(out, "sapporo_turf_speed_mismatch_z").quantile(0.70),
    }
    body = num(out, "body_prev_weight", np.nan).fillna(num(out, "前走馬体重", np.nan))
    checks = [
        ("top1_all", top1),
        ("top1_composite_hi", top1 & num(out, "sapporo_turf_composite_fit_z").ge(q["composite_hi"])),
        ("top1_venue_hi", top1 & num(out, "sapporo_turf_venue_fit_z").ge(q["venue_hi"])),
        ("top1_hokkaido_hi", top1 & num(out, "sapporo_turf_hokkaido_fit_z").ge(q["hokkaido_hi"])),
        ("top1_power_hi", top1 & num(out, "sapporo_turf_power_fit_z").ge(q["power_hi"])),
        ("top1_speed_mismatch_hi", top1 & num(out, "sapporo_turf_speed_mismatch_z").ge(q["mismatch_hi"])),
        ("top1_body_480plus", top1 & body.ge(480)),
        ("top1_body_500plus", top1 & body.ge(500)),
        ("top1_prev_hokkaido", top1 & out["prev_hokkaido_flag"].eq(1)),
        ("top1_prev_non_hokkaido", top1 & out["prev_hokkaido_flag"].eq(0)),
        ("top1_hakodate_to_sapporo", top1 & out["hakodate_to_sapporo_flag"].eq(1)),
        ("top1_sapporo_repeat", top1 & out["sapporo_repeat_flag"].eq(1)),
        ("top1_first_from_mainland", top1 & out["sapporo_first_from_mainland_flag"].eq(1)),
        ("top1_1200", top1 & out["distance_group"].eq("<=1200")),
        ("top1_1500", top1 & out["distance_group"].eq("1500")),
        ("top1_1800", top1 & out["distance_group"].eq("1700-1800")),
        ("top1_2000", top1 & out["distance_group"].eq("2000")),
        ("top1_2400plus", top1 & out["distance_group"].eq("2400+")),
        ("top3_pop5plus_composite_hi", top3 & num(out, "人気").ge(5) & num(out, "sapporo_turf_composite_fit_z").ge(q["composite_hi"])),
        ("top3_pop5plus_power_hi", top3 & num(out, "人気").ge(5) & num(out, "sapporo_turf_power_fit_z").ge(q["power_hi"])),
    ]
    rows = []
    for name, mask in checks:
        rows.append({"segment": name, **metrics(out[mask])})
    return pd.DataFrame(rows).sort_values(["win_roi", "place_roi"], ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Sapporo turf / western turf aptitude factors.")
    parser.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(project_path(args.train_csv), low_memory=False)
    test = pd.read_csv(project_path(args.test_csv), low_memory=False)
    base_model: SimpleRaceRanker = pickle.load(project_path(args.base_model).open("rb"))
    scored = prepare_scored(train, test, base_model)
    report = segment_report(scored)
    report.to_csv(out_dir / "sapporo_turf_aptitude_segments.csv", index=False, encoding="utf-8-sig")
    scored.to_csv(out_dir / "sapporo_turf_aptitude_scored.csv", index=False, encoding="utf-8-sig")

    show = report.copy()
    pct = [c for c in show.columns if c.endswith("_rate") or c.endswith("_roi")]
    show[pct] = show[pct] * 100.0
    print(show.to_string(index=False))
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
