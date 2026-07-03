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
from scripts.evaluate_sapporo_dirt1700_policy import SAPPORO_D1700_WEIGHTS, is_sapporo_dirt1700  # noqa: E402
from src.train.simple_ranker import SimpleRaceRanker  # noqa: E402


DEFAULT_OUT = Path("outputs/analysis/sapporo_stay_transport")
HORSE_COL = "血統登録番号"
DATE_COL = "日付"


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else Path.cwd() / path


def add_prev_venue_by_history(train_x: pd.DataFrame, test_x: pd.DataFrame) -> pd.DataFrame:
    train = train_x.copy()
    test = test_x.copy()
    train["_split_for_prev_venue"] = "train"
    test["_split_for_prev_venue"] = "test"
    test["_test_row_id"] = np.arange(len(test))
    train["_test_row_id"] = -1
    all_df = pd.concat([train, test], ignore_index=True, sort=False)
    all_df["_date_num"] = num(all_df, DATE_COL, np.nan)
    all_df["_race_num"] = num(all_df, RACE_COL, np.nan)
    ordered = all_df.sort_values([HORSE_COL, "_date_num", "_race_num"], kind="mergesort")
    prev_venue = ordered.groupby(HORSE_COL, sort=False)["場所"].shift()
    prev_date = ordered.groupby(HORSE_COL, sort=False)["_date_num"].shift()
    all_df.loc[ordered.index, "prev_venue"] = prev_venue
    all_df.loc[ordered.index, "prev_date_for_venue"] = prev_date
    out = all_df[all_df["_split_for_prev_venue"].eq("test")].sort_values("_test_row_id", kind="mergesort").copy()
    out = out.drop(columns=["_split_for_prev_venue", "_test_row_id", "_date_num", "_race_num"], errors="ignore")
    prev = out["prev_venue"].astype("string").fillna("")
    venue = out["場所"].astype("string").fillna("")
    out["prev_hakodate_flag"] = prev.eq("函館").astype(float)
    out["prev_sapporo_flag"] = prev.eq("札幌").astype(float)
    out["prev_hokkaido_flag"] = prev.isin(["函館", "札幌"]).astype(float)
    out["prev_same_venue_flag"] = prev.eq(venue).astype(float)
    out["prev_local_flag"] = prev.isin(["函館", "札幌", "福島", "新潟", "小倉"]).astype(float)
    out["prev_central_flag"] = prev.isin(["東京", "中山", "京都", "阪神", "中京"]).astype(float)
    out["sapporo_first_from_mainland_flag"] = (venue.eq("札幌") & out["prev_central_flag"].eq(1)).astype(float)
    out["sapporo_stay_like_flag"] = (venue.eq("札幌") & out["prev_hokkaido_flag"].eq(1)).astype(float)
    out["hakodate_to_sapporo_flag"] = (venue.eq("札幌") & out["prev_hakodate_flag"].eq(1)).astype(float)
    out["sapporo_repeat_flag"] = (venue.eq("札幌") & out["prev_sapporo_flag"].eq(1)).astype(float)
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
    sapporo = test_x[test_x["場所"].astype(str).eq("札幌")].copy()
    sapporo = add_sapporo_components(sapporo, "expected_lap_score")
    sapporo["surface_group"] = np.where(sapporo["芝・ダ"].astype(str).str.contains("芝", na=False), "turf", "dirt")
    sapporo["sapporo_score"] = sapporo["expected_lap_score"]
    dirt1700 = is_sapporo_dirt1700(sapporo)
    sapporo.loc[dirt1700, "sapporo_score"] = score_with_weights(sapporo.loc[dirt1700], "expected_lap_score", SAPPORO_D1700_WEIGHTS)
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
        "avg_interval": float(num(part, "間隔").mean()),
        "avg_body_delta": float(num(part, "前走馬体重増減").mean()),
    }


def segment_report(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    out["rank"] = out.groupby(RACE_COL)["sapporo_score"].rank(ascending=False, method="first").astype(int)
    top1 = out["rank"].eq(1)
    top3 = out["rank"].le(3)
    checks = [
        ("top1_all", top1),
        ("top1_prev_hokkaido", top1 & out["prev_hokkaido_flag"].eq(1)),
        ("top1_prev_non_hokkaido", top1 & out["prev_hokkaido_flag"].eq(0)),
        ("top1_hakodate_to_sapporo", top1 & out["hakodate_to_sapporo_flag"].eq(1)),
        ("top1_sapporo_repeat", top1 & out["sapporo_repeat_flag"].eq(1)),
        ("top1_first_from_mainland", top1 & out["sapporo_first_from_mainland_flag"].eq(1)),
        ("top1_prev_local", top1 & out["prev_local_flag"].eq(1)),
        ("top1_prev_central", top1 & out["prev_central_flag"].eq(1)),
        ("top1_turf_prev_hokkaido", top1 & out["surface_group"].eq("turf") & out["prev_hokkaido_flag"].eq(1)),
        ("top1_turf_prev_non_hokkaido", top1 & out["surface_group"].eq("turf") & out["prev_hokkaido_flag"].eq(0)),
        ("top1_dirt_prev_hokkaido", top1 & out["surface_group"].eq("dirt") & out["prev_hokkaido_flag"].eq(1)),
        ("top1_dirt_prev_non_hokkaido", top1 & out["surface_group"].eq("dirt") & out["prev_hokkaido_flag"].eq(0)),
        ("top3_pop5plus_prev_hokkaido", top3 & num(out, "人気").ge(5) & out["prev_hokkaido_flag"].eq(1)),
        ("top3_pop5plus_prev_non_hokkaido", top3 & num(out, "人気").ge(5) & out["prev_hokkaido_flag"].eq(0)),
    ]
    rows = []
    for name, mask in checks:
        rows.append({"segment": name, **metrics(out[mask])})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Sapporo stay/transport factors from previous venue.")
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
    report.to_csv(out_dir / "sapporo_stay_transport_segments.csv", index=False, encoding="utf-8-sig")
    scored.to_csv(out_dir / "sapporo_stay_transport_scored.csv", index=False, encoding="utf-8-sig")

    show = report.copy()
    pct = [c for c in show.columns if c.endswith("_rate") or c.endswith("_roi")]
    show[pct] = show[pct] * 100.0
    print(show.to_string(index=False))
    print(f"\nOutput: {out_dir}")


if __name__ == "__main__":
    main()
