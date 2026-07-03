from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_TEST = "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/body_weight_backfilled/test_features_with_same_day_bias_v3_retro_body_context.csv"
DEFAULT_MODEL = "models/body_context_same_day_bias_v3_retro/baseline_ranker.pkl"


def _num(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index)
    values = frame[column]
    if values.dtype == object or str(values.dtype).startswith("string"):
        values = values.astype("string").str.replace(",", "", regex=False).str.replace("+", "", regex=False)
    return pd.to_numeric(values, errors="coerce")


def _month_from_date(values: pd.Series) -> pd.Series:
    text = values.astype("string").fillna("").str.replace(".0", "", regex=False)
    digits = text.str.replace(r"\D", "", regex=True)
    month = pd.to_numeric(digits.str[-4:-2], errors="coerce")
    return month


def _season_bucket(month: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [
                month.isin([12, 1, 2]),
                month.isin([3, 4, 5]),
                month.isin([6, 7, 8]),
                month.isin([9, 10, 11]),
            ],
            ["winter", "spring", "summer", "autumn"],
            default="unknown",
        ),
        index=month.index,
    )


def _surface_bucket(values: pd.Series) -> pd.Series:
    text = values.astype("string").fillna("")
    return pd.Series(
        np.select(
            [text.str.contains("芝", regex=False, na=False), text.str.contains("ダ", regex=False, na=False)],
            ["turf", "dirt"],
            default="other",
        ),
        index=values.index,
    )


def _distance_bucket(values: pd.Series) -> pd.Series:
    distance = pd.to_numeric(values, errors="coerce")
    return pd.Series(
        np.select(
            [distance <= 1300, distance.between(1400, 1800), distance.between(1900, 2400), distance >= 2500],
            ["sprint_1300", "mile_1600", "middle_2000", "long_2600"],
            default="unknown",
        ),
        index=values.index,
    )


def _class_bucket(values: pd.Series) -> pd.Series:
    text = values.astype("string").fillna("")
    return pd.Series(
        np.select(
            [
                text.str.contains("新馬", na=False),
                text.str.contains("未勝利", na=False),
                text.str.contains("１勝|1勝|500万", regex=True, na=False),
                text.str.contains("２勝|2勝|1000万", regex=True, na=False),
                text.str.contains("３勝|3勝|1600万", regex=True, na=False),
                text.str.contains("オープン|OP|Ｌ|L", regex=True, na=False),
                text.str.contains("Ｇ|G", regex=True, na=False),
            ],
            ["newcomer", "maiden", "class_1win", "class_2win", "class_3win", "open", "graded"],
            default="other",
        ),
        index=values.index,
    )


def _score_frame(frame: pd.DataFrame, model: object, race_col: str) -> pd.DataFrame:
    out = frame.copy()
    out["ai_score"] = model.predict(out)
    out["ai_rank"] = out.groupby(race_col)["ai_score"].rank(ascending=False, method="first").astype(int)
    out["month"] = _month_from_date(out["日付"])
    out["season_bucket"] = _season_bucket(out["month"])
    out["surface_simple"] = _surface_bucket(out["芝・ダ"])
    out["distance_bucket_simple"] = _distance_bucket(out["距離"])
    out["class_bucket"] = _class_bucket(out["クラス名"])
    weight = _num(out, "前走馬体重")
    out["prev_body_weight_num"] = weight
    out["body_weight_bucket"] = pd.cut(
        weight,
        bins=[-np.inf, 439, 459, 479, 499, 519, 539, np.inf],
        labels=["<=439", "440-459", "460-479", "480-499", "500-519", "520-539", "540+"],
    ).astype("string")
    out.loc[weight.isna(), "body_weight_bucket"] = "missing"
    if "body_weight_rank_in_race" not in out.columns:
        out["body_weight_rank_in_race"] = weight.groupby(out[race_col]).rank(ascending=False, method="average")
    return out


def _metrics(frame: pd.DataFrame, race_col: str, label: str) -> dict[str, object]:
    rows = len(frame)
    if rows == 0:
        return {
            "segment": label,
            "bets": 0,
            "races": 0,
            "win_rate": 0.0,
            "top3_rate": 0.0,
            "win_roi": 0.0,
            "place_roi": 0.0,
            "avg_popularity": None,
            "avg_odds": None,
            "avg_prev_body_weight": None,
        }
    win_pay = _num(frame, "単勝配当", 0.0).fillna(0.0).where(frame["target_win"].eq(1), 0.0)
    place_pay = _num(frame, "複勝配当", 0.0).fillna(0.0).where(frame["target_top3"].eq(1), 0.0)
    stake = rows * 100.0
    return {
        "segment": label,
        "bets": int(rows),
        "races": int(frame[race_col].nunique()),
        "win_rate": float(frame["target_win"].mean()),
        "top3_rate": float(frame["target_top3"].mean()),
        "win_roi": float(win_pay.sum() / stake),
        "place_roi": float(place_pay.sum() / stake),
        "avg_popularity": float(_num(frame, "人気").mean()),
        "avg_odds": float(_num(frame, "単勝オッズ").mean()),
        "avg_prev_body_weight": float(frame["prev_body_weight_num"].mean()),
    }


def _split_by_date(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    date = _num(frame, "日付")
    dates = np.array(sorted(date.dropna().unique()))
    cutoff = dates[max(1, min(len(dates) - 1, int(len(dates) * 0.5)))]
    return date < cutoff, date >= cutoff


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze body-weight x season overlays.")
    parser.add_argument("--test-csv", default=DEFAULT_TEST)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="outputs/analysis/body_weight_season_segments")
    parser.add_argument("--race-col", default="レースID(新/馬番無)")
    args = parser.parse_args()

    with Path(args.model).open("rb") as f:
        model = pickle.load(f)
    frame = pd.read_csv(args.test_csv, encoding="utf-8-sig", low_memory=False)
    scored = _score_frame(frame, model, args.race_col)
    discovery_mask, validation_mask = _split_by_date(scored)

    masks: dict[str, pd.Series] = {
        "ai_top1": scored["ai_rank"].eq(1),
        "ai_top3": scored["ai_rank"].le(3),
        "ai_top5": scored["ai_rank"].le(5),
        "big_500": scored["prev_body_weight_num"].ge(500),
        "big_520": scored["prev_body_weight_num"].ge(520),
        "mid_460_499": scored["prev_body_weight_num"].between(460, 499, inclusive="both"),
        "small_460": scored["prev_body_weight_num"].le(460),
        "race_heavy_top3": scored["body_weight_rank_in_race"].le(3),
        "race_heavy_top5": scored["body_weight_rank_in_race"].le(5),
        "age2": _num(scored, "年齢").eq(2),
        "age3": _num(scored, "年齢").eq(3),
    }
    for col in ["season_bucket", "surface_simple", "distance_bucket_simple", "class_bucket"]:
        for value in sorted(scored[col].dropna().unique()):
            masks[f"{col}={value}"] = scored[col].eq(value)

    combos: list[tuple[str, ...]] = []
    body_terms = ["big_500", "big_520", "mid_460_499", "small_460", "race_heavy_top3", "race_heavy_top5"]
    season_terms = [
        "season_bucket=winter",
        "season_bucket=spring",
        "season_bucket=summer",
        "season_bucket=autumn",
    ]
    surface_terms = ["surface_simple=turf", "surface_simple=dirt"]
    distance_terms = [
        "distance_bucket_simple=sprint_1300",
        "distance_bucket_simple=mile_1600",
        "distance_bucket_simple=middle_2000",
        "distance_bucket_simple=long_2600",
    ]
    class_terms = ["class_bucket=newcomer", "class_bucket=maiden", "class_bucket=class_1win", "class_bucket=graded"]
    age_terms = ["age2", "age3"]
    for ai in ["ai_top1", "ai_top3", "ai_top5"]:
        for body in body_terms:
            for season in season_terms:
                combos.append((ai, body, season))
                for surface in surface_terms:
                    combos.append((ai, body, season, surface))
                for distance in distance_terms:
                    combos.append((ai, body, season, distance))
                for cls in class_terms:
                    combos.append((ai, body, season, cls))
                for age in age_terms:
                    combos.append((ai, body, season, age))

    rows = []
    for combo in combos:
        mask = pd.Series(True, index=scored.index)
        for term in combo:
            mask &= masks[term]
        all_part = scored[mask]
        if len(all_part) < 80:
            continue
        discovery = scored[mask & discovery_mask]
        validation = scored[mask & validation_mask]
        if len(discovery) < 35 or len(validation) < 35:
            continue
        row = {"segment": "&".join(combo)}
        for prefix, part in [("all", all_part), ("discovery", discovery), ("validation", validation)]:
            metrics = _metrics(part, args.race_col, row["segment"])
            row.update({f"{prefix}_{k}": v for k, v in metrics.items() if k != "segment"})
        row["validation_edge_score"] = (
            row["validation_win_roi"] + 0.35 * row["validation_place_roi"] + 0.20 * row["validation_top3_rate"]
        )
        row["roi_stability_win"] = row["validation_win_roi"] - row["discovery_win_roi"]
        rows.append(row)

    segments = pd.DataFrame(rows)
    if not segments.empty:
        segments = segments.sort_values(["validation_edge_score", "validation_win_roi", "validation_place_roi"], ascending=False)

    bucket_rows = []
    for cols in [
        ["season_bucket", "body_weight_bucket"],
        ["season_bucket", "surface_simple", "body_weight_bucket"],
        ["season_bucket", "distance_bucket_simple", "body_weight_bucket"],
    ]:
        for key, part in scored.groupby(cols, dropna=False):
            if len(part) < 80:
                continue
            key_tuple = key if isinstance(key, tuple) else (key,)
            label = "&".join(f"{col}={value}" for col, value in zip(cols, key_tuple))
            bucket_rows.append(_metrics(part, args.race_col, label))
    bucket_df = pd.DataFrame(bucket_rows).sort_values(["win_roi", "place_roi"], ascending=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    segments.to_csv(output_dir / "body_weight_season_segments.csv", index=False, encoding="utf-8-sig")
    segments.head(100).to_csv(output_dir / "top_body_weight_season_segments.csv", index=False, encoding="utf-8-sig")
    bucket_df.to_csv(output_dir / "season_body_weight_bucket_metrics.csv", index=False, encoding="utf-8-sig")
    summary = {
        "output_dir": str(output_dir),
        "rows": int(len(scored)),
        "races": int(scored[args.race_col].nunique()),
        "top_segments": segments.head(30).to_dict(orient="records") if not segments.empty else [],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
