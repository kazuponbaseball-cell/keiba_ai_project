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


def _going_bucket(values: pd.Series) -> pd.Series:
    text = values.astype("string").fillna("")
    return pd.Series(
        np.select(
            [
                text.str.contains("良", regex=False, na=False),
                text.str.contains("稍", regex=False, na=False),
                text.str.contains("重", regex=False, na=False),
                text.str.contains("不", regex=False, na=False),
            ],
            ["firm", "slightly_yielding", "yielding", "muddy"],
            default="unknown",
        ),
        index=values.index,
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


def _score_frame(frame: pd.DataFrame, model: object, race_col: str) -> pd.DataFrame:
    out = frame.copy()
    out["ai_score"] = model.predict(out)
    out["ai_rank"] = out.groupby(race_col)["ai_score"].rank(ascending=False, method="first").astype(int)
    out["going_bucket"] = _going_bucket(out["馬場状態"])
    out["surface_simple"] = _surface_bucket(out["芝・ダ"])
    out["distance_bucket_simple"] = _distance_bucket(out["距離"])
    out["odds_decimal"] = _num(out, "単勝オッズ")

    for col in [
        "bloodline_high_confidence_fit_score",
        "bloodline_lift_fit_score",
        "bloodline_pair_fit_score",
        "bloodline_reliability_score",
        "sire_going_top3_rate",
        "bms_going_top3_rate",
        "sire_going_lift",
        "bms_going_lift",
        "sire_surface_lift",
        "bms_surface_lift",
    ]:
        out[col] = _num(out, col, 0.0).fillna(0.0)

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
            "avg_bloodline_score": None,
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
        "avg_odds": float(frame["odds_decimal"].mean()),
        "avg_bloodline_score": float(frame["bloodline_high_confidence_fit_score"].mean()),
    }


def _quantiles(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "blood_hi": float(frame["bloodline_high_confidence_fit_score"].quantile(0.75)),
        "blood_top": float(frame["bloodline_high_confidence_fit_score"].quantile(0.90)),
        "lift_hi": float(frame["bloodline_lift_fit_score"].quantile(0.75)),
        "lift_top": float(frame["bloodline_lift_fit_score"].quantile(0.90)),
        "sire_going_hi": float(frame["sire_going_lift"].quantile(0.75)),
        "bms_going_hi": float(frame["bms_going_lift"].quantile(0.75)),
    }


def _split_by_date(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    date = _num(frame, "日付")
    dates = np.array(sorted(date.dropna().unique()))
    cutoff = dates[max(1, min(len(dates) - 1, int(len(dates) * 0.5)))]
    return date < cutoff, date >= cutoff


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze going x bloodline positive overlays.")
    parser.add_argument("--test-csv", default=DEFAULT_TEST)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="outputs/analysis/bloodline_going_segments")
    parser.add_argument("--race-col", default="レースID(新/馬番無)")
    args = parser.parse_args()

    with Path(args.model).open("rb") as f:
        model = pickle.load(f)
    frame = pd.read_csv(args.test_csv, encoding="utf-8-sig", low_memory=False)
    scored = _score_frame(frame, model, args.race_col)
    discovery_mask, validation_mask = _split_by_date(scored)
    q = _quantiles(scored[discovery_mask])

    base_masks: dict[str, pd.Series] = {
        "ai_top1": scored["ai_rank"].eq(1),
        "ai_top3": scored["ai_rank"].le(3),
        "ai_top5": scored["ai_rank"].le(5),
        "blood_hi": scored["bloodline_high_confidence_fit_score"].ge(q["blood_hi"]),
        "blood_top10": scored["bloodline_high_confidence_fit_score"].ge(q["blood_top"]),
        "blood_lift_hi": scored["bloodline_lift_fit_score"].ge(q["lift_hi"]),
        "blood_lift_top10": scored["bloodline_lift_fit_score"].ge(q["lift_top"]),
        "sire_going_lift_hi": scored["sire_going_lift"].ge(q["sire_going_hi"]),
        "bms_going_lift_hi": scored["bms_going_lift"].ge(q["bms_going_hi"]),
        "reliable": scored["bloodline_reliability_score"].ge(0.7),
    }

    for col in ["going_bucket", "surface_simple", "distance_bucket_simple"]:
        for value in sorted(scored[col].dropna().unique()):
            base_masks[f"{col}={value}"] = scored[col].eq(value)

    combos: list[tuple[str, ...]] = []
    blood_terms = [
        "blood_hi",
        "blood_top10",
        "blood_lift_hi",
        "blood_lift_top10",
        "sire_going_lift_hi",
        "bms_going_lift_hi",
    ]
    going_terms = [
        "going_bucket=firm",
        "going_bucket=slightly_yielding",
        "going_bucket=yielding",
        "going_bucket=muddy",
    ]
    surface_terms = ["surface_simple=turf", "surface_simple=dirt"]
    distance_terms = [
        "distance_bucket_simple=sprint_1300",
        "distance_bucket_simple=mile_1600",
        "distance_bucket_simple=middle_2000",
        "distance_bucket_simple=long_2600",
    ]
    for ai in ["ai_top1", "ai_top3", "ai_top5"]:
        for blood in blood_terms:
            combos.append((ai, blood))
            combos.append((ai, blood, "reliable"))
            for going in going_terms:
                combos.append((ai, blood, going))
                combos.append((ai, blood, going, "reliable"))
                for surface in surface_terms:
                    combos.append((ai, blood, going, surface))
                for distance in distance_terms:
                    combos.append((ai, blood, going, distance))

    rows = []
    for combo in combos:
        mask = pd.Series(True, index=scored.index)
        missing = False
        for term in combo:
            if term not in base_masks:
                missing = True
                break
            mask &= base_masks[term]
        if missing:
            continue
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
        segments = segments.sort_values(
            ["validation_edge_score", "validation_win_roi", "validation_place_roi"], ascending=False
        )

    bucket_rows = []
    for cols in [
        ["going_bucket"],
        ["going_bucket", "surface_simple"],
        ["going_bucket", "surface_simple", "distance_bucket_simple"],
    ]:
        for key, part in scored.groupby(cols, dropna=False):
            if len(part) < 80:
                continue
            key_text = key if isinstance(key, tuple) else (key,)
            label = "&".join(f"{col}={value}" for col, value in zip(cols, key_text))
            bucket_rows.append(_metrics(part, args.race_col, label))
    bucket_df = pd.DataFrame(bucket_rows).sort_values(["win_roi", "place_roi"], ascending=False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    segments.to_csv(output_dir / "bloodline_going_segments.csv", index=False, encoding="utf-8-sig")
    segments.head(100).to_csv(output_dir / "top_bloodline_going_segments.csv", index=False, encoding="utf-8-sig")
    bucket_df.to_csv(output_dir / "going_bucket_metrics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "output_dir": str(output_dir),
        "rows": int(len(scored)),
        "races": int(scored[args.race_col].nunique()),
        "thresholds": q,
        "top_segments": segments.head(30).to_dict(orient="records") if not segments.empty else [],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
