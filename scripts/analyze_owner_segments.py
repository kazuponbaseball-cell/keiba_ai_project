from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index)
    if series.dtype == object or str(series.dtype).startswith("string"):
        series = series.astype("string").str.replace(",", "", regex=False).str.replace("+", "", regex=False)
    return pd.to_numeric(series, errors="coerce")


def _metrics(frame: pd.DataFrame, label: str, race_col: str) -> dict[str, object]:
    rows = len(frame)
    if rows == 0:
        return {"label": label, "bets": 0, "races": 0}
    win_pay = _num(frame.get("単勝配当"), frame.index, 0.0).fillna(0.0).where(frame["target_win"].eq(1), 0.0)
    place_pay = _num(frame.get("複勝配当"), frame.index, 0.0).fillna(0.0).where(frame["target_top3"].eq(1), 0.0)
    return {
        "label": label,
        "bets": int(rows),
        "races": int(frame[race_col].nunique()),
        "win_rate": float(frame["target_win"].mean()),
        "top3_rate": float(frame["target_top3"].mean()),
        "win_roi": float(win_pay.sum() / (rows * 100.0)),
        "place_roi": float(place_pay.sum() / (rows * 100.0)),
        "avg_popularity": float(_num(frame.get("人気"), frame.index).mean()),
        "avg_odds": float(_num(frame.get("単勝オッズ"), frame.index).mean()),
    }


def _safe_quantile(series: pd.Series, q: float, fallback: float) -> float:
    value = _num(series, series.index).quantile(q)
    return float(value) if pd.notna(value) else fallback


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-csv",
        default="data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/body_weight_backfilled/owner_enriched/test_features_with_same_day_bias_v3_retro_body_owner.csv",
    )
    parser.add_argument("--model", default="models/body_owner_context_same_day_bias_v3_retro/baseline_ranker.pkl")
    parser.add_argument("--output-dir", default="outputs/analysis/owner_segments")
    parser.add_argument("--race-col", default="レースID(新/馬番無)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.test_csv, low_memory=False)
    with Path(args.model).open("rb") as f:
        model = pickle.load(f)

    frame["ai_score"] = model.predict(frame)
    frame["ai_rank"] = frame.groupby(args.race_col)["ai_score"].rank(ascending=False, method="first").astype(int)

    owner_starts = _num(frame.get("owner_starts"), frame.index, 0).fillna(0)
    owner_top3 = _num(frame.get("owner_top3_rate"), frame.index)
    owner_pop = _num(frame.get("owner_popularity_outperform_rate"), frame.index)
    owner_context = _num(frame.get("owner_context_fit_score"), frame.index)
    owner_synergy = _num(frame.get("owner_trainer_synergy_score"), frame.index)
    owner_pair_starts = _num(frame.get("owner_trainer_pair_starts"), frame.index, 0).fillna(0)
    owner_class = _num(frame.get("owner_class_top3_rate"), frame.index)
    owner_surface = _num(frame.get("owner_surface_top3_rate"), frame.index)
    owner_venue = _num(frame.get("owner_venue_top3_rate"), frame.index)

    q_top3_75 = _safe_quantile(owner_top3[owner_starts.ge(20)], 0.75, 0.4)
    q_pop_75 = _safe_quantile(owner_pop[owner_starts.ge(20)], 0.75, 0.4)
    q_context_75 = _safe_quantile(owner_context[owner_starts.ge(20)], 0.75, 0.03)
    q_class_75 = _safe_quantile(owner_class[owner_starts.ge(20)], 0.75, 0.4)
    q_surface_75 = _safe_quantile(owner_surface[owner_starts.ge(20)], 0.75, 0.4)
    q_venue_75 = _safe_quantile(owner_venue[owner_starts.ge(20)], 0.75, 0.4)

    masks: dict[str, pd.Series] = {
        "ai_top1_all": frame["ai_rank"].eq(1),
        "ai_top3_all": frame["ai_rank"].le(3),
        "ai_top1_owner20_top3_q75": frame["ai_rank"].eq(1) & owner_starts.ge(20) & owner_top3.ge(q_top3_75),
        "ai_top1_owner50_top3_q75": frame["ai_rank"].eq(1) & owner_starts.ge(50) & owner_top3.ge(q_top3_75),
        "ai_top1_owner20_pop_outperform_q75": frame["ai_rank"].eq(1) & owner_starts.ge(20) & owner_pop.ge(q_pop_75),
        "ai_top1_owner20_context_q75": frame["ai_rank"].eq(1) & owner_starts.ge(20) & owner_context.ge(q_context_75),
        "ai_top1_owner20_class_q75": frame["ai_rank"].eq(1) & owner_starts.ge(20) & owner_class.ge(q_class_75),
        "ai_top1_owner20_surface_q75": frame["ai_rank"].eq(1) & owner_starts.ge(20) & owner_surface.ge(q_surface_75),
        "ai_top1_owner20_venue_q75": frame["ai_rank"].eq(1) & owner_starts.ge(20) & owner_venue.ge(q_venue_75),
        "ai_top3_owner20_context_q75": frame["ai_rank"].le(3) & owner_starts.ge(20) & owner_context.ge(q_context_75),
        "ai_top3_owner_trainer_synergy_pos": frame["ai_rank"].le(3) & owner_pair_starts.ge(5) & owner_synergy.gt(0),
        "ai_top1_owner_trainer_synergy_pos": frame["ai_rank"].eq(1) & owner_pair_starts.ge(5) & owner_synergy.gt(0),
        "ai_top1_owner_low_sample_lt5": frame["ai_rank"].eq(1) & owner_starts.lt(5),
    }

    rows = [_metrics(frame[mask.fillna(False)], label, args.race_col) for label, mask in masks.items()]
    summary = pd.DataFrame(rows).sort_values(["win_roi", "place_roi"], ascending=False)
    summary.to_csv(output_dir / "owner_segment_summary.csv", index=False, encoding="utf-8-sig")
    detail = {
        "thresholds": {
            "owner_top3_q75_min20": q_top3_75,
            "owner_popularity_outperform_q75_min20": q_pop_75,
            "owner_context_fit_q75_min20": q_context_75,
            "owner_class_top3_q75_min20": q_class_75,
            "owner_surface_top3_q75_min20": q_surface_75,
            "owner_venue_top3_q75_min20": q_venue_75,
        },
        "output_dir": str(output_dir),
        "summary": summary.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps({"output_dir": str(output_dir), "thresholds": detail["thresholds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
