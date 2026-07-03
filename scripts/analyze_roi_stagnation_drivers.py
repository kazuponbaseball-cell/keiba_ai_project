from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import ensure_dir, project_path


def _num(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        cleaned = series.astype(str).str.replace(r"[(),]", "", regex=True).replace({"nan": np.nan, "": np.nan})
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def _rate(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else 0.0


def _summarize(frame: pd.DataFrame, label_col: str, *, min_n: int = 1) -> pd.DataFrame:
    rows = []
    for label, g in frame.groupby(label_col, dropna=False):
        if len(g) < min_n:
            continue
        stake = len(g) * 100.0
        rows.append(
            {
                label_col: str(label),
                "rows": int(len(g)),
                "races": int(g["race_id"].nunique()),
                "win_rate": _rate(g["is_win"]),
                "quinella_rate": _rate(g["is_quinella"]),
                "place_rate": _rate(g["is_place"]),
                "win_roi": float(g["win_return"].sum() / stake) if stake else 0.0,
                "place_roi": float(g["place_return"].sum() / stake) if stake else 0.0,
                "avg_pop": float(g["pop_rank_num"].mean()),
                "avg_odds": float(g["odds_num"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _ai_rank_bucket(rank: pd.Series) -> pd.Series:
    r = _num(rank)
    return np.select(
        [r.eq(1), r.eq(2), r.eq(3), r.eq(4), r.ge(5)],
        ["AI1", "AI2", "AI3", "AI4", "AI5plus"],
        default="unknown",
    )


def _pop_bucket(pop: pd.Series) -> pd.Series:
    p = _num(pop)
    return np.select(
        [p.eq(1), p.eq(2), p.eq(3), p.between(4, 6), p.between(7, 9), p.ge(10)],
        ["pop1", "pop2", "pop3", "pop4_6", "pop7_9", "pop10plus"],
        default="unknown",
    )


def _distance_bin(dist: pd.Series) -> pd.Series:
    d = _num(dist)
    return pd.cut(
        d,
        bins=[0, 1300, 1600, 1900, 2400, 4000],
        labels=["sprint", "mile", "middle", "long", "extended"],
        include_lowest=True,
    ).astype(str)


def _field_bin(field: pd.Series) -> pd.Series:
    f = _num(field)
    return pd.cut(
        f,
        bins=[0, 10, 13, 16, 30],
        labels=["small", "medium", "large", "full"],
        include_lowest=True,
    ).astype(str)


def _rpci_bin(rpci: pd.Series) -> pd.Series:
    r = _num(rpci)
    return pd.cut(
        r,
        bins=[0, 45, 50, 55, 60, 100],
        labels=["fast_lt45", "45_50", "50_55", "55_60", "slow_60plus"],
        include_lowest=True,
    ).astype(str)


def _class_group(cls: pd.Series) -> pd.Series:
    c = cls.astype(str)
    return np.select(
        [
            c.str.contains("新馬", na=False),
            c.str.contains("未勝利", na=False),
            c.str.contains("1勝", na=False),
            c.str.contains("2勝", na=False),
            c.str.contains("3勝", na=False),
            c.str.contains("OP|オープン|G", na=False),
        ],
        ["newcomer", "maiden", "1win", "2win", "3win", "open"],
        default="other",
    )


def _style_bucket(front: pd.Series, closer: pd.Series) -> pd.Series:
    f = _num(front).fillna(0)
    c = _num(closer).fillna(0)
    return np.select(
        [f.ge(0.45), f.ge(0.25), c.ge(0.45), c.ge(0.25)],
        ["front", "stalker", "closer", "mid_closer"],
        default="neutral",
    )


def _load_prediction(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df.rename(
        columns={
            "レースID(新/馬番無)": "race_id",
            "馬番": "horse_no",
            "馬名": "horse_name",
            "確定着順": "finish",
            "人気": "popularity",
            "単勝オッズ": "odds",
            "単勝配当": "win_pay",
            "複勝配当": "place_pay",
            "芝・ダ": "surface",
            "距離": "distance",
            "場所": "venue",
        }
    )
    df["race_id"] = df["race_id"].astype(str)
    df["horse_no"] = _num(df["horse_no"])
    df["finish_num"] = _num(df["finish"])
    df["ai_rank_num"] = _num(df["ai_rank"])
    df["pop_rank_num"] = _num(df.get("pop_rank", df.get("popularity")))
    df["odds_num"] = _num(df["odds"])
    df["win_return"] = _num(df["win_pay"]).where(df["finish_num"].eq(1), 0.0).fillna(0.0)
    df["place_return"] = _num(df["place_pay"]).where(df["finish_num"].le(3), 0.0).fillna(0.0)
    df["is_win"] = df["finish_num"].eq(1)
    df["is_quinella"] = df["finish_num"].le(2)
    df["is_place"] = df["finish_num"].le(3)
    df["ai_rank_bucket"] = _ai_rank_bucket(df["ai_rank_num"])
    df["pop_bucket"] = _pop_bucket(df["pop_rank_num"])
    df["ai_pop_bucket"] = df["ai_rank_bucket"].astype(str) + "_" + df["pop_bucket"].astype(str)
    df["ai_market_gap"] = df["pop_rank_num"] - df["ai_rank_num"]
    df["market_relation"] = np.select(
        [
            df["ai_market_gap"].ge(3),
            df["ai_market_gap"].between(1, 2),
            df["ai_market_gap"].between(-1, 0),
            df["ai_market_gap"].le(-2),
        ],
        ["AI_much_higher_than_market", "AI_higher_than_market", "market_agrees_or_higher", "market_much_higher_than_AI"],
        default="unknown",
    )
    if "distance" in df.columns:
        df["distance_bin"] = _distance_bin(df["distance"])
    return df


def _load_feature_lookup(path: Path, race_col: str = "レースID(新/馬番無)") -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    horse_col = "馬番" if "馬番" in df.columns else None
    if horse_col is None:
        candidates = [c for c in df.columns if "馬番" in c or "umaban" in c.lower()]
        horse_col = candidates[0]
    useful = [
        race_col,
        horse_col,
        "クラス名",
        "頭数",
        "出走頭数",
        "枠番",
        "馬場状態",
        "PCI",
        "PCI3",
        "RPCI",
        "expected_pace",
        "front_running_tendency",
        "closing_tendency",
        "race_front_runner_count",
        "race_front_runner_ratio",
        "race_closer_count",
        "race_closer_ratio",
        "race_pace_collapse_risk",
        "race_slow_pace_risk",
        "pace_fit_score",
        "front_advantage_score",
        "positioning_advantage_score",
        "draw_pace_fit_score",
        "horse_front_run_rate_past5",
        "horse_closer_rate_past5",
        "same_day_bias_ready",
        "same_day_bias_volatility",
        "same_day_bias_fit_score",
        "same_day_pop_adjusted_pace_fit_score",
        "same_day_projected_front_load_score",
        "same_day_projected_closer_load_score",
        "same_day_front_collapse_index",
        "same_day_closer_blocked_index",
    ]
    cols = []
    for c in useful:
        if c in df.columns and c not in cols:
            cols.append(c)
    out = df[cols].copy()
    out = out.rename(columns={race_col: "race_id", horse_col: "horse_no"})
    out["race_id"] = out["race_id"].astype(str)
    out["horse_no"] = _num(out["horse_no"])
    return out.drop_duplicates(["race_id", "horse_no"])


def _enrich(df: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(lookup, on=["race_id", "horse_no"], how="left")
    if "クラス名" in out.columns:
        out["class_group"] = _class_group(out["クラス名"])
    if "出走頭数" in out.columns:
        out["field_bin"] = _field_bin(out["出走頭数"])
    elif "頭数" in out.columns:
        out["field_bin"] = _field_bin(out["頭数"])
    if "RPCI" in out.columns:
        out["rpci_bin"] = _rpci_bin(out["RPCI"])
    if "PCI" in out.columns:
        out["pci_bin"] = _rpci_bin(out["PCI"])
    if "horse_front_run_rate_past5" in out.columns and "horse_closer_rate_past5" in out.columns:
        out["style_bucket"] = _style_bucket(out["horse_front_run_rate_past5"], out["horse_closer_rate_past5"])
    if "same_day_bias_volatility" in out.columns:
        vol = _num(out["same_day_bias_volatility"])
        out["bias_volatility_bin"] = pd.cut(vol, bins=[-999, 0.2, 0.5, 999], labels=["low", "mid", "high"]).astype(str)
    return out


def _top1_loss_segments(df: pd.DataFrame, min_n: int) -> pd.DataFrame:
    top1 = df[df["ai_rank_num"].eq(1)].copy()
    losses = top1[~top1["is_win"]].copy()
    cols = [
        "venue",
        "surface",
        "distance_bin",
        "class_group",
        "field_bin",
        "馬場状態",
        "枠番",
        "pop_bucket",
        "style_bucket",
        "rpci_bin",
        "pci_bin",
        "expected_pace",
        "bias_volatility_bin",
    ]
    rows = []
    for col in cols:
        if col not in top1.columns:
            continue
        tmp_top1 = top1.copy()
        tmp_losses = losses.copy()
        tmp_top1[col] = tmp_top1[col].astype(str)
        tmp_losses[col] = tmp_losses[col].astype(str)
        all_summary = _summarize(tmp_top1, col, min_n=min_n)
        loss_counts = tmp_losses.groupby(col, dropna=False).size().rename("top1_loss_count").reset_index()
        merged = all_summary.merge(loss_counts, on=col, how="left").fillna({"top1_loss_count": 0})
        merged["segment"] = col
        merged = merged.rename(columns={col: "value"})
        merged["loss_share"] = merged["top1_loss_count"] / max(1, len(losses))
        merged["win_rate_lift_vs_top1"] = merged["win_rate"] - float(top1["is_win"].mean())
        rows.append(merged)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _race_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    top1 = df[df["ai_rank_num"].eq(1)].copy()
    race = top1[
        [
            "race_id",
            "is_win",
            "is_quinella",
            "is_place",
            "win_return",
            "place_return",
            "venue",
            "surface",
            "distance_bin",
            "class_group",
            "field_bin",
            "馬場状態",
            "expected_pace",
            "rpci_bin",
            "style_bucket",
            "pop_bucket",
            "pop_rank_num",
            "odds_num",
            "ai_score",
        ]
        + [c for c in ["ai_score_gap_to_second", "race_front_runner_count", "race_closer_count", "race_pace_collapse_risk", "same_day_bias_volatility"] if c in top1.columns]
    ].copy()
    if "ai_score_gap_to_second" in race.columns:
        race["confidence_bin"] = pd.cut(_num(race["ai_score_gap_to_second"]), bins=[-999, 0.05, 0.1, 0.2, 999], labels=["low", "mid", "high", "very_high"]).astype(str)
    race["favorite_strength_bin"] = pd.cut(race["odds_num"], bins=[0, 1.5, 2.5, 4, 999], labels=["odds_lt1.5", "1.5_2.5", "2.5_4", "4plus"], include_lowest=True).astype(str)
    return race


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ROI stagnation drivers and AI weak race structures.")
    parser.add_argument("--prediction-csv", default="outputs/evaluation_workout_optimized_core_same_day_bias_v3_retro/20260614_185944/prediction_detail.csv")
    parser.add_argument("--feature-csv", default="data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features_with_same_day_bias_v3_retro.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/roi_stagnation_drivers_v1")
    parser.add_argument("--min-segment-n", type=int, default=50)
    args = parser.parse_args()

    pred = _load_prediction(project_path(args.prediction_csv))
    lookup = _load_feature_lookup(project_path(args.feature_csv))
    df = _enrich(pred, lookup)
    out_dir = ensure_dir(project_path(args.output_dir))

    ai_rank_summary = _summarize(df[df["ai_rank_num"].le(20)], "ai_rank_bucket")
    ai_rank_pop_summary = _summarize(df[df["ai_rank_num"].le(3)], "ai_pop_bucket", min_n=20)
    market_gap_summary = _summarize(df[df["ai_rank_num"].le(5)], "market_relation", min_n=20)
    top1_loss_segments = _top1_loss_segments(df, min_n=args.min_segment_n)

    race = _race_difficulty(df)
    race_cols = ["venue", "surface", "distance_bin", "class_group", "field_bin", "馬場状態", "expected_pace", "rpci_bin", "style_bucket", "pop_bucket", "confidence_bin", "favorite_strength_bin"]
    race_segments = []
    for col in race_cols:
        if col in race.columns:
            race_segments.append(_summarize(race, col, min_n=max(20, args.min_segment_n // 2)).assign(segment=col).rename(columns={col: "value"}))
    race_difficulty_segments = pd.concat(race_segments, ignore_index=True, sort=False) if race_segments else pd.DataFrame()

    # Compact policy-oriented lists.
    weak = top1_loss_segments.sort_values(["win_rate", "win_roi", "rows"], ascending=[True, True, False]).head(40)
    strong = top1_loss_segments.sort_values(["win_rate", "place_roi", "rows"], ascending=[False, False, False]).head(40)
    market_value = market_gap_summary.sort_values(["win_roi", "place_roi"], ascending=[False, False])

    df.to_csv(out_dir / "prediction_detail_enriched.csv", index=False, encoding="utf-8-sig")
    ai_rank_summary.to_csv(out_dir / "ai_rank_summary.csv", index=False, encoding="utf-8-sig")
    ai_rank_pop_summary.to_csv(out_dir / "ai_rank_popularity_summary.csv", index=False, encoding="utf-8-sig")
    market_gap_summary.to_csv(out_dir / "ai_market_gap_summary.csv", index=False, encoding="utf-8-sig")
    top1_loss_segments.to_csv(out_dir / "ai_top1_loss_structure_segments.csv", index=False, encoding="utf-8-sig")
    race_difficulty_segments.to_csv(out_dir / "race_difficulty_segments.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "ai_rank_summary": ai_rank_summary.to_dict(orient="records"),
        "ai_rank_popularity_top": ai_rank_pop_summary.sort_values(["ai_pop_bucket"]).to_dict(orient="records"),
        "market_gap_summary": market_gap_summary.to_dict(orient="records"),
        "top1_weak_segments": weak.to_dict(orient="records"),
        "top1_strong_segments": strong.to_dict(orient="records"),
        "race_difficulty_best": race_difficulty_segments.sort_values(["win_rate", "place_roi"], ascending=[False, False]).head(30).to_dict(orient="records"),
        "race_difficulty_worst": race_difficulty_segments.sort_values(["win_rate", "win_roi"], ascending=[True, True]).head(30).to_dict(orient="records"),
        "interpretation_seed": {
            "model_problem_ranking": [
                "見送り不足",
                "オッズ期待値不足",
                "展開予測不足",
                "モデル/目的関数不足",
                "特徴量不足",
            ],
            "reason": "AI top1 has reasonable hit rates but sub-100 ROI, suggesting selection and bet/no-bet are more important than adding broad features.",
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
