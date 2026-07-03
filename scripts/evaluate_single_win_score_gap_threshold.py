from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


RACE_COL = "レースID(新/馬番無)"
DATE_COL = "日付"
RACE_NO_COL = "Ｒ"
VENUE_COL = "場所"
HORSE_COL = "馬名"
POPULARITY_COL = "人気"
ODDS_COL = "単勝オッズ"
WIN_PAY_COL = "単勝配当"


def num(series: pd.Series | None, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if series is None:
        if index is None:
            raise ValueError("index is required when series is None")
        return pd.Series(default, index=index)
    if series.dtype == object or str(series.dtype).startswith("string"):
        series = (
            series.astype("string")
            .str.replace(",", "", regex=False)
            .str.replace("+", "", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
        )
    return pd.to_numeric(series, errors="coerce")


def race_year(frame: pd.DataFrame) -> pd.Series:
    race_id_year = frame[RACE_COL].astype(str).str.slice(0, 4)
    year = pd.to_numeric(race_id_year, errors="coerce")
    if year.notna().any():
        return year.astype("Int64")
    raw_date = num(frame.get(DATE_COL), frame.index)
    return (2000 + np.floor(raw_date / 10000)).astype("Int64")


def max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    equity = pnl.cumsum()
    peak = equity.cummax()
    return float((peak - equity).max())


def add_scores(frame: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    with model_path.open("rb") as f:
        model = pickle.load(f)
    out = frame.copy()
    out["ai_score"] = model.predict(out)
    out["ai_rank"] = out.groupby(RACE_COL)["ai_score"].rank(ascending=False, method="first").astype(int)
    out["_race_second_score"] = out.groupby(RACE_COL)["ai_score"].transform(
        lambda s: s.sort_values(ascending=False).iloc[1] if len(s) > 1 else np.nan
    )
    out["ai_score_gap_to_second"] = (out["ai_score"] - out["_race_second_score"]).where(out["ai_rank"].eq(1), 0.0)
    out["year"] = race_year(out)
    out["popularity_num"] = num(out.get(POPULARITY_COL), out.index)
    out["odds_decimal"] = num(out.get(ODDS_COL), out.index)
    out["win_pay"] = num(out.get(WIN_PAY_COL), out.index, 0.0).fillna(0.0)
    return out


def ai1_table(scored: pd.DataFrame) -> pd.DataFrame:
    out = scored[scored["ai_rank"].eq(1)].copy()
    out = out[out[RACE_COL].notna()].copy()
    out["target_win"] = num(out.get("target_win"), out.index, 0.0).fillna(0.0).astype(int)
    out["target_top3"] = num(out.get("target_top3"), out.index, 0.0).fillna(0.0).astype(int)
    out["win_return"] = out["win_pay"].where(out["target_win"].eq(1), 0.0)
    out["pnl_win_flat100"] = out["win_return"] - 100.0
    out = out.sort_values(["year", DATE_COL, RACE_COL]).reset_index(drop=True)
    keep = [
        "year",
        DATE_COL,
        VENUE_COL,
        RACE_NO_COL,
        RACE_COL,
        HORSE_COL,
        "ai_score",
        "ai_score_gap_to_second",
        "target_win",
        "target_top3",
        "popularity_num",
        "odds_decimal",
        "win_pay",
        "win_return",
        "pnl_win_flat100",
    ]
    return out[[c for c in keep if c in out.columns]]


def metrics(frame: pd.DataFrame, label: str) -> dict[str, object]:
    bets = int(len(frame))
    if bets == 0:
        return {
            "label": label,
            "bets": 0,
            "races": 0,
            "win_rate": 0.0,
            "top3_rate": 0.0,
            "win_roi": 0.0,
            "profit_yen_flat100": 0.0,
            "max_drawdown_yen_flat100": 0.0,
        }
    stake = bets * 100.0
    ret = float(frame["win_return"].sum())
    pnl = frame["pnl_win_flat100"]
    return {
        "label": label,
        "bets": bets,
        "races": int(frame[RACE_COL].nunique()),
        "win_rate": float(frame["target_win"].mean()),
        "top3_rate": float(frame["target_top3"].mean()),
        "win_roi": ret / stake,
        "profit_yen_flat100": ret - stake,
        "max_drawdown_yen_flat100": max_drawdown(pnl),
        "avg_odds": float(frame["odds_decimal"].mean()),
        "median_odds": float(frame["odds_decimal"].median()),
        "avg_popularity": float(frame["popularity_num"].mean()),
        "avg_gap": float(frame["ai_score_gap_to_second"].mean()),
        "median_gap": float(frame["ai_score_gap_to_second"].median()),
    }


def odds_mask(frame: pd.DataFrame, odds_policy: str) -> pd.Series:
    odds = frame["odds_decimal"]
    policies = {
        "all": odds.notna(),
        "odds_1p5_6": odds.between(1.5, 6.0, inclusive="both"),
        "odds_1p5_10": odds.between(1.5, 10.0, inclusive="both"),
        "odds_2_8": odds.between(2.0, 8.0, inclusive="both"),
        "odds_2_12": odds.between(2.0, 12.0, inclusive="both"),
        "odds_3_15": odds.between(3.0, 15.0, inclusive="both"),
        "odds_5_30": odds.between(5.0, 30.0, inclusive="both"),
    }
    if odds_policy not in policies:
        raise ValueError(f"Unknown odds policy: {odds_policy}")
    return policies[odds_policy].fillna(False)


def evaluate_grid(ai1: pd.DataFrame, min_bets: int) -> pd.DataFrame:
    max_gap = float(ai1["ai_score_gap_to_second"].max())
    thresholds = sorted(
        set(
            [0.0]
            + [round(float(x), 3) for x in np.arange(0.005, min(max_gap + 0.005, 0.301), 0.005)]
            + [round(float(ai1["ai_score_gap_to_second"].quantile(q)), 6) for q in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]]
        )
    )
    rows: list[dict[str, object]] = []
    for odds_policy in ["all", "odds_1p5_6", "odds_1p5_10", "odds_2_8", "odds_2_12", "odds_3_15", "odds_5_30"]:
        om = odds_mask(ai1, odds_policy)
        for threshold in thresholds:
            selected = ai1[om & ai1["ai_score_gap_to_second"].ge(threshold)].copy()
            if len(selected) < min_bets:
                continue
            row = metrics(selected, f"gap_ge_{threshold:.6f}__{odds_policy}")
            row["gap_threshold"] = threshold
            row["odds_policy"] = odds_policy
            rows.append(row)
    return pd.DataFrame(rows)


def yearly_metrics(ai1: pd.DataFrame, policy_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, policy in policy_rows.iterrows():
        threshold = float(policy["gap_threshold"])
        om = odds_mask(ai1, str(policy["odds_policy"]))
        selected = ai1[om & ai1["ai_score_gap_to_second"].ge(threshold)].copy()
        for year, part in selected.groupby("year"):
            row = metrics(part, str(policy["label"]))
            row["year"] = int(year)
            row["gap_threshold"] = threshold
            row["odds_policy"] = str(policy["odds_policy"])
            rows.append(row)
    return pd.DataFrame(rows)


def walk_forward(ai1: pd.DataFrame, min_train_bets: int, min_valid_bets: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = sorted(int(y) for y in ai1["year"].dropna().unique())
    rows: list[dict[str, object]] = []
    selected_parts: list[pd.DataFrame] = []
    if len(years) < 2:
        return pd.DataFrame(), pd.DataFrame()
    for valid_year in years[1:]:
        train = ai1[ai1["year"].lt(valid_year)].copy()
        valid = ai1[ai1["year"].eq(valid_year)].copy()
        grid = evaluate_grid(train, min_train_bets)
        if grid.empty:
            continue
        # Choose a train threshold that is profitable, not too thin, and not driven only by hit-rate noise.
        ranked = grid.copy()
        ranked["selection_score"] = (
            ranked["win_roi"] * np.sqrt(np.maximum(ranked["win_rate"], 0.001)) * np.log1p(ranked["bets"])
            - ranked["max_drawdown_yen_flat100"] / 100000.0
        )
        ranked = ranked.sort_values(["selection_score", "win_roi", "bets"], ascending=[False, False, False])
        chosen = ranked.iloc[0]
        mask = odds_mask(valid, str(chosen["odds_policy"])) & valid["ai_score_gap_to_second"].ge(float(chosen["gap_threshold"]))
        valid_sel = valid[mask].copy()
        if len(valid_sel) < min_valid_bets:
            # Still record it, but mark it as thin.
            thin = True
        else:
            thin = False
        train_metric = metrics(train[odds_mask(train, str(chosen["odds_policy"])) & train["ai_score_gap_to_second"].ge(float(chosen["gap_threshold"]))], "train_selected")
        valid_metric = metrics(valid_sel, "valid_selected")
        row = {
            "valid_year": valid_year,
            "chosen_gap_threshold": float(chosen["gap_threshold"]),
            "chosen_odds_policy": str(chosen["odds_policy"]),
            "thin_valid": thin,
            **{f"train_{k}": v for k, v in train_metric.items() if k != "label"},
            **{f"valid_{k}": v for k, v in valid_metric.items() if k != "label"},
        }
        rows.append(row)
        valid_sel["valid_year"] = valid_year
        valid_sel["chosen_gap_threshold"] = float(chosen["gap_threshold"])
        valid_sel["chosen_odds_policy"] = str(chosen["odds_policy"])
        selected_parts.append(valid_sel)
    return pd.DataFrame(rows), pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AI1 score-gap thresholds for win-only betting.")
    parser.add_argument(
        "--test-csv",
        default="data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus_rebuilt_20260623/test_features.csv",
    )
    parser.add_argument(
        "--model",
        default="models/workout_lap_pedigree_interactions_confirmed_opponent_2023plus_rebuilt_20260623/baseline_ranker.pkl",
    )
    parser.add_argument("--output-dir", default="outputs/analysis/single_win_score_gap_threshold_v1")
    parser.add_argument("--min-bets", type=int, default=40)
    parser.add_argument("--min-train-bets", type=int, default=80)
    parser.add_argument("--min-valid-bets", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.test_csv, low_memory=False)
    scored = add_scores(df, Path(args.model))
    ai1 = ai1_table(scored)
    ai1.to_csv(out_dir / "ai1_scored_rows.csv", index=False, encoding="utf-8-sig")

    baseline = pd.DataFrame([metrics(ai1, "ai1_all")])
    grid = evaluate_grid(ai1, args.min_bets)
    if not grid.empty:
        grid["roi_over_baseline"] = grid["win_roi"] - float(baseline.iloc[0]["win_roi"])
        grid["robust_score"] = (
            grid["win_roi"] * np.sqrt(np.maximum(grid["win_rate"], 0.001)) * np.log1p(grid["bets"])
            - grid["max_drawdown_yen_flat100"] / 100000.0
        )
        grid = grid.sort_values(["robust_score", "win_roi", "bets"], ascending=[False, False, False])
    year = yearly_metrics(ai1, grid.head(30)) if not grid.empty else pd.DataFrame()
    wf_summary, wf_selected = walk_forward(ai1, args.min_train_bets, args.min_valid_bets)

    baseline.to_csv(out_dir / "baseline_ai1_all.csv", index=False, encoding="utf-8-sig")
    grid.to_csv(out_dir / "threshold_grid.csv", index=False, encoding="utf-8-sig")
    grid.head(50).to_csv(out_dir / "top_thresholds.csv", index=False, encoding="utf-8-sig")
    year.to_csv(out_dir / "top_thresholds_yearly.csv", index=False, encoding="utf-8-sig")
    wf_summary.to_csv(out_dir / "walk_forward_summary.csv", index=False, encoding="utf-8-sig")
    wf_selected.to_csv(out_dir / "walk_forward_selected_rows.csv", index=False, encoding="utf-8-sig")

    summary = {
        "input_csv": args.test_csv,
        "model": args.model,
        "rows": int(len(df)),
        "ai1_races": int(len(ai1)),
        "years": [int(y) for y in sorted(ai1["year"].dropna().unique())],
        "baseline": baseline.iloc[0].to_dict(),
        "top_thresholds": grid.head(10).to_dict(orient="records") if not grid.empty else [],
        "walk_forward": wf_summary.to_dict(orient="records"),
        "outputs": {
            "ai1_scored_rows": str(out_dir / "ai1_scored_rows.csv"),
            "threshold_grid": str(out_dir / "threshold_grid.csv"),
            "top_thresholds": str(out_dir / "top_thresholds.csv"),
            "top_thresholds_yearly": str(out_dir / "top_thresholds_yearly.csv"),
            "walk_forward_summary": str(out_dir / "walk_forward_summary.csv"),
            "walk_forward_selected_rows": str(out_dir / "walk_forward_selected_rows.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
