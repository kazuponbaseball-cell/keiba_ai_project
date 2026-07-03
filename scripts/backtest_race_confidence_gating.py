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


def _safe_col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return _num(df[name]).fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def _normalize(series: pd.Series) -> pd.Series:
    s = _num(series).replace([np.inf, -np.inf], np.nan)
    lo = s.quantile(0.05)
    hi = s.quantile(0.95)
    if pd.isna(lo) or pd.isna(hi) or hi <= lo:
        return pd.Series(0.5, index=series.index)
    return ((s.clip(lo, hi) - lo) / (hi - lo)).fillna(0.5)


def _parse_date(series: pd.Series) -> pd.Series:
    raw = series.astype(str).str.replace(r"\D", "", regex=True)
    return pd.to_datetime(raw.str[:8], format="%Y%m%d", errors="coerce")


def _max_drawdown_by_race(frame: pd.DataFrame, stake_col: str, return_col: str) -> float:
    if frame.empty:
        return 0.0
    ordered = frame.sort_values(["race_date", "race_id"]).copy()
    pnl = ordered[return_col].fillna(0.0) - ordered[stake_col].fillna(0.0)
    equity = pnl.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    return float(drawdown.min())


def _metrics(frame: pd.DataFrame, label: str, stake_col: str, return_col: str) -> dict:
    if frame.empty:
        return {
            "policy": label,
            "races": 0,
            "stake_yen": 0,
            "return_yen": 0,
            "profit_yen": 0,
            "roi": 0.0,
            "win_rate": 0.0,
            "place_rate": 0.0,
            "max_drawdown_yen": 0.0,
            "avg_confidence": 0.0,
            "avg_pop": 0.0,
            "avg_odds": 0.0,
        }
    stake = float(frame[stake_col].sum())
    ret = float(frame[return_col].sum())
    return {
        "policy": label,
        "races": int(frame["race_id"].nunique()),
        "stake_yen": int(round(stake)),
        "return_yen": int(round(ret)),
        "profit_yen": int(round(ret - stake)),
        "roi": float(ret / stake) if stake else 0.0,
        "win_rate": float(frame["is_win"].mean()),
        "place_rate": float(frame["is_place"].mean()),
        "max_drawdown_yen": int(round(_max_drawdown_by_race(frame, stake_col, return_col))),
        "avg_confidence": float(frame["race_confidence_score"].mean()),
        "avg_pop": float(frame["pop_rank_num"].mean()),
        "avg_odds": float(frame["odds_num"].mean()),
    }


def _yearly_metrics(frame: pd.DataFrame, label: str, stake_col: str, return_col: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for year, g in frame.groupby("year"):
        row = _metrics(g, label, stake_col, return_col)
        row["year"] = int(year) if pd.notna(year) else -1
        rows.append(row)
    return pd.DataFrame(rows).sort_values("year")


def _prepare_top1(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "ai_score_gap_to_second" not in df.columns:
        score = _num(df["ai_score"])
        second = (
            df.assign(_score=score)
            .sort_values(["race_id", "_score"], ascending=[True, False])
            .groupby("race_id")["_score"]
            .nth(1)
        )
        df["_second_score"] = df["race_id"].map(second)
        df["ai_score_gap_to_second"] = _num(df["ai_score"]) - _num(df["_second_score"])

    top1 = df[_num(df["ai_rank_num"]).eq(1)].copy()
    top1["race_date"] = _parse_date(top1["日付S"]) if "日付S" in top1.columns else pd.NaT
    top1["year"] = top1["race_date"].dt.year

    top1["field_size_num"] = _safe_col(top1, "出走頭数", np.nan)
    if top1["field_size_num"].isna().all():
        top1["field_size_num"] = _safe_col(top1, "頭数", np.nan)
    top1["frame_num"] = _safe_col(top1, "枠番", np.nan)

    gap = _normalize(top1["ai_score_gap_to_second"])
    odds = _num(top1["odds_num"]).fillna(top1["odds_num"].median())
    pop = _num(top1["pop_rank_num"]).fillna(top1["pop_rank_num"].median())
    field = _num(top1["field_size_num"]).fillna(top1["field_size_num"].median())
    pressure = _safe_col(top1, "race_early_pressure_score", 0.0)
    front_count = _safe_col(top1, "race_front_runner_count_y", 0.0)
    if front_count.eq(0).all():
        front_count = _safe_col(top1, "race_front_runner_count_x", 0.0)
    pace_fit = _safe_col(top1, "pace_fit_score", 0.0)
    draw_fit = _safe_col(top1, "draw_pace_fit_score", 0.0)
    bias_fit = _safe_col(top1, "same_day_pop_adjusted_pace_fit_score", 0.0)

    # This is intentionally conservative and uses mostly pre-race style inputs.
    confidence = 0.0
    confidence += 2.0 * gap
    confidence += 0.35 * _normalize(pace_fit)
    confidence += 0.25 * _normalize(draw_fit)
    confidence += 0.20 * _normalize(bias_fit)
    confidence += np.where(pop.le(3), 0.25, 0.0)
    confidence += np.where(odds.between(2.0, 8.0), 0.15, 0.0)
    confidence -= np.where(odds.lt(1.8), 0.25, 0.0)
    confidence -= np.where(odds.gt(15.0), 0.25, 0.0)
    confidence -= np.where(field.ge(16), 0.30, 0.0)
    confidence -= np.where(field.le(7), 0.10, 0.0)
    confidence -= np.where(top1.get("expected_pace", "").astype(str).eq("fast"), 0.25, 0.0)
    confidence -= np.where(pressure.gt(0.65), 0.20, 0.0)
    confidence -= np.where(front_count.ge(4), 0.15, 0.0)
    confidence -= np.where(top1.get("馬場状態", "").astype(str).isin(["重", "不"]), 0.25, 0.0)
    confidence -= np.where(top1.get("venue", "").astype(str).isin(["札幌", "小倉"]), 0.15, 0.0)
    confidence -= np.where(top1.get("class_group", "").astype(str).eq("open"), 0.20, 0.0)

    top1["race_confidence_score_raw"] = confidence
    top1["race_confidence_score"] = _normalize(top1["race_confidence_score_raw"])
    top1["confidence_pct_rank"] = top1["race_confidence_score"].rank(pct=True, ascending=False, method="first")

    top1["win_stake"] = 100.0
    top1["place_stake"] = 100.0
    top1["win_return_100"] = _num(top1["win_return"]).fillna(0.0)
    top1["place_return_100"] = _num(top1["place_return"]).fillna(0.0)
    return top1


def _top_pct_backtest(top1: pd.DataFrame, pcts: list[float], bet_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    yearly = []
    ret_col = "win_return_100" if bet_type == "win" else "place_return_100"
    stake_col = "stake_100"
    for pct in pcts:
        selected = top1[top1["confidence_pct_rank"].le(pct)].copy()
        selected[stake_col] = 100.0
        label = f"{bet_type}_top{int(pct * 100)}pct"
        rows.append(_metrics(selected, label, stake_col, ret_col))
        y = _yearly_metrics(selected, label, stake_col, ret_col)
        if not y.empty:
            yearly.append(y)
    return pd.DataFrame(rows), pd.concat(yearly, ignore_index=True, sort=False) if yearly else pd.DataFrame()


def _tiered_backtest(top1: pd.DataFrame, bet_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ret_base = "win_return_100" if bet_type == "win" else "place_return_100"
    policies = [
        ("tier_20_50_70", 0.20, 0.50, 0.70, 500.0, 200.0, 50.0),
        ("tier_30_60_80", 0.30, 0.60, 0.80, 400.0, 150.0, 50.0),
        ("tier_40_70_90", 0.40, 0.70, 0.90, 300.0, 100.0, 30.0),
    ]
    rows = []
    yearly = []
    for name, a, b, c, stake_a, stake_b, stake_c in policies:
        selected = top1[top1["confidence_pct_rank"].le(c)].copy()
        selected["tier"] = np.select(
            [
                selected["confidence_pct_rank"].le(a),
                selected["confidence_pct_rank"].le(b),
                selected["confidence_pct_rank"].le(c),
            ],
            ["A", "B", "C"],
            default="skip",
        )
        selected["tier_stake"] = np.select(
            [selected["tier"].eq("A"), selected["tier"].eq("B"), selected["tier"].eq("C")],
            [stake_a, stake_b, stake_c],
            default=0.0,
        )
        selected["tier_return"] = selected[ret_base] * selected["tier_stake"] / 100.0
        label = f"{bet_type}_{name}"
        rows.append(_metrics(selected, label, "tier_stake", "tier_return"))
        by_tier = selected.groupby("tier", as_index=False).apply(
            lambda g: pd.Series(_metrics(g, f"{label}_{g.name}", "tier_stake", "tier_return"))
        )
        for _, row in by_tier.iterrows():
            d = row.to_dict()
            d["policy"] = f"{label}_{d.get('tier', '')}"
            rows.append(d)
        y = _yearly_metrics(selected, label, "tier_stake", "tier_return")
        if not y.empty:
            yearly.append(y)
    return pd.DataFrame(rows), pd.concat(yearly, ignore_index=True, sort=False) if yearly else pd.DataFrame()


def _value_stake_backtest(top1: pd.DataFrame, bet_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ret_base = "win_return_100" if bet_type == "win" else "place_return_100"
    danger = (
        top1.get("馬場状態", "").astype(str).isin(["重", "不"])
        | top1.get("expected_pace", "").astype(str).eq("fast")
        | _num(top1["field_size_num"]).ge(16)
        | top1.get("venue", "").astype(str).isin(["札幌", "小倉"])
        | top1.get("class_group", "").astype(str).eq("open")
    )
    policies = {
        "pop2_9_scaled": [
            (top1["pop_rank_num"].between(4, 9), 500.0),
            (top1["pop_rank_num"].between(2, 3), 200.0),
        ],
        "pop2_9_scaled_no_danger": [
            (top1["pop_rank_num"].between(4, 9) & ~danger, 500.0),
            (top1["pop_rank_num"].between(2, 3) & ~danger, 200.0),
        ],
        "broad_1thin_value_thick": [
            (top1["pop_rank_num"].eq(1) & ~danger, 50.0),
            (top1["pop_rank_num"].between(2, 3) & ~danger, 150.0),
            (top1["pop_rank_num"].between(4, 9) & ~danger, 500.0),
        ],
        "broad_include_danger_thin": [
            (top1["pop_rank_num"].eq(1), 30.0),
            (top1["pop_rank_num"].between(2, 3), 150.0),
            (top1["pop_rank_num"].between(4, 9), 400.0),
        ],
    }
    rows = []
    yearly = []
    for name, rules in policies.items():
        selected = top1.copy()
        stake = pd.Series(0.0, index=selected.index)
        for condition, amount in rules:
            stake = pd.Series(np.where(condition, np.maximum(stake, amount), stake), index=selected.index)
        selected["value_stake"] = stake
        selected = selected[selected["value_stake"].gt(0)].copy()
        selected["value_return"] = selected[ret_base] * selected["value_stake"] / 100.0
        label = f"{bet_type}_{name}"
        rows.append(_metrics(selected, label, "value_stake", "value_return"))
        y = _yearly_metrics(selected, label, "value_stake", "value_return")
        if not y.empty:
            yearly.append(y)
    return pd.DataFrame(rows), pd.concat(yearly, ignore_index=True, sort=False) if yearly else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest race confidence gating without over-squeezing race count.")
    parser.add_argument("--input-csv", default="outputs/analysis/roi_stagnation_drivers_v1/prediction_detail_enriched.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/race_confidence_gating_v1")
    args = parser.parse_args()

    top1 = _prepare_top1(project_path(args.input_csv))
    out_dir = ensure_dir(project_path(args.output_dir))

    pcts = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00]
    win_pct, win_year = _top_pct_backtest(top1, pcts, "win")
    place_pct, place_year = _top_pct_backtest(top1, pcts, "place")
    win_tier, win_tier_year = _tiered_backtest(top1, "win")
    place_tier, place_tier_year = _tiered_backtest(top1, "place")
    win_value, win_value_year = _value_stake_backtest(top1, "win")
    place_value, place_value_year = _value_stake_backtest(top1, "place")

    top1.to_csv(out_dir / "race_confidence_scored_top1.csv", index=False, encoding="utf-8-sig")
    win_pct.to_csv(out_dir / "top_pct_win_summary.csv", index=False, encoding="utf-8-sig")
    place_pct.to_csv(out_dir / "top_pct_place_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat([win_year, place_year], ignore_index=True, sort=False).to_csv(out_dir / "top_pct_yearly.csv", index=False, encoding="utf-8-sig")
    win_tier.to_csv(out_dir / "tiered_win_summary.csv", index=False, encoding="utf-8-sig")
    place_tier.to_csv(out_dir / "tiered_place_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat([win_tier_year, place_tier_year], ignore_index=True, sort=False).to_csv(out_dir / "tiered_yearly.csv", index=False, encoding="utf-8-sig")
    win_value.to_csv(out_dir / "value_stake_win_summary.csv", index=False, encoding="utf-8-sig")
    place_value.to_csv(out_dir / "value_stake_place_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat([win_value_year, place_value_year], ignore_index=True, sort=False).to_csv(out_dir / "value_stake_yearly.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "input_csv": str(project_path(args.input_csv)),
        "note": "Confidence score avoids post-race PCI/RPCI and is designed as a practical bet-sizing gate, not a new prediction feature.",
        "top_pct_win": win_pct.to_dict(orient="records"),
        "top_pct_place": place_pct.to_dict(orient="records"),
        "tiered_win": win_tier.to_dict(orient="records"),
        "tiered_place": place_tier.to_dict(orient="records"),
        "value_stake_win": win_value.to_dict(orient="records"),
        "value_stake_place": place_value.to_dict(orient="records"),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
