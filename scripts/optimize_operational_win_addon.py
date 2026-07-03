from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import ensure_dir, project_path


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def _num(series: pd.Series | None, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if series is None:
        if index is None:
            return pd.Series(dtype=float)
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _metrics(tickets: pd.DataFrame, label: str) -> dict:
    if tickets.empty:
        return {
            "policy": label,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi": 0.0,
            "ticket_hit_rate": 0.0,
            "race_hit_rate": 0.0,
            "max_drawdown_yen": 0.0,
        }

    t = tickets.copy()
    t["stake_yen"] = _num(t.get("stake_yen"), t.index, 0.0).fillna(0.0)
    t["return_yen"] = _num(t.get("return_yen"), t.index, 0.0).fillna(0.0)
    t["hit"] = t.get("hit", False).astype(bool)
    stake = float(t["stake_yen"].sum())
    ret = float(t["return_yen"].sum())

    by_race = (
        t.groupby("race_id", as_index=False)
        .agg(stake_yen=("stake_yen", "sum"), return_yen=("return_yen", "sum"), hit=("hit", "max"))
        .sort_values("race_id")
    )
    equity = (by_race["return_yen"] - by_race["stake_yen"]).cumsum()
    drawdown = equity - equity.cummax()

    out = {
        "policy": label,
        "tickets": int(len(t)),
        "races": int(t["race_id"].nunique()),
        "avg_stake_per_race": float(by_race["stake_yen"].mean()) if not by_race.empty else 0.0,
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake,
        "roi": ret / stake if stake else 0.0,
        "ticket_hit_rate": float(t["hit"].mean()) if len(t) else 0.0,
        "race_hit_rate": float(by_race["hit"].mean()) if len(by_race) else 0.0,
        "max_drawdown_yen": float(drawdown.min()) if not drawdown.empty else 0.0,
    }
    for ticket_type, g in t.groupby("ticket_type"):
        st = float(_num(g.get("stake_yen"), g.index, 0.0).sum())
        out[f"{ticket_type}_tickets"] = int(len(g))
        out[f"{ticket_type}_roi"] = float(_num(g.get("return_yen"), g.index, 0.0).sum() / st) if st else 0.0
        out[f"{ticket_type}_hit_rate"] = float(g["hit"].mean()) if len(g) else 0.0
    return out


def _load_base_tickets(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"race_id": str}, low_memory=False)
    if "operation_profile" in df.columns:
        df = df[~df["operation_profile"].astype(str).eq("skip")].copy()
    if "stake_yen" not in df.columns:
        df["stake_yen"] = 100.0
    if "return_yen" not in df.columns:
        pay = np.where(df.get("ticket_type", "").astype(str).eq("wide"), _num(df.get("wide_pay"), df.index, 0.0), _num(df.get("umaren_pay"), df.index, 0.0))
        df["return_yen"] = pay * df["stake_yen"] / 100.0
    if "hit" not in df.columns:
        df["hit"] = _num(df.get("return_yen"), df.index, 0.0).gt(0)
    df["ticket_type"] = df.get("ticket_type", "").astype(str)
    df["year"] = pd.to_numeric(df["race_id"].str.slice(0, 4), errors="coerce")
    return df


def _win_candidates(scored: pd.DataFrame) -> pd.DataFrame:
    df = scored.copy()
    df["race_id"] = df["race_id"].astype(str)
    df["horse_no"] = _num(df.get("horse_no"), df.index, 0).astype("Int64")
    df["year"] = pd.to_numeric(df.get("year", df["race_id"].str.slice(0, 4)), errors="coerce")
    df["ai_rank_num"] = _num(df.get("ai_rank_num"), df.index, 999.0).fillna(999.0)
    df["pop_rank_num"] = _num(df.get("pop_rank_num"), df.index, 999.0).fillna(999.0)
    df["odds"] = _num(df.get("market_odds_live_or_final"), df.index, np.nan).fillna(_num(df.get("odds_num"), df.index, np.nan))
    df["win_score"] = _num(df.get("win_suitability_score"), df.index, 0.0).fillna(0.0)
    df["win_prob"] = _num(df.get("ai_win_prob_proxy"), df.index, 0.0).fillna(0.0)
    df["win_ev"] = _num(df.get("win_ev_proxy"), df.index, 0.0).fillna(0.0)
    df["overlay"] = _num(df.get("market_overlay_score"), df.index, 0.0).fillna(0.0)
    df["danger"] = _num(df.get("danger_popular_hybrid_score"), df.index, np.nan).fillna(
        _num(df.get("danger_favorite_score"), df.index, 0.0)
    )
    df["skip_risk"] = _num(df.get("skip_risk_score"), df.index, 0.0).fillna(0.0)
    df["difficulty"] = _num(df.get("race_difficulty_model_score"), df.index, np.nan).fillna(
        _num(df.get("target_race_difficulty"), df.index, 0.5)
    )
    df["is_win"] = _num(df.get("is_win"), df.index, 0.0).fillna(0.0).gt(0)
    df["win_pay"] = _num(df.get("win_pay"), df.index, np.nan).fillna(_num(df.get("単勝配当"), df.index, 0.0)).fillna(0.0)
    df["win_return_base100"] = np.where(df["is_win"], df["win_pay"], 0.0)
    df["_candidate_order"] = (
        df["win_ev"].rank(method="first", ascending=False)
        + df["win_score"].rank(method="first", ascending=False) / 100000.0
        + df["overlay"].rank(method="first", ascending=False) / 1000000000.0
    )
    return df


def _grid() -> list[dict]:
    rows: list[dict] = []
    for ai_rank_max, score_min, ev_min, overlay_min, odds_min, odds_max, danger_max, diff_max, skip_max in product(
        [1, 2],
        [0.70, 0.80],
        [1.05, 1.20],
        [0.55],
        [1.8, 3.0],
        [12.0, 80.0],
        [0.50],
        [0.75, 1.01],
        [0.80],
    ):
        if odds_min >= odds_max:
            continue
        rows.append(
            {
                "ai_rank_max": ai_rank_max,
                "score_min": score_min,
                "ev_min": ev_min,
                "overlay_min": overlay_min,
                "odds_min": odds_min,
                "odds_max": odds_max,
                "danger_max": danger_max,
                "diff_max": diff_max,
                "skip_max": skip_max,
            }
        )
    return rows


def _select_wins(candidates: pd.DataFrame, params: dict, stake_yen: int, max_wins_per_race: int) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    c = candidates.copy()
    mask = (
        c["ai_rank_num"].le(params["ai_rank_max"])
        & c["win_score"].ge(params["score_min"])
        & c["win_ev"].ge(params["ev_min"])
        & c["overlay"].ge(params["overlay_min"])
        & c["odds"].between(params["odds_min"], params["odds_max"])
        & c["danger"].le(params["danger_max"])
        & c["difficulty"].le(params["diff_max"])
        & c["skip_risk"].le(params["skip_max"])
    )
    selected = c[mask].copy()
    if selected.empty:
        return selected
    selected = selected.sort_values(["race_id", "_candidate_order"], ascending=[True, True]).groupby("race_id", as_index=False).head(max_wins_per_race)
    selected["ticket_type"] = "win"
    selected["stake_yen"] = float(stake_yen)
    selected["return_yen"] = selected["win_return_base100"] * selected["stake_yen"] / 100.0
    selected["hit"] = selected["is_win"]
    selected["operation_profile"] = "win_addon"
    selected["operation_profile_label"] = "単勝追加"
    selected["operation_strength_rank"] = 3
    selected["anchor_no"] = selected["horse_no"]
    selected["anchor_name"] = selected.get("horse_name", "").astype(str)
    selected["partner_no"] = pd.NA
    selected["partner_name"] = ""
    selected["pair_quinella_score"] = selected["win_score"]
    selected["race_difficulty_score"] = selected["difficulty"]
    return selected


def _choose_policy(
    train: pd.DataFrame,
    min_train_races: int,
    min_hit_rate: float,
    stake_yen: int,
    max_wins_per_race: int,
) -> tuple[dict | None, dict | None]:
    best_params: dict | None = None
    best_metrics: dict | None = None
    best_score = -np.inf
    for params in _grid():
        selected = _select_wins(train, params, stake_yen, max_wins_per_race)
        if selected.empty:
            continue
        races = int(selected["race_id"].nunique())
        hit_rate = float(selected["hit"].mean())
        if races < min_train_races or hit_rate < min_hit_rate:
            continue
        stake = float(len(selected) * stake_yen)
        ret = float(_num(selected.get("return_yen"), selected.index, 0.0).sum())
        by_race = selected.groupby("race_id", as_index=False).agg(stake_yen=("stake_yen", "sum"), return_yen=("return_yen", "sum"))
        equity = (by_race["return_yen"] - by_race["stake_yen"]).cumsum()
        drawdown = equity - equity.cummax()
        metrics = {
            "policy": "train_win_addon",
            "tickets": int(len(selected)),
            "races": races,
            "stake_yen": stake,
            "return_yen": ret,
            "profit_yen": ret - stake,
            "roi": ret / stake if stake else 0.0,
            "ticket_hit_rate": hit_rate,
            "race_hit_rate": hit_rate,
            "max_drawdown_yen": float(drawdown.min()) if not drawdown.empty else 0.0,
        }
        score = (
            (metrics["roi"] - 1.0) * 100.0
            + metrics["race_hit_rate"] * 12.0
            + np.log1p(metrics["races"]) * 0.45
            - max(0.0, abs(metrics["max_drawdown_yen"]) / 10000.0) * 0.15
        )
        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = metrics
    return best_params, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored-csv", default="outputs/analysis/risk_models_v1/investment_features_with_risk_models.csv")
    parser.add_argument("--base-tickets-csv", default="outputs/analysis/operational_ticket_profiles_v1/ticket_profiles.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/operational_win_addon_v1")
    parser.add_argument("--stake-yen", type=int, default=100)
    parser.add_argument("--max-wins-per-race", type=int, default=1)
    parser.add_argument("--min-train-races", type=int, default=120)
    parser.add_argument("--min-hit-rate", type=float, default=0.08)
    args = parser.parse_args()

    out_dir = ensure_dir(project_path(args.output_dir))
    scored = pd.read_csv(project_path(args.scored_csv), dtype={"race_id": str}, low_memory=False)
    base = _load_base_tickets(project_path(args.base_tickets_csv))
    candidates = _win_candidates(scored)

    years = sorted(int(y) for y in candidates["year"].dropna().unique())
    wf_rows: list[dict] = []
    ticket_frames: list[pd.DataFrame] = []

    for year in years[1:]:
        train = candidates[candidates["year"].lt(year)].copy()
        test = candidates[candidates["year"].eq(year)].copy()
        params, train_metrics = _choose_policy(
            train,
            args.min_train_races,
            args.min_hit_rate,
            args.stake_yen,
            args.max_wins_per_race,
        )
        if params is None:
            wf_rows.append({"year": year, "selected": False})
            continue
        test_selected = _select_wins(test, params, args.stake_yen, args.max_wins_per_race)
        test_metrics = _metrics(test_selected, f"test_{year}_win_addon")
        ticket_frames.append(test_selected.assign(test_year=year))
        wf_rows.append(
            {
                "year": year,
                "selected": True,
                **{f"param_{k}": v for k, v in params.items()},
                **{f"train_{k}": v for k, v in (train_metrics or {}).items() if k != "policy"},
                **{f"test_{k}": v for k, v in test_metrics.items() if k != "policy"},
            }
        )

    win_tickets = pd.concat(ticket_frames, ignore_index=True, sort=False) if ticket_frames else pd.DataFrame()
    test_years = sorted(win_tickets["year"].dropna().astype(int).unique()) if not win_tickets.empty else years[1:]
    base_test = base[base["year"].isin(test_years)].copy() if test_years else base.iloc[0:0].copy()
    combined = pd.concat([base_test, win_tickets], ignore_index=True, sort=False)

    wf = pd.DataFrame(wf_rows)
    summary = {
        "input": {
            "scored_csv": args.scored_csv,
            "base_tickets_csv": args.base_tickets_csv,
            "stake_yen": args.stake_yen,
            "max_wins_per_race": args.max_wins_per_race,
            "min_train_races": args.min_train_races,
            "min_hit_rate": args.min_hit_rate,
            "test_years": test_years,
        },
        "base_operational": _metrics(base_test, "base_operational"),
        "win_addon": _metrics(win_tickets, "win_addon"),
        "combined": _metrics(combined, "base_plus_win_addon"),
    }
    summary["delta_roi"] = summary["combined"]["roi"] - summary["base_operational"]["roi"]
    summary["delta_profit_yen"] = summary["combined"]["profit_yen"] - summary["base_operational"]["profit_yen"]

    wf.to_csv(out_dir / "walkforward_win_summary.csv", index=False, encoding="utf-8-sig")
    win_tickets.to_csv(out_dir / "win_addon_tickets.csv", index=False, encoding="utf-8-sig")
    combined.to_csv(out_dir / "combined_ticket_profiles.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [summary["base_operational"], summary["win_addon"], summary["combined"]]
    ).to_csv(out_dir / "metrics.csv", index=False, encoding="utf-8-sig")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
