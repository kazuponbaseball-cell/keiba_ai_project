from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


UNIVERSE = Path("outputs/analysis/dynamic_pair_ticket_allocation_quinella_model_v1/pair_candidate_universe.csv")
RACES = Path("data/processed/normalized/races.csv")
OUT = Path("outputs/analysis/sparse_roi_strategy_v1")


VENUE_CODE = {
    "01": "Sapporo",
    "02": "Hakodate",
    "03": "Fukushima",
    "04": "Niigata",
    "05": "Tokyo",
    "06": "Nakayama",
    "07": "Chukyo",
    "08": "Kyoto",
    "09": "Hanshin",
    "10": "Kokura",
}


def num(s: pd.Series | None, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if s is None:
        if index is None:
            raise ValueError("index required")
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(s, errors="coerce")


def clip01(s: pd.Series) -> pd.Series:
    return num(s).fillna(0.0).clip(0.0, 1.0)


def max_drawdown(profit: pd.Series) -> float:
    if profit.empty:
        return 0.0
    equity = profit.cumsum()
    dd = equity - equity.cummax()
    return float(dd.min())


def load_universe() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE, dtype={"race_id": str}, low_memory=False)
    df["race_id"] = df["race_id"].astype(str)
    df["year"] = df["race_id"].str[:4].astype(int)
    df["venue"] = df["race_id"].str.zfill(16).str[8:10].map(VENUE_CODE).fillna("Unknown")

    races = pd.read_csv(RACES, dtype=str, low_memory=False)
    info = races.iloc[:, [0, 7, 12]].copy()
    info.columns = ["race_id", "surface_raw", "going_raw"]
    info = info.drop_duplicates("race_id")
    going_map = {"\u826f": "Good", "\u7a0d": "Yielding", "\u7a0d\u91cd": "Yielding", "\u91cd": "Soft", "\u4e0d": "Heavy", "\u4e0d\u826f": "Heavy"}
    df = df.merge(info, on="race_id", how="left")
    df["going"] = df["going_raw"].map(going_map).fillna("Unknown")

    idx = df.index
    for col in [
        "wide_axis_score",
        "wide_partner_score",
        "projected_front5_prob",
        "market_overlay_score",
        "late_value_survives_score",
        "pair_score",
        "pair_quinella_score",
        "anchor_quinella_score",
        "partner_quinella_score",
        "anchor_danger",
        "partner_danger",
        "anchor_odds",
        "partner_odds",
        "wide_pay",
        "umaren_pay",
    ]:
        df[col] = num(df.get(col), idx, 0.0).fillna(0.0)
    df["wide_hit"] = df["wide_hit"].astype(bool)
    df["umaren_hit"] = df["umaren_hit"].astype(bool)

    # Pre-race proxy. This intentionally avoids using actual pair payoffs as a filter.
    df["wide_quote_proxy"] = 100.0 * (np.sqrt((df["anchor_odds"].clip(1.0) * df["partner_odds"].clip(1.0))) * 0.45).clip(1.1, 120.0)
    df["umaren_quote_proxy"] = 100.0 * (df["anchor_odds"].clip(1.0) * df["partner_odds"].clip(1.0) * 0.32).clip(1.3, 260.0)
    df["sparse_score"] = (
        0.20 * clip01(df["pair_quinella_score"])
        + 0.18 * clip01(df["pair_score"])
        + 0.18 * clip01(df["market_overlay_score"])
        + 0.17 * clip01(df["late_value_survives_score"])
        + 0.16 * clip01(df["projected_front5_prob"])
        + 0.08 * clip01(df["partner_quinella_score"])
        + 0.05 * num(df["partner_odds"]).ge(8.0).astype(float)
        - 0.10 * clip01(df["anchor_danger"])
        - 0.12 * clip01(df["partner_danger"])
    ).clip(0.0, 1.0)
    return df


def grid() -> list[dict]:
    rows: list[dict] = []
    venue_policies = {
        "skip_hakodate": set(VENUE_CODE.values()) - {"Hakodate"},
        "positive_venues": {"Fukushima", "Niigata", "Chukyo", "Hanshin", "Kokura", "Tokyo"},
    }
    going_policies = {
        "skip_heavy": {"Good", "Yielding", "Soft", "Unknown"},
        "skip_soft_heavy": {"Good", "Yielding", "Unknown"},
    }
    for (
        venue_policy,
        going_policy,
        score_q,
        axis_min,
        partner_min,
        front_min,
        market_min,
        late_min,
        pair_q_min,
        partner_odds_min,
        partner_odds_max,
        anchor_danger_max,
        partner_danger_max,
        pairs_per_race,
        ticket_mode,
    ) in product(
        venue_policies.keys(),
        going_policies.keys(),
        [0.90, 0.95, 0.975],
        [0.62],
        [0.60],
        [0.45, 0.60],
        [0.45, 0.60],
        [0.45],
        [0.56, 0.64],
        [6.0, 10.0, 12.0],
        [40.0],
        [0.55],
        [0.35],
        [1],
        ["wide_only", "wide_umaren"],
    ):
        rows.append(
            {
                "venue_policy": venue_policy,
                "venue_allowed": venue_policies[venue_policy],
                "going_policy": going_policy,
                "going_allowed": going_policies[going_policy],
                "score_q": score_q,
                "axis_min": axis_min,
                "partner_min": partner_min,
                "front_min": front_min,
                "market_min": market_min,
                "late_min": late_min,
                "pair_q_min": pair_q_min,
                "partner_odds_min": partner_odds_min,
                "partner_odds_max": partner_odds_max,
                "anchor_danger_max": anchor_danger_max,
                "partner_danger_max": partner_danger_max,
                "pairs_per_race": pairs_per_race,
                "ticket_mode": ticket_mode,
                "wide_stake": 200.0,
                "umaren_stake": 100.0 if ticket_mode == "wide_umaren" else 0.0,
                "umaren_pair_q_min": max(pair_q_min, 0.60),
                "umaren_quote_min": 900.0,
            }
        )
    return rows


def select_pairs(df: pd.DataFrame, params: dict, score_threshold: float) -> pd.DataFrame:
    mask = (
        df["venue"].isin(params["venue_allowed"])
        & df["going"].isin(params["going_allowed"])
        & df["sparse_score"].ge(score_threshold)
        & df["wide_axis_score"].ge(params["axis_min"])
        & df["wide_partner_score"].ge(params["partner_min"])
        & df["projected_front5_prob"].ge(params["front_min"])
        & df["market_overlay_score"].ge(params["market_min"])
        & df["late_value_survives_score"].ge(params["late_min"])
        & df["pair_quinella_score"].ge(params["pair_q_min"])
        & df["partner_odds"].between(params["partner_odds_min"], params["partner_odds_max"])
        & df["anchor_danger"].le(params["anchor_danger_max"])
        & df["partner_danger"].le(params["partner_danger_max"])
    )
    selected = df[mask].copy()
    if selected.empty:
        return selected
    return (
        selected.sort_values(["race_id", "sparse_score", "pair_quinella_score", "market_overlay_score"], ascending=[True, False, False, False])
        .groupby("race_id", as_index=False)
        .head(int(params["pairs_per_race"]))
    )


def tickets_from_pairs(pairs: pd.DataFrame, params: dict) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    base = pairs.copy()
    base["pair_key"] = base["race_id"] + ":" + base["anchor_no"].astype(str) + "-" + base["partner_no"].astype(str)

    wide = base.copy()
    wide["ticket_type"] = "wide"
    wide["stake_yen"] = params["wide_stake"]
    wide["hit"] = wide["wide_hit"].astype(bool)
    wide["return_yen"] = wide["wide_pay"].where(wide["hit"], 0.0) * wide["stake_yen"] / 100.0
    frames.append(wide)

    if params["umaren_stake"] > 0:
        mask = (
            base["pair_quinella_score"].ge(params["umaren_pair_q_min"])
            & base["umaren_quote_proxy"].ge(params["umaren_quote_min"])
            & base["partner_odds"].le(25.0)
        )
        umaren = base[mask].copy()
        if not umaren.empty:
            umaren["ticket_type"] = "umaren"
            umaren["stake_yen"] = params["umaren_stake"]
            umaren["hit"] = umaren["umaren_hit"].astype(bool)
            umaren["return_yen"] = umaren["umaren_pay"].where(umaren["hit"], 0.0) * umaren["stake_yen"] / 100.0
            frames.append(umaren)

    out = pd.concat(frames, ignore_index=True, sort=False)
    out["year"] = out["race_id"].str[:4].astype(int)
    out["ticket_key"] = out["ticket_type"] + ":" + out["pair_key"]
    return out


def metrics(tickets: pd.DataFrame, label: str) -> dict:
    if tickets.empty:
        return {
            "label": label,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi": 0.0,
            "race_hit_rate": 0.0,
            "max_drawdown_yen": 0.0,
        }
    stake = float(tickets["stake_yen"].sum())
    ret = float(tickets["return_yen"].sum())
    by_race = tickets.groupby("race_id", sort=False).agg(stake=("stake_yen", "sum"), ret=("return_yen", "sum"), hit=("hit", "max"))
    profit = by_race["ret"] - by_race["stake"]
    return {
        "label": label,
        "tickets": int(len(tickets)),
        "races": int(tickets["race_id"].nunique()),
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake,
        "roi": ret / stake if stake else 0.0,
        "ticket_hit_rate": float(tickets["hit"].mean()),
        "race_hit_rate": float(by_race["hit"].mean()),
        "max_drawdown_yen": max_drawdown(profit),
        "wide_tickets": int((tickets["ticket_type"] == "wide").sum()),
        "umaren_tickets": int((tickets["ticket_type"] == "umaren").sum()),
    }


def evaluate(df: pd.DataFrame, params: dict, score_threshold: float, label: str) -> tuple[dict, pd.DataFrame]:
    pairs = select_pairs(df, params, score_threshold)
    tickets = tickets_from_pairs(pairs, params)
    return metrics(tickets, label), tickets


def policy_score(m: dict) -> float:
    if m["races"] < 25:
        return -1e9
    # Target sparse, high ROI, but reject one-hit tiny policies.
    return float(m["roi"]) * np.sqrt(max(float(m["race_hit_rate"]), 0.001)) * np.log1p(float(m["races"])) + float(m["profit_yen"]) / 100000.0


def walkforward_search(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    params_grid = grid()
    all_train_rows: list[pd.DataFrame] = []
    wf_rows: list[dict] = []
    ticket_frames: list[pd.DataFrame] = []
    for test_year in [2025, 2026]:
        train = df[df["year"] < test_year].copy()
        test = df[df["year"] == test_year].copy()
        scored: list[dict] = []
        for i, params in enumerate(params_grid):
            threshold = float(train["sparse_score"].quantile(params["score_q"]))
            m, _ = evaluate(train, params, threshold, f"train_{test_year}_grid_{i}")
            m.update({k: v for k, v in params.items() if k not in {"venue_allowed", "going_allowed"}})
            m["score_threshold"] = threshold
            m["grid_id"] = i
            m["test_year"] = test_year
            m["selection_score"] = policy_score(m)
            scored.append(m)
        train_grid = pd.DataFrame(scored).sort_values(["selection_score", "roi"], ascending=[False, False])
        all_train_rows.append(train_grid.head(100))
        best = train_grid.iloc[0].to_dict()
        best_params = params_grid[int(best["grid_id"])]
        threshold = float(best["score_threshold"])
        test_m, test_t = evaluate(test, best_params, threshold, f"wf_test_{test_year}")
        test_m.update({k: v for k, v in best_params.items() if k not in {"venue_allowed", "going_allowed"}})
        test_m["score_threshold"] = threshold
        test_m["test_year"] = test_year
        test_m["train_roi"] = float(best["roi"])
        test_m["train_races"] = int(best["races"])
        test_m["train_profit_yen"] = float(best["profit_yen"])
        wf_rows.append(test_m)
        if not test_t.empty:
            tmp = test_t.copy()
            tmp["test_year"] = test_year
            tmp["selected_grid_id"] = int(best["grid_id"])
            ticket_frames.append(tmp)
    return (
        pd.concat(all_train_rows, ignore_index=True, sort=False),
        pd.DataFrame(wf_rows),
        pd.concat(ticket_frames, ignore_index=True, sort=False) if ticket_frames else pd.DataFrame(),
    )


def top_train_apply_2025_2026(df: pd.DataFrame) -> pd.DataFrame:
    train = df[df["year"] == 2024].copy()
    test = df[df["year"].isin([2025, 2026])].copy()
    rows: list[dict] = []
    for i, params in enumerate(grid()):
        threshold = float(train["sparse_score"].quantile(params["score_q"]))
        train_m, _ = evaluate(train, params, threshold, f"train2024_grid_{i}")
        train_m["selection_score"] = policy_score(train_m)
        train_m["grid_id"] = i
        train_m["score_threshold"] = threshold
        train_m.update({k: v for k, v in params.items() if k not in {"venue_allowed", "going_allowed"}})
        if train_m["selection_score"] <= -1e8:
            continue
        test_m, _ = evaluate(test, params, threshold, f"test2025_2026_grid_{i}")
        row = {f"train_{k}": v for k, v in train_m.items()}
        row.update({f"test_{k}": v for k, v in test_m.items()})
        row["grid_id"] = i
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["test_roi", "test_profit_yen"], ascending=[False, False])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_universe()
    train_grid, wf_summary, wf_tickets = walkforward_search(df)
    train_grid.to_csv(OUT / "walkforward_train_top100.csv", index=False, encoding="utf-8-sig")
    wf_summary.to_csv(OUT / "walkforward_summary.csv", index=False, encoding="utf-8-sig")
    wf_tickets.to_csv(OUT / "walkforward_selected_tickets.csv", index=False, encoding="utf-8-sig")

    top = top_train_apply_2025_2026(df)
    top.to_csv(OUT / "train2024_policy_test2025_2026_ranked.csv", index=False, encoding="utf-8-sig")
    print("WALKFORWARD")
    display_cols = [
        "label",
        "test_year",
        "races",
        "tickets",
        "stake_yen",
        "return_yen",
        "profit_yen",
        "roi",
        "race_hit_rate",
        "max_drawdown_yen",
        "venue_policy",
        "going_policy",
        "score_q",
        "ticket_mode",
        "train_roi",
        "train_races",
    ]
    print(wf_summary[display_cols].to_string(index=False))
    print("\nTOP TRAIN2024 -> TEST2025_2026")
    top_cols = [
        "grid_id",
        "train_races",
        "train_roi",
        "train_profit_yen",
        "test_races",
        "test_roi",
        "test_profit_yen",
        "test_race_hit_rate",
        "train_venue_policy",
        "train_going_policy",
        "train_score_q",
        "train_ticket_mode",
    ]
    print(top[top_cols].head(20).to_string(index=False))
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
