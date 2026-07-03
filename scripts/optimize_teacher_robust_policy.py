from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from optimize_teacher_false_positive_filter import (
    SCORED,
    add_false_positive_features,
    apply_filter,
    base_select,
    metrics,
    policy_grid,
    stake_selected,
)


OUT = Path("outputs/analysis/teacher_robust_policy_v1")


def race_blocks(df: pd.DataFrame, n_blocks: int = 4) -> dict[str, int]:
    races = sorted(df["race_id"].dropna().astype(str).unique())
    if not races:
        return {}
    return {race_id: min(i * n_blocks // len(races), n_blocks - 1) for i, race_id in enumerate(races)}


def race_level(tickets: pd.DataFrame) -> pd.DataFrame:
    if tickets.empty:
        return pd.DataFrame(columns=["race_id", "stake_yen", "return_yen", "profit_yen", "hit"])
    race = (
        tickets.groupby("race_id", sort=False)
        .agg(stake_yen=("stake_yen", "sum"), return_yen=("return_yen", "sum"), hit=("hit", "max"))
        .reset_index()
    )
    race["profit_yen"] = race["return_yen"] - race["stake_yen"]
    return race


def roi_after_removing_top_profit_races(tickets: pd.DataFrame, top_n: int = 5) -> float:
    race = race_level(tickets)
    if race.empty or len(race) <= top_n:
        return 0.0
    kept = race.sort_values("profit_yen", ascending=False).iloc[top_n:]
    stake = float(kept["stake_yen"].sum())
    return float(kept["return_yen"].sum() / stake) if stake else 0.0


def block_metrics(tickets: pd.DataFrame, blocks: dict[str, int], n_blocks: int = 4) -> dict[str, float]:
    out: dict[str, float] = {}
    if tickets.empty:
        for b in range(n_blocks):
            out[f"block{b + 1}_races"] = 0
            out[f"block{b + 1}_roi"] = 0.0
            out[f"block{b + 1}_hit_rate"] = 0.0
        out["min_block_roi"] = 0.0
        out["median_block_roi"] = 0.0
        out["min_block_races"] = 0
        return out

    tmp = tickets.copy()
    tmp["block"] = tmp["race_id"].map(blocks).fillna(-1).astype(int)
    rois: list[float] = []
    race_counts: list[int] = []
    for b in range(n_blocks):
        block_tickets = tmp[tmp["block"].eq(b)].copy()
        m = metrics(block_tickets, f"block{b + 1}")
        out[f"block{b + 1}_races"] = m["races"]
        out[f"block{b + 1}_roi"] = m["roi"]
        out[f"block{b + 1}_hit_rate"] = m["race_hit_rate"]
        rois.append(float(m["roi"]))
        race_counts.append(int(m["races"]))
    out["min_block_roi"] = float(min(rois)) if rois else 0.0
    out["median_block_roi"] = float(np.median(rois)) if rois else 0.0
    out["min_block_races"] = int(min(race_counts)) if race_counts else 0
    return out


def robust_score(row: dict) -> float:
    if row["train_races"] < 80:
        return -1e9
    if row["train_race_hit_rate"] < 0.07:
        return -1e9
    if row["min_block_races"] < 12:
        return -1e9
    if row["top5_removed_roi"] < 0.80:
        return -1e9

    drawdown_penalty = abs(row["train_max_drawdown_yen"]) / 120000.0
    concentration_penalty = max(row["train_roi"] - row["top5_removed_roi"], 0.0) * 0.45
    return float(
        1.25 * row["min_block_roi"]
        + 1.10 * row["median_block_roi"]
        + 0.90 * row["top5_removed_roi"]
        + 0.35 * row["train_roi"]
        + 2.25 * row["train_race_hit_rate"]
        + 0.12 * np.log1p(row["train_races"])
        - drawdown_penalty
        - concentration_penalty
    )


def select_policy(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    params_rows = policy_grid()
    base_cache_train: dict[tuple[float, int], pd.DataFrame] = {}
    base_cache_test: dict[tuple[float, int], pd.DataFrame] = {}
    for params in params_rows:
        key = (float(params["coverage"]), int(params["max_tickets_per_race"]))
        if key not in base_cache_train:
            base_cache_train[key] = base_select(train, key[0], key[1])
            base_cache_test[key] = base_select(test, key[0], key[1])

    blocks = race_blocks(train, n_blocks=4)
    rows: list[dict] = []
    for i, params in enumerate(params_rows):
        key = (float(params["coverage"]), int(params["max_tickets_per_race"]))
        filt_train = stake_selected(apply_filter(base_cache_train[key], params))
        filt_test = stake_selected(apply_filter(base_cache_test[key], params))

        m_train = metrics(filt_train, f"train_{i}")
        m_test = metrics(filt_test, f"test_{i}")
        row = params | {"grid_id": i}
        row.update({f"train_{k}": v for k, v in m_train.items()})
        row.update(block_metrics(filt_train, blocks, n_blocks=4))
        row["top5_removed_roi"] = roi_after_removing_top_profit_races(filt_train, top_n=5)
        row.update({f"test_{k}": v for k, v in m_test.items()})
        row["robust_score"] = robust_score(row)
        rows.append(row)

    grid = pd.DataFrame(rows).sort_values(["robust_score", "train_roi"], ascending=[False, False])
    best = grid.iloc[0]
    best_params = {k: best[k] for k in params_rows[0].keys()}
    key = (float(best_params["coverage"]), int(best_params["max_tickets_per_race"]))
    best_train = stake_selected(apply_filter(base_cache_train[key], best_params))
    best_test = stake_selected(apply_filter(base_cache_test[key], best_params))
    return grid, best_train, best_test


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scored = pd.read_csv(SCORED, dtype={"race_id": str}, low_memory=False)
    scored = add_false_positive_features(scored)
    train = scored[scored["year"].eq(2025)].copy()
    test = scored[scored["year"].eq(2026)].copy()
    grid, best_train, best_test = select_policy(train, test)

    grid.to_csv(OUT / "robust_policy_grid.csv", index=False, encoding="utf-8-sig")
    best_train.to_csv(OUT / "best_robust_train_2025_tickets.csv", index=False, encoding="utf-8-sig")
    best_test.to_csv(OUT / "best_robust_test_2026_tickets.csv", index=False, encoding="utf-8-sig")

    print("ROBUST TEACHER POLICY")
    show = [
        "grid_id",
        "coverage",
        "wide_partner_odds_max",
        "wide_roi_max",
        "wide_quote_max",
        "wide_joint_min",
        "umaren_partner_odds_max",
        "umaren_roi_max",
        "umaren_quote_max",
        "umaren_joint_min",
        "train_races",
        "train_roi",
        "train_profit_yen",
        "train_race_hit_rate",
        "min_block_roi",
        "median_block_roi",
        "top5_removed_roi",
        "test_races",
        "test_roi",
        "test_profit_yen",
        "test_race_hit_rate",
        "robust_score",
    ]
    print(grid[show].head(30).to_string(index=False))
    print("\nBEST TRAIN")
    print(metrics(best_train, "best_train"))
    print("\nBEST TEST")
    print(metrics(best_test, "best_test"))
    print("\nBEST TEST TICKETS")
    cols = [
        "race_id",
        "ticket_type",
        "anchor_name",
        "partner_name",
        "teacher_edge_score",
        "quality_support_score",
        "fragile_overlay_score",
        "runtime_expected_roi",
        "partner_odds",
        "quote_proxy",
        "strength_joint_score",
        "hit",
        "pay_per100",
        "stake_yen",
        "return_yen",
    ]
    print(best_test[cols].to_string(index=False))
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
