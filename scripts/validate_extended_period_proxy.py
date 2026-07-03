from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


UNIVERSE = Path("outputs/analysis/dynamic_pair_ticket_allocation_quinella_model_v1/pair_candidate_universe.csv")
FINAL = Path("outputs/analysis/final_operational_quality_v1/standard_explained_tickets.csv")
OUT = Path("outputs/analysis/extended_period_validation_v1")


PARAMS = {
    "mode": "wide_umaren_strict_fixed_proxy",
    "axis_min": 0.62,
    "partner_min": 0.60,
    "partner_odds_min": 6.0,
    "partner_odds_max": 40.0,
    "front_min": 0.45,
    "pairs_per_race": 2,
    "anchor_danger_max": 0.55,
    "partner_danger_max": 0.35,
    "wide_stake": 200.0,
    "umaren_stake": 100.0,
    "umaren_pair_score_min": 0.74,
    "umaren_quinella_min": 0.0,
    "anchor_quinella_min": 0.0,
    "partner_quinella_min": 0.54,
    "umaren_partner_odds_max": 25.0,
    "umaren_pay_min": 1200.0,
}


def num(s: pd.Series | None, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if s is None:
        if index is None:
            raise ValueError("index is required when series is None")
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(s, errors="coerce")


def max_drawdown(profits: pd.Series) -> float:
    if profits.empty:
        return 0.0
    equity = profits.cumsum()
    dd = equity - equity.cummax()
    return float(dd.min())


def add_tickets_from_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if pairs.empty:
        return pd.DataFrame()
    base = pairs.copy()
    base["pair_key"] = base["race_id"].astype(str) + ":" + base["anchor_no"].astype(str) + "-" + base["partner_no"].astype(str)

    wide = base.copy()
    wide["ticket_type"] = "wide"
    wide["stake_yen"] = PARAMS["wide_stake"]
    wide["hit"] = wide["wide_hit"].astype(bool)
    wide["return_yen"] = num(wide["wide_pay"]).fillna(0.0).where(wide["hit"], 0.0) * wide["stake_yen"] / 100.0
    frames.append(wide)

    umaren_mask = (
        num(base["pair_score"]).ge(PARAMS["umaren_pair_score_min"])
        & num(base["pair_quinella_score"]).ge(PARAMS["umaren_quinella_min"])
        & num(base["anchor_quinella_score"]).ge(PARAMS["anchor_quinella_min"])
        & num(base["partner_quinella_score"]).ge(PARAMS["partner_quinella_min"])
        & num(base["partner_odds"]).le(PARAMS["umaren_partner_odds_max"])
        & num(base["umaren_pay"]).ge(PARAMS["umaren_pay_min"])
    )
    umaren = base[umaren_mask].copy()
    if not umaren.empty:
        umaren["ticket_type"] = "umaren"
        umaren["stake_yen"] = PARAMS["umaren_stake"]
        umaren["hit"] = umaren["umaren_hit"].astype(bool)
        umaren["return_yen"] = num(umaren["umaren_pay"]).fillna(0.0).where(umaren["hit"], 0.0) * umaren["stake_yen"] / 100.0
        frames.append(umaren)

    out = pd.concat(frames, ignore_index=True, sort=False)
    out["ticket_key"] = out["ticket_type"] + ":" + out["pair_key"]
    out["year"] = out["race_id"].astype(str).str[:4].astype(int)
    return out


def select_pairs(universe: pd.DataFrame) -> pd.DataFrame:
    u = universe.copy()
    mask = (
        num(u["wide_axis_score"]).ge(PARAMS["axis_min"])
        & num(u["wide_partner_score"]).ge(PARAMS["partner_min"])
        & num(u["partner_odds"]).between(PARAMS["partner_odds_min"], PARAMS["partner_odds_max"])
        & num(u["projected_front5_prob"]).ge(PARAMS["front_min"])
        & num(u["anchor_danger"]).le(PARAMS["anchor_danger_max"])
        & num(u["partner_danger"]).le(PARAMS["partner_danger_max"])
    )
    selected = u[mask].copy()
    if selected.empty:
        return selected
    return (
        selected.sort_values(["race_id", "pair_score", "market_overlay_score"], ascending=[True, False, False])
        .groupby("race_id", as_index=False)
        .head(int(PARAMS["pairs_per_race"]))
    )


def metrics(tickets: pd.DataFrame, label: str) -> dict:
    if tickets.empty:
        return {"label": label, "tickets": 0, "races": 0, "stake_yen": 0, "return_yen": 0, "profit_yen": 0, "roi_pct": 0}
    stake = float(num(tickets["stake_yen"]).sum())
    ret = float(num(tickets["return_yen"]).sum())
    by_race = tickets.groupby("race_id", sort=False).agg(stake=("stake_yen", "sum"), ret=("return_yen", "sum"), hit=("hit", "max"))
    by_race["profit"] = by_race["ret"] - by_race["stake"]
    return {
        "label": label,
        "tickets": int(len(tickets)),
        "races": int(tickets["race_id"].nunique()),
        "stake_yen": round(stake),
        "return_yen": round(ret),
        "profit_yen": round(ret - stake),
        "roi_pct": round(ret / stake * 100, 1) if stake else 0.0,
        "ticket_hit_rate_pct": round(float(tickets["hit"].mean() * 100), 1),
        "race_hit_rate_pct": round(float(by_race["hit"].mean() * 100), 1),
        "max_drawdown_yen": round(max_drawdown(by_race["profit"])),
        "avg_stake_per_race_yen": round(float(stake / tickets["race_id"].nunique()), 0),
    }


def final_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    final = pd.read_csv(FINAL, dtype={"race_id": str}, low_memory=False)
    final = final[pd.to_numeric(final["runtime_stake_yen"], errors="coerce").fillna(0).gt(0)].copy()
    final["stake_yen"] = pd.to_numeric(final["runtime_stake_yen"], errors="coerce").fillna(0.0)
    final["return_yen"] = pd.to_numeric(final["runtime_return_yen"], errors="coerce").fillna(0.0)
    final["hit"] = final["hit"].astype(bool)
    final["year"] = final["race_id"].astype(str).str[:4].astype(int)
    rows = [metrics(final, "current_final_2025_2026_exact")]
    yearly = []
    for y, g in final.groupby("year"):
        yearly.append(metrics(g, f"current_final_{int(y)}"))
    return pd.DataFrame(rows), pd.DataFrame(yearly)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(UNIVERSE, dtype={"race_id": str}, low_memory=False)
    pairs = select_pairs(universe)
    tickets = add_tickets_from_pairs(pairs)
    tickets.to_csv(OUT / "fixed_proxy_selected_tickets_2024_2026.csv", index=False, encoding="utf-8-sig")

    summary_rows = [metrics(tickets, "fixed_proxy_2024_2026_all")]
    yearly_rows = []
    for y, g in tickets.groupby("year"):
        yearly_rows.append(metrics(g, f"fixed_proxy_{int(y)}"))
    proxy_summary = pd.DataFrame(summary_rows)
    proxy_yearly = pd.DataFrame(yearly_rows)

    exact_summary, exact_yearly = final_metrics()
    summary = pd.concat([exact_summary, proxy_summary], ignore_index=True)
    yearly = pd.concat([exact_yearly, proxy_yearly], ignore_index=True)
    summary.to_csv(OUT / "extended_validation_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUT / "extended_validation_yearly.csv", index=False, encoding="utf-8-sig")

    venue_code_map = {
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
    tickets["venue"] = tickets["race_id"].astype(str).str.zfill(16).str[8:10].map(venue_code_map).fillna("Unknown")
    venue_rows = [metrics(g, f"fixed_proxy_{v}") | {"venue": v} for v, g in tickets.groupby("venue")]
    venue = pd.DataFrame(venue_rows).sort_values("profit_yen", ascending=False)
    venue.to_csv(OUT / "fixed_proxy_by_venue.csv", index=False, encoding="utf-8-sig")

    print("SUMMARY")
    print(summary.to_string(index=False))
    print("\nYEARLY")
    print(yearly.to_string(index=False))
    print("\nVENUE")
    print(venue.to_string(index=False))
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
