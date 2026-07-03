from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_strongest_final_strength_model import evaluate, policy_grid, policy_score, threshold_from_coverage


ROOT = Path(__file__).resolve().parents[1]
PAIR_UNIVERSE = ROOT / "outputs" / "analysis" / "strongest_final_strength_model_v1" / "pair_strength_universe.csv"
RUNNER_DATA = ROOT / "data" / "datasets" / "train" / "baseline_temporal_test_dataset.csv"
OUT = ROOT / "outputs" / "analysis" / "pace_regime_overlay_v1"


VENUE_JA_TO_EN = {
    "札幌": "Sapporo",
    "函館": "Hakodate",
    "福島": "Fukushima",
    "新潟": "Niigata",
    "東京": "Tokyo",
    "中山": "Nakayama",
    "中京": "Chukyo",
    "京都": "Kyoto",
    "阪神": "Hanshin",
    "小倉": "Kokura",
}


def num(s: pd.Series | None, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if s is None:
        if index is None:
            raise ValueError("index is required")
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(s, errors="coerce")


def norm01(s: pd.Series, lo: float | None = None, hi: float | None = None) -> pd.Series:
    x = num(s).replace([np.inf, -np.inf], np.nan)
    if lo is None:
        lo = float(x.quantile(0.05)) if x.notna().any() else 0.0
    if hi is None:
        hi = float(x.quantile(0.95)) if x.notna().any() else 1.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.5, index=x.index)
    return ((x.clip(lo, hi) - lo) / (hi - lo)).fillna(0.5).clip(0.0, 1.0)


def zrace(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(16)


def first_existing(columns: Iterable[str], candidates: list[str]) -> str:
    existing = set(columns)
    for c in candidates:
        if c in existing:
            return c
    raise KeyError(f"missing columns: {candidates}")


def normalize_surface(s: pd.Series) -> pd.Series:
    t = s.astype("string").fillna("")
    return pd.Series(
        np.select(
            [
                t.str.contains("芝", regex=False) | t.str.contains("ЋЕ", regex=False),
                t.str.contains("ダ", regex=False) | t.str.contains("ѓ_", regex=False),
            ],
            ["turf", "dirt"],
            default="unknown",
        ),
        index=s.index,
    )


def normalize_going(s: pd.Series) -> pd.Series:
    t = s.astype("string").fillna("")
    return pd.Series(
        np.select(
            [
                t.str.contains("良", regex=False) | t.str.contains("Good", case=False, regex=False),
                t.str.contains("稍", regex=False) | t.str.contains("Yield", case=False, regex=True),
                t.str.contains("不", regex=False) | t.str.contains("Heavy", case=False, regex=False),
                t.str.contains("重", regex=False) | t.str.contains("Soft", case=False, regex=False),
            ],
            ["Good", "Yielding", "Heavy", "Soft"],
            default="Unknown",
        ),
        index=s.index,
    )


def distance_bin(s: pd.Series) -> pd.Series:
    d = num(s)
    return pd.Series(
        np.select(
            [d <= 1400, d <= 1800, d <= 2200, d > 2200],
            ["sprint", "mile", "middle", "long"],
            default="unknown",
        ),
        index=s.index,
    )


def venue_group(s: pd.Series) -> pd.Series:
    v = s.astype("string")
    local = v.isin(["Sapporo", "Hakodate", "Fukushima", "Kokura"])
    return pd.Series(np.where(local, "local_small", "major"), index=s.index)


def style_from_corner(corner4: float, field_size: float) -> str:
    if pd.isna(corner4) or pd.isna(field_size):
        return "unknown"
    if corner4 <= 3:
        return "front"
    if corner4 <= max(5, field_size * 0.40):
        return "stalker"
    if corner4 >= field_size * 0.65:
        return "closer"
    return "mid"


def build_race_frame(path: Path) -> pd.DataFrame:
    header = list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)
    race_col = first_existing(header, ["レースID(新/馬番無)", "race_id"])
    finish_col = first_existing(header, ["確定着順", "finish_num", "finish"])
    corner_col = first_existing(header, ["4角", "4角.1", "corner4"])
    usecols = [
        race_col,
        finish_col,
        corner_col,
        first_existing(header, ["場所", "venue"]),
        first_existing(header, ["芝・ダ", "surface"]),
        first_existing(header, ["距離", "distance"]),
        first_existing(header, ["馬場状態", "going"]),
        first_existing(header, ["頭数", "field_size"]),
        first_existing(header, ["race_front_runner_count"]),
        first_existing(header, ["race_front_runner_ratio"]),
        first_existing(header, ["race_closer_count"]),
        first_existing(header, ["race_closer_ratio"]),
        first_existing(header, ["race_early_pressure_score"]),
    ]
    df = pd.read_csv(path, usecols=list(dict.fromkeys(usecols)), encoding="utf-8-sig", low_memory=False)
    df = df.rename(
        columns={
            race_col: "race_id",
            finish_col: "finish",
            corner_col: "corner4",
            "場所": "venue_ja",
            "venue": "venue_ja",
            "芝・ダ": "surface_raw",
            "surface": "surface_raw",
            "距離": "distance",
            "distance": "distance",
            "馬場状態": "going_raw",
            "going": "going_raw",
            "頭数": "field_size",
            "field_size": "field_size",
        }
    )
    df["race_id"] = zrace(df["race_id"])
    df["year"] = df["race_id"].str[:4].astype(int)
    df["finish"] = num(df["finish"])
    df["corner4"] = num(df["corner4"])
    df["field_size"] = num(df["field_size"]).fillna(df.groupby("race_id")["race_id"].transform("size"))
    df["venue"] = df["venue_ja"].map(VENUE_JA_TO_EN).fillna(df["venue_ja"].astype("string"))
    df["surface_type"] = normalize_surface(df["surface_raw"])
    df["going_norm"] = normalize_going(df["going_raw"])
    df["distance_bin"] = distance_bin(df["distance"])
    df["venue_group"] = venue_group(df["venue"])

    rows: list[dict] = []
    race_feature_cols = [
        "race_front_runner_count",
        "race_front_runner_ratio",
        "race_closer_count",
        "race_closer_ratio",
        "race_early_pressure_score",
    ]
    for race_id, g in df.groupby("race_id", sort=False):
        first = g.iloc[0]
        field_size = float(num(g["field_size"]).dropna().iloc[0]) if num(g["field_size"]).notna().any() else float(len(g))
        top3 = g[g["finish"].between(1, 3)].copy()
        styles = [style_from_corner(float(c), field_size) for c in top3["corner4"]]
        front_stalker_count = sum(x in {"front", "stalker"} for x in styles)
        closer_count = sum(x == "closer" for x in styles)
        avg_corner4 = float(top3["corner4"].mean()) if not top3.empty else np.nan
        actual_shape = "mixed"
        if front_stalker_count >= 2 and (pd.isna(avg_corner4) or avg_corner4 <= max(5, field_size * 0.45)):
            actual_shape = "front_stalker"
        elif closer_count >= 2 or (pd.notna(avg_corner4) and avg_corner4 >= field_size * 0.55):
            actual_shape = "closer"
        row = {
            "race_id": race_id,
            "year": int(first["year"]),
            "venue": first["venue"],
            "venue_group": first["venue_group"],
            "surface_type": first["surface_type"],
            "going_norm": first["going_norm"],
            "distance": float(first["distance"]) if pd.notna(first["distance"]) else np.nan,
            "distance_bin": first["distance_bin"],
            "field_size": field_size,
            "actual_bias_shape": actual_shape,
            "target_front_stalker": float(actual_shape == "front_stalker"),
            "target_closer": float(actual_shape == "closer"),
            "top3_avg_corner4": avg_corner4,
        }
        for c in race_feature_cols:
            row[c] = float(num(g[c]).mean()) if c in g else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def smoothed_group_rates(train: pd.DataFrame, keys: list[str], global_front: float, global_closer: float, smoothing: float = 20.0) -> pd.DataFrame:
    if not keys:
        return pd.DataFrame()
    grouped = (
        train.groupby(keys, dropna=False)
        .agg(
            regime_n=("race_id", "size"),
            raw_front=("target_front_stalker", "mean"),
            raw_closer=("target_closer", "mean"),
        )
        .reset_index()
    )
    grouped["front_survival_regime_score"] = (
        grouped["raw_front"] * grouped["regime_n"] + global_front * smoothing
    ) / (grouped["regime_n"] + smoothing)
    grouped["collapse_conversion_regime_score"] = (
        grouped["raw_closer"] * grouped["regime_n"] + global_closer * smoothing
    ) / (grouped["regime_n"] + smoothing)
    return grouped[keys + ["regime_n", "front_survival_regime_score", "collapse_conversion_regime_score"]]


def attach_rate_by_hierarchy(apply: pd.DataFrame, train_high: pd.DataFrame) -> pd.DataFrame:
    out = apply.copy()
    global_front = float(train_high["target_front_stalker"].mean()) if len(train_high) else 0.75
    global_closer = float(train_high["target_closer"].mean()) if len(train_high) else 0.09
    out["front_survival_regime_score"] = global_front
    out["collapse_conversion_regime_score"] = global_closer
    out["pace_regime_sample_size"] = 0.0
    hierarchies = [
        ["venue", "surface_type", "distance_bin", "going_norm"],
        ["venue", "surface_type", "distance_bin"],
        ["venue", "surface_type"],
        ["venue_group", "surface_type", "distance_bin"],
        ["surface_type", "distance_bin"],
        ["venue_group", "surface_type"],
    ]
    for keys in reversed(hierarchies):
        rates = smoothed_group_rates(train_high, keys, global_front, global_closer)
        if rates.empty:
            continue
        before = len(out)
        merged = out[keys].merge(rates, on=keys, how="left")
        usable = num(merged["regime_n"]).ge(8)
        out.loc[usable.values, "front_survival_regime_score"] = merged.loc[usable, "front_survival_regime_score"].to_numpy()
        out.loc[usable.values, "collapse_conversion_regime_score"] = merged.loc[usable, "collapse_conversion_regime_score"].to_numpy()
        out.loc[usable.values, "pace_regime_sample_size"] = merged.loc[usable, "regime_n"].to_numpy()
        assert len(out) == before
    return out


def add_regime_overlay_for_split(race_frame: pd.DataFrame, train_year: int, pairs: pd.DataFrame) -> pd.DataFrame:
    train = race_frame[race_frame["year"] < train_year].copy()
    high_threshold = float(train["race_early_pressure_score"].quantile(0.75)) if len(train) else 0.45
    train_high = train[num(train["race_early_pressure_score"]).ge(high_threshold)].copy()
    race_apply = race_frame[race_frame["race_id"].isin(pairs["race_id"].astype(str))].copy()
    race_apply = attach_rate_by_hierarchy(race_apply, train_high)
    lo = float(train["race_early_pressure_score"].quantile(0.25)) if len(train) else 0.20
    hi = float(train["race_early_pressure_score"].quantile(0.85)) if len(train) else 0.70
    race_apply["pressure_regime_intensity"] = norm01(race_apply["race_early_pressure_score"], lo=lo, hi=hi)
    race_apply["position_lock_score"] = (
        0.50
        + 0.55 * (race_apply["front_survival_regime_score"] - race_apply["collapse_conversion_regime_score"])
        + 0.08 * race_apply["venue_group"].eq("local_small").astype(float)
        + 0.05 * race_apply["going_norm"].isin(["Soft", "Heavy"]).astype(float)
    ).clip(0.0, 1.0)
    race_cols = [
        "race_id",
        "race_early_pressure_score",
        "race_front_runner_count",
        "race_closer_count",
        "front_survival_regime_score",
        "collapse_conversion_regime_score",
        "pace_regime_sample_size",
        "pressure_regime_intensity",
        "position_lock_score",
    ]
    out = pairs.merge(race_apply[race_cols], on="race_id", how="left")
    for c, default in [
        ("front_survival_regime_score", 0.75),
        ("collapse_conversion_regime_score", 0.09),
        ("pressure_regime_intensity", 0.50),
        ("position_lock_score", 0.65),
        ("pace_regime_sample_size", 0.0),
    ]:
        out[c] = num(out[c], out.index, default).fillna(default)
    front_pair = num(out.get("projected_front5_prob"), out.index, 0.5).fillna(0.5).clip(0.0, 1.0)
    pressure = num(out["pressure_regime_intensity"]).clip(0.0, 1.0)
    front_survival = num(out["front_survival_regime_score"]).clip(0.0, 1.0)
    closer_conversion = num(out["collapse_conversion_regime_score"]).clip(0.0, 1.0)
    out["regime_front_fit_score"] = (front_pair * front_survival * pressure).clip(0.0, 1.0)
    out["regime_front_collapse_risk_score"] = (front_pair * closer_conversion * pressure).clip(0.0, 1.0)
    out["regime_position_edge_score"] = (
        0.58 * out["regime_front_fit_score"]
        + 0.22 * num(out["position_lock_score"]).clip(0.0, 1.0)
        + 0.20 * (1.0 - out["regime_front_collapse_risk_score"])
    ).clip(0.0, 1.0)
    out["pace_regime_adjusted_pair_score"] = (
        0.94 * num(out["strongest_pair_score"]).fillna(0.0)
        + 0.06 * norm01(out["regime_position_edge_score"], lo=0.35, hi=0.82)
        + 0.020 * (out["front_survival_regime_score"] - 0.75).clip(-0.20, 0.20)
        - 0.025 * (out["collapse_conversion_regime_score"] - 0.10).clip(-0.05, 0.25) * front_pair
    ).clip(0.0, 1.0)
    return out


def evaluate_with_score(df: pd.DataFrame, score_col: str, label_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grids = policy_grid()
    train_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    ticket_frames: list[pd.DataFrame] = []
    for test_year in [2025, 2026]:
        work = df.copy()
        if score_col != "strongest_pair_score":
            work["strongest_pair_score_original"] = work["strongest_pair_score"]
            work["strongest_pair_score"] = work[score_col]
        train = work[work["year"] < test_year].copy()
        test = work[work["year"] == test_year].copy()
        rows: list[dict] = []
        for i, params in enumerate(grids):
            threshold = threshold_from_coverage(train, params)
            m, _ = evaluate(train, params, threshold, f"{label_prefix}_train_{test_year}_{i}")
            row = m | {k: v for k, v in params.items() if k not in {"venue_allowed", "going_allowed"}}
            row["grid_id"] = i
            row["score_threshold"] = threshold
            row["test_year"] = test_year
            row["selection_score"] = policy_score(m)
            rows.append(row)
        grid = pd.DataFrame(rows).sort_values(["selection_score", "roi"], ascending=[False, False])
        train_rows.append(grid.head(100))
        best = grid.iloc[0]
        params = grids[int(best["grid_id"])]
        m, tickets = evaluate(test, params, float(best["score_threshold"]), f"{label_prefix}_wf_test_{test_year}")
        m.update({k: v for k, v in params.items() if k not in {"venue_allowed", "going_allowed"}})
        m["test_year"] = test_year
        m["train_roi"] = float(best["roi"])
        m["train_races"] = int(best["races"])
        m["score_threshold"] = float(best["score_threshold"])
        m["score_col"] = score_col
        summary_rows.append(m)
        if not tickets.empty:
            tickets = tickets.copy()
            tickets["test_year"] = test_year
            tickets["score_col"] = score_col
            ticket_frames.append(tickets)
    return (
        pd.concat(train_rows, ignore_index=True, sort=False),
        pd.DataFrame(summary_rows),
        pd.concat(ticket_frames, ignore_index=True, sort=False) if ticket_frames else pd.DataFrame(),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    race_frame = build_race_frame(RUNNER_DATA)
    pairs = pd.read_csv(PAIR_UNIVERSE, dtype={"race_id": str}, low_memory=False)
    pairs["race_id"] = zrace(pairs["race_id"])
    enriched_parts: list[pd.DataFrame] = []
    for year, part in pairs.groupby("year", sort=True):
        enriched_parts.append(add_regime_overlay_for_split(race_frame, int(year), part.copy()))
    enriched = pd.concat(enriched_parts, ignore_index=True, sort=False)
    enriched.to_csv(OUT / "pair_strength_universe_with_pace_regime.csv", index=False, encoding="utf-8-sig")

    base_train, base_summary, base_tickets = evaluate_with_score(enriched, "strongest_pair_score", "baseline")
    reg_train, reg_summary, reg_tickets = evaluate_with_score(enriched, "pace_regime_adjusted_pair_score", "pace_regime")
    train_grid = pd.concat([base_train.assign(policy_variant="baseline"), reg_train.assign(policy_variant="pace_regime")], ignore_index=True)
    summary = pd.concat([base_summary.assign(policy_variant="baseline"), reg_summary.assign(policy_variant="pace_regime")], ignore_index=True)
    tickets = pd.concat([base_tickets.assign(policy_variant="baseline"), reg_tickets.assign(policy_variant="pace_regime")], ignore_index=True)
    train_grid.to_csv(OUT / "walkforward_train_top100_compare.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "walkforward_summary_compare.csv", index=False, encoding="utf-8-sig")
    tickets.to_csv(OUT / "walkforward_selected_tickets_compare.csv", index=False, encoding="utf-8-sig")

    compact_cols = [
        "policy_variant",
        "test_year",
        "candidate_races",
        "races",
        "race_selection_rate",
        "tickets",
        "stake_yen",
        "return_yen",
        "profit_yen",
        "roi",
        "race_hit_rate",
        "max_drawdown_yen",
        "wide_tickets",
        "umaren_tickets",
        "coverage",
        "venue_policy",
        "going_policy",
        "train_roi",
        "train_races",
    ]
    compact = summary[compact_cols].copy()
    compact.to_csv(OUT / "summary_compact.csv", index=False, encoding="utf-8-sig")
    totals = (
        summary.groupby("policy_variant", as_index=False)
        .agg(
            races=("races", "sum"),
            tickets=("tickets", "sum"),
            stake_yen=("stake_yen", "sum"),
            return_yen=("return_yen", "sum"),
            profit_yen=("profit_yen", "sum"),
            avg_race_hit_rate=("race_hit_rate", "mean"),
        )
    )
    totals["roi"] = totals["return_yen"] / totals["stake_yen"]
    totals.to_csv(OUT / "summary_totals.csv", index=False, encoding="utf-8-sig")
    with (OUT / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "output_dir": str(OUT),
                "race_frame_races": int(race_frame["race_id"].nunique()),
                "pair_rows": int(len(enriched)),
                "compact": compact.to_dict(orient="records"),
                "totals": totals.to_dict(orient="records"),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("PACE REGIME OVERLAY WALKFORWARD")
    print(compact.to_string(index=False))
    print("\nTOTALS")
    print(totals.to_string(index=False))
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
