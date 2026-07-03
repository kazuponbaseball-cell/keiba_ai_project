from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


UNIVERSE = Path("outputs/analysis/dynamic_pair_ticket_allocation_quinella_model_v1/pair_candidate_universe.csv")
RACES = Path("data/processed/normalized/races.csv")
RUNNERS_WITH_RESULT = Path("data/datasets/train/baseline_temporal_test_dataset.csv")
OUT = Path("outputs/analysis/strict_pair_probability_roi_v1")


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
            raise ValueError("index is required")
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(s, errors="coerce")


def clip01(s: pd.Series | np.ndarray | float) -> pd.Series:
    if isinstance(s, pd.Series):
        x = s
    else:
        x = pd.Series(s)
    return pd.to_numeric(x, errors="coerce").fillna(0.0).clip(0.0, 1.0)


def norm01(s: pd.Series, lo: float | None = None, hi: float | None = None) -> pd.Series:
    x = num(s).replace([np.inf, -np.inf], np.nan)
    if lo is None:
        lo = float(x.quantile(0.05)) if x.notna().any() else 0.0
    if hi is None:
        hi = float(x.quantile(0.95)) if x.notna().any() else 1.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.5, index=x.index)
    return ((x.clip(lo, hi) - lo) / (hi - lo)).fillna(0.5).clip(0.0, 1.0)


def max_drawdown(profit: pd.Series) -> float:
    if profit.empty:
        return 0.0
    equity = profit.cumsum()
    dd = equity - equity.cummax()
    return float(dd.min())


def load_actual_front() -> pd.DataFrame:
    race_col = "\u30ec\u30fc\u30b9ID(\u65b0/\u99ac\u756a\u7121)"
    horse_col = "\u99ac\u756a"
    corner_col = "4\u89d2"
    finish_col = "\u78ba\u5b9a\u7740\u9806"
    usecols = [race_col, horse_col, corner_col, finish_col]
    df = pd.read_csv(RUNNERS_WITH_RESULT, usecols=usecols, dtype={race_col: str}, low_memory=False)
    out = pd.DataFrame(
        {
            "race_id": df[race_col].astype(str),
            "horse_no": num(df[horse_col]).astype("Int64"),
            "actual_corner4": num(df[corner_col]),
            "finish_num_lookup": num(df[finish_col]),
        }
    )
    out["actual_front5"] = out["actual_corner4"].le(5).where(out["actual_corner4"].notna(), np.nan)
    return out.drop_duplicates(["race_id", "horse_no"], keep="last")


def load_universe() -> pd.DataFrame:
    df = pd.read_csv(UNIVERSE, dtype={"race_id": str}, low_memory=False)
    df["race_id"] = df["race_id"].astype(str)
    df["year"] = df["race_id"].str[:4].astype(int)
    df["venue"] = df["race_id"].str.zfill(16).str[8:10].map(VENUE_CODE).fillna("Unknown")

    races = pd.read_csv(RACES, dtype=str, low_memory=False)
    race_info = races.iloc[:, [0, 7, 12]].copy()
    race_info.columns = ["race_id", "surface_raw", "going_raw"]
    race_info = race_info.drop_duplicates("race_id")
    going_map = {"\u826f": "Good", "\u7a0d": "Yielding", "\u7a0d\u91cd": "Yielding", "\u91cd": "Soft", "\u4e0d": "Heavy", "\u4e0d\u826f": "Heavy"}
    df = df.merge(race_info, on="race_id", how="left")
    df["going"] = df["going_raw"].map(going_map).fillna("Unknown")

    front = load_actual_front()
    anchor_front = front.rename(
        columns={
            "horse_no": "anchor_no",
            "actual_corner4": "anchor_actual_corner4",
            "actual_front5": "anchor_actual_front5",
            "finish_num_lookup": "anchor_finish_lookup",
        }
    )
    partner_front = front.rename(
        columns={
            "horse_no": "partner_no",
            "actual_corner4": "partner_actual_corner4",
            "actual_front5": "partner_actual_front5",
            "finish_num_lookup": "partner_finish_lookup",
        }
    )
    df["anchor_no"] = num(df["anchor_no"]).astype("Int64")
    df["partner_no"] = num(df["partner_no"]).astype("Int64")
    df = df.merge(anchor_front, on=["race_id", "anchor_no"], how="left")
    df = df.merge(partner_front, on=["race_id", "partner_no"], how="left")

    for col in [
        "anchor_odds",
        "partner_odds",
        "anchor_win_score",
        "partner_win_score",
        "anchor_place_score",
        "partner_place_score",
        "anchor_quinella_score",
        "partner_quinella_score",
        "pair_score",
        "pair_quinella_score",
        "wide_axis_score",
        "wide_partner_score",
        "market_overlay_score",
        "late_value_survives_score",
        "projected_front5_prob",
        "anchor_danger",
        "partner_danger",
        "wide_pay",
        "umaren_pay",
    ]:
        df[col] = num(df.get(col), df.index, 0.0).fillna(0.0)
    df["wide_hit"] = df["wide_hit"].astype(bool)
    df["umaren_hit"] = df["umaren_hit"].astype(bool)
    df["wide_quote_proxy"] = 100.0 * (np.sqrt((df["anchor_odds"].clip(1.0) * df["partner_odds"].clip(1.0))) * 0.45).clip(1.1, 120.0)
    df["umaren_quote_proxy"] = 100.0 * (df["anchor_odds"].clip(1.0) * df["partner_odds"].clip(1.0) * 0.32).clip(1.3, 260.0)
    return df


@dataclass
class Calibrator:
    feature: str
    label: str
    bins: pd.DataFrame
    fallback: float

    def apply(self, df: pd.DataFrame) -> pd.Series:
        x = num(df[self.feature]).fillna(0.0)
        out = pd.Series(self.fallback, index=df.index, dtype=float)
        if self.bins.empty:
            return out.clip(0.001, 0.95)
        ordered = self.bins.sort_values("raw_min")
        for _, row in ordered.iterrows():
            mask = x.between(float(row["raw_min"]), float(row["raw_max"]), inclusive="both")
            out.loc[mask] = float(row["prob"])
        low = x.lt(float(ordered["raw_min"].min()))
        high = x.gt(float(ordered["raw_max"].max()))
        out.loc[low] = float(ordered.iloc[0]["prob"])
        out.loc[high] = float(ordered.iloc[-1]["prob"])
        return out.clip(0.001, 0.95)


def fit_calibrator(train: pd.DataFrame, feature: str, label: str, bins: int = 8, smoothing: float = 20.0) -> Calibrator:
    use = train[[feature, label]].copy()
    use[feature] = num(use[feature]).fillna(0.0)
    use[label] = num(use[label]).fillna(0.0)
    fallback = float(use[label].mean()) if len(use) else 0.01
    if len(use) < max(40, bins * 8) or use[feature].nunique() < 4:
        return Calibrator(feature, label, pd.DataFrame(), fallback)
    ranked = use[feature].rank(method="first")
    try:
        use["_bin"] = pd.qcut(ranked, bins, labels=False, duplicates="drop")
    except ValueError:
        return Calibrator(feature, label, pd.DataFrame(), fallback)
    grouped = (
        use.groupby("_bin", observed=True)
        .agg(raw_min=(feature, "min"), raw_max=(feature, "max"), n=(label, "size"), hit_rate=(label, "mean"))
        .reset_index(drop=True)
        .sort_values("raw_min")
    )
    grouped["prob"] = (grouped["hit_rate"] * grouped["n"] + fallback * smoothing) / (grouped["n"] + smoothing)
    # Monotonic calibration: higher raw score should not reduce probability.
    grouped["prob"] = grouped["prob"].cummax().clip(0.001, 0.95)
    return Calibrator(feature, label, grouped, fallback)


def build_raw_probability_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["front_raw"] = (
        0.64 * clip01(out["projected_front5_prob"])
        + 0.16 * clip01(out["wide_partner_score"])
        + 0.10 * norm01(out["partner_odds"], lo=3.0, hi=30.0)
        + 0.10 * clip01(out["market_overlay_score"])
        - 0.12 * clip01(out["partner_danger"])
    ).clip(0.0, 1.0)
    place_joint = np.sqrt((clip01(out["anchor_place_score"]) * clip01(out["partner_place_score"])).clip(0.0, 1.0))
    front_bonus = clip01(out["front_raw"])
    out["wide_joint_raw"] = (
        0.24 * clip01(out["pair_quinella_score"])
        + 0.20 * clip01(out["pair_score"])
        + 0.17 * place_joint
        + 0.14 * front_bonus
        + 0.12 * clip01(out["market_overlay_score"])
        + 0.09 * clip01(out["late_value_survives_score"])
        + 0.04 * norm01(out["partner_odds"], lo=4.0, hi=40.0)
        - 0.10 * clip01(out["anchor_danger"])
        - 0.12 * clip01(out["partner_danger"])
    ).clip(0.0, 1.0)
    out["umaren_joint_raw"] = (
        0.34 * clip01(out["pair_quinella_score"])
        + 0.17 * clip01(out["anchor_quinella_score"])
        + 0.17 * clip01(out["partner_quinella_score"])
        + 0.12 * clip01(out["anchor_win_score"])
        + 0.08 * front_bonus
        + 0.07 * clip01(out["market_overlay_score"])
        + 0.05 * clip01(out["late_value_survives_score"])
        - 0.08 * clip01(out["anchor_danger"])
        - 0.12 * clip01(out["partner_danger"])
    ).clip(0.0, 1.0)
    out["partner_actual_front5_label"] = num(out["partner_actual_front5"]).fillna(0.0)
    out["wide_label"] = out["wide_hit"].astype(float)
    out["umaren_label"] = out["umaren_hit"].astype(float)
    return out


def add_calibrated_probs(train: pd.DataFrame, apply_to: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    front_cal = fit_calibrator(train, "front_raw", "partner_actual_front5_label", bins=8, smoothing=30.0)
    tmp_train = train.copy()
    tmp_apply = apply_to.copy()
    tmp_train["front5_prob_cal"] = front_cal.apply(tmp_train)
    tmp_apply["front5_prob_cal"] = front_cal.apply(tmp_apply)

    # Joint probability raw scores use the calibrated front probability.
    for d in (tmp_train, tmp_apply):
        d["wide_joint_model_raw"] = (
            0.82 * num(d["wide_joint_raw"]).fillna(0.0)
            + 0.18 * num(d["front5_prob_cal"]).fillna(0.0)
        ).clip(0.0, 1.0)
        d["umaren_joint_model_raw"] = (
            0.88 * num(d["umaren_joint_raw"]).fillna(0.0)
            + 0.12 * num(d["front5_prob_cal"]).fillna(0.0)
        ).clip(0.0, 1.0)

    wide_cal = fit_calibrator(tmp_train, "wide_joint_model_raw", "wide_label", bins=10, smoothing=25.0)
    umaren_cal = fit_calibrator(tmp_train, "umaren_joint_model_raw", "umaren_label", bins=10, smoothing=25.0)
    tmp_apply["wide_hit_prob_cal"] = wide_cal.apply(tmp_apply)
    tmp_apply["umaren_hit_prob_cal"] = umaren_cal.apply(tmp_apply)
    tmp_apply["wide_ev_proxy"] = tmp_apply["wide_hit_prob_cal"] * tmp_apply["wide_quote_proxy"] / 100.0
    tmp_apply["umaren_ev_proxy"] = tmp_apply["umaren_hit_prob_cal"] * tmp_apply["umaren_quote_proxy"] / 100.0
    tmp_apply["strict_rank_score"] = (
        0.44 * norm01(tmp_apply["wide_hit_prob_cal"])
        + 0.24 * norm01(tmp_apply["wide_ev_proxy"], lo=0.5, hi=2.2)
        + 0.18 * norm01(tmp_apply["umaren_ev_proxy"], lo=0.3, hi=2.2)
        + 0.14 * norm01(tmp_apply["front5_prob_cal"])
    )
    meta = {
        "front_fallback": front_cal.fallback,
        "wide_fallback": wide_cal.fallback,
        "umaren_fallback": umaren_cal.fallback,
    }
    return tmp_apply, meta


def policy_grid() -> list[dict]:
    venue_policies = {
        "skip_hakodate": set(VENUE_CODE.values()) - {"Hakodate"},
        "positive_venues": {"Fukushima", "Niigata", "Chukyo", "Hanshin", "Kokura", "Tokyo"},
    }
    going_policies = {
        "skip_heavy": {"Good", "Yielding", "Soft", "Unknown"},
        "skip_soft_heavy": {"Good", "Yielding", "Unknown"},
    }
    rows: list[dict] = []
    for (
        coverage,
        venue_policy,
        going_policy,
        wide_ev_min,
        umaren_ev_min,
        front_min,
        market_min,
        partner_odds_min,
        ticket_mode,
    ) in product(
        [0.05, 0.10],
        venue_policies.keys(),
        going_policies.keys(),
        [1.25, 1.50],
        [1.50],
        [0.25, 0.40],
        [0.35],
        [10.0, 12.0],
        ["wide_umaren"],
    ):
        rows.append(
            {
                "coverage": coverage,
                "venue_policy": venue_policy,
                "venue_allowed": venue_policies[venue_policy],
                "going_policy": going_policy,
                "going_allowed": going_policies[going_policy],
                "wide_ev_min": wide_ev_min,
                "umaren_ev_min": umaren_ev_min,
                "front_min": front_min,
                "market_min": market_min,
                "partner_odds_min": partner_odds_min,
                "ticket_mode": ticket_mode,
                "wide_stake": 200.0,
                "umaren_stake": 100.0 if ticket_mode == "wide_umaren" else 0.0,
            }
        )
    return rows


def select_pairs(df: pd.DataFrame, params: dict, threshold: float) -> pd.DataFrame:
    work = df.copy()
    work["policy_rank_score"] = num(work.get("strict_rank_score"), work.index, 0.0).fillna(0.0)
    mask = (
        work["venue"].isin(params["venue_allowed"])
        & work["going"].isin(params["going_allowed"])
        & work["policy_rank_score"].ge(threshold)
        & work["wide_ev_proxy"].ge(params["wide_ev_min"])
        & work["front5_prob_cal"].ge(params["front_min"])
        & work["market_overlay_score"].ge(params["market_min"])
        & work["partner_odds"].ge(params["partner_odds_min"])
        & work["partner_danger"].le(0.35)
        & work["anchor_danger"].le(0.55)
    )
    selected = work[mask].copy()
    if selected.empty:
        return selected
    return (
        selected.sort_values(["race_id", "policy_rank_score", "wide_ev_proxy"], ascending=[True, False, False])
        .groupby("race_id", as_index=False)
        .head(1)
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
        umaren = base[
            base["umaren_ev_proxy"].ge(params["umaren_ev_min"])
            & base["umaren_hit_prob_cal"].ge(0.025)
            & base["partner_odds"].le(40.0)
        ].copy()
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


def evaluate(apply_df: pd.DataFrame, params: dict, threshold: float, label: str) -> tuple[dict, pd.DataFrame]:
    pairs = select_pairs(apply_df, params, threshold)
    tickets = tickets_from_pairs(pairs, params)
    return metrics(tickets, label), tickets


def policy_score(m: dict) -> float:
    if m["races"] < 40:
        return -1e9
    return float(m["roi"]) * np.sqrt(max(float(m["race_hit_rate"]), 0.001)) * np.log1p(float(m["races"])) + float(m["profit_yen"]) / 100000.0


def threshold_from_coverage(df: pd.DataFrame, coverage: float) -> float:
    return float(num(df.get("strict_rank_score"), df.index, 0.0).fillna(0.0).quantile(1.0 - coverage))


def walkforward(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grid_rows = policy_grid()
    train_rows: list[pd.DataFrame] = []
    wf_rows: list[dict] = []
    ticket_frames: list[pd.DataFrame] = []
    for test_year in [2025, 2026]:
        train_raw = df[df["year"] < test_year].copy()
        test_raw = df[df["year"] == test_year].copy()
        train_scored, meta = add_calibrated_probs(train_raw, train_raw)
        test_scored, _ = add_calibrated_probs(train_raw, test_raw)
        rows: list[dict] = []
        for i, params in enumerate(grid_rows):
            threshold = threshold_from_coverage(train_scored, params["coverage"])
            m, _ = evaluate(train_scored, params, threshold, f"train_{test_year}_{i}")
            row = m | {k: v for k, v in params.items() if k not in {"venue_allowed", "going_allowed"}}
            row["grid_id"] = i
            row["score_threshold"] = threshold
            row["test_year"] = test_year
            row["selection_score"] = policy_score(m)
            row.update(meta)
            rows.append(row)
        train_grid = pd.DataFrame(rows).sort_values(["selection_score", "roi"], ascending=[False, False])
        train_rows.append(train_grid.head(100))
        best = train_grid.iloc[0]
        params = grid_rows[int(best["grid_id"])]
        m, tickets = evaluate(test_scored, params, float(best["score_threshold"]), f"wf_test_{test_year}")
        m.update({k: v for k, v in params.items() if k not in {"venue_allowed", "going_allowed"}})
        m["test_year"] = test_year
        m["train_roi"] = float(best["roi"])
        m["train_races"] = int(best["races"])
        m["train_profit_yen"] = float(best["profit_yen"])
        m["score_threshold"] = float(best["score_threshold"])
        wf_rows.append(m)
        if not tickets.empty:
            tmp = tickets.copy()
            tmp["test_year"] = test_year
            tmp["selected_grid_id"] = int(best["grid_id"])
            ticket_frames.append(tmp)
    return pd.concat(train_rows, ignore_index=True), pd.DataFrame(wf_rows), pd.concat(ticket_frames, ignore_index=True, sort=False) if ticket_frames else pd.DataFrame()


def train_val_holdout(df: pd.DataFrame) -> pd.DataFrame:
    grid_rows = policy_grid()
    train_raw = df[df["year"] == 2024].copy()
    val_raw = df[df["year"] == 2025].copy()
    hold_raw = df[df["year"] == 2026].copy()
    train_scored, meta = add_calibrated_probs(train_raw, train_raw)
    val_scored, _ = add_calibrated_probs(train_raw, val_raw)
    hold_scored, _ = add_calibrated_probs(pd.concat([train_raw, val_raw], ignore_index=True, sort=False), hold_raw)
    rows: list[dict] = []
    for i, params in enumerate(grid_rows):
        threshold = threshold_from_coverage(train_scored, params["coverage"])
        m_train, _ = evaluate(train_scored, params, threshold, f"train_2024_{i}")
        m_val, _ = evaluate(val_scored, params, threshold, f"val_2025_{i}")
        m_hold, _ = evaluate(hold_scored, params, threshold, f"hold_2026_{i}")
        row = {k: v for k, v in params.items() if k not in {"venue_allowed", "going_allowed"}}
        row["grid_id"] = i
        row["score_threshold"] = threshold
        row.update({f"train_{k}": v for k, v in m_train.items()})
        row.update({f"val_{k}": v for k, v in m_val.items()})
        row.update({f"hold_{k}": v for k, v in m_hold.items()})
        row.update(meta)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["dev_score"] = (
        out["val_roi"] * np.sqrt(out["val_race_hit_rate"].clip(lower=0.001)) * np.log1p(out["val_races"])
        + 0.25 * out["train_roi"]
        + out["val_profit_yen"] / 100000.0
    )
    return out.sort_values(["dev_score", "val_profit_yen"], ascending=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = build_raw_probability_features(load_universe())
    train_grid, wf_summary, wf_tickets = walkforward(df)
    holdout = train_val_holdout(df)
    train_grid.to_csv(OUT / "walkforward_train_top100.csv", index=False, encoding="utf-8-sig")
    wf_summary.to_csv(OUT / "walkforward_summary.csv", index=False, encoding="utf-8-sig")
    wf_tickets.to_csv(OUT / "walkforward_selected_tickets.csv", index=False, encoding="utf-8-sig")
    holdout.to_csv(OUT / "train2024_val2025_hold2026_grid.csv", index=False, encoding="utf-8-sig")

    print("WALKFORWARD")
    cols = [
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
        "coverage",
        "ticket_mode",
        "train_roi",
        "train_races",
    ]
    print(wf_summary[cols].to_string(index=False))
    print("\nTRAIN2024 -> VAL2025 -> HOLD2026 TOP")
    hcols = [
        "grid_id",
        "train_races",
        "train_roi",
        "train_profit_yen",
        "val_races",
        "val_roi",
        "val_profit_yen",
        "val_race_hit_rate",
        "hold_races",
        "hold_roi",
        "hold_profit_yen",
        "hold_race_hit_rate",
        "venue_policy",
        "going_policy",
        "coverage",
        "wide_ev_min",
        "umaren_ev_min",
        "front_min",
        "market_min",
        "partner_odds_min",
        "ticket_mode",
    ]
    print(holdout[hcols].head(25).to_string(index=False))
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
