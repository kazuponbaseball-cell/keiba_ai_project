from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIVERSE = (
    ROOT
    / "outputs"
    / "analysis"
    / "pair_joint_probability_v2_rebuilt_20260623"
    / "pair_universe_with_joint_v2_features.csv"
)
DEFAULT_RACE_SHAPE = ROOT / "outputs" / "analysis" / "queue_shape_race_quality_v1" / "race_queue_shape_validation.csv"
OUT_DIR = ROOT / "outputs" / "analysis" / "shape_adjusted_pair_selection_v1"


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, **kwargs)


def ncol(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def clip01(s: pd.Series | np.ndarray | float) -> pd.Series:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return pd.Series(s).clip(0.0, 1.0)


def bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[col]
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def pair_key(row: pd.Series) -> str:
    a = int(row["horse_a"]) if pd.notna(row["horse_a"]) else int(row["anchor_no"])
    b = int(row["horse_b"]) if pd.notna(row["horse_b"]) else int(row["partner_no"])
    lo, hi = sorted([a, b])
    return f"{row['race_id']}:{lo}-{hi}"


def load_universe(path: Path, race_shape_path: Path) -> pd.DataFrame:
    df = read_csv(path, dtype={"race_id": str})
    if df.empty:
        return df
    df["race_id"] = df["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    if "year" not in df.columns:
        df["year"] = df["race_id"].str[:4].astype(int)
    for col in ["horse_a", "horse_b"]:
        if col not in df.columns:
            df[col] = np.nan
    df["pair_key_norm"] = df.apply(pair_key, axis=1)
    df = df.drop_duplicates(["race_id", "pair_key_norm"], keep="first").copy()

    race_shape = read_csv(race_shape_path, dtype={"race_id": str})
    race_shape["race_id"] = race_shape["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    keep = [
        "race_id",
        "queue_shape_label",
        "queue_clarity_score",
        "queue_duel_risk_score",
        "queue_front_load_score",
        "queue_top_gap",
        "queue_candidate_count",
        "actual_shape",
    ]
    race_shape = race_shape[[c for c in keep if c in race_shape.columns]].drop_duplicates("race_id")
    df = df.merge(race_shape, on="race_id", how="left")
    df["queue_shape_label"] = df["queue_shape_label"].fillna("unknown")
    return df


def add_shape_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    base = ncol(out, "pair_quinella_score", 0.5).clip(0.0, 1.0)
    base_rank = base.groupby(out["race_id"]).rank(pct=True).fillna(0.5)
    overlay = ncol(out, "market_overlay_score", 0.0).clip(0.0, 1.0)
    late = ncol(out, "late_value_survives_score", 0.0).clip(0.0, 1.0)
    front_max = ncol(out, "front_pair_max", ncol(out, "projected_front5_prob", 0.0)).clip(0.0, 1.0)
    front_min = ncol(out, "front_pair_min", 0.0).clip(0.0, 1.0)
    closer_max = ncol(out, "closer_pair_max", 0.0).clip(0.0, 1.0)
    diversity = ncol(out, "style_diversity", 0.0).clip(0.0, 1.0)
    clash = ncol(out, "front_front_clash", 0.0).clip(0.0, 1.0)
    front_slow = ncol(out, "front_front_slow_fit", 0.0).clip(0.0, 1.0)
    collapse = ncol(out, "collapse_fit", ncol(out, "race_pace_collapse", 0.0)).clip(0.0, 1.0)
    duel = ncol(out, "queue_duel_risk_score", 0.0).clip(0.0, 1.0)
    clarity = ncol(out, "queue_clarity_score", 0.0).clip(0.0, 1.0)

    label = out["queue_shape_label"].astype(str)
    single_fit = (0.46 * front_max + 0.30 * front_slow + 0.14 * front_min + 0.10 * base).clip(0.0, 1.0)
    no_clear_fit = (0.50 * front_max + 0.22 * front_slow + 0.18 * base + 0.10 * overlay).clip(0.0, 1.0)
    duel_fit = (0.35 * closer_max + 0.24 * collapse + 0.21 * diversity + 0.12 * front_max + 0.08 * base).clip(0.0, 1.0)
    matched_fit = (0.30 * closer_max + 0.25 * collapse + 0.24 * diversity + 0.13 * front_max + 0.08 * base).clip(0.0, 1.0)
    mixed_fit = (0.30 * base + 0.25 * front_max + 0.25 * closer_max + 0.20 * diversity).clip(0.0, 1.0)

    fit = pd.Series(0.5, index=out.index, dtype=float)
    fit = fit.where(~label.eq("single_leader_clear"), single_fit)
    fit = fit.where(~label.eq("no_clear_leader"), no_clear_fit)
    fit = fit.where(~label.eq("front_duel_dense"), duel_fit)
    fit = fit.where(~label.eq("matched_speed_duel"), matched_fit)
    fit = fit.where(~label.eq("mixed_queue"), mixed_fit)
    fit = fit.where(~label.eq("unknown"), mixed_fit)

    front_burn = (duel * clash * (0.45 + 0.55 * front_min)).clip(0.0, 1.0)
    dead_slow_closer = ((1.0 - duel) * clarity * closer_max * (1.0 - front_max)).clip(0.0, 1.0)
    no_clear_uncertainty = (label.eq("no_clear_leader").astype(float) * (1.0 - front_max) * 0.35).clip(0.0, 1.0)
    risk = np.maximum.reduce([front_burn.to_numpy(), dead_slow_closer.to_numpy(), no_clear_uncertainty.to_numpy()])

    out["shape_base_score"] = base
    out["shape_base_rank_score"] = base_rank
    out["shape_value_score"] = (0.55 * overlay + 0.45 * late).clip(0.0, 1.0)
    out["shape_pair_fit_score"] = fit.clip(0.0, 1.0)
    out["shape_pair_risk_score"] = pd.Series(risk, index=out.index).clip(0.0, 1.0)
    out["shape_pair_front_fit_component"] = (0.55 * front_max + 0.25 * front_slow + 0.20 * front_min).clip(0.0, 1.0)
    out["shape_pair_closer_fit_component"] = (0.45 * closer_max + 0.30 * collapse + 0.25 * diversity).clip(0.0, 1.0)
    return out


def gate_mask(df: pd.DataFrame, gate: str) -> pd.Series:
    q = ncol(df, "pair_quinella_score", 0.0)
    overlay = ncol(df, "market_overlay_score", 0.0)
    late = ncol(df, "late_value_survives_score", 0.0)
    skip = ncol(df, "skip_risk_score", 0.0)
    anchor_danger = ncol(df, "anchor_danger", 0.0)
    partner_danger = ncol(df, "partner_danger", 0.0)
    partner_odds = ncol(df, "partner_odds", 999.0)
    anchor_odds = ncol(df, "anchor_odds", 999.0)
    front = ncol(df, "projected_front5_prob", 0.0)
    value = np.maximum(overlay, late)
    danger_sum = anchor_danger + partner_danger

    if gate == "all":
        return pd.Series(True, index=df.index)
    if gate == "value_loose":
        return q.ge(0.58) & value.ge(0.55) & skip.le(0.48) & partner_odds.ge(4.0)
    if gate == "value_mid":
        return q.ge(0.62) & value.ge(0.65) & skip.le(0.42) & danger_sum.le(0.90) & partner_odds.ge(5.0)
    if gate == "value_strong":
        return q.ge(0.66) & value.ge(0.72) & skip.le(0.36) & danger_sum.le(0.75) & partner_odds.ge(6.0)
    if gate == "front_value_strong":
        return q.ge(0.64) & value.ge(0.68) & skip.le(0.40) & front.ge(0.60) & partner_odds.ge(6.0)
    if gate == "price_sane_strong":
        return (
            q.ge(0.64)
            & value.ge(0.68)
            & skip.le(0.42)
            & danger_sum.le(0.85)
            & partner_odds.between(5.0, 80.0)
            & anchor_odds.between(2.0, 60.0)
        )
    raise ValueError(f"Unknown gate: {gate}")


def ticket_return(frame: pd.DataFrame, ticket_type: str) -> pd.Series:
    if ticket_type == "umaren":
        pay = ncol(frame, "umaren_pay", 0.0)
        hit = bool_col(frame, "umaren_hit")
        return pay.where(hit, 0.0)
    if ticket_type == "wide":
        pay = ncol(frame, "wide_pay", 0.0)
        hit = bool_col(frame, "wide_hit")
        return pay.where(hit, 0.0)
    raise ValueError(ticket_type)


def hit_flag(frame: pd.DataFrame, ticket_type: str) -> pd.Series:
    return bool_col(frame, f"{ticket_type}_hit")


def select_top_per_race(frame: pd.DataFrame, score_col: str, ticket_type: str, policy: str, gate: str, weight: float) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    cols = ["race_id", score_col, "shape_base_score", "market_overlay_score"]
    sort_cols = [c for c in cols if c in frame.columns]
    selected = (
        frame.sort_values(sort_cols, ascending=[True] + [False] * (len(sort_cols) - 1))
        .drop_duplicates("race_id", keep="first")
        .copy()
    )
    selected["policy"] = policy
    selected["gate"] = gate
    selected["shape_weight"] = weight
    selected["ticket_type"] = ticket_type
    selected["stake_yen"] = 100.0
    selected["return_yen"] = ticket_return(selected, ticket_type)
    selected["hit"] = hit_flag(selected, ticket_type)
    selected["ticket_key"] = selected.apply(
        lambda r: f"{ticket_type}:{r['race_id']}:{int(r['horse_a'])}-{int(r['horse_b'])}", axis=1
    )
    return selected


def metrics(frame: pd.DataFrame, policy: str) -> dict[str, Any]:
    stake = float(frame["stake_yen"].sum()) if not frame.empty else 0.0
    ret = float(frame["return_yen"].sum()) if not frame.empty else 0.0
    return {
        "policy": policy,
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()) if not frame.empty else 0,
        "stake_yen": round(stake, 1),
        "return_yen": round(ret, 1),
        "profit_yen": round(ret - stake, 1),
        "roi_pct": round(ret / stake * 100, 1) if stake > 0 else 0.0,
        "hit_rate_pct": round(float(frame["hit"].mean() * 100), 1) if not frame.empty else 0.0,
        "avg_shape_fit": round(float(frame["shape_pair_fit_score"].mean()), 3) if not frame.empty else np.nan,
        "avg_shape_risk": round(float(frame["shape_pair_risk_score"].mean()), 3) if not frame.empty else np.nan,
        "changed_rate_pct": round(float(frame.get("changed_from_base", pd.Series(False, index=frame.index)).mean() * 100), 1)
        if not frame.empty
        else 0.0,
    }


def evaluate(df: pd.DataFrame, gates: list[str], weights: list[float], ticket_types: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selections: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []

    for gate in gates:
        gated = df[gate_mask(df, gate)].copy()
        if gated.empty:
            continue
        for ticket_type in ticket_types:
            base = select_top_per_race(gated, "shape_base_rank_score", ticket_type, f"{ticket_type}_{gate}_base", gate, 0.0)
            base_keys = base[["race_id", "pair_key_norm"]].rename(columns={"pair_key_norm": "base_pair_key_norm"})
            for weight in weights:
                if weight == 0:
                    selected = base.copy()
                    selected["policy"] = f"{ticket_type}_{gate}_base"
                    selected["shape_weight"] = 0.0
                else:
                    score_col = f"shape_adjusted_score_w{weight:.2f}"
                    risk_weight = min(0.12, weight * 0.75)
                    gated[score_col] = (
                        (1.0 - weight) * gated["shape_base_rank_score"]
                        + weight * gated["shape_pair_fit_score"]
                        + 0.06 * gated["shape_value_score"]
                        - risk_weight * gated["shape_pair_risk_score"]
                    )
                    selected = select_top_per_race(
                        gated,
                        score_col,
                        ticket_type,
                        f"{ticket_type}_{gate}_shape_w{weight:.2f}",
                        gate,
                        weight,
                    )
                selected = selected.merge(base_keys, on="race_id", how="left")
                selected["changed_from_base"] = selected["pair_key_norm"].ne(selected["base_pair_key_norm"])
                selections.append(selected)
                m = metrics(selected, selected["policy"].iloc[0] if not selected.empty else f"{ticket_type}_{gate}_{weight}")
                m["gate"] = gate
                m["ticket_type"] = ticket_type
                m["shape_weight"] = weight
                changed = selected[selected["changed_from_base"]].copy()
                cm = metrics(changed, "changed_only")
                m["changed_tickets"] = cm["tickets"]
                m["changed_roi_pct"] = cm["roi_pct"]
                m["changed_hit_rate_pct"] = cm["hit_rate_pct"]
                summary_rows.append(m)
                for year, gy in selected.groupby("year"):
                    ym = metrics(gy, selected["policy"].iloc[0])
                    ym["year"] = int(year)
                    ym["gate"] = gate
                    ym["ticket_type"] = ticket_type
                    ym["shape_weight"] = weight
                    yearly_rows.append(ym)

    detail = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    yearly = pd.DataFrame(yearly_rows)
    return summary, yearly, detail


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest race-shape adjusted pair pickup selection.")
    parser.add_argument("--universe", default=str(DEFAULT_UNIVERSE))
    parser.add_argument("--race-shape", default=str(DEFAULT_RACE_SHAPE))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--weights", default="0,0.04,0.08,0.12,0.16,0.20,0.28")
    parser.add_argument("--gates", default="all,value_loose,value_mid,value_strong,front_value_strong,price_sane_strong")
    parser.add_argument("--ticket-types", default="umaren,wide")
    args = parser.parse_args()

    universe_path = Path(args.universe)
    universe_path = universe_path if universe_path.is_absolute() else ROOT / universe_path
    race_shape_path = Path(args.race_shape)
    race_shape_path = race_shape_path if race_shape_path.is_absolute() else ROOT / race_shape_path
    out_dir = Path(args.output_dir)
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = [float(x.strip()) for x in args.weights.split(",") if x.strip()]
    gates = [x.strip() for x in args.gates.split(",") if x.strip()]
    ticket_types = [x.strip() for x in args.ticket_types.split(",") if x.strip()]

    df = load_universe(universe_path, race_shape_path)
    df = add_shape_scores(df)
    summary, yearly, detail = evaluate(df, gates, weights, ticket_types)
    if not summary.empty:
        summary = summary.sort_values(["roi_pct", "tickets"], ascending=[False, False])
    if not yearly.empty:
        yearly = yearly.sort_values(["policy", "year"])

    df[
        [
            "race_id",
            "year",
            "pair_key_norm",
            "anchor_no",
            "partner_no",
            "queue_shape_label",
            "shape_base_score",
            "shape_base_rank_score",
            "shape_pair_fit_score",
            "shape_pair_risk_score",
            "shape_pair_front_fit_component",
            "shape_pair_closer_fit_component",
        ]
    ].to_csv(out_dir / "pair_shape_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "shape_adjusted_pair_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(out_dir / "shape_adjusted_pair_yearly.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(out_dir / "shape_adjusted_pair_detail.csv", index=False, encoding="utf-8-sig")

    top = summary.head(30).replace({np.nan: None}).to_dict(orient="records") if not summary.empty else []
    by_base = summary[summary["shape_weight"].eq(0)].sort_values(["gate", "ticket_type"]).replace({np.nan: None}).to_dict(orient="records")
    report = {
        "universe": str(universe_path.relative_to(ROOT)),
        "rows": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "years": {str(k): int(v) for k, v in df["year"].value_counts().sort_index().to_dict().items()},
        "top_policies": top,
        "base_policies": by_base,
        "note": "This is a shadow pickup test. Shape weights are not yet promoted to production BUY gates; compare against base policy and yearly stability before adoption.",
    }
    (out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
