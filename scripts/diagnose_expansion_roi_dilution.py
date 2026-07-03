from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


VENUE_CODE_MAP = {
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


def num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def infer_venue(race_id: pd.Series) -> pd.Series:
    return race_id.astype(str).str.zfill(16).str[8:10].map(VENUE_CODE_MAP).fillna("Unknown")


def max_drawdown(profit: pd.Series) -> float:
    if profit.empty:
        return 0.0
    equity = profit.cumsum()
    return float((equity - equity.cummax()).min())


def ticket_key(df: pd.DataFrame) -> pd.Series:
    if "ticket_key" in df.columns:
        return df["ticket_key"].astype(str)
    a = num(df.get("horse_a"), df.index, np.nan)
    b = num(df.get("horse_b"), df.index, np.nan)
    anchor = num(df.get("anchor_no"), df.index, np.nan)
    partner = num(df.get("partner_no"), df.index, np.nan)
    a = a.fillna(np.minimum(anchor, partner))
    b = b.fillna(np.maximum(anchor, partner))
    return (
        df["ticket_type"].astype(str)
        + ":"
        + df["race_id"].astype(str)
        + ":"
        + a.astype("Int64").astype(str)
        + "-"
        + b.astype("Int64").astype(str)
    )


def normalize(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df.copy()
    out["source_label"] = label
    out["race_id"] = out["race_id"].astype(str)
    out["year"] = num(out.get("year"), out.index, np.nan).fillna(out["race_id"].str[:4].astype(int)).astype(int)
    out["venue_eval"] = out.get("venue", infer_venue(out["race_id"]))
    out["venue_eval"] = out["venue_eval"].astype("string").fillna("")
    out["venue_eval"] = out["venue_eval"].mask(out["venue_eval"].eq(""), infer_venue(out["race_id"]))
    out["ticket_type"] = out.get("ticket_type", "").astype(str)
    out["ticket_key_eval"] = ticket_key(out)
    stake = num(out.get("runtime_stake_yen"), out.index, np.nan)
    out["eval_stake_yen"] = stake.where(stake.gt(0), num(out.get("stake_yen"), out.index, 0.0)).fillna(0.0)
    ret = num(out.get("runtime_return_yen"), out.index, np.nan)
    out["eval_return_yen"] = ret.where(ret.ge(0), num(out.get("return_yen"), out.index, 0.0)).fillna(0.0)
    out["eval_profit_yen"] = out["eval_return_yen"] - out["eval_stake_yen"]
    out["hit_eval"] = out.get("hit", False).astype(bool)
    out["anchor_pop_bin"] = pd.cut(
        num(out.get("anchor_pop"), out.index, np.nan),
        bins=[0, 1, 2, 3, 5, 9, 99],
        labels=["1", "2", "3", "4-5", "6-9", "10+"],
        include_lowest=True,
    ).astype("string").fillna("unknown")
    out["partner_pop_bin"] = pd.cut(
        num(out.get("partner_pop"), out.index, np.nan),
        bins=[0, 3, 5, 9, 99],
        labels=["1-3", "4-5", "6-9", "10+"],
        include_lowest=True,
    ).astype("string").fillna("unknown")
    out["partner_odds_bin"] = pd.cut(
        num(out.get("partner_odds"), out.index, np.nan),
        bins=[0, 10, 20, 40, 80, 999],
        labels=["<=10", "10-20", "20-40", "40-80", "80+"],
        include_lowest=True,
    ).astype("string").fillna("unknown")
    out["overlay_bin"] = pd.cut(
        num(out.get("market_overlay_score"), out.index, np.nan),
        bins=[-0.001, 0.55, 0.70, 0.85, 1.001],
        labels=["low", "mid", "high", "extreme"],
        include_lowest=True,
    ).astype("string").fillna("unknown")
    out["front5_bin"] = pd.cut(
        num(out.get("projected_front5_prob"), out.index, np.nan),
        bins=[-0.001, 0.45, 0.60, 0.75, 1.001],
        labels=["low", "mid", "high", "very_high"],
        include_lowest=True,
    ).astype("string").fillna("unknown")
    out["pair_score_bin"] = pd.cut(
        num(out.get("pair_score"), out.index, np.nan),
        bins=[-0.001, 0.60, 0.70, 0.80, 1.001],
        labels=["low", "mid", "high", "very_high"],
        include_lowest=True,
    ).astype("string").fillna("unknown")
    out["pair_quinella_bin"] = pd.cut(
        num(out.get("pair_quinella_score"), out.index, np.nan),
        bins=[-0.001, 0.50, 0.60, 0.70, 1.001],
        labels=["low", "mid", "high", "very_high"],
        include_lowest=True,
    ).astype("string").fillna("unknown")
    out["danger_bin"] = pd.cut(
        num(out.get("partner_danger"), out.index, np.nan).fillna(0.0)
        + num(out.get("anchor_danger"), out.index, np.nan).fillna(0.0),
        bins=[-0.001, 0.30, 0.70, 2.001],
        labels=["low", "mid", "high"],
        include_lowest=True,
    ).astype("string").fillna("unknown")
    if "going" in out.columns:
        out["going_eval"] = out["going"].astype("string").fillna("unknown")
    elif "going_raw" in out.columns:
        out["going_eval"] = out["going_raw"].astype("string").fillna("unknown")
    elif "馬場状態" in out.columns:
        out["going_eval"] = out["馬場状態"].astype("string").fillna("unknown")
    else:
        out["going_eval"] = "unknown"
    if "surface_raw" in out.columns:
        out["surface_eval"] = out["surface_raw"].astype("string").fillna("unknown")
    elif "surface" in out.columns:
        out["surface_eval"] = out["surface"].astype("string").fillna("unknown")
    else:
        out["surface_eval"] = "unknown"
    return out[out["eval_stake_yen"].gt(0)].copy()


def metrics(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {
            "label": label,
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
    race = (
        df.groupby("race_id", sort=False)
        .agg(stake_yen=("eval_stake_yen", "sum"), return_yen=("eval_return_yen", "sum"), hit=("hit_eval", "max"))
        .reset_index()
    )
    race["profit_yen"] = race["return_yen"] - race["stake_yen"]
    stake = float(df["eval_stake_yen"].sum())
    ret = float(df["eval_return_yen"].sum())
    return {
        "label": label,
        "tickets": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake,
        "roi": ret / stake if stake else 0.0,
        "ticket_hit_rate": float(df["hit_eval"].mean()),
        "race_hit_rate": float(race["hit"].mean()) if len(race) else 0.0,
        "max_drawdown_yen": max_drawdown(race["profit_yen"]),
    }


def grouped(df: pd.DataFrame, columns: list[str], min_tickets: int, label: str) -> pd.DataFrame:
    rows = []
    for key, part in df.groupby(columns, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        if len(part) < min_tickets:
            continue
        row = {col: val for col, val in zip(columns, key)}
        row.update(metrics(part, label))
        rows.append(row)
    return pd.DataFrame(rows)


def top_profit_removed(df: pd.DataFrame, top_n: int) -> dict:
    race = (
        df.groupby("race_id", sort=False)
        .agg(stake_yen=("eval_stake_yen", "sum"), return_yen=("eval_return_yen", "sum"))
        .reset_index()
    )
    race["profit_yen"] = race["return_yen"] - race["stake_yen"]
    kept = set(race.sort_values("profit_yen", ascending=False).iloc[top_n:]["race_id"])
    return metrics(df[df["race_id"].isin(kept)].copy(), f"minus_top{top_n}")


def analyze_combos(df: pd.DataFrame, dimensions: list[str], min_tickets: int) -> pd.DataFrame:
    frames = []
    for width in [2, 3]:
        for cols in combinations(dimensions, width):
            part = grouped(df, list(cols), min_tickets, "combo")
            if part.empty:
                continue
            part.insert(0, "combo_dims", "+".join(cols))
            frames.append(part)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose why expanded ticket range dilutes ROI compared with final standard selection.")
    parser.add_argument("--expanded-csv", default="outputs/analysis/extended_period_validation_v1/fixed_proxy_selected_tickets_2024_2026.csv")
    parser.add_argument("--standard-csv", default="outputs/analysis/final_operational_quality_v1/standard_explained_tickets.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/expansion_roi_dilution_v1")
    parser.add_argument("--min-tickets", type=int, default=40)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    expanded = normalize(pd.read_csv(args.expanded_csv, dtype={"race_id": str}, low_memory=False), "expanded")
    standard = normalize(pd.read_csv(args.standard_csv, dtype={"race_id": str}, low_memory=False), "standard")
    standard_keys = set(standard["ticket_key_eval"])
    expanded["in_standard"] = expanded["ticket_key_eval"].isin(standard_keys)
    extra = expanded[~expanded["in_standard"]].copy()
    overlap = expanded[expanded["in_standard"]].copy()

    summary_rows = [
        metrics(standard, "standard_final"),
        metrics(expanded, "expanded_all"),
        metrics(overlap, "expanded_overlap_standard_keys"),
        metrics(extra, "expanded_extra_not_in_standard"),
        top_profit_removed(standard, 5) | {"label": "standard_minus_top5"},
        top_profit_removed(standard, 10) | {"label": "standard_minus_top10"},
        top_profit_removed(expanded, 5) | {"label": "expanded_minus_top5"},
        top_profit_removed(expanded, 10) | {"label": "expanded_minus_top10"},
        top_profit_removed(extra, 5) | {"label": "extra_minus_top5"},
        top_profit_removed(extra, 10) | {"label": "extra_minus_top10"},
    ]
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "dilution_summary.csv", index=False, encoding="utf-8-sig")

    dims = [
        "ticket_type",
        "venue_eval",
        "year",
        "going_eval",
        "surface_eval",
        "anchor_pop_bin",
        "partner_pop_bin",
        "partner_odds_bin",
        "overlay_bin",
        "front5_bin",
        "pair_score_bin",
        "pair_quinella_bin",
        "danger_bin",
    ]
    seg_frames = []
    for col in dims:
        s_extra = grouped(extra, [col], args.min_tickets, "extra")
        if not s_extra.empty:
            s_extra.insert(0, "segment_dim", col)
            seg_frames.append(s_extra)
    segment_summary = pd.concat(seg_frames, ignore_index=True, sort=False) if seg_frames else pd.DataFrame()
    segment_summary.to_csv(out_dir / "extra_segment_summary.csv", index=False, encoding="utf-8-sig")

    combo_summary = analyze_combos(extra, dims, args.min_tickets)
    combo_summary.to_csv(out_dir / "extra_combo_summary.csv", index=False, encoding="utf-8-sig")

    if not segment_summary.empty:
        good = segment_summary[(segment_summary["roi"].ge(1.25)) & (segment_summary["races"].ge(40))].sort_values(
            ["roi", "profit_yen"], ascending=False
        )
        bad = segment_summary[(segment_summary["roi"].lt(0.90)) & (segment_summary["races"].ge(40))].sort_values(
            ["profit_yen", "roi"], ascending=[True, True]
        )
    else:
        good = pd.DataFrame()
        bad = pd.DataFrame()
    if not combo_summary.empty:
        combo_good = combo_summary[(combo_summary["roi"].ge(1.35)) & (combo_summary["races"].ge(30))].sort_values(
            ["roi", "profit_yen"], ascending=False
        )
        combo_bad = combo_summary[(combo_summary["roi"].lt(0.85)) & (combo_summary["races"].ge(30))].sort_values(
            ["profit_yen", "roi"], ascending=[True, True]
        )
    else:
        combo_good = pd.DataFrame()
        combo_bad = pd.DataFrame()

    good.head(100).to_csv(out_dir / "candidate_expand_segments.csv", index=False, encoding="utf-8-sig")
    bad.head(100).to_csv(out_dir / "danger_dilution_segments.csv", index=False, encoding="utf-8-sig")
    combo_good.head(150).to_csv(out_dir / "candidate_expand_combos.csv", index=False, encoding="utf-8-sig")
    combo_bad.head(150).to_csv(out_dir / "danger_dilution_combos.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "summary": summary.to_dict(orient="records"),
        "top_candidate_segments": good.head(10).to_dict(orient="records") if not good.empty else [],
        "top_danger_segments": bad.head(10).to_dict(orient="records") if not bad.empty else [],
        "top_candidate_combos": combo_good.head(10).to_dict(orient="records") if not combo_good.empty else [],
        "top_danger_combos": combo_bad.head(10).to_dict(orient="records") if not combo_bad.empty else [],
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
