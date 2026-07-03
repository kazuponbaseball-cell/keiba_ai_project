from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_track_regime_lap_context import (  # noqa: E402
    DEFAULT_FRONT,
    DEFAULT_TICKETS,
    DEFAULT_TRACK,
    load_races,
    load_ticket_context,
    metrics,
    race_segments,
    text,
    truthy,
)


DEFAULT_OUT = ROOT / "outputs/analysis/track_regime_lap_context_by_venue_v1"

VENUE_NAMES = {
    1: "札幌",
    2: "函館",
    3: "福島",
    4: "新潟",
    5: "東京",
    6: "中山",
    7: "中京",
    8: "京都",
    9: "阪神",
    10: "小倉",
}


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value):
        return None
    return value


def add_venue_name(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    code = pd.to_numeric(out.get("venue_code"), errors="coerce")
    out["venue_name"] = code.map(VENUE_NAMES).fillna(out.get("venue", "")).fillna("").astype(str)
    out["venue_code_num"] = code
    return out


def race_by_venue(races: pd.DataFrame) -> pd.DataFrame:
    rows = []
    masks = {
        "all_turf_track": pd.Series(True, index=races.index),
        "opening_week": races["opening_week_flag"].eq(1),
        "fresh_or_change": races["fresh_or_change"].fillna(False).astype(bool),
        "inside_front_top20": races["inside_front_bias_prior"].ge(races["inside_front_bias_prior"].quantile(0.80)),
        "outer_late_top20": races["outer_late_bias_prior"].ge(races["outer_late_bias_prior"].quantile(0.80)),
        "fresh_high_pressure": races["fresh_or_change"].fillna(False).astype(bool)
        & races["pre_front_load_signal"].ge(races["pre_front_load_signal"].quantile(0.60)),
        "fresh_pred_fastish": races["fresh_or_change"].fillna(False).astype(bool)
        & races["pred_fastish"].fillna(False).astype(bool),
    }
    for venue, vg in races.groupby("venue_name", dropna=False):
        for name, mask in masks.items():
            sub = races.loc[vg.index.intersection(mask[mask.fillna(False)].index)].copy()
            if len(sub) < 8:
                continue
            rows.append(
                {
                    "venue": venue,
                    "segment": name,
                    "races": int(len(sub)),
                    "front_survival_rate_pct": float(sub["actual_front_survival"].mean() * 100),
                    "front_collapse_rate_pct": float(sub["actual_front_collapse"].mean() * 100),
                    "winner_front5_rate_pct": float(sub["winner_front5"].mean() * 100),
                    "actual_fast_or_frontload_rate_pct": float(truthy(sub["actual_fast_or_frontload"]).mean() * 100),
                    "avg_front3f_vs_course_prior": float(
                        (pd.to_numeric(sub["front3f_sec"], errors="coerce") - pd.to_numeric(sub["course_front3f_prior_sec"], errors="coerce")).mean()
                    ),
                    "avg_rpci": float(pd.to_numeric(sub["rpci"], errors="coerce").mean()),
                    "avg_inside_front_prior": float(pd.to_numeric(sub["inside_front_bias_prior"], errors="coerce").mean()),
                    "avg_outer_late_prior": float(pd.to_numeric(sub["outer_late_bias_prior"], errors="coerce").mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["segment", "front_survival_rate_pct", "races"], ascending=[True, False, False])


def build_ticket_masks(t: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, pd.Series]:
    front_any = truthy(t.get("pair_pred_front5_any", pd.Series(False, index=t.index)))
    front_both = truthy(t.get("pair_pred_front5_both", pd.Series(False, index=t.index)))
    leader_any = truthy(t.get("pair_pred_leader_any", pd.Series(False, index=t.index)))
    complement = truthy(t.get("pair_pred_front_complement", pd.Series(False, index=t.index)))
    pressure60 = t["pre_front_load_signal"].ge(thresholds["pressure_q60"])
    survival70 = t["context_survival_prior"].ge(t.loc[t["year"].lt(2026), "context_survival_prior"].quantile(0.70))
    inside80 = t["inside_front_bias_prior"].ge(thresholds["inside_q80"])
    outer80 = t["outer_late_bias_prior"].ge(thresholds["outer_q80"])
    wear80 = t["inner_wear_proxy"].ge(thresholds["wear_q80"])
    fresh = t["fresh_or_change"].fillna(False).astype(bool)
    fastish = t["pred_fastish"].fillna(False).astype(bool)
    return {
        "base_turf_track": pd.Series(True, index=t.index),
        "fresh_front_any": fresh & front_any,
        "fresh_fastish_front_any": fresh & fastish & front_any,
        "fresh_pressure_front_any": fresh & pressure60 & front_any,
        "fresh_pressure_survival_front_any": fresh & pressure60 & survival70 & front_any,
        "inside_q80_front_any": inside80 & front_any,
        "inside_q80_front_both": inside80 & front_both,
        "inside_q80_leader_any": inside80 & leader_any,
        "outer_q80_front_any": outer80 & front_any,
        "outer_q80_front_complement": outer80 & complement,
        "outer_q80_avoid_front_both": outer80 & ~front_both,
        "wear_q80_front_any": wear80 & front_any,
    }


def ticket_by_venue(t: pd.DataFrame, thresholds: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    masks = build_ticket_masks(t, thresholds)
    rows = []
    by_year = []
    for venue, vg in t.groupby("venue_name", dropna=False):
        for ticket_type in ["wide", "umaren"]:
            tm = text(vg, "ticket_type").eq(ticket_type)
            for name, mask in masks.items():
                sub = vg.loc[tm & mask.reindex(vg.index).fillna(False)].copy()
                if len(sub) < 20:
                    continue
                label = f"{ticket_type}::{name}"
                row = metrics(sub, label)
                row["venue"] = venue
                row["ticket_type"] = ticket_type
                row["segment"] = name
                rows.append(row)
                for year, gy in sub.groupby("year", dropna=False):
                    if len(gy) < 5:
                        continue
                    yr = metrics(gy, label)
                    yr["venue"] = venue
                    yr["ticket_type"] = ticket_type
                    yr["segment"] = name
                    yr["year"] = int(year)
                    by_year.append(yr)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out[
            [
                "venue",
                "ticket_type",
                "segment",
                "policy",
                "tickets",
                "races",
                "stake_yen",
                "return_yen",
                "profit_yen",
                "roi_pct",
                "hit_rate_pct",
                "roi_ex_top1_pct",
                "roi_ex_top5_pct",
                "top_return_share_pct",
            ]
        ].sort_values(["ticket_type", "segment", "roi_ex_top5_pct", "tickets"], ascending=[True, True, False, False])
    year = pd.DataFrame(by_year)
    if not year.empty:
        year = year.sort_values(["ticket_type", "segment", "venue", "year"])
    return out, year


def venue_recommendations(ticket_metrics: pd.DataFrame, year_metrics: pd.DataFrame) -> pd.DataFrame:
    if ticket_metrics.empty:
        return pd.DataFrame()
    rows = []
    for _, row in ticket_metrics.iterrows():
        if row["ticket_type"] != "wide":
            continue
        if int(row["tickets"]) < 80:
            continue
        y = year_metrics[
            (year_metrics["venue"].eq(row["venue"]))
            & (year_metrics["ticket_type"].eq(row["ticket_type"]))
            & (year_metrics["segment"].eq(row["segment"]))
        ]
        y2026 = y[y["year"].eq(2026)]
        y2025 = y[y["year"].eq(2025)]
        status = "shadow_only"
        reason = []
        if row["roi_ex_top5_pct"] >= 115 and row["hit_rate_pct"] >= 10:
            status = "watchlist"
            reason.append("全体ROIは良好")
        if not y2026.empty and float(y2026["roi_pct"].iloc[0]) < 80:
            status = "shadow_only"
            reason.append("2026で崩れ")
        if not y2026.empty and int(y2026["tickets"].iloc[0]) < 50:
            reason.append("2026件数不足")
        if not y2025.empty and not y2026.empty and float(y2025["roi_pct"].iloc[0]) >= 120 and float(y2026["roi_pct"].iloc[0]) >= 100:
            status = "promotion_candidate"
            reason.append("2025/2026ともプラス")
        rows.append(
            {
                "venue": row["venue"],
                "segment": row["segment"],
                "tickets": int(row["tickets"]),
                "races": int(row["races"]),
                "roi_pct": float(row["roi_pct"]),
                "roi_ex_top5_pct": float(row["roi_ex_top5_pct"]),
                "hit_rate_pct": float(row["hit_rate_pct"]),
                "status": status,
                "reason": " / ".join(reason) if reason else "明確な採用根拠不足",
            }
        )
    return pd.DataFrame(rows).sort_values(["status", "roi_ex_top5_pct", "tickets"], ascending=[True, False, False])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track-races", type=Path, default=DEFAULT_TRACK)
    parser.add_argument("--front-races", type=Path, default=DEFAULT_FRONT)
    parser.add_argument("--tickets", type=Path, default=DEFAULT_TICKETS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    races = add_venue_name(load_races(args.track_races, args.front_races))
    _, thresholds = race_segments(races)
    tickets = load_ticket_context(args.tickets, races)
    tickets = tickets.merge(races[["race_id", "venue_code", "venue", "venue_name"]].drop_duplicates("race_id"), on="race_id", how="left")
    tickets = add_venue_name(tickets)

    race_out = race_by_venue(races)
    ticket_out, ticket_year = ticket_by_venue(tickets, thresholds)
    rec = venue_recommendations(ticket_out, ticket_year)

    race_out.to_csv(args.out_dir / "race_track_regime_lap_by_venue.csv", index=False, encoding="utf-8-sig")
    ticket_out.to_csv(args.out_dir / "ticket_track_regime_lap_by_venue.csv", index=False, encoding="utf-8-sig")
    ticket_year.to_csv(args.out_dir / "ticket_track_regime_lap_by_venue_year.csv", index=False, encoding="utf-8-sig")
    rec.to_csv(args.out_dir / "venue_track_regime_recommendations.csv", index=False, encoding="utf-8-sig")

    top_watch = rec[rec["status"].isin(["promotion_candidate", "watchlist"])].head(20)
    top_risk = ticket_out[
        (ticket_out["ticket_type"].eq("wide"))
        & (ticket_out["tickets"].ge(80))
        & (ticket_out["roi_ex_top5_pct"].lt(80))
    ].sort_values(["roi_ex_top5_pct", "tickets"]).head(20)
    summary = {
        "output_dir": str(args.out_dir.relative_to(ROOT)),
        "race_count": int(len(races)),
        "ticket_count": int(len(tickets)),
        "venues": sorted(races["venue_name"].dropna().unique().tolist()),
        "top_watchlist": top_watch.to_dict(orient="records"),
        "top_risk_segments": top_risk.to_dict(orient="records"),
        "status_counts": rec["status"].value_counts().to_dict() if not rec.empty else {},
        "note": (
            "This breaks track-regime/lap policies down by venue. Promotion still requires live shadow accumulation, "
            "because 2026 venue samples are thin for many courses."
        ),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Track Regime Lap Context By Venue v1",
                "",
                "Venue-level breakdown for opening week / course-change / inside-front / outer-late lap policies.",
                "",
                "Key outputs:",
                "- race_track_regime_lap_by_venue.csv",
                "- ticket_track_regime_lap_by_venue.csv",
                "- ticket_track_regime_lap_by_venue_year.csv",
                "- venue_track_regime_recommendations.csv",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
