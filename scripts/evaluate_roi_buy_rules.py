from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import ensure_dir, project_path


def _metrics(frame: pd.DataFrame, label: str) -> dict[str, object]:
    if frame.empty:
        return {
            "rule": label,
            "tickets": 0,
            "races": 0,
            "hit_rate": 0.0,
            "roi": 0.0,
            "profit_flat100": 0.0,
            "max_drawdown_flat100": 0.0,
        }
    ordered = frame.sort_values(["date_key", "race_id_for_sort", "strategy", "a_horse", "b_horse"]).copy()
    profit = ordered["wide_return"].fillna(0.0) - 100.0
    curve = profit.cumsum()
    drawdown = curve.cummax() - curve
    return {
        "rule": label,
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()),
        "hit_rate": float(frame["wide_hit"].mean()),
        "roi": float(frame["wide_return"].sum() / (len(frame) * 100.0)),
        "profit_flat100": float(frame["wide_return"].sum() - len(frame) * 100.0),
        "max_drawdown_flat100": float(drawdown.max()) if not drawdown.empty else 0.0,
    }


def _yearly_metrics(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for year, part in frame.groupby("year"):
        row = _metrics(part, label)
        row["year"] = int(year)
        rows.append(row)
    return pd.DataFrame(rows)


def _rules(df: pd.DataFrame) -> dict[str, pd.Series]:
    bad = (
        df["partner_style"].isin(["closer"])
        | df["class_group"].isin(["newcomer"])
        | df["field_size_bin"].isin(["full"])
        | df["going"].isin(["稍"])
        | df["venue"].isin(["福島"])
    )
    return {
        "baseline_all": pd.Series(True, index=df.index),
        "strong_value_only": df["strategy"].eq("anchor_x_strong_value_top1"),
        "partner_front": df["partner_style"].eq("front"),
        "partner_front_odds20_100": df["partner_style"].eq("front") & df["partner_odds_band"].isin(["20_50", "50_100"]),
        "partner_front_pop10plus": df["partner_style"].eq("front") & df["partner_pop_band"].eq("pop10plus"),
        "strong_front": df["strategy"].eq("anchor_x_strong_value_top1") & df["partner_style"].eq("front"),
        "strong_front_or_pop10": df["strategy"].eq("anchor_x_strong_value_top1") & (df["partner_style"].eq("front") | df["partner_pop_band"].eq("pop10plus")),
        "avoid_bad_only": ~bad,
        "front_or_pop10_no_bad": (df["partner_style"].eq("front") | df["partner_pop_band"].eq("pop10plus")) & ~bad,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed buy/skip rules for ROI-first wide tickets.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/roi_segments_walkforward_v1/wide_pair_tickets_enriched.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/roi_buy_rules_v1")
    args = parser.parse_args()

    df = pd.read_csv(project_path(args.tickets_csv), low_memory=False)
    out_dir = ensure_dir(project_path(args.output_dir))

    summary_rows = []
    yearly_frames = []
    detail_frames = []
    for name, mask in _rules(df).items():
        part = df[mask].copy()
        summary_rows.append(_metrics(part, name))
        yearly = _yearly_metrics(part, name)
        yearly_frames.append(yearly)
        if not part.empty:
            tmp = part.copy()
            tmp["buy_rule"] = name
            detail_frames.append(tmp)

    summary = pd.DataFrame(summary_rows).sort_values(["roi", "profit_flat100"], ascending=[False, False])
    yearly = pd.concat(yearly_frames, ignore_index=True, sort=False) if yearly_frames else pd.DataFrame()
    details = pd.concat(detail_frames, ignore_index=True, sort=False) if detail_frames else pd.DataFrame()

    summary.to_csv(out_dir / "buy_rule_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(out_dir / "buy_rule_yearly_roi.csv", index=False, encoding="utf-8-sig")
    details.to_csv(out_dir / "buy_rule_ticket_details.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "summary": summary.to_dict(orient="records"),
        "recommended_primary_rule": "partner_front_odds20_100",
        "recommended_balanced_rule": "front_or_pop10_no_bad",
        "recommended_skip_logic": "skip partner closers, newcomers, full fields, slightly-heavy going, and Fukushima for this wide overlay until revalidated.",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
