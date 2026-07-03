from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_hit_first_rollover_model import _consecutive_metrics, _daily_distribution, _flat_metrics
from src.utils.paths import ensure_dir, project_path


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _add_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "odds" in out.columns:
        odds = _num(out["odds"])
        out["odds_bin"] = pd.cut(
            odds,
            bins=[0, 1.2, 1.4, 1.6, 1.8, 2.1],
            labels=["lt1.2", "1.2-1.4", "1.4-1.6", "1.6-1.8", "1.8-2.0"],
            include_lowest=True,
        ).astype(str)
    if "距離" in out.columns:
        dist = _num(out["距離"])
        out["distance_bin"] = pd.cut(
            dist,
            bins=[0, 1300, 1600, 1900, 2400, 4000],
            labels=["sprint", "mile", "middle", "long", "extended"],
            include_lowest=True,
        ).astype(str)
    if "出走頭数" in out.columns:
        field = _num(out["出走頭数"])
    elif "頭数" in out.columns:
        field = _num(out["頭数"])
    else:
        field = pd.Series(np.nan, index=out.index)
    out["field_bin"] = pd.cut(
        field,
        bins=[0, 10, 13, 16, 30],
        labels=["small", "medium", "large", "full"],
        include_lowest=True,
    ).astype(str)
    if "ai_score_gap_to_second" in out.columns:
        gap = _num(out["ai_score_gap_to_second"])
        out["gap_bin"] = pd.cut(
            gap,
            bins=[0, 0.2, 0.25, 0.35, 999],
            labels=["0.20-0.25", "0.25-0.35", "0.35plus", "huge"],
            include_lowest=True,
        ).astype(str)
    if "jockey_venue_top3_rate" in out.columns:
        jv = _num(out["jockey_venue_top3_rate"])
        out["jockey_venue_bin"] = pd.cut(
            jv,
            bins=[0, 0.35, 0.45, 0.60, 1.0],
            labels=["0.30-0.35", "0.35-0.45", "0.45-0.60", "0.60plus"],
            include_lowest=True,
        ).astype(str)
    if "horse_closer_rate_past5" in out.columns:
        closer = _num(out["horse_closer_rate_past5"])
        out["closer_rate_bin"] = pd.cut(
            closer,
            bins=[-0.01, 0.0, 0.2, 0.5, 1.0],
            labels=["zero", "0-0.2", "0.2-0.5", "0.5plus"],
            include_lowest=True,
        ).astype(str)
    if "クラス名" in out.columns:
        cls = out["クラス名"].astype(str)
        out["class_group"] = np.select(
            [
                cls.str.contains("新馬", na=False),
                cls.str.contains("未勝利", na=False),
                cls.str.contains("1勝", na=False),
                cls.str.contains("2勝", na=False),
                cls.str.contains("3勝", na=False),
                cls.str.contains("OP|オープン|G", na=False),
            ],
            ["newcomer", "maiden", "1win", "2win", "3win", "open"],
            default="other",
        )
    return out


def _segment_summary(df: pd.DataFrame, col: str, min_n: int) -> pd.DataFrame:
    rows = []
    if col not in df.columns:
        return pd.DataFrame()
    for value, g in df.groupby(col, dropna=False):
        if len(g) < min_n:
            continue
        hits = g[g["hit"]]
        rows.append(
            {
                "segment": col,
                "value": str(value),
                "tickets": int(len(g)),
                "days": int(g["date_key"].nunique()) if "date_key" in g.columns else 0,
                "hit_rate": float(g["hit"].mean()),
                "avg_hit_return": float(hits["return_per100"].mean()) if len(hits) else 0.0,
                "median_hit_return": float(hits["return_per100"].median()) if len(hits) else 0.0,
                "roi_reference": float(g["return_per100"].sum() / (len(g) * 100.0)),
            }
        )
    return pd.DataFrame(rows)


def _evaluate_named_rules(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rules: dict[str, pd.Series] = {
        "base_all": pd.Series(True, index=df.index),
    }
    if "芝・ダ" in df.columns:
        rules["dirt_only"] = df["芝・ダ"].astype(str).str.contains("ダ", na=False)
        rules["turf_only"] = df["芝・ダ"].astype(str).str.contains("芝", na=False)
    if "class_group" in df.columns:
        rules["no_maiden"] = ~df["class_group"].isin(["maiden", "newcomer"])
        rules["maiden_only"] = df["class_group"].eq("maiden")
        rules["dirt_no_maiden"] = rules.get("dirt_only", pd.Series(True, index=df.index)) & rules["no_maiden"]
    if "馬場状態" in df.columns:
        going = df["馬場状態"].astype(str)
        rules["good_only"] = going.eq("良")
        rules["not_bad_going"] = going.isin(["良", "稍", "稍重"])
    if "field_bin" in df.columns:
        rules["not_full"] = ~df["field_bin"].eq("full")
        rules["small_medium"] = df["field_bin"].isin(["small", "medium"])
    if "odds_bin" in df.columns:
        rules["odds_1_2_to_1_6"] = df["odds_bin"].isin(["1.2-1.4", "1.4-1.6"])
        rules["odds_le_1_6"] = df["odds_bin"].isin(["lt1.2", "1.2-1.4", "1.4-1.6"])

    combo = dict(rules)
    for a in ["dirt_only", "no_maiden", "not_full", "odds_le_1_6", "good_only", "not_bad_going"]:
        for b in ["dirt_only", "no_maiden", "not_full", "odds_le_1_6", "good_only", "not_bad_going"]:
            if a >= b or a not in rules or b not in rules:
                continue
            combo[f"{a}__{b}"] = rules[a] & rules[b]
    if all(k in rules for k in ["dirt_only", "no_maiden", "not_full"]):
        combo["dirt_no_maiden_not_full"] = rules["dirt_only"] & rules["no_maiden"] & rules["not_full"]

    summary_rows = []
    consecutive_frames = []
    for name, mask in combo.items():
        part = df[mask.fillna(False)].copy()
        if len(part) < 8:
            continue
        row = _flat_metrics(part, name)
        summary_rows.append(row)
        cm = _consecutive_metrics(part.sort_values("sort_key"), [2, 3, 4, 5, 6])
        if not cm.empty:
            cm["rule"] = name
            consecutive_frames.append(cm)
    summary = pd.DataFrame(summary_rows).sort_values(["hit_rate", "tickets"], ascending=[False, False])
    consecutive = pd.concat(consecutive_frames, ignore_index=True, sort=False) if consecutive_frames else pd.DataFrame()
    return summary, consecutive


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze segments of danger-filtered hit-first rollover candidates.")
    parser.add_argument("--input-csv", default="outputs/analysis/hit_first_danger_filters_v1/validation_danger_filtered.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/hit_first_filtered_segments_v1")
    parser.add_argument("--min-segment-n", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv(project_path(args.input_csv), low_memory=False)
    df = _add_bins(df)
    segment_cols = [
        "芝・ダ",
        "class_group",
        "馬場状態",
        "distance_bin",
        "field_bin",
        "odds_bin",
        "gap_bin",
        "jockey_venue_bin",
        "closer_rate_bin",
        "枠番",
        "strategy_name",
    ]
    segments = pd.concat(
        [_segment_summary(df, col, args.min_segment_n) for col in segment_cols],
        ignore_index=True,
        sort=False,
    )
    rules, consecutive = _evaluate_named_rules(df)
    daily = _daily_distribution(df)

    out_dir = ensure_dir(project_path(args.output_dir))
    df.to_csv(out_dir / "filtered_candidates_with_bins.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(out_dir / "filtered_segment_summary.csv", index=False, encoding="utf-8-sig")
    rules.to_csv(out_dir / "filtered_rule_summary.csv", index=False, encoding="utf-8-sig")
    consecutive.to_csv(out_dir / "filtered_rule_consecutive_summary.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(out_dir / "filtered_daily_distribution.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "base": _flat_metrics(df, "base_all"),
        "best_segments": segments.sort_values(["hit_rate", "tickets"], ascending=[False, False]).head(20).to_dict(orient="records"),
        "weak_segments": segments.sort_values(["hit_rate", "tickets"], ascending=[True, False]).head(20).to_dict(orient="records"),
        "best_rules": rules.head(20).to_dict(orient="records"),
        "daily_occurrence": {
            "days": int(daily.shape[0]) if not daily.empty else 0,
            "avg_per_day": float(daily["qualifying_races"].mean()) if not daily.empty else 0.0,
            "median_per_day": float(daily["qualifying_races"].median()) if not daily.empty else 0.0,
            "max_per_day": int(daily["qualifying_races"].max()) if not daily.empty else 0,
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
