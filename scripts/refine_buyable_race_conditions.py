from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.paths import ensure_dir, project_path


def _num(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        cleaned = series.astype(str).str.replace(r"[(),]", "", regex=True).replace({"nan": np.nan, "": np.nan})
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def _metrics(g: pd.DataFrame) -> dict:
    if g.empty:
        return {}
    stake = len(g) * 100.0
    return {
        "races": int(g["race_id"].nunique()),
        "win_rate": float(g["is_win"].mean()),
        "place_rate": float(g["is_place"].mean()),
        "win_roi": float(_num(g["win_return_100"]).sum() / stake),
        "place_roi": float(_num(g["place_return_100"]).sum() / stake),
        "avg_pop": float(_num(g["pop_rank_num"]).mean()),
        "avg_odds": float(_num(g["odds_num"]).mean()),
        "avg_confidence": float(_num(g["race_confidence_score"]).mean()),
    }


def _add_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["pop_band"] = np.select(
        [
            _num(out["pop_rank_num"]).eq(1),
            _num(out["pop_rank_num"]).between(2, 3),
            _num(out["pop_rank_num"]).between(4, 6),
            _num(out["pop_rank_num"]).between(7, 9),
            _num(out["pop_rank_num"]).ge(10),
        ],
        ["pop1", "pop2_3", "pop4_6", "pop7_9", "pop10plus"],
        default="unknown",
    )
    out["odds_band"] = pd.cut(
        _num(out["odds_num"]),
        bins=[0, 2, 4, 8, 15, 999],
        labels=["odds_lt2", "odds2_4", "odds4_8", "odds8_15", "odds15plus"],
        include_lowest=True,
    ).astype(str)
    out["field_band"] = pd.cut(
        _num(out["field_size_num"]),
        bins=[0, 10, 13, 15, 99],
        labels=["field_small", "field_medium", "field_large", "field_full16plus"],
        include_lowest=True,
    ).astype(str)
    out["going_band"] = np.select(
        [
            out.get("馬場状態", "").astype(str).isin(["良"]),
            out.get("馬場状態", "").astype(str).isin(["稍"]),
            out.get("馬場状態", "").astype(str).isin(["重", "不"]),
        ],
        ["good", "yielding", "heavy_bad"],
        default="unknown",
    )
    out["venue_band"] = np.select(
        [
            out.get("venue", "").astype(str).isin(["札幌", "函館", "福島", "小倉"]),
            out.get("venue", "").astype(str).isin(["東京", "京都", "阪神", "中山", "中京", "新潟"]),
        ],
        ["local", "main"],
        default="unknown",
    )
    out["class_band"] = out.get("class_group", "unknown").astype(str)
    out["pace_band"] = out.get("expected_pace", "unknown").astype(str)
    out["surface_band"] = out.get("surface", "unknown").astype(str)
    out["distance_band"] = out.get("distance_bin", "unknown").astype(str)
    out["conf_band"] = pd.cut(
        _num(out["race_confidence_score"]),
        bins=[-0.01, 0.4, 0.65, 0.85, 1.01],
        labels=["conf_low", "conf_mid", "conf_high", "conf_very_high"],
        include_lowest=True,
    ).astype(str)
    out["danger_base"] = (
        out["going_band"].eq("heavy_bad")
        | out["pace_band"].eq("fast")
        | out["field_band"].eq("field_full16plus")
        | out.get("venue", "").astype(str).isin(["札幌", "小倉"])
        | out["class_band"].eq("open")
    )
    return out


def _segment_search(df: pd.DataFrame, cols: list[str], min_n: int) -> pd.DataFrame:
    rows = []
    for depth in [1, 2, 3]:
        for combo in combinations(cols, depth):
            grouped = df.groupby(list(combo), dropna=False)
            for keys, g in grouped:
                if len(g) < min_n:
                    continue
                if not isinstance(keys, tuple):
                    keys = (keys,)
                m = _metrics(g)
                m["segment"] = " & ".join(f"{c}={v}" for c, v in zip(combo, keys))
                m["depth"] = depth
                rows.append(m)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["roi_avg"] = (out["win_roi"] + out["place_roi"]) / 2
    out["hit_roi_balance"] = out["roi_avg"] * np.sqrt(out["place_rate"].clip(lower=0.001))
    return out.sort_values(["roi_avg", "races"], ascending=[False, False])


def _policy_rows(df: pd.DataFrame) -> pd.DataFrame:
    policies = {
        "baseline_ai1_all": pd.Series(True, index=df.index),
        "value_core_pop2_9_no_danger": df["pop_band"].isin(["pop2_3", "pop4_6", "pop7_9"]) & ~df["danger_base"],
        "value_core_pop4_9_no_danger": df["pop_band"].isin(["pop4_6", "pop7_9"]) & ~df["danger_base"],
        "main_good_pop2_6": df["venue_band"].eq("main") & df["going_band"].eq("good") & df["pop_band"].isin(["pop2_3", "pop4_6"]),
        "main_good_pop2_9_not_fast": df["venue_band"].eq("main") & df["going_band"].eq("good") & df["pop_band"].isin(["pop2_3", "pop4_6", "pop7_9"]) & ~df["pace_band"].eq("fast"),
        "local_buyable_pop2_6_good": df["venue_band"].eq("local") & df["going_band"].eq("good") & df["pop_band"].isin(["pop2_3", "pop4_6"]) & ~df["pace_band"].eq("fast"),
        "heavy_bad_exception_pop2_6": df["going_band"].eq("heavy_bad") & df["pop_band"].isin(["pop2_3", "pop4_6"]) & df["conf_band"].isin(["conf_high", "conf_very_high"]),
        "full_field_exception_pop2_6": df["field_band"].eq("field_full16plus") & df["pop_band"].isin(["pop2_3", "pop4_6"]) & df["conf_band"].isin(["conf_high", "conf_very_high"]),
        "open_exception_pop2_6": df["class_band"].eq("open") & df["pop_band"].isin(["pop2_3", "pop4_6"]) & df["conf_band"].isin(["conf_high", "conf_very_high"]),
        "avoid_pop1_danger": df["pop_band"].eq("pop1") & df["danger_base"],
        "avoid_fast_pop1": df["pop_band"].eq("pop1") & df["pace_band"].eq("fast"),
    }
    rows = []
    for name, cond in policies.items():
        g = df[cond].copy()
        m = _metrics(g)
        if m:
            m["policy"] = name
            m["share"] = float(len(g) / len(df))
            rows.append(m)
    return pd.DataFrame(rows).sort_values(["win_roi", "place_roi"], ascending=[False, False])


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine buyable and avoid race conditions for AI top1 betting.")
    parser.add_argument("--input-csv", default="outputs/analysis/race_confidence_gating_v1/race_confidence_scored_top1.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/buyable_race_conditions_v1")
    parser.add_argument("--min-n", type=int, default=80)
    args = parser.parse_args()

    df = pd.read_csv(project_path(args.input_csv), low_memory=False)
    df = _add_bins(df)
    out_dir = ensure_dir(project_path(args.output_dir))

    cols = [
        "pop_band",
        "odds_band",
        "venue",
        "venue_band",
        "surface_band",
        "distance_band",
        "going_band",
        "field_band",
        "pace_band",
        "class_band",
        "conf_band",
    ]
    segments = _segment_search(df, cols, args.min_n)
    danger_segments = _segment_search(df[df["danger_base"]], cols, max(30, args.min_n // 2))
    safe_segments = _segment_search(df[~df["danger_base"]], cols, args.min_n)
    policies = _policy_rows(df)

    buyable = segments[
        (segments["races"].ge(args.min_n))
        & (segments["win_roi"].ge(0.98))
        & (segments["place_roi"].ge(0.98))
        & (segments["place_rate"].ge(0.45))
    ].copy()
    avoid = segments[
        (segments["races"].ge(args.min_n))
        & (segments["win_roi"].le(0.85))
        & (segments["place_roi"].le(0.90))
    ].copy()
    danger_buyable = danger_segments[
        (danger_segments["races"].ge(max(30, args.min_n // 2)))
        & (danger_segments["win_roi"].ge(0.95))
        & (danger_segments["place_roi"].ge(0.95))
    ].copy()

    df.to_csv(out_dir / "ai1_condition_scored.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(out_dir / "condition_segments_all.csv", index=False, encoding="utf-8-sig")
    safe_segments.to_csv(out_dir / "condition_segments_safe_base.csv", index=False, encoding="utf-8-sig")
    danger_segments.to_csv(out_dir / "condition_segments_danger_base.csv", index=False, encoding="utf-8-sig")
    policies.to_csv(out_dir / "candidate_policy_summary.csv", index=False, encoding="utf-8-sig")
    buyable.head(80).to_csv(out_dir / "buyable_conditions_top.csv", index=False, encoding="utf-8-sig")
    avoid.head(80).to_csv(out_dir / "avoid_conditions_top.csv", index=False, encoding="utf-8-sig")
    danger_buyable.head(80).to_csv(out_dir / "danger_but_buyable_conditions_top.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "candidate_policies": policies.to_dict(orient="records"),
        "buyable_top20": buyable.head(20).to_dict(orient="records"),
        "avoid_top20": avoid.head(20).to_dict(orient="records"),
        "danger_but_buyable_top20": danger_buyable.head(20).to_dict(orient="records"),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
