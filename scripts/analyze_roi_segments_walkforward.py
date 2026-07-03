from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_market_edge_pair_strategy import (  # noqa: E402
    _add_market_edge,
    _attach_wide_payoffs,
    _build_pair_candidates,
    _load_wide_payoffs,
)
from scripts.evaluate_ticket_strategies import _add_model_columns, _col, _num  # noqa: E402
from src.data.loaders import load_json_config  # noqa: E402
from src.utils.paths import ensure_dir, project_path  # noqa: E402


SEGMENT_COLUMNS = [
    "strategy",
    "venue",
    "surface",
    "distance_bin",
    "class_group",
    "going",
    "field_size_bin",
    "anchor_pop_band",
    "partner_pop_band",
    "partner_odds_band",
    "anchor_frame",
    "partner_frame",
    "anchor_style",
    "partner_style",
    "lap_profile",
    "rpci_bin",
    "field_market_shape",
]


def _label_bins(values: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(pd.to_numeric(values, errors="coerce"), bins=bins, labels=labels, include_lowest=True).astype("string").fillna("unknown")


def _year_from_date(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    yy = pd.to_numeric(text.str[:2], errors="coerce")
    return (2000 + yy).astype("Int64")


def _class_group(values: pd.Series) -> pd.Series:
    text = values.astype(str)
    out = pd.Series("other", index=values.index, dtype="string")
    out[text.str.contains("新馬", na=False)] = "newcomer"
    out[text.str.contains("未勝利", na=False)] = "maiden"
    out[text.str.contains("1勝", na=False)] = "1win"
    out[text.str.contains("2勝", na=False)] = "2win"
    out[text.str.contains("3勝", na=False)] = "3win"
    out[text.str.contains("OP|ｵｰﾌﾟﾝ|オープン", na=False)] = "open"
    out[text.str.contains("G1|G2|G3|重賞", na=False)] = "graded"
    return out


def _style(corner: pd.Series, field_size: pd.Series) -> pd.Series:
    pos = pd.to_numeric(corner, errors="coerce")
    heads = pd.to_numeric(field_size, errors="coerce")
    ratio = pos / heads.replace(0, np.nan)
    out = pd.Series("unknown", index=corner.index, dtype="string")
    out[ratio <= 0.25] = "front"
    out[(ratio > 0.25) & (ratio <= 0.50)] = "stalker"
    out[(ratio > 0.50) & (ratio <= 0.75)] = "midpack"
    out[ratio > 0.75] = "closer"
    return out


def _lap_profile(rpci: pd.Series) -> pd.Series:
    x = pd.to_numeric(rpci, errors="coerce")
    out = pd.Series("unknown", index=rpci.index, dtype="string")
    out[x <= 45] = "high_pace"
    out[(x > 45) & (x <= 50)] = "sustained"
    out[(x > 50) & (x <= 55)] = "balanced"
    out[x > 55] = "slow"
    return out


def _race_market_shape(scored: pd.DataFrame, race_col: str) -> pd.Series:
    prob = 1.0 / pd.to_numeric(scored["odds_decimal"], errors="coerce").replace(0, np.nan)
    share = prob / prob.groupby(scored[race_col]).transform("sum")
    entropy = -(share * np.log(share.replace(0, np.nan))).groupby(scored[race_col]).transform("sum")
    fav = share.groupby(scored[race_col]).transform("max")
    shape = pd.Series("normal", index=scored.index, dtype="string")
    shape[fav >= 0.38] = "strong_favorite"
    shape[(fav < 0.25) & (entropy >= entropy.quantile(0.65))] = "chaotic"
    shape[(fav < 0.30) & (entropy < entropy.quantile(0.35))] = "two_or_three_strong"
    return shape


def _ticket_metrics(frame: pd.DataFrame, label: str) -> dict[str, object]:
    if frame.empty:
        return {
            "label": label,
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
        "label": label,
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()),
        "hit_rate": float(frame["wide_hit"].mean()),
        "roi": float(frame["wide_return"].sum() / (len(frame) * 100.0)),
        "profit_flat100": float(frame["wide_return"].sum() - len(frame) * 100.0),
        "max_drawdown_flat100": float(drawdown.max()) if not drawdown.empty else 0.0,
    }


def _segment_metrics(tickets: pd.DataFrame, segment_col: str, min_tickets: int) -> pd.DataFrame:
    rows = []
    for value, part in tickets.groupby(segment_col, dropna=False):
        if len(part) < min_tickets:
            continue
        row = _ticket_metrics(part, str(value))
        row["segment"] = segment_col
        row["value"] = str(value)
        rows.append(row)
    return pd.DataFrame(rows)


def _profitable_rules(train: pd.DataFrame, *, min_tickets: int, min_roi: float) -> set[tuple[str, str]]:
    rules: set[tuple[str, str]] = set()
    for col in SEGMENT_COLUMNS:
        seg = _segment_metrics(train, col, min_tickets)
        if seg.empty:
            continue
        good = seg[(seg["roi"] >= min_roi) & (seg["profit_flat100"] > 0)]
        rules.update((col, str(value)) for value in good["value"])
    return rules


def _bad_rules(train: pd.DataFrame, *, min_tickets: int, max_roi: float) -> set[tuple[str, str]]:
    rules: set[tuple[str, str]] = set()
    for col in SEGMENT_COLUMNS:
        seg = _segment_metrics(train, col, min_tickets)
        if seg.empty:
            continue
        bad = seg[(seg["roi"] <= max_roi) & (seg["profit_flat100"] < 0)]
        rules.update((col, str(value)) for value in bad["value"])
    return rules


def _rule_score(frame: pd.DataFrame, rules: set[tuple[str, str]]) -> pd.Series:
    score = pd.Series(0, index=frame.index, dtype=int)
    for col, value in rules:
        score += frame[col].astype(str).eq(value).astype(int)
    return score


def _walk_forward(tickets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = sorted(int(y) for y in tickets["year"].dropna().unique())
    rows = []
    detail_frames = []
    for year in years[1:]:
        train = tickets[tickets["year"] < year].copy()
        valid = tickets[tickets["year"] == year].copy()
        if train.empty or valid.empty:
            continue

        policies = {
            "baseline_all": pd.Series(True, index=valid.index),
        }
        good_strict = _profitable_rules(train, min_tickets=80, min_roi=1.10)
        good_loose = _profitable_rules(train, min_tickets=120, min_roi=1.03)
        bad = _bad_rules(train, min_tickets=120, max_roi=0.80)
        policies["wf_good_rules_2plus_roi110"] = _rule_score(valid, good_strict) >= 2
        policies["wf_good_rules_1plus_roi103"] = _rule_score(valid, good_loose) >= 1
        policies["wf_drop_bad_rules_roi080"] = _rule_score(valid, bad) == 0
        policies["wf_good_1plus_and_drop_bad"] = ((_rule_score(valid, good_loose) >= 1) & (_rule_score(valid, bad) == 0))

        for name, mask in policies.items():
            part = valid[mask].copy()
            metric = _ticket_metrics(part, f"{name}_{year}")
            metric["policy"] = name
            metric["year"] = year
            metric["train_tickets"] = int(len(train))
            metric["good_rule_count_strict"] = len(good_strict)
            metric["good_rule_count_loose"] = len(good_loose)
            metric["bad_rule_count"] = len(bad)
            rows.append(metric)
            if not part.empty:
                tmp = part.copy()
                tmp["policy"] = name
                tmp["eval_year"] = year
                detail_frames.append(tmp)
    return pd.DataFrame(rows), pd.concat(detail_frames, ignore_index=True, sort=False) if detail_frames else pd.DataFrame()


def _enrich_tickets(scored: pd.DataFrame, tickets: pd.DataFrame, race_col: str) -> pd.DataFrame:
    date_col = _col(scored, ["日付"])
    venue_col = _col(scored, ["場所"])
    surface_col = _col(scored, ["芝・ダ", "芝ダ"])
    distance_col = _col(scored, ["距離"])
    class_col = _col(scored, ["クラス名"])
    going_col = _col(scored, ["馬場状態"])
    heads_col = _col(scored, ["頭数", "出走頭数"])
    frame_col = _col(scored, ["枠番"])
    horse_no_col = _col(scored, ["馬番"])
    corner_col = _col(scored, ["4角", "4角.1"])
    rpci_col = _col(scored, ["RPCI"])

    scored = scored.copy()
    scored["race_id"] = scored[race_col].astype(str)
    scored["year"] = _year_from_date(scored[date_col])
    scored["date_key"] = scored[date_col].astype(str).str.zfill(6)
    scored["field_market_shape"] = _race_market_shape(scored, race_col)

    meta_cols = [
        "race_id",
        "horse_name_for_ticket",
        date_col,
        venue_col,
        surface_col,
        distance_col,
        class_col,
        going_col,
        heads_col,
        frame_col,
        horse_no_col,
        corner_col,
        rpci_col,
        "year",
        "date_key",
        "field_market_shape",
    ]
    meta = scored[meta_cols].drop_duplicates()
    base = tickets.copy()
    base["race_id"] = base[race_col].astype(str)
    base = base.merge(meta.rename(columns={"horse_name_for_ticket": "a_horse"}), on=["race_id", "a_horse"], how="left")
    base = base.merge(
        meta.rename(columns={"horse_name_for_ticket": "b_horse"}),
        on=["race_id", "b_horse"],
        how="left",
        suffixes=("_a", "_b"),
    )

    base["venue"] = base[f"{venue_col}_a"].astype("string")
    base["surface"] = base[f"{surface_col}_a"].astype("string")
    base["distance_bin"] = _label_bins(base[f"{distance_col}_a"], [0, 1200, 1600, 2000, 2400, 10000], ["sprint", "mile", "middle", "classic", "long"])
    base["class_group"] = _class_group(base[f"{class_col}_a"])
    base["going"] = base[f"{going_col}_a"].astype("string")
    base["field_size_bin"] = _label_bins(base[f"{heads_col}_a"], [0, 9, 13, 16, 30], ["small", "medium", "large", "full"])
    base["anchor_pop_band"] = _label_bins(base["a_popularity"], [0, 1, 3, 6, 9, 99], ["fav1", "fav2_3", "pop4_6", "pop7_9", "pop10plus"])
    base["partner_pop_band"] = _label_bins(base["b_popularity"], [0, 1, 3, 6, 9, 99], ["fav1", "fav2_3", "pop4_6", "pop7_9", "pop10plus"])
    base["partner_odds_band"] = _label_bins(base["b_odds"], [0, 10, 20, 50, 100, 10000], ["lt10", "10_20", "20_50", "50_100", "100plus"])
    base["anchor_frame"] = base[f"{frame_col}_a"].astype("string")
    base["partner_frame"] = base[f"{frame_col}_b"].astype("string")
    base["anchor_style"] = _style(base[f"{corner_col}_a"], base[f"{heads_col}_a"])
    base["partner_style"] = _style(base[f"{corner_col}_b"], base[f"{heads_col}_a"])
    base["lap_profile"] = _lap_profile(base[f"{rpci_col}_a"])
    base["rpci_bin"] = _label_bins(base[f"{rpci_col}_a"], [0, 45, 50, 55, 100], ["rpci_le45", "rpci_45_50", "rpci_50_55", "rpci_gt55"])
    base["field_market_shape"] = base["field_market_shape_a"].astype("string")
    base["year"] = base["year_a"].astype("Int64")
    base["date_key"] = base["date_key_a"].astype(str)
    base["race_id_for_sort"] = base["race_id"]
    base["wide_return"] = base["wide_pay"].fillna(0.0).where(base["wide_hit"], 0.0)
    return base


def main() -> None:
    parser = argparse.ArgumentParser(description="ROI-first segment diagnostics and walk-forward gating for wide pair tickets.")
    parser.add_argument("--config", default="config/baseline_features_workout_optimized_core_same_day_bias.json")
    parser.add_argument("--model", default="models/workout_optimized_core_same_day_bias_v3_retro/baseline_ranker.pkl")
    parser.add_argument("--test-csv", default="data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/test_features_with_same_day_bias_v3_retro.csv")
    parser.add_argument("--wide-payoff-csv", default="data/processed/target/wide_payoffs.csv")
    parser.add_argument("--min-segment-tickets", type=int, default=60)
    parser.add_argument("--output-dir", default="outputs/analysis/roi_segments_walkforward")
    args = parser.parse_args()

    config = load_json_config(args.config)
    race_col = config["data"]["race_id_column"]
    df = pd.read_csv(project_path(args.test_csv), low_memory=False)
    scored = _add_model_columns(df, project_path(args.model), race_col)
    scored = _add_market_edge(scored, race_col)

    tickets = _build_pair_candidates(scored, race_col)
    from scripts.evaluate_market_edge_pair_strategy import _add_horse_numbers  # local import avoids widening public API

    tickets = _add_horse_numbers(tickets, scored, race_col)
    tickets = _attach_wide_payoffs(tickets, _load_wide_payoffs(args.wide_payoff_csv, race_col), race_col)
    enriched = _enrich_tickets(scored, tickets, race_col)

    out_dir = ensure_dir(project_path(args.output_dir))
    segment_frames = []
    for col in SEGMENT_COLUMNS:
        seg = _segment_metrics(enriched, col, args.min_segment_tickets)
        if not seg.empty:
            segment_frames.append(seg)
    segment_summary = pd.concat(segment_frames, ignore_index=True, sort=False) if segment_frames else pd.DataFrame()
    high = segment_summary.sort_values(["roi", "profit_flat100"], ascending=[False, False]).head(80)
    low = segment_summary.sort_values(["roi", "profit_flat100"], ascending=[True, True]).head(80)

    median_hit = float(segment_summary["hit_rate"].median()) if not segment_summary.empty else 0.0
    hit_high_roi_low = segment_summary[(segment_summary["hit_rate"] >= median_hit) & (segment_summary["roi"] < 0.90)].sort_values("profit_flat100")
    hit_low_roi_high = segment_summary[(segment_summary["hit_rate"] <= median_hit) & (segment_summary["roi"] > 1.10)].sort_values("roi", ascending=False)
    no_bet = segment_summary[(segment_summary["tickets"] >= 100) & (segment_summary["roi"] < 0.80)].sort_values("profit_flat100")

    wf_summary, wf_details = _walk_forward(enriched)
    yearly = (
        wf_details.groupby(["policy", "eval_year"], as_index=False)
        .apply(lambda g: pd.Series(_ticket_metrics(g, f"{g.name[0]}_{g.name[1]}")))
        .reset_index(drop=True)
        if not wf_details.empty
        else pd.DataFrame()
    )

    enriched.to_csv(out_dir / "wide_pair_tickets_enriched.csv", index=False, encoding="utf-8-sig")
    segment_summary.to_csv(out_dir / "segment_summary.csv", index=False, encoding="utf-8-sig")
    high.to_csv(out_dir / "high_roi_segments.csv", index=False, encoding="utf-8-sig")
    low.to_csv(out_dir / "low_roi_segments.csv", index=False, encoding="utf-8-sig")
    hit_high_roi_low.to_csv(out_dir / "hit_high_roi_low_segments.csv", index=False, encoding="utf-8-sig")
    hit_low_roi_high.to_csv(out_dir / "hit_low_roi_high_segments.csv", index=False, encoding="utf-8-sig")
    no_bet.to_csv(out_dir / "no_bet_candidate_segments.csv", index=False, encoding="utf-8-sig")
    wf_summary.to_csv(out_dir / "walkforward_policy_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(out_dir / "walkforward_yearly_roi.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "baseline": _ticket_metrics(enriched, "all_wide_pair_tickets"),
        "top_high_roi_segments": high.head(20).to_dict(orient="records"),
        "top_low_roi_segments": low.head(20).to_dict(orient="records"),
        "hit_high_roi_low_count": int(len(hit_high_roi_low)),
        "hit_low_roi_high_count": int(len(hit_low_roi_high)),
        "no_bet_candidate_count": int(len(no_bet)),
        "walkforward_summary": wf_summary.to_dict(orient="records"),
        "implemented_existing_data_features": [
            "distance_bin",
            "class_group",
            "field_size_bin",
            "anchor/partner popularity bands",
            "partner odds band",
            "anchor/partner style from 4角 relative position",
            "lap_profile and rpci_bin from race RPCI",
            "field_market_shape from live-available win odds distribution",
        ],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
