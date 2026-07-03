from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data" / "datasets" / "train" / "baseline_temporal_test_dataset.csv"
DEFAULT_INVEST = (
    ROOT
    / "outputs"
    / "analysis"
    / "investment_decision_features_rebuilt_20260623"
    / "investment_features_scored.csv"
)
DEFAULT_STRONGEST = (
    ROOT
    / "outputs"
    / "analysis"
    / "age_weight_surface_overlay_mcs_v4_v2"
    / "best_policy_tickets.csv"
)
DEFAULT_OUT = ROOT / "outputs" / "analysis" / "trainer_venue_expedition_v1"


VENUE_REGION = {
    "東京": "east",
    "中山": "east",
    "福島": "east",
    "新潟": "east",
    "京都": "west",
    "阪神": "west",
    "中京": "west",
    "小倉": "west",
    "札幌": "hokkaido",
    "函館": "hokkaido",
}
LOCAL_VENUES = {"札幌", "函館", "福島", "新潟", "小倉"}


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def normalize_race_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(16)


def find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"missing any of columns: {candidates}")


def add_shifted_group_rates(
    df: pd.DataFrame,
    group_cols: list[str],
    prefix: str,
    sort_cols: list[str],
) -> pd.DataFrame:
    out = df.copy()
    ordered = out.sort_values(sort_cols, kind="mergesort")
    grouped = ordered.groupby(group_cols, sort=False, dropna=False)

    starts = grouped.cumcount().astype(float)
    prev_win = grouped["is_win"].cumsum() - ordered["is_win"]
    prev_top3 = grouped["is_top3"].cumsum() - ordered["is_top3"]
    prev_score = grouped["score_for_history"].cumsum() - ordered["score_for_history"]
    prev_pop_out = grouped["is_pop_outperform"].cumsum() - ordered["is_pop_outperform"]

    denom = starts.replace(0.0, np.nan)
    out.loc[ordered.index, f"{prefix}_starts"] = starts
    out.loc[ordered.index, f"{prefix}_win_rate"] = (prev_win / denom).fillna(0.0)
    out.loc[ordered.index, f"{prefix}_top3_rate"] = (prev_top3 / denom).fillna(0.0)
    out.loc[ordered.index, f"{prefix}_avg_score"] = (prev_score / denom).fillna(0.0)
    out.loc[ordered.index, f"{prefix}_pop_outperform_rate"] = (prev_pop_out / denom).fillna(0.0)
    return out


def load_and_build_runner_features(train_path: Path) -> pd.DataFrame:
    df = read_csv(train_path)

    date_col = find_col(df, ["日付S", "日付"])
    venue_col = find_col(df, ["場所", "venue"])
    race_col = find_col(df, ["レースID(新/馬番無)", "race_id"])
    horse_no_col = find_col(df, ["馬番", "horse_no"])
    trainer_col = find_col(df, ["調教師コード", "trainer_code"])
    finish_col = find_col(df, ["確定着順", "finish_num", "finish"])
    pop_col = find_col(df, ["人気", "pop_rank", "popularity"])

    work = df[
        [
            date_col,
            venue_col,
            race_col,
            horse_no_col,
            trainer_col,
            finish_col,
            pop_col,
            "target_score" if "target_score" in df.columns else finish_col,
            "芝・ダ" if "芝・ダ" in df.columns else venue_col,
            "距離" if "距離" in df.columns else venue_col,
            "クラス名" if "クラス名" in df.columns else venue_col,
            "馬場状態" if "馬場状態" in df.columns else venue_col,
            "異常コード" if "異常コード" in df.columns else finish_col,
        ]
    ].copy()

    # Deduplicate any placeholder fallback columns selected above.
    work = work.loc[:, ~work.columns.duplicated()]
    rename = {
        date_col: "date",
        venue_col: "venue",
        race_col: "race_id",
        horse_no_col: "horse_no",
        trainer_col: "trainer_code",
        finish_col: "finish_num",
        pop_col: "pop_rank",
    }
    if "target_score" in work.columns:
        rename["target_score"] = "target_score"
    if "芝・ダ" in work.columns:
        rename["芝・ダ"] = "surface"
    if "距離" in work.columns:
        rename["距離"] = "distance"
    if "クラス名" in work.columns:
        rename["クラス名"] = "class_name"
    if "馬場状態" in work.columns:
        rename["馬場状態"] = "going"
    if "異常コード" in work.columns:
        rename["異常コード"] = "abnormal_code"
    work = work.rename(columns=rename)

    work["race_id"] = normalize_race_id(work["race_id"])
    work["horse_no"] = to_num(work["horse_no"]).astype("Int64")
    work["trainer_code"] = to_num(work["trainer_code"]).astype("Int64").astype("string")
    work["finish_num"] = to_num(work["finish_num"])
    work["pop_rank"] = to_num(work["pop_rank"])
    work["date_key"] = (
        work["date"]
        .astype("string")
        .str.replace(".", "", regex=False)
        .str.replace("/", "", regex=False)
        .str.replace("-", "", regex=False)
    )
    # "2026.2.15" becomes "2026215"; prefer race_id year when needed, but order
    # still works because the original file is temporal and race_id is sortable.
    work["date_key"] = pd.to_numeric(work["date_key"], errors="coerce")
    work["race_sort"] = pd.to_numeric(work["race_id"], errors="coerce")

    if "abnormal_code" in work.columns:
        work = work[to_num(work["abnormal_code"]).fillna(0).eq(0)].copy()
    work = work[work["finish_num"].notna() & work["horse_no"].notna()].copy()

    work["is_win"] = work["finish_num"].eq(1).astype(float)
    work["is_top3"] = work["finish_num"].le(3).astype(float)
    if "target_score" in work.columns:
        work["score_for_history"] = to_num(work["target_score"]).fillna(0.0)
    else:
        work["score_for_history"] = np.maximum(0.0, 1.0 - (work["finish_num"] - 1.0) / 18.0)
    work["is_pop_outperform"] = (work["finish_num"] <= work["pop_rank"]).fillna(False).astype(float)
    work["venue_region"] = work["venue"].map(VENUE_REGION).fillna("unknown")
    work["local_venue_flag"] = work["venue"].isin(LOCAL_VENUES).astype(float)

    sort_cols = ["race_sort", "horse_no"]
    ordered = work.sort_values(sort_cols, kind="mergesort")
    grouped_trainer = ordered.groupby("trainer_code", sort=False, dropna=False)
    trainer_starts = grouped_trainer.cumcount().astype(float)
    prev_east = grouped_trainer["venue_region"].transform(lambda s: (s.eq("east")).cumsum())
    prev_west = grouped_trainer["venue_region"].transform(lambda s: (s.eq("west")).cumsum())
    prev_hokkaido = grouped_trainer["venue_region"].transform(lambda s: (s.eq("hokkaido")).cumsum())
    prev_east = prev_east - ordered["venue_region"].eq("east").astype(int)
    prev_west = prev_west - ordered["venue_region"].eq("west").astype(int)
    prev_hokkaido = prev_hokkaido - ordered["venue_region"].eq("hokkaido").astype(int)
    denom = trainer_starts.replace(0.0, np.nan)
    east_share = (prev_east / denom).fillna(0.0)
    west_share = (prev_west / denom).fillna(0.0)
    hokkaido_share = (prev_hokkaido / denom).fillna(0.0)

    base = np.full(len(ordered), "unknown", dtype=object)
    base[(trainer_starts >= 20) & (east_share >= 0.65)] = "east"
    base[(trainer_starts >= 20) & (west_share >= 0.65)] = "west"
    base[(trainer_starts >= 20) & (hokkaido_share >= 0.40)] = "hokkaido"
    work.loc[ordered.index, "trainer_total_starts_past"] = trainer_starts
    work.loc[ordered.index, "trainer_east_share_past"] = east_share
    work.loc[ordered.index, "trainer_west_share_past"] = west_share
    work.loc[ordered.index, "trainer_hokkaido_share_past"] = hokkaido_share
    work.loc[ordered.index, "trainer_base_region_past"] = base

    work["trainer_expedition_flag"] = (
        ((work["trainer_base_region_past"].eq("east")) & work["venue_region"].isin(["west", "hokkaido"]))
        | ((work["trainer_base_region_past"].eq("west")) & work["venue_region"].isin(["east", "hokkaido"]))
    ).astype(float)
    work["trainer_home_region_flag"] = (
        work["trainer_base_region_past"].isin(["east", "west", "hokkaido"])
        & work["trainer_base_region_past"].eq(work["venue_region"])
    ).astype(float)

    work = add_shifted_group_rates(
        work,
        ["trainer_code", "venue"],
        "trainer_venue",
        sort_cols,
    )
    work = add_shifted_group_rates(
        work,
        ["trainer_code", "local_venue_flag"],
        "trainer_local",
        sort_cols,
    )
    work = add_shifted_group_rates(
        work,
        ["trainer_code", "trainer_expedition_flag"],
        "trainer_expedition",
        sort_cols,
    )
    work = add_shifted_group_rates(
        work,
        ["trainer_code", "venue", "surface"],
        "trainer_venue_surface",
        sort_cols,
    )

    keep_cols = [
        "race_id",
        "horse_no",
        "trainer_code",
        "venue",
        "surface",
        "distance",
        "class_name",
        "going",
        "venue_region",
        "local_venue_flag",
        "trainer_base_region_past",
        "trainer_total_starts_past",
        "trainer_east_share_past",
        "trainer_west_share_past",
        "trainer_hokkaido_share_past",
        "trainer_expedition_flag",
        "trainer_home_region_flag",
        "trainer_venue_starts",
        "trainer_venue_win_rate",
        "trainer_venue_top3_rate",
        "trainer_venue_avg_score",
        "trainer_venue_pop_outperform_rate",
        "trainer_local_starts",
        "trainer_local_win_rate",
        "trainer_local_top3_rate",
        "trainer_local_avg_score",
        "trainer_local_pop_outperform_rate",
        "trainer_expedition_starts",
        "trainer_expedition_win_rate",
        "trainer_expedition_top3_rate",
        "trainer_expedition_avg_score",
        "trainer_expedition_pop_outperform_rate",
        "trainer_venue_surface_starts",
        "trainer_venue_surface_win_rate",
        "trainer_venue_surface_top3_rate",
        "trainer_venue_surface_avg_score",
    ]
    return work[keep_cols].copy()


def summarize_flat(df: pd.DataFrame, mask: pd.Series, label: str) -> dict[str, object]:
    sub = df[mask.fillna(False)].copy()
    n = len(sub)
    if n == 0:
        return {
            "segment": label,
            "tickets": 0,
            "races": 0,
            "win_hit_rate": np.nan,
            "place_hit_rate": np.nan,
            "win_roi": np.nan,
            "place_roi": np.nan,
            "avg_odds": np.nan,
            "avg_pop": np.nan,
        }
    stake = n * 100.0
    return {
        "segment": label,
        "tickets": n,
        "races": int(sub["race_id"].nunique()),
        "win_hit_rate": float(sub["is_win"].mean()),
        "place_hit_rate": float(sub["is_place"].mean()),
        "win_roi": float(sub["win_return"].sum() / stake * 100.0),
        "place_roi": float(sub["place_return"].sum() / stake * 100.0),
        "avg_odds": float(sub["odds_num"].mean()),
        "avg_pop": float(sub["pop_rank_num"].mean()),
    }


def summarize_yearly(df: pd.DataFrame, mask: pd.Series, label: str) -> pd.DataFrame:
    sub = df[mask.fillna(False)].copy()
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for year, g in sub.groupby("year"):
        stake = len(g) * 100.0
        rows.append(
            {
                "segment": label,
                "year": int(year),
                "tickets": len(g),
                "races": int(g["race_id"].nunique()),
                "win_hit_rate": float(g["is_win"].mean()),
                "place_hit_rate": float(g["is_place"].mean()),
                "win_roi": float(g["win_return"].sum() / stake * 100.0),
                "place_roi": float(g["place_return"].sum() / stake * 100.0),
            }
        )
    return pd.DataFrame(rows)


def evaluate_runner_segments(features: pd.DataFrame, invest_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    invest = read_csv(invest_path)
    invest["race_id"] = normalize_race_id(invest["race_id"])
    invest["horse_no"] = to_num(invest["horse_no"]).astype("Int64")
    invest["year"] = invest["race_id"].str.slice(0, 4).astype(int)
    for col in ["ai_rank_num", "pop_rank_num", "odds_num", "win_return", "place_return"]:
        invest[col] = to_num(invest[col]).fillna(0.0)
    invest["is_win"] = invest["is_win"].astype(str).str.lower().isin(["true", "1", "1.0"]).astype(float)
    invest["is_place"] = invest["is_place"].astype(str).str.lower().isin(["true", "1", "1.0"]).astype(float)

    merged = invest.merge(features, on=["race_id", "horse_no"], how="left", suffixes=("", "_trainer"))

    tv_hi = (merged["trainer_venue_starts"] >= 10) & (merged["trainer_venue_top3_rate"] >= 0.35)
    tv_very_hi = (merged["trainer_venue_starts"] >= 20) & (merged["trainer_venue_top3_rate"] >= 0.40)
    tv_lo = (merged["trainer_venue_starts"] >= 10) & (merged["trainer_venue_top3_rate"] <= 0.18)
    tv_surface_hi = (
        (merged["trainer_venue_surface_starts"] >= 8)
        & (merged["trainer_venue_surface_top3_rate"] >= 0.35)
    )
    local_hi = (
        merged["venue"].isin(LOCAL_VENUES)
        & (merged["trainer_local_starts"] >= 20)
        & (merged["trainer_local_top3_rate"] >= 0.33)
    )
    expedition_hi = (
        (merged["trainer_expedition_flag"] == 1)
        & (merged["trainer_expedition_starts"] >= 10)
        & (merged["trainer_expedition_top3_rate"] >= 0.30)
    )
    expedition_lo = (
        (merged["trainer_expedition_flag"] == 1)
        & (merged["trainer_expedition_starts"] >= 10)
        & (merged["trainer_expedition_top3_rate"] <= 0.18)
    )

    segments = {
        "all_runners": pd.Series(True, index=merged.index),
        "ai1": merged["ai_rank_num"].eq(1),
        "ai_top3": merged["ai_rank_num"].between(1, 3),
        "ai1_trainer_venue_hi": merged["ai_rank_num"].eq(1) & tv_hi,
        "ai1_trainer_venue_very_hi": merged["ai_rank_num"].eq(1) & tv_very_hi,
        "ai1_trainer_venue_low": merged["ai_rank_num"].eq(1) & tv_lo,
        "ai_top3_trainer_venue_hi": merged["ai_rank_num"].between(1, 3) & tv_hi,
        "ai_top3_trainer_venue_low": merged["ai_rank_num"].between(1, 3) & tv_lo,
        "ai_top3_trainer_venue_surface_hi": merged["ai_rank_num"].between(1, 3) & tv_surface_hi,
        "ai_top3_local_strong_stable": merged["ai_rank_num"].between(1, 3) & local_hi,
        "ai_top3_expedition_strong": merged["ai_rank_num"].between(1, 3) & expedition_hi,
        "ai_top3_expedition_weak": merged["ai_rank_num"].between(1, 3) & expedition_lo,
        "overlay_ai_top3_trainer_venue_hi_odds4p": merged["ai_rank_num"].between(1, 3)
        & tv_hi
        & (merged["odds_num"] >= 4),
        "overlay_ai_top3_local_strong_odds4p": merged["ai_rank_num"].between(1, 3)
        & local_hi
        & (merged["odds_num"] >= 4),
        "watch_cut_ai_top3_trainer_venue_low": merged["ai_rank_num"].between(1, 3) & tv_lo,
    }

    rows = [summarize_flat(merged, mask, label) for label, mask in segments.items()]
    yearly = pd.concat(
        [summarize_yearly(merged, mask, label) for label, mask in segments.items()],
        ignore_index=True,
    )
    return merged, pd.DataFrame(rows), yearly


def evaluate_strongest_overlay(features: pd.DataFrame, strongest_path: Path) -> pd.DataFrame:
    if not strongest_path.exists():
        return pd.DataFrame()
    tickets = read_csv(strongest_path)
    if tickets.empty:
        return pd.DataFrame()

    tickets["race_id"] = normalize_race_id(tickets["race_id"])
    feat = features.copy()
    feat["horse_no"] = to_num(feat["horse_no"]).astype("Int64")
    anchor = feat.add_prefix("anchor_tv_").rename(
        columns={"anchor_tv_race_id": "race_id", "anchor_tv_horse_no": "anchor_no"}
    )
    partner = feat.add_prefix("partner_tv_").rename(
        columns={"partner_tv_race_id": "race_id", "partner_tv_horse_no": "partner_no"}
    )
    tickets["anchor_no"] = to_num(tickets["anchor_no"]).astype("Int64")
    tickets["partner_no"] = to_num(tickets["partner_no"]).astype("Int64")
    merged = tickets.merge(anchor, on=["race_id", "anchor_no"], how="left")
    merged = merged.merge(partner, on=["race_id", "partner_no"], how="left")

    for col in ["return_yen", "stake_yen"]:
        merged[col] = to_num(merged[col]).fillna(0.0)
    merged["year"] = merged["race_id"].str.slice(0, 4).astype(int)

    def mask_hi(side: str) -> pd.Series:
        return (
            (merged[f"{side}_tv_trainer_venue_starts"] >= 10)
            & (merged[f"{side}_tv_trainer_venue_top3_rate"] >= 0.35)
        )

    def mask_low(side: str) -> pd.Series:
        return (
            (merged[f"{side}_tv_trainer_venue_starts"] >= 10)
            & (merged[f"{side}_tv_trainer_venue_top3_rate"] <= 0.18)
        )

    def local_hi(side: str) -> pd.Series:
        return (
            merged[f"{side}_tv_venue"].isin(LOCAL_VENUES)
            & (merged[f"{side}_tv_trainer_local_starts"] >= 20)
            & (merged[f"{side}_tv_trainer_local_top3_rate"] >= 0.33)
        )

    def exp_lo(side: str) -> pd.Series:
        return (
            (merged[f"{side}_tv_trainer_expedition_flag"] == 1)
            & (merged[f"{side}_tv_trainer_expedition_starts"] >= 10)
            & (merged[f"{side}_tv_trainer_expedition_top3_rate"] <= 0.18)
        )

    segments = {
        "strongest_all": pd.Series(True, index=merged.index),
        "strongest_either_trainer_venue_hi": mask_hi("anchor") | mask_hi("partner"),
        "strongest_both_trainer_venue_hi": mask_hi("anchor") & mask_hi("partner"),
        "strongest_neither_trainer_venue_hi": ~(mask_hi("anchor") | mask_hi("partner")),
        "strongest_either_trainer_venue_low": mask_low("anchor") | mask_low("partner"),
        "strongest_no_trainer_venue_low": ~(mask_low("anchor") | mask_low("partner")),
        "strongest_either_local_strong_stable": local_hi("anchor") | local_hi("partner"),
        "strongest_either_weak_expedition": exp_lo("anchor") | exp_lo("partner"),
        "strongest_no_weak_expedition": ~(exp_lo("anchor") | exp_lo("partner")),
    }
    rows = []
    for label, mask in segments.items():
        sub = merged[mask.fillna(False)].copy()
        if sub.empty:
            rows.append({"segment": label, "tickets": 0, "races": 0, "roi": np.nan, "profit": 0.0, "hit_rate": np.nan})
            continue
        stake = sub["stake_yen"].sum()
        rows.append(
            {
                "segment": label,
                "tickets": len(sub),
                "races": int(sub["race_id"].nunique()),
                "hit_rate": float(to_num(sub["hit"]).fillna(0).mean()) if "hit" in sub.columns else np.nan,
                "roi": float(sub["return_yen"].sum() / stake * 100.0) if stake else np.nan,
                "profit": float(sub["return_yen"].sum() - stake),
                "stake": float(stake),
                "return": float(sub["return_yen"].sum()),
            }
        )
    return pd.DataFrame(rows)


def build_leaderboards(features: pd.DataFrame, invest_merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    runner = features.copy()
    # Leaderboard based on realized historical rows, not as-of rates, for human inspection only.
    # Rebuild from investment-merged to include payouts for recent eval years.
    m = invest_merged.copy()
    m["trainer_venue_key"] = m["trainer_code"].astype("string") + "_" + m["venue"].astype("string")
    grp = m.groupby(["trainer_code", "venue"], dropna=False)
    leader = grp.agg(
        starts=("race_id", "size"),
        races=("race_id", "nunique"),
        win_rate=("is_win", "mean"),
        top3_rate=("is_place", "mean"),
        win_return=("win_return", "sum"),
        place_return=("place_return", "sum"),
        avg_pop=("pop_rank_num", "mean"),
        avg_odds=("odds_num", "mean"),
    ).reset_index()
    leader["win_roi"] = leader["win_return"] / (leader["starts"] * 100.0) * 100.0
    leader["place_roi"] = leader["place_return"] / (leader["starts"] * 100.0) * 100.0
    leader = leader[leader["starts"] >= 10].sort_values(["place_roi", "top3_rate"], ascending=False)

    local = m[m["venue"].isin(LOCAL_VENUES)].groupby(["trainer_code"], dropna=False).agg(
        starts=("race_id", "size"),
        venues=("venue", "nunique"),
        top3_rate=("is_place", "mean"),
        place_return=("place_return", "sum"),
        win_return=("win_return", "sum"),
        avg_pop=("pop_rank_num", "mean"),
    ).reset_index()
    local["place_roi"] = local["place_return"] / (local["starts"] * 100.0) * 100.0
    local["win_roi"] = local["win_return"] / (local["starts"] * 100.0) * 100.0
    local = local[local["starts"] >= 20].sort_values(["place_roi", "top3_rate"], ascending=False)
    return leader, local


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--investment", type=Path, default=DEFAULT_INVEST)
    parser.add_argument("--strongest", type=Path, default=DEFAULT_STRONGEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    features = load_and_build_runner_features(args.train)
    features.to_csv(args.out_dir / "runner_trainer_venue_features.csv", index=False, encoding="utf-8-sig")

    merged, segment_summary, yearly = evaluate_runner_segments(features, args.investment)
    merged.to_csv(args.out_dir / "investment_with_trainer_venue_features.csv", index=False, encoding="utf-8-sig")
    segment_summary.to_csv(args.out_dir / "runner_segment_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(args.out_dir / "runner_segment_yearly.csv", index=False, encoding="utf-8-sig")

    strongest = evaluate_strongest_overlay(features, args.strongest)
    strongest.to_csv(args.out_dir / "strongest_overlay_summary.csv", index=False, encoding="utf-8-sig")

    leader, local = build_leaderboards(features, merged)
    leader.head(200).to_csv(args.out_dir / "trainer_venue_recent_leaders.csv", index=False, encoding="utf-8-sig")
    local.head(200).to_csv(args.out_dir / "trainer_local_recent_leaders.csv", index=False, encoding="utf-8-sig")

    summary = {
        "runner_features": str(args.out_dir / "runner_trainer_venue_features.csv"),
        "runner_segment_summary": str(args.out_dir / "runner_segment_summary.csv"),
        "runner_segment_yearly": str(args.out_dir / "runner_segment_yearly.csv"),
        "strongest_overlay_summary": str(args.out_dir / "strongest_overlay_summary.csv"),
        "trainer_venue_recent_leaders": str(args.out_dir / "trainer_venue_recent_leaders.csv"),
        "trainer_local_recent_leaders": str(args.out_dir / "trainer_local_recent_leaders.csv"),
        "rows": {
            "features": int(len(features)),
            "investment_merged": int(len(merged)),
            "strongest_segments": int(len(strongest)),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
