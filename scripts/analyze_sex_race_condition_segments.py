from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


COL_DATE = "\u65e5\u4ed8"
COL_VENUE = "\u5834\u6240"
COL_RACE_NO = "\uff32"
COL_RACE_NAME = "\u30ec\u30fc\u30b9\u540d"
COL_CLASS_NAME = "\u30af\u30e9\u30b9\u540d"
COL_HORSE_NAME = "\u99ac\u540d"
COL_SEX = "\u6027\u5225"
COL_AGE = "\u5e74\u9f62"
COL_CARRIED_WEIGHT = "\u65a4\u91cf"
COL_HORSE_NO = "\u99ac\u756a"
COL_POPULARITY = "\u4eba\u6c17"
COL_WIN_ODDS = "\u5358\u52dd\u30aa\u30c3\u30ba"
COL_SURFACE = "\u829d\u30fb\u30c0"
COL_DISTANCE = "\u8ddd\u96e2"
COL_TRACK_CONDITION = "\u99ac\u5834\u72b6\u614b"
COL_RACE_ID = "\u30ec\u30fc\u30b9ID(\u65b0/\u99ac\u756a\u7121)"
COL_FINISH = "\u78ba\u5b9a\u7740\u9806"
COL_RACE_TYPE = "\u7af6\u8d70\u7a2e\u5225"


DEFAULT_FEATURES = [
    Path(
        "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/"
        "body_weight_backfilled/owner_breeder_enriched/"
        "train_features_with_same_day_bias_v3_retro_body_breeder.csv"
    ),
    Path(
        "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/"
        "body_weight_backfilled/owner_breeder_enriched/"
        "test_features_with_same_day_bias_v3_retro_body_breeder.csv"
    ),
]

DEFAULT_TICKETS = [
    Path(
        "outputs/analysis/mcs_pbo_runtime_overlay_v4_operational_gates/"
        "mcs_full_margin095_s0304_danger020_skip03119_selected_tickets.csv"
    ),
    Path(
        "outputs/analysis/mcs_pbo_runtime_overlay_v4_operational_gates/"
        "mcs_full_margin095_s0304_skip03119_selected_tickets.csv"
    ),
    Path(
        "outputs/analysis/mcs_pbo_runtime_overlay_v4_operational_gates/"
        "recommended_runtime_tickets.csv"
    ),
]


def clean_race_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(16)


def read_runner_features(paths: list[Path]) -> pd.DataFrame:
    usecols = [
        COL_DATE,
        COL_VENUE,
        COL_RACE_NO,
        COL_RACE_NAME,
        COL_CLASS_NAME,
        COL_HORSE_NAME,
        COL_SEX,
        COL_AGE,
        COL_CARRIED_WEIGHT,
        COL_HORSE_NO,
        COL_POPULARITY,
        COL_WIN_ODDS,
        COL_SURFACE,
        COL_DISTANCE,
        COL_TRACK_CONDITION,
        COL_RACE_ID,
        COL_FINISH,
        COL_RACE_TYPE,
    ]
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(
            path,
            usecols=usecols,
            encoding="utf-8-sig",
            dtype={COL_RACE_ID: "string"},
            low_memory=False,
        )
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No feature files were found.")

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(
        columns={
            COL_DATE: "date",
            COL_VENUE: "venue",
            COL_RACE_NO: "race_no",
            COL_RACE_NAME: "race_name",
            COL_CLASS_NAME: "class_name",
            COL_HORSE_NAME: "horse_name",
            COL_SEX: "sex",
            COL_AGE: "age",
            COL_CARRIED_WEIGHT: "carried_weight",
            COL_HORSE_NO: "horse_no",
            COL_POPULARITY: "popularity",
            COL_WIN_ODDS: "win_odds",
            COL_SURFACE: "surface",
            COL_DISTANCE: "distance",
            COL_TRACK_CONDITION: "track_condition",
            COL_RACE_ID: "race_id",
            COL_FINISH: "finish",
            COL_RACE_TYPE: "race_type_code",
        }
    )
    df["race_id"] = clean_race_id(df["race_id"])
    df["horse_no"] = pd.to_numeric(df["horse_no"], errors="coerce").astype("Int64")
    for col in ["age", "carried_weight", "popularity", "win_odds", "distance", "finish"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["race_id", "horse_no"]).drop_duplicates(
        ["race_id", "horse_no"], keep="last"
    )
    return df


def build_race_context(runners: pd.DataFrame) -> pd.DataFrame:
    def first_non_null(x: pd.Series):
        x = x.dropna()
        return x.iloc[0] if len(x) else pd.NA

    grouped = runners.groupby("race_id", sort=False)
    race = grouped.agg(
        date=("date", first_non_null),
        venue=("venue", first_non_null),
        race_no=("race_no", first_non_null),
        race_name=("race_name", first_non_null),
        class_name=("class_name", first_non_null),
        surface=("surface", first_non_null),
        distance=("distance", first_non_null),
        track_condition=("track_condition", first_non_null),
        race_type_code=("race_type_code", first_non_null),
        runners=("horse_no", "count"),
        female_count=("sex", lambda s: (s == "\u725d").sum()),
        male_count=("sex", lambda s: (s == "\u7261").sum()),
        gelding_count=("sex", lambda s: (s == "\u30bb").sum()),
        weight_min=("carried_weight", "min"),
        weight_max=("carried_weight", "max"),
        weight_std=("carried_weight", "std"),
        weight_nunique=("carried_weight", "nunique"),
    ).reset_index()
    race["female_share"] = race["female_count"] / race["runners"].where(race["runners"] > 0)
    race["weight_spread"] = race["weight_max"] - race["weight_min"]
    race["race_name_str"] = race["race_name"].fillna("").astype(str)
    race["name_female_limited"] = race["race_name_str"].str.contains("\u725d", regex=False)
    race["female_only_race"] = race["female_count"].eq(race["runners"]) | race[
        "name_female_limited"
    ]
    race["mixed_with_female"] = race["female_count"].gt(0) & race["male_count"].gt(0)
    race["gelding_present"] = race["gelding_count"].gt(0)
    race["race_sex_mix"] = "male_or_gelding_only"
    race.loc[race["mixed_with_female"], "race_sex_mix"] = "mixed_with_female"
    race.loc[race["female_only_race"], "race_sex_mix"] = "female_only"
    race.loc[race["gelding_present"], "race_sex_mix"] = (
        race.loc[race["gelding_present"], "race_sex_mix"] + "_with_gelding"
    )

    race["weight_spread_bucket"] = pd.cut(
        race["weight_spread"],
        bins=[-0.01, 1.0, 2.5, 4.0, 99.0],
        labels=["0-1kg", "1-2.5kg", "2.5-4kg", "4kg+"],
    ).astype("string")
    race["weight_type_proxy"] = "allowance_or_sex_age_proxy"
    race.loc[race["weight_spread"].le(1.0), "weight_type_proxy"] = "fixed_weight_proxy"
    race.loc[race["weight_spread"].ge(4.0), "weight_type_proxy"] = "handicap_proxy"
    race["distance_bucket"] = pd.cut(
        race["distance"],
        bins=[0, 1300, 1600, 2000, 2400, 10000],
        labels=["sprint", "mile", "middle", "long", "stamina"],
    ).astype("string")
    return race.drop(columns=["race_name_str"])


def read_tickets(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if "race_id" not in df.columns:
        raise ValueError(f"{path} has no race_id column.")
    df["race_id"] = clean_race_id(df["race_id"])
    for col in ["anchor_no", "partner_no"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    stake_col = "eval_stake_yen" if "eval_stake_yen" in df.columns else "stake_yen"
    return_col = "eval_return_yen" if "eval_return_yen" in df.columns else "return_yen"
    df["stake_for_eval"] = pd.to_numeric(df[stake_col], errors="coerce").fillna(0.0)
    df["return_for_eval"] = pd.to_numeric(df[return_col], errors="coerce").fillna(0.0)
    df["hit_for_eval"] = df["return_for_eval"].gt(0)
    return df


def enrich_tickets(tickets: pd.DataFrame, runners: pd.DataFrame, race: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "race_id",
        "horse_no",
        "horse_name",
        "sex",
        "age",
        "carried_weight",
        "popularity",
        "win_odds",
    ]
    lookup = runners[keep].copy()
    anchor = lookup.add_prefix("anchor_").rename(
        columns={"anchor_race_id": "race_id", "anchor_horse_no": "anchor_no"}
    )
    partner = lookup.add_prefix("partner_").rename(
        columns={"partner_race_id": "race_id", "partner_horse_no": "partner_no"}
    )
    df = tickets.merge(anchor, on=["race_id", "anchor_no"], how="left")
    df = df.merge(partner, on=["race_id", "partner_no"], how="left")
    df = df.merge(race, on="race_id", how="left")
    for col in [
        "date",
        "venue",
        "race_no",
        "race_name",
        "class_name",
        "surface",
        "distance",
        "track_condition",
        "race_type_code",
    ]:
        y_col = f"{col}_y"
        x_col = f"{col}_x"
        if y_col in df.columns:
            df[col] = df[y_col]
            if x_col in df.columns:
                df[col] = df[col].combine_first(df[x_col])

    df["anchor_sex"] = df["anchor_sex"].fillna("unknown")
    df["partner_sex"] = df["partner_sex"].fillna("unknown")
    df["sex_pair_ordered"] = df["anchor_sex"] + "-" + df["partner_sex"]
    df["sex_pair_unordered"] = df.apply(
        lambda r: "-".join(sorted([str(r["anchor_sex"]), str(r["partner_sex"])])), axis=1
    )
    df["female_in_ticket"] = df[["anchor_sex", "partner_sex"]].eq("\u725d").any(axis=1)
    df["gelding_in_ticket"] = df[["anchor_sex", "partner_sex"]].eq("\u30bb").any(axis=1)
    df["same_sex_pair"] = df["anchor_sex"].eq(df["partner_sex"])
    df["anchor_weight_minus_partner"] = df["anchor_carried_weight"] - df["partner_carried_weight"]
    df["anchor_weight_relation"] = "even"
    df.loc[df["anchor_weight_minus_partner"].le(-1.5), "anchor_weight_relation"] = "anchor_lighter"
    df.loc[df["anchor_weight_minus_partner"].ge(1.5), "anchor_weight_relation"] = "anchor_heavier"
    return df


def summarize(df: pd.DataFrame, policy: str, group_col: str, min_tickets: int) -> pd.DataFrame:
    rows = []
    for value, g in df.groupby(group_col, dropna=False, sort=False):
        tickets = len(g)
        if tickets < min_tickets:
            continue
        stake = float(g["stake_for_eval"].sum())
        ret = float(g["return_for_eval"].sum())
        max_ret = float(g["return_for_eval"].max()) if tickets else 0.0
        max_idx = g["return_for_eval"].idxmax() if tickets else None
        stake_ex_top = stake
        ret_ex_top = ret
        if max_idx is not None:
            stake_ex_top -= float(g.loc[max_idx, "stake_for_eval"])
            ret_ex_top -= float(g.loc[max_idx, "return_for_eval"])
        rows.append(
            {
                "policy": policy,
                "segment": group_col,
                "value": "missing" if pd.isna(value) else str(value),
                "tickets": tickets,
                "races": int(g["race_id"].nunique()),
                "stake_yen": round(stake, 1),
                "return_yen": round(ret, 1),
                "profit_yen": round(ret - stake, 1),
                "roi_pct": round((ret / stake * 100.0) if stake else 0.0, 1),
                "hit_rate_pct": round(float(g["hit_for_eval"].mean() * 100.0), 1),
                "max_return_share_pct": round((max_ret / ret * 100.0) if ret else 0.0, 1),
                "top1_removed_roi_pct": round(
                    (ret_ex_top / stake_ex_top * 100.0) if stake_ex_top else 0.0, 1
                ),
            }
        )
    return pd.DataFrame(rows)


def overall_row(df: pd.DataFrame, policy: str) -> dict:
    stake = float(df["stake_for_eval"].sum())
    ret = float(df["return_for_eval"].sum())
    return {
        "policy": policy,
        "tickets": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "stake_yen": round(stake, 1),
        "return_yen": round(ret, 1),
        "profit_yen": round(ret - stake, 1),
        "roi_pct": round((ret / stake * 100.0) if stake else 0.0, 1),
        "hit_rate_pct": round(float(df["hit_for_eval"].mean() * 100.0), 1),
    }


def write_findings(summary: pd.DataFrame, coverage: list[dict], out_path: Path) -> None:
    lines = ["# Sex / Race Condition Segment Check", ""]
    lines.append("## Coverage")
    for item in coverage:
        lines.append(
            f"- {item['policy']}: tickets={item['tickets']}, "
            f"anchor_join={item['anchor_join_rate_pct']}%, "
            f"partner_join={item['partner_join_rate_pct']}%, "
            f"race_join={item['race_join_rate_pct']}%"
        )
    lines.append("")
    lines.append("## Notable Segments")
    candidate = summary[(summary["tickets"] >= 10) & (summary["segment"] != "overall")].copy()
    if not candidate.empty:
        top = candidate.sort_values(["roi_pct", "tickets"], ascending=[False, False]).head(12)
        low = candidate.sort_values(["roi_pct", "tickets"], ascending=[True, False]).head(12)
        lines.append("### High ROI")
        for _, r in top.iterrows():
            lines.append(
                f"- {r['policy']} / {r['segment']}={r['value']}: "
                f"tickets={r['tickets']}, ROI={r['roi_pct']}%, hit={r['hit_rate_pct']}%, "
                f"top1_removed={r['top1_removed_roi_pct']}%"
            )
        lines.append("")
        lines.append("### Low ROI")
        for _, r in low.iterrows():
            lines.append(
                f"- {r['policy']} / {r['segment']}={r['value']}: "
                f"tickets={r['tickets']}, ROI={r['roi_pct']}%, hit={r['hit_rate_pct']}%, "
                f"top1_removed={r['top1_removed_roi_pct']}%"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/analysis/sex_race_condition_segments")
    parser.add_argument("--min-tickets", type=int, default=8)
    parser.add_argument("--ticket", action="append", default=[])
    parser.add_argument("--feature", action="append", default=[])
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_paths = [Path(p) for p in args.feature] if args.feature else DEFAULT_FEATURES
    ticket_paths = [Path(p) for p in args.ticket] if args.ticket else DEFAULT_TICKETS

    runners = read_runner_features(feature_paths)
    race = build_race_context(runners)

    group_cols = [
        "anchor_sex",
        "partner_sex",
        "sex_pair_ordered",
        "sex_pair_unordered",
        "female_in_ticket",
        "gelding_in_ticket",
        "same_sex_pair",
        "race_sex_mix",
        "female_only_race",
        "mixed_with_female",
        "gelding_present",
        "weight_type_proxy",
        "weight_spread_bucket",
        "anchor_weight_relation",
        "surface",
        "distance_bucket",
        "class_name",
        "track_condition",
    ]

    summaries = []
    coverage = []
    for ticket_path in ticket_paths:
        if not ticket_path.exists():
            continue
        policy = ticket_path.stem.replace("_selected_tickets", "").replace("_runtime_tickets", "")
        tickets = read_tickets(ticket_path)
        enriched = enrich_tickets(tickets, runners, race)
        enriched.to_csv(out_dir / f"{policy}_enriched_tickets.csv", index=False, encoding="utf-8-sig")
        overall = overall_row(enriched, policy)
        overall.update({"segment": "overall", "value": "all", "max_return_share_pct": "", "top1_removed_roi_pct": ""})
        summaries.append(pd.DataFrame([overall]))
        for col in group_cols:
            if col in enriched.columns:
                summaries.append(summarize(enriched, policy, col, args.min_tickets))
        coverage.append(
            {
                "policy": policy,
                "tickets": int(len(enriched)),
                "anchor_join_rate_pct": round(float(enriched["anchor_sex"].ne("unknown").mean() * 100), 1),
                "partner_join_rate_pct": round(float(enriched["partner_sex"].ne("unknown").mean() * 100), 1),
                "race_join_rate_pct": round(float(enriched["race_name"].notna().mean() * 100), 1),
                "source": str(ticket_path),
            }
        )

    if not summaries:
        raise FileNotFoundError("No ticket files were found.")
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(out_dir / "segment_summary.csv", index=False, encoding="utf-8-sig")
    (out_dir / "coverage.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_findings(summary, coverage, out_dir / "findings.md")
    print(json.dumps({"output_dir": str(out_dir), "coverage": coverage}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
