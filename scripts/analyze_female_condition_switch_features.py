from __future__ import annotations

import argparse
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
COL_HORSE_NO = "\u99ac\u756a"
COL_POPULARITY = "\u4eba\u6c17"
COL_WIN_ODDS = "\u5358\u52dd\u30aa\u30c3\u30ba"
COL_SURFACE = "\u829d\u30fb\u30c0"
COL_DISTANCE = "\u8ddd\u96e2"
COL_TRACK_CONDITION = "\u99ac\u5834\u72b6\u614b"
COL_RACE_ID = "\u30ec\u30fc\u30b9ID(\u65b0/\u99ac\u756a\u7121)"
COL_HORSE_ID = "\u8840\u7d71\u767b\u9332\u756a\u53f7"
COL_FINISH = "\u78ba\u5b9a\u7740\u9806"
COL_ABNORMAL = "\u7570\u5e38\u30b3\u30fc\u30c9"
COL_WIN_PAY = "\u5358\u52dd\u914d\u5f53"
COL_PLACE_PAY = "\u8907\u52dd\u914d\u5f53"


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


def clean_race_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(16)


def parse_race_date(race_id: pd.Series) -> pd.Series:
    return pd.to_datetime(race_id.astype(str).str.slice(0, 8), format="%Y%m%d", errors="coerce")


def read_features(paths: list[Path]) -> pd.DataFrame:
    usecols = [
        COL_DATE,
        COL_VENUE,
        COL_RACE_NO,
        COL_RACE_NAME,
        COL_CLASS_NAME,
        COL_HORSE_NAME,
        COL_SEX,
        COL_AGE,
        COL_HORSE_NO,
        COL_POPULARITY,
        COL_WIN_ODDS,
        COL_SURFACE,
        COL_DISTANCE,
        COL_TRACK_CONDITION,
        COL_RACE_ID,
        COL_HORSE_ID,
        COL_FINISH,
        COL_ABNORMAL,
        COL_WIN_PAY,
        COL_PLACE_PAY,
    ]
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        frames.append(
            pd.read_csv(
                path,
                usecols=usecols,
                encoding="utf-8-sig",
                dtype={COL_RACE_ID: "string", COL_HORSE_ID: "string"},
                low_memory=False,
            )
        )
    if not frames:
        raise FileNotFoundError("No feature files were found.")

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(
        columns={
            COL_DATE: "date_raw",
            COL_VENUE: "venue",
            COL_RACE_NO: "race_no",
            COL_RACE_NAME: "race_name",
            COL_CLASS_NAME: "class_name",
            COL_HORSE_NAME: "horse_name",
            COL_SEX: "sex",
            COL_AGE: "age",
            COL_HORSE_NO: "horse_no",
            COL_POPULARITY: "popularity",
            COL_WIN_ODDS: "win_odds",
            COL_SURFACE: "surface",
            COL_DISTANCE: "distance",
            COL_TRACK_CONDITION: "track_condition",
            COL_RACE_ID: "race_id",
            COL_HORSE_ID: "horse_id",
            COL_FINISH: "finish",
            COL_ABNORMAL: "abnormal_code",
            COL_WIN_PAY: "win_pay",
            COL_PLACE_PAY: "place_pay",
        }
    )
    df["race_id"] = clean_race_id(df["race_id"])
    df["race_date"] = parse_race_date(df["race_id"])
    for col in [
        "age",
        "horse_no",
        "popularity",
        "win_odds",
        "distance",
        "finish",
        "abnormal_code",
        "win_pay",
        "place_pay",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["race_id", "horse_id", "race_date", "horse_no"])
    df = df.drop_duplicates(["race_id", "horse_no"], keep="last")
    df = df[df["finish"].between(1, 30)]
    return df


def add_race_sex_context(df: pd.DataFrame) -> pd.DataFrame:
    race = (
        df.groupby("race_id", sort=False)
        .agg(
            runners=("horse_no", "count"),
            female_count=("sex", lambda s: (s == "\u725d").sum()),
            male_count=("sex", lambda s: (s == "\u7261").sum()),
            gelding_count=("sex", lambda s: (s == "\u30bb").sum()),
            race_name=("race_name", "first"),
        )
        .reset_index()
    )
    race["race_name_str"] = race["race_name"].fillna("").astype(str)
    race["race_name_has_female"] = race["race_name_str"].str.contains("\u725d", regex=False)
    race["female_only_race"] = race["female_count"].eq(race["runners"]) | race[
        "race_name_has_female"
    ]
    race["race_sex_condition"] = "mixed_or_open"
    race.loc[race["female_only_race"], "race_sex_condition"] = "female_only"
    return df.merge(
        race[["race_id", "female_count", "male_count", "gelding_count", "female_only_race", "race_sex_condition"]],
        on="race_id",
        how="left",
    )


def add_previous_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["horse_id", "race_date", "race_id"]).copy()
    prev_cols = [
        "race_id",
        "race_date",
        "race_name",
        "class_name",
        "venue",
        "surface",
        "distance",
        "track_condition",
        "race_sex_condition",
        "female_only_race",
        "finish",
        "popularity",
        "win_odds",
    ]
    for col in prev_cols:
        df[f"prev_{col}"] = df.groupby("horse_id", sort=False)[col].shift(1)

    df["days_since_prev"] = (df["race_date"] - df["prev_race_date"]).dt.days
    df["prev_top3"] = pd.to_numeric(df["prev_finish"], errors="coerce").le(3)
    df["prev_win"] = pd.to_numeric(df["prev_finish"], errors="coerce").eq(1)
    df["prev_outside_top3"] = pd.to_numeric(df["prev_finish"], errors="coerce").ge(4)
    df["prev_bad_finish"] = pd.to_numeric(df["prev_finish"], errors="coerce").ge(6)
    df["current_win"] = df["finish"].eq(1)
    df["current_top3"] = df["finish"].le(3)
    df["popularity_outperform"] = df["finish"].lt(df["popularity"])
    df["win_return"] = df["win_pay"].fillna(0)
    df["place_return"] = df["place_pay"].fillna(0)

    df["female_condition_transition"] = "no_prev"
    has_prev = df["prev_race_sex_condition"].notna()
    df.loc[
        has_prev & df["prev_female_only_race"].astype("boolean").fillna(False) & df["female_only_race"],
        "female_condition_transition",
    ] = "female_only_to_female_only"
    df.loc[
        has_prev & df["prev_female_only_race"].astype("boolean").fillna(False) & ~df["female_only_race"],
        "female_condition_transition",
    ] = "female_only_to_mixed"
    df.loc[
        has_prev & ~df["prev_female_only_race"].astype("boolean").fillna(False) & df["female_only_race"],
        "female_condition_transition",
    ] = "mixed_to_female_only"
    df.loc[
        has_prev & ~df["prev_female_only_race"].astype("boolean").fillna(False) & ~df["female_only_race"],
        "female_condition_transition",
    ] = "mixed_to_mixed"

    df["prev_result_bucket"] = "no_prev"
    df.loc[df["prev_win"], "prev_result_bucket"] = "prev_win"
    df.loc[df["prev_top3"] & ~df["prev_win"], "prev_result_bucket"] = "prev_2nd_3rd"
    df.loc[df["prev_outside_top3"] & ~df["prev_bad_finish"], "prev_result_bucket"] = "prev_4th_5th"
    df.loc[df["prev_bad_finish"], "prev_result_bucket"] = "prev_6th_or_worse"
    df["popularity_bucket"] = pd.cut(
        df["popularity"],
        bins=[0, 3, 6, 10, 99],
        labels=["1-3人気", "4-6人気", "7-10人気", "11人気以下"],
    ).astype("string")
    return df


def summarize(df: pd.DataFrame, group_cols: list[str], min_rows: int = 20) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        starts = len(g)
        if starts < min_rows:
            continue
        stake = starts * 100.0
        win_ret = float(g["win_return"].sum())
        place_ret = float(g["place_return"].sum())
        mean_market_prob = float((1 / g["win_odds"].where(g["win_odds"] > 0)).mean())
        row = {
            col: ("missing" if pd.isna(val) else str(val))
            for col, val in zip(group_cols, key)
        }
        row.update(
            {
                "starts": starts,
                "races": int(g["race_id"].nunique()),
                "win_rate_pct": round(float(g["current_win"].mean() * 100), 2),
                "top3_rate_pct": round(float(g["current_top3"].mean() * 100), 2),
                "avg_popularity": round(float(g["popularity"].mean()), 2),
                "avg_win_odds": round(float(g["win_odds"].mean()), 2),
                "win_roi_pct": round(win_ret / stake * 100, 1),
                "place_roi_pct": round(place_ret / stake * 100, 1),
                "popularity_outperform_pct": round(
                    float(g["popularity_outperform"].mean() * 100), 2
                ),
                "actual_vs_market_win_ratio": round(
                    (float(g["current_win"].mean()) / mean_market_prob)
                    if mean_market_prob
                    else 0.0,
                    2,
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_notes(out_dir: Path, female: pd.DataFrame, summaries: dict[str, pd.DataFrame]) -> None:
    lines = ["# Female Race-Condition Switch Feature Check", ""]
    lines.append(f"- female starts with previous race: {int(female['prev_race_id'].notna().sum())}")
    lines.append(f"- female total starts: {len(female)}")
    lines.append("")
    transition = summaries["transition"]
    lines.append("## Transition Summary")
    for _, r in transition.sort_values("starts", ascending=False).iterrows():
        lines.append(
            f"- {r['female_condition_transition']}: starts={r['starts']}, "
            f"win={r['win_rate_pct']}%, top3={r['top3_rate_pct']}%, "
            f"winROI={r['win_roi_pct']}%, placeROI={r['place_roi_pct']}%, "
            f"actual/market={r['actual_vs_market_win_ratio']}"
        )
    lines.append("")
    lines.append("## Feature Reading")
    lines.append(
        "- If `female_only_to_mixed` is weak after a previous good run, prior female-only form may need a discount in mixed/open races."
    )
    lines.append(
        "- If `mixed_to_female_only` rebounds after a previous poor run, mixed-race defeats should be discounted less when returning to female-only races."
    )
    lines.append(
        "- Treat this as a context feature first, not an immediate hard filter. The signal should be blended with class, odds, and distance/surface."
    )
    (out_dir / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/analysis/female_condition_switch_features")
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--min-rows", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_paths = [Path(p) for p in args.feature] if args.feature else DEFAULT_FEATURES

    df = read_features(feature_paths)
    df = add_race_sex_context(df)
    df = add_previous_context(df)
    female = df[df["sex"].eq("\u725d")].copy()
    female = female[female["prev_race_id"].notna()].copy()

    summaries = {
        "transition": summarize(female, ["female_condition_transition"], args.min_rows),
        "transition_prev_result": summarize(
            female, ["female_condition_transition", "prev_result_bucket"], args.min_rows
        ),
        "transition_popularity": summarize(
            female, ["female_condition_transition", "popularity_bucket"], args.min_rows
        ),
        "transition_surface": summarize(
            female, ["female_condition_transition", "surface"], args.min_rows
        ),
        "transition_class": summarize(
            female, ["female_condition_transition", "class_name"], args.min_rows
        ),
    }
    for name, summary in summaries.items():
        summary.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    keep = [
        "race_date",
        "race_id",
        "venue",
        "race_no",
        "race_name",
        "class_name",
        "horse_name",
        "age",
        "popularity",
        "win_odds",
        "finish",
        "win_pay",
        "place_pay",
        "surface",
        "distance",
        "female_condition_transition",
        "prev_race_date",
        "prev_race_name",
        "prev_class_name",
        "prev_finish",
        "prev_popularity",
        "prev_win_odds",
        "prev_race_sex_condition",
        "prev_result_bucket",
    ]
    female[keep].to_csv(out_dir / "female_starts_with_transition.csv", index=False, encoding="utf-8-sig")
    write_notes(out_dir, female, summaries)
    print(
        {
            "rows": int(len(female)),
            "output_dir": str(out_dir),
            "transition_rows": int(len(summaries["transition"])),
        }
    )


if __name__ == "__main__":
    main()
