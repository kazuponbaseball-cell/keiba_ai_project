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
COL_INTERVAL = "\u9593\u9694"
COL_LAYOFF_START = "\u4f11\u307f\u660e\u3051\uff5e\u6226\u76ee"

MALE = "\u7261"
FEMALE = "\u725d"
GELDING = "\u30bb"


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
        COL_INTERVAL,
        COL_LAYOFF_START,
    ]
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
        available = [c for c in usecols if c in header.columns]
        frames.append(
            pd.read_csv(
                path,
                usecols=available,
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
            COL_INTERVAL: "interval_weeks_raw",
            COL_LAYOFF_START: "layoff_start_no",
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
        "interval_weeks_raw",
        "layoff_start_no",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["race_id", "horse_id", "race_date", "horse_no", "finish"])
    df = df.drop_duplicates(["race_id", "horse_no"], keep="last")
    df = df[df["finish"].between(1, 30)]
    return df


def add_gelding_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["horse_id", "race_date", "race_id"]).copy()
    out["prev_sex"] = out.groupby("horse_id", sort=False)["sex"].shift(1)
    out["prev_race_date"] = out.groupby("horse_id", sort=False)["race_date"].shift(1)
    out["prev_finish"] = out.groupby("horse_id", sort=False)["finish"].shift(1)
    out["prev_popularity"] = out.groupby("horse_id", sort=False)["popularity"].shift(1)
    out["prev_surface"] = out.groupby("horse_id", sort=False)["surface"].shift(1)
    out["prev_distance"] = out.groupby("horse_id", sort=False)["distance"].shift(1)
    out["days_since_prev"] = (out["race_date"] - out["prev_race_date"]).dt.days
    out["is_gelding"] = out["sex"].eq(GELDING)
    out["known_gelding_debut"] = out["is_gelding"] & out["prev_sex"].eq(MALE)
    out["known_gelding_debut_from_female"] = out["is_gelding"] & out["prev_sex"].eq(FEMALE)
    out["first_seen_as_gelding"] = out["is_gelding"] & out["prev_sex"].isna()
    out["established_gelding"] = out["is_gelding"] & out["prev_sex"].eq(GELDING)
    out["has_pre_gelding_male_record"] = (
        out.groupby("horse_id", sort=False)["sex"]
        .transform(lambda s: s.eq(MALE).cummax().shift(1).fillna(False))
        .astype(bool)
    )
    out["known_gelding_transition"] = out["is_gelding"] & out["has_pre_gelding_male_record"]
    out["gelding_start_no_since_transition"] = 0
    for _, idx in out.groupby("horse_id", sort=False).groups.items():
        part = out.loc[idx]
        start_no = []
        seen_transition = False
        count = 0
        previous_sex = None
        for _, row in part.iterrows():
            if row["sex"] == GELDING and previous_sex == MALE:
                seen_transition = True
                count = 1
            elif row["sex"] == GELDING and seen_transition:
                count += 1
            else:
                count = 0 if row["sex"] != GELDING else count
            start_no.append(count if row["sex"] == GELDING and seen_transition else 0)
            previous_sex = row["sex"]
        out.loc[idx, "gelding_start_no_since_transition"] = start_no

    out["gelding_phase"] = "non_gelding"
    out.loc[out["first_seen_as_gelding"], "gelding_phase"] = "first_seen_as_gelding_unknown_timing"
    out.loc[out["known_gelding_debut"], "gelding_phase"] = "known_gelding_debut"
    out.loc[out["gelding_start_no_since_transition"].eq(2), "gelding_phase"] = "known_gelding_second_start"
    out.loc[out["gelding_start_no_since_transition"].eq(3), "gelding_phase"] = "known_gelding_third_start"
    out.loc[out["gelding_start_no_since_transition"].ge(4), "gelding_phase"] = "known_gelding_4plus_start"
    out.loc[out["is_gelding"] & out["gelding_phase"].eq("non_gelding"), "gelding_phase"] = "established_gelding_unknown_transition"

    out["prev_result_bucket"] = "no_prev"
    out.loc[out["prev_finish"].eq(1), "prev_result_bucket"] = "prev_win"
    out.loc[out["prev_finish"].between(2, 3), "prev_result_bucket"] = "prev_2nd_3rd"
    out.loc[out["prev_finish"].between(4, 5), "prev_result_bucket"] = "prev_4th_5th"
    out.loc[out["prev_finish"].ge(6), "prev_result_bucket"] = "prev_6th_or_worse"
    out["layoff_bucket"] = pd.cut(
        out["days_since_prev"],
        bins=[-1, 35, 70, 140, 365, 99999],
        labels=["0-35d", "36-70d", "71-140d", "141-365d", "366d+"],
    ).astype("string")
    out["popularity_bucket"] = pd.cut(
        out["popularity"],
        bins=[0, 3, 6, 10, 99],
        labels=["1-3人気", "4-6人気", "7-10人気", "11人気以下"],
    ).astype("string")
    out["age_bucket"] = pd.cut(
        out["age"],
        bins=[0, 3, 4, 5, 6, 99],
        labels=["2-3歳", "4歳", "5歳", "6歳", "7歳以上"],
    ).astype("string")
    out["distance_bucket"] = pd.cut(
        out["distance"],
        bins=[0, 1300, 1600, 2000, 2400, 10000],
        labels=["sprint", "mile", "middle", "long", "stamina"],
    ).astype("string")
    out["distance_change_bucket"] = "same_or_unknown"
    diff = out["distance"] - out["prev_distance"]
    out.loc[diff.le(-200), "distance_change_bucket"] = "shorten"
    out.loc[diff.ge(200), "distance_change_bucket"] = "extend"
    out.loc[diff.abs().lt(200), "distance_change_bucket"] = "same"
    out["surface_change_bucket"] = "same_or_unknown"
    out.loc[out["prev_surface"].notna() & out["surface"].ne(out["prev_surface"]), "surface_change_bucket"] = "surface_switch"
    out.loc[out["prev_surface"].notna() & out["surface"].eq(out["prev_surface"]), "surface_change_bucket"] = "same_surface"
    out["current_win"] = out["finish"].eq(1)
    out["current_top3"] = out["finish"].le(3)
    out["win_return"] = out["win_pay"].fillna(0)
    out["place_return"] = out["place_pay"].fillna(0)
    return out


def summarize(df: pd.DataFrame, group_cols: list[str], min_rows: int) -> pd.DataFrame:
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
        market_prob = (1 / g["win_odds"].where(g["win_odds"] > 0)).replace([float("inf")], pd.NA)
        row = {col: ("missing" if pd.isna(val) else str(val)) for col, val in zip(group_cols, key)}
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
                "actual_vs_market_win_ratio": round(
                    (float(g["current_win"].mean()) / float(market_prob.mean()))
                    if market_prob.notna().any() and float(market_prob.mean()) > 0
                    else 0.0,
                    2,
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def scenario_metrics(df: pd.DataFrame) -> pd.DataFrame:
    scenarios = {
        "all_gelding": df[df["is_gelding"]],
        "known_gelding_debut": df[df["known_gelding_debut"]],
        "known_gelding_debut_popular_1_3": df[df["known_gelding_debut"] & df["popularity"].between(1, 3)],
        "known_gelding_debut_not_popular": df[df["known_gelding_debut"] & ~df["popularity"].between(1, 3)],
        "known_gelding_debut_0_70d": df[df["known_gelding_debut"] & df["layoff_bucket"].isin(["0-35d", "36-70d"])],
        "known_gelding_debut_71d_plus": df[df["known_gelding_debut"] & ~df["layoff_bucket"].isin(["0-35d", "36-70d"])],
        "known_gelding_second_third": df[df["gelding_start_no_since_transition"].isin([2, 3])],
        "established_gelding_transition_4plus": df[df["gelding_start_no_since_transition"].ge(4)],
        "non_gelding_male": df[df["sex"].eq(MALE)],
    }
    rows = []
    for name, g in scenarios.items():
        if g.empty:
            continue
        starts = len(g)
        stake = starts * 100.0
        win_ret = float(g["win_return"].sum())
        place_ret = float(g["place_return"].sum())
        market_prob = (1 / g["win_odds"].where(g["win_odds"] > 0)).replace([float("inf")], pd.NA)
        row = {
            "starts": starts,
            "races": int(g["race_id"].nunique()),
            "win_rate_pct": round(float(g["current_win"].mean() * 100), 2),
            "top3_rate_pct": round(float(g["current_top3"].mean() * 100), 2),
            "avg_popularity": round(float(g["popularity"].mean()), 2),
            "avg_win_odds": round(float(g["win_odds"].mean()), 2),
            "win_roi_pct": round(win_ret / stake * 100, 1),
            "place_roi_pct": round(place_ret / stake * 100, 1),
            "actual_vs_market_win_ratio": round(
                (float(g["current_win"].mean()) / float(market_prob.mean()))
                if market_prob.notna().any() and float(market_prob.mean()) > 0
                else 0.0,
                2,
            ),
            "scenario": name,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def write_findings(out_dir: Path, df: pd.DataFrame, summaries: dict[str, pd.DataFrame]) -> None:
    lines = ["# Gelding Transition Check", ""]
    lines.append(
        "Proxy definition: `known_gelding_debut` means the same horse was male in the previous observed start and is gelding in the current start."
    )
    lines.append("")
    scen = summaries["scenarios"]
    if not scen.empty:
        lines.append("## Main Scenarios")
        for _, r in scen.iterrows():
            lines.append(
                f"- {r['scenario']}: starts={int(r['starts'])}, win={r['win_rate_pct']}%, "
                f"top3={r['top3_rate_pct']}%, winROI={r['win_roi_pct']}%, placeROI={r['place_roi_pct']}%"
            )
    lines.append("")
    lines.append("## Caution")
    lines.append("- Actual surgery dates are not available in this dataset; sex-label transition is used as a practical proxy.")
    lines.append("- Very small segments should be treated as hypothesis candidates, not final rules.")
    (out_dir / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze gelding debut and post-gelding conditions.")
    parser.add_argument("--output-dir", default="outputs/analysis/gelding_transition_features")
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--min-rows", type=int, default=20)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [Path(p) for p in args.feature] if args.feature else DEFAULT_FEATURES
    df = add_gelding_context(read_features(paths))

    summaries = {
        "scenarios": scenario_metrics(df),
        "phase": summarize(df, ["gelding_phase"], args.min_rows),
        "phase_popularity": summarize(df, ["gelding_phase", "popularity_bucket"], args.min_rows),
        "phase_layoff": summarize(df, ["gelding_phase", "layoff_bucket"], args.min_rows),
        "phase_prev_result": summarize(df, ["gelding_phase", "prev_result_bucket"], args.min_rows),
        "phase_surface": summarize(df, ["gelding_phase", "surface"], args.min_rows),
        "phase_distance": summarize(df, ["gelding_phase", "distance_bucket"], args.min_rows),
        "phase_age": summarize(df, ["gelding_phase", "age_bucket"], args.min_rows),
        "phase_class": summarize(df, ["gelding_phase", "class_name"], args.min_rows),
        "phase_surface_change": summarize(df, ["gelding_phase", "surface_change_bucket"], args.min_rows),
        "phase_distance_change": summarize(df, ["gelding_phase", "distance_change_bucket"], args.min_rows),
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
        "sex",
        "prev_sex",
        "gelding_phase",
        "gelding_start_no_since_transition",
        "age",
        "popularity",
        "win_odds",
        "finish",
        "win_pay",
        "place_pay",
        "surface",
        "distance",
        "track_condition",
        "days_since_prev",
        "layoff_bucket",
        "prev_finish",
        "prev_result_bucket",
        "surface_change_bucket",
        "distance_change_bucket",
    ]
    df[df["is_gelding"] | df["known_gelding_transition"]][keep].to_csv(
        out_dir / "gelding_starts_with_context.csv", index=False, encoding="utf-8-sig"
    )
    write_findings(out_dir, df, summaries)
    meta = {
        "rows": int(len(df)),
        "gelding_rows": int(df["is_gelding"].sum()),
        "known_gelding_debut_rows": int(df["known_gelding_debut"].sum()),
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
