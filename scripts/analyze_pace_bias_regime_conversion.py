from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "data"
    / "datasets"
    / "cache"
    / "workout_lap_pedigree_interactions_confirmed_opponent_2023plus"
    / "train_features.csv"
)
DEFAULT_OUT = ROOT / "outputs" / "analysis" / "pace_bias_regime_conversion_v1"


def read_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)


def pick(existing: set[str], names: list[str]) -> str | None:
    for name in names:
        if name in existing:
            return name
    return None


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def norm_race_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(16)


def distance_bin(distance: pd.Series) -> pd.Series:
    d = num(distance)
    return pd.Series(
        np.select(
            [d <= 1400, d <= 1800, d <= 2200, d > 2200],
            ["sprint", "mile", "middle", "long"],
            default="unknown",
        ),
        index=distance.index,
    )


def going_bin(going: pd.Series) -> pd.Series:
    g = going.astype("string").fillna("")
    return pd.Series(
        np.select(
            [g.str.contains("良", regex=False), g.str.contains("稍", regex=False), g.str.contains("重|不", regex=True)],
            ["firm_good", "yielding", "soft_heavy"],
            default="unknown",
        ),
        index=going.index,
    )


def venue_group(venue: pd.Series) -> pd.Series:
    v = venue.astype("string")
    local = v.isin(["札幌", "函館", "福島", "小倉"])
    big = v.isin(["東京", "阪神", "京都", "中京", "中山", "新潟"])
    return pd.Series(np.select([local, big], ["local_small", "major"], default="unknown"), index=venue.index)


def style_from_corner(corner4: float, field_size: float) -> str:
    if pd.isna(corner4) or pd.isna(field_size):
        return "unknown"
    if corner4 <= 3:
        return "front"
    if corner4 <= max(5, field_size * 0.40):
        return "stalker"
    if corner4 >= field_size * 0.65:
        return "closer"
    return "mid"


def build_race_table(path: Path) -> pd.DataFrame:
    header = read_header(path)
    existing = set(header)
    race_col = pick(existing, ["レースID(新/馬番無)", "race_id"])
    finish_col = pick(existing, ["確定着順", "finish_num", "finish"])
    corner_col = pick(existing, ["4角.1", "4角", "corner4"])
    if not race_col or not finish_col or not corner_col:
        raise KeyError("race/finish/corner columns are required")

    base_cols = [
        race_col,
        finish_col,
        corner_col,
        pick(existing, ["日付", "date"]),
        pick(existing, ["日付S", "date_s"]),
        pick(existing, ["場所", "venue"]),
        pick(existing, ["Ｒ", "race_no"]),
        pick(existing, ["レース名", "race_name"]),
        pick(existing, ["芝・ダ", "surface"]),
        pick(existing, ["距離", "distance"]),
        pick(existing, ["馬場状態", "going"]),
        pick(existing, ["頭数", "field_size"]),
        pick(existing, ["出走頭数", "field_size2"]),
        pick(existing, ["人気", "popularity"]),
        pick(existing, ["単勝オッズ", "win_odds"]),
        pick(existing, ["異常コード", "abnormal_code"]),
    ]
    feature_cols = [
        "race_front_runner_count",
        "race_front_runner_ratio",
        "race_closer_count",
        "race_closer_ratio",
        "race_early_pressure_score",
        "race_pace_collapse_risk",
        "race_slow_pace_risk",
        "race_deep_closer_count",
        "front_pressure_rank_score",
        "pace_fit_score",
        "front_advantage_score",
        "closer_advantage_score",
        "draw_pace_fit_score",
        "PCI",
        "PCI3",
        "RPCI",
        "Ave-3F",
    ]
    usecols = [c for c in base_cols + feature_cols if c and c in existing]
    df = pd.read_csv(path, usecols=usecols, encoding="utf-8-sig", low_memory=False)
    rename = {
        race_col: "race_id",
        finish_col: "finish",
        corner_col: "corner4",
    }
    optional_map = {
        "日付": "date",
        "date": "date",
        "日付S": "date_s",
        "date_s": "date_s",
        "場所": "venue",
        "venue": "venue",
        "Ｒ": "race_no",
        "race_no": "race_no",
        "レース名": "race_name",
        "race_name": "race_name",
        "芝・ダ": "surface",
        "surface": "surface",
        "距離": "distance",
        "distance": "distance",
        "馬場状態": "going",
        "going": "going",
        "頭数": "field_size",
        "出走頭数": "field_size2",
        "人気": "popularity",
        "popularity": "popularity",
        "単勝オッズ": "win_odds",
        "win_odds": "win_odds",
        "異常コード": "abnormal_code",
        "abnormal_code": "abnormal_code",
    }
    rename.update({k: v for k, v in optional_map.items() if k in df.columns})
    df = df.rename(columns=rename)
    df["race_id"] = norm_race_id(df["race_id"])
    df["year"] = df["race_id"].str.slice(0, 4).astype(int)
    df["finish"] = num(df["finish"])
    df["corner4"] = num(df["corner4"])
    df["field_size"] = num(df.get("field_size", pd.Series(np.nan, index=df.index))).fillna(
        num(df.get("field_size2", pd.Series(np.nan, index=df.index)))
    )
    if "abnormal_code" in df.columns:
        df = df[num(df["abnormal_code"]).fillna(0).eq(0)].copy()
    df = df[df["finish"].notna()].copy()

    race_rows: list[dict[str, Any]] = []
    race_feature_cols = [c for c in feature_cols if c in df.columns]
    for race_id, g in df.groupby("race_id", sort=False):
        first = g.iloc[0]
        field_size = float(num(g["field_size"]).dropna().iloc[0]) if num(g["field_size"]).notna().any() else float(len(g))
        top3 = g[g["finish"].between(1, 3)].copy()
        top3_styles = [style_from_corner(c, field_size) for c in top3["corner4"]]
        front_stalker_count = sum(s in {"front", "stalker"} for s in top3_styles)
        closer_count = sum(s == "closer" for s in top3_styles)
        avg_corner4 = float(top3["corner4"].mean()) if not top3.empty else np.nan
        actual_shape = "mixed"
        if front_stalker_count >= 2 and (pd.isna(avg_corner4) or avg_corner4 <= max(5, field_size * 0.45)):
            actual_shape = "front_stalker"
        elif closer_count >= 2 or (pd.notna(avg_corner4) and avg_corner4 >= field_size * 0.55):
            actual_shape = "closer"

        row: dict[str, Any] = {
            "race_id": race_id,
            "year": int(first["year"]),
            "date_s": first.get("date_s", ""),
            "venue": first.get("venue", ""),
            "race_no": first.get("race_no", np.nan),
            "race_name": first.get("race_name", ""),
            "surface": first.get("surface", ""),
            "distance": first.get("distance", np.nan),
            "going": first.get("going", ""),
            "field_size": field_size,
            "distance_bin": distance_bin(pd.Series([first.get("distance", np.nan)])).iloc[0],
            "going_bin": going_bin(pd.Series([first.get("going", "")])).iloc[0],
            "venue_group": venue_group(pd.Series([first.get("venue", "")])).iloc[0],
            "actual_bias_shape": actual_shape,
            "target_front_stalker": int(actual_shape == "front_stalker"),
            "target_closer": int(actual_shape == "closer"),
            "top3_avg_corner4": avg_corner4,
            "top3_front_stalker_count": int(front_stalker_count),
            "top3_closer_count": int(closer_count),
            "winner_corner4": float(g.loc[g["finish"].eq(1), "corner4"].iloc[0]) if g["finish"].eq(1).any() else np.nan,
            "winner_pop": float(num(g.loc[g["finish"].eq(1), "popularity"]).iloc[0]) if "popularity" in g and g["finish"].eq(1).any() else np.nan,
        }
        for col in race_feature_cols:
            row[col] = float(num(g[col]).mean())
        race_rows.append(row)
    races = pd.DataFrame(race_rows)
    races["pressure_bucket"] = pd.qcut(
        races["race_pace_collapse_risk"].rank(method="first"),
        q=4,
        labels=["low", "mid_low", "mid_high", "high"],
    )
    races["front_count_bucket"] = pd.cut(
        races["race_front_runner_count"],
        bins=[-0.1, 1.5, 3.5, 5.5, 99],
        labels=["0-1", "2-3", "4-5", "6+"],
    )
    return races


def summarize(group: pd.DataFrame, by: list[str], min_races: int = 15) -> pd.DataFrame:
    out = (
        group.groupby(by, dropna=False)
        .agg(
            races=("race_id", "size"),
            front_stalker_rate=("target_front_stalker", "mean"),
            closer_rate=("target_closer", "mean"),
            avg_top3_corner4=("top3_avg_corner4", "mean"),
            avg_pressure=("race_pace_collapse_risk", "mean"),
            avg_front_count=("race_front_runner_count", "mean"),
            avg_winner_pop=("winner_pop", "mean"),
        )
        .reset_index()
    )
    out = out[out["races"] >= min_races].copy()
    return out.sort_values(["front_stalker_rate", "races"], ascending=[False, False])


def model_compare(races: pd.DataFrame) -> pd.DataFrame:
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except Exception as exc:
        return pd.DataFrame([{"model": "sklearn_unavailable", "error": str(exc)}])

    numeric_pace = [
        "race_pace_collapse_risk",
        "race_slow_pace_risk",
        "race_early_pressure_score",
        "race_front_runner_count",
        "race_closer_count",
        "field_size",
    ]
    numeric_context = numeric_pace + [
        "distance",
        "race_front_runner_ratio",
        "race_closer_ratio",
        "race_deep_closer_count",
        "front_advantage_score",
        "draw_pace_fit_score",
    ]
    cats = ["venue", "surface", "going_bin", "distance_bin", "venue_group", "front_count_bucket"]
    for col in numeric_context:
        if col not in races.columns:
            races[col] = 0.0
    for col in cats:
        if col not in races.columns:
            races[col] = "unknown"

    rows = []
    splits = [(2025, races["year"] < 2025, races["year"].eq(2025)), (2026, races["year"] < 2026, races["year"].eq(2026))]
    configs = [
        ("pace_only", numeric_pace, []),
        ("pace_plus_course_context", numeric_context, cats),
    ]
    for test_year, train_mask, test_mask in splits:
        train = races[train_mask].copy()
        test = races[test_mask].copy()
        if len(train) < 100 or len(test) < 30:
            continue
        y_train = train["target_front_stalker"].astype(int)
        y_test = test["target_front_stalker"].astype(int)

        # Baseline: slow risk minus collapse risk means "front should survive".
        baseline_score = (
            pd.to_numeric(test["race_slow_pace_risk"], errors="coerce").fillna(0)
            - pd.to_numeric(test["race_pace_collapse_risk"], errors="coerce").fillna(0)
        )
        try:
            rows.append(
                {
                    "model": "simple_slow_minus_collapse",
                    "test_year": test_year,
                    "train_races": len(train),
                    "test_races": len(test),
                    "auc": roc_auc_score(y_test, baseline_score),
                    "accuracy": accuracy_score(y_test, baseline_score >= baseline_score.median()),
                }
            )
        except Exception:
            pass

        for name, nums, cat_cols in configs:
            transformers = []
            if nums:
                transformers.append(("num", StandardScaler(), nums))
            if cat_cols:
                transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
            pre = ColumnTransformer(transformers)
            model = Pipeline(
                [
                    ("pre", pre),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
                ]
            )
            model.fit(train[nums + cat_cols], y_train)
            prob = model.predict_proba(test[nums + cat_cols])[:, 1]
            rows.append(
                {
                    "model": name,
                    "test_year": test_year,
                    "train_races": len(train),
                    "test_races": len(test),
                    "auc": roc_auc_score(y_test, prob),
                    "accuracy": accuracy_score(y_test, prob >= 0.5),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    races = build_race_table(args.input_csv)
    races.to_csv(args.out_dir / "race_pace_bias_table.csv", index=False, encoding="utf-8-sig")

    high_pressure = races[races["pressure_bucket"].eq("high")].copy()
    summaries = {
        "high_pressure_by_venue_surface": summarize(high_pressure, ["venue", "surface"], min_races=8),
        "high_pressure_by_venue_surface_going": summarize(high_pressure, ["venue", "surface", "going_bin"], min_races=5),
        "high_pressure_by_venue_surface_distance": summarize(high_pressure, ["venue", "surface", "distance_bin"], min_races=5),
        "all_by_expected_pressure_bucket": summarize(races, ["pressure_bucket", "venue_group", "surface"], min_races=20),
    }
    for name, df in summaries.items():
        df.to_csv(args.out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    model = model_compare(races)
    model.to_csv(args.out_dir / "front_stalker_model_comparison.csv", index=False, encoding="utf-8-sig")

    high_front = high_pressure[high_pressure["target_front_stalker"].eq(1)].copy()
    high_front.to_csv(args.out_dir / "high_pressure_front_survival_races.csv", index=False, encoding="utf-8-sig")
    high_closer = high_pressure[high_pressure["target_closer"].eq(1)].copy()
    high_closer.to_csv(args.out_dir / "high_pressure_closer_conversion_races.csv", index=False, encoding="utf-8-sig")

    payload = {
        "input": str(args.input_csv),
        "out_dir": str(args.out_dir),
        "races": int(len(races)),
        "years": [int(x) for x in sorted(races["year"].dropna().unique())],
        "overall_front_stalker_rate": float(races["target_front_stalker"].mean()),
        "overall_closer_rate": float(races["target_closer"].mean()),
        "high_pressure_races": int(len(high_pressure)),
        "high_pressure_front_stalker_rate": float(high_pressure["target_front_stalker"].mean()) if len(high_pressure) else None,
        "high_pressure_closer_rate": float(high_pressure["target_closer"].mean()) if len(high_pressure) else None,
        "model_comparison": model.to_dict(orient="records"),
        "outputs": {name: str(args.out_dir / f"{name}.csv") for name in summaries},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
