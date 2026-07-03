from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_json_config, model_categorical_features, model_numeric_features  # noqa: E402
from src.train.simple_ranker import SimpleRaceRanker  # noqa: E402


COL_DATE = "日付"
COL_SEX = "性別"
COL_HORSE_NO = "馬番"
COL_POPULARITY = "人気"
COL_DISTANCE = "距離"
COL_SURFACE = "芝・ダ"
COL_WIN_PAY = "単勝配当"
COL_PLACE_PAY = "複勝配当"

MALE = "牡"
GELDING = "セ"

NEW_NUMERIC_FEATURES = [
    "is_gelding_flag",
    "known_gelding_debut_flag",
    "known_gelding_second_start_flag",
    "known_gelding_third_start_flag",
    "known_gelding_4plus_start_flag",
    "gelding_start_no_since_transition_capped",
    "gelding_days_since_prev",
    "gelding_debut_popular_1_3_flag",
    "gelding_debut_unpopular_flag",
    "gelding_debut_surface_switch_flag",
    "gelding_debut_prev_good_flag",
    "gelding_debut_prev_bad_flag",
    "gelding_second_popular_1_3_flag",
    "gelding_second_unpopular_flag",
    "gelding_second_shorten_flag",
    "gelding_second_surface_switch_flag",
    "gelding_established_flag",
]

NEW_CATEGORICAL_FEATURES = [
    "gelding_phase",
    "gelding_phase_popularity_bucket",
    "gelding_phase_surface_change",
    "gelding_phase_distance_change",
    "gelding_phase_prev_result",
]


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _clean_race_id(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(16)


def _parse_date(df: pd.DataFrame, race_id_col: str) -> pd.Series:
    rid = df[race_id_col].astype(str).str.extract(r"(\d{8})", expand=False)
    parsed = pd.to_datetime(rid, format="%Y%m%d", errors="coerce")
    if COL_DATE in df.columns:
        raw = df[COL_DATE].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
        yy = pd.to_numeric(raw.str.slice(0, 2), errors="coerce")
        mm = raw.str.slice(2, 4)
        dd = raw.str.slice(4, 6)
        year = np.where(yy >= 70, 1900 + yy, 2000 + yy)
        raw_date = pd.to_datetime(
            pd.Series(year, index=df.index).astype("Int64").astype(str) + mm + dd,
            format="%Y%m%d",
            errors="coerce",
        )
        parsed = parsed.fillna(raw_date)
    return parsed


def _pick_col(df: pd.DataFrame, candidates: list[str], fallback: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    return fallback


def _bucket_popularity(popularity: pd.Series) -> pd.Series:
    p = _num(popularity)
    out = pd.Series("missing", index=popularity.index, dtype="object")
    out.loc[p.between(1, 3)] = "pop_1_3"
    out.loc[p.between(4, 6)] = "pop_4_6"
    out.loc[p.between(7, 10)] = "pop_7_10"
    out.loc[p.ge(11)] = "pop_11_plus"
    return out.astype("string")


def _bucket_prev_result(prev_finish: pd.Series) -> pd.Series:
    f = _num(prev_finish)
    out = pd.Series("no_prev", index=prev_finish.index, dtype="object")
    out.loc[f.eq(1)] = "prev_win"
    out.loc[f.between(2, 3)] = "prev_2_3"
    out.loc[f.between(4, 5)] = "prev_4_5"
    out.loc[f.ge(6)] = "prev_6_plus"
    return out.astype("string")


def add_gelding_transition_features(
    frame: pd.DataFrame,
    race_id_col: str,
    horse_id_col: str,
    horse_no_col: str,
    sex_col: str,
    finish_col: str,
) -> pd.DataFrame:
    out = frame.copy()
    out["_gelding_order"] = np.arange(len(out))
    out["_race_id_norm"] = _clean_race_id(out[race_id_col])
    out["_horse_id_norm"] = out[horse_id_col].astype("string")
    out["_horse_no_num"] = _num(out[horse_no_col]).astype("Int64")
    out["_race_date"] = _parse_date(out, race_id_col)

    work = out.sort_values(["_horse_id_norm", "_race_date", "_race_id_norm", "_horse_no_num"]).copy()
    work["_sex"] = work[sex_col].astype("string")
    work["_is_gelding"] = work["_sex"].eq(GELDING)
    work["_prev_sex"] = work.groupby("_horse_id_norm", sort=False)["_sex"].shift(1)
    work["_prev_race_date"] = work.groupby("_horse_id_norm", sort=False)["_race_date"].shift(1)
    work["_prev_finish"] = work.groupby("_horse_id_norm", sort=False)[finish_col].shift(1)
    work["_prev_surface"] = work.groupby("_horse_id_norm", sort=False)[COL_SURFACE].shift(1) if COL_SURFACE in work.columns else pd.NA
    work["_prev_distance"] = work.groupby("_horse_id_norm", sort=False)[COL_DISTANCE].shift(1) if COL_DISTANCE in work.columns else pd.NA

    work["gelding_days_since_prev"] = (
        work["_race_date"] - work["_prev_race_date"]
    ).dt.days.where(work["_prev_race_date"].notna(), 0).fillna(0).clip(lower=0, upper=999)
    work["_known_gelding_debut"] = (work["_is_gelding"] & work["_prev_sex"].eq(MALE)).fillna(False)

    transition_no = work["_known_gelding_debut"].fillna(False).astype(int).groupby(work["_horse_id_norm"], sort=False).cumsum()
    after_known_transition = work["_is_gelding"] & transition_no.gt(0)
    work["_gelding_start_no"] = after_known_transition.astype(int).groupby(
        [work["_horse_id_norm"], transition_no],
        sort=False,
    ).cumsum()
    work.loc[~after_known_transition, "_gelding_start_no"] = 0

    work["is_gelding_flag"] = work["_is_gelding"].astype(float)
    work["known_gelding_debut_flag"] = work["_known_gelding_debut"].astype(float)
    work["known_gelding_second_start_flag"] = work["_gelding_start_no"].eq(2).astype(float)
    work["known_gelding_third_start_flag"] = work["_gelding_start_no"].eq(3).astype(float)
    work["known_gelding_4plus_start_flag"] = work["_gelding_start_no"].ge(4).astype(float)
    work["gelding_start_no_since_transition_capped"] = work["_gelding_start_no"].clip(upper=4).astype(float)
    work["gelding_established_flag"] = (
        work["_is_gelding"] & work["_prev_sex"].eq(GELDING) & work["_gelding_start_no"].eq(0)
    ).astype(float)

    pop = _num(work[COL_POPULARITY]) if COL_POPULARITY in work.columns else pd.Series(np.nan, index=work.index)
    prev_finish = _num(work["_prev_finish"])
    surface_change = pd.Series("same_or_unknown", index=work.index, dtype="object")
    if COL_SURFACE in work.columns:
        cur_surface = work[COL_SURFACE].astype("string")
        prev_surface = work["_prev_surface"].astype("string")
        surface_change.loc[prev_surface.notna() & cur_surface.ne(prev_surface)] = "surface_switch"
        surface_change.loc[prev_surface.notna() & cur_surface.eq(prev_surface)] = "same_surface"
    distance_change = pd.Series("same_or_unknown", index=work.index, dtype="object")
    if COL_DISTANCE in work.columns:
        diff = _num(work[COL_DISTANCE]) - _num(work["_prev_distance"])
        distance_change.loc[diff.le(-200)] = "shorten"
        distance_change.loc[diff.ge(200)] = "extend"
        distance_change.loc[diff.abs().lt(200)] = "same"
    work["_surface_change"] = surface_change.astype("string")
    work["_distance_change"] = distance_change.astype("string")

    debut = work["known_gelding_debut_flag"].eq(1)
    second = work["known_gelding_second_start_flag"].eq(1)
    work["gelding_debut_popular_1_3_flag"] = (debut & pop.between(1, 3)).astype(float)
    work["gelding_debut_unpopular_flag"] = (debut & pop.ge(4)).astype(float)
    work["gelding_debut_surface_switch_flag"] = (debut & work["_surface_change"].eq("surface_switch")).astype(float)
    work["gelding_debut_prev_good_flag"] = (debut & prev_finish.le(3)).astype(float)
    work["gelding_debut_prev_bad_flag"] = (debut & prev_finish.ge(6)).astype(float)
    work["gelding_second_popular_1_3_flag"] = (second & pop.between(1, 3)).astype(float)
    work["gelding_second_unpopular_flag"] = (second & pop.ge(4)).astype(float)
    work["gelding_second_shorten_flag"] = (second & work["_distance_change"].eq("shorten")).astype(float)
    work["gelding_second_surface_switch_flag"] = (second & work["_surface_change"].eq("surface_switch")).astype(float)

    work["gelding_phase"] = "non_gelding"
    work.loc[work["_is_gelding"] & work["_prev_sex"].isna(), "gelding_phase"] = "first_seen_as_gelding_unknown_timing"
    work.loc[debut, "gelding_phase"] = "known_gelding_debut"
    work.loc[second, "gelding_phase"] = "known_gelding_second_start"
    work.loc[work["known_gelding_third_start_flag"].eq(1), "gelding_phase"] = "known_gelding_third_start"
    work.loc[work["known_gelding_4plus_start_flag"].eq(1), "gelding_phase"] = "known_gelding_4plus_start"
    work.loc[
        work["_is_gelding"] & work["gelding_phase"].eq("non_gelding"),
        "gelding_phase",
    ] = "established_gelding_unknown_transition"

    work["_pop_bucket"] = _bucket_popularity(pop)
    work["_prev_result_bucket"] = _bucket_prev_result(prev_finish)
    for col, source in [
        ("gelding_phase_popularity_bucket", "_pop_bucket"),
        ("gelding_phase_surface_change", "_surface_change"),
        ("gelding_phase_distance_change", "_distance_change"),
        ("gelding_phase_prev_result", "_prev_result_bucket"),
    ]:
        work[col] = work["gelding_phase"].astype(str) + "__" + work[source].astype(str)

    for col in NEW_NUMERIC_FEATURES:
        work[col] = _num(work[col]).fillna(0.0)
    for col in NEW_CATEGORICAL_FEATURES:
        work[col] = work[col].astype("string").fillna("missing")

    keep = ["_gelding_order", *NEW_NUMERIC_FEATURES, *NEW_CATEGORICAL_FEATURES]
    enriched = out.merge(work[keep], on="_gelding_order", how="left")
    for col in NEW_NUMERIC_FEATURES:
        enriched[col] = _num(enriched[col]).fillna(0.0)
    for col in NEW_CATEGORICAL_FEATURES:
        enriched[col] = enriched[col].astype("string").fillna("missing")
    return enriched.drop(
        columns=[
            c
            for c in enriched.columns
            if c.startswith("_gelding") or c.startswith("_race_id_norm") or c.startswith("_horse_id_norm")
            or c.startswith("_horse_no_num") or c.startswith("_race_date")
        ],
        errors="ignore",
    )


def metric_summary(df: pd.DataFrame, scores: np.ndarray, race_col: str, rank_col: str) -> dict[str, Any]:
    scored = df.copy()
    scored["ai_score"] = scores
    scored["ai_rank"] = scored.groupby(race_col)["ai_score"].rank(ascending=False, method="first").astype(int)
    top1 = scored[scored["ai_rank"] == 1]
    top3 = scored[scored["ai_rank"] <= 3]
    rank = _num(scored[rank_col])

    win_pay_col = _pick_col(scored, [COL_WIN_PAY, "win_pay"], COL_WIN_PAY)
    place_pay_col = _pick_col(scored, [COL_PLACE_PAY, "place_pay"], COL_PLACE_PAY)
    win_pay = _num(top1.get(win_pay_col, pd.Series(0, index=top1.index))).fillna(0).where(
        _num(top1[rank_col]).eq(1), 0.0
    )
    place_pay = _num(top1.get(place_pay_col, pd.Series(0, index=top1.index))).fillna(0).where(
        _num(top1[rank_col]).le(3), 0.0
    )
    top3_win_pay = _num(top3.get(win_pay_col, pd.Series(0, index=top3.index))).fillna(0).where(
        _num(top3[rank_col]).eq(1), 0.0
    )
    top3_place_pay = _num(top3.get(place_pay_col, pd.Series(0, index=top3.index))).fillna(0).where(
        _num(top3[rank_col]).le(3), 0.0
    )
    return {
        "rows": int(len(scored)),
        "races": int(scored[race_col].nunique()),
        "top1_win_rate_pct": round(float(_num(top1[rank_col]).eq(1).mean() * 100), 3),
        "top1_top3_rate_pct": round(float(_num(top1[rank_col]).le(3).mean() * 100), 3),
        "top1_win_roi_pct": round(float(win_pay.sum() / (len(top1) * 100.0) * 100), 3) if len(top1) else 0,
        "top1_place_roi_pct": round(float(place_pay.sum() / (len(top1) * 100.0) * 100), 3) if len(top1) else 0,
        "top3_contains_winner_rate_pct": round(
            float(top3.groupby(race_col).apply(lambda g: _num(g[rank_col]).eq(1).any()).mean() * 100),
            3,
        )
        if len(top3)
        else 0,
        "top3_win_roi_pct": round(float(top3_win_pay.sum() / (len(top3) * 100.0) * 100), 3) if len(top3) else 0,
        "top3_place_roi_pct": round(float(top3_place_pay.sum() / (len(top3) * 100.0) * 100), 3) if len(top3) else 0,
        "winner_mean_ai_rank": round(float(scored.loc[rank.eq(1), "ai_rank"].mean()), 3),
    }


def fit_eval(
    train: pd.DataFrame,
    test: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    config: dict[str, Any],
) -> tuple[SimpleRaceRanker, dict[str, Any]]:
    model = SimpleRaceRanker(
        numeric_features=numeric,
        categorical_features=categorical,
        categorical_top_k=int(config["training"].get("categorical_top_k", 80)),
        ridge_alpha=float(config["training"].get("ridge_alpha", 10.0)),
    ).fit(train, "target_score")
    race_col = config["data"]["race_id_column"]
    rank_col = config["data"]["rank_column"]
    return model, metric_summary(test, model.predict(test), race_col, rank_col)


def coefficient_table(model: SimpleRaceRanker) -> pd.DataFrame:
    names = model.feature_names_ or []
    coefs = model.coefficients_ if model.coefficients_ is not None else []
    rows = []
    for name, coef in zip(names, coefs):
        if name in NEW_NUMERIC_FEATURES or any(name.startswith(f"{cat}=") for cat in NEW_CATEGORICAL_FEATURES):
            rows.append({"feature": name, "coef": float(coef), "abs_coef": abs(float(coef))})
    return pd.DataFrame(rows).sort_values("abs_coef", ascending=False)


def model_segment_summary(
    df: pd.DataFrame,
    scores: np.ndarray,
    variant: str,
    race_col: str,
    rank_col: str,
) -> pd.DataFrame:
    scored = df.copy()
    scored["ai_score"] = scores
    scored["ai_rank"] = scored.groupby(race_col)["ai_score"].rank(ascending=False, method="first").astype(int)
    scored["_actual_win"] = _num(scored[rank_col]).eq(1)
    scored["_actual_top3"] = _num(scored[rank_col]).le(3)
    win_pay_col = _pick_col(scored, [COL_WIN_PAY, "win_pay"], COL_WIN_PAY)
    place_pay_col = _pick_col(scored, [COL_PLACE_PAY, "place_pay"], COL_PLACE_PAY)

    rows = []
    for segment, frame in [
        ("top1_gelding_phase", scored[scored["ai_rank"].eq(1)]),
        ("all_gelding_phase", scored[scored["gelding_phase"].ne("non_gelding")]),
    ]:
        if frame.empty:
            continue
        for val, g in frame.groupby("gelding_phase", dropna=False):
            if len(g) < 20:
                continue
            win_pay = _num(g.get(win_pay_col, pd.Series(0, index=g.index))).fillna(0).where(g["_actual_win"], 0)
            place_pay = _num(g.get(place_pay_col, pd.Series(0, index=g.index))).fillna(0).where(g["_actual_top3"], 0)
            rows.append(
                {
                    "variant": variant,
                    "segment": segment,
                    "gelding_phase": str(val),
                    "rows": int(len(g)),
                    "races": int(g[race_col].nunique()),
                    "win_rate_pct": round(float(g["_actual_win"].mean() * 100), 3),
                    "top3_rate_pct": round(float(g["_actual_top3"].mean() * 100), 3),
                    "win_roi_pct": round(float(win_pay.sum() / (len(g) * 100.0) * 100), 3),
                    "place_roi_pct": round(float(place_pay.sum() / (len(g) * 100.0) * 100), 3),
                    "mean_ai_rank": round(float(g["ai_rank"].mean()), 3),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B test gelding transition features in the horse ranker.")
    parser.add_argument("--config", default="config/baseline_features_workout_content_bridge_vertical_context_safe_dedup.json")
    parser.add_argument("--train-csv", default="outputs/analysis/vertical_horse_context_v1/train_features_with_vertical_context.csv")
    parser.add_argument("--test-csv", default="outputs/analysis/vertical_horse_context_v1/test_features_with_vertical_context.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/gelding_transition_model_v1")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_json_config(args.config)
    race_col = config["data"]["race_id_column"]
    horse_col = config["data"]["horse_id_column"]
    rank_col = config["data"]["rank_column"]

    header = pd.read_csv(args.train_csv, nrows=0, low_memory=False)
    horse_no_col = COL_HORSE_NO if COL_HORSE_NO in header.columns else "horse_no"
    sex_col = COL_SEX if COL_SEX in header.columns else "sex"
    needed = list(
        dict.fromkeys(
            [
                race_col,
                horse_col,
                horse_no_col,
                sex_col,
                rank_col,
                COL_DATE,
                COL_POPULARITY,
                COL_SURFACE,
                COL_DISTANCE,
                COL_WIN_PAY,
                COL_PLACE_PAY,
                "target_score",
                *model_numeric_features(config),
                *model_categorical_features(config),
            ]
        )
    )
    usecols_train = [c for c in needed if c in header.columns]
    train = pd.read_csv(args.train_csv, usecols=usecols_train, low_memory=False)
    header_test = pd.read_csv(args.test_csv, nrows=0, low_memory=False)
    usecols_test = [c for c in needed if c in header_test.columns]
    test = pd.read_csv(args.test_csv, usecols=usecols_test, low_memory=False)

    combined = pd.concat([train.assign(_split="train"), test.assign(_split="test")], ignore_index=True)
    combined = add_gelding_transition_features(
        combined,
        race_id_col=race_col,
        horse_id_col=horse_col,
        horse_no_col=horse_no_col,
        sex_col=sex_col,
        finish_col=rank_col,
    )
    train_x = combined[combined["_split"].eq("train")].drop(columns=["_split"]).copy()
    test_x = combined[combined["_split"].eq("test")].drop(columns=["_split"]).copy()

    train_x.to_csv(out_dir / "train_with_gelding_transition.csv", index=False, encoding="utf-8-sig")
    test_x.to_csv(out_dir / "test_with_gelding_transition.csv", index=False, encoding="utf-8-sig")

    base_numeric = model_numeric_features(config)
    base_categorical = model_categorical_features(config)
    plus_numeric = list(dict.fromkeys([*base_numeric, *NEW_NUMERIC_FEATURES]))
    plus_categorical = list(dict.fromkeys([*base_categorical, *NEW_CATEGORICAL_FEATURES]))

    base_model, base_metrics = fit_eval(train_x, test_x, base_numeric, base_categorical, config)
    plus_model, plus_metrics = fit_eval(train_x, test_x, plus_numeric, plus_categorical, config)
    base_scores = base_model.predict(test_x)
    plus_scores = plus_model.predict(test_x)
    rows = [
        {
            "variant": "baseline_retrained",
            "numeric_features": len(base_numeric),
            "categorical_features": len(base_categorical),
            **base_metrics,
        },
        {
            "variant": "plus_gelding_transition",
            "numeric_features": len(plus_numeric),
            "categorical_features": len(plus_categorical),
            **plus_metrics,
        },
    ]
    summary = pd.DataFrame(rows)
    diff = summary.set_index("variant").diff().tail(1).reset_index(drop=True)
    diff.insert(0, "variant", "plus_minus_baseline")
    pd.concat([summary, diff], ignore_index=True).to_csv(out_dir / "model_ab_summary.csv", index=False, encoding="utf-8-sig")

    coef = coefficient_table(plus_model)
    coef.to_csv(out_dir / "gelding_transition_feature_coefficients.csv", index=False, encoding="utf-8-sig")
    seg = pd.concat(
        [
            model_segment_summary(test_x, base_scores, "baseline_retrained", race_col, rank_col),
            model_segment_summary(test_x, plus_scores, "plus_gelding_transition", race_col, rank_col),
        ],
        ignore_index=True,
    )
    seg.to_csv(out_dir / "segment_model_ab_summary.csv", index=False, encoding="utf-8-sig")

    scored_test = test_x.copy()
    scored_test["baseline_score"] = base_scores
    scored_test["plus_score"] = plus_scores
    scored_test["baseline_rank"] = scored_test.groupby(race_col)["baseline_score"].rank(ascending=False, method="first").astype(int)
    scored_test["plus_rank"] = scored_test.groupby(race_col)["plus_score"].rank(ascending=False, method="first").astype(int)
    scored_test["rank_delta_plus_minus_base"] = scored_test["plus_rank"] - scored_test["baseline_rank"]
    scored_test.to_csv(out_dir / "test_scored_ab.csv", index=False, encoding="utf-8-sig")

    with (out_dir / "gelding_transition_ranker.pkl").open("wb") as f:
        pickle.dump(plus_model, f)
    meta = {
        "config": args.config,
        "train_csv": args.train_csv,
        "test_csv": args.test_csv,
        "new_numeric_features": NEW_NUMERIC_FEATURES,
        "new_categorical_features": NEW_CATEGORICAL_FEATURES,
        "summary": pd.concat([summary, diff], ignore_index=True).to_dict(orient="records"),
        "segment_summary": seg.to_dict(orient="records"),
        "coefficient_top": coef.head(30).to_dict(orient="records"),
        "notes": [
            "Actual gelding surgery dates are not available; sex-label transition from male to gelding is used as a practical proxy.",
            "All transition features use current entry information plus only prior horse history.",
            "Current popularity is treated as an operationally available market feature and should be refreshed close to betting time.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
