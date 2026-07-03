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


COL_RACE_ID = "\u30ec\u30fc\u30b9ID(\u65b0/\u99ac\u756a\u7121)"
COL_HORSE_ID = "\u8840\u7d71\u767b\u9332\u756a\u53f7"
COL_HORSE_NO = "\u99ac\u756a"
COL_RACE_NAME = "\u30ec\u30fc\u30b9\u540d"
COL_SEX = "\u6027\u5225"
COL_FINISH = "\u78ba\u5b9a\u7740\u9806"
COL_DATE = "\u65e5\u4ed8"
COL_WIN_PAY = "\u5358\u52dd\u914d\u5f53"
COL_PLACE_PAY = "\u8907\u52dd\u914d\u5f53"

FEMALE = "\u725d"

NEW_NUMERIC_FEATURES = [
    "female_current_only_flag",
    "female_prev_only_flag",
    "female_only_to_mixed_flag",
    "mixed_to_female_only_flag",
    "female_only_to_female_only_flag",
    "mixed_to_mixed_female_flag",
    "prev_female_only_win_flag",
    "prev_female_only_top3_flag",
    "prev_mixed_win_flag",
    "prev_mixed_top3_flag",
    "female_only_to_mixed_prev_good_flag",
    "female_only_to_mixed_prev_bad_flag",
    "mixed_to_female_only_prev_good_flag",
    "mixed_to_female_only_prev_bad_flag",
    "female_condition_switch_days_since_prev",
]

NEW_CATEGORICAL_FEATURES = [
    "female_condition_transition",
    "female_transition_prev_result_bucket",
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
        raw_date = pd.to_datetime(pd.Series(year, index=df.index).astype("Int64").astype(str) + mm + dd, format="%Y%m%d", errors="coerce")
        parsed = parsed.fillna(raw_date)
    return parsed


def _pick_col(df: pd.DataFrame, candidates: list[str], fallback: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    return fallback


def add_female_condition_switch_features(
    frame: pd.DataFrame,
    race_id_col: str,
    horse_id_col: str,
    horse_no_col: str,
    race_name_col: str,
    sex_col: str,
    finish_col: str,
) -> pd.DataFrame:
    out = frame.copy()
    out["_female_switch_order"] = np.arange(len(out))
    out["_race_id_norm"] = _clean_race_id(out[race_id_col])
    out["_horse_id_norm"] = out[horse_id_col].astype("string")
    out["_horse_no_num"] = _num(out[horse_no_col]).astype("Int64")
    out["_race_date"] = _parse_date(out, race_id_col)

    sex = out[sex_col].astype("string")
    out["_is_female_horse"] = sex.eq(FEMALE)
    race_name = out[race_name_col].astype("string").fillna("")

    race_ctx = (
        out.groupby("_race_id_norm", sort=False)
        .agg(
            runners=("_horse_no_num", "count"),
            female_count=("_is_female_horse", "sum"),
            race_name=("_race_id_norm", lambda s: race_name.loc[s.index].iloc[0] if len(s) else ""),
        )
        .reset_index()
    )
    race_ctx["race_name_has_female"] = race_ctx["race_name"].astype(str).str.contains(FEMALE, regex=False)
    race_ctx["female_current_only_flag"] = (
        race_ctx["female_count"].eq(race_ctx["runners"]) | race_ctx["race_name_has_female"]
    ).astype(float)
    out = out.merge(
        race_ctx[["_race_id_norm", "female_current_only_flag"]],
        on="_race_id_norm",
        how="left",
    )

    work = out.sort_values(["_horse_id_norm", "_race_date", "_race_id_norm", "_horse_no_num"]).copy()
    prev_cols = ["female_current_only_flag", finish_col, "_race_date"]
    for col in prev_cols:
        work[f"prev__{col}"] = work.groupby("_horse_id_norm", sort=False)[col].shift(1)
    work["female_prev_only_flag"] = _num(work["prev__female_current_only_flag"]).fillna(0.0)
    prev_finish = _num(work[f"prev__{finish_col}"])
    work["prev_win_flag"] = prev_finish.eq(1).astype(float)
    work["prev_top3_flag"] = prev_finish.le(3).astype(float)
    work["prev_bad_flag"] = prev_finish.ge(6).astype(float)
    work["female_condition_switch_days_since_prev"] = (
        work["_race_date"] - work["prev___race_date"]
    ).dt.days.where(work["prev___race_date"].notna(), 0).fillna(0).clip(lower=0, upper=365)

    is_female = work["_is_female_horse"].astype(bool)
    cur_only = _num(work["female_current_only_flag"]).fillna(0.0).gt(0.5)
    prev_only = _num(work["female_prev_only_flag"]).fillna(0.0).gt(0.5)
    has_prev = work["prev___race_date"].notna()

    work["female_only_to_mixed_flag"] = (is_female & has_prev & prev_only & ~cur_only).astype(float)
    work["mixed_to_female_only_flag"] = (is_female & has_prev & ~prev_only & cur_only).astype(float)
    work["female_only_to_female_only_flag"] = (is_female & has_prev & prev_only & cur_only).astype(float)
    work["mixed_to_mixed_female_flag"] = (is_female & has_prev & ~prev_only & ~cur_only).astype(float)
    work["prev_female_only_win_flag"] = (is_female & prev_only & work["prev_win_flag"].eq(1)).astype(float)
    work["prev_female_only_top3_flag"] = (is_female & prev_only & work["prev_top3_flag"].eq(1)).astype(float)
    work["prev_mixed_win_flag"] = (is_female & ~prev_only & work["prev_win_flag"].eq(1)).astype(float)
    work["prev_mixed_top3_flag"] = (is_female & ~prev_only & work["prev_top3_flag"].eq(1)).astype(float)
    work["female_only_to_mixed_prev_good_flag"] = (
        work["female_only_to_mixed_flag"].eq(1) & work["prev_top3_flag"].eq(1)
    ).astype(float)
    work["female_only_to_mixed_prev_bad_flag"] = (
        work["female_only_to_mixed_flag"].eq(1) & work["prev_bad_flag"].eq(1)
    ).astype(float)
    work["mixed_to_female_only_prev_good_flag"] = (
        work["mixed_to_female_only_flag"].eq(1) & work["prev_top3_flag"].eq(1)
    ).astype(float)
    work["mixed_to_female_only_prev_bad_flag"] = (
        work["mixed_to_female_only_flag"].eq(1) & work["prev_bad_flag"].eq(1)
    ).astype(float)

    work["female_condition_transition"] = "not_female"
    work.loc[is_female & ~has_prev, "female_condition_transition"] = "female_no_prev"
    work.loc[work["female_only_to_mixed_flag"].eq(1), "female_condition_transition"] = "female_only_to_mixed"
    work.loc[work["mixed_to_female_only_flag"].eq(1), "female_condition_transition"] = "mixed_to_female_only"
    work.loc[work["female_only_to_female_only_flag"].eq(1), "female_condition_transition"] = "female_only_to_female_only"
    work.loc[work["mixed_to_mixed_female_flag"].eq(1), "female_condition_transition"] = "mixed_to_mixed_female"

    work["female_transition_prev_result_bucket"] = "not_female_or_no_prev"
    work.loc[is_female & has_prev & prev_finish.eq(1), "female_transition_prev_result_bucket"] = "prev_win"
    work.loc[
        is_female & has_prev & prev_finish.between(2, 3),
        "female_transition_prev_result_bucket",
    ] = "prev_2nd_3rd"
    work.loc[
        is_female & has_prev & prev_finish.between(4, 5),
        "female_transition_prev_result_bucket",
    ] = "prev_4th_5th"
    work.loc[
        is_female & has_prev & prev_finish.ge(6),
        "female_transition_prev_result_bucket",
    ] = "prev_6th_or_worse"
    work.loc[
        is_female & has_prev,
        "female_transition_prev_result_bucket",
    ] = work.loc[is_female & has_prev, "female_condition_transition"].astype(str) + "__" + work.loc[
        is_female & has_prev, "female_transition_prev_result_bucket"
    ].astype(str)

    for col in NEW_NUMERIC_FEATURES:
        work[col] = _num(work[col]).fillna(0.0)
    keep = ["_female_switch_order", *NEW_NUMERIC_FEATURES, *NEW_CATEGORICAL_FEATURES]
    enriched = out.merge(work[keep], on="_female_switch_order", how="left", suffixes=("", "_feature"))
    for col in NEW_NUMERIC_FEATURES:
        enriched[col] = _num(enriched[col]).fillna(0.0)
    for col in NEW_CATEGORICAL_FEATURES:
        enriched[col] = enriched[col].astype("string").fillna("missing")
    return enriched.drop(columns=[c for c in enriched.columns if c.startswith("_female_switch") or c.startswith("_race_id_norm") or c.startswith("_horse_id_norm") or c.startswith("_horse_no_num") or c.startswith("_race_date") or c.startswith("_is_female")], errors="ignore")


def metric_summary(df: pd.DataFrame, scores: np.ndarray, race_col: str, rank_col: str) -> dict[str, Any]:
    scored = df.copy()
    scored["ai_score"] = scores
    scored["ai_rank"] = scored.groupby(race_col)["ai_score"].rank(ascending=False, method="first").astype(int)
    top1 = scored[scored["ai_rank"] == 1]
    top3 = scored[scored["ai_rank"] <= 3]
    rank = _num(scored[rank_col])
    scored["target_win"] = rank.eq(1)
    scored["target_top3"] = rank.le(3)

    win_pay_col = _pick_col(scored, [COL_WIN_PAY, "win_pay", "単勝配当"], COL_WIN_PAY)
    place_pay_col = _pick_col(scored, [COL_PLACE_PAY, "place_pay", "複勝配当"], COL_PLACE_PAY)
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
        "top3_contains_winner_rate_pct": round(float(top3.groupby(race_col).apply(lambda g: _num(g[rank_col]).eq(1).any()).mean() * 100), 3)
        if len(top3)
        else 0,
        "top3_win_roi_pct": round(float(top3_win_pay.sum() / (len(top3) * 100.0) * 100), 3) if len(top3) else 0,
        "top3_place_roi_pct": round(float(top3_place_pay.sum() / (len(top3) * 100.0) * 100), 3) if len(top3) else 0,
        "winner_mean_ai_rank": round(float(scored.loc[_num(scored[rank_col]).eq(1), "ai_rank"].mean()), 3),
    }


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
    win_pay_col = _pick_col(scored, [COL_WIN_PAY, "win_pay", "単勝配当"], COL_WIN_PAY)
    place_pay_col = _pick_col(scored, [COL_PLACE_PAY, "place_pay", "複勝配当"], COL_PLACE_PAY)

    rows = []
    for group_col, frame in [
        ("top1_transition", scored[scored["ai_rank"].eq(1)]),
        ("all_female_transition", scored[scored["female_condition_transition"].ne("not_female")]),
    ]:
        if frame.empty:
            continue
        for val, g in frame.groupby("female_condition_transition", dropna=False):
            if len(g) < 20:
                continue
            win_pay = _num(g.get(win_pay_col, pd.Series(0, index=g.index))).fillna(0).where(g["_actual_win"], 0)
            place_pay = _num(g.get(place_pay_col, pd.Series(0, index=g.index))).fillna(0).where(g["_actual_top3"], 0)
            rows.append(
                {
                    "variant": variant,
                    "segment": group_col,
                    "female_condition_transition": str(val),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B test female-only/open switch features in the horse ranker.")
    parser.add_argument("--config", default="config/baseline_features_workout_content_bridge_vertical_context_safe_dedup.json")
    parser.add_argument("--train-csv", default="outputs/analysis/vertical_horse_context_v1/train_features_with_vertical_context.csv")
    parser.add_argument("--test-csv", default="outputs/analysis/vertical_horse_context_v1/test_features_with_vertical_context.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/female_condition_switch_model_v1")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_json_config(args.config)
    race_col = config["data"]["race_id_column"]
    horse_col = config["data"]["horse_id_column"]
    rank_col = config["data"]["rank_column"]

    header = pd.read_csv(args.train_csv, nrows=0, low_memory=False)
    race_name_col = COL_RACE_NAME if COL_RACE_NAME in header.columns else "race_name"
    sex_col = COL_SEX if COL_SEX in header.columns else "sex"
    horse_no_col = COL_HORSE_NO if COL_HORSE_NO in header.columns else "horse_no"

    needed = list(
        dict.fromkeys(
            [
                race_col,
                horse_col,
                horse_no_col,
                race_name_col,
                sex_col,
                rank_col,
                COL_DATE,
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
    combined = add_female_condition_switch_features(
        combined,
        race_id_col=race_col,
        horse_id_col=horse_col,
        horse_no_col=horse_no_col,
        race_name_col=race_name_col,
        sex_col=sex_col,
        finish_col=rank_col,
    )
    train_x = combined[combined["_split"].eq("train")].drop(columns=["_split"]).copy()
    test_x = combined[combined["_split"].eq("test")].drop(columns=["_split"]).copy()

    train_x.to_csv(out_dir / "train_with_female_condition_switch.csv", index=False, encoding="utf-8-sig")
    test_x.to_csv(out_dir / "test_with_female_condition_switch.csv", index=False, encoding="utf-8-sig")

    base_numeric = model_numeric_features(config)
    base_categorical = model_categorical_features(config)
    plus_numeric = list(dict.fromkeys([*base_numeric, *NEW_NUMERIC_FEATURES]))
    plus_categorical = list(dict.fromkeys([*base_categorical, *NEW_CATEGORICAL_FEATURES]))

    base_model, base_metrics = fit_eval(train_x, test_x, base_numeric, base_categorical, config)
    plus_model, plus_metrics = fit_eval(train_x, test_x, plus_numeric, plus_categorical, config)
    base_scores = base_model.predict(test_x)
    plus_scores = plus_model.predict(test_x)
    rows = [
        {"variant": "baseline_retrained", "numeric_features": len(base_numeric), "categorical_features": len(base_categorical), **base_metrics},
        {"variant": "plus_female_condition_switch", "numeric_features": len(plus_numeric), "categorical_features": len(plus_categorical), **plus_metrics},
    ]
    summary = pd.DataFrame(rows)
    diff = summary.set_index("variant").diff().tail(1).reset_index(drop=True)
    diff.insert(0, "variant", "plus_minus_baseline")
    pd.concat([summary, diff], ignore_index=True).to_csv(out_dir / "model_ab_summary.csv", index=False, encoding="utf-8-sig")

    coef = coefficient_table(plus_model)
    coef.to_csv(out_dir / "female_condition_feature_coefficients.csv", index=False, encoding="utf-8-sig")
    seg = pd.concat(
        [
            model_segment_summary(test_x, base_scores, "baseline_retrained", race_col, rank_col),
            model_segment_summary(test_x, plus_scores, "plus_female_condition_switch", race_col, rank_col),
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

    with (out_dir / "female_condition_switch_ranker.pkl").open("wb") as f:
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
            "All female-condition switch features use current race entries/race name and prior horse race only.",
            "No current finish, payoff, or future female-only results are used as model features.",
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
