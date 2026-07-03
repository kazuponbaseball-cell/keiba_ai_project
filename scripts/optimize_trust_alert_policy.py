from __future__ import annotations

import argparse
import json
import pickle
import sys
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


RACE_COL = "\u30ec\u30fc\u30b9ID(\u65b0/\u99ac\u756a\u7121)"
DATE_COL = "\u65e5\u4ed8"
VENUE_COL = "\u5834\u6240"
AGE_COL = "\u5e74\u9f62"
SURFACE_COL = "\u829d\u30fb\u30c0"
DISTANCE_COL = "\u8ddd\u96e2"
CLASS_COL = "\u30af\u30e9\u30b9\u540d"
GOING_COL = "\u99ac\u5834\u72b6\u614b"
POPULARITY_COL = "\u4eba\u6c17"
ODDS_COL = "\u5358\u52dd\u30aa\u30c3\u30ba"
WIN_PAY_COL = "\u5358\u52dd\u914d\u5f53"
PLACE_PAY_COL = "\u8907\u52dd\u914d\u5f53"


def _num(series: pd.Series | None, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if series is None:
        if index is None:
            raise ValueError("index required")
        return pd.Series(default, index=index)
    if series.dtype == object or str(series.dtype).startswith("string"):
        series = (
            series.astype("string")
            .str.replace(",", "", regex=False)
            .str.replace("+", "", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
        )
    return pd.to_numeric(series, errors="coerce")


def _q(frame: pd.DataFrame, column: str, quantile: float, default: float = 0.0) -> float:
    if column not in frame.columns:
        return default
    values = _num(frame[column]).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return default
    return float(values.quantile(quantile))


def _score(frame: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    with model_path.open("rb") as f:
        model = pickle.load(f)
    out = frame.copy()
    out["ai_score"] = model.predict(out)
    out["ai_rank"] = out.groupby(RACE_COL)["ai_score"].rank(ascending=False, method="first").astype(int)
    out["popularity_num"] = _num(out.get(POPULARITY_COL), out.index)
    out["odds_decimal"] = _num(out.get(ODDS_COL), out.index)
    out["pop_rank"] = out.groupby(RACE_COL)["popularity_num"].rank(ascending=True, method="first")
    out["ai_pop_gap"] = out["ai_rank"] - out["pop_rank"]
    second = out.groupby(RACE_COL)["ai_score"].transform(
        lambda s: s.sort_values(ascending=False).iloc[1] if len(s) > 1 else np.nan
    )
    out["ai_score_gap_to_second"] = (out["ai_score"] - second).where(out["ai_rank"].eq(1), 0.0).fillna(0.0)

    odds = out["odds_decimal"].replace(0, np.nan)
    implied = 1.0 / odds
    out["_implied_share"] = implied / implied.groupby(out[RACE_COL]).transform("sum")
    out["market_top3_share"] = out.groupby(RACE_COL)["_implied_share"].transform(
        lambda s: float(s.sort_values(ascending=False).head(3).sum())
    )
    out["favorite_implied_share"] = out.groupby(RACE_COL)["_implied_share"].transform("max")
    out["odds_entropy"] = out.groupby(RACE_COL)["_implied_share"].transform(
        lambda s: float(-(s.dropna() * np.log(s.dropna())).sum())
    )

    age = _num(out.get(AGE_COL), out.index)
    distance = _num(out.get(DISTANCE_COL), out.index)
    surface = out.get(SURFACE_COL, pd.Series("", index=out.index)).astype("string")
    going = out.get(GOING_COL, pd.Series("", index=out.index)).astype("string")
    class_name = out.get(CLASS_COL, pd.Series("", index=out.index)).astype("string")
    out["is_young"] = age.le(3)
    out["is_age2"] = age.eq(2)
    out["is_turf"] = surface.str.contains("\u829d", na=False)
    out["is_dirt"] = surface.str.contains("\u30c0", na=False)
    out["is_soft_or_heavy"] = going.str.contains("\u91cd|\u4e0d|\u7a0d", regex=True, na=False)
    out["is_sprint"] = distance.le(1400)
    out["is_mile"] = distance.between(1500, 1800)
    out["is_middle"] = distance.between(1900, 2400)
    out["is_long"] = distance.ge(2500)
    out["is_newcomer"] = class_name.str.contains("\u65b0\u99ac", na=False)
    out["is_maiden"] = class_name.str.contains("\u672a\u52dd\u5229", na=False)
    out["is_open_plus"] = class_name.str.contains("\u30aa\u30fc\u30d7\u30f3|OP|L|G", regex=True, na=False)
    return out


def _split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = _num(frame[DATE_COL]).dropna().sort_values().unique()
    cutoff = dates[len(dates) // 2]
    date_num = _num(frame[DATE_COL])
    return frame[date_num <= cutoff].copy(), frame[date_num > cutoff].copy()


def _metrics(frame: pd.DataFrame, label: str) -> dict[str, object]:
    rows = len(frame)
    if rows == 0:
        return {"policy": label, "bets": 0, "races": 0}
    win_pay = _num(frame.get(WIN_PAY_COL), frame.index, 0.0).fillna(0.0).where(frame["target_win"].eq(1), 0.0)
    place_pay = _num(frame.get(PLACE_PAY_COL), frame.index, 0.0).fillna(0.0).where(frame["target_top3"].eq(1), 0.0)
    return {
        "policy": label,
        "bets": int(rows),
        "races": int(frame[RACE_COL].nunique()),
        "win_rate": float(frame["target_win"].mean()),
        "top3_rate": float(frame["target_top3"].mean()),
        "win_roi": float(win_pay.sum() / (rows * 100.0)),
        "place_roi": float(place_pay.sum() / (rows * 100.0)),
        "avg_popularity": float(frame["popularity_num"].mean()),
        "avg_odds": float(frame["odds_decimal"].mean()),
        "avg_ai_rank": float(frame["ai_rank"].mean()),
    }


def _trust_library(discovery: pd.DataFrame) -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    q = lambda col, pct: _q(discovery, col, pct)
    lib: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
        "ai_top1": lambda x: x["ai_rank"].eq(1),
        "ai_top2": lambda x: x["ai_rank"].le(2),
        "favorite": lambda x: x["popularity_num"].eq(1),
        "pop_le2": lambda x: x["popularity_num"].le(2),
        "pop_le3": lambda x: x["popularity_num"].le(3),
        "fav_odds_le15": lambda x: x["popularity_num"].eq(1) & x["odds_decimal"].le(1.5),
        "fav_odds_le20": lambda x: x["popularity_num"].eq(1) & x["odds_decimal"].le(2.0),
        "fav_odds_le25": lambda x: x["popularity_num"].eq(1) & x["odds_decimal"].le(2.5),
        "market_top3_hi": lambda x: x["market_top3_share"].ge(q("market_top3_share", 0.70)),
        "market_top3_mid_hi": lambda x: x["market_top3_share"].ge(q("market_top3_share", 0.60)),
        "favorite_share_hi": lambda x: x["favorite_implied_share"].ge(q("favorite_implied_share", 0.70)),
        "ai_gap005": lambda x: x["ai_score_gap_to_second"].ge(0.05),
        "ai_market_agree": lambda x: x["ai_rank"].eq(1) & x["popularity_num"].eq(1),
        "not_newcomer_maiden": lambda x: ~(x["is_newcomer"] | x["is_maiden"]),
        "not_soft_heavy": lambda x: ~x["is_soft_or_heavy"],
        "turf": lambda x: x["is_turf"],
        "dirt": lambda x: x["is_dirt"],
        "sprint": lambda x: x["is_sprint"],
        "mile": lambda x: x["is_mile"],
        "middle": lambda x: x["is_middle"],
        "young": lambda x: x["is_young"],
    }
    feature_thresholds = {
        "jockey_top3_hi": "jockey_top3_rate",
        "trainer_top3_hi": "trainer_top3_rate",
        "jockey_trainer_pair_hi": "jockey_trainer_pair_top3_rate",
        "rotation_fit_hi": "rotation_fit_score",
        "pace_fit_hi": "pace_fit_score",
        "lap_fit_hi": "lap_aptitude_fit_score",
        "lap_reliable_hi": "lap_aptitude_reliability_score",
        "blood_fit_hi": "bloodline_high_confidence_fit_score",
        "blood_lift_hi": "bloodline_lift_fit_score",
        "bias_fit_hi": "bias_adjusted_recent_score",
        "draw_pace_hi": "draw_pace_fit_score",
        "member_level_hi": "race_member_level_rank_score",
        "confirmed_member_hi": "confirmed_member_level_adjusted_score",
        "workout_hi": "workout_knowledge_grade_score",
        "body_maturity_hi": "body_young_maturity_score",
        "same_day_bias_fit_hi": "same_day_bias_fit_score",
        "retro_resistant_hi": "past3_retro_bias_resistant_score",
    }
    for name, col in feature_thresholds.items():
        if col in discovery.columns:
            threshold = q(col, 0.65)
            lib[name] = lambda x, c=col, t=threshold: _num(x[c]).fillna(-np.inf).ge(t)
    return lib


def _alert_library(discovery: pd.DataFrame) -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    q = lambda col, pct: _q(discovery, col, pct)
    lib: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
        "fav_odds_gt25": lambda x: x["popularity_num"].eq(1) & x["odds_decimal"].gt(2.5),
        "fav_odds_gt30": lambda x: x["popularity_num"].eq(1) & x["odds_decimal"].gt(3.0),
        "ai_disagrees_fav": lambda x: x["popularity_num"].eq(1) & x["ai_rank"].gt(1),
        "market_top3_low": lambda x: x["market_top3_share"].le(q("market_top3_share", 0.30)),
        "odds_entropy_hi": lambda x: x["odds_entropy"].ge(q("odds_entropy", 0.70)),
        "newcomer_or_maiden": lambda x: x["is_newcomer"] | x["is_maiden"],
        "soft_or_heavy": lambda x: x["is_soft_or_heavy"],
    }
    low_features = {
        "fav_pace_fit_low": "pace_fit_score",
        "fav_lap_reliable_low": "lap_aptitude_reliability_score",
        "fav_blood_fit_low": "bloodline_high_confidence_fit_score",
        "fav_bias_fit_low": "bias_adjusted_recent_score",
        "fav_draw_pace_low": "draw_pace_fit_score",
        "fav_jockey_top3_low": "jockey_top3_rate",
        "fav_trainer_top3_low": "trainer_top3_rate",
        "fav_rotation_fit_low": "rotation_fit_score",
    }
    for name, col in low_features.items():
        if col in discovery.columns:
            threshold = q(col, 0.35)
            lib[name] = lambda x, c=col, t=threshold: _num(x[c]).fillna(np.inf).le(t)
    high_risk = {
        "pace_collapse_hi": "race_pace_collapse_risk",
        "early_pressure_hi": "race_early_pressure_score",
        "depth_hi": "race_member_depth_score",
        "same_day_volatility_hi": "same_day_bias_volatility",
    }
    for name, col in high_risk.items():
        if col in discovery.columns:
            threshold = q(col, 0.70)
            lib[name] = lambda x, c=col, t=threshold: _num(x[c]).fillna(-np.inf).ge(t)
    return lib


def _precompute(frame: pd.DataFrame, lib: dict[str, Callable[[pd.DataFrame], pd.Series]]) -> dict[str, pd.Series]:
    return {name: func(frame).fillna(False).astype(bool) for name, func in lib.items()}


def _mask(masks: dict[str, pd.Series], combo: tuple[str, ...], index: pd.Index) -> pd.Series:
    out = pd.Series(True, index=index)
    for name in combo:
        out &= masks[name]
    return out


def _search(
    discovery: pd.DataFrame,
    validation: pd.DataFrame,
    lib: dict[str, Callable[[pd.DataFrame], pd.Series]],
    *,
    mode: str,
    min_discovery: int,
    min_validation: int,
    max_combo_size: int,
) -> pd.DataFrame:
    discovery_masks = _precompute(discovery, lib)
    validation_masks = _precompute(validation, lib)
    rows: list[dict[str, object]] = []
    names = list(lib)
    for size in range(1, max_combo_size + 1):
        for combo in combinations(names, size):
            if mode == "trust" and not any(name in combo for name in ["ai_top1", "favorite", "ai_market_agree", "fav_odds_le20", "fav_odds_le25"]):
                continue
            if mode == "alert" and not any(name in combo for name in ["fav_odds_gt25", "fav_odds_gt30", "ai_disagrees_fav"]):
                continue
            label = "&".join(combo)
            dpart = discovery[_mask(discovery_masks, combo, discovery.index)]
            vpart = validation[_mask(validation_masks, combo, validation.index)]
            if len(dpart) < min_discovery or len(vpart) < min_validation:
                continue
            d = _metrics(dpart, label)
            v = _metrics(vpart, label)
            rows.append(
                {
                    "policy": label,
                    **{f"discovery_{k}": value for k, value in d.items() if k != "policy"},
                    **{f"validation_{k}": value for k, value in v.items() if k != "policy"},
                    "min_top3_rate": min(float(d["top3_rate"]), float(v["top3_rate"])),
                    "mean_top3_rate": (float(d["top3_rate"]) + float(v["top3_rate"])) / 2.0,
                    "min_win_rate": min(float(d["win_rate"]), float(v["win_rate"])),
                    "mean_win_rate": (float(d["win_rate"]) + float(v["win_rate"])) / 2.0,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    if mode == "trust":
        return result.sort_values(
            ["min_top3_rate", "validation_top3_rate", "validation_bets"],
            ascending=[False, False, False],
        )
    return result.sort_values(
        ["validation_top3_rate", "min_top3_rate", "validation_bets"],
        ascending=[True, True, False],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-csv",
        default="data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/body_weight_backfilled/test_features_with_same_day_bias_v3_retro_body_context.csv",
    )
    parser.add_argument("--model", default="models/body_context_same_day_bias_v3_retro/baseline_ranker.pkl")
    parser.add_argument("--output-dir", default="outputs/analysis/trust_alert_policy")
    parser.add_argument("--max-combo-size", type=int, default=4)
    args = parser.parse_args()

    scored = _score(pd.read_csv(args.test_csv, low_memory=False), Path(args.model))
    discovery, validation = _split(scored)
    trust = _search(
        discovery,
        validation,
        _trust_library(discovery),
        mode="trust",
        min_discovery=120,
        min_validation=80,
        max_combo_size=args.max_combo_size,
    )
    alert = _search(
        discovery,
        validation,
        _alert_library(discovery),
        mode="alert",
        min_discovery=120,
        min_validation=80,
        max_combo_size=args.max_combo_size,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trust.to_csv(output_dir / "trust_policy_candidates.csv", index=False, encoding="utf-8-sig")
    alert.to_csv(output_dir / "alert_policy_candidates.csv", index=False, encoding="utf-8-sig")
    payload = {
        "output_dir": str(output_dir),
        "trust_top": trust.head(30).to_dict(orient="records") if not trust.empty else [],
        "alert_top": alert.head(30).to_dict(orient="records") if not alert.empty else [],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Trust candidates")
    print(trust.head(20).to_string(index=False) if not trust.empty else "none")
    print("\nAlert candidates")
    print(alert.head(20).to_string(index=False) if not alert.empty else "none")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
