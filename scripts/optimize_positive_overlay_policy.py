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
AGE_COL = "\u5e74\u9f62"
SURFACE_COL = "\u829d\u30fb\u30c0"
CLASS_COL = "\u30af\u30e9\u30b9\u540d"
POPULARITY_COL = "\u4eba\u6c17"
ODDS_COL = "\u5358\u52dd\u30aa\u30c3\u30ba"
WIN_PAY_COL = "\u5358\u52dd\u914d\u5f53"
PLACE_PAY_COL = "\u8907\u52dd\u914d\u5f53"


def _num(series: pd.Series | None, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if series is None:
        if index is None:
            raise ValueError("index required when series is None")
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


def _q(frame: pd.DataFrame, column: str, q: float, default: float = 0.0) -> float:
    if column not in frame.columns:
        return default
    values = _num(frame[column]).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return default
    return float(values.quantile(q))


def _metrics(frame: pd.DataFrame, label: str) -> dict[str, object]:
    rows = len(frame)
    if rows == 0:
        return {"policy": label, "bets": 0, "races": 0}
    win_pay = _num(frame.get(WIN_PAY_COL), frame.index, 0.0).fillna(0.0).where(frame["target_win"].eq(1), 0.0)
    place_pay = _num(frame.get(PLACE_PAY_COL), frame.index, 0.0).fillna(0.0).where(frame["target_top3"].eq(1), 0.0)
    stake = rows * 100.0
    return {
        "policy": label,
        "bets": int(rows),
        "races": int(frame[RACE_COL].nunique()),
        "win_rate": float(frame["target_win"].mean()),
        "top3_rate": float(frame["target_top3"].mean()),
        "win_roi": float(win_pay.sum() / stake),
        "place_roi": float(place_pay.sum() / stake),
        "profit_win_flat100": float(win_pay.sum() - stake),
        "profit_place_flat100": float(place_pay.sum() - stake),
        "avg_popularity": float(_num(frame.get(POPULARITY_COL), frame.index).mean()),
        "avg_odds": float(_num(frame.get(ODDS_COL), frame.index).mean()),
        "avg_ai_rank": float(frame["ai_rank"].mean()),
    }


def _split_discovery_validation(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = _num(frame[DATE_COL]).dropna().sort_values().unique()
    cutoff = dates[len(dates) // 2]
    date_num = _num(frame[DATE_COL])
    return frame[date_num <= cutoff].copy(), frame[date_num > cutoff].copy()


def _add_context(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    age = _num(out.get(AGE_COL), out.index)
    surface = out.get(SURFACE_COL, pd.Series("", index=out.index)).astype("string")
    class_name = out.get(CLASS_COL, pd.Series("", index=out.index)).astype("string")
    out["is_turf"] = surface.str.contains("\u829d", na=False)
    out["is_dirt"] = surface.str.contains("\u30c0", na=False)
    out["is_age2"] = age.eq(2)
    out["is_age3"] = age.eq(3)
    out["is_young"] = age.le(3)
    out["is_newcomer"] = class_name.str.contains("\u65b0\u99ac", na=False)
    out["is_maiden"] = class_name.str.contains("\u672a\u52dd\u5229", na=False)
    out["is_1win"] = class_name.str.contains("1\u52dd|500\u4e07", regex=True, na=False)
    out["is_open_plus"] = class_name.str.contains("\u30aa\u30fc\u30d7\u30f3|OP|L|G", regex=True, na=False)
    return out


def _score(frame: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    with model_path.open("rb") as f:
        model = pickle.load(f)
    out = _add_context(frame)
    out["ai_score"] = model.predict(out)
    out["ai_rank"] = out.groupby(RACE_COL)["ai_score"].rank(ascending=False, method="first").astype(int)
    pop = _num(out.get(POPULARITY_COL), out.index)
    out["popularity_num"] = pop
    out["pop_rank"] = out.groupby(RACE_COL)["popularity_num"].rank(ascending=True, method="first")
    out["ai_pop_gap"] = out["ai_rank"] - out["pop_rank"]
    second = out.groupby(RACE_COL)["ai_score"].transform(lambda s: s.sort_values(ascending=False).iloc[1] if len(s) > 1 else np.nan)
    out["ai_score_gap_to_second"] = (out["ai_score"] - second).where(out["ai_rank"].eq(1), 0.0).fillna(0.0)
    return out


def _condition_library(discovery: pd.DataFrame) -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    q = lambda col, quant: _q(discovery, col, quant)
    lib: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
        "ai_top1": lambda x: x["ai_rank"].eq(1),
        "ai_top2": lambda x: x["ai_rank"].le(2),
        "ai_top3": lambda x: x["ai_rank"].le(3),
        "pop_le3": lambda x: x["popularity_num"].le(3),
        "pop_4_9": lambda x: x["popularity_num"].between(4, 9),
        "pop_ge5": lambda x: x["popularity_num"].ge(5),
        "odds_ge7": lambda x: _num(x.get(ODDS_COL), x.index).ge(7),
        "market_gap3": lambda x: x["ai_pop_gap"].le(-3),
        "top1_gap005": lambda x: x["ai_score_gap_to_second"].ge(0.05),
        "turf_young": lambda x: x["is_turf"] & x["is_young"],
        "turf_2yo": lambda x: x["is_turf"] & x["is_age2"],
        "newcomer_or_maiden": lambda x: x["is_newcomer"] | x["is_maiden"],
    }

    high_cols = {
        "body_young_maturity_hi": "body_young_maturity_score",
        "body_age2_big500": "body_age2_big500_flag",
        "body_age2_heavy_top3": "body_age2_race_heavy_top3_flag",
        "body_age3_heavy_top5": "body_age3_race_heavy_top5_flag",
        "blood_high_conf_hi": "bloodline_high_confidence_fit_score",
        "blood_lift_hi": "bloodline_lift_fit_score",
        "lap_fit_hi": "lap_aptitude_fit_score",
        "lap_reliability_hi": "lap_aptitude_reliability_score",
        "bias_adjusted_hi": "bias_adjusted_recent_score",
        "draw_pace_hi": "draw_pace_fit_score",
        "workout_knowledge_hi": "workout_knowledge_grade_score",
        "workout_load_hi": "workout_load_density_score",
        "same_day_bias_fit_hi": "same_day_bias_fit_score",
        "same_day_pop_fit_hi": "same_day_pop_adjusted_pace_fit_score",
        "retro_resistant_hi": "past3_retro_bias_resistant_score",
        "retro_adversity_hi": "past3_retro_bias_adversity_score",
        "owner_pop_outperform_hi": "owner_popularity_outperform_rate",
        "owner_context_hi": "owner_context_fit_score",
        "breeder_context_hi": "breeder_context_fit_score",
        "breeder_young_turf_hi": "breeder_young_turf_fit_score",
    }
    for name, col in high_cols.items():
        if col not in discovery.columns:
            continue
        if col.endswith("_flag"):
            lib[name] = lambda x, c=col: _num(x[c]).fillna(0).ge(1)
        else:
            threshold = q(col, 0.70)
            lib[name] = lambda x, c=col, t=threshold: _num(x[c]).fillna(-np.inf).ge(t)

    for flag in [
        "breeder_northern_turf_young_flag",
        "breeder_shadai_turf_young_flag",
        "breeder_shadai_group_flag",
        "breeder_northern_farm_flag",
        "breeder_shadai_farm_flag",
    ]:
        if flag in discovery.columns:
            lib[flag] = lambda x, c=flag: _num(x[c]).fillna(0).ge(1)

    return lib


def _precompute(frame: pd.DataFrame, lib: dict[str, Callable[[pd.DataFrame], pd.Series]]) -> dict[str, pd.Series]:
    return {name: func(frame).fillna(False).astype(bool) for name, func in lib.items()}


def _combo_mask(masks: dict[str, pd.Series], combo: tuple[str, ...], index: pd.Index) -> pd.Series:
    mask = pd.Series(True, index=index)
    for name in combo:
        mask &= masks[name]
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", default="outputs/analysis/optimized_positive_overlay")
    parser.add_argument("--min-discovery-bets", type=int, default=80)
    parser.add_argument("--min-validation-bets", type=int, default=50)
    parser.add_argument("--max-combo-size", type=int, default=4)
    args = parser.parse_args()

    frame = pd.read_csv(args.test_csv, low_memory=False)
    scored = _score(frame, Path(args.model))
    discovery, validation = _split_discovery_validation(scored)
    lib = _condition_library(discovery)
    discovery_masks = _precompute(discovery, lib)
    validation_masks = _precompute(validation, lib)

    rows: list[dict[str, object]] = []
    for size in range(1, args.max_combo_size + 1):
        for combo in combinations(lib.keys(), size):
            if not any(name.startswith("ai_top") for name in combo):
                continue
            label = "&".join(combo)
            dpart = discovery[_combo_mask(discovery_masks, combo, discovery.index)]
            if len(dpart) < args.min_discovery_bets:
                continue
            vpart = validation[_combo_mask(validation_masks, combo, validation.index)]
            if len(vpart) < args.min_validation_bets:
                continue
            d = _metrics(dpart, label)
            v = _metrics(vpart, label)
            rows.append(
                {
                    "policy": label,
                    **{f"discovery_{k}": value for k, value in d.items() if k != "policy"},
                    **{f"validation_{k}": value for k, value in v.items() if k != "policy"},
                    "min_win_roi": min(float(d["win_roi"]), float(v["win_roi"])),
                    "min_place_roi": min(float(d["place_roi"]), float(v["place_roi"])),
                    "mean_win_roi": (float(d["win_roi"]) + float(v["win_roi"])) / 2.0,
                    "mean_place_roi": (float(d["place_roi"]) + float(v["place_roi"])) / 2.0,
                }
            )

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(
            ["min_win_roi", "validation_win_roi", "validation_bets", "mean_place_roi"],
            ascending=[False, False, False, False],
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "optimized_overlay_policies.csv", index=False, encoding="utf-8-sig")
    top = summary.head(50).to_dict(orient="records") if not summary.empty else []
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "condition_count": len(lib),
                "policies_tested": int(len(summary)),
                "top_policies": top,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.head(30).to_string(index=False) if not summary.empty else "no policies")
    print(json.dumps({"output_dir": str(output_dir), "condition_count": len(lib), "policies_tested": int(len(summary))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
