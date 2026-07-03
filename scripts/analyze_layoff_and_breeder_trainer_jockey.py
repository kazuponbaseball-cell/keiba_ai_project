from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


RACE_COL = "\u30ec\u30fc\u30b9ID(\u65b0/\u99ac\u756a\u7121)"
HORSE_COL = "\u8840\u7d71\u767b\u9332\u756a\u53f7"
DATE_COL = "\u65e5\u4ed8"
TRAINER_COL = "\u8abf\u6559\u5e2b\u30b3\u30fc\u30c9"
JOCKEY_COL = "\u9a0e\u624b\u30b3\u30fc\u30c9"
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


def _resolve_raw_csv(raw_csv: str) -> Path:
    path = Path(raw_csv)
    if path.exists():
        return path
    matches = list(Path("date/raw").glob("*.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(raw_csv)


def _metrics(frame: pd.DataFrame, label: str) -> dict[str, object]:
    rows = len(frame)
    if rows == 0:
        return {"segment": label, "bets": 0, "races": 0}
    win_pay = _num(frame.get(WIN_PAY_COL), frame.index, 0.0).fillna(0.0).where(frame["target_win"].eq(1), 0.0)
    place_pay = _num(frame.get(PLACE_PAY_COL), frame.index, 0.0).fillna(0.0).where(frame["target_top3"].eq(1), 0.0)
    return {
        "segment": label,
        "bets": int(rows),
        "races": int(frame[RACE_COL].nunique()),
        "win_rate": float(frame["target_win"].mean()),
        "top3_rate": float(frame["target_top3"].mean()),
        "win_roi": float(win_pay.sum() / (rows * 100.0)),
        "place_roi": float(place_pay.sum() / (rows * 100.0)),
        "avg_popularity": float(_num(frame.get(POPULARITY_COL), frame.index).mean()),
        "avg_odds": float(_num(frame.get(ODDS_COL), frame.index).mean()),
        "avg_ai_rank": float(frame["ai_rank"].mean()) if "ai_rank" in frame.columns else None,
    }


def _score(frame: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    with model_path.open("rb") as f:
        model = pickle.load(f)
    out = frame.copy()
    out["ai_score"] = model.predict(out)
    out["ai_rank"] = out.groupby(RACE_COL)["ai_score"].rank(ascending=False, method="first").astype(int)
    out["popularity_num"] = _num(out.get(POPULARITY_COL), out.index)
    return out


def _load_raw_master(raw_csv: Path) -> pd.DataFrame:
    cols = [
        RACE_COL,
        HORSE_COL,
        "\u99ac\u540d",
        "\u9a0e\u624b",
        "\u8abf\u6559\u5e2b",
        "\u7a2e\u7261\u99ac",
        "\u6bcd\u7236\u99ac",
    ]
    return pd.read_csv(raw_csv, encoding="cp932", usecols=lambda c: c in cols, low_memory=False).drop_duplicates(
        [RACE_COL, HORSE_COL]
    )


def _add_previous_combo_stats(frame: pd.DataFrame, keys: list[str], prefix: str) -> pd.DataFrame:
    group = frame.groupby(keys, dropna=False, sort=False)
    starts = group.cumcount()
    out = pd.DataFrame(index=frame.index)
    out[f"{prefix}_starts"] = starts.astype(float)
    out[f"{prefix}_win_rate"] = (group["target_win"].cumsum() - frame["target_win"]) / starts.replace(0, np.nan)
    out[f"{prefix}_top3_rate"] = (group["target_top3"].cumsum() - frame["target_top3"]) / starts.replace(0, np.nan)
    pop_out = frame["rank_num"].notna() & frame["popularity_num"].notna() & frame["rank_num"].lt(frame["popularity_num"])
    out[f"{prefix}_popularity_outperform_rate"] = (group["_pop_out"].cumsum() - frame["_pop_out"]) / starts.replace(0, np.nan)
    return out


def _prepare_combo_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    train["_split"] = "train"
    test["_split"] = "test"
    frame = pd.concat([train, test], ignore_index=True, sort=False)
    frame["_date_num"] = _num(frame.get(DATE_COL), frame.index, 0).fillna(0)
    frame["_race_num"] = _num(frame.get(RACE_COL), frame.index, 0).fillna(0)
    frame["_orig_order"] = np.arange(len(frame))
    frame = frame.sort_values(["_date_num", "_race_num", "_orig_order"], kind="mergesort").reset_index(drop=True)
    frame["target_win"] = _num(frame["target_win"], frame.index, 0).fillna(0.0)
    frame["target_top3"] = _num(frame["target_top3"], frame.index, 0).fillna(0.0)
    frame["rank_num"] = _num(frame.get("\u78ba\u5b9a\u7740\u9806"), frame.index)
    frame["popularity_num"] = _num(frame.get(POPULARITY_COL), frame.index)
    frame["_pop_out"] = (frame["rank_num"].notna() & frame["popularity_num"].notna() & frame["rank_num"].lt(frame["popularity_num"])).astype(float)

    key_sets = [
        (["breeder_group_for_model", TRAINER_COL, JOCKEY_COL], "breedergrp_trainer_jockey"),
        (["breeder_group_for_model", TRAINER_COL], "breedergrp_trainer"),
        (["breeder_group_for_model", JOCKEY_COL], "breedergrp_jockey"),
        (["breeder_name", TRAINER_COL, JOCKEY_COL], "breeder_trainer_jockey"),
    ]
    for keys, prefix in key_sets:
        if all(k in frame.columns for k in keys):
            stats = _add_previous_combo_stats(frame, keys, prefix)
            for col in stats.columns:
                frame[col] = stats[col]

    frame = frame.sort_values("_orig_order", kind="mergesort")
    train_out = frame[frame["_split"].eq("train")].drop(columns=[c for c in frame.columns if c.startswith("_")])
    test_out = frame[frame["_split"].eq("test")].drop(columns=[c for c in frame.columns if c.startswith("_")])
    return train_out.reset_index(drop=True), test_out.reset_index(drop=True)


def _layoff_summary(frame: pd.DataFrame, group_col: str, min_count: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group, part in frame.groupby(group_col, dropna=False):
        fresh = part[_num(part.get("rotation_fresh_start_flag"), part.index, 0).fillna(0).ge(1)]
        second = part[_num(part.get("rotation_second_after_layoff_flag"), part.index, 0).fillna(0).ge(1)]
        third = part[_num(part.get("rotation_third_after_layoff_flag"), part.index, 0).fillna(0).ge(1)]
        if len(fresh) < min_count or len(second) < min_count:
            continue
        fresh_top3 = float(fresh["target_top3"].mean())
        second_top3 = float(second["target_top3"].mean())
        third_top3 = float(third["target_top3"].mean()) if len(third) >= min_count else np.nan
        rows.append(
            {
                "group_col": group_col,
                "group": group,
                "fresh_bets": int(len(fresh)),
                "fresh_win_rate": float(fresh["target_win"].mean()),
                "fresh_top3_rate": fresh_top3,
                "second_bets": int(len(second)),
                "second_win_rate": float(second["target_win"].mean()),
                "second_top3_rate": second_top3,
                "third_bets": int(len(third)),
                "third_top3_rate": third_top3,
                "second_lift_vs_fresh": second_top3 - fresh_top3,
                "third_lift_vs_fresh": third_top3 - fresh_top3 if np.isfinite(third_top3) else np.nan,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["second_lift_vs_fresh", "second_bets"], ascending=[False, False])


def _combo_segment_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    candidates = {
        "btj_group_top3_hi": ("breedergrp_trainer_jockey_starts", "breedergrp_trainer_jockey_top3_rate", 10),
        "bt_group_top3_hi": ("breedergrp_trainer_starts", "breedergrp_trainer_top3_rate", 20),
        "bj_group_top3_hi": ("breedergrp_jockey_starts", "breedergrp_jockey_top3_rate", 20),
        "btj_name_top3_hi": ("breeder_trainer_jockey_starts", "breeder_trainer_jockey_top3_rate", 5),
    }
    for label, (starts_col, rate_col, min_starts) in candidates.items():
        if starts_col not in frame.columns or rate_col not in frame.columns:
            continue
        rate = _num(frame[rate_col], frame.index)
        starts = _num(frame[starts_col], frame.index, 0).fillna(0)
        for q in [0.60, 0.70, 0.80]:
            threshold = rate[starts.ge(min_starts)].quantile(q)
            mask = starts.ge(min_starts) & rate.ge(threshold)
            for rank_label, rank_mask in {
                "all": pd.Series(True, index=frame.index),
                "ai_top1": frame["ai_rank"].eq(1),
                "ai_top3": frame["ai_rank"].le(3),
                "favorite": frame["popularity_num"].eq(1),
            }.items():
                part = frame[mask & rank_mask]
                if len(part) < 80:
                    continue
                row = _metrics(part, f"{label}_q{int(q*100)}_{rank_label}")
                row["starts_col"] = starts_col
                row["rate_col"] = rate_col
                row["min_starts"] = min_starts
                row["threshold"] = float(threshold)
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["top3_rate", "win_rate", "bets"], ascending=[False, False, False])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-csv",
        default="data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/body_weight_backfilled/owner_breeder_enriched/train_features_with_same_day_bias_v3_retro_body_breeder.csv",
    )
    parser.add_argument(
        "--test-csv",
        default="data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/body_weight_backfilled/owner_breeder_enriched/test_features_with_same_day_bias_v3_retro_body_breeder.csv",
    )
    parser.add_argument("--model", default="models/body_context_same_day_bias_v3_retro/baseline_ranker.pkl")
    parser.add_argument("--raw-csv", default="date/raw/全競走馬成績.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/layoff_breeder_trainer_jockey")
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv, low_memory=False)
    test = pd.read_csv(args.test_csv, low_memory=False)
    raw = _load_raw_master(_resolve_raw_csv(args.raw_csv))
    train = train.merge(raw, on=[RACE_COL, HORSE_COL], how="left")
    test = test.merge(raw, on=[RACE_COL, HORSE_COL], how="left")
    train_combo, test_combo = _prepare_combo_features(train, test)
    scored = _score(test_combo, Path(args.model))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_dir / "test_scored_with_combo_features.csv", index=False, encoding="utf-8-sig")

    layoff_tables = []
    for col, min_count in [
        ("\u8abf\u6559\u5e2b", 30),
        ("\u7a2e\u7261\u99ac", 40),
        ("\u6bcd\u7236\u99ac", 40),
        ("breeder_group_for_model", 40),
        ("\u8abf\u6559\u5e2b\u30b3\u30fc\u30c9", 30),
    ]:
        if col in scored.columns:
            table = _layoff_summary(scored, col, min_count)
            if not table.empty:
                layoff_tables.append(table)
                table.to_csv(output_dir / f"layoff_summary_{col}.csv", index=False, encoding="utf-8-sig")
    layoff = pd.concat(layoff_tables, ignore_index=True) if layoff_tables else pd.DataFrame()
    layoff.to_csv(output_dir / "layoff_summary_all.csv", index=False, encoding="utf-8-sig")

    base_rows = []
    for label, mask in {
        "fresh_all": _num(scored.get("rotation_fresh_start_flag"), scored.index, 0).ge(1),
        "second_after_layoff_all": _num(scored.get("rotation_second_after_layoff_flag"), scored.index, 0).ge(1),
        "third_after_layoff_all": _num(scored.get("rotation_third_after_layoff_flag"), scored.index, 0).ge(1),
        "fresh_ai_top1": _num(scored.get("rotation_fresh_start_flag"), scored.index, 0).ge(1) & scored["ai_rank"].eq(1),
        "second_ai_top1": _num(scored.get("rotation_second_after_layoff_flag"), scored.index, 0).ge(1) & scored["ai_rank"].eq(1),
    }.items():
        base_rows.append(_metrics(scored[mask.fillna(False)], label))
    base_summary = pd.DataFrame(base_rows)
    base_summary.to_csv(output_dir / "layoff_base_segments.csv", index=False, encoding="utf-8-sig")

    combo_summary = _combo_segment_summary(scored)
    combo_summary.to_csv(output_dir / "breeder_trainer_jockey_combo_segments.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(output_dir),
        "layoff_base": base_summary.to_dict(orient="records"),
        "layoff_top": layoff.head(30).to_dict(orient="records") if not layoff.empty else [],
        "combo_top": combo_summary.head(30).to_dict(orient="records") if not combo_summary.empty else [],
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Layoff base")
    print(base_summary.to_string(index=False))
    print("\nLayoff top")
    print(layoff.head(20).to_string(index=False) if not layoff.empty else "none")
    print("\nBreeder x trainer x jockey combos")
    print(combo_summary.head(20).to_string(index=False) if not combo_summary.empty else "none")
    print(json.dumps({"output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
