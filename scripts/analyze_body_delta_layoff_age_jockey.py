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
AGE_COL = "\u5e74\u9f62"
JOCKEY_COL = "\u9a0e\u624b"
JOCKEY_CODE_COL = "\u9a0e\u624b\u30b3\u30fc\u30c9"
POPULARITY_COL = "\u4eba\u6c17"
ODDS_COL = "\u5358\u52dd\u30aa\u30c3\u30ba"
WIN_PAY_COL = "\u5358\u52dd\u914d\u5f53"
PLACE_PAY_COL = "\u8907\u52dd\u914d\u5f53"
BODY_WEIGHT_COL = "\u99ac\u4f53\u91cd"
BODY_DELTA_COL = "\u99ac\u4f53\u91cd\u5897\u6e1b"


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


def _score(frame: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    with model_path.open("rb") as f:
        model = pickle.load(f)
    out = frame.copy()
    out["ai_score"] = model.predict(out)
    out["ai_rank"] = out.groupby(RACE_COL)["ai_score"].rank(ascending=False, method="first").astype(int)
    out["popularity_num"] = _num(out.get(POPULARITY_COL), out.index)
    out["odds_decimal"] = _num(out.get(ODDS_COL), out.index)
    return out


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
        "avg_popularity": float(frame["popularity_num"].mean()),
        "avg_odds": float(frame["odds_decimal"].mean()),
        "avg_ai_rank": float(frame["ai_rank"].mean()) if "ai_rank" in frame.columns else None,
        "avg_body_delta": float(_num(frame.get(BODY_DELTA_COL), frame.index).mean()) if BODY_DELTA_COL in frame else None,
    }


def _load_body_jockey_raw(raw_csv: Path) -> pd.DataFrame:
    cols = [RACE_COL, HORSE_COL, BODY_WEIGHT_COL, BODY_DELTA_COL, JOCKEY_COL, JOCKEY_CODE_COL]
    return pd.read_csv(raw_csv, encoding="cp932", usecols=lambda c: c in cols, low_memory=False).drop_duplicates(
        [RACE_COL, HORSE_COL]
    )


def _add_body_bins(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    delta = _num(out.get(BODY_DELTA_COL), out.index)
    age = _num(out.get(AGE_COL), out.index)
    out["body_delta_bucket"] = pd.cut(
        delta,
        bins=[-999, -16, -10, -4, 4, 10, 16, 999],
        labels=["loss_17plus", "loss_11_16", "loss_5_10", "flat_pm4", "gain_5_10", "gain_11_16", "gain_17plus"],
    )
    out["age_bucket"] = np.select(
        [age.eq(2), age.eq(3), age.between(4, 5), age.ge(6)],
        ["age2", "age3", "age4_5", "age6plus"],
        default="unknown",
    )
    out["fresh_flag"] = _num(out.get("rotation_fresh_start_flag"), out.index, 0).fillna(0).ge(1)
    out["second_after_layoff_flag2"] = _num(out.get("rotation_second_after_layoff_flag"), out.index, 0).fillna(0).ge(1)
    out["layoff_9w_plus_flag"] = (
        _num(out.get("rotation_layoff_9_16w_flag"), out.index, 0).fillna(0).ge(1)
        | _num(out.get("rotation_long_layoff_17w_plus_flag"), out.index, 0).fillna(0).ge(1)
    )
    return out


def _segment_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_cols, prefix, min_rows in [
        (["body_delta_bucket"], "delta", 120),
        (["age_bucket", "body_delta_bucket"], "age_delta", 80),
        (["fresh_flag", "body_delta_bucket"], "fresh_delta", 40),
        (["age_bucket", "fresh_flag", "body_delta_bucket"], "age_fresh_delta", 30),
        (["second_after_layoff_flag2", "body_delta_bucket"], "second_delta", 20),
    ]:
        grouped = frame.groupby(group_cols, dropna=False, observed=False)
        for key, part in grouped:
            if len(part) < min_rows:
                continue
            key_tuple = key if isinstance(key, tuple) else (key,)
            label_bits = [prefix, *[str(x) for x in key_tuple]]
            rows.append(_metrics(part, ":".join(label_bits)))
            ai1 = part[part["ai_rank"].eq(1)]
            if len(ai1) >= max(15, min_rows // 3):
                rows.append(_metrics(ai1, ":".join([*label_bits, "ai_top1"])))
            fav = part[part["popularity_num"].eq(1)]
            if len(fav) >= max(15, min_rows // 3):
                rows.append(_metrics(fav, ":".join([*label_bits, "favorite"])))
    return pd.DataFrame(rows).sort_values(["top3_rate", "win_rate", "bets"], ascending=[False, False, False])


def _jockey_profiles(frame: pd.DataFrame, min_rows: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_code = frame.groupby([JOCKEY_CODE_COL, JOCKEY_COL], dropna=False)
    rows = []
    for (code, name), part in by_code:
        if len(part) < min_rows:
            continue
        fav = part[part["popularity_num"].le(3)]
        longshot = part[part["popularity_num"].ge(5)]
        ai1 = part[part["ai_rank"].eq(1)]
        if len(fav) < 30 or len(longshot) < 30:
            continue
        row = {
            "jockey_code": code,
            "jockey": name,
            "bets": int(len(part)),
            "win_rate": float(part["target_win"].mean()),
            "top3_rate": float(part["target_top3"].mean()),
            "favorite_bets": int(len(fav)),
            "favorite_win_rate": float(fav["target_win"].mean()),
            "favorite_top3_rate": float(fav["target_top3"].mean()),
            "favorite_win_roi": _metrics(fav, "tmp")["win_roi"],
            "favorite_place_roi": _metrics(fav, "tmp")["place_roi"],
            "longshot_bets": int(len(longshot)),
            "longshot_win_rate": float(longshot["target_win"].mean()),
            "longshot_top3_rate": float(longshot["target_top3"].mean()),
            "longshot_win_roi": _metrics(longshot, "tmp")["win_roi"],
            "longshot_place_roi": _metrics(longshot, "tmp")["place_roi"],
            "longshot_pop_outperform_rate": float(
                (_num(longshot.get("\u78ba\u5b9a\u7740\u9806"), longshot.index).lt(longshot["popularity_num"])).mean()
            ),
            "ai1_bets": int(len(ai1)),
            "ai1_win_rate": float(ai1["target_win"].mean()) if len(ai1) else np.nan,
            "ai1_top3_rate": float(ai1["target_top3"].mean()) if len(ai1) else np.nan,
        }
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return table, table, table
    axis = table[table["favorite_bets"].ge(50)].sort_values(
        ["favorite_top3_rate", "favorite_win_rate", "favorite_bets"], ascending=[False, False, False]
    )
    value = table[table["longshot_bets"].ge(50)].sort_values(
        ["longshot_place_roi", "longshot_pop_outperform_rate", "longshot_bets"], ascending=[False, False, False]
    )
    balanced = table.sort_values(["ai1_top3_rate", "favorite_top3_rate", "bets"], ascending=[False, False, False])
    return table, axis, value, balanced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-csv",
        default="data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/body_weight_backfilled/test_features_with_same_day_bias_v3_retro_body_context.csv",
    )
    parser.add_argument("--model", default="models/body_context_same_day_bias_v3_retro/baseline_ranker.pkl")
    parser.add_argument("--raw-csv", default="date/raw/全競走馬成績.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/body_delta_layoff_age_jockey")
    args = parser.parse_args()

    base = pd.read_csv(args.test_csv, low_memory=False)
    raw = _load_body_jockey_raw(_resolve_raw_csv(args.raw_csv))
    frame = base.merge(raw, on=[RACE_COL, HORSE_COL], how="left", suffixes=("", "_raw"))
    frame = _add_body_bins(_score(frame, Path(args.model)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    segment = _segment_table(frame)
    segment.to_csv(output_dir / "body_delta_layoff_age_segments.csv", index=False, encoding="utf-8-sig")

    all_jockeys, axis, value, balanced = _jockey_profiles(frame, min_rows=80)
    all_jockeys.to_csv(output_dir / "jockey_profile_all.csv", index=False, encoding="utf-8-sig")
    axis.head(80).to_csv(output_dir / "jockey_axis_candidates.csv", index=False, encoding="utf-8-sig")
    value.head(80).to_csv(output_dir / "jockey_value_candidates.csv", index=False, encoding="utf-8-sig")
    balanced.head(80).to_csv(output_dir / "jockey_ai1_candidates.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(output_dir),
        "body_delta_coverage": float(frame[BODY_DELTA_COL].notna().mean()),
        "top_body_segments": segment.head(30).to_dict(orient="records"),
        "axis_jockey_top": axis.head(20).to_dict(orient="records"),
        "value_jockey_top": value.head(20).to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Body delta / layoff / age")
    print(segment.head(30).to_string(index=False))
    print("\nAxis jockey candidates")
    print(axis.head(20).to_string(index=False) if not axis.empty else "none")
    print("\nValue jockey candidates")
    print(value.head(20).to_string(index=False) if not value.empty else "none")
    print(json.dumps({"output_dir": str(output_dir), "body_delta_coverage": payload["body_delta_coverage"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
