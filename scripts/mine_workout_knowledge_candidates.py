from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.workout_knowledge import TRAINER_NAMES


DEFAULT_TRAIN = (
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus_rebuilt_20260623/"
    "train_features.csv"
)
DEFAULT_VALID = (
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus_rebuilt_20260623/"
    "test_features.csv"
)

SPECS = {
    "lap": ["_trainer_code", "_lap"],
    "course_lap": ["_trainer_code", "_course", "_lap"],
    "surface_lap": ["_trainer_code", "_surface", "_lap"],
    "pattern": ["_trainer_code", "_pattern"],
    "course_pattern": ["_trainer_code", "_course", "_pattern"],
    "surface_course_lap": ["_trainer_code", "_surface", "_course", "_lap"],
    "latest_total_z": ["_trainer_code", "_latest_total_z"],
    "latest_final1_z": ["_trainer_code", "_latest_final1_z"],
    "best_total_z": ["_trainer_code", "_best_total_z"],
    "best_final1_z": ["_trainer_code", "_best_final1_z"],
    "course_total_z": ["_trainer_code", "_course", "_latest_total_z"],
    "course_final1_z": ["_trainer_code", "_course", "_latest_final1_z"],
    "days_lap": ["_trainer_code", "_days_bucket", "_lap"],
    "count_lap": ["_trainer_code", "_count_bucket", "_lap"],
    "flag_a1": ["_trainer_code", "_flag_a1"],
    "flag_a2": ["_trainer_code", "_flag_a2"],
    "flag_a3": ["_trainer_code", "_flag_a3"],
    "flag_b1": ["_trainer_code", "_flag_b1"],
    "flag_b2": ["_trainer_code", "_flag_b2"],
    "flag_b3": ["_trainer_code", "_flag_b3"],
    "flag_fast_final": ["_trainer_code", "_flag_fast_final"],
    "flag_strong_finish": ["_trainer_code", "_flag_strong_finish"],
    "flag_partner_win": ["_trainer_code", "_flag_partner_win"],
}


def first_existing(df: pd.DataFrame, candidates: Iterable[str], *, required: bool = True) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"None of these columns exist: {list(candidates)}")
    return None


def num(values: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if values is None:
        return pd.Series(default, index=index, dtype="float64")
    if values.dtype == object or str(values.dtype).startswith("string"):
        values = (
            values.astype("string")
            .str.replace(",", "", regex=False)
            .str.replace("(", "", regex=False)
            .str.replace(")", "", regex=False)
        )
    return pd.to_numeric(values, errors="coerce").fillna(default)


def text(values: pd.Series | None, index: pd.Index, default: str = "unknown") -> pd.Series:
    if values is None:
        return pd.Series(default, index=index, dtype="string")
    out = values.astype("string").fillna(default)
    out = out.mask(out.str.len().fillna(0).eq(0), default)
    return out


def pay_yen(pay: pd.Series, odds: pd.Series, hit: pd.Series) -> pd.Series:
    parsed_pay = num(pay, pay.index, np.nan)
    parsed_odds = num(odds, odds.index, np.nan)
    adjusted = parsed_pay.copy()
    adjusted = adjusted.mask(adjusted.lt(50.0) & parsed_odds.notna(), parsed_odds * 100.0)
    adjusted = adjusted.mask(adjusted.isna() & parsed_odds.notna(), parsed_odds * 100.0)
    return adjusted.fillna(0.0).where(hit, 0.0)


def add_workout_rule_tokens(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    out["_course"] = text(out.get("workout_latest_course_bucket"), idx)
    out["_lap"] = text(out.get("workout_latest_lap_group"), idx)
    out["_pattern"] = text(out.get("workout_latest_pattern_bucket"), idx)
    out["_surface"] = text(out.get("芝・ダ"), idx)
    out["_class"] = text(out.get("クラス名"), idx)

    def bucket_z(col: str, prefix: str) -> pd.Series:
        z = num(out.get(col), idx, np.nan)
        return pd.Series(
            np.select(
                [
                    z.le(-1.00),
                    z.le(-0.50),
                    z.lt(0.50),
                    z.lt(1.00),
                    z.ge(1.00),
                ],
                [
                    f"{prefix}_very_fast",
                    f"{prefix}_fast",
                    f"{prefix}_normal",
                    f"{prefix}_slow",
                    f"{prefix}_very_slow",
                ],
                default=f"{prefix}_unknown",
            ),
            index=idx,
            dtype="string",
        )

    out["_latest_total_z"] = bucket_z("workout_latest_total_vs_trainer_z", "latest_total")
    out["_latest_final1_z"] = bucket_z("workout_latest_final1_vs_trainer_z", "latest_final1")
    out["_best_total_z"] = bucket_z("workout_best_total_vs_trainer_z", "best_total")
    out["_best_final1_z"] = bucket_z("workout_best_final1_vs_trainer_z", "best_final1")

    days = num(out.get("workout_latest_days_before_race"), idx, np.nan)
    out["_days_bucket"] = pd.Series(
        np.select(
            [
                days.le(3.0),
                days.le(5.0),
                days.le(8.0),
                days.gt(8.0),
            ],
            ["days_0_3", "days_4_5", "days_6_8", "days_9plus"],
            default="days_unknown",
        ),
        index=idx,
        dtype="string",
    )
    count = num(out.get("workout_count"), idx, 0.0)
    out["_count_bucket"] = pd.Series(
        np.select(
            [
                count.le(1.0),
                count.le(3.0),
                count.le(7.0),
                count.gt(7.0),
            ],
            ["count_0_1", "count_2_3", "count_4_7", "count_8plus"],
            default="count_unknown",
        ),
        index=idx,
        dtype="string",
    )
    for flag in ["a1", "a2", "a3", "b1", "b2", "b3", "fast_final", "strong_finish", "partner_win"]:
        source = {
            "a1": "workout_a1_flag",
            "a2": "workout_a2_flag",
            "a3": "workout_a3_flag",
            "b1": "workout_b1_flag",
            "b2": "workout_b2_flag",
            "b3": "workout_b3_flag",
            "fast_final": "workout_fast_final_flag",
            "strong_finish": "workout_strong_finish_flag",
            "partner_win": "workout_partner_win_flag",
        }[flag]
        out[f"_flag_{flag}"] = np.where(num(out.get(source), idx, 0.0).ge(1.0), f"{flag}_yes", f"{flag}_no")
    return out


def prepare(df: pd.DataFrame, *, split: str) -> pd.DataFrame:
    out = add_workout_rule_tokens(df)
    idx = out.index
    trainer_col = first_existing(out, ["調教師コード", "trainer_code"])
    rank_col = first_existing(out, ["確定着順", "finish_position", "rank"])
    race_col = first_existing(out, ["レースID(新/馬番無)", "race_id"], required=False)
    out["_trainer_code"] = num(out[trainer_col], idx, -1).astype("Int64")
    out["_registered_trainer"] = out["_trainer_code"].astype(int).isin({int(c) for c in TRAINER_NAMES})
    out["_rank"] = num(out[rank_col], idx, 999.0)
    out["_win"] = out["_rank"].eq(1.0)
    out["_top3"] = out["_rank"].between(1.0, 3.0)
    out["_odds"] = num(out.get("単勝オッズ"), idx, np.nan)
    out["_win_return_yen"] = pay_yen(out.get("単勝配当", pd.Series(np.nan, index=idx)), out["_odds"], out["_win"])
    out["_place_return_yen"] = pay_yen(out.get("複勝配当", pd.Series(np.nan, index=idx)), out["_odds"], out["_top3"])
    out["_target_score"] = num(out.get("target_score"), idx, np.nan)
    if out["_target_score"].isna().all():
        field = out.groupby(race_col)["_rank"].transform("max").replace(0, np.nan) if race_col else pd.Series(np.nan, index=idx)
        out["_target_score"] = ((field + 1.0 - out["_rank"]) / field).clip(0.0, 1.0)
    out["_popularity"] = num(out.get("人気"), idx, np.nan)
    out["_workout_count"] = num(out.get("workout_count"), idx, 0.0)
    out["_split"] = split
    return out


def metrics(df: pd.DataFrame, prefix: str) -> dict[str, float | int]:
    rows = len(df)
    if rows == 0:
        return {
            f"{prefix}_starts": 0,
            f"{prefix}_win_rate": 0.0,
            f"{prefix}_top3_rate": 0.0,
            f"{prefix}_avg_score": 0.0,
            f"{prefix}_avg_popularity": np.nan,
            f"{prefix}_avg_odds": np.nan,
            f"{prefix}_win_roi_pct": 0.0,
            f"{prefix}_place_roi_pct": 0.0,
        }
    stake = rows * 100.0
    return {
        f"{prefix}_starts": int(rows),
        f"{prefix}_win_rate": round(float(df["_win"].mean()), 4),
        f"{prefix}_top3_rate": round(float(df["_top3"].mean()), 4),
        f"{prefix}_avg_score": round(float(df["_target_score"].mean()), 4),
        f"{prefix}_avg_popularity": round(float(df["_popularity"].mean()), 2),
        f"{prefix}_avg_odds": round(float(df["_odds"].mean()), 2),
        f"{prefix}_win_roi_pct": round(float(df["_win_return_yen"].sum() / stake * 100.0), 1),
        f"{prefix}_place_roi_pct": round(float(df["_place_return_yen"].sum() / stake * 100.0), 1),
    }


def summarize_group(df: pd.DataFrame, group_cols: list[str], prefix: str) -> pd.DataFrame:
    rows = []
    for key, part in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(group_cols, key)}
        row.update(metrics(part, prefix))
        rows.append(row)
    return pd.DataFrame(rows)


def mine(train: pd.DataFrame, valid: pd.DataFrame, *, min_train: int, min_valid: int) -> pd.DataFrame:
    train = train[
        ~train["_registered_trainer"] & train["_trainer_code"].ge(0) & train["_workout_count"].gt(0.0)
    ].copy()
    valid = valid[
        ~valid["_registered_trainer"] & valid["_trainer_code"].ge(0) & valid["_workout_count"].gt(0.0)
    ].copy()
    trainer_train_base = summarize_group(train, ["_trainer_code"], "trainer_train")
    trainer_valid_base = summarize_group(valid, ["_trainer_code"], "trainer_valid")
    outputs = []
    for family, cols in SPECS.items():
        train_summary = summarize_group(train, cols, "train")
        valid_summary = summarize_group(valid, cols, "valid")
        merged = train_summary.merge(valid_summary, on=cols, how="inner")
        merged = merged.merge(trainer_train_base, on="_trainer_code", how="left")
        merged = merged.merge(trainer_valid_base, on="_trainer_code", how="left")
        merged["family"] = family
        merged["rule_key"] = merged[cols].astype("string").agg("|".join, axis=1)
        for col in ["_lap", "_course", "_surface", "_pattern", "_latest_total_z", "_latest_final1_z", "_best_total_z", "_best_final1_z", "_days_bucket", "_count_bucket", "_flag_a1", "_flag_a2", "_flag_a3", "_flag_b1", "_flag_b2", "_flag_b3", "_flag_fast_final", "_flag_strong_finish", "_flag_partner_win"]:
            if col not in merged.columns:
                merged[col] = pd.NA
        outputs.append(merged)
    if not outputs:
        return pd.DataFrame()
    out = pd.concat(outputs, ignore_index=True, sort=False)
    out = out[(out["train_starts"] >= min_train) & (out["valid_starts"] >= min_valid)].copy()
    out = out[
        ~out["rule_key"].astype("string").str.contains("unknown", case=False, na=False)
        & ~out["rule_key"].astype("string").str.endswith("_no", na=False)
    ].copy()
    out["train_top3_lift_vs_trainer"] = out["train_top3_rate"] - out["trainer_train_top3_rate"]
    out["valid_top3_lift_vs_trainer"] = out["valid_top3_rate"] - out["trainer_valid_top3_rate"]
    out["train_score_lift_vs_trainer"] = out["train_avg_score"] - out["trainer_train_avg_score"]
    out["valid_score_lift_vs_trainer"] = out["valid_avg_score"] - out["trainer_valid_avg_score"]
    out["train_win_roi_lift_vs_trainer"] = out["train_win_roi_pct"] - out["trainer_train_win_roi_pct"]
    out["valid_win_roi_lift_vs_trainer"] = out["valid_win_roi_pct"] - out["trainer_valid_win_roi_pct"]
    out["valid_edge_score"] = (
        np.log1p(out["valid_starts"]) * 0.35
        + out["valid_score_lift_vs_trainer"].fillna(0.0) * 18.0
        + out["valid_top3_lift_vs_trainer"].fillna(0.0) * 10.0
        + (out["valid_win_roi_pct"].fillna(0.0) - 100.0) / 80.0
        + (out["valid_place_roi_pct"].fillna(0.0) - 100.0) / 120.0
    )
    out["action"] = np.select(
        [
            (
                out["train_score_lift_vs_trainer"].ge(0.015)
                & out["valid_score_lift_vs_trainer"].ge(0.010)
                & out["valid_top3_lift_vs_trainer"].ge(0.015)
                & (out["valid_win_roi_pct"].ge(105.0) | out["valid_place_roi_pct"].ge(105.0))
            ),
            (
                out["train_score_lift_vs_trainer"].ge(0.005)
                & out["valid_score_lift_vs_trainer"].ge(0.000)
                & (out["valid_win_roi_pct"].ge(95.0) | out["valid_place_roi_pct"].ge(95.0))
            ),
        ],
        ["candidate_rule", "shadow_only"],
        default="reject",
    )
    return out.sort_values(["action", "valid_edge_score", "valid_starts"], ascending=[True, False, False])


def validation_segments(valid: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    base = valid[
        ~valid["_registered_trainer"] & valid["_trainer_code"].ge(0) & valid["_workout_count"].gt(0.0)
    ].copy()
    if base.empty:
        return pd.DataFrame()
    candidate_sets: dict[str, dict[str, set[str]]] = {}
    for action in ["candidate_rule", "shadow_only"]:
        candidate_sets[action] = {}
        part = candidates[candidates["action"].eq(action)]
        for family in SPECS:
            candidate_sets[action][family] = set(part.loc[part["family"].eq(family), "rule_key"].astype("string"))

    flags = pd.DataFrame(index=base.index)
    for action in ["candidate_rule", "shadow_only"]:
        action_flag = pd.Series(False, index=base.index)
        for family, cols in SPECS.items():
            keys = base[cols].astype("string").agg("|".join, axis=1)
            action_flag |= keys.isin(candidate_sets[action][family])
        flags[action] = action_flag
    base["_auto_candidate_rule"] = flags["candidate_rule"]
    base["_auto_shadow_only"] = flags["shadow_only"] & ~flags["candidate_rule"]
    base["_auto_any"] = base["_auto_candidate_rule"] | base["_auto_shadow_only"]

    segments = {
        "valid_all_unregistered_with_workout": base,
        "auto_candidate_rule_any": base[base["_auto_candidate_rule"]],
        "auto_shadow_only_any": base[base["_auto_shadow_only"]],
        "auto_candidate_or_shadow_any": base[base["_auto_any"]],
        "auto_no_candidate": base[~base["_auto_any"]],
    }
    return pd.DataFrame([{"segment": name, **metrics(part, "valid")} for name, part in segments.items()])


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine unregistered-trainer workout knowledge candidates.")
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN)
    parser.add_argument("--valid-csv", default=DEFAULT_VALID)
    parser.add_argument("--output-dir", default="outputs/analysis/workout_knowledge_auto_mine_v1")
    parser.add_argument("--min-train", type=int, default=28)
    parser.add_argument("--min-valid", type=int, default=8)
    args = parser.parse_args()

    train_raw = pd.read_csv(args.train_csv, low_memory=False)
    valid_raw = pd.read_csv(args.valid_csv, low_memory=False)
    train = prepare(train_raw, split="train")
    valid = prepare(valid_raw, split="valid")
    candidates = mine(train, valid, min_train=args.min_train, min_valid=args.min_valid)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "workout_auto_knowledge_candidates_all.csv"
    priority_path = out_dir / "workout_auto_knowledge_candidates_priority.csv"
    segment_path = out_dir / "workout_auto_knowledge_validation_segments.csv"
    summary_path = out_dir / "summary.json"
    candidates.to_csv(all_path, index=False, encoding="utf-8-sig")
    priority = candidates[candidates["action"].isin(["candidate_rule", "shadow_only"])].copy()
    priority.to_csv(priority_path, index=False, encoding="utf-8-sig")
    segment_summary = validation_segments(valid, candidates)
    segment_summary.to_csv(segment_path, index=False, encoding="utf-8-sig")
    summary = {
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "candidate_rows": int(len(candidates)),
        "candidate_rule_rows": int((candidates["action"] == "candidate_rule").sum()) if not candidates.empty else 0,
        "shadow_only_rows": int((candidates["action"] == "shadow_only").sum()) if not candidates.empty else 0,
        "all_candidates_csv": str(all_path),
        "priority_candidates_csv": str(priority_path),
        "validation_segments_csv": str(segment_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not priority.empty:
        cols = [
            "_trainer_code",
            "family",
            "rule_key",
            "action",
            "train_starts",
            "valid_starts",
            "valid_win_roi_pct",
            "valid_place_roi_pct",
            "valid_top3_lift_vs_trainer",
            "valid_score_lift_vs_trainer",
            "valid_edge_score",
        ]
        print(priority[cols].head(30).to_string(index=False))
    if not segment_summary.empty:
        print(segment_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
