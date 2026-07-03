from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_TRAIN_FEATURES = (
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus_rebuilt_20260623/"
    "train_features_with_same_day_bias_v3_retro.csv"
)
DEFAULT_TEST_FEATURES = (
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus_rebuilt_20260623/"
    "test_features_with_same_day_bias_v3_retro.csv"
)
DEFAULT_MODEL = "models/workout_lap_pedigree_interactions_confirmed_opponent_2023plus_rebuilt_20260623/baseline_ranker.pkl"
DEFAULT_PAIR_CANDIDATES = "outputs/analysis/dynamic_pair_ticket_allocation_rebuilt_20260623/pair_candidate_universe.csv"
DEFAULT_DYNAMIC_TICKETS = "outputs/analysis/dynamic_pair_ticket_allocation_rebuilt_20260623/walkforward_selected_tickets.csv"
DEFAULT_PURGED_TICKETS = "outputs/analysis/purged_walkforward_mcs_pbo_rebuilt_20260623/purged_walkforward_selected_tickets.csv"
DEFAULT_OUT = "outputs/analysis/pedigree_ancestor_tag_overlay_rebuilt_20260623"
DEFAULT_DEEP_PEDIGREE_MASTER = "data/processed/target/deep_pedigree_master.csv"

PEDIGREE_COLUMNS = ["種牡馬", "母馬", "母父馬", "母母馬", "父父馬", "父母馬"]
DEEP_PEDIGREE_COLUMNS = [
    "父馬",
    "母馬",
    "父父馬",
    "父母馬",
    "母父馬",
    "母母馬",
    "父父父馬",
    "父父母馬",
    "父母父馬",
    "父母母馬",
    "母父父馬",
    "母父母馬",
    "母母父馬",
    "母母母馬",
]
DEEP_PEDIGREE_META: dict[str, tuple[int, str]] = {
    "父馬": (1, "sire"),
    "母馬": (1, "dam"),
    "父父馬": (2, "sire"),
    "父母馬": (2, "sire"),
    "母父馬": (2, "dam"),
    "母母馬": (2, "dam"),
    "父父父馬": (3, "sire"),
    "父父母馬": (3, "sire"),
    "父母父馬": (3, "sire"),
    "父母母馬": (3, "sire"),
    "母父父馬": (3, "dam"),
    "母父母馬": (3, "dam"),
    "母母父馬": (3, "dam"),
    "母母母馬": (3, "dam"),
}

TAG_PATTERNS: dict[str, list[str]] = {
    "storm_cat": [
        "Storm Cat",
        "ストームキャット",
        "Hennessy",
        "ヘニーヒューズ",
        "Henny Hughes",
        "Scat Daddy",
        "ヨハネスブルグ",
        "Johannesburg",
        "Tale of the Cat",
        "Giant's Causeway",
        "Harlan",
        "Harlan's Holiday",
        "Into Mischief",
        "Forestry",
        "Bluegrass Cat",
        "Bernstein",
    ],
    "roberto": [
        "Roberto",
        "ロベルト",
        "Brian's Time",
        "ブライアンズタイム",
        "Kris S.",
        "Kris S",
        "シンボリクリスエス",
        "Symboli Kris S",
        "Dynaformer",
        "Silver Hawk",
        "グラスワンダー",
        "Grass Wonder",
        "スクリーンヒーロー",
        "Screen Hero",
        "モーリス",
        "Maurice",
        "タニノギムレット",
        "Tanino Gimlet",
        "エピファネイア",
        "Epiphaneia",
    ],
    "nureyev": [
        "Nureyev",
        "ヌレイエフ",
        "Theatrical",
        "シアトリカル",
        "Pivotal",
        "Polar Falcon",
        "Soviet Star",
        "Spinning World",
        "Peintre Celebre",
    ],
    "danzig": [
        "Danzig",
        "ダンジグ",
        "Danehill",
        "デインヒル",
        "Green Desert",
        "War Front",
        "ウォーフロント",
        "Hard Spun",
        "ハードスパン",
        "Dansili",
        "Invincible Spirit",
        "Chief's Crown",
        "チーフズクラウン",
    ],
    "sadlers_wells": [
        "Sadler's Wells",
        "Sadlers Wells",
        "サドラーズウェルズ",
        "Galileo",
        "ガリレオ",
        "Montjeu",
        "モンジュー",
        "Frankel",
        "フランケル",
        "High Chaparral",
        "El Prado",
        "エルプラド",
    ],
    "mr_prospector": [
        "Mr. Prospector",
        "Mr Prospector",
        "ミスタープロスペクター",
        "Gone West",
        "ゴーンウエスト",
        "Forty Niner",
        "フォーティナイナー",
        "Seeking the Gold",
        "Smart Strike",
        "Fappiano",
        "Unbridled",
        "Distorted Humor",
        "Machiavellian",
        "Street Cry",
        "Woodman",
        "Kingmambo",
        "キングマンボ",
    ],
    "northern_dancer": [
        "Northern Dancer",
        "ノーザンダンサー",
        "Danzig",
        "ダンジグ",
        "Sadler's Wells",
        "Sadlers Wells",
        "サドラーズウェルズ",
        "Nureyev",
        "ヌレイエフ",
        "Storm Bird",
        "ストームバード",
        "Lyphard",
        "リファール",
        "Nijinsky",
        "ニジンスキー",
        "Vice Regent",
        "ヴァイスリージェント",
        "Deputy Minister",
        "Green Desert",
        "Danehill",
        "Galileo",
        "Montjeu",
    ],
    "halo": [
        "Halo",
        "ヘイロー",
        "Sunday Silence",
        "サンデーサイレンス",
        "Deep Impact",
        "ディープインパクト",
        "Stay Gold",
        "ステイゴールド",
        "Heart's Cry",
        "ハーツクライ",
        "Black Tide",
        "ブラックタイド",
        "Daiwa Major",
        "ダイワメジャー",
        "Kizuna",
        "キズナ",
        "Orfevre",
        "オルフェーヴル",
        "Gold Ship",
        "ゴールドシップ",
    ],
    "hail_to_reason": [
        "Hail to Reason",
        "ヘイルトゥリーズン",
        "Halo",
        "ヘイロー",
        "Roberto",
        "ロベルト",
        "Sunday Silence",
        "サンデーサイレンス",
        "Brian's Time",
        "ブライアンズタイム",
        "Symboli Kris S",
        "シンボリクリスエス",
    ],
    "kingmambo": [
        "Kingmambo",
        "キングマンボ",
        "King Kamehameha",
        "キングカメハメハ",
        "Lemon Drop Kid",
        "El Condor Pasa",
        "エルコンドルパサー",
        "Rulership",
        "ルーラーシップ",
        "Lord Kanaloa",
        "ロードカナロア",
        "Duramente",
        "ドゥラメンテ",
        "Rey de Oro",
        "レイデオロ",
    ],
    "deputy_minister": [
        "Deputy Minister",
        "デピュティミニスター",
        "French Deputy",
        "フレンチデピュティ",
        "クロフネ",
        "Kurofune",
        "Vice Regent",
        "ヴァイスリージェント",
        "Awesome Again",
        "Silver Deputy",
        "Dehere",
        "デヒア",
    ],
    "blushing_groom": [
        "Blushing Groom",
        "ブラッシンググルーム",
        "Rainbow Quest",
        "Rahy",
        "Nashwan",
        "Groom Dancer",
        "Fantastic Light",
    ],
}


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    values = frame[col]
    if values.dtype == object or str(values.dtype).startswith("string"):
        values = values.astype("string").str.replace(",", "", regex=False).str.replace("+", "", regex=False)
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def norm01(series: pd.Series, lo: float | None = None, hi: float | None = None) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if lo is None:
        lo = float(x.quantile(0.05)) if x.notna().any() else 0.0
    if hi is None:
        hi = float(x.quantile(0.95)) if x.notna().any() else 1.0
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return pd.Series(0.5, index=x.index, dtype=float)
    return ((x.clip(lo, hi) - lo) / (hi - lo)).fillna(0.5).clip(0.0, 1.0)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def top_removed_roi(ret: pd.Series, stake: pd.Series, top_n: int) -> float:
    if len(ret) <= top_n:
        return 0.0
    drop_idx = ret.sort_values(ascending=False).index[:top_n]
    ret2 = ret.drop(index=drop_idx)
    stake2 = stake.drop(index=drop_idx)
    return float(ret2.sum() / stake2.sum()) if stake2.sum() > 0 else 0.0


def max_drawdown_by_race(frame: pd.DataFrame, stake_col: str, return_col: str) -> float:
    if frame.empty:
        return 0.0
    tmp = frame.copy()
    tmp["_stake"] = num(tmp, stake_col)
    tmp["_return"] = num(tmp, return_col)
    tmp["_profit"] = tmp["_return"] - tmp["_stake"]
    sort_cols = [c for c in ["year", "race_id"] if c in tmp.columns]
    if sort_cols:
        tmp = tmp.sort_values(sort_cols, kind="mergesort")
    race_profit = tmp.groupby("race_id", sort=False)["_profit"].sum()
    equity = race_profit.cumsum()
    dd = equity - equity.cummax()
    return float(dd.min()) if len(dd) else 0.0


def ticket_metrics(frame: pd.DataFrame, stake_col: str, return_col: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi": 0.0,
            "ticket_hit_rate": 0.0,
            "race_hit_rate": 0.0,
            "max_drawdown_yen": 0.0,
            "top5_removed_roi": 0.0,
            "top10_removed_roi": 0.0,
        }
    stake = num(frame, stake_col)
    ret = num(frame, return_col)
    hit = ret.gt(0)
    race_hit = frame.assign(_hit=hit).groupby("race_id")["_hit"].max()
    return {
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()) if "race_id" in frame.columns else 0,
        "stake_yen": float(stake.sum()),
        "return_yen": float(ret.sum()),
        "profit_yen": float(ret.sum() - stake.sum()),
        "roi": float(ret.sum() / stake.sum()) if stake.sum() > 0 else 0.0,
        "ticket_hit_rate": float(hit.mean()) if len(hit) else 0.0,
        "race_hit_rate": float(race_hit.mean()) if len(race_hit) else 0.0,
        "max_drawdown_yen": max_drawdown_by_race(frame, stake_col, return_col),
        "top5_removed_roi": top_removed_roi(ret, stake, 5),
        "top10_removed_roi": top_removed_roi(ret, stake, 10),
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}" if math.isfinite(value) else ""
            vals.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def tag_regex(terms: list[str]) -> re.Pattern[str]:
    return re.compile("|".join(re.escape(term.lower()) for term in terms))


def available_pedigree_columns(frame: pd.DataFrame) -> list[str]:
    deep_cols = [c for c in DEEP_PEDIGREE_COLUMNS if c in frame.columns]
    if deep_cols:
        return deep_cols
    return [c for c in PEDIGREE_COLUMNS if c in frame.columns]


def race_condition_masks(frame: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    surface = frame.get("surface", pd.Series("", index=frame.index)).astype(str)
    distance = num(frame, "distance", np.nan)
    going = frame.get("going", pd.Series("", index=frame.index)).astype(str)
    venue = frame.get("venue", pd.Series("", index=frame.index)).astype(str)
    race_class = frame.get("race_class", pd.Series("", index=frame.index)).astype(str)
    frame_no = num(frame, "partner_frame_no", np.nan)
    if frame_no.isna().all():
        frame_no = num(frame, "frame_no", np.nan)
    fast_clock = num(frame, "partner_fast_clock_aptitude_score", np.nan)
    if fast_clock.isna().all():
        fast_clock = num(frame, "fast_clock_aptitude_score", np.nan)
    fast_lap = num(frame, "partner_horse_fast_lap_score_past5", np.nan)
    if fast_lap.isna().all():
        fast_lap = num(frame, "horse_fast_lap_score_past5", np.nan)
    lap_fit = num(frame, "partner_lap_aptitude_fit_score", np.nan)
    if lap_fit.isna().all():
        lap_fit = num(frame, "lap_aptitude_fit_score", np.nan)
    bad_going = going.isin(["稍", "重", "不", "稍重", "不良"])
    local = venue.isin(["札幌", "函館", "福島", "新潟", "小倉"])
    small_turn = venue.isin(["札幌", "函館", "福島", "中山", "小倉"])
    local_small_turn = venue.isin(["札幌", "函館", "福島", "小倉"])
    outer_draw = frame_no.ge(7)
    inner_draw = frame_no.le(2)
    middle_draw = frame_no.between(3, 6, inclusive="both")
    fast_clock_ok = fast_clock.ge(0.62)
    very_fast_clock_ok = fast_clock.ge(0.72)
    fast_lap_ok = fast_lap.ge(0.55)
    lap_fit_ok = lap_fit.ge(0.58)
    return [
        ("dirt", surface.eq("ダ")),
        ("turf", surface.eq("芝")),
        ("dirt_sprint", surface.eq("ダ") & distance.le(1400)),
        ("turf_sprint", surface.eq("芝") & distance.le(1400)),
        ("bad_going", bad_going),
        ("local", local),
        ("local_dirt", local & surface.eq("ダ")),
        ("local_turf", local & surface.eq("芝")),
        ("small_turn", small_turn),
        ("local_small_turn", local_small_turn),
        ("small_turn_dirt", small_turn & surface.eq("ダ")),
        ("small_turn_turf", small_turn & surface.eq("芝")),
        ("outer_draw", outer_draw),
        ("inner_draw", inner_draw),
        ("middle_draw", middle_draw),
        ("outer_dirt_sprint", outer_draw & surface.eq("ダ") & distance.le(1400)),
        ("outer_small_turn", outer_draw & small_turn),
        ("inner_small_turn", inner_draw & small_turn),
        ("fast_clock_aptitude", fast_clock_ok),
        ("very_fast_clock_aptitude", very_fast_clock_ok),
        ("fast_clock_turf", fast_clock_ok & surface.eq("芝")),
        ("fast_clock_dirt", fast_clock_ok & surface.eq("ダ")),
        ("fast_clock_good_turf", fast_clock_ok & surface.eq("芝") & going.eq("良")),
        ("fast_clock_small_turn", fast_clock_ok & small_turn),
        ("fast_lap_aptitude", fast_lap_ok),
        ("lap_fit_high", lap_fit_ok),
        ("maiden_or_new", race_class.str.contains("新馬|未勝", regex=True, na=False)),
        ("first_or_lower", race_class.str.contains("新馬|未勝|1勝", regex=True, na=False)),
    ]


def add_pedigree_tags(profile: pd.DataFrame) -> pd.DataFrame:
    out = profile.copy()
    pedigree_cols = available_pedigree_columns(out)
    pedigree_text = out[pedigree_cols].fillna("").astype(str).agg(" ".join, axis=1)
    pedigree_lower = pedigree_text.str.lower()
    out["pedigree_text"] = pedigree_text
    tag_cols = []
    for tag, terms in TAG_PATTERNS.items():
        regex = tag_regex(terms)
        col = f"tag_{tag}"
        out[col] = pedigree_lower.str.contains(regex, na=False)
        out[f"{col}_count"] = 0
        out[f"{col}_sire_side_count"] = 0
        out[f"{col}_dam_side_count"] = 0
        out[f"{col}_min_generation"] = np.nan
        tag_cols.append(col)
        for pedigree_col in pedigree_cols:
            hit = out[pedigree_col].fillna("").astype(str).str.lower().str.contains(regex, na=False)
            out[f"{col}_count"] += hit.astype(int)
            generation, side = DEEP_PEDIGREE_META.get(pedigree_col, (np.nan, "unknown"))
            if side == "sire":
                out[f"{col}_sire_side_count"] += hit.astype(int)
            elif side == "dam":
                out[f"{col}_dam_side_count"] += hit.astype(int)
            if not pd.isna(generation):
                current = out[f"{col}_min_generation"]
                out[f"{col}_min_generation"] = np.where(
                    hit & pd.isna(current),
                    generation,
                    np.where(hit, np.minimum(pd.to_numeric(current, errors="coerce").fillna(generation), generation), current),
                )
        out[f"{col}_cross_flag"] = out[f"{col}_count"].ge(2)
    out["tag_any_focus"] = out[tag_cols].any(axis=1)
    out["tag_count_focus"] = out[tag_cols].sum(axis=1)
    out["tag_total_ancestor_hits"] = out[[f"{col}_count" for col in tag_cols]].sum(axis=1)
    return out


def load_deep_pedigree_master(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    deep = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, dtype={"血統登録番号": str})
    keep = ["血統登録番号", "ancestor_count", "pedigree_complete_14_flag", *DEEP_PEDIGREE_COLUMNS]
    keep = [c for c in keep if c in deep.columns]
    return deep[keep].drop_duplicates("血統登録番号", keep="last")


def load_profile(
    feature_csv: Path,
    model_csv: Path | None = None,
    model_path: Path | None = None,
    deep_pedigree_master: Path | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(feature_csv, encoding="utf-8-sig", low_memory=False)
    out = pd.DataFrame(index=frame.index)
    out["race_id"] = frame["レースID(新/馬番無)"].astype(str)
    out["horse_no"] = num(frame, "馬番").astype("Int64").astype(str)
    out["血統登録番号"] = frame.get("血統登録番号", pd.Series("", index=frame.index)).astype(str).str.replace(".0", "", regex=False)
    out["horse_name"] = frame["馬名"].astype(str)
    out["venue"] = frame.get("場所", pd.Series("", index=frame.index)).astype(str)
    out["surface"] = frame.get("芝・ダ", pd.Series("", index=frame.index)).astype(str)
    out["distance"] = num(frame, "距離", np.nan)
    out["going"] = frame.get("馬場状態", pd.Series("", index=frame.index)).astype(str)
    out["race_class"] = frame.get("クラス名", pd.Series("", index=frame.index)).astype(str)
    out["frame_no"] = num(frame, "枠番", np.nan)
    out["finish"] = num(frame, "確定着順", np.nan)
    out["popularity"] = num(frame, "人気", np.nan)
    out["odds"] = num(frame, "単勝オッズ", np.nan)
    out["win_pay_100"] = num(frame, "単勝配当", 0.0)
    out["place_pay_100"] = num(frame, "複勝配当", 0.0)
    avg_time = num(frame, "past3_avg_time_value", 0.0)
    best_time = num(frame, "past3_best_time_value", 0.0)
    prev_class_time = num(frame, "prev_class_time_value_score", 0.0)
    fast_lap = num(frame, "horse_fast_lap_score_past5", 0.0).clip(0.0, 1.0)
    lap_fit = num(frame, "lap_aptitude_fit_score", 0.0)
    out["horse_fast_lap_score_past5"] = fast_lap
    out["lap_aptitude_fit_score"] = lap_fit
    out["fast_clock_aptitude_score"] = (
        0.34 * norm01(avg_time, lo=-0.12, hi=0.18)
        + 0.26 * norm01(best_time, lo=-0.08, hi=0.24)
        + 0.20 * norm01(prev_class_time, lo=-0.10, hi=0.22)
        + 0.12 * fast_lap
        + 0.08 * norm01(lap_fit, lo=0.0, hi=1.0)
    ).clip(0.0, 1.0)
    for col in PEDIGREE_COLUMNS:
        out[col] = frame[col].astype(str) if col in frame.columns else ""
    deep = load_deep_pedigree_master(deep_pedigree_master)
    if not deep.empty and "血統登録番号" in out.columns:
        legacy_drop = [c for c in DEEP_PEDIGREE_COLUMNS if c in out.columns]
        out = out.drop(columns=legacy_drop, errors="ignore")
        out = out.merge(deep, on="血統登録番号", how="left")
    if model_path is not None:
        with model_path.open("rb") as f:
            model = pickle.load(f)
        pred = pd.Series(model.predict(frame), index=frame.index, dtype=float)
        out["ai_score"] = pred
        out["ai_rank"] = pred.groupby(out["race_id"]).rank(ascending=False, method="first")
    elif model_csv is not None:
        raise NotImplementedError(model_csv)
    else:
        out["ai_score"] = np.nan
        out["ai_rank"] = np.nan
    out = add_pedigree_tags(out)
    return out.drop_duplicates(["race_id", "horse_no"], keep="last")


def add_side_profile(frame: pd.DataFrame, profile: pd.DataFrame, side: str, no_col: str) -> pd.DataFrame:
    out = frame.copy()
    out["race_id"] = out["race_id"].astype(str)
    out[no_col] = num(out, no_col).astype("Int64").astype(str)
    side_cols = [
        "race_id",
        "horse_no",
        "horse_name",
        "frame_no",
        "fast_clock_aptitude_score",
        "horse_fast_lap_score_past5",
        "lap_aptitude_fit_score",
        "tag_any_focus",
        "tag_count_focus",
        "tag_total_ancestor_hits",
        *[f"tag_{tag}" for tag in TAG_PATTERNS],
        *[f"tag_{tag}_count" for tag in TAG_PATTERNS],
        *[f"tag_{tag}_sire_side_count" for tag in TAG_PATTERNS],
        *[f"tag_{tag}_dam_side_count" for tag in TAG_PATTERNS],
        *[f"tag_{tag}_min_generation" for tag in TAG_PATTERNS],
        *[f"tag_{tag}_cross_flag" for tag in TAG_PATTERNS],
        "pedigree_text",
    ]
    side_cols = [c for c in side_cols if c in profile.columns]
    side_profile = profile[side_cols].add_prefix(f"{side}_").rename(
        columns={f"{side}_race_id": "race_id", f"{side}_horse_no": no_col}
    )
    return out.merge(side_profile, on=["race_id", no_col], how="left")


def add_race_profile(frame: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["race_id"] = out["race_id"].astype(str)
    race_cols = ["race_id", "venue", "surface", "distance", "going", "race_class"]
    race_profile = profile[race_cols].drop_duplicates("race_id", keep="last")
    return out.merge(race_profile, on="race_id", how="left")


def add_pair_tag_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for side in ["anchor", "partner"]:
        out[f"{side}_tag_any_focus"] = out.get(f"{side}_tag_any_focus", False).fillna(False).astype(bool)
        out[f"{side}_tag_count_focus"] = num(out, f"{side}_tag_count_focus", 0)
        for tag in TAG_PATTERNS:
            out[f"{side}_tag_{tag}"] = out.get(f"{side}_tag_{tag}", False).fillna(False).astype(bool)
    out["either_tag_any_focus"] = out["anchor_tag_any_focus"] | out["partner_tag_any_focus"]
    out["both_tag_any_focus"] = out["anchor_tag_any_focus"] & out["partner_tag_any_focus"]
    out["partner_projected_front_ok"] = num(out, "projected_front5_prob").ge(0.45)
    out["partner_tag_front_ok"] = out["partner_tag_any_focus"] & out["partner_projected_front_ok"]
    out["partner_tag_value_like"] = out["partner_tag_any_focus"] & (
        num(out, "partner_pop", 99).ge(4) | num(out, "partner_odds", 0).ge(8.0)
    )
    out["partner_tag_front_value_like"] = out["partner_tag_front_ok"] & out["partner_tag_value_like"]
    return out


def quality_core_mask(frame: pd.DataFrame) -> pd.Series:
    pair_q = num(frame, "pair_quinella_score").quantile(0.75)
    overlay_q = num(frame, "market_overlay_score").quantile(0.75)
    late_q = num(frame, "late_value_survives_score").quantile(0.50)
    front_q = num(frame, "projected_front5_prob").quantile(0.50)
    return (
        num(frame, "pair_quinella_score").ge(pair_q)
        & num(frame, "market_overlay_score").ge(overlay_q)
        & num(frame, "late_value_survives_score").ge(late_q)
        & num(frame, "projected_front5_prob").ge(front_q)
        & num(frame, "anchor_danger").le(0.70)
        & num(frame, "partner_danger").le(0.70)
    )


def evaluate_runner(profile: pd.DataFrame) -> pd.DataFrame:
    frame = profile.copy()
    frame["stake_100"] = 100.0
    frame["win_return_100"] = np.where(num(frame, "finish", 99).eq(1), num(frame, "win_pay_100"), 0.0)
    frame["place_return_100"] = np.where(num(frame, "finish", 99).le(3), num(frame, "place_pay_100"), 0.0)
    masks: list[tuple[str, pd.Series]] = [
        ("all_runners", pd.Series(True, index=frame.index)),
        ("ai_top1", num(frame, "ai_rank", 99).le(1)),
        ("ai_top3", num(frame, "ai_rank", 99).le(3)),
        ("ai_top5", num(frame, "ai_rank", 99).le(5)),
        ("tag_any_focus", frame["tag_any_focus"]),
        ("tag_any_focus_ai_top3", frame["tag_any_focus"] & num(frame, "ai_rank", 99).le(3)),
        ("tag_any_focus_ai_top5", frame["tag_any_focus"] & num(frame, "ai_rank", 99).le(5)),
        ("tag_any_focus_pop4plus_ai_top5", frame["tag_any_focus"] & num(frame, "ai_rank", 99).le(5) & num(frame, "popularity", 99).ge(4)),
    ]
    for tag in TAG_PATTERNS:
        masks.append((f"tag_{tag}_ai_top5", frame[f"tag_{tag}"] & num(frame, "ai_rank", 99).le(5)))
        for condition_name, condition_mask in race_condition_masks(frame):
            masks.append(
                (
                    f"tag_{tag}_{condition_name}_ai_top5",
                    frame[f"tag_{tag}"] & num(frame, "ai_rank", 99).le(5) & condition_mask,
                )
            )
    rows = []
    for name, mask in masks:
        part = frame[mask.fillna(False)]
        for bet, ret_col in [("win", "win_return_100"), ("place", "place_return_100")]:
            m = ticket_metrics(part, "stake_100", ret_col)
            rows.append({"source": "runner", "segment": name, "bet_type": bet, **m})
    return pd.DataFrame(rows)


def evaluate_selected(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    action = frame.get("runtime_action", pd.Series("BUY", index=frame.index)).astype(str)
    buy = action.eq("BUY")
    masks: list[tuple[str, pd.Series]] = [
        ("all_buy_tickets", buy),
        ("partner_tag_any_focus", buy & frame["partner_tag_any_focus"]),
        ("partner_no_focus_tag", buy & ~frame["partner_tag_any_focus"]),
        ("anchor_tag_any_focus", buy & frame["anchor_tag_any_focus"]),
        ("either_tag_any_focus", buy & frame["either_tag_any_focus"]),
        ("partner_tag_front_ok", buy & frame["partner_tag_front_ok"]),
        ("partner_tag_value_like", buy & frame["partner_tag_value_like"]),
        ("partner_tag_front_value_like", buy & frame["partner_tag_front_value_like"]),
    ]
    for tag in TAG_PATTERNS:
        masks.append((f"partner_tag_{tag}", buy & frame[f"partner_tag_{tag}"]))
        masks.append((f"partner_tag_{tag}_front_ok", buy & frame[f"partner_tag_{tag}"] & frame["partner_projected_front_ok"]))
        for condition_name, condition_mask in race_condition_masks(frame):
            masks.append((f"partner_tag_{tag}_{condition_name}", buy & frame[f"partner_tag_{tag}"] & condition_mask))
    rows = []
    for name, mask in masks:
        part = frame[mask.fillna(False)]
        for ticket_type in sorted(part["ticket_type"].dropna().unique()):
            ticket_part = part[part["ticket_type"].eq(ticket_type)]
            rows.append({"source": label, "segment": name, "bet_type": ticket_type, **ticket_metrics(ticket_part, "stake_yen", "return_yen")})
        rows.append({"source": label, "segment": name, "bet_type": "all", **ticket_metrics(part, "stake_yen", "return_yen")})
    return pd.DataFrame(rows)


def evaluate_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["stake_100"] = 100.0
    out["umaren_return_100"] = np.where(out["umaren_hit"].astype(bool), num(out, "umaren_pay"), 0.0)
    out["wide_return_100"] = np.where(out["wide_hit"].astype(bool), num(out, "wide_pay"), 0.0)
    qcore = quality_core_mask(out)
    masks: list[tuple[str, pd.Series]] = [
        ("all_pair_candidates", pd.Series(True, index=out.index)),
        ("quality_core", qcore),
        ("partner_tag_any_focus", out["partner_tag_any_focus"]),
        ("partner_tag_any_focus_quality_core", out["partner_tag_any_focus"] & qcore),
        ("partner_tag_front_ok_quality_core", out["partner_tag_front_ok"] & qcore),
        ("partner_tag_front_value_quality_core", out["partner_tag_front_value_like"] & qcore),
        ("either_tag_any_focus_quality_core", out["either_tag_any_focus"] & qcore),
    ]
    for tag in TAG_PATTERNS:
        masks.append((f"partner_tag_{tag}_quality_core", out[f"partner_tag_{tag}"] & qcore))
        masks.append((f"partner_tag_{tag}_front_quality_core", out[f"partner_tag_{tag}"] & out["partner_projected_front_ok"] & qcore))
        for condition_name, condition_mask in race_condition_masks(out):
            masks.append((f"partner_tag_{tag}_{condition_name}_quality_core", out[f"partner_tag_{tag}"] & qcore & condition_mask))
    rows = []
    for name, mask in masks:
        part = out[mask.fillna(False)]
        for bet, ret_col in [("wide", "wide_return_100"), ("umaren", "umaren_return_100")]:
            rows.append({"source": "pair_candidate_universe", "segment": name, "bet_type": bet, **ticket_metrics(part, "stake_100", ret_col)})
    return pd.DataFrame(rows)


def evaluate_selected_by_year(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    masks: list[tuple[str, pd.Series]] = [
        ("all_buy_tickets", pd.Series(True, index=frame.index)),
        ("partner_tag_any_focus", frame["partner_tag_any_focus"]),
        ("anchor_tag_any_focus", frame["anchor_tag_any_focus"]),
        ("either_tag_any_focus", frame["either_tag_any_focus"]),
    ]
    for tag in TAG_PATTERNS:
        masks.append((f"partner_tag_{tag}", frame[f"partner_tag_{tag}"]))
    rows = []
    for name, mask in masks:
        subset = frame[mask.fillna(False)]
        for year, part in subset.groupby("year", sort=True):
            rows.append(
                {
                    "source": label,
                    "segment": name,
                    "year": int(year),
                    **ticket_metrics(part, "stake_yen", "return_yen"),
                }
            )
    return pd.DataFrame(rows)


def render_review(summary: dict[str, Any], segment_summary: pd.DataFrame, runner_summary: pd.DataFrame) -> str:
    def pick(source: str, bet_type: str, contains: str, n: int = 12) -> list[dict[str, Any]]:
        part = segment_summary[
            segment_summary["source"].eq(source)
            & segment_summary["bet_type"].eq(bet_type)
            & segment_summary["segment"].astype(str).str.contains(contains, regex=False)
        ].copy()
        part = part.sort_values(["races", "roi"], ascending=[False, False]).head(n)
        cols = ["segment", "bet_type", "tickets", "races", "roi", "ticket_hit_rate", "top5_removed_roi", "profit_yen"]
        return part[cols].to_dict("records")

    runner = runner_summary[
        runner_summary["segment"].astype(str).str.contains("tag_", regex=False)
        & runner_summary["bet_type"].isin(["win", "place"])
    ].copy()
    runner = runner.sort_values(["races", "roi"], ascending=[False, False]).head(12)
    runner_cols = ["segment", "bet_type", "tickets", "races", "roi", "ticket_hit_rate", "top5_removed_roi", "profit_yen"]

    lines = [
        "# 系統タグ検証",
        "",
        "## 目的",
        "",
        "Storm Cat / Roberto / Nureyev / Danzig 系のような手作業タグを、BUY拡張または準候補昇格に使えるかを確認した。",
        "資料で提案された主要祖先タグまで広げ、取得済み血統列で拾える範囲の proxy として検証した。",
        "直名だけでは拾えないため、Scat Daddy、Hennessy、Brian's Time、Danehill、Kingmambo、Deputy Minister など近い系統名も同一タグに含めた。",
        "",
        "## 結論",
        "",
        "- 系統タグ単独をBUY条件に昇格するのはまだ危険。",
        "- 既存の最強版BUYに対して、相手馬がフォーカス系統タグを持つだけでは安定改善とは言い切れない。",
        "- 使うなら、血統単独加点ではなく、`前目確率`・`妙味残存`・`ペア確率`を満たした準候補の説明/補助タグに留めるのが安全。",
        "- 取得済み列は完全な5代血統表ではないため、`*_5gen` ではなく `available_pedigree_tag` として扱うのが正確。",
        "- 特にタグ別・条件別では母数がさらに薄くなるため、採用前にT-5/T-3スナップショットでシャドー運用が必要。",
        "",
        "## 入力",
        "",
        f"- train_features: `{summary['inputs']['train_features']}`",
        f"- test_features: `{summary['inputs']['test_features']}`",
        f"- dynamic_tickets: `{summary['inputs']['dynamic_tickets']}`",
        f"- purged_tickets: `{summary['inputs']['purged_tickets']}`",
        f"- pair_candidates: `{summary['inputs']['pair_candidates']}`",
        "",
        "## ランナー単体",
        "",
        markdown_table(runner[runner_cols].to_dict("records")),
        "",
        "## Dynamic BUY",
        "",
        markdown_table(pick("dynamic_selected_tickets", "all", "tag_")),
        "",
        "## Purged BUY",
        "",
        markdown_table(pick("purged_selected_tickets", "all", "tag_")),
        "",
        "## ペア候補 universe",
        "",
        markdown_table(pick("pair_candidate_universe", "umaren", "tag_")),
        "",
        "## 採用判断",
        "",
        "現時点では `採用見送り / シャドー継続`。タグは評価理由の補助としては有用だが、回収率を押し上げる主因としては未確認。",
        "次にやるなら、完全5代血統表の取り込み、母系側タグ・馬場・距離変化・前目確率との交互作用をT-5/T-3固定スナップショット上で検証する。",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-features", default=DEFAULT_TRAIN_FEATURES)
    parser.add_argument("--test-features", default=DEFAULT_TEST_FEATURES)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--pair-candidates", default=DEFAULT_PAIR_CANDIDATES)
    parser.add_argument("--dynamic-tickets", default=DEFAULT_DYNAMIC_TICKETS)
    parser.add_argument("--purged-tickets", default=DEFAULT_PURGED_TICKETS)
    parser.add_argument("--deep-pedigree-master", default=DEFAULT_DEEP_PEDIGREE_MASTER)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_features = project_path(args.train_features)
    test_features = project_path(args.test_features)
    model_path = project_path(args.model)
    pair_candidates_path = project_path(args.pair_candidates)
    dynamic_tickets_path = project_path(args.dynamic_tickets)
    purged_tickets_path = project_path(args.purged_tickets)
    deep_pedigree_master_path = project_path(args.deep_pedigree_master) if args.deep_pedigree_master else None

    profile = load_profile(test_features, model_path=model_path, deep_pedigree_master=deep_pedigree_master_path)
    profile.to_csv(out_dir / "runner_pedigree_tag_profile.csv", index=False, encoding="utf-8-sig")

    dynamic = pd.read_csv(dynamic_tickets_path, encoding="utf-8-sig", low_memory=False)
    purged = pd.read_csv(purged_tickets_path, encoding="utf-8-sig", low_memory=False)
    candidates = pd.read_csv(pair_candidates_path, encoding="utf-8-sig", low_memory=False)

    for name, frame in [("dynamic", dynamic), ("purged", purged), ("candidates", candidates)]:
        frame["race_id"] = frame["race_id"].astype(str)
        frame["anchor_no"] = num(frame, "anchor_no").astype("Int64").astype(str)
        frame["partner_no"] = num(frame, "partner_no").astype("Int64").astype(str)

    dynamic = add_race_profile(dynamic, profile)
    purged = add_race_profile(purged, profile)
    candidates = add_race_profile(candidates, profile)

    dynamic = add_pair_tag_flags(add_side_profile(add_side_profile(dynamic, profile, "anchor", "anchor_no"), profile, "partner", "partner_no"))
    purged = add_pair_tag_flags(add_side_profile(add_side_profile(purged, profile, "anchor", "anchor_no"), profile, "partner", "partner_no"))
    candidates = add_pair_tag_flags(add_side_profile(add_side_profile(candidates, profile, "anchor", "anchor_no"), profile, "partner", "partner_no"))

    dynamic.to_csv(out_dir / "dynamic_selected_with_pedigree_tags.csv", index=False, encoding="utf-8-sig")
    purged.to_csv(out_dir / "purged_selected_with_pedigree_tags.csv", index=False, encoding="utf-8-sig")

    runner_summary = evaluate_runner(profile)
    segment_summary = pd.concat(
        [
            evaluate_selected(dynamic, "dynamic_selected_tickets"),
            evaluate_selected(purged, "purged_selected_tickets"),
            evaluate_candidates(candidates),
        ],
        ignore_index=True,
    )
    runner_summary.to_csv(out_dir / "runner_tag_segments.csv", index=False, encoding="utf-8-sig")
    segment_summary.to_csv(out_dir / "segment_summary.csv", index=False, encoding="utf-8-sig")
    selected_year_summary = pd.concat(
        [
            evaluate_selected_by_year(dynamic, "dynamic_selected_tickets"),
            evaluate_selected_by_year(purged, "purged_selected_tickets"),
        ],
        ignore_index=True,
    )
    selected_year_summary.to_csv(out_dir / "selected_year_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "inputs": {
            "train_features": str(train_features),
            "test_features": str(test_features),
            "model": str(model_path),
            "dynamic_tickets": str(dynamic_tickets_path),
            "purged_tickets": str(purged_tickets_path),
            "pair_candidates": str(pair_candidates_path),
            "deep_pedigree_master": str(deep_pedigree_master_path) if deep_pedigree_master_path else "",
        },
        "tag_patterns": TAG_PATTERNS,
        "runner_tag_counts": {
            tag: int(profile[f"tag_{tag}"].sum()) for tag in TAG_PATTERNS
        }
        | {"any_focus": int(profile["tag_any_focus"].sum()), "all_runners": int(len(profile))},
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "review.md").write_text(render_review(summary, segment_summary, runner_summary), encoding="utf-8")

    print(json.dumps(json_ready({"ok": True, "out_dir": str(out_dir), **summary["runner_tag_counts"]}), ensure_ascii=False))


if __name__ == "__main__":
    main()
