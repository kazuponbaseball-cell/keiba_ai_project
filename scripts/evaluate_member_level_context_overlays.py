from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RUNNER_CACHES = [
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/body_weight_backfilled/owner_breeder_enriched/train_features_with_same_day_bias_v3_retro_body_breeder.csv",
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus/body_weight_backfilled/owner_breeder_enriched/test_features_with_same_day_bias_v3_retro_body_breeder.csv",
]


def _num(series: pd.Series | None, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    if series.dtype == object or str(series.dtype).startswith("string"):
        series = series.astype("string").str.replace(",", "", regex=False).str.replace("+", "", regex=False)
    return pd.to_numeric(series, errors="coerce")


def _id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True).str.strip().str.zfill(16)


def _clip01(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(0.0, 1.0)


def _text(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series("", index=index, dtype="string")
    return series.astype("string").fillna("")


def _class_level(text: pd.Series) -> pd.Series:
    t = text.astype("string").fillna("")
    out = pd.Series(0.35, index=t.index, dtype=float)
    out[t.str.contains("新馬|メイクデビュー", regex=True)] = 0.15
    out[t.str.contains("未勝利", regex=False)] = 0.25
    out[t.str.contains("1勝|１勝|500万|５００万", regex=True)] = 0.45
    out[t.str.contains("2勝|２勝|1000万|１０００万", regex=True)] = 0.60
    out[t.str.contains("3勝|３勝|1600万|１６００万", regex=True)] = 0.72
    out[t.str.contains("OP|オープン|リステッド|L$", regex=True)] = 0.82
    out[t.str.contains("G3|Ｇ３", regex=True)] = 0.90
    out[t.str.contains("G2|Ｇ２", regex=True)] = 0.95
    out[t.str.contains("G1|Ｇ１", regex=True)] = 1.00
    return out


def _rank01(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index, dtype=float)
    return values.rank(pct=True).fillna(0.5)


def _load_history(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, encoding="cp932", nrows=0)
    wanted = [
        "日付S",
        "場所",
        "Ｒ",
        "レース名",
        "クラス名",
        "性別",
        "年齢",
        "斤量",
        "頭数",
        "出走頭数",
        "馬番",
        "人気",
        "単勝オッズ",
        "着順",
        "芝・ダ",
        "距離",
        "馬場状態",
        "年齢限定",
        "限定",
        "性別限定",
        "重量種別",
        "キャリア",
        "レースID(新/馬番無)",
        "前走レース名",
        "前クラス名",
        "前走斤量",
        "前走頭数",
        "前走人気",
        "前走単勝オッズ",
        "前走着順",
        "前芝・ダ",
        "前距離",
        "前走馬場状態",
        "前走着差タイム",
        "前走馬体重",
        "前走レースID(新/馬番無)",
        "前走クラスコード",
        "前走重量コード",
    ]
    usecols = [c for c in wanted if c in header.columns]
    df = pd.read_csv(path, encoding="cp932", usecols=usecols, low_memory=False)
    df = df.rename(columns={"レースID(新/馬番無)": "race_id", "馬番": "horse_no"})
    df["race_id"] = _id(df["race_id"])
    if "前走レースID(新/馬番無)" in df.columns:
        df["_prev_race_id"] = _id(df["前走レースID(新/馬番無)"])
    else:
        df["_prev_race_id"] = pd.Series(pd.NA, index=df.index, dtype="string")

    race_info_cols = [c for c in ["race_id", "レース名", "クラス名", "年齢限定", "性別限定", "重量種別"] if c in df.columns]
    race_info = df[race_info_cols].drop_duplicates("race_id", keep="last").rename(
        columns={
            "race_id": "_prev_race_id",
            "レース名": "_prev_actual_レース名",
            "クラス名": "_prev_actual_クラス名",
            "年齢限定": "_prev_actual_年齢限定",
            "性別限定": "_prev_actual_性別限定",
            "重量種別": "_prev_actual_重量種別",
        }
    )
    df = df.merge(race_info, on="_prev_race_id", how="left")
    df["horse_no"] = _num(df["horse_no"], df.index).astype("Int64")
    idx = df.index

    race_name = _text(df.get("レース名"), idx)
    class_name = _text(df.get("クラス名"), idx)
    prev_race_name = _text(df.get("_prev_actual_レース名"), idx).where(
        _text(df.get("_prev_actual_レース名"), idx).str.len().gt(0),
        _text(df.get("前走レース名"), idx),
    )
    prev_class_name = _text(df.get("_prev_actual_クラス名"), idx).where(
        _text(df.get("_prev_actual_クラス名"), idx).str.len().gt(0),
        _text(df.get("前クラス名"), idx),
    )
    age_limit = _text(df.get("年齢限定"), idx)
    sex_limit = _text(df.get("性別限定"), idx)
    weight_type = _text(df.get("重量種別"), idx)
    prev_age_limit = _text(df.get("_prev_actual_年齢限定"), idx)
    prev_sex_limit = _text(df.get("_prev_actual_性別限定"), idx)
    prev_weight_type = _text(df.get("_prev_actual_重量種別"), idx)

    current_text = class_name + " " + race_name
    prev_text = prev_class_name + " " + prev_race_name
    df["ctx_current_class_level"] = _class_level(current_text)
    df["ctx_prev_class_level"] = _class_level(prev_text)
    df["ctx_class_delta"] = df["ctx_current_class_level"] - df["ctx_prev_class_level"]

    df["ctx_is_handicap"] = (weight_type + " " + race_name).str.contains("ハンデ|H", regex=True).astype(float)
    df["ctx_is_female_only"] = (sex_limit + " " + race_name).str.contains("牝", regex=False).astype(float)
    df["ctx_is_young_only"] = (
        (age_limit + " " + race_name).str.contains("2歳|２歳|3歳|３歳", regex=True)
        & ~(age_limit + " " + race_name).str.contains("以上", regex=False)
    ).astype(float)
    df["ctx_prev_female_like"] = (prev_sex_limit + " " + prev_text).str.contains("牝", regex=False).astype(float)
    df["ctx_prev_young_like"] = (
        (prev_age_limit + " " + prev_text).str.contains("2歳|２歳|3歳|３歳", regex=True)
        & ~(prev_age_limit + " " + prev_text).str.contains("以上", regex=False)
    ).astype(float)
    df["ctx_prev_handicap_like"] = (prev_weight_type + " " + prev_text).str.contains("ハンデ|H", regex=True).astype(float)

    field = _num(df.get("前走頭数"), idx, np.nan).fillna(_num(df.get("頭数"), idx, np.nan)).replace(0, np.nan)
    prev_finish = _num(df.get("前走着順"), idx, np.nan)
    prev_margin = _num(df.get("前走着差タイム"), idx, np.nan).clip(lower=-0.5, upper=3.0)
    prev_pop = _num(df.get("前走人気"), idx, np.nan)
    finish_score = ((field + 1.0 - prev_finish) / field).clip(0.0, 1.0).fillna(0.45)
    close_score = (1.0 - (prev_margin.clip(lower=0.0) / 1.2)).clip(0.0, 1.0).fillna(0.45)
    over_pop_score = ((prev_pop - prev_finish) / field).clip(-0.5, 0.8).fillna(0.0)
    over_pop_score = ((over_pop_score + 0.5) / 1.3).clip(0.0, 1.0)

    df["ctx_prev_finish_score"] = finish_score
    df["ctx_prev_close_score"] = close_score
    df["ctx_prev_over_pop_score"] = over_pop_score
    df["ctx_prev_content_score"] = _clip01(
        0.36 * finish_score
        + 0.26 * close_score
        + 0.24 * df["ctx_prev_class_level"]
        + 0.14 * over_pop_score
    )
    df["ctx_prev_strong_class_good_run"] = _clip01(
        df["ctx_prev_class_level"] * (0.56 * finish_score + 0.32 * close_score + 0.12 * over_pop_score)
    )

    protected_prev = ((df["ctx_prev_female_like"] > 0) | (df["ctx_prev_young_like"] > 0)).astype(float)
    current_mixed = ((df["ctx_is_female_only"] == 0) & (df["ctx_is_young_only"] == 0)).astype(float)
    df["ctx_protected_to_mixed_risk"] = _clip01(protected_prev * current_mixed * (df["ctx_class_delta"] + 0.35))
    df["ctx_handicap_class_blur"] = _clip01(df["ctx_is_handicap"] * (0.35 + 0.65 * df["ctx_prev_class_level"]))
    df["ctx_young_class_uncertainty"] = _clip01(df["ctx_is_young_only"] * (1.0 - _num(df.get("キャリア"), idx, 3).fillna(3).clip(0, 6) / 6.0))
    df["ctx_class_context_penalty"] = _clip01(
        0.42 * df["ctx_protected_to_mixed_risk"]
        + 0.36 * df["ctx_handicap_class_blur"]
        + 0.22 * df["ctx_young_class_uncertainty"]
    )
    df["ctx_mechanical_member_score"] = _clip01(0.58 * df["ctx_prev_class_level"] + 0.42 * df["ctx_prev_content_score"])
    df["ctx_conditional_member_score"] = _clip01(
        0.42 * df["ctx_prev_content_score"]
        + 0.34 * df["ctx_prev_strong_class_good_run"]
        + 0.16 * np.maximum(df["ctx_current_class_level"], df["ctx_prev_class_level"])
        + 0.08 * df["ctx_prev_over_pop_score"]
        - 0.28 * df["ctx_class_context_penalty"]
    )
    keep = [
        "race_id",
        "horse_no",
        "日付S",
        "場所",
        "Ｒ",
        "レース名",
        "クラス名",
        "年齢限定",
        "性別限定",
        "重量種別",
        "_prev_race_id",
        "_prev_actual_レース名",
        "_prev_actual_クラス名",
        "_prev_actual_年齢限定",
        "_prev_actual_性別限定",
        "_prev_actual_重量種別",
        "ctx_current_class_level",
        "ctx_prev_class_level",
        "ctx_class_delta",
        "ctx_is_handicap",
        "ctx_is_female_only",
        "ctx_is_young_only",
        "ctx_prev_female_like",
        "ctx_prev_young_like",
        "ctx_prev_content_score",
        "ctx_prev_strong_class_good_run",
        "ctx_class_context_penalty",
        "ctx_mechanical_member_score",
        "ctx_conditional_member_score",
    ]
    return df[[c for c in keep if c in df.columns]].drop_duplicates(["race_id", "horse_no"], keep="last")


def _load_runner_cache(paths: list[Path]) -> pd.DataFrame:
    wanted = [
        "レースID(新/馬番無)",
        "馬番",
        "クラス名",
        "前クラス名",
        "年齢限定",
        "性別限定",
        "重量種別",
        "class_move_score",
        "rotation_class_up_flag",
        "rotation_class_down_flag",
        "rotation_same_class_flag",
        "prev_class_time_value_score",
        "bias_adjusted_recent_score",
        "race_member_depth_score",
        "prev_race_member_level",
        "past3_avg_race_member_level",
        "past3_max_race_member_level",
        "prev_performance_vs_member_level",
        "past3_avg_performance_vs_member_level",
        "prev_race_next_starters_count",
        "prev_race_next_starters_ratio",
        "prev_race_confirmed_strength_score",
        "prev_race_confirmed_depth_score",
        "prev_confirmed_opponent_good_run_score",
        "prev_confirmed_opponent_excuse_score",
        "past3_confirmed_opponent_strength",
        "confirmed_member_level_adjusted_score",
        "content_common_opponent_adjusted_score",
        "weight_diff",
        "race_weight_light_rank_score",
    ]
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        header = pd.read_csv(path, nrows=0, low_memory=False)
        usecols = [c for c in wanted if c in header.columns]
        if "レースID(新/馬番無)" not in usecols or "馬番" not in usecols:
            continue
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        df = df.rename(columns={"レースID(新/馬番無)": "race_id", "馬番": "horse_no"})
        df["race_id"] = _id(df["race_id"])
        df["horse_no"] = _num(df["horse_no"], df.index).astype("Int64")
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["race_id", "horse_no"])
    return pd.concat(frames, ignore_index=True, sort=False).drop_duplicates(["race_id", "horse_no"], keep="last")


def _merge_cache_features(runners: pd.DataFrame, cache: pd.DataFrame) -> pd.DataFrame:
    if cache.empty:
        return runners
    suffix_cols = [c for c in cache.columns if c not in {"race_id", "horse_no"}]
    cache = cache.rename(columns={c: f"cache_{c}" for c in suffix_cols})
    return runners.merge(cache, on=["race_id", "horse_no"], how="left")


def _merge_runner(pairs: pd.DataFrame, runners: pd.DataFrame, prefix: str, horse_col: str) -> pd.DataFrame:
    runner = runners.copy()
    rename = {c: f"{prefix}_{c}" for c in runner.columns if c not in {"race_id", "horse_no"}}
    runner = runner.rename(columns=rename)
    out = pairs.copy()
    out[horse_col] = _num(out[horse_col], out.index).astype("Int64")
    return out.merge(runner, left_on=["race_id", horse_col], right_on=["race_id", "horse_no"], how="left").drop(
        columns=["horse_no"], errors="ignore"
    )


def _enrich_pairs(pair_path: Path, runners: pd.DataFrame) -> pd.DataFrame:
    pairs = pd.read_csv(pair_path, dtype={"race_id": str}, low_memory=False)
    pairs["race_id"] = _id(pairs["race_id"])
    if "year" not in pairs.columns:
        pairs["year"] = _num(pairs["race_id"].str[:4], pairs.index).astype(int)
    pairs = _merge_runner(pairs, runners, "anchor", "anchor_no")
    pairs = _merge_runner(pairs, runners, "partner", "partner_no")
    idx = pairs.index
    a_mech = _num(pairs.get("anchor_ctx_mechanical_member_score"), idx, 0.5).fillna(0.5)
    b_mech = _num(pairs.get("partner_ctx_mechanical_member_score"), idx, 0.5).fillna(0.5)
    a_cond = _num(pairs.get("anchor_ctx_conditional_member_score"), idx, 0.5).fillna(0.5)
    b_cond = _num(pairs.get("partner_ctx_conditional_member_score"), idx, 0.5).fillna(0.5)
    a_pen = _num(pairs.get("anchor_ctx_class_context_penalty"), idx, 0.0).fillna(0.0)
    b_pen = _num(pairs.get("partner_ctx_class_context_penalty"), idx, 0.0).fillna(0.0)
    pairs["pair_mechanical_member_score"] = _clip01(0.52 * np.maximum(a_mech, b_mech) + 0.48 * ((a_mech + b_mech) / 2.0))
    pairs["pair_conditional_member_score"] = _clip01(0.52 * np.maximum(a_cond, b_cond) + 0.48 * ((a_cond + b_cond) / 2.0))
    pairs["pair_member_context_penalty"] = _clip01(np.maximum(a_pen, b_pen) * 0.62 + ((a_pen + b_pen) / 2.0) * 0.38)

    a_cache_raw = (
        0.30 * _num(pairs.get("anchor_cache_confirmed_member_level_adjusted_score"), idx, 0.0).fillna(0.0)
        + 0.22 * _num(pairs.get("anchor_cache_prev_race_member_level"), idx, 0.0).fillna(0.0)
        + 0.18 * _num(pairs.get("anchor_cache_past3_max_race_member_level"), idx, 0.0).fillna(0.0)
        + 0.14 * _num(pairs.get("anchor_cache_prev_confirmed_opponent_good_run_score"), idx, 0.0).fillna(0.0)
        + 0.10 * _num(pairs.get("anchor_cache_prev_class_time_value_score"), idx, 0.0).fillna(0.0)
        + 0.06 * _num(pairs.get("anchor_cache_rotation_class_down_flag"), idx, 0.0).fillna(0.0)
        - 0.06 * _num(pairs.get("anchor_cache_rotation_class_up_flag"), idx, 0.0).fillna(0.0)
    )
    b_cache_raw = (
        0.30 * _num(pairs.get("partner_cache_confirmed_member_level_adjusted_score"), idx, 0.0).fillna(0.0)
        + 0.22 * _num(pairs.get("partner_cache_prev_race_member_level"), idx, 0.0).fillna(0.0)
        + 0.18 * _num(pairs.get("partner_cache_past3_max_race_member_level"), idx, 0.0).fillna(0.0)
        + 0.14 * _num(pairs.get("partner_cache_prev_confirmed_opponent_good_run_score"), idx, 0.0).fillna(0.0)
        + 0.10 * _num(pairs.get("partner_cache_prev_class_time_value_score"), idx, 0.0).fillna(0.0)
        + 0.06 * _num(pairs.get("partner_cache_rotation_class_down_flag"), idx, 0.0).fillna(0.0)
        - 0.06 * _num(pairs.get("partner_cache_rotation_class_up_flag"), idx, 0.0).fillna(0.0)
    )
    a_cache = _rank01(a_cache_raw)
    b_cache = _rank01(b_cache_raw)
    pairs["pair_cache_member_score"] = _clip01(0.52 * np.maximum(a_cache, b_cache) + 0.48 * ((a_cache + b_cache) / 2.0))

    a_coverage = _num(pairs.get("anchor_cache_prev_race_next_starters_ratio"), idx, 0.0).fillna(0.0).clip(0.0, 1.0)
    b_coverage = _num(pairs.get("partner_cache_prev_race_next_starters_ratio"), idx, 0.0).fillna(0.0).clip(0.0, 1.0)
    pairs["pair_member_confirmation_coverage"] = _clip01(
        0.52 * np.maximum(a_coverage, b_coverage) + 0.48 * ((a_coverage + b_coverage) / 2.0)
    )
    pairs["pair_member_confirmation_pending_score"] = (1.0 - pairs["pair_member_confirmation_coverage"]).clip(0.0, 1.0)

    a_provisional_raw = (
        0.32 * _num(pairs.get("anchor_cache_prev_race_member_level"), idx, 0.0).fillna(0.0)
        + 0.22 * _num(pairs.get("anchor_cache_race_member_depth_score"), idx, 0.0).fillna(0.0)
        + 0.18 * _num(pairs.get("anchor_cache_bias_adjusted_recent_score"), idx, 0.0).fillna(0.0)
        + 0.14 * _num(pairs.get("anchor_cache_prev_class_time_value_score"), idx, 0.0).fillna(0.0)
        + 0.10 * _num(pairs.get("anchor_cache_past3_avg_race_member_level"), idx, 0.0).fillna(0.0)
        + 0.04 * _num(pairs.get("anchor_cache_rotation_class_down_flag"), idx, 0.0).fillna(0.0)
    )
    b_provisional_raw = (
        0.32 * _num(pairs.get("partner_cache_prev_race_member_level"), idx, 0.0).fillna(0.0)
        + 0.22 * _num(pairs.get("partner_cache_race_member_depth_score"), idx, 0.0).fillna(0.0)
        + 0.18 * _num(pairs.get("partner_cache_bias_adjusted_recent_score"), idx, 0.0).fillna(0.0)
        + 0.14 * _num(pairs.get("partner_cache_prev_class_time_value_score"), idx, 0.0).fillna(0.0)
        + 0.10 * _num(pairs.get("partner_cache_past3_avg_race_member_level"), idx, 0.0).fillna(0.0)
        + 0.04 * _num(pairs.get("partner_cache_rotation_class_down_flag"), idx, 0.0).fillna(0.0)
    )
    a_provisional = _rank01(a_provisional_raw)
    b_provisional = _rank01(b_provisional_raw)
    a_pending_aware = (a_coverage * a_cache + (1.0 - a_coverage) * a_provisional).clip(0.0, 1.0)
    b_pending_aware = (b_coverage * b_cache + (1.0 - b_coverage) * b_provisional).clip(0.0, 1.0)
    pairs["pair_pending_aware_member_score"] = _clip01(
        0.52 * np.maximum(a_pending_aware, b_pending_aware) + 0.48 * ((a_pending_aware + b_pending_aware) / 2.0)
    )
    pairs["pair_pending_opportunity_score"] = _clip01(
        pairs["pair_pending_aware_member_score"] * pairs["pair_member_confirmation_pending_score"]
    )

    pairs["score_base"] = _num(pairs.get("strongest_pair_score"), idx, 0.0).fillna(0.0)
    pairs["score_mechanical_member"] = pairs["score_base"] + 0.12 * (_rank01(pairs["pair_mechanical_member_score"]) - 0.5)
    pairs["score_conditional_member"] = (
        pairs["score_base"]
        + 0.12 * (_rank01(pairs["pair_conditional_member_score"]) - 0.5)
        - 0.10 * (_rank01(pairs["pair_member_context_penalty"]) - 0.5)
    )
    pairs["score_cache_member"] = pairs["score_base"] + 0.12 * (_rank01(pairs["pair_cache_member_score"]) - 0.5)
    pairs["score_blended_member_context"] = (
        pairs["score_base"]
        + 0.08 * (_rank01(pairs["pair_cache_member_score"]) - 0.5)
        + 0.05 * (_rank01(pairs["pair_conditional_member_score"]) - 0.5)
        - 0.07 * (_rank01(pairs["pair_member_context_penalty"]) - 0.5)
    )
    pairs["score_pending_aware_member"] = pairs["score_base"] + 0.10 * (
        _rank01(pairs["pair_pending_aware_member_score"]) - 0.5
    )
    pairs["score_pending_opportunity_member"] = (
        pairs["score_base"]
        + 0.08 * (_rank01(pairs["pair_pending_aware_member_score"]) - 0.5)
        + 0.04 * (_rank01(pairs["pair_pending_opportunity_score"]) - 0.5)
    )
    return pairs


def _metrics(df: pd.DataFrame, score_col: str, ticket: str, coverage: float, year: int | None = None) -> dict:
    work = df[df["year"].eq(year)].copy() if year is not None else df.copy()
    if work.empty:
        return {"score": score_col, "ticket": ticket, "coverage": coverage, "year": year or "all", "tickets": 0}
    n = max(1, int(np.ceil(len(work) * coverage)))
    selected = work.sort_values(score_col, ascending=False).head(n).copy()
    if ticket == "umaren":
        pay = _num(selected.get("umaren_pay"), selected.index, 0.0).fillna(0.0)
        hit = selected.get("umaren_hit", False).astype(bool)
    else:
        pay = _num(selected.get("wide_pay"), selected.index, 0.0).fillna(0.0)
        hit = selected.get("wide_hit", False).astype(bool)
    stake = float(len(selected) * 100.0)
    ret = float(pay.where(hit, 0.0).sum())
    race_hits = selected.loc[hit, "race_id"].nunique()
    return {
        "score": score_col,
        "ticket": ticket,
        "coverage": coverage,
        "year": year if year is not None else "all",
        "tickets": int(len(selected)),
        "races": int(selected["race_id"].nunique()),
        "hits": int(hit.sum()),
        "hit_rate": float(hit.mean()) if len(selected) else 0.0,
        "race_hit_rate": float(race_hits / selected["race_id"].nunique()) if selected["race_id"].nunique() else 0.0,
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake,
        "roi": ret / stake if stake else 0.0,
        "avg_score": float(selected[score_col].mean()),
    }


def _segment_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for feature in [
        "pair_mechanical_member_score",
        "pair_conditional_member_score",
        "pair_cache_member_score",
        "pair_pending_aware_member_score",
        "pair_pending_opportunity_score",
        "pair_member_confirmation_coverage",
        "pair_member_context_penalty",
    ]:
        tmp = df.copy()
        bin_id = pd.qcut(_rank01(tmp[feature]), 5, labels=False, duplicates="drop")
        max_bin = int(pd.to_numeric(bin_id, errors="coerce").max()) if pd.notna(bin_id).any() else 0
        labels = {0: "q1_low", max(0, max_bin): "q_high"}
        tmp["bin"] = bin_id.map(lambda value: labels.get(int(value), f"q{int(value) + 1}") if pd.notna(value) else "unknown")
        for (year, bin_label), g in tmp.groupby(["year", "bin"], observed=True):
            for ticket in ["umaren", "wide"]:
                if ticket == "umaren":
                    hit = g.get("umaren_hit", False).astype(bool)
                    pay = _num(g.get("umaren_pay"), g.index, 0.0).fillna(0.0)
                else:
                    hit = g.get("wide_hit", False).astype(bool)
                    pay = _num(g.get("wide_pay"), g.index, 0.0).fillna(0.0)
                stake = len(g) * 100.0
                ret = float(pay.where(hit, 0.0).sum())
                rows.append(
                    {
                        "feature": feature,
                        "year": int(year),
                        "bin": str(bin_label),
                        "ticket": ticket,
                        "rows": int(len(g)),
                        "hit_rate": float(hit.mean()) if len(g) else 0.0,
                        "roi": ret / stake if stake else 0.0,
                        "profit_yen": ret - stake,
                        "avg_feature": float(g[feature].mean()),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate member-level overlays by race condition context.")
    parser.add_argument("--pair-csv", default="outputs/analysis/strongest_final_strength_model_v1/pair_strength_universe.csv")
    parser.add_argument("--history-csv", default=None)
    parser.add_argument("--runner-cache", action="append", default=None)
    parser.add_argument("--output-dir", default="outputs/analysis/member_level_context_overlays_v1")
    args = parser.parse_args()

    root = Path.cwd()
    pair_path = root / args.pair_csv
    history_path = Path(args.history_csv) if args.history_csv else next((root / "date" / "raw").glob("*.csv"))
    cache_paths = [root / p for p in (args.runner_cache or DEFAULT_RUNNER_CACHES)]
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    runners = _load_history(history_path)
    runner_cache = _load_runner_cache(cache_paths)
    runners = _merge_cache_features(runners, runner_cache)
    enriched = _enrich_pairs(pair_path, runners)
    enriched.to_csv(out_dir / "pair_context_enriched.csv", index=False, encoding="utf-8-sig")

    score_cols = [
        "score_base",
        "score_mechanical_member",
        "score_conditional_member",
        "score_cache_member",
        "score_blended_member_context",
        "score_pending_aware_member",
        "score_pending_opportunity_member",
    ]
    coverages = [0.0025, 0.005, 0.01, 0.02, 0.05]
    rows: list[dict] = []
    for score in score_cols:
        for ticket in ["umaren", "wide"]:
            for cov in coverages:
                rows.append(_metrics(enriched, score, ticket, cov, None))
                for year in sorted(enriched["year"].dropna().astype(int).unique()):
                    rows.append(_metrics(enriched, score, ticket, cov, int(year)))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "coverage_metrics.csv", index=False, encoding="utf-8-sig")
    segments = _segment_rows(enriched)
    segments.to_csv(out_dir / "member_context_segments.csv", index=False, encoding="utf-8-sig")

    train = metrics[metrics["year"].isin([2024, 2025])].copy()
    valid = metrics[metrics["year"].eq(2026)].copy()
    compare_cols = ["ticket", "coverage", "score", "tickets", "races", "roi", "hit_rate", "profit_yen"]
    best_train = (
        train[(train["ticket"].eq("umaren")) & (train["tickets"].ge(20))]
        .sort_values(["roi", "profit_yen"], ascending=False)
        .head(20)[compare_cols]
    )
    summary = {
        "pair_csv": str(pair_path),
        "history_csv": str(history_path),
        "runner_cache_rows": int(len(runner_cache)),
        "runner_rows": int(len(runners)),
        "pair_rows": int(len(enriched)),
        "output_dir": str(out_dir),
        "top_train_umaren": best_train.to_dict(orient="records"),
        "validation_2026": valid[valid["ticket"].eq("umaren")][compare_cols].to_dict(orient="records"),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
