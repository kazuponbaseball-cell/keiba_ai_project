from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BREAKDOWN_DIR = ROOT / "outputs" / "analysis" / "retro_lap_adversity_v1" / "breakdown_v1"
OUT_DIR = BREAKDOWN_DIR / "policy_filter_v1"


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)


def hit_series(df: pd.DataFrame) -> pd.Series:
    s = df["hit"]
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi_pct": 0.0,
            "hit_rate_pct": 0.0,
            "roi_ex_top1_pct": 0.0,
            "top_return_share_pct": 0.0,
            "anchor_top3_rate_pct": 0.0,
            "partner_top3_rate_pct": 0.0,
            "both_top3_rate_pct": 0.0,
        }
    ret_series = pd.to_numeric(df["return_yen"], errors="coerce").fillna(0.0)
    stake = len(df) * 100.0
    ret = float(ret_series.sum())
    top = float(ret_series.max())
    ex = df.drop(ret_series.idxmax()) if top > 0 else df.iloc[0:0]
    ex_ret = float(pd.to_numeric(ex["return_yen"], errors="coerce").fillna(0.0).sum()) if not ex.empty else 0.0
    ex_stake = len(ex) * 100.0
    anchor_finish = pd.to_numeric(df["anchor_finish"], errors="coerce")
    partner_finish = pd.to_numeric(df["partner_finish"], errors="coerce")
    return {
        "tickets": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "stake_yen": round(stake, 1),
        "return_yen": round(ret, 1),
        "profit_yen": round(ret - stake, 1),
        "roi_pct": round(ret / stake * 100, 1) if stake else 0.0,
        "hit_rate_pct": round(float(hit_series(df).mean() * 100), 1),
        "roi_ex_top1_pct": round(ex_ret / ex_stake * 100, 1) if ex_stake else 0.0,
        "top_return_share_pct": round(top / ret * 100, 1) if ret else 0.0,
        "anchor_top3_rate_pct": round(float(anchor_finish.le(3).mean() * 100), 1),
        "partner_top3_rate_pct": round(float(partner_finish.le(3).mean() * 100), 1),
        "both_top3_rate_pct": round(float((anchor_finish.le(3) & partner_finish.le(3)).mean() * 100), 1),
    }


def evaluate_filters(df: pd.DataFrame) -> pd.DataFrame:
    conditions: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
        "小倉": lambda d: d["venue"].eq("小倉"),
        "函館": lambda d: d["venue"].eq("函館"),
        "阪神ダート": lambda d: d["venue"].eq("阪神") & d["surface"].eq("ダ"),
        "ダート": lambda d: d["surface"].eq("ダ"),
        "芝": lambda d: d["surface"].eq("芝"),
        "1勝": lambda d: d["class_name"].eq("1勝"),
        "未勝利": lambda d: d["class_name"].eq("未勝利"),
        "新馬": lambda d: d["class_name"].eq("新馬"),
        "オープン以上": lambda d: d["class_name"].astype(str).str.contains("OP|ｵｰﾌﾟﾝ|Ｇ", regex=True, na=False),
        "1800-1999": lambda d: d["distance_bucket"].eq("1800-1999"),
        "2000-2199": lambda d: d["distance_bucket"].eq("2000-2199"),
        "<=1199": lambda d: d["distance_bucket"].eq("<=1199"),
        "相手4-5人気": lambda d: d["partner_pop_bucket"].eq("4-5人気"),
        "相手6-8人気": lambda d: d["partner_pop_bucket"].eq("6-8人気"),
        "相手9-12人気": lambda d: d["partner_pop_bucket"].eq("9-12人気"),
        "相手13人気+": lambda d: d["partner_pop_bucket"].eq("13人気+"),
        "相手25-60倍": lambda d: d["partner_odds_bucket"].eq("25-60"),
        "相手60倍+": lambda d: d["partner_odds_bucket"].eq("60+"),
        "幾何オッズ4-7": lambda d: d["odds_geom_bucket"].eq("4-7"),
        "幾何オッズ12-25": lambda d: d["odds_geom_bucket"].eq("12-25"),
        "幾何オッズ25-60": lambda d: d["odds_geom_bucket"].eq("25-60"),
        "mixed_queue": lambda d: d["queue_shape_label"].eq("mixed_queue"),
        "front_duel_dense": lambda d: d["queue_shape_label"].eq("front_duel_dense"),
        "ダート×1勝": lambda d: d["surface"].eq("ダ") & d["class_name"].eq("1勝"),
        "ダート×未勝利": lambda d: d["surface"].eq("ダ") & d["class_name"].eq("未勝利"),
        "ダート×1800-1999": lambda d: d["surface"].eq("ダ") & d["distance_bucket"].eq("1800-1999"),
        "ダート×1勝×1800-1999": lambda d: d["surface"].eq("ダ") & d["class_name"].eq("1勝") & d["distance_bucket"].eq("1800-1999"),
        "ダート×1勝×相手9-12人気": lambda d: d["surface"].eq("ダ") & d["class_name"].eq("1勝") & d["partner_pop_bucket"].eq("9-12人気"),
        "ダート×1勝×1800-1999×相手9-12人気": lambda d: d["surface"].eq("ダ")
        & d["class_name"].eq("1勝")
        & d["distance_bucket"].eq("1800-1999")
        & d["partner_pop_bucket"].eq("9-12人気"),
        "ダート×front_duel_dense": lambda d: d["surface"].eq("ダ") & d["queue_shape_label"].eq("front_duel_dense"),
        "ダート×front_duel_dense×front_loaded": lambda d: d["surface"].eq("ダ")
        & d["queue_shape_label"].eq("front_duel_dense")
        & d["actual_lap_regime"].eq("front_loaded"),
        "mixed_queue×front_loaded×相手6-8人気": lambda d: d["queue_shape_label"].eq("mixed_queue")
        & d["actual_lap_regime"].eq("front_loaded")
        & d["partner_pop_bucket"].eq("6-8人気"),
        "芝×2000-2199": lambda d: d["surface"].eq("芝") & d["distance_bucket"].eq("2000-2199"),
        "芝×1勝×幾何オッズ12-25": lambda d: d["surface"].eq("芝")
        & d["class_name"].eq("1勝")
        & d["odds_geom_bucket"].eq("12-25"),
        "小倉×ダート": lambda d: d["venue"].eq("小倉") & d["surface"].eq("ダ"),
        "小倉×芝": lambda d: d["venue"].eq("小倉") & d["surface"].eq("芝"),
        "小倉×1勝": lambda d: d["venue"].eq("小倉") & d["class_name"].eq("1勝"),
        "小倉×ダート×1勝": lambda d: d["venue"].eq("小倉") & d["surface"].eq("ダ") & d["class_name"].eq("1勝"),
    }
    rows: list[dict] = []
    total = metrics(df)
    total["condition"] = "ALL"
    total["side"] = "ALL"
    total["share_pct"] = 100.0
    rows.append(total)
    for name, func in conditions.items():
        mask = func(df).fillna(False)
        for side, side_mask in [("IN", mask), ("OUT", ~mask)]:
            row = metrics(df[side_mask])
            row["condition"] = name
            row["side"] = side
            row["share_pct"] = round(float(side_mask.mean() * 100), 1)
            rows.append(row)
    out = pd.DataFrame(rows)
    return out[
        [
            "condition",
            "side",
            "tickets",
            "share_pct",
            "roi_pct",
            "hit_rate_pct",
            "profit_yen",
            "roi_ex_top1_pct",
            "top_return_share_pct",
            "anchor_top3_rate_pct",
            "partner_top3_rate_pct",
            "both_top3_rate_pct",
        ]
    ]


def evaluate_policy_candidates(df: pd.DataFrame) -> pd.DataFrame:
    policies: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
        "no_filter": lambda d: pd.Series(True, index=d.index),
        "exclude_2000_2199_and_under1199": lambda d: ~d["distance_bucket"].isin(["2000-2199", "<=1199"]),
        "exclude_longshot_tail": lambda d: ~d["partner_pop_bucket"].eq("13人気+") & ~d["odds_geom_bucket"].eq("25-60"),
        "exclude_all_period_weak_core": lambda d: ~(
            (d["surface"].eq("ダ") & d["queue_shape_label"].eq("front_duel_dense") & d["actual_lap_regime"].eq("front_loaded"))
            | (d["queue_shape_label"].eq("mixed_queue") & d["actual_lap_regime"].eq("front_loaded") & d["partner_pop_bucket"].eq("6-8人気"))
            | (d["surface"].eq("芝") & d["distance_bucket"].eq("2000-2199"))
            | (d["surface"].eq("芝") & d["class_name"].eq("1勝") & d["odds_geom_bucket"].eq("12-25"))
            | (d["venue"].eq("阪神") & d["surface"].eq("ダ"))
        ),
        "exclude_core_plus_bad_distance": lambda d: ~(
            (d["surface"].eq("ダ") & d["queue_shape_label"].eq("front_duel_dense") & d["actual_lap_regime"].eq("front_loaded"))
            | (d["queue_shape_label"].eq("mixed_queue") & d["actual_lap_regime"].eq("front_loaded") & d["partner_pop_bucket"].eq("6-8人気"))
            | (d["surface"].eq("芝") & d["distance_bucket"].eq("2000-2199"))
            | (d["surface"].eq("芝") & d["class_name"].eq("1勝") & d["odds_geom_bucket"].eq("12-25"))
            | (d["venue"].eq("阪神") & d["surface"].eq("ダ"))
            | d["distance_bucket"].isin(["2000-2199", "<=1199"])
        ),
        "exclude_2026_bad_looking_not_recommended": lambda d: ~(
            (d["venue"].eq("小倉"))
            | (d["surface"].eq("ダ") & d["class_name"].eq("1勝") & d["distance_bucket"].eq("1800-1999"))
            | (d["partner_pop_bucket"].eq("9-12人気"))
        ),
    }
    rows: list[dict] = []
    for name, func in policies.items():
        mask = func(df).fillna(False)
        row = metrics(df[mask])
        row["policy"] = name
        row["kept_share_pct"] = round(float(mask.mean() * 100), 1)
        rows.append(row)
        for year, g in df[mask].groupby("year"):
            yrow = metrics(g)
            yrow["policy"] = name
            yrow["year"] = int(year)
            yrow["kept_share_pct"] = round(float(mask[df["year"].eq(year)].mean() * 100), 1)
            rows.append(yrow)
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = read_csv(BREAKDOWN_DIR / "selected_low_risk_umaren.csv", dtype={"race_id": str})

    filters = evaluate_filters(df)
    policies = evaluate_policy_candidates(df)
    filters.to_csv(OUT_DIR / "condition_filter_matrix.csv", index=False, encoding="utf-8-sig")
    policies.to_csv(OUT_DIR / "policy_candidate_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "total": metrics(df),
        "weak_in_conditions": filters[(filters["side"].eq("IN")) & (filters["tickets"].ge(30))]
        .sort_values(["roi_pct", "tickets"], ascending=[True, False])
        .head(20)
        .replace({np.nan: None})
        .to_dict(orient="records"),
        "best_out_filters": filters[(filters["side"].eq("OUT")) & (filters["tickets"].ge(500))]
        .sort_values(["roi_pct", "tickets"], ascending=[False, False])
        .head(15)
        .replace({np.nan: None})
        .to_dict(orient="records"),
        "policy_candidates": policies[policies["year"].isna() if "year" in policies.columns else pd.Series(True, index=policies.index)]
        .replace({np.nan: None})
        .to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
