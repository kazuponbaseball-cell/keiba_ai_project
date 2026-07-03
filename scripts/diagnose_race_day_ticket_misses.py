from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def num(s: pd.Series | Any, default: float = np.nan) -> pd.Series | float:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce")
    try:
        if s is None or pd.isna(s) or str(s).strip() == "":
            return default
        return float(str(s).replace(",", ""))
    except Exception:
        return default


def zrace(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(16)


def norm_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin({"true", "1", "1.0", "yes", "y"})


def single_roi(ai: pd.DataFrame) -> pd.DataFrame:
    work = ai.copy()
    work["ai_rank_num"] = num(work["ai_rank"])
    work["win_odds_num"] = num(work["win_odds"]).fillna(0.0)
    work["is_win_bool"] = norm_bool(work["is_win"]) if "is_win" in work.columns else num(work["finish"]).eq(1)

    buckets: dict[str, pd.Series] = {
        "AI1": work["ai_rank_num"].eq(1),
        "AI2": work["ai_rank_num"].eq(2),
        "AI3": work["ai_rank_num"].eq(3),
        "AI4": work["ai_rank_num"].eq(4),
        "AI5": work["ai_rank_num"].eq(5),
        "AI1-2": work["ai_rank_num"].between(1, 2),
        "AI1-3": work["ai_rank_num"].between(1, 3),
        "AI1-5": work["ai_rank_num"].between(1, 5),
    }
    rows: list[dict[str, Any]] = []
    for label, mask in buckets.items():
        part = work[mask].copy()
        stake = float(len(part) * 100)
        ret = float(np.where(part["is_win_bool"], part["win_odds_num"] * 100.0, 0.0).sum())
        rows.append(
            {
                "bucket": label,
                "bets": int(len(part)),
                "races": int(part["race_id"].nunique()),
                "wins": int(part["is_win_bool"].sum()),
                "win_rate": float(part["is_win_bool"].mean()) if len(part) else np.nan,
                "stake_yen": stake,
                "return_yen": ret,
                "profit_yen": ret - stake,
                "win_roi_pct": ret / stake * 100.0 if stake else np.nan,
                "avg_win_odds": float(part["win_odds_num"].mean()) if len(part) else np.nan,
                "median_win_odds": float(part["win_odds_num"].median()) if len(part) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def pair_key(df: pd.DataFrame, race_col: str, a_col: str, b_col: str) -> pd.Series:
    a = num(df[a_col]).astype("Int64")
    b = num(df[b_col]).astype("Int64")
    lo = pd.concat([a, b], axis=1).min(axis=1).astype("Int64").astype(str)
    hi = pd.concat([a, b], axis=1).max(axis=1).astype("Int64").astype(str)
    return zrace(df[race_col]) + ":" + lo + "-" + hi


def ticket_required_finish(ticket_type: str) -> int:
    raw = str(ticket_type).lower()
    if "wide" in raw or "ワイド" in raw:
        return 3
    return 2


def classify_ticket(row: pd.Series) -> tuple[str, str]:
    if bool(row.get("hit_bool")):
        return "的中", "的中"

    reasons: list[str] = []
    main = "未分類"
    if bool(row.get("pace_miss_flag")):
        reasons.append("展開読みミス")
    if bool(row.get("ai1_good_not_in_ticket_flag")):
        reasons.append("AI1位を買い目化できず")
    if bool(row.get("umaren_to_wide_would_hit_flag")):
        reasons.append("券種選択ミス(馬連→ワイドなら的中)")
    if bool(row.get("a_required_flag")) and not bool(row.get("b_required_flag")):
        main = "相手選定ミス"
        reasons.append("軸は条件内・相手が不足")
    elif bool(row.get("b_required_flag")) and not bool(row.get("a_required_flag")):
        main = "軸選定ミス"
        reasons.append("相手は条件内・軸が不足")
    elif not bool(row.get("a_required_flag")) and not bool(row.get("b_required_flag")):
        if bool(row.get("ai1_good_not_in_ticket_flag")):
            main = "買い目変換ミス"
        elif bool(row.get("pace_miss_flag")):
            main = "展開読みミス"
        else:
            main = "能力評価/候補選定ミス"
        reasons.append("両馬とも必要着順外")
    else:
        main = "券種/払い戻し条件ミス"

    if bool(row.get("partner_rank_floor_fail_flag")):
        reasons.append("相手の能力下限に不安")
    if bool(row.get("favorite_not_in_ticket_flag")):
        reasons.append("人気・本命筋の軽視")
    return main, " / ".join(dict.fromkeys(reasons))


def build_diagnosis(
    ai: pd.DataFrame,
    pnl: pd.DataFrame,
    bias: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    ai = ai.copy()
    ai["race_id"] = zrace(ai["race_id"])
    ai["horse_no_num"] = num(ai["horse_no"]).astype("Int64")
    ai_small = ai[
        [
            "race_id",
            "horse_no_num",
            "horse_name",
            "ai_rank",
            "ai_score",
            "finish",
            "popularity",
            "win_odds",
            "corner4",
            "actual_style",
            "is_top2",
            "is_top3",
        ]
    ].copy()

    pnl = pnl.copy()
    pnl["race_id"] = zrace(pnl["raceId"])
    pnl["a_no_num"] = num(pnl["aNo"]).astype("Int64")
    pnl["b_no_num"] = num(pnl["bNo"]).astype("Int64")
    pnl["ticket_pair_key"] = pair_key(pnl, "raceId", "aNo", "bNo")
    pnl["hit_bool"] = norm_bool(pnl["hit"])

    a = ai_small.rename(
        columns={
            "horse_no_num": "a_no_num",
            "horse_name": "a_horse_name_ai",
            "ai_rank": "a_ai_rank",
            "ai_score": "a_ai_score",
            "finish": "a_finish",
            "popularity": "a_popularity",
            "win_odds": "a_win_odds",
            "corner4": "a_corner4",
            "actual_style": "a_actual_style",
            "is_top2": "a_is_top2",
            "is_top3": "a_is_top3",
        }
    )
    b = ai_small.rename(
        columns={
            "horse_no_num": "b_no_num",
            "horse_name": "b_horse_name_ai",
            "ai_rank": "b_ai_rank",
            "ai_score": "b_ai_score",
            "finish": "b_finish",
            "popularity": "b_popularity",
            "win_odds": "b_win_odds",
            "corner4": "b_corner4",
            "actual_style": "b_actual_style",
            "is_top2": "b_is_top2",
            "is_top3": "b_is_top3",
        }
    )
    out = pnl.merge(a, on=["race_id", "a_no_num"], how="left").merge(b, on=["race_id", "b_no_num"], how="left")

    bias_small = bias.copy()
    bias_small["race_id"] = zrace(bias_small["race_id"])
    keep_bias = [
        c
        for c in [
            "race_id",
            "expected_pace",
            "predicted_bias_shape",
            "actual_bias_shape",
            "mismatch",
            "top3_numbers",
            "winner_no",
            "winner_pop",
            "top3_avg_corner4",
            "top3_front_stalker_count",
            "top3_closer_count",
        ]
        if c in bias_small.columns
    ]
    out = out.merge(bias_small[keep_bias], on="race_id", how="left")

    if not candidates.empty:
        cand = candidates.copy()
        cand["race_id"] = zrace(cand["race_id"])
        if {"anchor_horse_no", "partner_horse_no"}.issubset(cand.columns):
            cand["candidate_pair_key"] = pair_key(cand, "race_id", "anchor_horse_no", "partner_horse_no")
            cand_cols = [
                c
                for c in [
                    "candidate_pair_key",
                    "strongest_current_score",
                    "pair_quinella_score",
                    "pair_score",
                    "ticket_hit_prob",
                    "runtime_expected_roi",
                    "min_odds_margin_ratio",
                    "skip_risk_score",
                    "projected_front5_prob",
                    "position_front_value_score",
                    "pace_fit_pair_score",
                    "pace_regime_front_survival_score",
                    "pace_regime_collapse_conversion_score",
                    "pace_regime_collapse_warning_flag",
                    "ticket_danger_popular_score",
                    "race_difficulty_score",
                    "partner_ai_rank_num",
                    "anchor_ai_rank_num",
                ]
                if c in cand.columns
            ]
            cand = cand[cand_cols].drop_duplicates("candidate_pair_key", keep="first")
            out = out.merge(cand, left_on="ticket_pair_key", right_on="candidate_pair_key", how="left")

    required = out["ticketType"].map(ticket_required_finish)
    out["required_finish"] = required
    out["a_required_flag"] = num(out["a_finish"]).le(required)
    out["b_required_flag"] = num(out["b_finish"]).le(required)
    out["both_top3_flag"] = num(out["a_finish"]).le(3) & num(out["b_finish"]).le(3)
    out["umaren_to_wide_would_hit_flag"] = out["ticketType"].astype(str).str.lower().str.contains("umaren") & out[
        "both_top3_flag"
    ] & ~out["hit_bool"]

    ai1 = ai[num(ai["ai_rank"]).eq(1)].copy()
    ai1 = ai1[
        [
            "race_id",
            "horse_no_num",
            "horse_name",
            "finish",
            "popularity",
            "win_odds",
            "actual_style",
            "is_top2",
            "is_top3",
        ]
    ].rename(
        columns={
            "horse_no_num": "ai1_no",
            "horse_name": "ai1_name",
            "finish": "ai1_finish",
            "popularity": "ai1_popularity",
            "win_odds": "ai1_win_odds",
            "actual_style": "ai1_actual_style",
            "is_top2": "ai1_is_top2",
            "is_top3": "ai1_is_top3",
        }
    )
    out = out.merge(ai1, on="race_id", how="left")
    out["ai1_in_ticket_flag"] = (num(out["ai1_no"]).eq(num(out["aNo"]))) | (num(out["ai1_no"]).eq(num(out["bNo"])))
    out["ai1_good_not_in_ticket_flag"] = norm_bool(out["ai1_is_top2"]) & ~out["ai1_in_ticket_flag"]

    actual = out["actual_bias_shape"].astype(str)
    predicted = out["predicted_bias_shape"].astype(str)
    out["pace_miss_flag"] = (
        (predicted.eq("front_stalker") & actual.eq("closer"))
        | (predicted.eq("closer_watch") & actual.eq("front_stalker"))
    )
    out["pace_match_flag"] = (
        (predicted.eq("front_stalker") & actual.eq("front_stalker"))
        | (predicted.eq("closer_watch") & actual.eq("closer"))
    )
    out["partner_rank_floor_fail_flag"] = num(out["b_ai_rank"]).gt(5) | num(out.get("partner_ai_rank_num")).gt(5)
    out["favorite_not_in_ticket_flag"] = (
        out["finishTop3"].astype(str).str.split("-").str[0].ne(out["aNo"].astype(str))
        & out["finishTop3"].astype(str).str.split("-").str[0].ne(out["bNo"].astype(str))
        & num(out["winner_pop"]).le(2)
    )

    classified = out.apply(classify_ticket, axis=1, result_type="expand")
    out["primary_miss_cause"] = classified[0]
    out["miss_cause_detail"] = classified[1]
    return out


def summarize_diagnosis(diag: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cause = (
        diag.groupby(["decisionGroup", "decisionLabel", "ticketType", "primary_miss_cause"], dropna=False)
        .agg(
            tickets=("race_id", "size"),
            races=("race_id", "nunique"),
            stake_yen=("stakeYen", "sum"),
            payout_yen=("payoutYen", "sum"),
            hits=("hit_bool", "sum"),
            pace_miss=("pace_miss_flag", "sum"),
            ai1_good_not_in_ticket=("ai1_good_not_in_ticket_flag", "sum"),
            umaren_to_wide_would_hit=("umaren_to_wide_would_hit_flag", "sum"),
        )
        .reset_index()
    )
    cause["profit_yen"] = cause["payout_yen"] - cause["stake_yen"]
    cause["roi_pct"] = np.where(cause["stake_yen"] > 0, cause["payout_yen"] / cause["stake_yen"] * 100.0, np.nan)

    final_or_all = (
        diag.groupby(["decisionGroup", "primary_miss_cause"], dropna=False)
        .agg(
            tickets=("race_id", "size"),
            races=("race_id", "nunique"),
            stake_yen=("stakeYen", "sum"),
            payout_yen=("payoutYen", "sum"),
            hits=("hit_bool", "sum"),
        )
        .reset_index()
    )
    final_or_all["profit_yen"] = final_or_all["payout_yen"] - final_or_all["stake_yen"]
    final_or_all["roi_pct"] = np.where(
        final_or_all["stake_yen"] > 0, final_or_all["payout_yen"] / final_or_all["stake_yen"] * 100.0, np.nan
    )
    return cause, final_or_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", default="outputs/analysis/race_day_review_20260627")
    parser.add_argument("--pnl-detail-csv", default="outputs/analysis/current_live_pnl/current_live_pnl_detail.csv")
    parser.add_argument("--candidate-csv", default="outputs/analysis/current_strongest_runtime_v1/current_strongest_all_candidates.csv")
    parser.add_argument("--out-dir", default="outputs/analysis/ticket_miss_diagnosis_20260627")
    args = parser.parse_args()

    review_dir = ROOT / args.review_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ai = read_csv(review_dir / "ai_rank_results_all_horses.csv")
    pnl = read_csv(ROOT / args.pnl_detail_csv)
    bias = read_csv(review_dir / "pace_bias_review_by_race.csv")
    candidates_path = ROOT / args.candidate_csv
    candidates = read_csv(candidates_path) if candidates_path.exists() else pd.DataFrame()

    roi = single_roi(ai)
    roi.to_csv(out_dir / "ai_rank_win_roi.csv", index=False, encoding="utf-8-sig")

    diag = build_diagnosis(ai, pnl, bias, candidates)
    diag.to_csv(out_dir / "ticket_miss_diagnosis.csv", index=False, encoding="utf-8-sig")
    cause, by_group = summarize_diagnosis(diag)
    cause.to_csv(out_dir / "miss_cause_summary.csv", index=False, encoding="utf-8-sig")
    by_group.to_csv(out_dir / "miss_cause_by_decision_group.csv", index=False, encoding="utf-8-sig")

    final = diag[diag["decisionGroup"].eq("final_buy")].copy()
    final_cols = [
        "raceLabel",
        "raceName",
        "ticketLabel",
        "aNo",
        "aName",
        "a_ai_rank",
        "a_finish",
        "a_actual_style",
        "bNo",
        "bName",
        "b_ai_rank",
        "b_finish",
        "b_actual_style",
        "finishTop3",
        "predicted_bias_shape",
        "actual_bias_shape",
        "mismatch",
        "primary_miss_cause",
        "miss_cause_detail",
        "ai1_no",
        "ai1_name",
        "ai1_finish",
        "ai1_in_ticket_flag",
        "liveOdds",
        "stakeYen",
        "payoutYen",
    ]
    final[[c for c in final_cols if c in final.columns]].to_csv(
        out_dir / "final_buy_miss_diagnosis.csv", index=False, encoding="utf-8-sig"
    )

    summary = {
        "output_dir": str(out_dir),
        "ai_rank_win_roi": roi.to_dict(orient="records"),
        "final_buy_miss_causes": final["primary_miss_cause"].value_counts(dropna=False).to_dict(),
        "all_ticket_miss_causes": diag["primary_miss_cause"].value_counts(dropna=False).to_dict(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
