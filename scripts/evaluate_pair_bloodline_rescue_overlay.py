from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RUNNER_SCORES = "outputs/analysis/maternal_pedigree_rescue_challenger_v1/runner_scores.csv"
DEFAULT_PAIR_CANDIDATES = "outputs/analysis/dynamic_pair_ticket_allocation_quinella_model_v1/pair_candidate_universe.csv"
DEFAULT_SELECTED_TICKETS = "outputs/analysis/mcs_pbo_runtime_overlay_v3/recommended_all_tickets.csv"
DEFAULT_OUT = "outputs/analysis/pair_bloodline_rescue_overlay_v1"


RUNNER_SCORE_COLS = [
    "maternal_rescue_fit_score",
    "maternal_family_lower_bound_score",
    "maternal_family_reliability_score",
    "pure_bloodline_lift_fit_score",
    "combined_pedigree_rescue_score",
    "low_career_pedigree_rescue_score",
    "ai_market_residual_z",
    "ai_rank_market_residual",
    "fc_condition_uncertainty_score",
    "キャリア",
    "人気",
    "単勝オッズ",
    "ai_rank",
]


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(__file__).resolve().parents[1] / p


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


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


def top_removed_roi(return_values: pd.Series, stake_values: pd.Series, top_n: int) -> float:
    if len(return_values) <= top_n:
        return 0.0
    order = return_values.sort_values(ascending=False).index[:top_n]
    ret = return_values.drop(index=order).sum()
    stake = stake_values.drop(index=order).sum()
    return float(ret / stake) if stake > 0 else 0.0


def max_drawdown_by_race(frame: pd.DataFrame, *, stake_col: str, return_col: str) -> float:
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
    drawdown = equity - equity.cummax()
    return float(drawdown.min()) if len(drawdown) else 0.0


def ticket_metrics(frame: pd.DataFrame, *, stake_col: str, return_col: str) -> dict[str, Any]:
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
        "max_drawdown_yen": max_drawdown_by_race(frame, stake_col=stake_col, return_col=return_col),
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


def load_runner_scores(path: Path) -> pd.DataFrame:
    scores = pd.read_csv(path)
    if "ai_rank" in scores.columns:
        scores = scores.rename(columns={"ai_rank": "runner_ai_rank"})
    needed = ["レースID(新/馬番無)", "馬名", *[c for c in RUNNER_SCORE_COLS if c in scores.columns]]
    if "runner_ai_rank" in scores.columns:
        needed.append("runner_ai_rank")
    needed = list(dict.fromkeys(needed))
    scores = scores[needed].copy()
    scores["race_id"] = scores["レースID(新/馬番無)"].astype(str)
    scores["horse_name"] = scores["馬名"].astype(str)
    scores = scores.drop(columns=["レースID(新/馬番無)", "馬名"])
    return scores.drop_duplicates(["race_id", "horse_name"], keep="last")


def add_runner_side(frame: pd.DataFrame, scores: pd.DataFrame, *, side: str, name_col: str) -> pd.DataFrame:
    out = frame.copy()
    out["race_id"] = out["race_id"].astype(str)
    out[name_col] = out[name_col].astype(str)
    side_scores = scores.add_prefix(f"{side}_")
    side_scores = side_scores.rename(
        columns={f"{side}_race_id": "race_id", f"{side}_horse_name": name_col}
    )
    return out.merge(side_scores, on=["race_id", name_col], how="left")


def add_bloodline_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    score_cols = [
        "maternal_rescue_fit_score",
        "maternal_family_lower_bound_score",
        "pure_bloodline_lift_fit_score",
        "combined_pedigree_rescue_score",
        "low_career_pedigree_rescue_score",
    ]
    for side in ["anchor", "partner"]:
        for col in score_cols:
            full = f"{side}_{col}"
            out[full] = num(out, full, 0.0)

    partner_combo = num(out, "partner_combined_pedigree_rescue_score")
    partner_low = num(out, "partner_low_career_pedigree_rescue_score")
    partner_mat = num(out, "partner_maternal_rescue_fit_score")
    partner_pure = num(out, "partner_pure_bloodline_lift_fit_score")
    partner_uncertainty = num(out, "partner_fc_condition_uncertainty_score")

    anchor_combo = num(out, "anchor_combined_pedigree_rescue_score")
    anchor_low = num(out, "anchor_low_career_pedigree_rescue_score")
    anchor_mat = num(out, "anchor_maternal_rescue_fit_score")
    anchor_pure = num(out, "anchor_pure_bloodline_lift_fit_score")
    anchor_uncertainty = num(out, "anchor_fc_condition_uncertainty_score")

    thresholds: dict[str, float] = {}
    for label, series in {
        "partner_combo_q75": partner_combo,
        "partner_combo_q90": partner_combo,
        "partner_low_q75": partner_low,
        "partner_low_q90": partner_low,
        "partner_mat_q75": partner_mat,
        "partner_pure_q75": partner_pure,
        "anchor_combo_q75": anchor_combo,
        "anchor_low_q75": anchor_low,
        "anchor_mat_q75": anchor_mat,
        "anchor_pure_q75": anchor_pure,
    }.items():
        q = 0.90 if label.endswith("q90") else 0.75
        thresholds[label] = float(series.quantile(q))

    out["partner_bloodline_rescue_hi"] = (
        partner_combo.ge(thresholds["partner_combo_q75"])
        | partner_low.ge(thresholds["partner_low_q75"])
        | (partner_mat.ge(thresholds["partner_mat_q75"]) & partner_uncertainty.ge(0.35))
        | (partner_pure.ge(thresholds["partner_pure_q75"]) & partner_uncertainty.ge(0.35))
    )
    out["partner_bloodline_rescue_top"] = partner_combo.ge(thresholds["partner_combo_q90"]) | partner_low.ge(
        thresholds["partner_low_q90"]
    )
    out["anchor_bloodline_rescue_hi"] = (
        anchor_combo.ge(thresholds["anchor_combo_q75"])
        | anchor_low.ge(thresholds["anchor_low_q75"])
        | (anchor_mat.ge(thresholds["anchor_mat_q75"]) & anchor_uncertainty.ge(0.35))
        | (anchor_pure.ge(thresholds["anchor_pure_q75"]) & anchor_uncertainty.ge(0.35))
    )
    out.attrs["bloodline_thresholds"] = thresholds
    return out


def build_candidate_segments(frame: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    q_pair = float(num(frame, "pair_quinella_score").quantile(0.75))
    q_pair_top = float(num(frame, "pair_quinella_score").quantile(0.90))
    q_overlay = float(num(frame, "market_overlay_score").quantile(0.75))
    q_late = float(num(frame, "late_value_survives_score").quantile(0.50))
    q_front = float(num(frame, "projected_front5_prob").quantile(0.50))

    quality_core = (
        num(frame, "pair_quinella_score").ge(q_pair)
        & num(frame, "market_overlay_score").ge(q_overlay)
        & num(frame, "late_value_survives_score").ge(q_late)
        & num(frame, "projected_front5_prob").ge(q_front)
        & num(frame, "anchor_danger").le(0.70)
        & num(frame, "partner_danger").le(0.70)
    )
    quality_top = quality_core & num(frame, "pair_quinella_score").ge(q_pair_top)
    partner_value = (num(frame, "partner_pop", 99).ge(4) | num(frame, "partner_odds", 0).ge(8.0)) & num(
        frame, "partner_ai_rank", 99
    ).le(8)
    anchor_ok = (num(frame, "anchor_quinella_model_rank", 99).le(8) | num(frame, "anchor_pop", 99).le(3)) & num(
        frame, "skip_risk_score"
    ).le(0.75)

    segments = [
        ("all_pair_candidates", pd.Series(True, index=frame.index)),
        ("quality_core", quality_core),
        ("quality_top", quality_top),
        ("partner_bloodline_hi_all", frame["partner_bloodline_rescue_hi"]),
        ("partner_bloodline_hi_quality_core", frame["partner_bloodline_rescue_hi"] & quality_core),
        ("partner_bloodline_top_quality_core", frame["partner_bloodline_rescue_top"] & quality_core),
        ("partner_bloodline_hi_quality_top", frame["partner_bloodline_rescue_hi"] & quality_top),
        ("partner_bloodline_hi_value_quality", frame["partner_bloodline_rescue_hi"] & quality_core & partner_value),
        (
            "partner_bloodline_hi_anchor_ok_value_quality",
            frame["partner_bloodline_rescue_hi"] & quality_core & partner_value & anchor_ok,
        ),
        ("anchor_bloodline_hi_quality_core", frame["anchor_bloodline_rescue_hi"] & quality_core),
        (
            "either_side_bloodline_hi_quality_core",
            (frame["anchor_bloodline_rescue_hi"] | frame["partner_bloodline_rescue_hi"]) & quality_core,
        ),
        (
            "both_sides_bloodline_hi_quality_core",
            frame["anchor_bloodline_rescue_hi"] & frame["partner_bloodline_rescue_hi"] & quality_core,
        ),
    ]
    return segments


def evaluate_candidate_universe(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["stake_100"] = 100.0
    out["umaren_return_100"] = np.where(out["umaren_hit"].astype(bool), num(out, "umaren_pay"), 0.0)
    out["wide_return_100"] = np.where(out["wide_hit"].astype(bool), num(out, "wide_pay"), 0.0)

    rows = []
    for segment, mask in build_candidate_segments(out):
        part = out[mask.fillna(False)].copy()
        rows.append({"scope": "candidate_universe", "bet_type": "umaren", "segment": segment, **ticket_metrics(part, stake_col="stake_100", return_col="umaren_return_100")})
        rows.append({"scope": "candidate_universe", "bet_type": "wide", "segment": segment, **ticket_metrics(part, stake_col="stake_100", return_col="wide_return_100")})
    return pd.DataFrame(rows)


def evaluate_selected_tickets(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    runtime_action = frame.get("runtime_action", pd.Series("BUY", index=frame.index)).astype(str)
    masks = [
        ("all_recommended_file", pd.Series(True, index=frame.index)),
        ("runtime_buy_only", runtime_action.eq("BUY")),
        ("runtime_skip_only", runtime_action.ne("BUY")),
        ("buy_partner_bloodline_hi", runtime_action.eq("BUY") & frame["partner_bloodline_rescue_hi"]),
        ("buy_partner_bloodline_top", runtime_action.eq("BUY") & frame["partner_bloodline_rescue_top"]),
        ("buy_partner_bloodline_not_hi", runtime_action.eq("BUY") & ~frame["partner_bloodline_rescue_hi"]),
        ("buy_anchor_bloodline_hi", runtime_action.eq("BUY") & frame["anchor_bloodline_rescue_hi"]),
        ("buy_either_side_bloodline_hi", runtime_action.eq("BUY") & (frame["anchor_bloodline_rescue_hi"] | frame["partner_bloodline_rescue_hi"])),
        ("buy_both_sides_bloodline_hi", runtime_action.eq("BUY") & frame["anchor_bloodline_rescue_hi"] & frame["partner_bloodline_rescue_hi"]),
    ]
    for segment, mask in masks:
        part = frame[mask.fillna(False)].copy()
        rows.append({"scope": "selected_tickets", "bet_type": "actual_stake", "segment": segment, **ticket_metrics(part, stake_col="stake_yen", return_col="return_yen")})
    return pd.DataFrame(rows)


def write_review(out_dir: Path, summary: dict[str, Any], segment_summary: pd.DataFrame) -> None:
    selected = segment_summary[segment_summary["scope"].eq("selected_tickets")].copy()
    candidates = segment_summary[segment_summary["scope"].eq("candidate_universe")].copy()
    key_rows = []
    for label in [
        "runtime_buy_only",
        "buy_partner_bloodline_hi",
        "buy_partner_bloodline_top",
        "buy_partner_bloodline_not_hi",
    ]:
        part = selected[selected["segment"].eq(label)]
        if not part.empty:
            key_rows.append(part.iloc[0].to_dict())
    for label in [
        "quality_core",
        "partner_bloodline_hi_quality_core",
        "partner_bloodline_hi_value_quality",
        "partner_bloodline_hi_anchor_ok_value_quality",
    ]:
        for bet_type in ["umaren", "wide"]:
            part = candidates[candidates["segment"].eq(label) & candidates["bet_type"].eq(bet_type)]
            if not part.empty:
                key_rows.append(part.iloc[0].to_dict())

    verdict = summary["verdict"]
    body = [
        "# Pair Bloodline Rescue Overlay v1",
        "",
        "外部AIの指摘に沿って、血統を単馬評価ではなく「馬連・ワイドの相手候補救済」として使えるかを検証した。",
        "",
        "## Verdict",
        "",
        f"- 採用判断: **{verdict['decision']}**",
        f"- 理由: {verdict['reason']}",
        f"- 次の扱い: {verdict['next_action']}",
        "",
        "## Key Metrics",
        "",
        markdown_table(key_rows),
        "",
        "## Notes",
        "",
        "- 候補宇宙は各ペア100円均等で、最強版の実購入金額とは別評価。",
        "- selected_tickets は `mcs_pbo_runtime_overlay_v3/recommended_all_tickets.csv` の実ステーク・実払戻で評価。",
        "- 血統スコアは race_id + 馬名で結合。欠損は0扱いだが、結合率は summary.json に保存。",
        "- この検証は採用可否の一次判定。閾値探索を増やす場合はMCS/PBO探索本数に含める必要がある。",
    ]
    (out_dir / "review.md").write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner-scores", default=DEFAULT_RUNNER_SCORES)
    parser.add_argument("--pair-candidates", default=DEFAULT_PAIR_CANDIDATES)
    parser.add_argument("--selected-tickets", default=DEFAULT_SELECTED_TICKETS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args()

    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner_scores = load_runner_scores(project_path(args.runner_scores))

    pair_candidates = pd.read_csv(project_path(args.pair_candidates))
    pair_candidates["race_id"] = pair_candidates["race_id"].astype(str)
    pair_candidates = add_runner_side(pair_candidates, runner_scores, side="anchor", name_col="anchor_name")
    pair_candidates = add_runner_side(pair_candidates, runner_scores, side="partner", name_col="partner_name")
    pair_candidates = add_bloodline_flags(pair_candidates)
    candidate_summary = evaluate_candidate_universe(pair_candidates)

    selected_tickets = pd.read_csv(project_path(args.selected_tickets), low_memory=False)
    selected_tickets["race_id"] = selected_tickets["race_id"].astype(str)
    selected_tickets = add_runner_side(selected_tickets, runner_scores, side="anchor", name_col="anchor_name")
    selected_tickets = add_runner_side(selected_tickets, runner_scores, side="partner", name_col="partner_name")
    selected_tickets = add_bloodline_flags(selected_tickets)
    selected_summary = evaluate_selected_tickets(selected_tickets)

    segment_summary = pd.concat([selected_summary, candidate_summary], ignore_index=True)
    segment_summary = segment_summary.sort_values(["scope", "bet_type", "segment"], kind="mergesort")
    segment_summary.to_csv(out_dir / "segment_summary.csv", index=False, encoding="utf-8-sig")

    profile_cols = [
        "race_id",
        "year",
        "ticket_type",
        "anchor_no",
        "anchor_name",
        "partner_no",
        "partner_name",
        "stake_yen",
        "return_yen",
        "runtime_action",
        "partner_bloodline_rescue_hi",
        "partner_bloodline_rescue_top",
        "anchor_bloodline_rescue_hi",
        "anchor_combined_pedigree_rescue_score",
        "partner_combined_pedigree_rescue_score",
        "anchor_low_career_pedigree_rescue_score",
        "partner_low_career_pedigree_rescue_score",
        "anchor_fc_condition_uncertainty_score",
        "partner_fc_condition_uncertainty_score",
    ]
    selected_tickets[[c for c in profile_cols if c in selected_tickets.columns]].to_csv(
        out_dir / "selected_ticket_bloodline_profile.csv", index=False, encoding="utf-8-sig"
    )

    buy = selected_summary[selected_summary["segment"].eq("runtime_buy_only")].iloc[0].to_dict()
    buy_partner_hi = selected_summary[selected_summary["segment"].eq("buy_partner_bloodline_hi")].iloc[0].to_dict()
    buy_partner_not_hi = selected_summary[selected_summary["segment"].eq("buy_partner_bloodline_not_hi")].iloc[0].to_dict()
    candidate_core = candidate_summary[
        candidate_summary["segment"].eq("quality_core") & candidate_summary["bet_type"].eq("umaren")
    ].iloc[0].to_dict()
    candidate_partner_hi = candidate_summary[
        candidate_summary["segment"].eq("partner_bloodline_hi_quality_core")
        & candidate_summary["bet_type"].eq("umaren")
    ].iloc[0].to_dict()
    candidate_core_wide = candidate_summary[
        candidate_summary["segment"].eq("quality_core") & candidate_summary["bet_type"].eq("wide")
    ].iloc[0].to_dict()
    candidate_partner_hi_wide = candidate_summary[
        candidate_summary["segment"].eq("partner_bloodline_hi_quality_core")
        & candidate_summary["bet_type"].eq("wide")
    ].iloc[0].to_dict()

    if buy_partner_hi["tickets"] >= 30 and buy_partner_hi["roi"] > buy_partner_not_hi["roi"] and buy_partner_hi["top10_removed_roi"] >= 1.0:
        decision = "shadow-positive"
        reason = "最強版BUY内では相手血統救済ありのROIが非救済を上回り、上位払戻除外後も100%を維持。"
        next_action = "本番採用ではなく、T-5/T-3スナップショットでシャドー昇格候補として継続監視。"
    elif candidate_partner_hi["tickets"] >= 50 and candidate_partner_hi["roi"] > candidate_core["roi"] and candidate_partner_hi["top10_removed_roi"] >= 1.0:
        decision = "candidate-shadow-only"
        reason = "候補宇宙では血統救済相手の改善余地があるが、最強版BUY内の優位が未確認。"
        next_action = "買い目拡張はせず、準候補台帳の分類タグとして残す。"
    else:
        decision = "do-not-adopt"
        reason = "血統救済相手が既存BUYや候補宇宙で安定した上乗せになっていない。"
        next_action = "血統はBUY拡張ではなく、経験不足馬の過小評価チェック・理由表示・シャドー検証に留める。"

    summary = {
        "runner_scores": str(project_path(args.runner_scores)),
        "pair_candidates": str(project_path(args.pair_candidates)),
        "selected_tickets": str(project_path(args.selected_tickets)),
        "output_dir": str(out_dir),
        "rows": {
            "runner_scores": int(len(runner_scores)),
            "pair_candidates": int(len(pair_candidates)),
            "selected_tickets": int(len(selected_tickets)),
        },
        "merge_rates": {
            "pair_anchor_score_rate": float(pair_candidates["anchor_combined_pedigree_rescue_score"].notna().mean()),
            "pair_partner_score_rate": float(pair_candidates["partner_combined_pedigree_rescue_score"].notna().mean()),
            "selected_anchor_score_rate": float(selected_tickets["anchor_combined_pedigree_rescue_score"].notna().mean()),
            "selected_partner_score_rate": float(selected_tickets["partner_combined_pedigree_rescue_score"].notna().mean()),
        },
        "bloodline_thresholds": pair_candidates.attrs.get("bloodline_thresholds", {}),
        "key_metrics": {
            "selected_buy": buy,
            "selected_buy_partner_bloodline_hi": buy_partner_hi,
            "selected_buy_partner_bloodline_not_hi": buy_partner_not_hi,
            "candidate_quality_core_umaren": candidate_core,
            "candidate_partner_bloodline_hi_quality_core_umaren": candidate_partner_hi,
            "candidate_quality_core_wide": candidate_core_wide,
            "candidate_partner_bloodline_hi_quality_core_wide": candidate_partner_hi_wide,
        },
        "verdict": {
            "decision": decision,
            "reason": reason,
            "next_action": next_action,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    write_review(out_dir, summary, segment_summary)

    print(json.dumps(json_ready(summary["verdict"]), ensure_ascii=False, indent=2))
    print(f"wrote: {out_dir}")


if __name__ == "__main__":
    main()
