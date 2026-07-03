from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_shape_adjusted_pair_selection import bool_col, ncol, ticket_return  # noqa: E402
from scripts.refine_retro_lap_partner_selection_v1 import (  # noqa: E402
    metrics as ticket_metrics,
    prepare_universe,
    price_sane_partner_base_pool,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "analysis" / "pair_covariance_synergy_v1"


def clip01(s: pd.Series | np.ndarray | float) -> pd.Series:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return pd.Series(s).clip(0.0, 1.0)


def add_covariance_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    label = out["queue_shape_label"].astype(str)
    front_min = clip01(ncol(out, "front_pair_min", 0.0))
    front_max = clip01(ncol(out, "front_pair_max", ncol(out, "projected_front5_prob", 0.0)))
    closer_max = clip01(ncol(out, "closer_pair_max", 0.0))
    front_closer = clip01(ncol(out, "front_closer_complement", 0.0))
    clash = clip01(ncol(out, "front_front_clash", 0.0))
    front_slow = clip01(ncol(out, "front_front_slow_fit", 0.0))
    collapse = clip01(ncol(out, "collapse_fit", 0.0))
    diversity = clip01(ncol(out, "style_diversity", 0.0))
    duel = clip01(ncol(out, "queue_duel_risk_score", 0.0))
    clarity = clip01(ncol(out, "queue_clarity_score", 0.0))
    shape_fit = clip01(ncol(out, "shape_pair_fit_score", 0.5))
    shape_risk = clip01(ncol(out, "shape_pair_risk_score", 0.0))
    joint = clip01(ncol(out, "joint_quality_score_v1", 0.5))
    retro = clip01(ncol(out, "retro_partner_support_v1", 0.5))

    single_fit = (
        0.36 * front_slow
        + 0.22 * front_max
        + 0.16 * front_min
        + 0.14 * (1.0 - clash)
        + 0.12 * shape_fit
    ).clip(0.0, 1.0)
    no_clear_fit = (
        0.28 * front_max
        + 0.22 * front_closer
        + 0.20 * diversity
        + 0.18 * shape_fit
        + 0.12 * (1.0 - clash)
    ).clip(0.0, 1.0)
    duel_fit = (
        0.32 * closer_max
        + 0.25 * collapse
        + 0.19 * diversity
        + 0.14 * front_closer
        + 0.10 * (1.0 - clash)
    ).clip(0.0, 1.0)
    matched_fit = (
        0.29 * closer_max
        + 0.24 * collapse
        + 0.22 * diversity
        + 0.15 * front_closer
        + 0.10 * front_max
    ).clip(0.0, 1.0)
    mixed_fit = (
        0.28 * diversity
        + 0.23 * front_closer
        + 0.21 * front_max
        + 0.17 * closer_max
        + 0.11 * shape_fit
    ).clip(0.0, 1.0)

    covariance_fit = pd.Series(0.5, index=out.index, dtype=float)
    covariance_fit = covariance_fit.where(~label.eq("single_leader_clear"), single_fit)
    covariance_fit = covariance_fit.where(~label.eq("no_clear_leader"), no_clear_fit)
    covariance_fit = covariance_fit.where(~label.eq("front_duel_dense"), duel_fit)
    covariance_fit = covariance_fit.where(~label.eq("matched_speed_duel"), matched_fit)
    covariance_fit = covariance_fit.where(~label.eq("mixed_queue"), mixed_fit)
    covariance_fit = covariance_fit.where(~label.eq("unknown"), mixed_fit)

    front_pair_burn = (clash * (0.45 + 0.55 * duel) * (0.45 + 0.55 * front_min)).clip(0.0, 1.0)
    same_role_fragile = ((1.0 - diversity) * (0.50 * duel + 0.25 * collapse + 0.25 * clarity)).clip(0.0, 1.0)
    dead_closer = ((1.0 - duel) * clarity * closer_max * (1.0 - front_max)).clip(0.0, 1.0)
    mutual_risk = np.maximum.reduce([front_pair_burn.to_numpy(), same_role_fragile.to_numpy(), dead_closer.to_numpy(), shape_risk.to_numpy()])
    mutual_risk = pd.Series(mutual_risk, index=out.index).clip(0.0, 1.0)

    out["pair_covariance_fit_score"] = covariance_fit
    out["pair_mutual_exclusion_risk_score"] = mutual_risk
    out["pair_style_complement_score"] = (0.45 * diversity + 0.35 * front_closer + 0.20 * (1.0 - clash)).clip(0.0, 1.0)
    out["pair_duel_closer_support_score"] = (0.50 * closer_max + 0.30 * collapse + 0.20 * diversity).clip(0.0, 1.0)
    out["pair_front_survival_support_score"] = (0.45 * front_slow + 0.30 * front_max + 0.25 * (1.0 - clash)).clip(0.0, 1.0)
    out["pair_covariance_score_v1"] = (
        0.42 * covariance_fit
        + 0.20 * shape_fit
        + 0.16 * joint
        + 0.10 * retro
        + 0.12 * (1.0 - mutual_risk)
    ).clip(0.0, 1.0)
    return out


@dataclass(frozen=True)
class CovPolicy:
    name: str
    score_func: Callable[[pd.DataFrame], pd.Series]
    pool_func: Callable[[pd.DataFrame], pd.Series]
    description: str


def quantile_thresholds(df: pd.DataFrame) -> dict[str, float]:
    pool = df[price_sane_partner_base_pool(df).fillna(False)].copy()
    cols = [
        "pair_covariance_score_v1",
        "pair_covariance_fit_score",
        "pair_mutual_exclusion_risk_score",
        "pair_style_complement_score",
        "pair_duel_closer_support_score",
        "pair_front_survival_support_score",
    ]
    out: dict[str, float] = {}
    for col in cols:
        x = pd.to_numeric(pool[col], errors="coerce").dropna()
        if x.empty:
            continue
        for q in [0.35, 0.45, 0.50, 0.60, 0.70, 0.75, 0.80]:
            out[f"{col}_q{int(q*100)}"] = float(x.quantile(q))
    return out


def make_policies(th: dict[str, float]) -> list[CovPolicy]:
    def pool_base(d: pd.DataFrame) -> pd.Series:
        return price_sane_partner_base_pool(d)

    def pool_cov_loose(d: pd.DataFrame) -> pd.Series:
        return (
            pool_base(d)
            & d["pair_covariance_score_v1"].ge(th.get("pair_covariance_score_v1_q45", 0.0))
            & d["pair_mutual_exclusion_risk_score"].le(th.get("pair_mutual_exclusion_risk_score_q80", 1.0))
        )

    def pool_cov_mid(d: pd.DataFrame) -> pd.Series:
        return (
            pool_base(d)
            & d["pair_covariance_score_v1"].ge(th.get("pair_covariance_score_v1_q50", 0.0))
            & d["pair_mutual_exclusion_risk_score"].le(th.get("pair_mutual_exclusion_risk_score_q75", 1.0))
        )

    def pool_cov_strict(d: pd.DataFrame) -> pd.Series:
        return (
            pool_base(d)
            & d["pair_covariance_score_v1"].ge(th.get("pair_covariance_score_v1_q60", 0.0))
            & d["pair_mutual_exclusion_risk_score"].le(th.get("pair_mutual_exclusion_risk_score_q70", 1.0))
        )

    def pool_no_mutual_exclusion(d: pd.DataFrame) -> pd.Series:
        return pool_base(d) & ~(
            d["pair_mutual_exclusion_risk_score"].ge(th.get("pair_mutual_exclusion_risk_score_q75", 1.0))
            & d["queue_shape_label"].astype(str).isin(["front_duel_dense", "matched_speed_duel", "mixed_queue"])
        )

    def pool_queue_specific(d: pd.DataFrame) -> pd.Series:
        label = d["queue_shape_label"].astype(str)
        front_race = label.isin(["single_leader_clear", "no_clear_leader"])
        duel_race = label.isin(["front_duel_dense", "matched_speed_duel"])
        mixed = label.eq("mixed_queue")
        support = (
            (front_race & d["pair_front_survival_support_score"].ge(th.get("pair_front_survival_support_score_q45", 0.0)))
            | (duel_race & d["pair_duel_closer_support_score"].ge(th.get("pair_duel_closer_support_score_q45", 0.0)))
            | (mixed & d["pair_style_complement_score"].ge(th.get("pair_style_complement_score_q45", 0.0)))
        )
        return pool_base(d) & support & d["pair_mutual_exclusion_risk_score"].le(th.get("pair_mutual_exclusion_risk_score_q80", 1.0))

    def score_base(d: pd.DataFrame) -> pd.Series:
        return d["segment_base_score"]

    def score_cov_light(d: pd.DataFrame) -> pd.Series:
        return 0.78 * d["segment_base_score"] + 0.22 * d["pair_covariance_score_v1"] - 0.05 * d["pair_mutual_exclusion_risk_score"]

    def score_cov_mid(d: pd.DataFrame) -> pd.Series:
        return 0.64 * d["segment_base_score"] + 0.36 * d["pair_covariance_score_v1"] - 0.08 * d["pair_mutual_exclusion_risk_score"]

    def score_cov_strong(d: pd.DataFrame) -> pd.Series:
        return 0.50 * d["segment_base_score"] + 0.50 * d["pair_covariance_score_v1"] - 0.12 * d["pair_mutual_exclusion_risk_score"]

    def score_queue_blend(d: pd.DataFrame) -> pd.Series:
        label = d["queue_shape_label"].astype(str)
        support = pd.Series(0.5, index=d.index)
        support = support.where(~label.isin(["single_leader_clear", "no_clear_leader"]), d["pair_front_survival_support_score"])
        support = support.where(~label.isin(["front_duel_dense", "matched_speed_duel"]), d["pair_duel_closer_support_score"])
        support = support.where(~label.eq("mixed_queue"), d["pair_style_complement_score"])
        return 0.67 * d["segment_base_score"] + 0.23 * support + 0.10 * d["pair_covariance_score_v1"] - 0.08 * d["pair_mutual_exclusion_risk_score"]

    return [
        CovPolicy("deployable_filter_base", score_base, pool_base, "事前弱条件除外後のベース"),
        CovPolicy("covariance_rerank_light", score_cov_light, pool_base, "ペア共分散を軽く加味して再ランキング"),
        CovPolicy("covariance_rerank_mid", score_cov_mid, pool_base, "ペア共分散を中程度加味して再ランキング"),
        CovPolicy("covariance_rerank_strong", score_cov_strong, pool_base, "ペア共分散を強めに加味して再ランキング"),
        CovPolicy("covariance_gate_loose", score_cov_light, pool_cov_loose, "共分散スコア下位/相互排除上位を軽く除外"),
        CovPolicy("covariance_gate_mid", score_cov_mid, pool_cov_mid, "共分散スコア下位/相互排除上位を中程度除外"),
        CovPolicy("covariance_gate_strict", score_cov_mid, pool_cov_strict, "共分散スコアと相互排除リスクを強めに見る"),
        CovPolicy("no_mutual_exclusion_gate", score_base, pool_no_mutual_exclusion, "同時好走を邪魔しやすい相互排除ペアだけ除外"),
        CovPolicy("queue_specific_synergy", score_queue_blend, pool_queue_specific, "隊列タイプごとに前/差し/補完の効き方を変える"),
    ]


def select_top_per_race(df: pd.DataFrame, policy: CovPolicy) -> pd.DataFrame:
    pool = df[policy.pool_func(df).fillna(False)].copy()
    if pool.empty:
        return pool
    pool["_cov_policy_score"] = policy.score_func(pool)
    selected = (
        pool.sort_values(["race_id", "_cov_policy_score", "pair_quinella_score", "market_overlay_score"], ascending=[True, False, False, False])
        .drop_duplicates("race_id", keep="first")
        .copy()
    )
    selected["policy"] = policy.name
    selected["description"] = policy.description
    selected["ticket_type"] = "umaren"
    selected["stake_yen"] = 100.0
    selected["return_yen"] = ticket_return(selected, "umaren")
    selected["hit"] = bool_col(selected, "umaren_hit")
    selected["anchor_top3"] = pd.to_numeric(selected["anchor_finish"], errors="coerce").le(3)
    selected["partner_top3"] = pd.to_numeric(selected["partner_finish"], errors="coerce").le(3)
    selected["both_top3"] = selected["anchor_top3"] & selected["partner_top3"]
    selected["ticket_key"] = selected.apply(lambda r: f"umaren:{r['race_id']}:{int(r['horse_a'])}-{int(r['horse_b'])}", axis=1)
    return selected


def policy_summary(df: pd.DataFrame, policies: list[CovPolicy]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selections = []
    summary_rows = []
    yearly_rows = []
    base_keys = None
    for policy in policies:
        selected = select_top_per_race(df, policy)
        if selected.empty:
            continue
        if policy.name == "deployable_filter_base":
            base_keys = selected[["race_id", "pair_key_norm"]].rename(columns={"pair_key_norm": "base_pair_key_norm"})
        selections.append(selected)
        row = ticket_metrics(selected, policy.name)
        row["description"] = policy.description
        row["avg_covariance"] = round(float(selected["pair_covariance_score_v1"].mean()), 3)
        row["avg_mutual_risk"] = round(float(selected["pair_mutual_exclusion_risk_score"].mean()), 3)
        summary_rows.append(row)
        for year, gy in selected.groupby("year"):
            yr = ticket_metrics(gy, policy.name)
            yr["year"] = int(year)
            yr["description"] = policy.description
            yr["avg_covariance"] = round(float(gy["pair_covariance_score_v1"].mean()), 3)
            yr["avg_mutual_risk"] = round(float(gy["pair_mutual_exclusion_risk_score"].mean()), 3)
            yearly_rows.append(yr)
    detail = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    if base_keys is not None and not detail.empty:
        detail = detail.merge(base_keys, on="race_id", how="left")
        detail["changed_from_base"] = detail["pair_key_norm"].ne(detail["base_pair_key_norm"])
        changed = detail[detail["changed_from_base"] & detail["policy"].ne("deployable_filter_base")].copy()
        for policy, gp in changed.groupby("policy"):
            ch = ticket_metrics(gp, f"{policy}_changed_only")
            ch["description"] = "ベースから相手差し替えになった分だけ"
            ch["avg_covariance"] = round(float(gp["pair_covariance_score_v1"].mean()), 3)
            ch["avg_mutual_risk"] = round(float(gp["pair_mutual_exclusion_risk_score"].mean()), 3)
            summary_rows.append(ch)
    summary = pd.DataFrame(summary_rows).sort_values(["roi_pct", "tickets"], ascending=[False, False])
    yearly = pd.DataFrame(yearly_rows).sort_values(["policy", "year"]) if yearly_rows else pd.DataFrame()
    return summary, yearly, detail


def component_segments(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("pair_covariance_score_v1", "high"),
        ("pair_covariance_fit_score", "high"),
        ("pair_mutual_exclusion_risk_score", "low"),
        ("pair_style_complement_score", "high"),
        ("pair_duel_closer_support_score", "high"),
        ("pair_front_survival_support_score", "high"),
        ("front_front_clash", "low"),
        ("style_diversity", "high"),
    ]
    for policy, base in selected.groupby("policy"):
        if policy != "deployable_filter_base":
            continue
        for col, direction in specs:
            if col not in base.columns:
                continue
            x = pd.to_numeric(base[col], errors="coerce")
            if x.notna().sum() < 20:
                continue
            q20, q40, q60, q80 = [float(x.quantile(q)) for q in [0.2, 0.4, 0.6, 0.8]]
            masks = {
                "bottom20": x.le(q20),
                "low40": x.le(q40),
                "mid40_60": x.between(q40, q60, inclusive="both"),
                "high40": x.ge(q60),
                "top20": x.ge(q80),
            }
            for bucket, mask in masks.items():
                sub = base[mask.fillna(False)].copy()
                if sub.empty:
                    continue
                row = ticket_metrics(sub, f"{col}_{bucket}")
                row["factor"] = col
                row["bucket"] = bucket
                row["direction"] = direction
                rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["factor", "roi_pct"], ascending=[True, False])
    return out


def same_pair_wide_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy, group in detail.groupby("policy"):
        stake = float(len(group) * 100.0)
        wide_hit = bool_col(group, "wide_hit")
        wide_return = pd.to_numeric(group["wide_pay"], errors="coerce").fillna(0.0).where(wide_hit, 0.0)
        rows.append(
            {
                "policy": policy,
                "tickets": int(len(group)),
                "wide_return_yen": round(float(wide_return.sum()), 1),
                "wide_roi_pct": round(float(wide_return.sum() / stake * 100), 1) if stake else 0.0,
                "wide_hit_rate_pct": round(float(wide_hit.mean() * 100), 1) if len(group) else 0.0,
                "umaren_roi_pct": round(float(pd.to_numeric(group["return_yen"], errors="coerce").fillna(0.0).sum() / stake * 100), 1)
                if stake
                else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["wide_roi_pct", "tickets"], ascending=[False, False])


def choose_policy(train: pd.DataFrame, policies: list[CovPolicy]) -> CovPolicy:
    rows = []
    for policy in policies:
        selected = select_top_per_race(train, policy)
        m = ticket_metrics(selected, policy.name)
        if m["tickets"] < 40:
            continue
        score = m["roi_pct"] - max(0.0, m["top_return_share_pct"] - 35.0) * 0.75 + m["both_top3_rate_pct"] * 0.25
        rows.append((score, m["roi_ex_top1_pct"], m["roi_pct"], policy.name, policy))
    if not rows:
        return policies[0]
    rows.sort(reverse=True)
    return rows[0][-1]


def walk_forward(df: pd.DataFrame, policies: list[CovPolicy]) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = sorted(df["month_block"].dropna().unique())
    rows = []
    min_train_months = 6
    for idx in range(min_train_months, len(months)):
        test_month = months[idx]
        train_months = months[: max(0, idx - 1)]
        train = df[df["month_block"].isin(train_months)]
        test = df[df["month_block"].eq(test_month)]
        chosen = choose_policy(train, policies)
        train_m = ticket_metrics(select_top_per_race(train, chosen), chosen.name)
        test_selected = select_top_per_race(test, chosen)
        test_m = ticket_metrics(test_selected, chosen.name)
        rows.append(
            {
                "test_month": test_month,
                "chosen_policy": chosen.name,
                "train_months": len(train_months),
                "train_roi_pct": train_m["roi_pct"],
                "train_tickets": train_m["tickets"],
                "test_tickets": test_m["tickets"],
                "test_roi_pct": test_m["roi_pct"],
                "test_profit_yen": test_m["profit_yen"],
                "test_hit_rate_pct": test_m["hit_rate_pct"],
                "test_both_top3_rate_pct": test_m["both_top3_rate_pct"],
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    stake = float(detail["test_tickets"].sum() * 100.0)
    ret = float((detail["test_profit_yen"] + detail["test_tickets"] * 100.0).sum())
    summary = pd.DataFrame(
        [
            {
                "months": int(len(detail)),
                "tickets": int(detail["test_tickets"].sum()),
                "return_yen": round(ret, 1),
                "profit_yen": round(ret - stake, 1),
                "roi_pct": round(ret / stake * 100, 1) if stake else 0.0,
                "positive_month_rate_pct": round(float(detail["test_profit_yen"].gt(0).mean() * 100), 1),
                "chosen_policy_counts": json.dumps(detail["chosen_policy"].value_counts().to_dict(), ensure_ascii=False),
            }
        ]
    )
    return detail, summary


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    work = df.replace({np.nan: ""})
    headers = [str(c) for c in work.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in work.to_numpy():
        lines.append("| " + " | ".join(str(v).replace("|", "/") for v in row) + " |")
    return "\n".join(lines)


def write_readme(
    summary: pd.DataFrame,
    yearly: pd.DataFrame,
    wf_summary: pd.DataFrame,
    segments: pd.DataFrame,
    wide_summary: pd.DataFrame,
) -> None:
    core = summary[~summary["policy"].astype(str).str.endswith("_changed_only")].head(10)
    lines = [
        "# Pair Covariance Synergy v1",
        "",
        "## Purpose",
        "馬連の相手選びを、単体能力ではなく2頭同時好走の相性で改善できるか検証する。",
        "",
        "## Fixed Policy Summary",
        md_table(
            core[
                [
                    "policy",
                    "tickets",
                    "roi_pct",
                    "hit_rate_pct",
                    "roi_ex_top1_pct",
                    "anchor_top3_rate_pct",
                    "partner_top3_rate_pct",
                    "both_top3_rate_pct",
                    "avg_covariance",
                    "avg_mutual_risk",
                ]
            ]
        ),
        "",
        "## Yearly Summary",
        md_table(
            yearly[
                [
                    "policy",
                    "year",
                    "tickets",
                    "roi_pct",
                    "hit_rate_pct",
                    "both_top3_rate_pct",
                    "avg_covariance",
                    "avg_mutual_risk",
                ]
            ]
        ),
        "",
        "## Walk-Forward Summary",
        md_table(wf_summary),
        "",
        "## Same Pair Wide Summary",
        md_table(
            wide_summary[
                [
                    "policy",
                    "tickets",
                    "wide_roi_pct",
                    "wide_hit_rate_pct",
                    "umaren_roi_pct",
                ]
            ].head(12)
        ),
        "",
        "## Component Segments On Base",
        md_table(
            segments[
                [
                    "factor",
                    "bucket",
                    "tickets",
                    "roi_pct",
                    "hit_rate_pct",
                    "both_top3_rate_pct",
                    "roi_ex_top1_pct",
                ]
            ].head(40)
        )
        if not segments.empty
        else "(no rows)",
        "",
        "## Interpretation",
        "- ROIが上がらない再ランキングは、相手を変えるほど悪化しやすい。",
        "- ゲートだけで改善する場合は、本番ではBUY昇格ではなく見送り/シャドーの追加理由にする。",
        "- `changed_only` が弱ければ、同一レース内の相手差し替えはまだ採用しない。",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = add_covariance_scores(prepare_universe())
    th = quantile_thresholds(df)
    policy_list = make_policies(th)
    summary, yearly, detail = policy_summary(df, policy_list)
    segments = component_segments(detail)
    wide_summary = same_pair_wide_summary(detail)
    wf_detail, wf_summary = walk_forward(df, policy_list)

    summary.to_csv(OUT_DIR / "pair_covariance_policy_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUT_DIR / "pair_covariance_policy_yearly.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(OUT_DIR / "pair_covariance_detail.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(OUT_DIR / "pair_covariance_component_segments.csv", index=False, encoding="utf-8-sig")
    wide_summary.to_csv(OUT_DIR / "pair_covariance_same_pair_wide_summary.csv", index=False, encoding="utf-8-sig")
    wf_detail.to_csv(OUT_DIR / "pair_covariance_walk_forward_detail.csv", index=False, encoding="utf-8-sig")
    wf_summary.to_csv(OUT_DIR / "pair_covariance_walk_forward_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "output_dir": str(OUT_DIR),
        "rows": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "thresholds": th,
        "top_fixed_policies": summary.head(14).replace({np.nan: None}).to_dict(orient="records"),
        "same_pair_wide_summary": wide_summary.head(10).replace({np.nan: None}).to_dict(orient="records"),
        "walk_forward_summary": wf_summary.replace({np.nan: None}).to_dict(orient="records"),
        "note": "Pre-race pair covariance test. No actual current race lap is used for policy scoring.",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(summary, yearly, wf_summary, segments, wide_summary)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
