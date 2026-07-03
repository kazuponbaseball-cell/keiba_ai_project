from __future__ import annotations

import itertools
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_shape_adjusted_pair_selection import (  # noqa: E402
    DEFAULT_RACE_SHAPE,
    DEFAULT_UNIVERSE,
    add_shape_scores,
    bool_col,
    gate_mask,
    load_universe,
    ncol,
    ticket_return,
)


ROOT = Path(__file__).resolve().parents[1]
RETRO_PAIR_PATH = ROOT / "outputs" / "analysis" / "retro_lap_adversity_v1" / "tickets_with_retro_lap_adversity.csv"
RACES_PATH = ROOT / "data" / "processed" / "normalized" / "races.csv"
OUT_DIR = ROOT / "outputs" / "analysis" / "retro_lap_partner_selection_v1"


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, **kwargs)


def clip01(s: pd.Series | np.ndarray | float) -> pd.Series:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    return pd.Series(s).clip(0.0, 1.0)


def norm01(s: pd.Series, lo: float | None = None, hi: float | None = None) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if lo is None:
        lo = float(x.quantile(0.05)) if x.notna().any() else 0.0
    if hi is None:
        hi = float(x.quantile(0.95)) if x.notna().any() else 1.0
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.5, index=x.index)
    return ((x.clip(lo, hi) - lo) / (hi - lo)).fillna(0.5).clip(0.0, 1.0)


def pair_key_norm(df: pd.DataFrame) -> pd.Series:
    a = pd.to_numeric(df["anchor_no"], errors="coerce")
    b = pd.to_numeric(df["partner_no"], errors="coerce")
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return df["race_id"].astype(str) + ":" + lo.astype("Int64").astype(str) + "-" + hi.astype("Int64").astype(str)


def distance_bucket(distance: pd.Series) -> pd.Series:
    d = pd.to_numeric(distance, errors="coerce")
    labels = pd.Series("unknown", index=distance.index, dtype=object)
    labels.loc[d.le(1199)] = "<=1199"
    labels.loc[d.between(1200, 1399)] = "1200-1399"
    labels.loc[d.between(1400, 1599)] = "1400-1599"
    labels.loc[d.between(1600, 1799)] = "1600-1799"
    labels.loc[d.between(1800, 1999)] = "1800-1999"
    labels.loc[d.between(2000, 2199)] = "2000-2199"
    labels.loc[d.ge(2200)] = "2200+"
    return labels


def pop_bucket(pop: pd.Series) -> pd.Series:
    p = pd.to_numeric(pop, errors="coerce")
    labels = pd.Series("unknown", index=pop.index, dtype=object)
    labels.loc[p.between(1, 1)] = "1人気"
    labels.loc[p.between(2, 3)] = "2-3人気"
    labels.loc[p.between(4, 5)] = "4-5人気"
    labels.loc[p.between(6, 8)] = "6-8人気"
    labels.loc[p.between(9, 12)] = "9-12人気"
    labels.loc[p.ge(13)] = "13人気+"
    return labels


def odds_bucket(x: pd.Series) -> pd.Series:
    v = pd.to_numeric(x, errors="coerce")
    labels = pd.Series("unknown", index=x.index, dtype=object)
    labels.loc[v.lt(2)] = "<2"
    labels.loc[v.between(2, 4, inclusive="left")] = "2-4"
    labels.loc[v.between(4, 7, inclusive="left")] = "4-7"
    labels.loc[v.between(7, 12, inclusive="left")] = "7-12"
    labels.loc[v.between(12, 25, inclusive="left")] = "12-25"
    labels.loc[v.between(25, 60, inclusive="left")] = "25-60"
    labels.loc[v.ge(60)] = "60+"
    return labels


def parse_date_from_race_id(race_id: pd.Series) -> pd.Series:
    return pd.to_datetime(race_id.astype(str).str.extract(r"(\d{8})", expand=False), format="%Y%m%d", errors="coerce")


def add_race_metadata(df: pd.DataFrame) -> pd.DataFrame:
    races = read_csv(RACES_PATH, dtype={"レースID(新/馬番無)": str})
    keep = ["レースID(新/馬番無)", "場所", "芝・ダ", "距離", "クラス名", "馬場状態"]
    races = races[[c for c in keep if c in races.columns]].drop_duplicates("レースID(新/馬番無)")
    races = races.rename(
        columns={
            "レースID(新/馬番無)": "race_id",
            "場所": "venue",
            "芝・ダ": "surface",
            "距離": "distance",
            "クラス名": "class_name",
            "馬場状態": "going",
        }
    )
    races["race_id"] = races["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    out = df.merge(races, on="race_id", how="left")
    out["distance_bucket"] = distance_bucket(out["distance"])
    out["partner_pop_bucket"] = pop_bucket(out["partner_pop"])
    out["anchor_pop_bucket"] = pop_bucket(out["anchor_pop"])
    if "odds_geom" in out.columns:
        out["odds_geom_bucket"] = odds_bucket(out["odds_geom"])
    else:
        geom = np.sqrt(pd.to_numeric(out["anchor_odds"], errors="coerce") * pd.to_numeric(out["partner_odds"], errors="coerce"))
        out["odds_geom_bucket"] = odds_bucket(geom)
    out["race_date"] = parse_date_from_race_id(out["race_id"])
    out["month_block"] = out["race_date"].dt.to_period("M").astype(str)
    return out


def merge_retro_features(df: pd.DataFrame) -> pd.DataFrame:
    retro_cols = [
        "race_id",
        "pair_key_norm",
        "retro_lap_pair_fit_score",
        "retro_lap_pair_risk_score",
        "retro_lap_pair_pos_max",
        "retro_lap_pair_pos_avg",
        "retro_lap_pair_negative_max",
        "retro_lap_pair_overhelped_max",
        "retro_lap_pair_evidence_min",
    ]
    retro = read_csv(RETRO_PAIR_PATH, dtype={"race_id": str}, usecols=lambda c: c in retro_cols)
    retro["race_id"] = retro["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    return df.merge(retro.drop_duplicates(["race_id", "pair_key_norm"]), on=["race_id", "pair_key_norm"], how="left")


def prepare_universe() -> pd.DataFrame:
    df = load_universe(DEFAULT_UNIVERSE, DEFAULT_RACE_SHAPE)
    df = add_shape_scores(df)
    df["pair_key_norm"] = pair_key_norm(df)
    df = merge_retro_features(df)
    df = add_race_metadata(df)
    q = ncol(df, "pair_quinella_score", 0.0).clip(0.0, 1.0)
    q_rank = q.groupby(df["race_id"]).rank(pct=True).fillna(0.5)
    df["segment_base_score"] = (
        0.15 * q_rank
        + 0.85 * ncol(df, "shape_pair_fit_score", 0.5).clip(0.0, 1.0)
        + 0.06 * ncol(df, "shape_value_score", 0.0).clip(0.0, 1.0)
        - 0.12 * ncol(df, "shape_pair_risk_score", 0.0).clip(0.0, 1.0)
    )
    partner_quality = (
        0.25 * clip01(df["partner_place_score"])
        + 0.24 * clip01(df["partner_quinella_score"])
        + 0.16 * clip01(df["partner_quinella_model_score_norm"])
        + 0.13 * clip01(df["wide_partner_score"])
        + 0.12 * clip01(df["partner_front5_prob_v2"] if "partner_front5_prob_v2" in df.columns else df["projected_front5_prob"])
        + 0.10 * (1.0 - clip01(df["partner_danger"]))
    ).clip(0.0, 1.0)
    joint_quality = (
        0.23 * clip01(df["pair_quinella_score"])
        + 0.20 * clip01(df["pair_score"])
        + 0.17 * clip01(df["joint_place_product"])
        + 0.12 * clip01(df["joint_q_product"])
        + 0.10 * clip01(df["market_overlay_score"])
        + 0.08 * clip01(df["late_value_survives_score"])
        + 0.10 * (1.0 - clip01(df["danger_sum"]))
    ).clip(0.0, 1.0)
    partner_rank = (
        0.50 * df["partner_quinella_score"].groupby(df["race_id"]).rank(pct=True).fillna(0.5)
        + 0.30 * df["partner_place_score"].groupby(df["race_id"]).rank(pct=True).fillna(0.5)
        + 0.20 * df["wide_partner_score"].groupby(df["race_id"]).rank(pct=True).fillna(0.5)
    ).clip(0.0, 1.0)
    df["partner_quality_score_v1"] = partner_quality
    df["joint_quality_score_v1"] = joint_quality
    df["partner_rank_score_v1"] = partner_rank
    df["partner_safety_score_v1"] = (1.0 - (0.70 * clip01(df["partner_danger"]) + 0.30 * clip01(df["danger_sum"]))).clip(0.0, 1.0)
    df["retro_partner_support_v1"] = (
        0.55 * clip01(df["retro_lap_pair_fit_score"])
        + 0.25 * clip01(df["retro_lap_pair_pos_max"])
        + 0.20 * (1.0 - clip01(df["retro_lap_pair_risk_score"]))
    ).clip(0.0, 1.0)
    return df


def retro_low_risk_mask(df: pd.DataFrame) -> pd.Series:
    valid = df[ncol(df, "retro_lap_pair_evidence_min", 0.0).gt(0)]
    threshold = float(valid["retro_lap_pair_risk_score"].quantile(0.60)) if not valid.empty else 0.0
    return ncol(df, "retro_lap_pair_risk_score", 1.0).le(threshold)


def deployable_weak_core_keep(df: pd.DataFrame) -> pd.Series:
    surface = df["surface"].astype(str)
    venue = df["venue"].astype(str)
    cls = df["class_name"].astype(str)
    dist = df["distance_bucket"].astype(str)
    partner_pop = df["partner_pop_bucket"].astype(str)
    odds_geom = df["odds_geom_bucket"].astype(str)
    queue = df["queue_shape_label"].astype(str)
    weak_core = (
        (surface.eq("ダ") & queue.eq("front_duel_dense"))
        | (queue.eq("mixed_queue") & partner_pop.eq("6-8人気"))
        | (surface.eq("芝") & dist.eq("2000-2199"))
        | (surface.eq("芝") & cls.eq("1勝") & odds_geom.eq("12-25"))
        | (venue.eq("阪神") & surface.eq("ダ"))
        | dist.isin(["2000-2199", "<=1199"])
    )
    return ~weak_core.fillna(False)


def price_sane_partner_base_pool(df: pd.DataFrame) -> pd.Series:
    return gate_mask(df, "price_sane_strong").fillna(False) & retro_low_risk_mask(df).fillna(False) & deployable_weak_core_keep(df).fillna(False)


@dataclass(frozen=True)
class PartnerPolicy:
    name: str
    score_func: Callable[[pd.DataFrame], pd.Series]
    pool_func: Callable[[pd.DataFrame], pd.Series]
    description: str


def policies() -> list[PartnerPolicy]:
    def pool_base(d: pd.DataFrame) -> pd.Series:
        return gate_mask(d, "price_sane_strong").fillna(False) & retro_low_risk_mask(d).fillna(False)

    def pool_deploy(d: pd.DataFrame) -> pd.Series:
        return price_sane_partner_base_pool(d)

    def pool_deploy_quality(d: pd.DataFrame) -> pd.Series:
        return pool_deploy(d) & d["partner_quality_score_v1"].ge(0.52) & d["partner_safety_score_v1"].ge(0.64)

    def pool_deploy_quality_strict(d: pd.DataFrame) -> pd.Series:
        return pool_deploy(d) & d["partner_quality_score_v1"].ge(0.58) & d["partner_safety_score_v1"].ge(0.68)

    def score_base(d: pd.DataFrame) -> pd.Series:
        return d["segment_base_score"]

    def score_quality(d: pd.DataFrame) -> pd.Series:
        return (
            0.50 * d["segment_base_score"]
            + 0.20 * d["joint_quality_score_v1"]
            + 0.18 * d["partner_quality_score_v1"]
            + 0.07 * d["retro_partner_support_v1"]
            + 0.05 * d["partner_rank_score_v1"]
        )

    def score_safety(d: pd.DataFrame) -> pd.Series:
        return (
            0.45 * d["segment_base_score"]
            + 0.24 * d["joint_quality_score_v1"]
            + 0.16 * d["partner_quality_score_v1"]
            + 0.10 * d["partner_safety_score_v1"]
            + 0.05 * d["retro_partner_support_v1"]
        )

    def score_rank(d: pd.DataFrame) -> pd.Series:
        return (
            0.42 * d["segment_base_score"]
            + 0.24 * d["partner_rank_score_v1"]
            + 0.16 * d["joint_quality_score_v1"]
            + 0.12 * d["partner_quality_score_v1"]
            + 0.06 * d["retro_partner_support_v1"]
        )

    return [
        PartnerPolicy("retro_low_risk_base", score_base, pool_base, "retro低リスク内で従来shape寄りスコア"),
        PartnerPolicy("deployable_filter_base", score_base, pool_deploy, "事前弱条件除外後、従来shape寄りスコア"),
        PartnerPolicy("partner_quality_rerank", score_quality, pool_deploy, "相手品質/連対寄りスコアを加味して再ランキング"),
        PartnerPolicy("partner_safety_rerank", score_safety, pool_deploy, "相手品質と危険度低さを加味して再ランキング"),
        PartnerPolicy("partner_rank_rerank", score_rank, pool_deploy, "レース内で相手上位度を強めて再ランキング"),
        PartnerPolicy("partner_quality_gate", score_quality, pool_deploy_quality, "相手品質/安全度の最低ラインも要求"),
        PartnerPolicy("partner_quality_gate_strict", score_quality, pool_deploy_quality_strict, "相手品質/安全度の最低ラインを強めに要求"),
    ]


def max_drawdown(profit: pd.Series) -> float:
    if profit.empty:
        return 0.0
    equity = profit.cumsum()
    peak = equity.cummax()
    return float((peak - equity).max())


def select_top_per_race(df: pd.DataFrame, policy: PartnerPolicy) -> pd.DataFrame:
    pool = df[policy.pool_func(df).fillna(False)].copy()
    if pool.empty:
        return pool
    pool["_partner_policy_score"] = policy.score_func(pool)
    selected = (
        pool.sort_values(["race_id", "_partner_policy_score", "joint_quality_score_v1", "market_overlay_score"], ascending=[True, False, False, False])
        .drop_duplicates("race_id", keep="first")
        .copy()
    )
    selected["policy"] = policy.name
    selected["policy_description"] = policy.description
    selected["ticket_type"] = "umaren"
    selected["stake_yen"] = 100.0
    selected["return_yen"] = ticket_return(selected, "umaren")
    selected["hit"] = bool_col(selected, "umaren_hit")
    selected["anchor_top3"] = pd.to_numeric(selected["anchor_finish"], errors="coerce").le(3)
    selected["partner_top3"] = pd.to_numeric(selected["partner_finish"], errors="coerce").le(3)
    selected["both_top3"] = selected["anchor_top3"] & selected["partner_top3"]
    selected["ticket_key"] = selected.apply(lambda r: f"umaren:{r['race_id']}:{int(r['horse_a'])}-{int(r['horse_b'])}", axis=1)
    return selected


def metrics(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {
            "policy": label,
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
            "max_drawdown_yen": 0.0,
        }
    stake = float(df["stake_yen"].sum())
    ret = float(df["return_yen"].sum())
    top = float(df["return_yen"].max())
    ex = df.drop(df["return_yen"].idxmax()) if top > 0 else df.iloc[0:0]
    ex_stake = float(ex["stake_yen"].sum()) if not ex.empty else 0.0
    ex_ret = float(ex["return_yen"].sum()) if not ex.empty else 0.0
    return {
        "policy": label,
        "tickets": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "stake_yen": round(stake, 1),
        "return_yen": round(ret, 1),
        "profit_yen": round(ret - stake, 1),
        "roi_pct": round(ret / stake * 100, 1) if stake else 0.0,
        "hit_rate_pct": round(float(df["hit"].mean() * 100), 1),
        "roi_ex_top1_pct": round(ex_ret / ex_stake * 100, 1) if ex_stake else 0.0,
        "top_return_share_pct": round(top / ret * 100, 1) if ret else 0.0,
        "anchor_top3_rate_pct": round(float(df["anchor_top3"].mean() * 100), 1),
        "partner_top3_rate_pct": round(float(df["partner_top3"].mean() * 100), 1),
        "both_top3_rate_pct": round(float(df["both_top3"].mean() * 100), 1),
        "max_drawdown_yen": round(max_drawdown(df.sort_values(["race_date", "race_id"])["return_yen"] - df.sort_values(["race_date", "race_id"])["stake_yen"]), 1),
    }


def fixed_policy_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows = []
    summary_rows = []
    year_rows = []
    baseline_keys = None
    for policy in policies():
        selected = select_top_per_race(df, policy)
        if selected.empty:
            continue
        if policy.name == "deployable_filter_base":
            baseline_keys = selected[["race_id", "pair_key_norm"]].rename(columns={"pair_key_norm": "deployable_base_pair"})
        selected_rows.append(selected)
        row = metrics(selected, policy.name)
        row["description"] = policy.description
        summary_rows.append(row)
        for year, gy in selected.groupby("year"):
            yr = metrics(gy, policy.name)
            yr["year"] = int(year)
            yr["description"] = policy.description
            year_rows.append(yr)
    detail = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    if baseline_keys is not None and not detail.empty:
        detail = detail.merge(baseline_keys, on="race_id", how="left")
        detail["changed_from_deployable_base"] = detail["pair_key_norm"].ne(detail["deployable_base_pair"])
        changed = detail[detail["changed_from_deployable_base"] & detail["policy"].ne("deployable_filter_base")].copy()
        for policy, gp in changed.groupby("policy"):
            c = metrics(gp, f"{policy}_changed_only")
            c["description"] = "deployable_filter_baseから相手差し替えになった分だけ"
            summary_rows.append(c)
    summary = pd.DataFrame(summary_rows).sort_values(["roi_pct", "tickets"], ascending=[False, False])
    yearly = pd.DataFrame(year_rows).sort_values(["policy", "year"]) if year_rows else pd.DataFrame()
    return summary, yearly, detail


def choose_policy(train: pd.DataFrame) -> PartnerPolicy:
    rows = []
    for policy in policies():
        if policy.name == "retro_low_risk_base":
            continue
        selected = select_top_per_race(train, policy)
        m = metrics(selected, policy.name)
        if m["tickets"] < 40:
            continue
        score = m["roi_pct"] - max(0.0, m["top_return_share_pct"] - 35.0) * 0.7 + m["partner_top3_rate_pct"] * 0.15
        rows.append((score, m["roi_ex_top1_pct"], m["roi_pct"], policy.name, policy))
    if not rows:
        return policies()[1]
    rows.sort(reverse=True)
    return rows[0][-1]


def walk_forward(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = sorted(df["month_block"].dropna().unique())
    rows = []
    min_train_months = 6
    for idx in range(min_train_months, len(months)):
        test_month = months[idx]
        train_months = months[: max(0, idx - 1)]
        train = df[df["month_block"].isin(train_months)]
        test = df[df["month_block"].eq(test_month)]
        chosen = choose_policy(train)
        train_m = metrics(select_top_per_race(train, chosen), chosen.name)
        selected = select_top_per_race(test, chosen)
        test_m = metrics(selected, chosen.name)
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
                "test_partner_top3_rate_pct": test_m["partner_top3_rate_pct"],
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
    cols = [str(c) for c in work.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in work.to_numpy():
        lines.append("| " + " | ".join(str(v).replace("|", "/") for v in row) + " |")
    return "\n".join(lines)


def write_readme(summary: pd.DataFrame, yearly: pd.DataFrame, wf_summary: pd.DataFrame) -> None:
    top = summary[~summary["policy"].astype(str).str.endswith("_changed_only")].head(8)
    lines = [
        "# Retro Lap Partner Selection v1",
        "",
        "## Purpose",
        "retro低リスク馬連について、同一レース内で相手馬を選び直すとROI/相手3着内率が改善するか検証する。",
        "",
        "## Fixed Policy Summary",
        md_table(
            top[
                [
                    "policy",
                    "tickets",
                    "roi_pct",
                    "hit_rate_pct",
                    "roi_ex_top1_pct",
                    "anchor_top3_rate_pct",
                    "partner_top3_rate_pct",
                    "both_top3_rate_pct",
                    "max_drawdown_yen",
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
                    "anchor_top3_rate_pct",
                    "partner_top3_rate_pct",
                    "both_top3_rate_pct",
                ]
            ]
        ),
        "",
        "## Walk-Forward Summary",
        md_table(wf_summary),
        "",
        "## Interpretation",
        "- `retro_low_risk_base` は前回の低リスク馬連セグメント相当。",
        "- `deployable_filter_base` はレース後ラップを使わない弱条件除外後のベース。",
        "- `partner_*` は相手馬の連対/複勝寄り品質、危険度、隊列相性を足して同一レース内で相手を選び直す。",
        "- 2026年序盤の崩れが改善するかを必ず年別で確認する。",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare_universe()
    summary, yearly, detail = fixed_policy_summary(df)
    wf_detail, wf_summary = walk_forward(df)

    summary.to_csv(OUT_DIR / "partner_selection_policy_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(OUT_DIR / "partner_selection_policy_yearly.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(OUT_DIR / "partner_selection_detail.csv", index=False, encoding="utf-8-sig")
    wf_detail.to_csv(OUT_DIR / "partner_selection_walk_forward_detail.csv", index=False, encoding="utf-8-sig")
    wf_summary.to_csv(OUT_DIR / "partner_selection_walk_forward_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "output_dir": str(OUT_DIR),
        "rows": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "top_fixed_policies": summary.head(12).replace({np.nan: None}).to_dict(orient="records"),
        "walk_forward_summary": wf_summary.replace({np.nan: None}).to_dict(orient="records"),
        "note": "Pre-race deployable partner re-ranking test. actual_lap_regime is not used in policy scoring.",
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(summary, yearly, wf_summary)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
