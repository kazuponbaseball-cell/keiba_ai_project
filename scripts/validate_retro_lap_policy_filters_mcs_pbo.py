from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BREAKDOWN_DIR = ROOT / "outputs" / "analysis" / "retro_lap_adversity_v1" / "breakdown_v1"
IN_PATH = BREAKDOWN_DIR / "selected_low_risk_umaren.csv"
OUT_DIR = BREAKDOWN_DIR / "policy_filter_v1" / "mcs_pbo_v1"


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)


def as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["race_id"] = out["race_id"].astype(str)
    out["race_date"] = pd.to_datetime(out["race_date"], errors="coerce")
    missing = out["race_date"].isna()
    if missing.any():
        inferred = out.loc[missing, "race_id"].astype(str).str.extract(r"(\d{8})", expand=False)
        out.loc[missing, "race_date"] = pd.to_datetime(inferred, format="%Y%m%d", errors="coerce")
    out = out[out["race_date"].notna()].copy()
    out["year"] = out["race_date"].dt.year
    out["month_block"] = out["race_date"].dt.to_period("M").astype(str)
    out["stake_yen"] = num(out, "stake_yen", 100.0).fillna(100.0)
    out["return_yen"] = num(out, "return_yen", 0.0).fillna(0.0)
    out["profit_yen"] = out["return_yen"] - out["stake_yen"]
    out["hit_bool"] = as_bool(out["hit"]) | out["return_yen"].gt(0)
    out["_order"] = np.arange(len(out))
    return out.sort_values(["race_date", "race_id", "_order"]).reset_index(drop=True)


def max_drawdown(profits: pd.Series) -> float:
    if profits.empty:
        return 0.0
    equity = profits.cumsum()
    peak = equity.cummax()
    return float((peak - equity).max())


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
            "max_drawdown_yen": 0.0,
        }
    stake = float(df["stake_yen"].sum())
    ret = float(df["return_yen"].sum())
    top_ret = float(df["return_yen"].max())
    ex = df.drop(df["return_yen"].idxmax()) if top_ret > 0 else df.iloc[0:0]
    ex_stake = float(ex["stake_yen"].sum()) if not ex.empty else 0.0
    ex_ret = float(ex["return_yen"].sum()) if not ex.empty else 0.0
    return {
        "tickets": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "stake_yen": round(stake, 1),
        "return_yen": round(ret, 1),
        "profit_yen": round(ret - stake, 1),
        "roi_pct": round(ret / stake * 100, 1) if stake else 0.0,
        "hit_rate_pct": round(float(df["hit_bool"].mean() * 100), 1),
        "roi_ex_top1_pct": round(ex_ret / ex_stake * 100, 1) if ex_stake else 0.0,
        "top_return_share_pct": round(top_ret / ret * 100, 1) if ret else 0.0,
        "max_drawdown_yen": round(max_drawdown(df["profit_yen"]), 1),
    }


@dataclass(frozen=True)
class Policy:
    name: str
    uses_post_race_lap: bool
    description: str

    def mask(self, df: pd.DataFrame) -> pd.Series:
        true = pd.Series(True, index=df.index)
        surface = df["surface"].astype(str)
        venue = df["venue"].astype(str)
        cls = df["class_name"].astype(str)
        dist = df["distance_bucket"].astype(str)
        partner_pop = df["partner_pop_bucket"].astype(str)
        odds_geom = df["odds_geom_bucket"].astype(str)
        queue = df["queue_shape_label"].astype(str)
        lap = df["actual_lap_regime"].astype(str)

        weak_core_actual = (
            (surface.eq("ダ") & queue.eq("front_duel_dense") & lap.eq("front_loaded"))
            | (queue.eq("mixed_queue") & lap.eq("front_loaded") & partner_pop.eq("6-8人気"))
            | (surface.eq("芝") & dist.eq("2000-2199"))
            | (surface.eq("芝") & cls.eq("1勝") & odds_geom.eq("12-25"))
            | (venue.eq("阪神") & surface.eq("ダ"))
        )
        weak_core_prerace = (
            (surface.eq("ダ") & queue.eq("front_duel_dense"))
            | (queue.eq("mixed_queue") & partner_pop.eq("6-8人気"))
            | (surface.eq("芝") & dist.eq("2000-2199"))
            | (surface.eq("芝") & cls.eq("1勝") & odds_geom.eq("12-25"))
            | (venue.eq("阪神") & surface.eq("ダ"))
        )

        if self.name == "no_filter":
            return true
        if self.name == "exclude_2000_2199_and_under1199":
            return ~dist.isin(["2000-2199", "<=1199"])
        if self.name == "exclude_longshot_tail":
            return ~(partner_pop.eq("13人気+") | odds_geom.eq("25-60"))
        if self.name == "exclude_all_period_weak_core_actual_lap":
            return ~weak_core_actual
        if self.name == "exclude_core_plus_bad_distance_actual_lap":
            return ~(weak_core_actual | dist.isin(["2000-2199", "<=1199"]))
        if self.name == "exclude_2026_bad_looking_not_recommended":
            return ~(
                venue.eq("小倉")
                | (surface.eq("ダ") & cls.eq("1勝") & dist.eq("1800-1999"))
                | partner_pop.eq("9-12人気")
            )
        if self.name == "exclude_prerace_weak_core":
            return ~weak_core_prerace
        if self.name == "exclude_prerace_core_plus_bad_distance":
            return ~(weak_core_prerace | dist.isin(["2000-2199", "<=1199"]))
        if self.name == "exclude_prerace_distance_and_tail":
            return ~(dist.isin(["2000-2199", "<=1199"]) | partner_pop.eq("13人気+") | odds_geom.eq("25-60"))
        raise ValueError(f"unknown policy: {self.name}")


POLICIES = [
    Policy("no_filter", False, "現行のretro低リスク馬連をそのまま買う"),
    Policy("exclude_2000_2199_and_under1199", False, "短距離端と2000-2199mを除外"),
    Policy("exclude_longshot_tail", False, "相手13人気以上と幾何オッズ25-60を除外"),
    Policy("exclude_all_period_weak_core_actual_lap", True, "弱い中核条件を除外。ただし実レース後ラップを含む診断用"),
    Policy("exclude_core_plus_bad_distance_actual_lap", True, "弱い中核条件に距離端を足して除外。ただし診断用"),
    Policy("exclude_2026_bad_looking_not_recommended", False, "2026悪化っぽい条件を除外。過剰反応確認用"),
    Policy("exclude_prerace_weak_core", False, "actual_lapを使わず、事前に分かる弱い中核条件だけ除外"),
    Policy("exclude_prerace_core_plus_bad_distance", False, "事前弱中核条件に距離端を足して除外"),
    Policy("exclude_prerace_distance_and_tail", False, "距離端と人気/オッズ尻尾を除外する事前ルール"),
]


def apply_policy(df: pd.DataFrame, policy: Policy) -> pd.DataFrame:
    return df[policy.mask(df).fillna(False)].copy()


def policy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy in POLICIES:
        sub = apply_policy(df, policy)
        row = metrics(sub)
        row.update(
            {
                "policy": policy.name,
                "uses_post_race_lap": policy.uses_post_race_lap,
                "description": policy.description,
                "kept_share_pct": round(float(policy.mask(df).fillna(False).mean() * 100), 1),
            }
        )
        rows.append(row)
        for year, g in sub.groupby("year", sort=True):
            y = metrics(g)
            y.update(
                {
                    "policy": policy.name,
                    "year": int(year),
                    "uses_post_race_lap": policy.uses_post_race_lap,
                    "description": policy.description,
                    "kept_share_pct": round(float(policy.mask(df[df["year"].eq(year)]).fillna(False).mean() * 100), 1),
                }
            )
            rows.append(y)
    return pd.DataFrame(rows)


def monthly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for policy in POLICIES:
        sub = apply_policy(df, policy)
        for month, g in sub.groupby("month_block", sort=True):
            row = metrics(g)
            row.update(
                {
                    "policy": policy.name,
                    "month_block": month,
                    "uses_post_race_lap": policy.uses_post_race_lap,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def choose_policy(train: pd.DataFrame, policies: list[Policy]) -> Policy:
    rows = []
    for policy in policies:
        sub = apply_policy(train, policy)
        m = metrics(sub)
        if m["tickets"] < 50:
            continue
        # Penalize extremely top-heavy rules during selection.
        score = m["roi_pct"] - max(0.0, m["top_return_share_pct"] - 45.0) * 0.6
        rows.append((score, m["roi_ex_top1_pct"], m["roi_pct"], policy.name, policy))
    if not rows:
        return policies[0]
    rows.sort(reverse=True)
    return rows[0][-1]


def walk_forward(df: pd.DataFrame, deployable_only: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    policies = [p for p in POLICIES if (not deployable_only or not p.uses_post_race_lap)]
    months = sorted(df["month_block"].unique())
    rows = []
    min_train_months = 6
    for idx in range(min_train_months, len(months)):
        test_month = months[idx]
        # One-month embargo before the test month to avoid reacting to very recent noise.
        train_months = months[: max(0, idx - 1)]
        train = df[df["month_block"].isin(train_months)]
        test = df[df["month_block"].eq(test_month)]
        chosen = choose_policy(train, policies)
        train_m = metrics(apply_policy(train, chosen))
        test_sub = apply_policy(test, chosen)
        test_m = metrics(test_sub)
        row = {
            "mode": "deployable_only" if deployable_only else "all_policies",
            "test_month": test_month,
            "chosen_policy": chosen.name,
            "chosen_uses_post_race_lap": chosen.uses_post_race_lap,
            "train_months": len(train_months),
            "train_roi_pct": train_m["roi_pct"],
            "train_tickets": train_m["tickets"],
            "test_roi_pct": test_m["roi_pct"],
            "test_tickets": test_m["tickets"],
            "test_return_yen": test_m["return_yen"],
            "test_profit_yen": test_m["profit_yen"],
            "test_hit_rate_pct": test_m["hit_rate_pct"],
        }
        rows.append(row)
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    summary_rows = []
    for mode, g in detail.groupby("mode"):
        stake = float(g["test_tickets"].sum() * 100.0)
        ret = float(g["test_return_yen"].sum())
        summary_rows.append(
            {
                "mode": mode,
                "months": int(len(g)),
                "tickets": int(g["test_tickets"].sum()),
                "return_yen": round(ret, 1),
                "profit_yen": round(ret - stake, 1),
                "roi_pct": round(ret / stake * 100, 1) if stake else 0.0,
                "positive_month_rate_pct": round(float(g["test_profit_yen"].gt(0).mean() * 100), 1),
                "chosen_policy_counts": json.dumps(g["chosen_policy"].value_counts().to_dict(), ensure_ascii=False),
            }
        )
    return detail, pd.DataFrame(summary_rows)


def assign_equal_time_blocks(df: pd.DataFrame, n_blocks: int = 8) -> pd.Series:
    dates = df[["race_date"]].drop_duplicates().sort_values("race_date").reset_index(drop=True)
    dates["block"] = pd.qcut(dates.index, q=n_blocks, labels=False, duplicates="drop")
    mapper = dates.set_index("race_date")["block"].to_dict()
    return df["race_date"].map(mapper).astype(int)


def pbo_analysis(df: pd.DataFrame, deployable_only: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    policies = [p for p in POLICIES if (not deployable_only or not p.uses_post_race_lap)]
    work = df.copy()
    work["pbo_block"] = assign_equal_time_blocks(work, 8)
    blocks = sorted(work["pbo_block"].unique())
    half = len(blocks) // 2
    rows = []
    for split_id, test_blocks in enumerate(itertools.combinations(blocks, half), start=1):
        test_blocks = set(test_blocks)
        train = work[~work["pbo_block"].isin(test_blocks)]
        test = work[work["pbo_block"].isin(test_blocks)]
        train_scores = []
        test_scores = []
        for policy in policies:
            train_m = metrics(apply_policy(train, policy))
            test_m = metrics(apply_policy(test, policy))
            train_scores.append((train_m["roi_pct"], train_m["tickets"], policy.name, policy))
            test_scores.append({"policy": policy.name, "roi_pct": test_m["roi_pct"], "tickets": test_m["tickets"]})
        train_scores = [r for r in train_scores if r[1] >= 50]
        if not train_scores:
            continue
        train_scores.sort(reverse=True)
        selected = train_scores[0][3]
        test_rank = pd.DataFrame(test_scores).sort_values("roi_pct", ascending=False).reset_index(drop=True)
        selected_row = test_rank[test_rank["policy"].eq(selected.name)].iloc[0]
        rank_pos = int(test_rank.index[test_rank["policy"].eq(selected.name)][0]) + 1
        rank_pct = 1.0 - ((rank_pos - 1) / max(1, len(test_rank) - 1))
        selected_test_m = metrics(apply_policy(test, selected))
        no_filter_test_m = metrics(apply_policy(test, POLICIES[0]))
        rows.append(
            {
                "mode": "deployable_only" if deployable_only else "all_policies",
                "split_id": split_id,
                "test_blocks": ",".join(map(str, sorted(test_blocks))),
                "selected_policy": selected.name,
                "selected_uses_post_race_lap": selected.uses_post_race_lap,
                "train_roi_pct": train_scores[0][0],
                "test_roi_pct": selected_test_m["roi_pct"],
                "test_tickets": selected_test_m["tickets"],
                "test_profit_yen": selected_test_m["profit_yen"],
                "test_rank_pct": round(rank_pct, 4),
                "test_under_median": bool(rank_pct < 0.5),
                "test_under_no_filter": bool(selected_test_m["roi_pct"] < no_filter_test_m["roi_pct"]),
                "no_filter_test_roi_pct": no_filter_test_m["roi_pct"],
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    summary_rows = []
    for mode, g in detail.groupby("mode"):
        summary_rows.append(
            {
                "mode": mode,
                "splits": int(len(g)),
                "avg_test_roi_pct": round(float(g["test_roi_pct"].mean()), 1),
                "median_test_roi_pct": round(float(g["test_roi_pct"].median()), 1),
                "pbo_under_median_pct": round(float(g["test_under_median"].mean() * 100), 1),
                "under_no_filter_pct": round(float(g["test_under_no_filter"].mean() * 100), 1),
                "positive_profit_split_pct": round(float(g["test_profit_yen"].gt(0).mean() * 100), 1),
                "selected_policy_counts": json.dumps(g["selected_policy"].value_counts().to_dict(), ensure_ascii=False),
            }
        )
    return detail, pd.DataFrame(summary_rows)


def bootstrap_months(df: pd.DataFrame, n_iter: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = sorted(df["month_block"].unique())
    month_index = {m: i for i, m in enumerate(months)}
    base_monthly = []
    policy_monthly = {}
    for policy in POLICIES:
        rows = []
        sub = apply_policy(df, policy)
        for m in months:
            g = sub[sub["month_block"].eq(m)]
            rows.append({"stake": float(g["stake_yen"].sum()), "return": float(g["return_yen"].sum())})
        arr = np.array([[r["stake"], r["return"]] for r in rows], dtype=float)
        policy_monthly[policy.name] = arr
        if policy.name == "no_filter":
            base_monthly = arr
    out_rows = []
    for policy in POLICIES:
        arr = policy_monthly[policy.name]
        rois = []
        profit_deltas = []
        for _ in range(n_iter):
            sample = rng.integers(0, len(months), len(months))
            stake = float(arr[sample, 0].sum())
            ret = float(arr[sample, 1].sum())
            base_stake = float(base_monthly[sample, 0].sum())
            base_ret = float(base_monthly[sample, 1].sum())
            rois.append(ret / stake * 100 if stake else 0.0)
            profit_deltas.append((ret - stake) - (base_ret - base_stake))
        rois_arr = np.array(rois)
        delta_arr = np.array(profit_deltas)
        full_m = metrics(apply_policy(df, policy))
        out_rows.append(
            {
                "policy": policy.name,
                "uses_post_race_lap": policy.uses_post_race_lap,
                "tickets": full_m["tickets"],
                "full_roi_pct": full_m["roi_pct"],
                "full_roi_ex_top1_pct": full_m["roi_ex_top1_pct"],
                "bootstrap_roi_p05": round(float(np.percentile(rois_arr, 5)), 1),
                "bootstrap_roi_p50": round(float(np.percentile(rois_arr, 50)), 1),
                "bootstrap_roi_p95": round(float(np.percentile(rois_arr, 95)), 1),
                "prob_roi_gt100_pct": round(float((rois_arr > 100).mean() * 100), 1),
                "prob_profit_gt_no_filter_pct": round(float((delta_arr > 0).mean() * 100), 1),
                "profit_delta_vs_no_filter_p05": round(float(np.percentile(delta_arr, 5)), 1),
                "profit_delta_vs_no_filter_p50": round(float(np.percentile(delta_arr, 50)), 1),
                "profit_delta_vs_no_filter_p95": round(float(np.percentile(delta_arr, 95)), 1),
            }
        )
    return pd.DataFrame(out_rows)


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    work = df.copy()
    work = work.replace({np.nan: ""})
    headers = [str(c) for c in work.columns]
    rows = [[str(v) for v in row] for row in work.to_numpy()]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def write_readme(
    policy_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    wf_summary: pd.DataFrame,
    pbo_summary: pd.DataFrame,
    boot_df: pd.DataFrame,
) -> None:
    deployable = policy_df[policy_df["year"].isna() & ~policy_df["uses_post_race_lap"]].copy()
    fixed_top = deployable.sort_values("roi_pct", ascending=False).head(5)
    lines = [
        "# Retro Lap Policy Filter MCS/PBO v1",
        "",
        "## Purpose",
        "retro低リスク馬連のフィルター候補が、探索当たりではなく月次で耐えるかを検証する。",
        "",
        "## Important Leak Check",
        "`actual_lap_regime` を使う候補はレース後に確定する情報を含むため、本番BUYには使わない。診断用として分離した。",
        "",
        "## Fixed Policy Summary",
        md_table(
            fixed_top[
                [
                    "policy",
                    "tickets",
                    "races",
                    "roi_pct",
                    "hit_rate_pct",
                    "roi_ex_top1_pct",
                    "top_return_share_pct",
                    "max_drawdown_yen",
                ]
            ]
        ),
        "",
        "## Walk-Forward Summary",
        md_table(wf_summary),
        "",
        "## PBO Summary",
        md_table(pbo_summary),
        "",
        "## Bootstrap Summary",
        md_table(
            boot_df.sort_values(["uses_post_race_lap", "prob_roi_gt100_pct", "full_roi_pct"], ascending=[True, False, False])[
                [
                    "policy",
                    "uses_post_race_lap",
                    "tickets",
                    "full_roi_pct",
                    "bootstrap_roi_p05",
                    "bootstrap_roi_p50",
                    "bootstrap_roi_p95",
                    "prob_roi_gt100_pct",
                    "prob_profit_gt_no_filter_pct",
                ]
            ]
        ),
        "",
        "## Files",
        "- `policy_metrics.csv`",
        "- `monthly_policy_metrics.csv`",
        "- `walk_forward_detail.csv` / `walk_forward_summary.csv`",
        "- `pbo_detail.csv` / `pbo_summary.csv`",
        "- `bootstrap_summary.csv`",
        "- `summary.json`",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare(read_csv(IN_PATH, dtype={"race_id": str}))

    policy_df = policy_metrics(df)
    monthly_df = monthly_metrics(df)
    wf_all_detail, wf_all_summary = walk_forward(df, deployable_only=False)
    wf_dep_detail, wf_dep_summary = walk_forward(df, deployable_only=True)
    wf_detail = pd.concat([wf_all_detail, wf_dep_detail], ignore_index=True)
    wf_summary = pd.concat([wf_all_summary, wf_dep_summary], ignore_index=True)
    pbo_all_detail, pbo_all_summary = pbo_analysis(df, deployable_only=False)
    pbo_dep_detail, pbo_dep_summary = pbo_analysis(df, deployable_only=True)
    pbo_detail = pd.concat([pbo_all_detail, pbo_dep_detail], ignore_index=True)
    pbo_summary = pd.concat([pbo_all_summary, pbo_dep_summary], ignore_index=True)
    boot_df = bootstrap_months(df)

    policy_df.to_csv(OUT_DIR / "policy_metrics.csv", index=False, encoding="utf-8-sig")
    monthly_df.to_csv(OUT_DIR / "monthly_policy_metrics.csv", index=False, encoding="utf-8-sig")
    wf_detail.to_csv(OUT_DIR / "walk_forward_detail.csv", index=False, encoding="utf-8-sig")
    wf_summary.to_csv(OUT_DIR / "walk_forward_summary.csv", index=False, encoding="utf-8-sig")
    pbo_detail.to_csv(OUT_DIR / "pbo_detail.csv", index=False, encoding="utf-8-sig")
    pbo_summary.to_csv(OUT_DIR / "pbo_summary.csv", index=False, encoding="utf-8-sig")
    boot_df.to_csv(OUT_DIR / "bootstrap_summary.csv", index=False, encoding="utf-8-sig")

    top_deployable = (
        policy_df[policy_df["year"].isna() & ~policy_df["uses_post_race_lap"]]
        .sort_values("roi_pct", ascending=False)
        .head(5)
        .replace({np.nan: None})
        .to_dict(orient="records")
    )
    report = {
        "input": str(IN_PATH),
        "output_dir": str(OUT_DIR),
        "total_tickets": int(len(df)),
        "date_range": [str(df["race_date"].min().date()), str(df["race_date"].max().date())],
        "top_deployable_fixed_policies": top_deployable,
        "walk_forward_summary": wf_summary.replace({np.nan: None}).to_dict(orient="records"),
        "pbo_summary": pbo_summary.replace({np.nan: None}).to_dict(orient="records"),
        "bootstrap_summary": boot_df.replace({np.nan: None}).to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(policy_df, monthly_df, wf_summary, pbo_summary, boot_df)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
