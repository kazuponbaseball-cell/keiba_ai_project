from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "outputs" / "analysis" / "pair_covariance_synergy_v1" / "pair_covariance_detail.csv"
OUT_DIR = ROOT / "outputs" / "analysis" / "pair_covariance_synergy_v1" / "mcs_pbo_v1"


CORE_POLICIES = [
    "deployable_filter_base",
    "covariance_rerank_light",
    "covariance_rerank_mid",
    "covariance_rerank_strong",
    "covariance_gate_loose",
    "covariance_gate_mid",
    "covariance_gate_strict",
    "no_mutual_exclusion_gate",
    "queue_specific_synergy",
]


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)


def as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["policy"].astype(str).isin(CORE_POLICIES)].copy()
    out["race_id"] = out["race_id"].astype(str)
    out["race_date"] = pd.to_datetime(out["race_date"], errors="coerce")
    missing = out["race_date"].isna()
    if missing.any():
        inferred = out.loc[missing, "race_id"].astype(str).str.extract(r"(\d{8})", expand=False)
        out.loc[missing, "race_date"] = pd.to_datetime(inferred, format="%Y%m%d", errors="coerce")
    out = out[out["race_date"].notna()].copy()
    out["year"] = out["race_date"].dt.year
    out["month_block"] = out["race_date"].dt.to_period("M").astype(str)
    out["stake_yen"] = pd.to_numeric(out["stake_yen"], errors="coerce").fillna(100.0)
    out["return_yen"] = pd.to_numeric(out["return_yen"], errors="coerce").fillna(0.0)
    out["hit_bool"] = as_bool(out["hit"]) | out["return_yen"].gt(0)
    out["wide_hit_bool"] = as_bool(out["wide_hit"])
    out["wide_return_yen"] = pd.to_numeric(out["wide_pay"], errors="coerce").fillna(0.0).where(out["wide_hit_bool"], 0.0)
    out["wide_profit_yen"] = out["wide_return_yen"] - out["stake_yen"]
    out["profit_yen"] = out["return_yen"] - out["stake_yen"]
    out["_order"] = np.arange(len(out))
    return out.sort_values(["race_date", "race_id", "policy", "_order"]).reset_index(drop=True)


def max_drawdown(profit: pd.Series) -> float:
    if profit.empty:
        return 0.0
    equity = profit.cumsum()
    return float((equity.cummax() - equity).max())


def metrics(df: pd.DataFrame, *, bet: str = "umaren") -> dict:
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
    if bet == "wide":
        ret_col = "wide_return_yen"
        hit_col = "wide_hit_bool"
        profit = df["wide_profit_yen"]
    else:
        ret_col = "return_yen"
        hit_col = "hit_bool"
        profit = df["profit_yen"]
    stake = float(df["stake_yen"].sum())
    ret = float(df[ret_col].sum())
    top_ret = float(df[ret_col].max())
    ex = df.drop(df[ret_col].idxmax()) if top_ret > 0 else df.iloc[0:0]
    ex_stake = float(ex["stake_yen"].sum()) if not ex.empty else 0.0
    ex_ret = float(ex[ret_col].sum()) if not ex.empty else 0.0
    return {
        "tickets": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "stake_yen": round(stake, 1),
        "return_yen": round(ret, 1),
        "profit_yen": round(ret - stake, 1),
        "roi_pct": round(ret / stake * 100, 1) if stake else 0.0,
        "hit_rate_pct": round(float(df[hit_col].mean() * 100), 1),
        "roi_ex_top1_pct": round(ex_ret / ex_stake * 100, 1) if ex_stake else 0.0,
        "top_return_share_pct": round(top_ret / ret * 100, 1) if ret else 0.0,
        "max_drawdown_yen": round(max_drawdown(profit), 1),
    }


def policy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bet in ["umaren", "wide"]:
        for policy, group in df.groupby("policy", sort=False):
            row = metrics(group, bet=bet)
            row.update({"policy": policy, "bet": bet})
            rows.append(row)
            for year, gy in group.groupby("year", sort=True):
                yr = metrics(gy, bet=bet)
                yr.update({"policy": policy, "bet": bet, "year": int(year)})
                rows.append(yr)
    return pd.DataFrame(rows)


def monthly_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bet in ["umaren", "wide"]:
        for policy, gp in df.groupby("policy", sort=False):
            for month, group in gp.groupby("month_block", sort=True):
                row = metrics(group, bet=bet)
                row.update({"policy": policy, "bet": bet, "month_block": month})
                rows.append(row)
    return pd.DataFrame(rows)


def choose_policy(train: pd.DataFrame, *, bet: str) -> str:
    rows = []
    for policy, group in train.groupby("policy"):
        m = metrics(group, bet=bet)
        if m["tickets"] < 40:
            continue
        score = m["roi_pct"] - max(0.0, m["top_return_share_pct"] - 35.0) * 0.8 + m["hit_rate_pct"] * 0.4
        rows.append((score, m["roi_ex_top1_pct"], m["roi_pct"], policy))
    if not rows:
        return "deployable_filter_base"
    rows.sort(reverse=True)
    return rows[0][-1]


def walk_forward(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    months = sorted(df["month_block"].unique())
    rows = []
    min_train_months = 6
    for bet in ["umaren", "wide"]:
        for idx in range(min_train_months, len(months)):
            test_month = months[idx]
            train_months = months[: max(0, idx - 1)]
            train = df[df["month_block"].isin(train_months)]
            test_all = df[df["month_block"].eq(test_month)]
            chosen = choose_policy(train, bet=bet)
            train_m = metrics(train[train["policy"].eq(chosen)], bet=bet)
            test = test_all[test_all["policy"].eq(chosen)]
            test_m = metrics(test, bet=bet)
            rows.append(
                {
                    "bet": bet,
                    "test_month": test_month,
                    "chosen_policy": chosen,
                    "train_months": len(train_months),
                    "train_roi_pct": train_m["roi_pct"],
                    "train_tickets": train_m["tickets"],
                    "test_tickets": test_m["tickets"],
                    "test_roi_pct": test_m["roi_pct"],
                    "test_profit_yen": test_m["profit_yen"],
                    "test_hit_rate_pct": test_m["hit_rate_pct"],
                }
            )
    detail = pd.DataFrame(rows)
    summary_rows = []
    for bet, group in detail.groupby("bet"):
        stake = float(group["test_tickets"].sum() * 100.0)
        ret = float((group["test_profit_yen"] + group["test_tickets"] * 100.0).sum())
        summary_rows.append(
            {
                "bet": bet,
                "months": int(len(group)),
                "tickets": int(group["test_tickets"].sum()),
                "return_yen": round(ret, 1),
                "profit_yen": round(ret - stake, 1),
                "roi_pct": round(ret / stake * 100, 1) if stake else 0.0,
                "positive_month_rate_pct": round(float(group["test_profit_yen"].gt(0).mean() * 100), 1),
                "chosen_policy_counts": json.dumps(group["chosen_policy"].value_counts().to_dict(), ensure_ascii=False),
            }
        )
    return detail, pd.DataFrame(summary_rows)


def assign_equal_time_blocks(df: pd.DataFrame, n_blocks: int = 8) -> pd.Series:
    dates = df[["race_date"]].drop_duplicates().sort_values("race_date").reset_index(drop=True)
    dates["block"] = pd.qcut(dates.index, q=n_blocks, labels=False, duplicates="drop")
    mapper = dates.set_index("race_date")["block"].to_dict()
    return df["race_date"].map(mapper).astype(int)


def pbo_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["pbo_block"] = assign_equal_time_blocks(work, 8)
    blocks = sorted(work["pbo_block"].unique())
    half = len(blocks) // 2
    rows = []
    for bet in ["umaren", "wide"]:
        for split_id, test_blocks in enumerate(itertools.combinations(blocks, half), start=1):
            test_blocks = set(test_blocks)
            train = work[~work["pbo_block"].isin(test_blocks)]
            test = work[work["pbo_block"].isin(test_blocks)]
            scores = []
            test_scores = []
            for policy, group in train.groupby("policy"):
                m = metrics(group, bet=bet)
                if m["tickets"] < 50:
                    continue
                scores.append((m["roi_pct"], m["tickets"], policy))
            for policy, group in test.groupby("policy"):
                tm = metrics(group, bet=bet)
                test_scores.append({"policy": policy, "roi_pct": tm["roi_pct"], "tickets": tm["tickets"]})
            if not scores:
                continue
            scores.sort(reverse=True)
            selected = scores[0][-1]
            test_rank = pd.DataFrame(test_scores).sort_values("roi_pct", ascending=False).reset_index(drop=True)
            selected_pos = int(test_rank.index[test_rank["policy"].eq(selected)][0]) + 1
            rank_pct = 1.0 - ((selected_pos - 1) / max(1, len(test_rank) - 1))
            selected_m = metrics(test[test["policy"].eq(selected)], bet=bet)
            base_m = metrics(test[test["policy"].eq("deployable_filter_base")], bet=bet)
            rows.append(
                {
                    "bet": bet,
                    "split_id": split_id,
                    "test_blocks": ",".join(map(str, sorted(test_blocks))),
                    "selected_policy": selected,
                    "train_roi_pct": scores[0][0],
                    "test_roi_pct": selected_m["roi_pct"],
                    "test_tickets": selected_m["tickets"],
                    "test_profit_yen": selected_m["profit_yen"],
                    "test_rank_pct": round(rank_pct, 4),
                    "test_under_median": bool(rank_pct < 0.5),
                    "test_under_base": bool(selected_m["roi_pct"] < base_m["roi_pct"]),
                    "base_test_roi_pct": base_m["roi_pct"],
                }
            )
    detail = pd.DataFrame(rows)
    summary_rows = []
    for bet, group in detail.groupby("bet"):
        summary_rows.append(
            {
                "bet": bet,
                "splits": int(len(group)),
                "avg_test_roi_pct": round(float(group["test_roi_pct"].mean()), 1),
                "median_test_roi_pct": round(float(group["test_roi_pct"].median()), 1),
                "pbo_under_median_pct": round(float(group["test_under_median"].mean() * 100), 1),
                "under_base_pct": round(float(group["test_under_base"].mean() * 100), 1),
                "positive_profit_split_pct": round(float(group["test_profit_yen"].gt(0).mean() * 100), 1),
                "selected_policy_counts": json.dumps(group["selected_policy"].value_counts().to_dict(), ensure_ascii=False),
            }
        )
    return detail, pd.DataFrame(summary_rows)


def bootstrap_months(df: pd.DataFrame, n_iter: int = 3000, seed: int = 43) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = sorted(df["month_block"].unique())
    rows = []
    for bet in ["umaren", "wide"]:
        monthly: dict[str, np.ndarray] = {}
        for policy in CORE_POLICIES:
            arr = []
            sub = df[df["policy"].eq(policy)]
            for month in months:
                g = sub[sub["month_block"].eq(month)]
                if bet == "wide":
                    arr.append([float(g["stake_yen"].sum()), float(g["wide_return_yen"].sum())])
                else:
                    arr.append([float(g["stake_yen"].sum()), float(g["return_yen"].sum())])
            monthly[policy] = np.array(arr, dtype=float)
        base = monthly["deployable_filter_base"]
        for policy in CORE_POLICIES:
            arr = monthly[policy]
            rois = []
            deltas = []
            for _ in range(n_iter):
                sample = rng.integers(0, len(months), len(months))
                stake = float(arr[sample, 0].sum())
                ret = float(arr[sample, 1].sum())
                base_stake = float(base[sample, 0].sum())
                base_ret = float(base[sample, 1].sum())
                rois.append(ret / stake * 100 if stake else 0.0)
                deltas.append((ret - stake) - (base_ret - base_stake))
            rois_arr = np.array(rois)
            delta_arr = np.array(deltas)
            full = metrics(df[df["policy"].eq(policy)], bet=bet)
            rows.append(
                {
                    "bet": bet,
                    "policy": policy,
                    "tickets": full["tickets"],
                    "full_roi_pct": full["roi_pct"],
                    "full_roi_ex_top1_pct": full["roi_ex_top1_pct"],
                    "bootstrap_roi_p05": round(float(np.percentile(rois_arr, 5)), 1),
                    "bootstrap_roi_p50": round(float(np.percentile(rois_arr, 50)), 1),
                    "bootstrap_roi_p95": round(float(np.percentile(rois_arr, 95)), 1),
                    "prob_roi_gt100_pct": round(float((rois_arr > 100).mean() * 100), 1),
                    "prob_profit_gt_base_pct": round(float((delta_arr > 0).mean() * 100), 1),
                    "profit_delta_vs_base_p05": round(float(np.percentile(delta_arr, 5)), 1),
                    "profit_delta_vs_base_p50": round(float(np.percentile(delta_arr, 50)), 1),
                    "profit_delta_vs_base_p95": round(float(np.percentile(delta_arr, 95)), 1),
                }
            )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    work = df.replace({np.nan: ""})
    headers = [str(c) for c in work.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in work.to_numpy():
        lines.append("| " + " | ".join(str(v).replace("|", "/") for v in row) + " |")
    return "\n".join(lines)


def write_readme(pm: pd.DataFrame, wf: pd.DataFrame, pbo: pd.DataFrame, boot: pd.DataFrame) -> None:
    fixed = pm[pm["year"].isna()].copy()
    lines = [
        "# Pair Covariance MCS/PBO v1",
        "",
        "## Purpose",
        "共分散ゲートが探索当たりではないかを、月次ウォークフォワード、PBO風検証、月次ブートストラップで確認する。",
        "",
        "## Fixed Policy Metrics",
        md_table(
            fixed[
                [
                    "bet",
                    "policy",
                    "tickets",
                    "roi_pct",
                    "hit_rate_pct",
                    "roi_ex_top1_pct",
                    "top_return_share_pct",
                    "max_drawdown_yen",
                ]
            ].sort_values(["bet", "roi_pct"], ascending=[True, False])
        ),
        "",
        "## Walk-Forward Summary",
        md_table(wf),
        "",
        "## PBO Summary",
        md_table(pbo),
        "",
        "## Bootstrap Summary",
        md_table(
            boot[
                [
                    "bet",
                    "policy",
                    "tickets",
                    "full_roi_pct",
                    "bootstrap_roi_p05",
                    "bootstrap_roi_p50",
                    "bootstrap_roi_p95",
                    "prob_roi_gt100_pct",
                    "prob_profit_gt_base_pct",
                ]
            ].sort_values(["bet", "prob_profit_gt_base_pct", "full_roi_pct"], ascending=[True, False, False])
        ),
        "",
        "## Interpretation",
        "- `covariance_gate_mid/strict` は固定ROIは高いが、ベースより利益が上かはブートストラップで確認する。",
        "- `positive_month_rate` が低い場合は正式BUY置換ではなく、信頼度ラベル/減額/シャドーが妥当。",
        "- 閾値は前段検証の全体分位点由来なので、完全な本番採用前にはT-5/T-3スナップショットでシャドー運用する。",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare(read_csv(IN_PATH, dtype={"race_id": str}))
    pm = policy_metrics(df)
    mm = monthly_metrics(df)
    wf_detail, wf_summary = walk_forward(df)
    pbo_detail, pbo_summary = pbo_analysis(df)
    boot = bootstrap_months(df)

    pm.to_csv(OUT_DIR / "policy_metrics.csv", index=False, encoding="utf-8-sig")
    mm.to_csv(OUT_DIR / "monthly_policy_metrics.csv", index=False, encoding="utf-8-sig")
    wf_detail.to_csv(OUT_DIR / "walk_forward_detail.csv", index=False, encoding="utf-8-sig")
    wf_summary.to_csv(OUT_DIR / "walk_forward_summary.csv", index=False, encoding="utf-8-sig")
    pbo_detail.to_csv(OUT_DIR / "pbo_detail.csv", index=False, encoding="utf-8-sig")
    pbo_summary.to_csv(OUT_DIR / "pbo_summary.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(OUT_DIR / "bootstrap_summary.csv", index=False, encoding="utf-8-sig")

    report = {
        "input": str(IN_PATH),
        "output_dir": str(OUT_DIR),
        "policy_metrics": pm[pm["year"].isna()].replace({np.nan: None}).to_dict(orient="records"),
        "walk_forward_summary": wf_summary.replace({np.nan: None}).to_dict(orient="records"),
        "pbo_summary": pbo_summary.replace({np.nan: None}).to_dict(orient="records"),
        "bootstrap_summary": boot.replace({np.nan: None}).to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(pm, wf_summary, pbo_summary, boot)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
