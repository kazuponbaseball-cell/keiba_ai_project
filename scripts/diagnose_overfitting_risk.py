from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_priority_a_non_day_factors import _metric, _num
from scripts.optimize_operational_win_addon import _json_default
from src.utils.paths import ensure_dir, project_path


CANDIDATES = {
    "current_base": "outputs/analysis/live_runtime_safety_overlay_v1/live_safety_overlaid_tickets.csv",
    "standard": "outputs/analysis/strategy_optimization_v1/recommended_standard_tickets.csv",
    "profit": "outputs/analysis/strategy_optimization_v1/recommended_profit_tickets.csv",
    "roi": "outputs/analysis/strategy_optimization_v1/recommended_roi_tickets.csv",
    "defensive": "outputs/analysis/final_operational_quality_v1/recommended_defensive_tickets.csv",
}


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(project_path(path), dtype={"race_id": str}, low_memory=False)
    if "year" not in df.columns:
        df["year"] = df["race_id"].astype(str).str[:4].astype(int)
    if "date_key" not in df.columns:
        df["date_key"] = df["race_id"].astype(str).str[:8]
    df["date"] = pd.to_datetime(df["date_key"], errors="coerce")
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    return df


def _selected(df: pd.DataFrame) -> pd.DataFrame:
    return df[_num(df.get("runtime_stake_yen"), df.index, 0.0).fillna(0.0).gt(0)].copy()


def _period_metrics(df: pd.DataFrame, label: str, period_col: str) -> pd.DataFrame:
    rows = []
    selected = _selected(df)
    for period, g in selected.groupby(period_col, dropna=False):
        m = _metric(g, f"{label}_{period}")
        m["strategy"] = label
        m["period_type"] = period_col
        m["period"] = str(period)
        rows.append(m)
    return pd.DataFrame(rows)


def _hit_dependency(df: pd.DataFrame, label: str) -> dict:
    selected = _selected(df).copy()
    selected["stake"] = _num(selected.get("runtime_stake_yen"), selected.index, 0.0).fillna(0.0)
    selected["ret"] = _num(selected.get("runtime_return_yen"), selected.index, 0.0).fillna(0.0)
    selected["profit"] = selected["ret"] - selected["stake"]
    race = selected.groupby("race_id", sort=False).agg(
        date=("date", "first"),
        stake=("stake", "sum"),
        ret=("ret", "sum"),
        profit=("profit", "sum"),
    ).reset_index()
    hits = race[race["ret"].gt(0)].sort_values("profit", ascending=False).copy()
    total_profit = float(race["profit"].sum())
    total_stake = float(race["stake"].sum())
    top1 = float(hits["profit"].head(1).sum())
    top3 = float(hits["profit"].head(3).sum())
    top5 = float(hits["profit"].head(5).sum())
    without_top1 = race.copy()
    without_top3 = race.copy()
    without_top5 = race.copy()
    for target, n in [(without_top1, 1), (without_top3, 3), (without_top5, 5)]:
        top_ids = set(hits["race_id"].head(n))
        target.loc[target["race_id"].isin(top_ids), "ret"] = 0.0
        target["profit"] = target["ret"] - target["stake"]
    return {
        "strategy": label,
        "races": int(len(race)),
        "hit_races": int(len(hits)),
        "total_stake_yen": total_stake,
        "total_profit_yen": total_profit,
        "top1_profit_yen": top1,
        "top3_profit_yen": top3,
        "top5_profit_yen": top5,
        "top1_profit_share": top1 / total_profit if total_profit else 0.0,
        "top3_profit_share": top3 / total_profit if total_profit else 0.0,
        "top5_profit_share": top5 / total_profit if total_profit else 0.0,
        "roi_without_top1": float(without_top1["ret"].sum() / without_top1["stake"].sum()) if without_top1["stake"].sum() else 0.0,
        "profit_without_top1_yen": float(without_top1["profit"].sum()),
        "roi_without_top3": float(without_top3["ret"].sum() / without_top3["stake"].sum()) if without_top3["stake"].sum() else 0.0,
        "profit_without_top3_yen": float(without_top3["profit"].sum()),
        "roi_without_top5": float(without_top5["ret"].sum() / without_top5["stake"].sum()) if without_top5["stake"].sum() else 0.0,
        "profit_without_top5_yen": float(without_top5["profit"].sum()),
    }


def _stability_score(monthly: pd.DataFrame, hitdep: dict) -> tuple[int, list[str]]:
    risk = 0
    notes = []
    months = monthly.copy()
    months["roi"] = pd.to_numeric(months["roi"], errors="coerce")
    months["profit_yen"] = pd.to_numeric(months["profit_yen"], errors="coerce")
    if len(months) < 12:
        risk += 2
        notes.append("月次サンプルが少ない")
    negative_rate = float((months["profit_yen"] < 0).mean()) if len(months) else 1.0
    if negative_rate >= 0.35:
        risk += 2
        notes.append("マイナス月が多い")
    elif negative_rate >= 0.20:
        risk += 1
        notes.append("マイナス月がやや多い")
    if float(hitdep.get("top3_profit_share", 0.0)) >= 0.65:
        risk += 2
        notes.append("上位3的中への利益依存が高い")
    elif float(hitdep.get("top3_profit_share", 0.0)) >= 0.45:
        risk += 1
        notes.append("上位3的中への利益依存が中程度")
    if float(hitdep.get("roi_without_top3", 0.0)) < 1.0:
        risk += 3
        notes.append("上位3的中を除くとROIが100%未満")
    elif float(hitdep.get("roi_without_top3", 0.0)) < 1.2:
        risk += 1
        notes.append("上位3的中除外後の余力が小さい")
    roi_cv = float(months["roi"].std() / months["roi"].mean()) if len(months) and months["roi"].mean() else 999.0
    if roi_cv >= 1.0:
        risk += 2
        notes.append("月次ROIのばらつきが大きい")
    elif roi_cv >= 0.65:
        risk += 1
        notes.append("月次ROIのばらつきが中程度")
    return risk, notes


def _make_report(out_dir: Path, summary: pd.DataFrame, period: pd.DataFrame, hitdep: pd.DataFrame, diagnosis: pd.DataFrame) -> None:
    standard_diag = diagnosis[diagnosis["strategy"].eq("standard")].iloc[0].to_dict()
    standard_hit = hitdep[hitdep["strategy"].eq("standard")].iloc[0].to_dict()
    standard_month = period[(period["strategy"].eq("standard")) & (period["period_type"].eq("month"))].copy()
    negative_months = int((pd.to_numeric(standard_month["profit_yen"], errors="coerce") < 0).sum())
    report = f"""# Overfitting Risk Diagnosis

## 判定

標準モードの過学習リスクは **{standard_diag['risk_label']}**。

過学習ではないと断言はできない。
ただし、上位3的中を除いてもROIが100%を上回るなら、完全な偶然依存ではない。

## Standard Mode

- races: {int(standard_hit['races'])}
- hit races: {int(standard_hit['hit_races'])}
- total profit: {standard_hit['total_profit_yen']:,.0f}
- top1 profit share: {standard_hit['top1_profit_share']*100:.1f}%
- top3 profit share: {standard_hit['top3_profit_share']*100:.1f}%
- top5 profit share: {standard_hit['top5_profit_share']*100:.1f}%
- ROI without top1: {standard_hit['roi_without_top1']*100:.1f}%
- ROI without top3: {standard_hit['roi_without_top3']*100:.1f}%
- ROI without top5: {standard_hit['roi_without_top5']*100:.1f}%
- negative months: {negative_months}/{len(standard_month)}

## Risk Reasons

{chr(10).join('- ' + x for x in str(standard_diag['notes']).split(' / ') if x)}

## What This Means

- 月次・四半期でプラスが広く出ていれば、単なる1発依存ではない。
- ただし高配当上位の寄与が大きい場合、バックテストのROIは実力より上振れている可能性がある。
- 実運用ではROI 300%台を期待値として扱わず、まずROI 120-150%維持を合格ラインにする。

## Required Next Validation

1. 完全未使用の今後20-30レースを本当の外部検証にする。
2. チューニング禁止期間を決める。
3. 標準モードの買い目を事前保存し、結果後に差し替えない。
4. 30レース時点でROI 100%未満なら金額を落とす。
5. 50-80レースでROI 120%以上、最大DDが想定内なら段階的に増額する。

## Outputs

- `strategy_summary_metrics.csv`
- `period_metrics.csv`
- `hit_dependency.csv`
- `overfitting_diagnosis.csv`
"""
    (out_dir / "overfitting_risk_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose overfitting risk for optimized betting strategies.")
    parser.add_argument("--output-dir", default="outputs/analysis/overfitting_diagnosis_v1")
    args = parser.parse_args()

    out_dir = ensure_dir(project_path(args.output_dir))
    summary_rows = []
    period_frames = []
    hit_rows = []
    diagnosis_rows = []
    for label, path in CANDIDATES.items():
        if not project_path(path).exists():
            continue
        df = _load(path)
        m = _metric(df, label)
        m["strategy"] = label
        summary_rows.append(m)
        for period_col in ["year", "quarter", "month"]:
            period_frames.append(_period_metrics(df, label, period_col))
        hit = _hit_dependency(df, label)
        hit_rows.append(hit)
    summary = pd.DataFrame(summary_rows)
    period = pd.concat(period_frames, ignore_index=True) if period_frames else pd.DataFrame()
    hitdep = pd.DataFrame(hit_rows)
    for _, row in hitdep.iterrows():
        monthly = period[(period["strategy"].eq(row["strategy"])) & (period["period_type"].eq("month"))]
        risk, notes = _stability_score(monthly, row.to_dict())
        label = "low" if risk <= 2 else "medium" if risk <= 5 else "high"
        diagnosis_rows.append({"strategy": row["strategy"], "risk_score": risk, "risk_label": label, "notes": " / ".join(notes) if notes else "大きな過学習警告は限定的"})
    diagnosis = pd.DataFrame(diagnosis_rows)

    summary.to_csv(out_dir / "strategy_summary_metrics.csv", index=False, encoding="utf-8-sig")
    period.to_csv(out_dir / "period_metrics.csv", index=False, encoding="utf-8-sig")
    hitdep.to_csv(out_dir / "hit_dependency.csv", index=False, encoding="utf-8-sig")
    diagnosis.to_csv(out_dir / "overfitting_diagnosis.csv", index=False, encoding="utf-8-sig")
    _make_report(out_dir, summary, period, hitdep, diagnosis)
    payload = {
        "output_dir": str(out_dir),
        "diagnosis": diagnosis.to_dict("records"),
        "standard_hit_dependency": hitdep[hitdep["strategy"].eq("standard")].to_dict("records"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
