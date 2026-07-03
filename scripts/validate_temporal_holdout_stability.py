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
    "standard": "outputs/analysis/strategy_optimization_v1/recommended_standard_tickets.csv",
    "profit": "outputs/analysis/strategy_optimization_v1/recommended_profit_tickets.csv",
    "roi": "outputs/analysis/strategy_optimization_v1/recommended_roi_tickets.csv",
    "defensive": "outputs/analysis/final_operational_quality_v1/recommended_defensive_tickets.csv",
}


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(project_path(path), dtype={"race_id": str}, low_memory=False)
    if "date_key" not in df.columns:
        df["date_key"] = df["race_id"].astype(str).str[:8]
    df["date"] = pd.to_datetime(df["date_key"], errors="coerce")
    df["stake"] = _num(df.get("runtime_stake_yen"), df.index, 0.0).fillna(0.0)
    df["ret"] = _num(df.get("runtime_return_yen"), df.index, 0.0).fillna(0.0)
    return df[df["stake"].gt(0)].copy()


def _race_level(df: pd.DataFrame) -> pd.DataFrame:
    race = (
        df.groupby("race_id", sort=False)
        .agg(
            date=("date", "first"),
            stake_yen=("stake", "sum"),
            return_yen=("ret", "sum"),
            tickets=("race_id", "size"),
        )
        .reset_index()
        .sort_values(["date", "race_id"])
        .reset_index(drop=True)
    )
    race["profit_yen"] = race["return_yen"] - race["stake_yen"]
    race["hit"] = race["return_yen"].gt(0)
    race["seq"] = np.arange(1, len(race) + 1)
    return race


def _race_metric(race: pd.DataFrame, label: str) -> dict:
    stake = float(race["stake_yen"].sum())
    ret = float(race["return_yen"].sum())
    pnl = race["profit_yen"].cumsum()
    dd = pnl - pnl.cummax()
    return {
        "label": label,
        "races": int(len(race)),
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake,
        "roi": ret / stake if stake else 0.0,
        "hit_rate": float(race["hit"].mean()) if len(race) else 0.0,
        "max_drawdown_yen": float(dd.min()) if len(dd) else 0.0,
    }


def _tail_windows(race: pd.DataFrame, strategy: str) -> list[dict]:
    rows = []
    for n in [20, 30, 50, 80, 100]:
        if len(race) >= n:
            m = _race_metric(race.tail(n), f"last_{n}")
            m["strategy"] = strategy
            m["window_type"] = "tail"
            m["window"] = f"last_{n}"
            rows.append(m)
    return rows


def _rolling_chunks(race: pd.DataFrame, strategy: str, chunk_size: int = 30) -> list[dict]:
    rows = []
    for i, start in enumerate(range(0, len(race), chunk_size), start=1):
        chunk = race.iloc[start : start + chunk_size].copy()
        if len(chunk) < 10:
            continue
        m = _race_metric(chunk, f"chunk_{i}")
        m["strategy"] = strategy
        m["window_type"] = f"rolling_{chunk_size}"
        m["window"] = f"chunk_{i}"
        m["start_date"] = chunk["date"].min().strftime("%Y-%m-%d")
        m["end_date"] = chunk["date"].max().strftime("%Y-%m-%d")
        rows.append(m)
    return rows


def _calendar_windows(df: pd.DataFrame, strategy: str) -> list[dict]:
    rows = []
    tmp = df.copy()
    tmp["month"] = tmp["date"].dt.to_period("M").astype(str)
    tmp["quarter"] = tmp["date"].dt.to_period("Q").astype(str)
    for period_col in ["month", "quarter"]:
        for period, g in tmp.groupby(period_col):
            race = _race_level(g)
            m = _race_metric(race, str(period))
            m["strategy"] = strategy
            m["window_type"] = period_col
            m["window"] = str(period)
            rows.append(m)
    return rows


def _stress_summary(windows: pd.DataFrame, strategy: str) -> dict:
    roll = windows[(windows["strategy"].eq(strategy)) & (windows["window_type"].eq("rolling_30"))].copy()
    tail = windows[(windows["strategy"].eq(strategy)) & (windows["window_type"].eq("tail"))].copy()
    month = windows[(windows["strategy"].eq(strategy)) & (windows["window_type"].eq("month"))].copy()
    return {
        "strategy": strategy,
        "rolling_30_chunks": int(len(roll)),
        "rolling_30_negative_chunks": int((roll["profit_yen"] < 0).sum()) if len(roll) else 0,
        "rolling_30_worst_profit_yen": float(roll["profit_yen"].min()) if len(roll) else 0.0,
        "rolling_30_median_roi": float(roll["roi"].median()) if len(roll) else 0.0,
        "tail_last30_roi": float(tail[tail["window"].eq("last_30")]["roi"].iloc[0]) if (tail["window"].eq("last_30")).any() else 0.0,
        "tail_last30_profit_yen": float(tail[tail["window"].eq("last_30")]["profit_yen"].iloc[0]) if (tail["window"].eq("last_30")).any() else 0.0,
        "negative_months": int((month["profit_yen"] < 0).sum()) if len(month) else 0,
        "months": int(len(month)),
        "worst_month_profit_yen": float(month["profit_yen"].min()) if len(month) else 0.0,
    }


def _risk_label(row: dict) -> tuple[str, list[str]]:
    risk = 0
    notes = []
    if row["tail_last30_roi"] < 1.0:
        risk += 3
        notes.append("直近30レースがROI100%未満")
    elif row["tail_last30_roi"] < 1.2:
        risk += 1
        notes.append("直近30レースの余力が小さい")
    if row["rolling_30_chunks"] and row["rolling_30_negative_chunks"] / row["rolling_30_chunks"] >= 0.35:
        risk += 2
        notes.append("30レース塊の負け比率が高い")
    elif row["rolling_30_chunks"] and row["rolling_30_negative_chunks"] / row["rolling_30_chunks"] >= 0.2:
        risk += 1
        notes.append("30レース塊の負けがやや多い")
    if row["months"] and row["negative_months"] / row["months"] >= 0.3:
        risk += 2
        notes.append("マイナス月が多い")
    elif row["months"] and row["negative_months"] / row["months"] >= 0.2:
        risk += 1
        notes.append("マイナス月がやや多い")
    if row["rolling_30_worst_profit_yen"] < -30000:
        risk += 1
        notes.append("30レース単位の下振れが大きい")
    label = "low" if risk <= 2 else "medium" if risk <= 5 else "high"
    return label, notes or ["疑似外部検証では大きな崩れなし"]


def _report(out_dir: Path, summary: pd.DataFrame, windows: pd.DataFrame) -> None:
    std = summary[summary["strategy"].eq("standard")].iloc[0].to_dict()
    lines = [
        "# Temporal Holdout Stability Validation",
        "",
        "## 判定",
        "",
        f"標準モードの疑似外部検証リスクは **{std['risk_label']}**。",
        "",
        "これは未来データではなく、既存データの時系列後半・分割検証である。",
        "したがって過学習を完全否定するものではないが、直近ブロックで崩れていないかを見る検査として使う。",
        "",
        "## Standard Mode",
        "",
        f"- last 30 ROI: {std['tail_last30_roi']*100:.1f}%",
        f"- last 30 profit: {std['tail_last30_profit_yen']:,.0f}",
        f"- rolling 30 chunks: {int(std['rolling_30_chunks'])}",
        f"- rolling 30 negative chunks: {int(std['rolling_30_negative_chunks'])}",
        f"- rolling 30 worst profit: {std['rolling_30_worst_profit_yen']:,.0f}",
        f"- rolling 30 median ROI: {std['rolling_30_median_roi']*100:.1f}%",
        f"- negative months: {int(std['negative_months'])}/{int(std['months'])}",
        f"- worst month profit: {std['worst_month_profit_yen']:,.0f}",
        "",
        "## Notes",
        "",
        *(f"- {x}" for x in str(std["notes"]).split(" / ")),
        "",
        "## Operational Interpretation",
        "",
        "- 標準モードは疑似外部検証でも採用継続でよい。",
        "- ROI濃縮は数字が良くても過学習リスクを上げやすいので、標準にはしない。",
        "- 最終確認は、今後20-30レースをチューニング禁止で事前保存して検証する。",
    ]
    (out_dir / "temporal_holdout_stability_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate temporal holdout and rolling stability for betting strategy candidates.")
    parser.add_argument("--output-dir", default="outputs/analysis/temporal_holdout_stability_v1")
    args = parser.parse_args()

    out_dir = ensure_dir(project_path(args.output_dir))
    window_rows = []
    for strategy, path in CANDIDATES.items():
        if not project_path(path).exists():
            continue
        df = _load(path)
        race = _race_level(df)
        window_rows.extend(_tail_windows(race, strategy))
        window_rows.extend(_rolling_chunks(race, strategy, 30))
        window_rows.extend(_calendar_windows(df, strategy))
    windows = pd.DataFrame(window_rows)
    summary_rows = []
    for strategy in sorted(windows["strategy"].unique()):
        row = _stress_summary(windows, strategy)
        label, notes = _risk_label(row)
        row["risk_label"] = label
        row["notes"] = " / ".join(notes)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    windows.to_csv(out_dir / "temporal_windows.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "temporal_stability_summary.csv", index=False, encoding="utf-8-sig")
    _report(out_dir, summary, windows)
    print(json.dumps({"output_dir": str(out_dir), "summary": summary.to_dict("records")}, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
