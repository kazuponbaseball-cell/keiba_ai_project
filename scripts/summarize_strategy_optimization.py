from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_priority_a_non_day_factors import _metric
from src.utils.paths import ensure_dir, project_path


CANDIDATES = [
    {
        "mode": "current_base",
        "label": "現行ベース",
        "role": "baseline",
        "path": "outputs/analysis/live_runtime_safety_overlay_v1/live_safety_overlaid_tickets.csv",
        "note": "ライブ安全装置まで入った現在の基準。",
    },
    {
        "mode": "a_only",
        "label": "Aのみ",
        "role": "candidate",
        "path": "outputs/analysis/priority_a_ticket_type_overlay_boost_umaren_v1/priority_a_ticket_type_overlaid_tickets.csv",
        "note": "馬連A上位を1.15倍。買える数を減らさず利益改善。",
    },
    {
        "mode": "b_only",
        "label": "Bのみ",
        "role": "candidate",
        "path": "outputs/analysis/priority_b_context_factors_v1/boost_high_b_110_tickets.csv",
        "note": "B上位を10%増額。文脈が良い買い目だけ厚くする。",
    },
    {
        "mode": "ab_profit",
        "label": "A+B 利益最大",
        "role": "profit",
        "path": "outputs/analysis/priority_ab_context_factors_v1/boost_high_b_110_tickets.csv",
        "note": "A馬連増額とB上位増額を重ねた利益最大候補。",
    },
    {
        "mode": "ab_balanced_blinker",
        "label": "A+B+馬具リスク減額",
        "role": "standard",
        "path": "outputs/analysis/optimized_stack_ab_blinker_reduce_v1/reduce_pair_first_blinker_50_tickets.csv",
        "note": "利益をほぼ維持しつつ、ROIと最大DDを改善する標準候補。",
    },
    {
        "mode": "abs_roi",
        "label": "A+B+S",
        "role": "roi_candidate",
        "path": "outputs/analysis/optimized_stack_ab_s_gate_v1/s_priority_selected_tickets.csv",
        "note": "SゲートでROI/的中率を濃縮。ただし買える数は減る。",
    },
    {
        "mode": "abs_roi_blinker",
        "label": "A+B+S+馬具リスク減額",
        "role": "roi",
        "path": "outputs/analysis/optimized_stack_abs_blinker_reduce_v1/reduce_pair_first_blinker_50_tickets.csv",
        "note": "ROI最大候補。標準より買える数は減る。",
    },
    {
        "mode": "ab_skip_low_b",
        "label": "A+B 低B見送り",
        "role": "watch",
        "path": "outputs/analysis/priority_ab_context_factors_v1/skip_low_b_tickets.csv",
        "note": "低Bを見送る濃縮案。利益低下が大きく標準にはしない。",
    },
]


def _load_candidate(row: dict) -> pd.DataFrame | None:
    path = project_path(row["path"])
    if not path.exists():
        return None
    return pd.read_csv(path, dtype={"race_id": str}, low_memory=False)


def _metrics_for_candidates() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    year_rows = []
    for candidate in CANDIDATES:
        df = _load_candidate(candidate)
        if df is None:
            continue
        if "year" not in df.columns:
            df["year"] = df["race_id"].astype(str).str[:4].astype(int)
        m = _metric(df, candidate["mode"])
        m["mode"] = candidate["mode"]
        m.update({k: candidate[k] for k in ["label", "role", "path", "note"]})
        summary_rows.append(m)
        for year, g in df.groupby("year"):
            ym = _metric(g, f"{candidate['mode']}_{int(year)}")
            ym.update({"mode": candidate["mode"], "label": candidate["label"], "year": int(year), "role": candidate["role"]})
            year_rows.append(ym)
    return pd.DataFrame(summary_rows), pd.DataFrame(year_rows)


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _fmt_yen(x: float) -> str:
    return f"{x:,.0f}"


def _write_report(out_dir: Path, summary: pd.DataFrame, yearly: pd.DataFrame) -> None:
    base = summary[summary["mode"].eq("current_base")].iloc[0]
    standard = summary[summary["role"].eq("standard")].iloc[0]
    profit = summary[summary["role"].eq("profit")].iloc[0]
    roi = summary[summary["role"].eq("roi")].iloc[0]

    def table(df: pd.DataFrame) -> str:
        label_col = "display_label" if "display_label" in df.columns else "label"
        cols = [label_col, "tickets", "races", "stake_yen", "return_yen", "profit_yen", "roi", "race_hit_rate", "max_drawdown_yen"]
        lines = ["| 戦略 | tickets | races | stake | return | profit | ROI | race hit | max DD |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for _, r in df[cols].iterrows():
            lines.append(
                f"| {r[label_col]} | {int(r['tickets'])} | {int(r['races'])} | {_fmt_yen(r['stake_yen'])} | {_fmt_yen(r['return_yen'])} | {_fmt_yen(r['profit_yen'])} | {_fmt_pct(r['roi'])} | {_fmt_pct(r['race_hit_rate'])} | {_fmt_yen(r['max_drawdown_yen'])} |"
            )
        return "\n".join(lines)

    yearly_focus = yearly[yearly["mode"].isin(["current_base", "ab_profit", "ab_balanced_blinker", "abs_roi_blinker"])].copy()
    yearly_focus["display_label"] = yearly_focus["year"].astype(str) + " " + yearly_focus["label"].astype(str)

    report = f"""# Strategy Optimization Review

## 結論

ここまで追加してきた要素は、いったん整理して使い分けるべき。

標準運用は **A+B+馬具リスク減額** を推奨する。

理由:

- 現行ベースよりROIが大きく改善する。
- 利益は利益最大案にかなり近い。
- 最大ドローダウンが改善する。
- 買えるレース数を減らさない。

## 推奨モード

| モード | 採用戦略 | 用途 |
|---|---|---|
| 標準 | A+B+馬具リスク減額 | 普段の運用。ROIと利益のバランス重視。 |
| 利益最大 | A+B 利益最大 | 強く取りに行く日。資金効率より総利益重視。 |
| ROI濃縮 | A+B+S+馬具リスク減額 | 買い目を絞ってROIを上げたい日。 |

## 全体比較

{table(summary.sort_values(["role", "profit_yen"], ascending=[True, False]))}

## 主要差分

- 現行ベース profit: {_fmt_yen(base['profit_yen'])}, ROI: {_fmt_pct(base['roi'])}
- 標準候補 profit: {_fmt_yen(standard['profit_yen'])}, ROI: {_fmt_pct(standard['roi'])}
- 利益最大 profit: {_fmt_yen(profit['profit_yen'])}, ROI: {_fmt_pct(profit['roi'])}
- ROI濃縮 profit: {_fmt_yen(roi['profit_yen'])}, ROI: {_fmt_pct(roi['roi'])}

標準候補は利益最大案より profit が {_fmt_yen(profit['profit_yen'] - standard['profit_yen'])} だけ低いが、stakeを {_fmt_yen(profit['stake_yen'] - standard['stake_yen'])} 抑え、ROIとDDが改善する。
そのため、日常運用では標準候補を優先する。

## 年度別確認

{table(yearly_focus.sort_values(['year', 'mode']))}

## 採用/保留の整理

採用:

- A: 馬連A上位を1.15倍
- B: B上位を10%増額
- 馬具: ペア馬券の初/再ブリンカー疑いは50%減額

条件付き採用:

- S: ROI濃縮モード。標準では見送りゲートにしない。

保留:

- 低B見送り: ROIは上がるが、買えるレースと利益が落ちる。
- 低B減額: ROIは上がるが、現段階では標準採用の優先度は低い。

## 出力

- `recommended_standard_tickets.csv`: 標準運用
- `recommended_profit_tickets.csv`: 利益最大
- `recommended_roi_tickets.csv`: ROI濃縮
- `strategy_comparison.csv`: 全候補比較
- `strategy_yearly_comparison.csv`: 年度別比較
"""
    (out_dir / "strategy_optimization_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and select optimized strategy stack.")
    parser.add_argument("--output-dir", default="outputs/analysis/strategy_optimization_v1")
    args = parser.parse_args()

    out_dir = ensure_dir(project_path(args.output_dir))
    summary, yearly = _metrics_for_candidates()
    summary.to_csv(out_dir / "strategy_comparison.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(out_dir / "strategy_yearly_comparison.csv", index=False, encoding="utf-8-sig")

    copy_map = {
        "ab_balanced_blinker": "recommended_standard_tickets.csv",
        "ab_profit": "recommended_profit_tickets.csv",
        "abs_roi_blinker": "recommended_roi_tickets.csv",
    }
    for mode, name in copy_map.items():
        src = summary[summary["mode"].eq(mode)]["path"].iloc[0]
        shutil.copyfile(project_path(src), out_dir / name)

    _write_report(out_dir, summary, yearly)
    print(summary[["mode", "label", "tickets", "races", "stake_yen", "return_yen", "profit_yen", "roi", "race_hit_rate", "max_drawdown_yen"]].to_string(index=False))
    print(f"\nWrote: {out_dir}")


if __name__ == "__main__":
    main()
