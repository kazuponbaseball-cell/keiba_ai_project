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


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(project_path(path), dtype={"race_id": str}, low_memory=False)
    if "year" not in df.columns:
        df["year"] = df["race_id"].astype(str).str[:4].astype(int)
    if "date_key" not in df.columns:
        df["date_key"] = df["race_id"].astype(str).str[:8]
    df["date_key"] = pd.to_datetime(df["date_key"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def _selected(df: pd.DataFrame) -> pd.DataFrame:
    return df[_num(df.get("runtime_stake_yen"), df.index, 0.0).fillna(0.0).gt(0)].copy()


def _safe_rate(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _scalar_num(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _monthly(df: pd.DataFrame, label: str) -> pd.DataFrame:
    s = _selected(df)
    g = s.groupby(pd.to_datetime(s["date_key"]).dt.to_period("M").astype(str), dropna=False)
    rows = []
    for month, x in g:
        m = _metric(x, label)
        m["month"] = month
        m["days"] = int(x["date_key"].nunique())
        rows.append(m)
    return pd.DataFrame(rows)


def _quarterly_recent(df: pd.DataFrame) -> pd.DataFrame:
    s = _selected(df).copy()
    s["date"] = pd.to_datetime(s["date_key"], errors="coerce")
    s["quarter"] = s["date"].dt.to_period("Q").astype(str)
    rows = []
    for q, x in s.groupby("quarter"):
        m = _metric(x, f"quarter_{q}")
        m["quarter"] = q
        rows.append(m)
    return pd.DataFrame(rows)


def _race_scaled_5k(df: pd.DataFrame) -> pd.DataFrame:
    s = _selected(df).copy()
    s["stake"] = _num(s.get("runtime_stake_yen"), s.index, 0.0).fillna(0.0)
    s["ret"] = _num(s.get("runtime_return_yen"), s.index, 0.0).fillna(0.0)
    race = s.groupby("race_id", sort=False).agg(
        date_key=("date_key", "first"),
        orig_stake=("stake", "sum"),
        orig_ret=("ret", "sum"),
        ticket_count=("race_id", "size"),
    ).reset_index()
    race["date"] = pd.to_datetime(race["date_key"], errors="coerce")
    race["mult"] = np.where(race["orig_stake"].gt(0), 5000.0 / race["orig_stake"], 0.0)
    race["stake_yen"] = 5000.0
    race["return_yen"] = race["orig_ret"] * race["mult"]
    race["profit_yen"] = race["return_yen"] - race["stake_yen"]
    race["hit"] = race["return_yen"].gt(0)
    return race.sort_values(["date", "race_id"]).reset_index(drop=True)


def _bankroll_stats(race: pd.DataFrame) -> dict:
    eq = race["profit_yen"].cumsum()
    dd = eq - eq.cummax()
    loss_streak = 0
    max_loss_streak = 0
    current_loss_amount = 0.0
    worst_loss_streak_amount = 0.0
    for profit in race["profit_yen"]:
        if profit < 0:
            loss_streak += 1
            current_loss_amount += float(profit)
        else:
            max_loss_streak = max(max_loss_streak, loss_streak)
            worst_loss_streak_amount = min(worst_loss_streak_amount, current_loss_amount)
            loss_streak = 0
            current_loss_amount = 0.0
    max_loss_streak = max(max_loss_streak, loss_streak)
    worst_loss_streak_amount = min(worst_loss_streak_amount, current_loss_amount)
    month = race.groupby(race["date"].dt.to_period("M").astype(str)).agg(
        races=("race_id", "nunique"),
        stake_yen=("stake_yen", "sum"),
        return_yen=("return_yen", "sum"),
        profit_yen=("profit_yen", "sum"),
        hits=("hit", "sum"),
    ).reset_index()
    month["roi"] = month["return_yen"] / month["stake_yen"]
    month["hit_rate"] = month["hits"] / month["races"]
    return {
        "races": int(len(race)),
        "hit_rate": _safe_rate(float(race["hit"].sum()), float(len(race))),
        "avg_month_profit_yen": float(month["profit_yen"].mean()) if len(month) else 0.0,
        "median_month_profit_yen": float(month["profit_yen"].median()) if len(month) else 0.0,
        "worst_month_profit_yen": float(month["profit_yen"].min()) if len(month) else 0.0,
        "best_month_profit_yen": float(month["profit_yen"].max()) if len(month) else 0.0,
        "max_drawdown_yen": float(dd.min()) if len(dd) else 0.0,
        "max_losing_streak_races": int(max_loss_streak),
        "worst_losing_streak_amount_yen": float(worst_loss_streak_amount),
    }


def _apply_defensive(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = df.copy()
    stake = _num(out.get("runtime_stake_yen"), out.index, 0.0).fillna(0.0)
    danger = _num(out.get("ticket_danger_popular_score"), out.index, 0.0).fillna(0.0)
    difficulty = _num(out.get("race_difficulty_score"), out.index, 0.0).fillna(0.0)
    b_score = _num(out.get("b_priority_net_score"), out.index, 0.5).fillna(0.5)
    live_alert = _num(out.get("live_alert_risk_score"), out.index, 0.0).fillna(0.0)
    first_blinker = _num(out.get("ticket_equipment_first_or_reapply_flag"), out.index, 0.0).fillna(0.0)
    risk_points = (
        danger.ge(0.55).astype(int)
        + difficulty.ge(0.66).astype(int)
        + b_score.lt(0.5533280795580451).astype(int)
        + live_alert.ge(0.40).astype(int)
        + first_blinker.eq(1).astype(int)
    )
    out["defensive_risk_points"] = risk_points
    out["pre_defensive_stake_yen"] = stake
    out["defensive_action"] = "KEEP"
    new_stake = stake.copy()
    if mode == "reduce_risk2_50":
        mask = stake.gt(0) & risk_points.ge(2)
        new_stake.loc[mask] = new_stake.loc[mask] * 0.5
        out.loc[mask, "defensive_action"] = "REDUCE_RISK2_50"
    elif mode == "reduce_risk1_30_risk2_60":
        mask1 = stake.gt(0) & risk_points.eq(1)
        mask2 = stake.gt(0) & risk_points.ge(2)
        new_stake.loc[mask1] = new_stake.loc[mask1] * 0.7
        new_stake.loc[mask2] = new_stake.loc[mask2] * 0.4
        out.loc[mask1, "defensive_action"] = "REDUCE_RISK1_30"
        out.loc[mask2, "defensive_action"] = "REDUCE_RISK2_60"
    elif mode == "skip_risk3_reduce_risk2_50":
        mask2 = stake.gt(0) & risk_points.eq(2)
        mask3 = stake.gt(0) & risk_points.ge(3)
        new_stake.loc[mask2] = new_stake.loc[mask2] * 0.5
        new_stake.loc[mask3] = 0.0
        out.loc[mask2, "defensive_action"] = "REDUCE_RISK2_50"
        out.loc[mask3, "defensive_action"] = "SKIP_RISK3"
    else:
        raise ValueError(mode)
    rounded = (np.floor(new_stake / 100.0) * 100.0).clip(lower=100.0)
    out["runtime_stake_yen"] = np.where(new_stake.gt(0), rounded, 0.0)
    pay = _num(out.get("runtime_backtest_pay_per100"), out.index, _num(out.get("quote_pay_proxy_per100"), out.index, 0.0)).fillna(0.0)
    out["runtime_return_yen"] = np.where(out.get("hit", False).astype(bool), pay * out["runtime_stake_yen"] / 100.0, 0.0)
    out["runtime_reason"] = out.get("runtime_reason", "").astype(str) + f"|defensive:{mode}"
    return out[out["runtime_stake_yen"].gt(0)].copy()


def _reason_columns(df: pd.DataFrame, mode_label: str) -> pd.DataFrame:
    out = df.copy()
    reasons = []
    risks = []
    stakes = []
    for _, row in out.iterrows():
        buy = []
        risk = []
        adj = []
        if str(row.get("priority_a_ticket_overlay_action", "")) == "BOOST_UMAREN_A_TOP":
            buy.append("馬連A上位")
            adj.append("A馬連増額")
        if str(row.get("priority_b_context_action", "")) == "BOOST_HIGH_B":
            buy.append("B文脈上位")
            adj.append("B上位増額")
        if _scalar_num(row.get("market_overlay_score"), 0.0) >= 0.60:
            buy.append("市場妙味")
        if _scalar_num(row.get("ticket_front_position_reliability_score"), 0.0) >= 0.60:
            buy.append("前位置信頼")
        if _scalar_num(row.get("ticket_danger_popular_score"), 0.0) >= 0.55:
            risk.append("危険人気")
        if _scalar_num(row.get("race_difficulty_score"), 0.0) >= 0.66:
            risk.append("難解レース")
        if _scalar_num(row.get("b_priority_net_score"), 0.5) < 0.5533280795580451:
            risk.append("B下位")
        if str(row.get("equipment_overlay_action", "")) == "REDUCE_PAIR_FIRST_BLINKER":
            risk.append("馬具変更リスク")
            adj.append("馬具50%減額")
        if str(row.get("live_safety_action", "")) not in ("", "KEEP") and not pd.isna(row.get("live_safety_action")):
            risk.append(f"ライブ警戒:{row.get('live_safety_action')}")
            adj.append("ライブ安全装置")
        if str(row.get("defensive_action", "")) not in ("", "KEEP") and not pd.isna(row.get("defensive_action")):
            risk.append(str(row.get("defensive_action")))
            adj.append("守備モード")
        reasons.append(" / ".join(buy[:4]) if buy else "基準条件クリア")
        risks.append(" / ".join(risk[:4]) if risk else "大きな警戒なし")
        stakes.append(" / ".join(adj[:4]) if adj else "基準金額")
    out["operational_mode"] = mode_label
    out["buy_reason_summary"] = reasons
    out["risk_reason_summary"] = risks
    out["stake_adjustment_summary"] = stakes
    out["dashboard_decision_label"] = np.where(_num(out.get("runtime_stake_yen"), out.index, 0.0).fillna(0.0).gt(0), "BUY", "SKIP")
    return out


def _live_rehearsal_status() -> pd.DataFrame:
    checks = [
        ("pair_odds_latest", "data/processed/live_odds/realtime_pair_odds_latest.csv"),
        ("single_odds_latest", "data/processed/live_odds/realtime_single_odds_latest.csv"),
        ("pair_odds_timeline", "data/processed/live_odds/realtime_pair_odds_timeline.csv"),
        ("single_odds_timeline", "data/processed/live_odds/realtime_single_odds_timeline.csv"),
        ("body_weight_latest", "data/processed/live_body_weight/body_weight_latest.csv"),
        ("dashboard_html", "outputs/ui/keiba_dashboard_aggressive_stake.html"),
        ("netkeiba_plan", "outputs/integration/netkeiba_bet_plan/netkeiba_bet_plan.csv"),
    ]
    rows = []
    for name, rel in checks:
        path = project_path(rel)
        rows.append(
            {
                "check": name,
                "path": rel,
                "exists": path.exists(),
                "size": path.stat().st_size if path.exists() else 0,
                "status": "OK" if path.exists() and path.stat().st_size > 0 else "MISSING_OR_EMPTY",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize operational quality: walk-forward, rehearsal, bankroll, defensive mode, and explainability.")
    parser.add_argument("--standard-csv", default="outputs/analysis/strategy_optimization_v1/recommended_standard_tickets.csv")
    parser.add_argument("--profit-csv", default="outputs/analysis/strategy_optimization_v1/recommended_profit_tickets.csv")
    parser.add_argument("--roi-csv", default="outputs/analysis/strategy_optimization_v1/recommended_roi_tickets.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/final_operational_quality_v1")
    args = parser.parse_args()

    out_dir = ensure_dir(project_path(args.output_dir))
    standard = _load(args.standard_csv)
    profit = _load(args.profit_csv)
    roi = _load(args.roi_csv)

    strategy_rows = []
    for label, df in [("standard", standard), ("profit", profit), ("roi", roi)]:
        m = _metric(df, label)
        m["label"] = label
        strategy_rows.append(m)
    pd.DataFrame(strategy_rows).to_csv(out_dir / "strategy_mode_metrics.csv", index=False, encoding="utf-8-sig")

    monthly_frames = []
    for label, df in [("standard", standard), ("profit", profit), ("roi", roi)]:
        x = _monthly(df, label)
        x["label"] = label
        monthly_frames.append(x)
    monthly = pd.concat(monthly_frames, ignore_index=True)
    monthly.to_csv(out_dir / "monthly_walkforward_metrics.csv", index=False, encoding="utf-8-sig")
    _quarterly_recent(standard).to_csv(out_dir / "standard_quarterly_metrics.csv", index=False, encoding="utf-8-sig")

    race5k = _race_scaled_5k(standard)
    race5k.to_csv(out_dir / "standard_5000_per_race_curve.csv", index=False, encoding="utf-8-sig")
    bankroll = _bankroll_stats(race5k)
    with (out_dir / "bankroll_stress_5000_per_race.json").open("w", encoding="utf-8") as f:
        json.dump(bankroll, f, ensure_ascii=False, indent=2, default=_json_default)

    defensive_rows = []
    best_defensive = None
    for mode in ["reduce_risk2_50", "reduce_risk1_30_risk2_60", "skip_risk3_reduce_risk2_50"]:
        d = _apply_defensive(standard, mode)
        d.to_csv(out_dir / f"defensive_{mode}_tickets.csv", index=False, encoding="utf-8-sig")
        m = _metric(d, mode)
        defensive_rows.append(m)
        if best_defensive is None or (m["profit_yen"] >= best_defensive[1]["profit_yen"] and m["roi"] >= 3.0):
            best_defensive = (mode, m, d)
    defensive_metrics = pd.DataFrame(defensive_rows)
    defensive_metrics.to_csv(out_dir / "defensive_mode_metrics.csv", index=False, encoding="utf-8-sig")

    standard_explained = _reason_columns(standard, "standard")
    standard_explained.to_csv(out_dir / "standard_explained_tickets.csv", index=False, encoding="utf-8-sig")
    if best_defensive is not None:
        defensive_explained = _reason_columns(best_defensive[2], f"defensive_{best_defensive[0]}")
        defensive_explained.to_csv(out_dir / "recommended_defensive_tickets.csv", index=False, encoding="utf-8-sig")

    live_status = _live_rehearsal_status()
    live_status.to_csv(out_dir / "live_rehearsal_status.csv", index=False, encoding="utf-8-sig")

    best_def = best_defensive[1] if best_defensive is not None else {}
    report = f"""# Final Operational Quality Review

## 1. Walk-Forward / Recent Validation

標準・利益最大・ROI濃縮の3モードを同じ指標で再確認した。

- 標準: profit {strategy_rows[0]['profit_yen']:,.0f}, ROI {strategy_rows[0]['roi']*100:.2f}%, races {strategy_rows[0]['races']}
- 利益最大: profit {strategy_rows[1]['profit_yen']:,.0f}, ROI {strategy_rows[1]['roi']*100:.2f}%, races {strategy_rows[1]['races']}
- ROI濃縮: profit {strategy_rows[2]['profit_yen']:,.0f}, ROI {strategy_rows[2]['roi']*100:.2f}%, races {strategy_rows[2]['races']}

標準は買えるレース数を維持しながら、利益最大との差が小さいため本線。

## 2. Race-Day Rehearsal

ライブ運用に必要なファイルの存在確認を `live_rehearsal_status.csv` に出した。
当日運用では以下の順で更新する。

1. TARGET/JV odds snapshot
2. odds normalize
3. odds timeline append
4. runtime BUY/REDUCE/WAIT/SKIP
5. A/B/馬具リスク調整
6. dashboard rebuild
7. netkeiba handoff export

## 3. Bankroll / Drawdown

1レース5,000円固定換算:

- 平均月間利益: {bankroll['avg_month_profit_yen']:,.0f}
- 中央月間利益: {bankroll['median_month_profit_yen']:,.0f}
- 最悪月: {bankroll['worst_month_profit_yen']:,.0f}
- 最高月: {bankroll['best_month_profit_yen']:,.0f}
- 最大DD: {bankroll['max_drawdown_yen']:,.0f}
- 最大連敗: {bankroll['max_losing_streak_races']} races

推奨ストップ:

- 1日損失 -30,000円で新規購入停止
- 月間損失 -100,000円でROI濃縮モードへ移行
- 最大DD -120,000円到達で当月停止

## 4. Defensive Mode

守備モード候補を3つ検証した。
最も実用的な候補:

- {best_defensive[0] if best_defensive else 'none'}
- profit {best_def.get('profit_yen', 0):,.0f}
- ROI {best_def.get('roi', 0)*100:.2f}%
- races {best_def.get('races', 0)}

守備モードは標準より利益が落ちる可能性があるため、常用ではなく、連敗・月間損失時の切替用。

## 5. Explainability / Dashboard

`standard_explained_tickets.csv` に以下を追加した。

- `buy_reason_summary`
- `risk_reason_summary`
- `stake_adjustment_summary`
- `dashboard_decision_label`

これにより、画面上で「なぜ買うか」「なぜ減額したか」「何を警戒しているか」を表示できる。

## Final Recommendation

通常日は `standard_explained_tickets.csv` を使う。
連敗・月間DDが深い日は `recommended_defensive_tickets.csv` または ROI濃縮モードに切り替える。
"""
    (out_dir / "final_operational_quality_report.md").write_text(report, encoding="utf-8")

    payload = {
        "output_dir": str(out_dir),
        "standard": strategy_rows[0],
        "profit": strategy_rows[1],
        "roi": strategy_rows[2],
        "bankroll_5000_per_race": bankroll,
        "best_defensive": best_defensive[0] if best_defensive else None,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
