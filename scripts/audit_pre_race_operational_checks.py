from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _num_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def _first_existing(df: pd.DataFrame, cols: Iterable[str], default: float = 0.0) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in cols:
        if col in df.columns:
            out = out.fillna(pd.to_numeric(df[col], errors="coerce"))
    return out.fillna(default).astype(float)


def _as_bool_series(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    out = pd.Series(False, index=df.index)
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col]
        if s.dtype == bool:
            out = out | s.fillna(False)
        else:
            text = s.astype(str).str.lower()
            num = pd.to_numeric(s, errors="coerce").fillna(0.0)
            out = out | text.isin(["true", "1", "yes"]) | num.gt(0)
    return out.fillna(False)


def _round_stake(stake: pd.Series, unit: int = 100) -> pd.Series:
    return (np.floor(stake.fillna(0.0).clip(lower=0.0) / unit) * unit).astype(float)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "race_id" in out.columns:
        out["race_id"] = out["race_id"].astype(str)
    else:
        out["race_id"] = np.arange(len(out)).astype(str)

    out["_date_sort"] = out.get("日付S", out.get("date", out.get("year", ""))).astype(str)
    out["_race_sort"] = _first_existing(out, ["Ｒ", "race_no", "raceNo"], 0.0)
    out["_base_stake"] = _first_existing(out, ["runtime_stake_yen", "eval_stake_yen", "stake_yen"], 0.0)
    out["_pay_per100"] = _first_existing(
        out,
        ["runtime_backtest_pay_per100", "runtime_pay_per100", "umaren_pay", "wide_pay", "win_pay", "trio_pay"],
        0.0,
    )
    hit_from_return = _first_existing(out, ["runtime_return_yen", "eval_return_yen", "return_yen"], 0.0).gt(0)
    hit_from_flags = _as_bool_series(out, ["hit_eval", "hit", "umaren_hit"])
    out["_hit_bool"] = hit_from_return | hit_from_flags
    out["_base_return"] = np.where(out["_hit_bool"], out["_base_stake"] * out["_pay_per100"] / 100.0, 0.0)
    out["_hit_prob"] = _first_existing(out, ["pair_calibrated_hit_prob", "ticket_hit_prob"], 0.0).clip(0, 1)
    out["_danger_score"] = _first_existing(
        out,
        ["ticket_danger_popular_score", "danger_sum", "danger_popular_hybrid_score", "danger_popular_model_score"],
        0.0,
    )
    out["_difficulty_score"] = _first_existing(
        out,
        ["race_difficulty_score", "race_difficulty_model_score", "difficulty", "skip_risk_score"],
        0.0,
    )
    out["_skip_risk_score"] = _first_existing(out, ["skip_risk_score", "skip_risk"], 0.0)
    out["_margin_ratio"] = _first_existing(out, ["runtime_odds_margin_ratio", "min_odds_margin_ratio"], 1.0)
    out["_expected_roi"] = _first_existing(out, ["runtime_expected_roi", "expected_roi_after_slippage"], 1.0)
    out["_runtime_odds"] = _first_existing(out, ["runtime_odds", "quote_pay_proxy_per100"], 0.0)
    quote_pay = _first_existing(out, ["quote_pay_proxy_per100", "runtime_pay_per100"], 0.0)
    out.loc[out["_runtime_odds"].gt(25), "_runtime_odds"] = out.loc[out["_runtime_odds"].gt(25), "_runtime_odds"] / 100.0
    out.loc[out["_runtime_odds"].le(0), "_runtime_odds"] = quote_pay[out["_runtime_odds"].le(0)] / 100.0
    return out


def _metrics(df: pd.DataFrame, label: str, stake: pd.Series, mask: pd.Series | None = None) -> dict:
    if mask is None:
        mask = pd.Series(True, index=df.index)
    selected = df.loc[mask].copy()
    stake = stake.reindex(selected.index).fillna(0.0)
    selected = selected.loc[stake.gt(0)].copy()
    stake = stake.loc[selected.index]
    if selected.empty:
        return {
            "policy": label,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi_pct": 0.0,
            "ticket_hit_rate_pct": 0.0,
            "race_hit_rate_pct": 0.0,
            "max_drawdown_yen": 0.0,
            "top5_removed_roi_pct": 0.0,
            "top10_removed_roi_pct": 0.0,
            "avg_danger": 0.0,
            "avg_difficulty": 0.0,
            "avg_margin_ratio": 0.0,
        }

    returns = pd.Series(
        np.where(selected["_hit_bool"], stake * selected["_pay_per100"] / 100.0, 0.0),
        index=selected.index,
    )
    profit = returns - stake
    stake_sum = float(stake.sum())
    return_sum = float(returns.sum())
    ordered = selected.assign(_stake=stake, _return=returns, _profit=profit).sort_values(
        ["_date_sort", "race_id", "_race_sort"], kind="mergesort"
    )
    race_profit = ordered.groupby("race_id", sort=False)["_profit"].sum()
    equity = race_profit.cumsum()
    drawdown = equity.cummax() - equity

    def removed_roi(n: int) -> float:
        keep = returns.sort_values(ascending=False).iloc[n:]
        if keep.empty:
            return 0.0
        keep_stake = stake.loc[keep.index].sum()
        return float(keep.sum() / keep_stake * 100.0) if keep_stake > 0 else 0.0

    race_hit = ordered.groupby("race_id", sort=False)["_return"].sum().gt(0).mean()
    return {
        "policy": label,
        "tickets": int(len(selected)),
        "races": int(selected["race_id"].nunique()),
        "stake_yen": round(stake_sum, 1),
        "return_yen": round(return_sum, 1),
        "profit_yen": round(return_sum - stake_sum, 1),
        "roi_pct": round(return_sum / stake_sum * 100.0, 1) if stake_sum > 0 else 0.0,
        "ticket_hit_rate_pct": round(float(returns.gt(0).mean() * 100.0), 1),
        "race_hit_rate_pct": round(float(race_hit * 100.0), 1),
        "max_drawdown_yen": round(float(drawdown.max()), 1) if not drawdown.empty else 0.0,
        "top5_removed_roi_pct": round(removed_roi(5), 1),
        "top10_removed_roi_pct": round(removed_roi(10), 1),
        "avg_danger": round(float(selected["_danger_score"].mean()), 3),
        "avg_difficulty": round(float(selected["_difficulty_score"].mean()), 3),
        "avg_margin_ratio": round(float(selected["_margin_ratio"].mean()), 2),
    }


def _gate_metrics(df: pd.DataFrame, col: str, label_prefix: str) -> pd.DataFrame:
    stake = df["_base_stake"]
    rows = [_metrics(df, f"{label_prefix}:base", stake)]
    values = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    fixed = [0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.65, 0.80]
    quantiles = []
    if not values.empty:
        quantiles = [float(values.quantile(q)) for q in [0.50, 0.60, 0.70, 0.80, 0.90]]
    thresholds = sorted(set(round(x, 4) for x in fixed + quantiles if math.isfinite(x)))
    for th in thresholds:
        rows.append(_metrics(df, f"{label_prefix}:{col}<= {th:g}", stake, df[col].le(th)))
    return pd.DataFrame(rows)


def _stake_policy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    rows.append(_metrics(df, "current_runtime_stake", df["_base_stake"]))
    rows.append(_metrics(df, "flat_100", pd.Series(100.0, index=df.index)))
    rows.append(_metrics(df, "flat_300", pd.Series(300.0, index=df.index)))
    rows.append(_metrics(df, "flat_500", pd.Series(500.0, index=df.index)))

    margin = df["_margin_ratio"].fillna(1.0)
    margin_units = pd.Series(100.0, index=df.index)
    margin_units += margin.ge(1.25) * 100.0
    margin_units += margin.ge(1.75) * 200.0
    margin_units += margin.ge(2.50) * 200.0
    margin_units += margin.ge(4.00) * 400.0
    rows.append(_metrics(df, "margin_ladder_100_1000", margin_units.clip(100.0, 1000.0)))

    ev = df["_expected_roi"].fillna(1.0)
    ev_units = pd.Series(100.0, index=df.index)
    ev_units += ev.ge(1.25) * 100.0
    ev_units += ev.ge(1.75) * 200.0
    ev_units += ev.ge(2.50) * 300.0
    ev_units += ev.ge(4.00) * 300.0
    rows.append(_metrics(df, "expected_roi_ladder_100_1000", ev_units.clip(100.0, 1000.0)))

    danger = df["_danger_score"].fillna(0.0)
    difficulty = df["_difficulty_score"].fillna(0.0)
    adjusted = df["_base_stake"].copy()
    adjusted = adjusted.where(danger.le(0.35), adjusted * 0.75)
    adjusted = adjusted.where(danger.le(0.55), adjusted * 0.50)
    adjusted = adjusted.where(~(danger.le(0.10) & difficulty.le(0.50)), adjusted * 1.10)
    rows.append(_metrics(df, "danger_difficulty_adjusted_current", _round_stake(adjusted).clip(100.0, 2000.0)))

    odds = df["_runtime_odds"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    p = df["_hit_prob"].clip(0.001, 0.95)
    b = (odds - 1.0).clip(lower=0.01)
    kelly_fraction = ((b * p) - (1.0 - p)) / b
    kelly_stake = _round_stake((10000.0 * 0.125 * kelly_fraction.clip(lower=0.0))).clip(0.0, 1000.0)
    rows.append(_metrics(df, "one_eighth_kelly_proxy_cap1000", kelly_stake))
    return pd.DataFrame(rows)


def _collect_config_features(obj) -> list[str]:
    features: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {"numeric_features", "categorical_features", "generated_numeric_features", "generated_categorical_features"}:
                if isinstance(value, list):
                    features.extend(str(x) for x in value)
            features.extend(_collect_config_features(value))
    elif isinstance(obj, list):
        for item in obj:
            features.extend(_collect_config_features(item))
    return list(dict.fromkeys(features))


def _leakage_scan(config_path: Path, tickets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        features = _collect_config_features(cfg)
        banned = [str(x) for x in cfg.get("leakage_banned_feature_keywords", [])]
        allowed_prefixes = [str(x) for x in cfg.get("leakage_allowed_prefixes", [])]
        for feature in features:
            allowed = any(feature.startswith(prefix) for prefix in allowed_prefixes)
            hits = [kw for kw in banned if kw in feature and not allowed]
            if hits:
                rows.append(
                    {
                        "scope": "model_config_feature",
                        "name": feature,
                        "status": "review",
                        "reason": ",".join(sorted(set(hits))),
                    }
                )
        if not rows:
            rows.append(
                {
                    "scope": "model_config_feature",
                    "name": str(config_path),
                    "status": "ok",
                    "reason": "official configured numeric/categorical features pass banned keyword scan",
                }
            )
    else:
        rows.append({"scope": "model_config_feature", "name": str(config_path), "status": "missing", "reason": "config not found"})

    result_patterns = re.compile(r"(pay|return|hit|finish|actual_|確定|配当|払戻|着順)", re.IGNORECASE)
    result_cols = [c for c in tickets.columns if result_patterns.search(str(c))]
    for col in result_cols[:80]:
        rows.append(
            {
                "scope": "ticket_backtest_column",
                "name": col,
                "status": "expected_backtest_only",
                "reason": "present for historical payoff/evaluation; do not use in live selection masks",
            }
        )
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame, limit: int | None = None) -> str:
    if limit is not None:
        df = df.head(limit)
    if df.empty:
        return "_no rows_"
    view = df.copy()
    view = view.astype(object).where(pd.notna(view), "")
    cols = [str(c) for c in view.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in view.iterrows():
        values = [str(row[c]).replace("|", "\\|") for c in view.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit pre-race operational gates: danger favorite, race difficulty, stake sizing, and leakage.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/mcs_pbo_runtime_overlay_v3/mcs_full_margin095_s0304_selected_tickets.csv")
    parser.add_argument("--config-json", default="config/baseline_features_workout_optimized_core_same_day_bias.json")
    parser.add_argument("--output-dir", default="outputs/analysis/pre_race_operational_checks_v1")
    args = parser.parse_args()

    tickets_path = _project_path(args.tickets_csv)
    out_dir = _project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(tickets_path, dtype={"race_id": str}, low_memory=False)
    df = _prepare(raw)

    base = pd.DataFrame([_metrics(df, "base_current_runtime", df["_base_stake"])])
    danger = _gate_metrics(df, "_danger_score", "danger_gate")
    difficulty = _gate_metrics(df, "_difficulty_score", "difficulty_gate")
    skip_risk = _gate_metrics(df, "_skip_risk_score", "skip_risk_gate")
    stake = _stake_policy_metrics(df)
    leakage = _leakage_scan(_project_path(args.config_json), raw)

    base.to_csv(out_dir / "base_metrics.csv", index=False, encoding="utf-8-sig")
    danger.to_csv(out_dir / "danger_gate_metrics.csv", index=False, encoding="utf-8-sig")
    difficulty.to_csv(out_dir / "difficulty_gate_metrics.csv", index=False, encoding="utf-8-sig")
    skip_risk.to_csv(out_dir / "skip_risk_gate_metrics.csv", index=False, encoding="utf-8-sig")
    stake.to_csv(out_dir / "stake_policy_metrics.csv", index=False, encoding="utf-8-sig")
    leakage.to_csv(out_dir / "leakage_scan.csv", index=False, encoding="utf-8-sig")

    danger_viable = danger[danger["tickets"].ge(100)].sort_values(["top10_removed_roi_pct", "roi_pct"], ascending=False)
    difficulty_viable = difficulty[difficulty["tickets"].ge(100)].sort_values(["top10_removed_roi_pct", "roi_pct"], ascending=False)
    skip_viable = skip_risk[skip_risk["tickets"].ge(100)].sort_values(["top10_removed_roi_pct", "roi_pct"], ascending=False)
    stake_sorted = stake.sort_values(["top10_removed_roi_pct", "roi_pct"], ascending=False)
    leak_review = leakage[leakage["status"].eq("review")]

    md = [
        "# Pre-race Operational Checks v1",
        "",
        f"- tickets_csv: `{tickets_path}`",
        f"- rows: {len(df)} / races: {df['race_id'].nunique()}",
        "- Note: ROI uses historical payoff columns only for evaluation. Gate masks use pre-race/risk/odds-derived columns.",
        "",
        "## Base",
        _md_table(base),
        "",
        "## Danger Favorite Gate: best viable rows",
        _md_table(danger_viable.head(8)),
        "",
        "## Race Difficulty Gate: best viable rows",
        _md_table(difficulty_viable.head(8)),
        "",
        "## Skip Risk Gate: best viable rows",
        _md_table(skip_viable.head(8)),
        "",
        "## Stake Policy Comparison",
        _md_table(stake_sorted),
        "",
        "## Leakage Scan",
        f"- model config review hits: {len(leak_review)}",
        f"- backtest result/payoff columns present in ticket CSV: {int((leakage['scope'] == 'ticket_backtest_column').sum())}",
        "- evaluation payoff columns are expected in historical backtests, but must remain outside live selection masks.",
        "",
    ]
    if not leak_review.empty:
        md.extend(["### Review hits", _md_table(leak_review.head(30)), ""])
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "base": base.iloc[0].to_dict(),
                "best_danger_viable": danger_viable.head(1).to_dict(orient="records"),
                "best_difficulty_viable": difficulty_viable.head(1).to_dict(orient="records"),
                "best_stake_policy": stake_sorted.head(1).to_dict(orient="records"),
                "leak_review_hits": int(len(leak_review)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
