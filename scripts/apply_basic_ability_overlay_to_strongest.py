from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TICKETS = ROOT / "outputs/analysis/optimized_stack_ab_s_gate_v1/s_priority_selected_tickets.csv"
DEFAULT_RUNNERS = ROOT / "outputs/analysis/basic_ability_transform_candidates_v1/runner_basic_ability_candidate_features.csv"
DEFAULT_OUT = ROOT / "outputs/analysis/basic_ability_overlay_strongest_v1"


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def num(s: pd.Series | None, index: pd.Index | None = None, default: float = np.nan) -> pd.Series:
    if s is None:
        if index is None:
            return pd.Series(dtype=float)
        return pd.Series(default, index=index)
    return pd.to_numeric(s, errors="coerce")


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def metrics(df: pd.DataFrame, name: str) -> dict[str, object]:
    if df.empty:
        return {
            "policy": name,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi": np.nan,
            "ticket_hit_rate": np.nan,
            "race_hit_rate": np.nan,
            "max_drawdown_yen": 0.0,
        }
    work = df.copy()
    work["_eval_stake_yen"] = num(work.get("eval_stake_yen"), work.index, np.nan)
    work["_eval_return_yen"] = num(work.get("eval_return_yen"), work.index, np.nan)
    if work["_eval_stake_yen"].isna().all():
        work["_eval_stake_yen"] = num(work.get("runtime_stake_yen"), work.index, np.nan)
    if work["_eval_return_yen"].isna().all():
        work["_eval_return_yen"] = num(work.get("runtime_return_yen"), work.index, np.nan)
    work["_eval_stake_yen"] = work["_eval_stake_yen"].fillna(num(work.get("stake_yen"), work.index, 0.0)).fillna(0.0)
    work["_eval_return_yen"] = work["_eval_return_yen"].fillna(num(work.get("return_yen"), work.index, 0.0)).fillna(0.0)
    work["profit_yen"] = work["_eval_return_yen"] - work["_eval_stake_yen"]
    work["_date_sort"] = pd.to_datetime(work.get("date_key"), errors="coerce")
    work = work.sort_values(["_date_sort", "race_id", "ticket_type"], na_position="last")
    race_profit = work.groupby("race_id", sort=False)["profit_yen"].sum()
    curve = race_profit.cumsum()
    peak = curve.cummax()
    dd = curve - peak
    return {
        "policy": name,
        "tickets": int(len(work)),
        "races": int(work["race_id"].nunique()),
        "stake_yen": float(work["_eval_stake_yen"].sum()),
        "return_yen": float(work["_eval_return_yen"].sum()),
        "profit_yen": float(work["profit_yen"].sum()),
        "roi": safe_div(float(work["_eval_return_yen"].sum()), float(work["_eval_stake_yen"].sum())),
        "ticket_hit_rate": float((work["_eval_return_yen"] > 0).mean()),
        "race_hit_rate": float((work.groupby("race_id")["_eval_return_yen"].sum() > 0).mean()),
        "max_drawdown_yen": float(dd.min()) if not dd.empty else 0.0,
    }


def available_cols(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)


def load_enriched(tickets_path: Path, runners_path: Path) -> pd.DataFrame:
    tickets = read_csv(tickets_path, dtype=str)
    tickets["race_id"] = tickets["race_id"].astype(str)
    tickets["anchor_no"] = num(tickets.get("anchor_no"), tickets.index)
    tickets["partner_no"] = num(tickets.get("partner_no"), tickets.index)
    tickets["stake_yen"] = num(tickets.get("stake_yen"), tickets.index, 0.0).fillna(0.0)
    tickets["return_yen"] = num(tickets.get("return_yen"), tickets.index, 0.0).fillna(0.0)
    tickets["eval_stake_yen"] = num(tickets.get("runtime_stake_yen"), tickets.index, np.nan)
    tickets["eval_return_yen"] = num(tickets.get("runtime_return_yen"), tickets.index, np.nan)
    tickets["eval_stake_yen"] = tickets["eval_stake_yen"].fillna(tickets["stake_yen"]).fillna(0.0)
    tickets["eval_return_yen"] = tickets["eval_return_yen"].fillna(tickets["return_yen"]).fillna(0.0)

    runner_cols = [
        "race_id",
        "horse_no",
        "source",
        "condition_adjusted_recent_ability_score",
        "recent_weighted_score_3",
        "ability_stability_score_3",
        "ability_floor_score_5",
        "wide_axis_reliability_ability_score",
        "win_ceiling_ability_score",
        "prev_stretch_gain_sec",
        "prev_late_improvement_score",
        "prev_market_underestimated_score",
        "collapse_risk_score",
    ]
    usecols = [c for c in runner_cols if c in available_cols(runners_path)]
    runners = read_csv(runners_path, dtype=str, usecols=usecols)
    runners["race_id"] = runners["race_id"].astype(str)
    runners["horse_no"] = num(runners.get("horse_no"), runners.index)
    order = runners.get("source", pd.Series("", index=runners.index)).map({"test": 0, "train": 1}).fillna(9)
    runners = runners.assign(_source_order=order).sort_values("_source_order")
    runners = runners.drop_duplicates(["race_id", "horse_no"], keep="first")

    anchor = runners.add_prefix("anchor_").rename(columns={"anchor_race_id": "race_id", "anchor_horse_no": "anchor_no"})
    partner = runners.add_prefix("partner_").rename(columns={"partner_race_id": "race_id", "partner_horse_no": "partner_no"})
    out = tickets.merge(anchor, on=["race_id", "anchor_no"], how="left").merge(
        partner, on=["race_id", "partner_no"], how="left"
    )

    features = [c for c in runner_cols if c not in {"race_id", "horse_no", "source"}]
    for feature in features:
        a = num(out.get(f"anchor_{feature}"), out.index)
        p = num(out.get(f"partner_{feature}"), out.index)
        out[f"pair_min_{feature}"] = pd.concat([a, p], axis=1).min(axis=1)
        out[f"pair_avg_{feature}"] = pd.concat([a, p], axis=1).mean(axis=1)
        out[f"pair_max_{feature}"] = pd.concat([a, p], axis=1).max(axis=1)
    return out


def quantile(series: pd.Series, q: float) -> float:
    vals = num(series).dropna()
    return float(vals.quantile(q)) if not vals.empty else float("nan")


def policy_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    floor = num(work.get("pair_min_ability_floor_score_5"), work.index)
    reliability = num(work.get("pair_min_wide_axis_reliability_ability_score"), work.index)
    recent = num(work.get("pair_avg_condition_adjusted_recent_ability_score"), work.index)
    stretch = num(work.get("pair_max_prev_stretch_gain_sec"), work.index)
    market_under = num(work.get("pair_max_prev_market_underestimated_score"), work.index)

    thresholds = {
        "floor_q20": quantile(floor, 0.20),
        "floor_q25": quantile(floor, 0.25),
        "reliability_q20": quantile(reliability, 0.20),
        "recent_q20": quantile(recent, 0.20),
        "recent_q80": quantile(recent, 0.80),
        "stretch_q80": quantile(stretch, 0.80),
        "market_under_q80": quantile(market_under, 0.80),
    }

    masks: list[tuple[str, pd.Series, str]] = [
        ("base_all", pd.Series(True, index=work.index), "現行S優先チケット全体"),
        (
            "skip_low_pair_floor_q20",
            floor.ge(thresholds["floor_q20"]) | floor.isna(),
            "能力底が下位20%のペアを見送り",
        ),
        (
            "skip_low_pair_floor_q25",
            floor.ge(thresholds["floor_q25"]) | floor.isna(),
            "能力底が下位25%のペアを見送り",
        ),
        (
            "skip_low_floor_or_low_recent_q20",
            (floor.ge(thresholds["floor_q20"]) | floor.isna())
            & (recent.ge(thresholds["recent_q20"]) | recent.isna()),
            "能力底または条件補正近走能力が下位20%なら見送り",
        ),
        (
            "strong_recent_top20_only",
            recent.ge(thresholds["recent_q80"]),
            "条件補正近走能力が上位20%のペアだけ買う",
        ),
        (
            "value_shadow_stretch_or_market_top20",
            stretch.ge(thresholds["stretch_q80"]) | market_under.ge(thresholds["market_under_q80"]),
            "直線伸び/市場過小評価の上位20%をシャドー候補として抽出",
        ),
        (
            "low_reliability_removed",
            reliability.ge(thresholds["reliability_q20"]) | reliability.isna(),
            "軸信頼度が下位20%のペアを見送り",
        ),
    ]

    rows: list[dict[str, object]] = []
    policy_rows: list[pd.DataFrame] = []
    for name, mask, desc in masks:
        selected = work.loc[mask.fillna(False)].copy()
        selected["basic_ability_overlay_policy"] = name
        selected["basic_ability_overlay_description"] = desc
        rows.append({**metrics(selected, name), "description": desc})
        policy_rows.append(selected)
    metrics_df = pd.DataFrame(rows)
    concatenated = pd.concat(policy_rows, ignore_index=True) if policy_rows else pd.DataFrame()
    thresholds_df = pd.DataFrame([thresholds])
    return metrics_df, thresholds_df, concatenated


def by_year_metrics(df: pd.DataFrame, policy_col: str = "basic_ability_overlay_policy") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if df.empty or policy_col not in df.columns:
        return pd.DataFrame()
    for (policy, year), group in df.groupby([policy_col, "test_year"], dropna=False):
        rows.append({**metrics(group, f"{policy}_{year}"), "policy_name": policy, "test_year": year})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickets-csv", default=str(DEFAULT_TICKETS))
    parser.add_argument("--runner-candidates-csv", default=str(DEFAULT_RUNNERS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    enriched = load_enriched(Path(args.tickets_csv), Path(args.runner_candidates_csv))
    metrics_df, thresholds_df, all_policy_rows = policy_table(enriched)
    year_df = by_year_metrics(all_policy_rows)

    enriched.to_csv(out_dir / "s_priority_tickets_with_basic_ability_overlay.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(out_dir / "policy_metrics.csv", index=False, encoding="utf-8-sig")
    thresholds_df.to_csv(out_dir / "policy_thresholds.csv", index=False, encoding="utf-8-sig")
    year_df.to_csv(out_dir / "policy_metrics_by_year.csv", index=False, encoding="utf-8-sig")

    base = metrics_df[metrics_df["policy"] == "base_all"].iloc[0].to_dict() if not metrics_df.empty else {}
    summary = {
        "output_dir": str(out_dir),
        "base": base,
        "policy_metrics": metrics_df.to_dict(orient="records"),
        "thresholds": thresholds_df.iloc[0].to_dict() if not thresholds_df.empty else {},
        "recommendation": {
            "adopt_now": [
                "Add pair_min_ability_floor_score_5 as a caution/reduction label.",
                "Do not expand BUY count from these features yet.",
                "Use prev_stretch_gain_sec and prev_market_underestimated_score as shadow/value labels.",
            ],
            "do_not_adopt": [
                "Do not use collapse_risk_score top quantile as a buy booster despite high in-sample ROI.",
                "Do not require high condition-adjusted ability only; it narrows tickets without improving ROI.",
            ],
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
