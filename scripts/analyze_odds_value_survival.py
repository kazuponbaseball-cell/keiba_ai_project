from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=str, low_memory=False)


def text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def num_series(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def bool_series(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(False, index=frame.index)
    raw = frame[col].astype(str).str.lower()
    numeric = pd.to_numeric(frame[col], errors="coerce").fillna(0)
    return raw.isin(["true", "1", "1.0", "yes", "y"]) | numeric.gt(0)


def normalize_ticket_type(value: Any, label: Any = "") -> str:
    raw = text(value).lower()
    lab = text(label)
    if raw in {"wide", "ワイド"} or lab == "ワイド":
        return "wide"
    if raw in {"umaren", "quinella", "馬連"} or lab == "馬連":
        return "umaren"
    return raw


def parse_stamp(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series.astype(str), format="%Y%m%d_%H%M%S", errors="coerce")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fallback = pd.to_datetime(series.astype(str), errors="coerce")
    parsed = parsed.fillna(fallback)
    return parsed


def pair_key(frame: pd.DataFrame, race_col: str, type_col: str, a_col: str, b_col: str) -> pd.Series:
    a = pd.to_numeric(frame[a_col], errors="coerce")
    b = pd.to_numeric(frame[b_col], errors="coerce")
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return (
        frame[race_col].astype(str)
        + ":"
        + frame[type_col].astype(str)
        + ":"
        + lo.astype("Int64").astype(str)
        + "-"
        + hi.astype("Int64").astype(str)
    )


def load_pnl(paths: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for raw in paths:
        path = project_path(raw)
        frame = read_csv_safe(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["_source_file"] = str(path)
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    out["race_id"] = out.get("raceId", out.get("race_id", "")).astype(str)
    out["ticket_type_norm"] = [
        normalize_ticket_type(kind, label)
        for kind, label in zip(out.get("ticketType", ""), out.get("ticketLabel", ""))
    ]
    out["a_no_norm"] = pd.to_numeric(out.get("aNo", out.get("a_no", "")), errors="coerce")
    out["b_no_norm"] = pd.to_numeric(out.get("bNo", out.get("b_no", "")), errors="coerce")
    out = out[out["race_id"].notna() & out["a_no_norm"].notna() & out["b_no_norm"].notna()].copy()
    out = out[out["ticket_type_norm"].isin(["wide", "umaren"])].copy()
    out["ticket_key"] = pair_key(out, "race_id", "ticket_type_norm", "a_no_norm", "b_no_norm")
    out["stake_yen"] = num_series(out, "stakeYen", 0.0)
    out["return_yen"] = num_series(out, "payoutYen", 0.0)
    out["profit_yen"] = out["return_yen"] - out["stake_yen"]
    out["hit_bool"] = bool_series(out, "hit") | out["return_yen"].gt(0)
    out["date_key"] = out.get("dateKey", out["race_id"].str.slice(0, 8)).astype(str)
    out["start_time"] = out.get("startTime", "").astype(str)
    out["decision_group"] = out.get("decisionGroup", "").astype(str)
    out["decision_label"] = out.get("decisionLabel", "").astype(str)
    out["display_live_odds"] = num_series(out, "liveOdds")
    out["display_live_pay"] = num_series(out, "livePay")
    return out


def load_timeline(path: str, labels: list[str]) -> pd.DataFrame:
    timeline = read_csv_safe(project_path(path))
    if timeline.empty:
        return pd.DataFrame()
    out = timeline.copy()
    out["race_id"] = out.get("race_id", "").astype(str)
    out["ticket_type_norm"] = [normalize_ticket_type(v) for v in out.get("ticket_type", "")]
    out["a_no_norm"] = pd.to_numeric(out.get("a_no", ""), errors="coerce")
    out["b_no_norm"] = pd.to_numeric(out.get("b_no", ""), errors="coerce")
    out["decision_label"] = out.get("decision_label", "").astype(str)
    out = out[out["ticket_type_norm"].isin(["wide", "umaren"])].copy()
    if labels:
        out = out[out["decision_label"].isin(labels)].copy()
    out = out[out["race_id"].notna() & out["a_no_norm"].notna() & out["b_no_norm"].notna()].copy()
    if out.empty:
        return out
    out["ticket_key"] = pair_key(out, "race_id", "ticket_type_norm", "a_no_norm", "b_no_norm")
    out["pay_per100"] = num_series(out, "live_pay_per100")
    out["odds"] = num_series(out, "live_odds")
    out["odds"] = out["odds"].fillna(out["pay_per100"] / 100.0)
    out["pay_per100"] = out["pay_per100"].fillna(out["odds"] * 100.0)
    out["_sort_at"] = parse_stamp(out.get("captured_at", out.get("snapshot_at", "")))
    fallback = parse_stamp(out.get("snapshot_at", ""))
    out["_sort_at"] = out["_sort_at"].fillna(fallback)
    out = out[out["odds"].gt(1.0) & out["odds"].lt(999.0)].copy()
    out = out.sort_values(["ticket_key", "decision_label", "_sort_at"])
    return out.drop_duplicates(["ticket_key", "decision_label"], keep="last")


def timeline_wide(timeline: pd.DataFrame) -> pd.DataFrame:
    if timeline.empty:
        return pd.DataFrame()
    index_cols = ["ticket_key", "race_id", "ticket_type_norm", "a_no_norm", "b_no_norm"]
    odds = timeline.pivot_table(index=index_cols, columns="decision_label", values="odds", aggfunc="last")
    pay = timeline.pivot_table(index=index_cols, columns="decision_label", values="pay_per100", aggfunc="last")
    stamps = timeline.pivot_table(index=index_cols, columns="decision_label", values="_sort_at", aggfunc="last")
    odds.columns = [f"odds_{col}" for col in odds.columns]
    pay.columns = [f"pay_{col}" for col in pay.columns]
    stamps.columns = [f"captured_{col}" for col in stamps.columns]
    return pd.concat([odds, pay, stamps], axis=1).reset_index()


def max_drawdown(profits: pd.Series) -> float:
    if profits.empty:
        return 0.0
    equity = profits.cumsum()
    peak = equity.cummax().clip(lower=0.0)
    return float((equity - peak).min())


def metric_row(rows: pd.DataFrame, label: str) -> dict[str, Any]:
    if rows.empty:
        return {
            "label": label,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi_pct": None,
            "ticket_hit_rate_pct": None,
            "race_hit_rate_pct": None,
        }
    ordered = rows.sort_values(["date_key", "start_time", "race_id", "ticket_key"]).copy()
    stake = float(ordered["stake_yen"].sum())
    ret = float(ordered["return_yen"].sum())
    race = (
        ordered.groupby("race_id", sort=False)
        .agg(
            stake_yen=("stake_yen", "sum"),
            return_yen=("return_yen", "sum"),
            profit_yen=("profit_yen", "sum"),
            hit=("return_yen", lambda s: bool((s > 0).any())),
            date_key=("date_key", "first"),
            start_time=("start_time", "first"),
        )
        .reset_index()
        .sort_values(["date_key", "start_time", "race_id"])
    )
    top_return = ordered["return_yen"].max() if len(ordered) else 0.0
    return {
        "label": label,
        "tickets": int(len(ordered)),
        "races": int(ordered["race_id"].nunique()),
        "stake_yen": round(stake, 1),
        "return_yen": round(ret, 1),
        "profit_yen": round(ret - stake, 1),
        "roi_pct": round(ret / stake * 100, 1) if stake else None,
        "ticket_hit_rate_pct": round(float(ordered["hit_bool"].mean()) * 100, 1),
        "race_hit_rate_pct": round(float(race["hit"].mean()) * 100, 1) if len(race) else None,
        "max_drawdown_yen": round(max_drawdown(race["profit_yen"]), 1),
        "avg_t5_odds": round(float(ordered["odds_T-5"].mean()), 2) if "odds_T-5" in ordered else None,
        "avg_t3_odds": round(float(ordered["odds_T-3"].mean()), 2) if "odds_T-3" in ordered else None,
        "avg_t5_to_t3_ratio": round(float(ordered["t5_to_t3_odds_ratio"].mean()), 3)
        if "t5_to_t3_odds_ratio" in ordered
        else None,
        "top_return_concentration_pct": round(top_return / ret * 100, 1) if ret else 0.0,
    }


def add_survival_features(rows: pd.DataFrame, from_label: str, to_label: str) -> pd.DataFrame:
    out = rows.copy()
    from_col = f"odds_{from_label}"
    to_col = f"odds_{to_label}"
    if from_col not in out.columns:
        out[from_col] = np.nan
    if to_col not in out.columns:
        out[to_col] = np.nan
    out["t5_to_t3_odds_ratio"] = out[to_col] / out[from_col].replace(0, np.nan)
    out["t5_to_t3_drop_rate"] = 1.0 - out["t5_to_t3_odds_ratio"]
    out["t5_to_t3_missing"] = out[from_col].isna() | out[to_col].isna()
    ratio = out["t5_to_t3_odds_ratio"]
    out["value_survival_bucket"] = np.select(
        [
            out["t5_to_t3_missing"],
            ratio.lt(0.80),
            ratio.lt(0.90),
            ratio.lt(1.00),
            ratio.lt(1.10),
            ratio.ge(1.10),
        ],
        [
            "missing",
            "crushed_lt80",
            "down_80_90",
            "mild_down_90_100",
            "stable_100_110",
            "drift_ge110",
        ],
        default="unknown",
    )
    out["value_survived_080"] = ratio.ge(0.80)
    out["value_survived_085"] = ratio.ge(0.85)
    out["value_survived_090"] = ratio.ge(0.90)
    out["value_survived_095"] = ratio.ge(0.95)
    out["value_survived_100"] = ratio.ge(1.00)
    out["value_crushed_20pct"] = ratio.lt(0.80)
    out["value_crushed_10pct"] = ratio.lt(0.90)
    return out


def summarize_segments(rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result_rows: list[dict[str, Any]] = []
    for col in columns:
        if col not in rows.columns:
            continue
        for value, part in rows.groupby(col, dropna=False):
            metric = metric_row(part, f"{col}={value}")
            metric["segment"] = col
            metric["segment_value"] = value
            result_rows.append(metric)
    return pd.DataFrame(result_rows)


def summarize_policies(rows: pd.DataFrame) -> pd.DataFrame:
    policies: list[tuple[str, pd.Series]] = [
        ("all_displayed", pd.Series(True, index=rows.index)),
        ("matched_t5_t3_only", ~rows["t5_to_t3_missing"]),
        ("drop_crush_lt80", rows["value_survived_080"]),
        ("drop_crush_lt85", rows["value_survived_085"]),
        ("drop_crush_lt90", rows["value_survived_090"]),
        ("drop_crush_lt95", rows["value_survived_095"]),
        ("stable_or_drift_only", rows["value_survived_100"]),
    ]
    for group in ["reference_candidate", "reference_watch", "reference_weak", "reference_skip"]:
        mask = rows["decision_group"].eq(group)
        policies.append((f"{group}_all", mask))
        policies.append((f"{group}_survive90", mask & rows["value_survived_090"]))
        policies.append((f"{group}_survive95", mask & rows["value_survived_095"]))
    for ticket_type in ["umaren", "wide"]:
        mask = rows["ticket_type_norm"].eq(ticket_type)
        policies.append((f"{ticket_type}_all", mask))
        policies.append((f"{ticket_type}_survive90", mask & rows["value_survived_090"]))
        policies.append((f"{ticket_type}_survive95", mask & rows["value_survived_095"]))
    return pd.DataFrame([metric_row(rows[mask.fillna(False)], label) for label, mask in policies])


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze whether live odds value survives from T-5 to T-3.")
    parser.add_argument(
        "--pnl-detail-csv",
        nargs="+",
        default=["outputs/analysis/current_live_pnl/current_live_pnl_detail.csv"],
        help="One or more current_live_pnl_detail.csv files.",
    )
    parser.add_argument("--pair-timeline-csv", default="data/processed/live_odds/realtime_pair_odds_timeline.csv")
    parser.add_argument("--from-label", default="T-5")
    parser.add_argument("--to-label", default="T-3")
    parser.add_argument("--extra-labels", nargs="*", default=["T-10", "manual", "final_check"])
    parser.add_argument("--output-dir", default="outputs/analysis/odds_value_survival")
    args = parser.parse_args()

    labels = list(dict.fromkeys([args.from_label, args.to_label, *args.extra_labels]))
    pnl = load_pnl(args.pnl_detail_csv)
    timeline = load_timeline(args.pair_timeline_csv, labels)
    odds = timeline_wide(timeline)
    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if pnl.empty:
        raise SystemExit("No PnL detail rows found.")
    merged = pnl.merge(odds, on="ticket_key", how="left", suffixes=("", "_timeline"))
    merged = add_survival_features(merged, args.from_label, args.to_label)

    segments = summarize_segments(
        merged,
        ["ticket_type_norm", "decision_group", "decision_label", "value_survival_bucket"],
    )
    policies = summarize_policies(merged)

    merged.to_csv(out_dir / "odds_value_survival_rows.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(out_dir / "odds_value_survival_segments.csv", index=False, encoding="utf-8-sig")
    policies.to_csv(out_dir / "odds_value_survival_policy_summary.csv", index=False, encoding="utf-8-sig")

    matched = merged[~merged["t5_to_t3_missing"]].copy()
    summary = {
        "output_dir": str(out_dir),
        "pnl_detail_csv": [str(project_path(p)) for p in args.pnl_detail_csv],
        "pair_timeline_csv": str(project_path(args.pair_timeline_csv)),
        "from_label": args.from_label,
        "to_label": args.to_label,
        "tickets": int(len(merged)),
        "races": int(merged["race_id"].nunique()),
        "matched_t5_t3_tickets": int(len(matched)),
        "matched_t5_t3_races": int(matched["race_id"].nunique()) if not matched.empty else 0,
        "missing_t5_t3_tickets": int(merged["t5_to_t3_missing"].sum()),
        "top_policy_rows": policies.head(20).to_dict(orient="records"),
        "bucket_rows": segments[segments["segment"].eq("value_survival_bucket")]
        .sort_values("tickets", ascending=False)
        .to_dict(orient="records"),
        "note": "This is a shadow-analysis table. Do not promote non-BUY tickets from a single weekend.",
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))
    print(policies.to_string(index=False))


if __name__ == "__main__":
    main()
