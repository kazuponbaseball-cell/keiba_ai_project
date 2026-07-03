from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


VENUE_CODE_MAP = {
    "01": "Sapporo",
    "02": "Hakodate",
    "03": "Fukushima",
    "04": "Niigata",
    "05": "Tokyo",
    "06": "Nakayama",
    "07": "Chukyo",
    "08": "Kyoto",
    "09": "Hanshin",
    "10": "Kokura",
}


DEFAULT_MODES = {
    "standard": "outputs/analysis/final_operational_quality_v1/standard_explained_tickets.csv",
    "defensive": "outputs/analysis/final_operational_quality_v1/recommended_defensive_tickets.csv",
    "balanced_teacher": "outputs/analysis/teacher_balanced_operational_policy_v1/balanced_2025_2026_tickets.csv",
    "robust_expansion": "",
}


def num(series: pd.Series | None, index: pd.Index, default: float = 0.0) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def safe_col(df: pd.DataFrame, name: str, default="") -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)


def infer_venue(race_id: pd.Series) -> pd.Series:
    return race_id.astype(str).str.zfill(16).str[8:10].map(VENUE_CODE_MAP).fillna("Unknown")


def infer_date_key(race_id: pd.Series) -> pd.Series:
    text = race_id.astype(str).str.zfill(16)
    date = pd.to_datetime(text.str[:8], format="%Y%m%d", errors="coerce")
    return date.dt.strftime("%Y-%m-%d")


def normalize_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = df.copy()
    out["race_id"] = out["race_id"].astype(str)
    if "year" not in out.columns:
        out["year"] = out["race_id"].str[:4].astype(int)
    if "date_key" not in out.columns:
        out["date_key"] = infer_date_key(out["race_id"])
    else:
        parsed = pd.to_datetime(out["date_key"], errors="coerce")
        fallback = infer_date_key(out["race_id"])
        out["date_key"] = parsed.dt.strftime("%Y-%m-%d").fillna(fallback)

    venue = safe_col(out, "venue", "")
    venue = venue.astype("string").fillna("")
    out["venue_eval"] = venue.mask(venue.eq(""), infer_venue(out["race_id"]))

    out["mode"] = mode
    out["operational_mode"] = mode
    out["ticket_type"] = safe_col(out, "ticket_type", "").astype(str)
    out["anchor_name"] = safe_col(out, "anchor_name", "")
    out["partner_name"] = safe_col(out, "partner_name", "")
    out["horse_a"] = num(out.get("horse_a"), out.index, np.nan)
    out["horse_b"] = num(out.get("horse_b"), out.index, np.nan)
    anchor_no = num(out.get("anchor_no"), out.index, np.nan)
    partner_no = num(out.get("partner_no"), out.index, np.nan)
    out["horse_a"] = out["horse_a"].fillna(np.minimum(anchor_no, partner_no))
    out["horse_b"] = out["horse_b"].fillna(np.maximum(anchor_no, partner_no))
    out["ticket_key"] = (
        out["race_id"].astype(str)
        + ":"
        + out["ticket_type"].astype(str)
        + ":"
        + out["horse_a"].astype("Int64").astype(str)
        + "-"
        + out["horse_b"].astype("Int64").astype(str)
    )

    stake = num(out.get("runtime_stake_yen"), out.index, np.nan)
    out["eval_stake_yen"] = stake.where(stake.gt(0), num(out.get("stake_yen"), out.index, 0.0)).fillna(0.0)
    ret = num(out.get("runtime_return_yen"), out.index, np.nan)
    out["eval_return_yen"] = ret.where(ret.ge(0), num(out.get("return_yen"), out.index, 0.0)).fillna(0.0)
    out["eval_profit_yen"] = out["eval_return_yen"] - out["eval_stake_yen"]
    out["eval_hit"] = safe_col(out, "hit", False).astype(bool)
    out["pay_per100_eval"] = num(out.get("runtime_backtest_pay_per100"), out.index, np.nan)
    if "pay_per100" in out.columns:
        out["pay_per100_eval"] = out["pay_per100_eval"].fillna(num(out["pay_per100"], out.index, 0.0))
    out["pay_per100_eval"] = out["pay_per100_eval"].fillna(0.0)

    if "operation_profile" not in out.columns:
        out["operation_profile"] = "buy_candidate"
    if "operation_profile_label" not in out.columns:
        out["operation_profile_label"] = mode
    if "operation_strength_rank" not in out.columns:
        out["operation_strength_rank"] = 2 if mode == "balanced_teacher" else 1
    if "runtime_action" not in out.columns:
        out["runtime_action"] = np.where(out["eval_stake_yen"].gt(0), "BUY", "SKIP")
    if "runtime_stake_yen" not in out.columns:
        out["runtime_stake_yen"] = out["eval_stake_yen"]
    if "runtime_return_yen" not in out.columns:
        out["runtime_return_yen"] = out["eval_return_yen"]
    if "runtime_ticket_status" not in out.columns:
        out["runtime_ticket_status"] = np.where(out["eval_stake_yen"].gt(0), "BUY", "SKIP")
    if "dashboard_decision_label" not in out.columns:
        out["dashboard_decision_label"] = np.where(out["eval_stake_yen"].gt(0), "BUY", "SKIP")
    if "buy_reason_summary" not in out.columns:
        out["buy_reason_summary"] = np.where(
            mode == "balanced_teacher",
            "teacher edge + false-positive filter",
            "standard operational signal",
        )
    if "risk_reason_summary" not in out.columns:
        out["risk_reason_summary"] = ""
    if "stake_adjustment_summary" not in out.columns:
        out["stake_adjustment_summary"] = mode
    return out[out["eval_stake_yen"].gt(0)].copy()


def max_drawdown(profit: pd.Series) -> float:
    if profit.empty:
        return 0.0
    equity = profit.cumsum()
    return float((equity - equity.cummax()).min())


def metrics(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        return {
            "label": label,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi": 0.0,
            "ticket_hit_rate": 0.0,
            "race_hit_rate": 0.0,
            "max_drawdown_yen": 0.0,
        }
    race = (
        df.groupby("race_id", sort=False)
        .agg(stake_yen=("eval_stake_yen", "sum"), return_yen=("eval_return_yen", "sum"), hit=("eval_hit", "max"))
        .reset_index()
    )
    race["profit_yen"] = race["return_yen"] - race["stake_yen"]
    stake = float(df["eval_stake_yen"].sum())
    ret = float(df["eval_return_yen"].sum())
    return {
        "label": label,
        "tickets": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake,
        "roi": ret / stake if stake else 0.0,
        "ticket_hit_rate": float(df["eval_hit"].mean()),
        "race_hit_rate": float(race["hit"].mean()) if len(race) else 0.0,
        "max_drawdown_yen": max_drawdown(race["profit_yen"]),
    }


def grouped_metrics(df: pd.DataFrame, by: list[str], prefix: str) -> pd.DataFrame:
    rows: list[dict] = []
    for key, part in df.groupby(by, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {name: value for name, value in zip(by, key)}
        row.update(metrics(part, prefix))
        rows.append(row)
    return pd.DataFrame(rows)


def top_hit_removed(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows: list[dict] = []
    for mode, part in df.groupby("mode"):
        race = (
            part.groupby("race_id", sort=False)
            .agg(stake_yen=("eval_stake_yen", "sum"), return_yen=("eval_return_yen", "sum"))
            .reset_index()
        )
        race["profit_yen"] = race["return_yen"] - race["stake_yen"]
        kept_ids = set(race.sort_values("profit_yen", ascending=False).iloc[top_n:]["race_id"])
        row = metrics(part[part["race_id"].isin(kept_ids)].copy(), f"{mode}_minus_top{top_n}")
        row["mode"] = mode
        row["removed_top_profit_races"] = top_n
        rows.append(row)
    return pd.DataFrame(rows)


def overlap_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    modes = sorted(df["mode"].unique())
    pivot = (
        df.assign(selected=1)
        .pivot_table(index="ticket_key", columns="mode", values="selected", aggfunc="max", fill_value=0)
        .reset_index()
    )
    for mode in modes:
        if mode not in pivot.columns:
            pivot[mode] = 0
    meta_cols = [
        "ticket_key",
        "race_id",
        "date_key",
        "venue_eval",
        "ticket_type",
        "anchor_name",
        "partner_name",
        "pay_per100_eval",
        "eval_hit",
    ]
    meta = df.sort_values(["mode", "ticket_key"]).drop_duplicates("ticket_key")[[c for c in meta_cols if c in df.columns]]
    detail = meta.merge(pivot, on="ticket_key", how="left")
    rows = []
    for left in modes:
        for right in modes:
            if left >= right:
                continue
            left_set = set(df[df["mode"].eq(left)]["ticket_key"])
            right_set = set(df[df["mode"].eq(right)]["ticket_key"])
            both = left_set & right_set
            rows.append(
                {
                    "left_mode": left,
                    "right_mode": right,
                    "left_tickets": len(left_set),
                    "right_tickets": len(right_set),
                    "overlap_tickets": len(both),
                    "left_only": len(left_set - right_set),
                    "right_only": len(right_set - left_set),
                    "jaccard": len(both) / len(left_set | right_set) if left_set | right_set else 0.0,
                }
            )
    return pd.DataFrame(rows), detail


def load_modes(paths: dict[str, str]) -> pd.DataFrame:
    frames = []
    for mode, path_text in paths.items():
        path = Path(path_text)
        if not path.exists():
            continue
        raw = pd.read_csv(path, dtype={"race_id": str}, low_memory=False)
        frames.append(normalize_mode(raw, mode))
    if not frames:
        raise FileNotFoundError("No mode input CSVs were found.")
    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare standard, defensive, and balanced teacher operational modes.")
    parser.add_argument("--standard-csv", default=DEFAULT_MODES["standard"])
    parser.add_argument("--defensive-csv", default=DEFAULT_MODES["defensive"])
    parser.add_argument("--balanced-csv", default=DEFAULT_MODES["balanced_teacher"])
    parser.add_argument("--robust-csv", default=DEFAULT_MODES["robust_expansion"])
    parser.add_argument("--output-dir", default="outputs/analysis/operational_mode_comparison_v1")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = {
        "standard": args.standard_csv,
        "defensive": args.defensive_csv,
        "balanced_teacher": args.balanced_csv,
    }
    if args.robust_csv:
        modes["robust_expansion"] = args.robust_csv
    combined = load_modes(modes)

    combined.to_csv(out_dir / "combined_operational_tickets.csv", index=False, encoding="utf-8-sig")
    mode_metrics = pd.DataFrame([metrics(g, mode) | {"mode": mode} for mode, g in combined.groupby("mode")])
    mode_metrics.to_csv(out_dir / "mode_metrics.csv", index=False, encoding="utf-8-sig")
    grouped_metrics(combined, ["mode", "year"], "mode_year").to_csv(out_dir / "mode_year_metrics.csv", index=False, encoding="utf-8-sig")
    grouped_metrics(combined, ["mode", "venue_eval"], "mode_venue").to_csv(out_dir / "mode_venue_metrics.csv", index=False, encoding="utf-8-sig")
    if "going" in combined.columns:
        grouped_metrics(combined, ["mode", "going"], "mode_going").to_csv(out_dir / "mode_going_metrics.csv", index=False, encoding="utf-8-sig")
    elif "馬場状態" in combined.columns:
        combined["going_eval"] = combined["馬場状態"].fillna("")
        grouped_metrics(combined, ["mode", "going_eval"], "mode_going").to_csv(out_dir / "mode_going_metrics.csv", index=False, encoding="utf-8-sig")
    grouped_metrics(combined, ["mode", "date_key"], "mode_day").to_csv(out_dir / "mode_day_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat([top_hit_removed(combined, 5), top_hit_removed(combined, 10)], ignore_index=True).to_csv(
        out_dir / "top_hit_removed_metrics.csv", index=False, encoding="utf-8-sig"
    )
    overlap, overlap_detail = overlap_summary(combined)
    overlap.to_csv(out_dir / "mode_overlap_summary.csv", index=False, encoding="utf-8-sig")
    overlap_detail.to_csv(out_dir / "mode_overlap_detail.csv", index=False, encoding="utf-8-sig")

    dashboard_cols = [
        "race_id",
        "year",
        "date_key",
        "venue_eval",
        "ticket_type",
        "anchor_no",
        "partner_no",
        "anchor_name",
        "partner_name",
        "horse_a",
        "horse_b",
        "stake_yen",
        "runtime_stake_yen",
        "runtime_return_yen",
        "ticket_hit_prob",
        "quote_pay_proxy_per100",
        "runtime_expected_roi",
        "operation_profile",
        "operation_profile_label",
        "operation_strength_rank",
        "operational_mode",
        "runtime_action",
        "runtime_ticket_status",
        "buy_reason_summary",
        "risk_reason_summary",
        "stake_adjustment_summary",
        "dashboard_decision_label",
        "pay_per100_eval",
        "eval_hit",
    ]
    dashboard = combined[[c for c in dashboard_cols if c in combined.columns]].copy()
    dashboard.to_csv(out_dir / "dashboard_operational_tickets.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "inputs": modes,
        "mode_metrics": mode_metrics.to_dict(orient="records"),
        "overlap": overlap.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
