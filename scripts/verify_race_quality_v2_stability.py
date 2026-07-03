from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIAG = ROOT / "outputs/analysis/race_quality_prediction_v2/race_quality_v2_diagnostics.csv"
DEFAULT_TICKETS = ROOT / "outputs/analysis/lap_pair_refinement_candidates_v1/s_priority_tickets_with_lap_pair_refinement.csv"
DEFAULT_OUT = ROOT / "outputs/analysis/race_quality_v2_verification_v1"
DEFAULT_SURFACE_SOURCES = [
    ROOT / "outputs/analysis/content_bridge_member_features_v1/train_features_with_content_bridge.csv",
    ROOT / "outputs/analysis/content_bridge_member_features_v1/test_features_with_content_bridge.csv",
]

VENUE_NAMES = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)


def num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def add_race_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["race_id"] = out["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    out["year"] = out["race_id"].str[:4]
    out["race_date"] = out["race_id"].str[:8]
    out["venue_code"] = out["race_id"].str[8:10]
    out["venue_name"] = out["venue_code"].map(VENUE_NAMES).fillna(out["venue_code"])
    out["race_no"] = out["race_id"].str[-2:]
    return out


def normalize_surface_value(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    lower = text.lower()
    if "芝" in text or "turf" in lower:
        return "芝"
    if "ダ" in text or "dirt" in lower:
        return "ダート"
    return None


def load_surface_map(paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            # The source columns are currently column 18: surface and 34: race ID.
            # Reading by index avoids Windows/PowerShell encoding glitches around Japanese column names.
            raw = read_csv(path, usecols=[18, 34])
        except Exception:
            continue
        if raw.shape[1] < 2:
            continue
        race_col = next((c for c in raw.columns if "ID" in str(c)), raw.columns[-1])
        surface_col = next((c for c in raw.columns if c != race_col), raw.columns[0])
        frame = pd.DataFrame(
            {
                "race_id": raw[race_col].astype(str).str.replace(r"\.0$", "", regex=True).str.strip(),
                "surface_from_features": raw[surface_col].map(normalize_surface_value),
            }
        )
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["race_id", "surface_from_features"])
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=["race_id", "surface_from_features"])
    merged = merged[merged["race_id"].str.len().ge(12)]
    if merged.empty:
        return pd.DataFrame(columns=["race_id", "surface_from_features"])

    def most_common(series: pd.Series) -> str:
        counts = series.value_counts()
        return str(counts.index[0])

    return (
        merged.groupby("race_id", as_index=False)["surface_from_features"]
        .agg(most_common)
        .sort_values("race_id")
        .reset_index(drop=True)
    )


def merge_surface(df: pd.DataFrame, surface_map: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    existing = out.get("surface", pd.Series(np.nan, index=out.index)).map(normalize_surface_value)
    out["surface_original_normalized"] = existing
    if surface_map.empty:
        out["surface"] = existing
        return out
    out = out.merge(surface_map, on="race_id", how="left")
    out["surface"] = out["surface_from_features"].combine_first(out["surface_original_normalized"])
    return out


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def prediction_metrics(frame: pd.DataFrame, segment: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "segment": segment,
            "races": 0,
            "v1_hit_rate": np.nan,
            "v2_hit_rate": np.nan,
            "v2_minus_v1": np.nan,
            "v2_avg_confidence": np.nan,
            "v2_avg_margin": np.nan,
            "sustain_recall": np.nan,
            "fast_recall": np.nan,
            "slow_recall": np.nan,
            "instant_recall": np.nan,
        }
    return {
        "segment": segment,
        "races": int(len(frame)),
        "v1_hit_rate": float(frame["v1_hit"].mean()),
        "v2_hit_rate": float(frame["v2_hit"].mean()),
        "v2_minus_v1": float(frame["v2_hit"].mean() - frame["v1_hit"].mean()),
        "v2_avg_confidence": float(num(frame.get("v2_confidence"), frame.index).mean()),
        "v2_avg_margin": float(num(frame.get("v2_margin"), frame.index).mean()),
        "sustain_recall": recall(frame, "sustain"),
        "fast_recall": recall(frame, "fast"),
        "slow_recall": recall(frame, "slow"),
        "instant_recall": recall(frame, "instant"),
    }


def recall(frame: pd.DataFrame, klass: str) -> float:
    sub = frame[frame["actual_lap_mode"].eq(klass)]
    return float(sub["v2_hit"].mean()) if not sub.empty else float("nan")


def summarize_prediction(diag: pd.DataFrame) -> dict[str, pd.DataFrame]:
    valid = diag[diag["actual_lap_mode"].isin(["fast", "slow", "instant", "sustain"])].copy()
    test = valid[valid["source"].astype(str).eq("test")].copy()
    out: dict[str, pd.DataFrame] = {}

    rows = [prediction_metrics(test, "test_all")]
    for year, g in test.groupby("year", sort=True):
        rows.append(prediction_metrics(g, f"year:{year}"))
    out["prediction_by_year"] = pd.DataFrame(rows)

    rows = [prediction_metrics(test, "test_all")]
    for venue, g in test.groupby("venue_name", sort=True):
        rows.append(prediction_metrics(g, f"venue:{venue}"))
    out["prediction_by_venue"] = pd.DataFrame(rows).sort_values(["races", "segment"], ascending=[False, True])

    rows = []
    for (year, venue), g in test.groupby(["year", "venue_name"], sort=True):
        if len(g) >= 40:
            rows.append(prediction_metrics(g, f"{year}:{venue}"))
    out["prediction_by_year_venue_min40"] = pd.DataFrame(rows).sort_values(
        ["v2_hit_rate", "races"], ascending=[True, False]
    )

    bin_source = test.copy()
    bin_source["v2_conf_bin"] = pd.qcut(
        bin_source["v2_confidence"].rank(method="first"),
        q=5,
        labels=["q1_low", "q2", "q3", "q4", "q5_high"],
    )
    rows = []
    for bin_name, g in bin_source.groupby("v2_conf_bin", observed=False):
        rows.append(prediction_metrics(g, str(bin_name)))
    out["prediction_by_v2_confidence_bin"] = pd.DataFrame(rows)

    rows = []
    for mode, g in test.groupby("actual_lap_mode", observed=False):
        rows.append(prediction_metrics(g, str(mode)))
    out["prediction_by_actual_mode"] = pd.DataFrame(rows).sort_values("v2_hit_rate")

    rows = []
    if "surface" in test.columns and test["surface"].notna().any():
        rows.append(prediction_metrics(test[test["surface"].notna()], "surface:ALL_FILLED"))
        for surface, g in test.dropna(subset=["surface"]).groupby("surface", sort=True):
            rows.append(prediction_metrics(g, f"surface:{surface}"))
    out["prediction_by_surface"] = pd.DataFrame(rows)

    return out


def ticket_key(frame: pd.DataFrame) -> pd.Series:
    a = num(frame.get("anchor_no"), frame.index).fillna(-1).astype(int).astype(str)
    b = num(frame.get("partner_no"), frame.index).fillna(-1).astype(int).astype(str)
    typ = frame.get("ticket_type", pd.Series("", index=frame.index)).astype(str)
    return frame["race_id"].astype(str) + ":" + a + "-" + b + ":" + typ


def ticket_metrics(frame: pd.DataFrame, segment: str, policy: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "policy": policy,
            "segment": segment,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "roi": np.nan,
            "hit_rate": np.nan,
            "roi_ex_top1_return": np.nan,
            "top_return_share": np.nan,
            "v2_race_read_hit_rate": np.nan,
        }
    stake = num(frame.get("runtime_stake_yen"), frame.index).fillna(num(frame.get("stake_yen"), frame.index)).fillna(0.0)
    ret = num(frame.get("runtime_return_yen"), frame.index).fillna(num(frame.get("return_yen"), frame.index)).fillna(0.0)
    stake_sum = float(stake.sum())
    ret_sum = float(ret.sum())
    if ret_sum > 0 and len(frame) > 1:
        top_idx = int(ret.to_numpy().argmax())
        roi_ex_top = safe_div(ret_sum - float(ret.iloc[top_idx]), stake_sum - float(stake.iloc[top_idx]))
        top_share = float(ret.max() / ret_sum)
    else:
        roi_ex_top = np.nan
        top_share = np.nan
    return {
        "policy": policy,
        "segment": segment,
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()),
        "stake_yen": stake_sum,
        "return_yen": ret_sum,
        "profit_yen": ret_sum - stake_sum,
        "roi": safe_div(ret_sum, stake_sum),
        "hit_rate": float(ret.gt(0).mean()),
        "roi_ex_top1_return": roi_ex_top,
        "top_return_share": top_share,
        "v2_race_read_hit_rate": float(frame["v2_hit"].mean()) if "v2_hit" in frame else np.nan,
        "avg_v2_confidence": float(num(frame.get("v2_confidence"), frame.index).mean()) if "v2_confidence" in frame else np.nan,
    }


def policy_masks(tickets: pd.DataFrame) -> dict[str, pd.Series]:
    lap_fit = num(tickets.get("pair_lap_same_race_fit_score"), tickets.index)
    lap_conf = num(tickets.get("pair_lap_race_confidence"), tickets.index)
    lap_contra = num(tickets.get("pair_lap_contradiction_score"), tickets.index)
    v2_conf = num(tickets.get("v2_confidence"), tickets.index)
    v2_margin = num(tickets.get("v2_margin"), tickets.index)
    thresholds = {
        "lap_fit_q30": float(lap_fit.quantile(0.30)),
        "lap_conf_q40": float(lap_conf.quantile(0.40)),
        "lap_contra_q80": float(lap_contra.quantile(0.80)),
        "v2_conf_q20": float(v2_conf.quantile(0.20)),
        "v2_margin_q30": float(v2_margin.quantile(0.30)),
    }
    masks = {
        "base_all": pd.Series(True, index=tickets.index),
        "lap_fit_q30": lap_fit.ge(thresholds["lap_fit_q30"]),
        "lap_fit_q30_v2_conf_q20": lap_fit.ge(thresholds["lap_fit_q30"]) & v2_conf.ge(thresholds["v2_conf_q20"]),
        "lap_combo_v2_conf_q20": lap_fit.ge(thresholds["lap_fit_q30"])
        & lap_conf.ge(thresholds["lap_conf_q40"])
        & lap_contra.le(thresholds["lap_contra_q80"])
        & v2_conf.ge(thresholds["v2_conf_q20"]),
        "lap_combo_v2_margin_q30": lap_fit.ge(thresholds["lap_fit_q30"])
        & lap_conf.ge(thresholds["lap_conf_q40"])
        & lap_contra.le(thresholds["lap_contra_q80"])
        & v2_margin.ge(thresholds["v2_margin_q30"]),
    }
    for mask in masks.values():
        mask.fillna(False, inplace=True)
    return masks


def prepare_tickets(tickets_path: Path, diag: pd.DataFrame) -> pd.DataFrame:
    tickets = read_csv(tickets_path)
    tickets = add_race_keys(tickets)
    keep = [
        "race_id",
        "v1_predicted_lap_mode",
        "v2_predicted_lap_mode",
        "actual_lap_mode",
        "v1_hit",
        "v2_hit",
        "v2_confidence",
        "v2_margin",
    ]
    out = tickets.merge(diag[[c for c in keep if c in diag.columns]].drop_duplicates("race_id"), on="race_id", how="left")
    ret = num(out.get("runtime_return_yen"), out.index).fillna(num(out.get("return_yen"), out.index)).fillna(0.0)
    out["ticket_hit"] = ret.gt(0)
    out["race_read_x_ticket_result"] = np.select(
        [
            out["ticket_hit"] & out["v2_hit"].fillna(False),
            out["ticket_hit"] & ~out["v2_hit"].fillna(False),
            ~out["ticket_hit"] & out["v2_hit"].fillna(False),
            ~out["ticket_hit"] & ~out["v2_hit"].fillna(False),
        ],
        [
            "ticket_hit_and_race_read_hit",
            "ticket_hit_but_race_read_miss",
            "ticket_miss_but_race_read_hit",
            "ticket_miss_and_race_read_miss",
        ],
        default="unknown",
    )
    return out


def summarize_tickets(tickets: pd.DataFrame) -> dict[str, pd.DataFrame]:
    masks = policy_masks(tickets)
    outputs: dict[str, pd.DataFrame] = {}

    rows = []
    for policy, mask in masks.items():
        sub = tickets.loc[mask].copy()
        rows.append(ticket_metrics(sub, "ALL", policy))
    outputs["ticket_policy_overall"] = pd.DataFrame(rows).sort_values("roi", ascending=False)

    rows = []
    for policy, mask in masks.items():
        sub = tickets.loc[mask].copy()
        for year, g in sub.groupby("year", sort=True):
            rows.append(ticket_metrics(g, str(year), policy))
    outputs["ticket_policy_by_year"] = pd.DataFrame(rows).sort_values(["policy", "segment"])

    rows = []
    for policy, mask in masks.items():
        sub = tickets.loc[mask].copy()
        for venue, g in sub.groupby("venue_name", sort=True):
            if len(g) >= 3:
                rows.append(ticket_metrics(g, venue, policy))
    outputs["ticket_policy_by_venue_min3"] = pd.DataFrame(rows).sort_values(["policy", "roi"], ascending=[True, False])

    rows = []
    for policy, mask in masks.items():
        sub = tickets.loc[mask].copy()
        if "surface" in sub.columns:
            for surface, g in sub.groupby("surface", sort=True):
                rows.append(ticket_metrics(g, str(surface), policy))
    outputs["ticket_policy_by_surface"] = pd.DataFrame(rows).sort_values(["policy", "segment"]) if rows else pd.DataFrame()

    rows = []
    for policy, mask in masks.items():
        sub = tickets.loc[mask].copy()
        for reason, g in sub.groupby("race_read_x_ticket_result", sort=True):
            rows.append(ticket_metrics(g, str(reason), policy))
    outputs["ticket_race_read_decomposition"] = pd.DataFrame(rows).sort_values(["policy", "segment"])

    return outputs


def to_md_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "(empty)"
    view = frame.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_report(out_dir: Path, outputs: dict[str, pd.DataFrame]) -> None:
    pred_year = outputs.get("prediction_by_year", pd.DataFrame())
    pred_venue = outputs.get("prediction_by_venue", pd.DataFrame())
    pred_surface = outputs.get("prediction_by_surface", pd.DataFrame())
    ticket_overall = outputs.get("ticket_policy_overall", pd.DataFrame())
    ticket_year = outputs.get("ticket_policy_by_year", pd.DataFrame())
    ticket_surface = outputs.get("ticket_policy_by_surface", pd.DataFrame())
    decomposed = outputs.get("ticket_race_read_decomposition", pd.DataFrame())

    lines = [
        "# Race Quality v2 Verification",
        "",
        "## Prediction By Year",
        to_md_table(pred_year),
        "",
        "## Prediction By Venue",
        to_md_table(pred_venue),
        "",
        "## Prediction By Surface",
        to_md_table(pred_surface),
        "",
        "## Ticket Policy Overall",
        to_md_table(ticket_overall),
        "",
        "## Ticket Policy By Year",
        to_md_table(ticket_year, max_rows=40),
        "",
        "## Ticket Policy By Surface",
        to_md_table(ticket_surface, max_rows=40),
        "",
        "## Race Read x Ticket Result",
        to_md_table(decomposed, max_rows=60),
        "",
        "## Interpretation",
        "- v2 is still shadow-only. It should not become a formal BUY gate until live non-empty snapshots are checked.",
        "- If a policy has high ROI but low ticket count and high top_return_share, treat it as promising but unstable.",
        "- The most useful next operational check is whether v2 columns are populated in the dashboard on the next live race day.",
    ]
    (out_dir / "verification_report.md").write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify race-quality v2 stability and ticket decomposition.")
    parser.add_argument("--diag", default=str(DEFAULT_DIAG))
    parser.add_argument("--tickets", default=str(DEFAULT_TICKETS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--surface-source",
        action="append",
        default=None,
        help="CSV with race surface columns. Defaults to content_bridge train/test feature files.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    surface_paths = [Path(p) for p in args.surface_source] if args.surface_source else DEFAULT_SURFACE_SOURCES
    surface_paths = [p if p.is_absolute() else ROOT / p for p in surface_paths]
    surface_map = load_surface_map(surface_paths)

    diag = merge_surface(add_race_keys(read_csv(Path(args.diag))), surface_map)
    prediction_outputs = summarize_prediction(diag)
    tickets = merge_surface(prepare_tickets(Path(args.tickets), diag), surface_map)
    ticket_outputs = summarize_tickets(tickets)
    outputs = {**prediction_outputs, **ticket_outputs}

    for name, frame in outputs.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    summary = {
        "diag_csv": str(Path(args.diag)),
        "tickets_csv": str(Path(args.tickets)),
        "surface_sources": [str(p) for p in surface_paths],
        "surface_map_races": int(surface_map["race_id"].nunique()) if not surface_map.empty else 0,
        "diag_surface_coverage": int(diag["surface"].notna().sum()) if "surface" in diag else 0,
        "ticket_surface_coverage": int(tickets["surface"].notna().sum()) if "surface" in tickets else 0,
        "outputs": {name: str(out_dir / f"{name}.csv") for name in outputs},
        "headline": {
            "test_v2_hit_rate": float(
                prediction_outputs["prediction_by_year"].loc[
                    prediction_outputs["prediction_by_year"]["segment"].eq("test_all"), "v2_hit_rate"
                ].iloc[0]
            ),
            "test_v1_hit_rate": float(
                prediction_outputs["prediction_by_year"].loc[
                    prediction_outputs["prediction_by_year"]["segment"].eq("test_all"), "v1_hit_rate"
                ].iloc[0]
            ),
            "best_ticket_policy": ticket_outputs["ticket_policy_overall"].head(1).replace({np.nan: None}).to_dict(orient="records"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir, outputs)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
