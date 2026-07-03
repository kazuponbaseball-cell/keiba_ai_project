from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TICKETS = ROOT / "outputs/analysis/continuous_race_pace_prediction_v1/ticket_predictions_joined.csv"
DEFAULT_OUT = ROOT / "outputs/analysis/priority_pace_pair_gate_v1"


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)


def num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def norm01(series: pd.Series | None, index: pd.Index, lo: float | None = None, hi: float | None = None) -> pd.Series:
    s = num(series, index).replace([np.inf, -np.inf], np.nan)
    if lo is None:
        lo = float(s.quantile(0.10)) if s.notna().any() else 0.0
    if hi is None:
        hi = float(s.quantile(0.90)) if s.notna().any() else 1.0
    if abs(hi - lo) < 1e-9:
        return pd.Series(0.5, index=index)
    return ((s - lo) / (hi - lo)).clip(0.0, 1.0).fillna(0.0)


def normalize_mode(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if text == "long_spurt":
        return "sustain"
    if text in {"fast", "slow", "instant", "sustain"}:
        return text
    return "unknown"


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def max_drawdown(profit: pd.Series) -> float:
    if profit.empty:
        return 0.0
    curve = profit.cumsum()
    dd = curve - curve.cummax()
    return float(dd.min())


def auc_score(y: pd.Series, score: pd.Series) -> float:
    yy = y.astype(bool)
    ss = pd.to_numeric(score, errors="coerce")
    mask = ss.notna()
    yy = yy[mask]
    ss = ss[mask]
    pos = int(yy.sum())
    neg = int((~yy).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = ss.rank(method="average")
    return float((ranks[yy].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["race_id"] = out["race_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    if "year" not in out.columns:
        out["year"] = out.get("year_x", out["race_id"].str[:4])
    out["year"] = pd.to_numeric(out["year"], errors="coerce").fillna(out["race_id"].str[:4].astype(int)).astype(int)
    if "surface" not in out.columns:
        out["surface"] = out.get("surface_y", out.get("surface_x", "unknown"))
    out["surface"] = out["surface"].astype("string").fillna("unknown")

    stake = num(out.get("runtime_stake_yen"), out.index).fillna(num(out.get("stake_yen"), out.index)).fillna(0.0)
    ret = num(out.get("runtime_return_yen"), out.index).fillna(num(out.get("return_yen"), out.index)).fillna(0.0)
    out["_stake"] = stake
    out["_return"] = ret
    out["_profit"] = ret - stake
    out["_hit"] = ret.gt(0)

    cont_mode = out.get("cont_predicted_lap_mode", pd.Series("unknown", index=out.index)).map(normalize_mode)
    v2_mode = out.get("v2_predicted_lap_mode", pd.Series("unknown", index=out.index)).map(normalize_mode)
    anchor_top = out.get("anchor_horse_lap_profile_top_mode", pd.Series("unknown", index=out.index)).map(normalize_mode)
    partner_top = out.get("partner_horse_lap_profile_top_mode", pd.Series("unknown", index=out.index)).map(normalize_mode)
    anchor_pred = out.get("anchor_predicted_lap_mode", pd.Series("unknown", index=out.index)).map(normalize_mode)
    partner_pred = out.get("partner_predicted_lap_mode", pd.Series("unknown", index=out.index)).map(normalize_mode)

    out["pace_mode_consensus_flag"] = cont_mode.eq(v2_mode) & cont_mode.ne("unknown")
    out["anchor_cont_mode_fit"] = (anchor_top.eq(cont_mode) | anchor_pred.eq(cont_mode)).astype(float)
    out["partner_cont_mode_fit"] = (partner_top.eq(cont_mode) | partner_pred.eq(cont_mode)).astype(float)
    out["pair_cont_mode_fit_raw"] = 0.5 * out["anchor_cont_mode_fit"] + 0.5 * out["partner_cont_mode_fit"]

    idx = out.index
    pair_lap_fit = norm01(out.get("pair_lap_same_race_fit_score"), idx, lo=0.18, hi=0.55)
    pair_lap_conf = norm01(out.get("pair_lap_race_confidence"), idx, lo=0.20, hi=0.58)
    pair_lap_contra_ok = 1.0 - norm01(out.get("pair_lap_contradiction_score"), idx, lo=0.05, hi=0.65)
    pair_profile_min = norm01(out.get("pair_min_lap_profile_fit_score"), idx, lo=0.12, hi=0.65)
    cont_conf = norm01(out.get("cont_confidence"), idx, lo=0.30, hi=0.43)
    cont_margin = norm01(out.get("cont_margin"), idx, lo=0.02, hi=0.10)
    v2_conf = norm01(out.get("v2_confidence"), idx, lo=0.26, hi=0.38)
    consensus = out["pace_mode_consensus_flag"].astype(float)

    out["continuous_pair_pace_fit_score"] = (
        0.26 * pair_lap_fit
        + 0.18 * pair_lap_conf
        + 0.18 * pair_lap_contra_ok
        + 0.14 * pair_profile_min
        + 0.10 * out["pair_cont_mode_fit_raw"]
        + 0.08 * cont_conf
        + 0.04 * cont_margin
        + 0.02 * consensus
    ).clip(0.0, 1.0)
    out["continuous_pair_readability_score"] = (
        0.34 * cont_conf + 0.24 * cont_margin + 0.22 * v2_conf + 0.20 * consensus
    ).clip(0.0, 1.0)

    pair_score = norm01(out.get("pair_score"), idx, lo=0.55, hi=0.86)
    pair_q = norm01(out.get("pair_quinella_score"), idx, lo=0.45, hi=0.80)
    market = norm01(out.get("market_overlay_score"), idx, lo=0.35, hi=0.92)
    late = norm01(out.get("late_value_survives_score"), idx, lo=0.35, hi=0.92)
    hit_prob = norm01(out.get("ticket_hit_prob"), idx, lo=0.10, hi=0.80)
    danger = 1.0 - norm01(
        num(out.get("anchor_danger"), idx, 0.0).fillna(0.0) + num(out.get("partner_danger"), idx, 0.0).fillna(0.0),
        idx,
        lo=0.05,
        hi=0.65,
    )
    out["continuous_pair_value_score"] = (
        0.22 * pair_score
        + 0.20 * pair_q
        + 0.18 * market
        + 0.14 * late
        + 0.14 * hit_prob
        + 0.12 * out["continuous_pair_pace_fit_score"]
    ).clip(0.0, 1.0)
    out["continuous_pair_formal_score"] = (
        0.42 * out["continuous_pair_value_score"]
        + 0.26 * out["continuous_pair_pace_fit_score"]
        + 0.17 * out["continuous_pair_readability_score"]
        + 0.15 * danger
    ).clip(0.0, 1.0)

    out["pace_pair_gate_label"] = np.select(
        [
            out["continuous_pair_formal_score"].ge(0.72) & out["pace_mode_consensus_flag"],
            out["continuous_pair_formal_score"].ge(0.64),
            out["continuous_pair_pace_fit_score"].lt(0.36),
        ],
        ["pace_pair_strong", "pace_pair_watch", "pace_pair_caution"],
        default="pace_pair_neutral",
    )
    return out


def metrics(frame: pd.DataFrame, policy: str, segment: str = "ALL") -> dict[str, Any]:
    if frame.empty:
        return {
            "policy": policy,
            "segment": segment,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi": np.nan,
            "hit_rate": np.nan,
        }
    stake = float(frame["_stake"].sum())
    ret = float(frame["_return"].sum())
    by_race = frame.groupby("race_id", sort=False).agg(
        stake=("_stake", "sum"),
        ret=("_return", "sum"),
        hit=("_hit", "max"),
    )
    if ret > 0 and len(frame) > 1:
        top_i = int(frame["_return"].to_numpy().argmax())
        top_return = float(frame["_return"].iloc[top_i])
        top_stake = float(frame["_stake"].iloc[top_i])
        roi_ex_top = safe_div(ret - top_return, stake - top_stake)
        top_share = safe_div(top_return, ret)
    else:
        roi_ex_top = np.nan
        top_share = np.nan
    return {
        "policy": policy,
        "segment": segment,
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()),
        "stake_yen": stake,
        "return_yen": ret,
        "profit_yen": ret - stake,
        "roi": safe_div(ret, stake),
        "hit_rate": float(frame["_hit"].mean()),
        "race_hit_rate": float(by_race["hit"].mean()),
        "roi_ex_top1_return": roi_ex_top,
        "top_return_share": top_share,
        "max_drawdown_yen": max_drawdown(by_race["ret"] - by_race["stake"]),
        "avg_formal_score": float(frame["continuous_pair_formal_score"].mean()),
        "avg_pace_fit_score": float(frame["continuous_pair_pace_fit_score"].mean()),
        "mode_consensus_rate": float(frame["pace_mode_consensus_flag"].mean()),
    }


def policy_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    fit = df["continuous_pair_pace_fit_score"]
    formal = df["continuous_pair_formal_score"]
    readable = df["continuous_pair_readability_score"]
    consensus = df["pace_mode_consensus_flag"].fillna(False)
    value = df["continuous_pair_value_score"]
    masks = {
        "base_formal_buy_all": pd.Series(True, index=df.index),
        "formal_score_q50": formal.ge(float(formal.quantile(0.50))),
        "formal_score_q60": formal.ge(float(formal.quantile(0.60))),
        "formal_score_q70": formal.ge(float(formal.quantile(0.70))),
        "pace_fit_q60": fit.ge(float(fit.quantile(0.60))),
        "pace_fit_q70": fit.ge(float(fit.quantile(0.70))),
        "consensus_and_value_q55": consensus & value.ge(float(value.quantile(0.55))),
        "readable_consensus_formal_q55": consensus
        & readable.ge(float(readable.quantile(0.45)))
        & formal.ge(float(formal.quantile(0.55))),
        "strong_label_only": df["pace_pair_gate_label"].eq("pace_pair_strong"),
        "watch_or_strong_label": df["pace_pair_gate_label"].isin(["pace_pair_strong", "pace_pair_watch"]),
    }
    return {k: v.fillna(False) for k, v in masks.items()}


def summarize_policies(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    masks = policy_masks(df)
    overall = []
    by_year = []
    by_surface = []
    by_ticket = []
    for policy, mask in masks.items():
        sub = df.loc[mask].copy()
        overall.append(metrics(sub, policy))
        for year, g in sub.groupby("year", dropna=False):
            by_year.append(metrics(g, policy, str(year)))
        for surface, g in sub.groupby("surface", dropna=False):
            by_surface.append(metrics(g, policy, str(surface)))
        for ticket_type, g in sub.groupby("ticket_type", dropna=False):
            by_ticket.append(metrics(g, policy, str(ticket_type)))
    return {
        "policy_overall": pd.DataFrame(overall).sort_values("roi", ascending=False),
        "policy_by_year": pd.DataFrame(by_year).sort_values(["policy", "segment"]) if by_year else pd.DataFrame(),
        "policy_by_surface": pd.DataFrame(by_surface).sort_values(["policy", "segment"]) if by_surface else pd.DataFrame(),
        "policy_by_ticket_type": pd.DataFrame(by_ticket).sort_values(["policy", "segment"]) if by_ticket else pd.DataFrame(),
    }


def score_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for score_col in [
        "continuous_pair_formal_score",
        "continuous_pair_pace_fit_score",
        "continuous_pair_readability_score",
        "continuous_pair_value_score",
        "pair_lap_same_race_fit_score",
        "pair_quinella_score",
        "pair_score",
        "ticket_hit_prob",
    ]:
        if score_col in df.columns:
            rows.append(
                {
                    "score": score_col,
                    "auc_hit": auc_score(df["_hit"], df[score_col]),
                    "corr_return": float(pd.to_numeric(df[score_col], errors="coerce").corr(df["_return"])),
                    "mean": float(pd.to_numeric(df[score_col], errors="coerce").mean()),
                }
            )
    return pd.DataFrame(rows).sort_values("auc_hit", ascending=False)


def decile_table(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["formal_score_decile"] = pd.qcut(
        work["continuous_pair_formal_score"].rank(method="first"),
        q=10,
        labels=[f"d{i}" for i in range(1, 11)],
    )
    rows = []
    for decile, g in work.groupby("formal_score_decile", observed=False):
        rows.append(metrics(g, "formal_score_decile", str(decile)))
    return pd.DataFrame(rows)


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


def write_report(out_dir: Path, outputs: dict[str, pd.DataFrame], summary: dict[str, Any]) -> None:
    lines = [
        "# Priority Pace Pair Gate v1",
        "",
        "## Policy Overall",
        to_md_table(outputs["policy_overall"]),
        "",
        "## Policy By Year",
        to_md_table(outputs["policy_by_year"], max_rows=60),
        "",
        "## Policy By Surface",
        to_md_table(outputs["policy_by_surface"], max_rows=60),
        "",
        "## Score Diagnostics",
        to_md_table(outputs["score_diagnostics"]),
        "",
        "## Formal Score Deciles",
        to_md_table(outputs["formal_score_deciles"], max_rows=20),
        "",
        "## Notes",
        "- This is an overlay on existing formal/S-priority tickets, not a new ticket generator.",
        "- Policies with very high top_return_share are still unstable even if ROI is high.",
        "- If adopted, use the score first as dashboard/LINE explanation and shadow gate, then promote only after live OOS accumulation.",
        "",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate continuous pace x pair-fit gates on current formal tickets.")
    parser.add_argument("--tickets", default=str(DEFAULT_TICKETS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tickets = prepare(read_csv(Path(args.tickets)))
    outputs = summarize_policies(tickets)
    outputs["score_diagnostics"] = score_diagnostics(tickets)
    outputs["formal_score_deciles"] = decile_table(tickets)
    outputs["tickets_scored"] = tickets

    for name, frame in outputs.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    best = outputs["policy_overall"].head(1).replace({np.nan: None}).to_dict(orient="records")
    practical_frame = outputs["policy_overall"][
        outputs["policy_overall"]["tickets"].ge(40)
        & outputs["policy_overall"]["top_return_share"].le(0.35)
        & outputs["policy_overall"]["roi_ex_top1_return"].ge(1.0)
    ].copy()
    practical = practical_frame.head(1).replace({np.nan: None}).to_dict(orient="records")
    base = outputs["policy_overall"].loc[
        outputs["policy_overall"]["policy"].eq("base_formal_buy_all")
    ].replace({np.nan: None}).to_dict(orient="records")
    summary = {
        "tickets_csv": str(Path(args.tickets)),
        "output_dir": str(out_dir),
        "input_tickets": int(len(tickets)),
        "input_races": int(tickets["race_id"].nunique()),
        "best_policy": best,
        "practical_policy_candidate": practical,
        "base_policy": base,
        "recommendation": (
            "Use continuous_pair_formal_score and continuous_pair_pace_fit_score as dashboard/shadow explanations first. "
            "formal_score_q70 is the practical candidate on historical tickets, but it should remain shadow-only until more live/OOS races accumulate."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir, outputs, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
