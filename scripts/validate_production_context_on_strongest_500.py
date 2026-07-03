from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_age_weight_surface_overlay import (  # noqa: E402
    apply_policy,
    json_ready,
    metrics,
    num,
    recalc_return,
)


OUT = ROOT / "outputs/analysis/production_context_on_strongest_500_v1"
CONTEXT_UNIVERSE = (
    ROOT
    / "outputs/analysis/production_context_combo_pair_probability_v1/pair_universe_with_production_context_features.csv"
)


POLICIES = {
    "mcs_v4_age_weight_578": {
        "tickets_csv": ROOT / "outputs/analysis/age_weight_surface_overlay_mcs_v4_v1/best_policy_tickets.csv",
        "policy": {
            "name": "age_positive_or_edge_p0.55_m2.0",
            "keep_positive_or_strong_edge": True,
            "positive_threshold": 0.55,
            "strong_min_margin": 2.0,
            "strong_min_expected_roi": 1.55,
        },
    },
    "fixed_edge_v2_556": {
        "tickets_csv": ROOT / "outputs/analysis/age_weight_surface_overlay_fixed_edge_v2/best_policy_tickets.csv",
        "policy": {
            "name": "strong_edge_only_m2.0",
            "strong_edge_only": True,
            "strong_min_margin": 2.0,
            "strong_min_expected_roi": 1.55,
        },
    },
}


CONTEXT_COLS = [
    "race_id",
    "_pair_lo",
    "_pair_hi",
    "front_load_survival_pair_fit",
    "front_load_collapse_front_risk",
    "front_load_collapse_closer_fit",
    "front_load_course_net_score",
    "front_load_course_label",
    "s_waveform_fit_score",
    "s_class_lap_fit_score",
    "s_distance_lap_fit_score",
    "s_priority_lap_support_score",
    "s_priority_lap_caution_score",
    "s_priority_lap_net_score",
    "s_priority_lap_label",
    "closer_course_pair_fit",
    "closer_course_front_risk",
    "closer_course_net_score",
    "closer_course_label",
]


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def normalize_race_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d+)", expand=False).str.zfill(16)


def read_context_universe() -> pd.DataFrame:
    if not CONTEXT_UNIVERSE.exists():
        raise FileNotFoundError(
            f"missing context universe: {CONTEXT_UNIVERSE}. "
            "Run scripts/validate_production_context_combo_pair_probability.py first."
        )
    ctx = pd.read_csv(
        CONTEXT_UNIVERSE,
        dtype={"race_id": str},
        usecols=lambda c: c in set(CONTEXT_COLS),
        low_memory=False,
    )
    ctx["race_id"] = normalize_race_id(ctx["race_id"])
    for col in ["_pair_lo", "_pair_hi"]:
        ctx[col] = pd.to_numeric(ctx[col], errors="coerce").astype("Int64")
    return ctx.drop_duplicates(["race_id", "_pair_lo", "_pair_hi"], keep="last")


def attach_context(tickets: pd.DataFrame, ctx: pd.DataFrame) -> pd.DataFrame:
    out = tickets.copy()
    out["race_id"] = normalize_race_id(out["race_id"])
    a = pd.to_numeric(out["anchor_no"], errors="coerce")
    b = pd.to_numeric(out["partner_no"], errors="coerce")
    out["_pair_lo"] = np.minimum(a, b).astype("Int64")
    out["_pair_hi"] = np.maximum(a, b).astype("Int64")
    out = out.merge(ctx, on=["race_id", "_pair_lo", "_pair_hi"], how="left", suffixes=("", "_ctx"))

    for col in CONTEXT_COLS:
        if col in {"race_id", "_pair_lo", "_pair_hi"}:
            continue
        if col not in out.columns:
            out[col] = 0.0 if not col.endswith("_label") else ""

    survival = num(out, "front_load_survival_pair_fit", 0).fillna(0).clip(0, 1)
    collapse = num(out, "front_load_collapse_front_risk", 0).fillna(0).clip(0, 1)
    closer_rescue = num(out, "front_load_collapse_closer_fit", 0).fillna(0).clip(0, 1)
    front_adj = (0.014 * survival - 0.018 * collapse + 0.008 * closer_rescue).clip(-0.05, 0.05)

    s_adj = (
        0.032 * num(out, "s_waveform_fit_score", 0).fillna(0).clip(0, 1)
        + 0.032 * num(out, "s_class_lap_fit_score", 0).fillna(0).clip(0, 1)
        + 0.032 * num(out, "s_distance_lap_fit_score", 0).fillna(0).clip(0, 1)
        - 0.032 * num(out, "s_priority_lap_caution_score", 0).fillna(0).clip(0, 1)
    ).clip(-0.06, 0.06)

    closer_adj = (
        0.040 * num(out, "closer_course_pair_fit", 0).fillna(0).clip(0, 1)
        - 0.048 * num(out, "closer_course_front_risk", 0).fillna(0).clip(0, 1)
    ).clip(-0.06, 0.06)

    prod_adj = (front_adj + s_adj + closer_adj).clip(-0.10, 0.10)
    out["prod_front_context_probability_adjustment"] = front_adj
    out["prod_s_priority_probability_adjustment"] = s_adj
    out["prod_closer_course_probability_adjustment"] = closer_adj
    out["production_context_probability_adjustment"] = prod_adj
    out["production_context_label"] = np.select(
        [
            prod_adj.ge(0.030),
            prod_adj.le(-0.030),
            num(out, "s_priority_lap_support_score", 0).fillna(0).ge(0.62)
            & num(out, "s_priority_lap_caution_score", 0).fillna(0).le(0.34),
            survival.ge(0.62) & collapse.le(0.42),
            num(out, "closer_course_pair_fit", 0).fillna(0).ge(0.30)
            & num(out, "closer_course_front_risk", 0).fillna(0).le(0.36),
        ],
        [
            "prod_context_support",
            "prod_context_caution",
            "prod_s_lap_support",
            "prod_front_survival_support",
            "prod_closer_context_support",
        ],
        default="prod_context_neutral",
    )
    out["production_context_matched"] = out["front_load_course_label"].notna() | out["s_priority_lap_label"].notna()
    return out


def evaluate_guard(df: pd.DataFrame, label: str, keep: pd.Series) -> tuple[dict, pd.DataFrame]:
    out = df.copy()
    out["_eval_stake"] = num(out, "_eval_stake", 0).fillna(0).where(keep, 0.0)
    out["_eval_return"] = recalc_return(out, out["_eval_stake"])
    return metrics(out, label), out


def add_year_metrics(df: pd.DataFrame, label: str) -> list[dict]:
    rows = []
    for year, part in df.groupby(num(df, "year", 0), dropna=False):
        if pd.isna(year):
            continue
        rows.append(metrics(part, f"{label}_{int(year)}"))
    return rows


def profit_concentration(df: pd.DataFrame) -> dict:
    selected = df[num(df, "_eval_stake", 0).fillna(0).gt(0)].copy()
    if selected.empty:
        return {"top1_hit_profit_share": 0.0, "top3_hit_profit_share": 0.0, "top5_hit_profit_share": 0.0}
    selected["_profit"] = num(selected, "_eval_return", 0).fillna(0) - num(selected, "_eval_stake", 0).fillna(0)
    total_profit = float(selected["_profit"].sum())
    hit_profits = selected.loc[selected["_profit"].gt(0), "_profit"].sort_values(ascending=False)
    if total_profit <= 0 or hit_profits.empty:
        return {"top1_hit_profit_share": 0.0, "top3_hit_profit_share": 0.0, "top5_hit_profit_share": 0.0}
    return {
        "top1_hit_profit_share": float(hit_profits.head(1).sum() / total_profit),
        "top3_hit_profit_share": float(hit_profits.head(3).sum() / total_profit),
        "top5_hit_profit_share": float(hit_profits.head(5).sum() / total_profit),
    }


def run_dataset(name: str, spec: dict, ctx: pd.DataFrame) -> tuple[list[dict], dict]:
    tickets_path = project_path(spec["tickets_csv"])
    tickets = pd.read_csv(tickets_path, dtype={"race_id": str}, low_memory=False)
    tickets["race_id"] = normalize_race_id(tickets["race_id"])
    if "year" not in tickets.columns:
        tickets["year"] = pd.to_numeric(tickets["race_id"].str[:4], errors="coerce")
    tickets["_base_stake"] = num(tickets, "runtime_stake_yen", 0).fillna(0)
    tickets["_base_return"] = num(tickets, "runtime_return_yen", 0).fillna(0)

    policy_eval = apply_policy(tickets, spec["policy"])
    enriched = attach_context(policy_eval, ctx)
    selected_base = num(enriched, "_eval_stake", 0).fillna(0).gt(0)
    adj = num(enriched, "production_context_probability_adjustment", 0).fillna(0)
    margin = num(enriched, "min_odds_margin_ratio", 0).fillna(0)
    expected_roi = num(enriched, "runtime_expected_roi", 0).fillna(0)

    guards = {
        "same_policy_reproduced": selected_base,
        "context_non_negative": selected_base & adj.ge(0.0),
        "context_not_caution_m003": selected_base & adj.ge(-0.030),
        "context_support_p001": selected_base & adj.ge(0.010),
        "context_support_p002": selected_base & adj.ge(0.020),
        "context_support_p003": selected_base & adj.ge(0.030),
        "context_support_or_very_strong_edge": selected_base
        & (adj.ge(0.010) | (margin.ge(3.0) & expected_roi.ge(1.80))),
    }

    rows: list[dict] = []
    best_by_roi: tuple[float, pd.DataFrame, str] | None = None
    for guard_name, keep in guards.items():
        label = f"{name}__{guard_name}"
        row, evaluated = evaluate_guard(enriched, label, keep)
        row.update(
            {
                "dataset": name,
                "guard": guard_name,
                "context_matched_rate": float(enriched.loc[selected_base, "production_context_matched"].mean())
                if selected_base.any()
                else 0.0,
                "avg_context_adjustment": float(adj[selected_base].mean()) if selected_base.any() else 0.0,
                "selected_avg_context_adjustment": float(adj[keep].mean()) if keep.any() else 0.0,
            }
        )
        row.update(profit_concentration(evaluated))
        rows.append(row)
        if best_by_roi is None or row["roi"] > best_by_roi[0]:
            best_by_roi = (row["roi"], evaluated, guard_name)
        evaluated.loc[num(evaluated, "_eval_stake", 0).fillna(0).gt(0)].to_csv(
            OUT / f"{label}_tickets.csv", index=False, encoding="utf-8-sig"
        )

    year_rows = []
    for guard_name, keep in guards.items():
        label = f"{name}__{guard_name}"
        _, evaluated = evaluate_guard(enriched, label, keep)
        for row in add_year_metrics(evaluated, label):
            row.update({"dataset": name, "guard": guard_name})
            year_rows.append(row)
    pd.DataFrame(year_rows).to_csv(OUT / f"{name}_yearly_metrics.csv", index=False, encoding="utf-8-sig")

    details = {
        "input": str(tickets_path),
        "rows": int(len(tickets)),
        "policy": spec["policy"],
        "base_selected_rows": int(selected_base.sum()),
        "context_matched_rate": float(enriched.loc[selected_base, "production_context_matched"].mean())
        if selected_base.any()
        else 0.0,
    }
    return rows, details


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = read_context_universe()
    all_rows: list[dict] = []
    details: dict[str, dict] = {}
    for name, spec in POLICIES.items():
        rows, detail = run_dataset(name, spec, ctx)
        all_rows.extend(rows)
        details[name] = detail

    metrics_df = pd.DataFrame(all_rows).sort_values(["dataset", "roi"], ascending=[True, False])
    metrics_df.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    payload = {
        "created_at": pd.Timestamp.now().isoformat(),
        "purpose": (
            "Same-basis validation: apply production context support/caution guards to the previously "
            "reported 500%+ strongest selected-ticket policies."
        ),
        "context_universe": str(CONTEXT_UNIVERSE),
        "details": details,
        "summary": metrics_df.to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json_ready(payload), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
