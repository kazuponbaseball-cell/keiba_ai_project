from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.apply_optimized_factor_gate import DEFAULT_PARAMS
from scripts.compare_priority_a_gate_profiles import PROFILES, _mask as profile_mask
from scripts.optimize_priority_s_betting_policy import _metrics
from scripts.summarize_factor_roi_effectiveness import _apply_gate as optimized_factor_mask
from src.utils.paths import ensure_dir, project_path


PROFILE_LABELS = {
    "strong": "強気",
    "standard": "標準",
    "broad": "広め",
    "skip": "見送り",
}


def _profile(name: str) -> dict:
    for profile in PROFILES:
        if profile["profile"] == name:
            return profile
    raise KeyError(name)


def _ticket_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["race_id"].astype(str)
        + ":"
        + pd.to_numeric(df["anchor_no"], errors="coerce").astype("Int64").astype(str)
        + "-"
        + pd.to_numeric(df["partner_no"], errors="coerce").astype("Int64").astype(str)
        + ":"
        + df["ticket_type"].astype(str)
    )


def _profile_metrics(tickets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for profile in ["strong", "standard", "broad", "skip"]:
        part = tickets[tickets["operation_profile"].eq(profile)].copy()
        row = _metrics(part, profile)
        row["profile_label"] = PROFILE_LABELS[profile]
        rows.append(row)
    actionable = tickets[~tickets["operation_profile"].eq("skip")].copy()
    row = _metrics(actionable, "actionable_total")
    row["profile_label"] = "購入候補合計"
    rows.append(row)
    return pd.DataFrame(rows)


def _bundle_metrics(tickets: pd.DataFrame) -> pd.DataFrame:
    bundles = [
        ("strong_bundle", "強気", ["strong"]),
        ("standard_bundle", "標準", ["strong", "standard"]),
        ("broad_bundle", "広め", ["strong", "standard", "broad"]),
    ]
    rows = []
    for policy, label, profiles in bundles:
        part = tickets[tickets["operation_profile"].isin(profiles)].copy()
        row = _metrics(part, policy)
        row["profile_label"] = label
        row["included_profiles"] = ",".join(profiles)
        rows.append(row)
    return pd.DataFrame(rows)


def _race_summary(tickets: pd.DataFrame) -> pd.DataFrame:
    actionable = tickets[~tickets["operation_profile"].eq("skip")].copy()
    if actionable.empty:
        return pd.DataFrame()
    order = {"strong": 3, "standard": 2, "broad": 1}
    work = actionable.copy()
    work["_profile_order"] = work["operation_profile"].map(order).fillna(0)
    by = (
        work.sort_values(["race_id", "_profile_order", "pair_quinella_score", "market_overlay_score"], ascending=[True, False, False, False])
        .groupby("race_id", as_index=False)
        .agg(
            race_profile=("operation_profile", "first"),
            race_profile_label=("operation_profile_label", "first"),
            tickets=("ticket_type", "size"),
            stake_yen=("stake_yen", "sum"),
            expected_ticket_types=("ticket_type", lambda s: ",".join(sorted(set(map(str, s))))),
            anchor_name=("anchor_name", "first"),
            partner_names=("partner_name", lambda s: " / ".join(dict.fromkeys(map(str, s)).keys())),
            max_pair_quinella_score=("pair_quinella_score", "max"),
            max_market_overlay_score=("market_overlay_score", "max"),
            min_anchor_vertical_overpopular=("anchor_vertical_overpopular_risk_score", "min"),
            max_partner_vertical_value=("partner_vertical_underpopular_value_score", "max"),
            race_difficulty_score=("race_difficulty_score", "max"),
            race_pace_collapse=("race_pace_collapse", "max"),
        )
    )
    return by.sort_values(["race_profile", "max_pair_quinella_score"], ascending=[True, False])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build operational strong/standard/broad/skip ticket profile outputs for UI and live use.")
    parser.add_argument("--tickets-csv", default="outputs/analysis/vertical_context_roi_v1/baseline_selected_tickets_with_vertical_scores.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/operational_ticket_profiles_v1")
    parser.add_argument("--params-json", default=None, help="Optional JSON object overriding optimized factor gate params.")
    args = parser.parse_args()

    tickets = pd.read_csv(project_path(args.tickets_csv), dtype={"race_id": str}, low_memory=False)
    tickets = tickets.copy()
    tickets["_ticket_key"] = _ticket_key(tickets)
    params = dict(DEFAULT_PARAMS)
    if args.params_json:
        params.update(json.loads(args.params_json))

    broad = tickets[optimized_factor_mask(tickets, params)].copy()
    middle = broad[profile_mask(broad, _profile("middle_175_candidate"))].copy()
    strong = broad[profile_mask(broad, _profile("strict_190"))].copy()

    middle_keys = set(middle["_ticket_key"])
    strong_keys = set(strong["_ticket_key"])
    broad_keys = set(broad["_ticket_key"])

    tickets["operation_profile"] = np.select(
        [
            tickets["_ticket_key"].isin(strong_keys),
            tickets["_ticket_key"].isin(middle_keys),
            tickets["_ticket_key"].isin(broad_keys),
        ],
        ["strong", "standard", "broad"],
        default="skip",
    )
    tickets["operation_profile_label"] = tickets["operation_profile"].map(PROFILE_LABELS)
    tickets["operation_action"] = np.where(tickets["operation_profile"].eq("skip"), "skip", "buy_candidate")
    tickets["operation_strength_rank"] = tickets["operation_profile"].map({"strong": 3, "standard": 2, "broad": 1, "skip": 0}).fillna(0).astype(int)

    out_dir = ensure_dir(project_path(args.output_dir))
    tickets.drop(columns=["_ticket_key"], errors="ignore").to_csv(out_dir / "ticket_profiles.csv", index=False, encoding="utf-8-sig")
    actionable = tickets[~tickets["operation_profile"].eq("skip")].drop(columns=["_ticket_key"], errors="ignore")
    actionable.to_csv(out_dir / "actionable_ticket_profiles.csv", index=False, encoding="utf-8-sig")
    race_summary = _race_summary(tickets.drop(columns=["_ticket_key"], errors="ignore"))
    race_summary.to_csv(out_dir / "race_profile_summary.csv", index=False, encoding="utf-8-sig")
    metrics = _profile_metrics(tickets.drop(columns=["_ticket_key"], errors="ignore"))
    metrics.to_csv(out_dir / "profile_metrics.csv", index=False, encoding="utf-8-sig")
    bundle_metrics = _bundle_metrics(tickets.drop(columns=["_ticket_key"], errors="ignore"))
    bundle_metrics.to_csv(out_dir / "profile_bundle_metrics.csv", index=False, encoding="utf-8-sig")

    payload = {
        "output_dir": str(out_dir),
        "params": params,
        "counts": tickets["operation_profile"].value_counts().to_dict(),
        "profile_metrics": metrics.to_dict(orient="records"),
        "profile_bundle_metrics": bundle_metrics.to_dict(orient="records"),
        "race_profiles": race_summary["race_profile"].value_counts().to_dict() if not race_summary.empty else {},
        "files": {
            "ticket_profiles": str(out_dir / "ticket_profiles.csv"),
            "actionable_ticket_profiles": str(out_dir / "actionable_ticket_profiles.csv"),
            "race_profile_summary": str(out_dir / "race_profile_summary.csv"),
            "profile_metrics": str(out_dir / "profile_metrics.csv"),
            "profile_bundle_metrics": str(out_dir / "profile_bundle_metrics.csv"),
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
