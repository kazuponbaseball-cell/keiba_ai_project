from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.strict_pair_probability_roi_protocol import (  # noqa: E402
    build_raw_probability_features,
    metrics,
    train_val_holdout,
    walkforward,
)
from scripts.validate_closer_course_context_pair_probability import (  # noqa: E402
    apply_variant as apply_closer_variant,
    attach_closer_context,
)
from scripts.validate_front_load_course_context_pair_probability import (  # noqa: E402
    apply_variant as apply_front_variant,
    attach_front_load_context,
)
from scripts.validate_lap_s_priority_combo_pair_probability import (  # noqa: E402
    apply_variant as apply_s_priority_variant,
    attach_s_priority_features,
    read_pair_universe,
)


OUT = ROOT / "outputs/analysis/production_context_combo_pair_probability_v1"


def aggregate_tickets(tickets: pd.DataFrame, label: str) -> dict:
    out = metrics(tickets, label)
    year_rows = []
    if not tickets.empty:
        for year, part in tickets.groupby("year", sort=True):
            row = metrics(part, f"{label}_{year}")
            year_rows.append(
                {
                    "year": int(year),
                    "roi": row["roi"],
                    "races": row["races"],
                    "tickets": row["tickets"],
                    "profit_yen": row["profit_yen"],
                    "max_drawdown_yen": row["max_drawdown_yen"],
                }
            )
    out["year_metrics"] = year_rows
    out["min_year_roi"] = min((row["roi"] for row in year_rows), default=0.0)
    return out


def run_variant(source: pd.DataFrame, name: str, *, front: str, s_priority: str, closer: str) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = source.copy()
    if front != "baseline":
        work = apply_front_variant(work, front)
    if s_priority != "baseline":
        work = apply_s_priority_variant(work, s_priority)
    if closer != "baseline":
        work = apply_closer_variant(work, closer)
    scored = build_raw_probability_features(work)
    train_grid, wf_summary, wf_tickets = walkforward(scored)
    holdout = train_val_holdout(scored)
    total = aggregate_tickets(wf_tickets, f"{name}_walkforward_total")
    total.update(
        {
            "variant": name,
            "front_variant": front,
            "s_priority_variant": s_priority,
            "closer_variant": closer,
        }
    )
    return total, train_grid, wf_summary, holdout


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    source = read_pair_universe()
    source = attach_front_load_context(source)
    source = attach_s_priority_features(source)
    source = attach_closer_context(source)
    source.to_csv(OUT / "pair_universe_with_production_context_features.csv", index=False, encoding="utf-8-sig")

    variants = {
        "baseline": ("baseline", "baseline", "baseline"),
        "production_context_v1": ("context_switch_light", "s_combo_strong", "closer_switch_strong"),
        "production_context_v1_no_closer": ("context_switch_light", "s_combo_strong", "baseline"),
        "production_context_v1_light_lap": ("context_switch_light", "s_combo_light", "closer_switch_strong"),
    }

    summaries = []
    holdouts = []
    for name, (front, s_priority, closer) in variants.items():
        total, train_grid, wf_summary, holdout = run_variant(
            source,
            name,
            front=front,
            s_priority=s_priority,
            closer=closer,
        )
        summaries.append(total)
        train_grid.to_csv(OUT / f"{name}_walkforward_train_top100.csv", index=False, encoding="utf-8-sig")
        wf_summary.to_csv(OUT / f"{name}_walkforward_summary.csv", index=False, encoding="utf-8-sig")
        holdout["variant"] = name
        holdouts.append(holdout)
        holdout.to_csv(OUT / f"{name}_train2024_val2025_hold2026_grid.csv", index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(summaries).sort_values(["roi", "profit_yen"], ascending=False)
    summary_df.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    if holdouts:
        pd.concat(holdouts, ignore_index=True).to_csv(OUT / "holdout_all_variants.csv", index=False, encoding="utf-8-sig")
    payload = {
        "created_at": pd.Timestamp.now().isoformat(),
        "purpose": "Combined validation for production-reflected front context, S-priority lap, and closer course context modifiers.",
        "summary": summary_df.to_dict(orient="records"),
        "outputs": [
            "summary.csv",
            "pair_universe_with_production_context_features.csv",
            "*_walkforward_summary.csv",
            "*_train2024_val2025_hold2026_grid.csv",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
