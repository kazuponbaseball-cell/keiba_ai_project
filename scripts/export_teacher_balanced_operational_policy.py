from __future__ import annotations

from pathlib import Path

import pandas as pd

from optimize_teacher_false_positive_filter import (
    SCORED,
    add_false_positive_features,
    apply_filter,
    base_select,
    metrics,
    stake_selected,
)


OUT = Path("outputs/analysis/teacher_balanced_operational_policy_v1")


BALANCED_POLICY = {
    "coverage": 0.08,
    "max_tickets_per_race": 1,
    "similarity_min": 0.46,
    "quality_min": 0.42,
    "fragile_max": 0.42,
    # Wide is allowed to be a little softer on joint quality. Over-tightening
    # removes the front-position/value patterns this strategy is trying to keep.
    "wide_partner_odds_max": 45.0,
    "wide_roi_min": 0.8,
    "wide_roi_max": 6.0,
    "wide_quote_min": 250.0,
    "wide_quote_max": 900.0,
    "wide_joint_min": 0.50,
    # Umaren needs stricter heat control because false positives explode faster.
    "umaren_partner_odds_min": 5.0,
    "umaren_partner_odds_max": 25.0,
    "umaren_roi_min": 1.0,
    "umaren_roi_max": 12.0,
    "umaren_quote_min": 800.0,
    "umaren_quote_max": 2400.0,
    "umaren_joint_min": 0.52,
}


def select(df: pd.DataFrame) -> pd.DataFrame:
    base = base_select(
        df,
        float(BALANCED_POLICY["coverage"]),
        int(BALANCED_POLICY["max_tickets_per_race"]),
    )
    return stake_selected(apply_filter(base, BALANCED_POLICY))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scored = pd.read_csv(SCORED, dtype={"race_id": str}, low_memory=False)
    scored = add_false_positive_features(scored)

    train = scored[scored["year"].eq(2025)].copy()
    test = scored[scored["year"].eq(2026)].copy()
    all_years = scored[scored["year"].isin([2025, 2026])].copy()

    train_tickets = select(train)
    test_tickets = select(test)
    all_tickets = pd.concat([train_tickets, test_tickets], ignore_index=True)

    train_tickets.to_csv(OUT / "balanced_train_2025_tickets.csv", index=False, encoding="utf-8-sig")
    test_tickets.to_csv(OUT / "balanced_test_2026_tickets.csv", index=False, encoding="utf-8-sig")
    all_tickets.to_csv(OUT / "balanced_2025_2026_tickets.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame(
        [
            metrics(train_tickets, "balanced_train_2025"),
            metrics(test_tickets, "balanced_test_2026"),
            metrics(all_tickets, "balanced_2025_2026"),
        ]
    )
    summary.to_csv(OUT / "balanced_summary.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([BALANCED_POLICY]).to_csv(OUT / "balanced_policy_params.csv", index=False, encoding="utf-8-sig")

    print("BALANCED OPERATIONAL POLICY")
    print(summary.to_string(index=False))
    print("\nPOLICY")
    print(pd.Series(BALANCED_POLICY).to_string())
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
