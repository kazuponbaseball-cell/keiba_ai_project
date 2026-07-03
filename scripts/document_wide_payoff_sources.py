from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
XLSX = PROJECT_ROOT / "docs" / "PC-KEIBAテーブル定義書.xlsx"
OUT_DIR = PROJECT_ROOT / "outputs" / "analysis" / "wide_payoff_source_check"


def extract_columns(sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(XLSX, sheet_name=sheet_name, header=None)
    starts = raw.index[raw.iloc[:, 0].astype(str).eq("No.")].tolist()
    if not starts:
        raise ValueError(f"No column header found in sheet: {sheet_name}")
    rows = []
    for _, row in raw.iloc[starts[0] + 1 :].iterrows():
        no = row.iloc[0]
        if pd.isna(no):
            continue
        try:
            int_no = int(no)
        except Exception:
            continue
        rows.append(
            {
                "no": int_no,
                "logical_name": row.iloc[1],
                "physical_name": row.iloc[2],
                "data_type": row.iloc[3],
                "not_null": row.iloc[4] if len(row) > 4 else None,
                "note": row.iloc[6] if len(row) > 6 else None,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payoff = extract_columns("払戻")
    wide_payoff = payoff[
        payoff["logical_name"].astype(str).str.contains("ワイド", na=False)
        | payoff["physical_name"].astype(str).str.contains("wide", na=False)
    ].copy()
    o3 = extract_columns("オッズ3(ワイド)")
    sokuho_o3 = extract_columns("速報系データ オッズ3(ワイド)")

    payoff.to_csv(OUT_DIR / "jvd_hr_columns.csv", index=False, encoding="utf-8-sig")
    wide_payoff.to_csv(OUT_DIR / "jvd_hr_wide_columns.csv", index=False, encoding="utf-8-sig")
    o3.to_csv(OUT_DIR / "jvd_o3_columns.csv", index=False, encoding="utf-8-sig")
    sokuho_o3.to_csv(OUT_DIR / "apd_sokuho_o3_columns.csv", index=False, encoding="utf-8-sig")

    summary = {
        "payoff_table": "jvd_hr",
        "wide_odds_table": "jvd_o3",
        "realtime_wide_odds_table": "apd_sokuho_o3",
        "wide_payoff_columns": wide_payoff[["logical_name", "physical_name", "data_type"]].to_dict(orient="records"),
        "wide_odds_columns": o3[["logical_name", "physical_name", "data_type"]].to_dict(orient="records"),
        "note": "The current all-race CSV lacks wide payoff columns, but PC-KEIBA/TARGET definitions show wide payoff fields in jvd_hr.",
    }
    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
