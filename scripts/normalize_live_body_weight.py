from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _project_path(path: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    p = Path(path)
    return p if p.is_absolute() else root / p


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace("+", "", regex=False), errors="coerce")


def _normalize_manual_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"race_id": str}, low_memory=False)
    rename = {
        "馬番": "horse_no",
        "umaban": "horse_no",
        "horse_number": "horse_no",
        "馬体重": "body_weight",
        "bataiju": "body_weight",
        "weight": "body_weight",
        "増減": "body_weight_diff",
        "馬体重増減": "body_weight_diff",
        "zogen_sa": "body_weight_diff",
        "diff": "body_weight_diff",
        "発表時刻": "snapshot_at",
        "happyo_tsukihi_jifun": "snapshot_at",
    }
    out = df.rename(columns={k: v for k, v in rename.items() if k in df.columns}).copy()
    required = {"race_id", "horse_no", "body_weight"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"body weight csv missing columns: {sorted(missing)}")
    if "body_weight_diff" not in out.columns:
        out["body_weight_diff"] = pd.NA
    if "snapshot_at" not in out.columns:
        out["snapshot_at"] = pd.NA
    out["horse_no"] = _num(out["horse_no"]).astype("Int64")
    out["body_weight"] = _num(out["body_weight"])
    out["body_weight_diff"] = _num(out["body_weight_diff"])
    out = out[out["race_id"].notna() & out["horse_no"].notna() & out["body_weight"].notna()].copy()
    out["race_id"] = out["race_id"].astype(str)
    out = out.sort_values(["race_id", "snapshot_at", "horse_no"])
    out = out.drop_duplicates(["race_id", "horse_no"], keep="last")
    return out[["race_id", "horse_no", "body_weight", "body_weight_diff", "snapshot_at"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize live body weight CSV for the dashboard.")
    parser.add_argument("--manual-csv", required=True)
    parser.add_argument("--output-csv", default="data/processed/live_body_weight/body_weight_latest.csv")
    parser.add_argument("--summary-json", default="outputs/analysis/live_body_weight/summary.json")
    args = parser.parse_args()

    out = _normalize_manual_csv(_project_path(args.manual_csv))
    output = _project_path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False, encoding="utf-8-sig")

    summary = {
        "output_csv": str(output),
        "rows": int(len(out)),
        "races": int(out["race_id"].nunique()) if not out.empty else 0,
    }
    summary_path = _project_path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
