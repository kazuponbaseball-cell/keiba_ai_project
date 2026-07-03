from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.workout_knowledge import (
    evaluate_workout_knowledge,
    prepare_workouts_for_knowledge,
    select_entry_workouts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate entries with trainer-specific workout knowledge rules.")
    parser.add_argument("--entry-csv", required=True)
    parser.add_argument("--workouts-csv", default="data/processed/target/workouts_20230101_20260613.csv")
    parser.add_argument("--output-csv", default="outputs/analysis/workout_knowledge_entry_eval.csv")
    parser.add_argument("--output-md", default="outputs/analysis/workout_knowledge_entry_eval.md")
    parser.add_argument("--lookback-days", type=int, default=21)
    args = parser.parse_args()

    entries = pd.read_csv(args.entry_csv, encoding="utf-8-sig", low_memory=False)
    workouts = pd.read_csv(args.workouts_csv, encoding="utf-8-sig", low_memory=False)
    workouts = prepare_workouts_for_knowledge(workouts)

    rows = []
    markdown = ["# 調教ナレッジ評価", ""]
    for _, entry in entries.iterrows():
        selected = select_entry_workouts(entry, workouts, lookback_days=args.lookback_days)
        result = evaluate_workout_knowledge(entry, selected)
        rows.append(
            {
                "馬名": result["horse_name"],
                "厩舎": result["trainer"],
                "調教師コード": result["trainer_code"],
                "芝・ダート": result["surface"],
                "調教内容": result["workout_content"],
                "該当パターン": result["matched_pattern"],
                "加点要素": " / ".join(result["plus_factors"]),
                "減点要素": " / ".join(result["minus_factors"]),
                "総合評価": result["grade"],
                "grade_score": result["grade_score"],
                "短評": result["comment"],
            }
        )
        markdown.extend(_format_markdown(result))

    out = pd.DataFrame(rows).sort_values(["grade_score", "馬名"], ascending=[False, True])
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")
    output_md.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"rows": int(len(out)), "output_csv": str(output_csv), "output_md": str(output_md)}, ensure_ascii=False, indent=2))


def _format_markdown(result: dict) -> list[str]:
    plus = result["plus_factors"] or ["なし"]
    minus = result["minus_factors"] or ["なし"]
    lines = [
        f"## 【{result['horse_name']}】",
        "",
        f"厩舎：{result['trainer']}",
        "",
        f"芝・ダート：{result['surface']}",
        "",
        f"調教内容：{result['workout_content']}",
        "",
        f"該当パターン：{result['matched_pattern']}",
        "",
        "加点要素：",
    ]
    lines.extend([f"・{item}" for item in plus])
    lines.extend(["", "減点要素："])
    lines.extend([f"・{item}" for item in minus])
    lines.extend(["", f"総合評価：{result['grade']}", "", f"短評：{result['comment']}", ""])
    return lines


if __name__ == "__main__":
    main()
