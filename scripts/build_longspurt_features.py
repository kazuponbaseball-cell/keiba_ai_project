from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features.longspurt import LongspurtConfig, build_longspurt_feature_set
from src.utils.paths import ensure_dir, project_path


def _rename_existing(df: pd.DataFrame, aliases: dict[str, list[str]]) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        if canonical in df.columns:
            continue
        for candidate in candidates:
            if candidate in df.columns:
                rename[candidate] = canonical
                break
    return df.rename(columns=rename)


def normalize_races(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "race_id": ["レースID(新/馬番無)", "レースID"],
        "date": ["日付"],
        "venue": ["場所"],
        "surface": ["芝・ダ"],
        "distance": ["距離"],
        "going": ["馬場状態"],
        "race_laps": ["レースラップ", "レースラップタイム"],
    }
    return _rename_existing(df, aliases)


def normalize_runners(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "race_id": ["レースID(新/馬番無)", "レースID"],
        "horse_id": ["血統登録番号"],
        "date": ["日付"],
        "finish": ["確定着順", "着順"],
        "popularity": ["人気"],
        "field_size": ["出走頭数", "頭数"],
        "corner1": ["1角"],
        "corner2": ["2角"],
        "corner3": ["3角"],
        "corner4": ["4角", "4角.1"],
        "final3f_rank": ["上り3F順"],
    }
    return _rename_existing(df, aliases)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build long-spurt race and horse features from CSV inputs.")
    parser.add_argument("--races-csv", required=True)
    parser.add_argument("--runners-csv", required=True)
    parser.add_argument("--output-dir", default="outputs/features/longspurt")
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--min-group-size", type=int, default=20)
    parser.add_argument("--fast-quantile", type=float, default=0.25)
    args = parser.parse_args()

    races = normalize_races(pd.read_csv(project_path(args.races_csv), encoding=args.encoding, low_memory=False))
    runners = normalize_runners(pd.read_csv(project_path(args.runners_csv), encoding=args.encoding, low_memory=False))
    config = LongspurtConfig(min_group_size=args.min_group_size, fast_quantile=args.fast_quantile)
    race_features, horse_features = build_longspurt_feature_set(races, runners, config)

    out_dir = ensure_dir(project_path(args.output_dir))
    race_path = out_dir / "race_longspurt_features.csv"
    horse_path = out_dir / "horse_longspurt_features.csv"
    race_features.to_csv(race_path, index=False, encoding="utf-8-sig")
    horse_features.to_csv(horse_path, index=False, encoding="utf-8-sig")
    print(f"race_features={race_path}")
    print(f"horse_features={horse_path}")
    print(f"races={len(race_features)}")
    print(f"runner_rows={len(horse_features)}")


if __name__ == "__main__":
    main()
