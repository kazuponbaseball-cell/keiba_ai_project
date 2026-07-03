from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FEATURES = [
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus_rebuilt_20260623/"
    "body_weight_backfilled/owner_breeder_enriched/"
    "train_features_with_same_day_bias_v3_retro_body_breeder.csv",
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus_rebuilt_20260623/"
    "body_weight_backfilled/owner_breeder_enriched/"
    "test_features_with_same_day_bias_v3_retro_body_breeder.csv",
]

DEFAULT_TICKETS = [
    "outputs/analysis/mcs_pbo_runtime_overlay_v4_operational_gates/recommended_runtime_tickets.csv",
    "outputs/analysis/mcs_pbo_runtime_overlay_v4_operational_gates/recommended_all_tickets.csv",
]

RUNNER_COLS = [
    "日付S",
    "場所",
    "レース名",
    "性別",
    "斤量",
    "頭数",
    "出走頭数",
    "枠番",
    "馬番",
    "人気",
    "単勝オッズ",
    "芝・ダ",
    "単勝配当",
    "複勝配当",
    "レースID(新/馬番無)",
    "血統登録番号",
    "確定着順",
]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, **kwargs)


def num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    raw = frame[col]
    if raw.dtype == object:
        raw = raw.astype(str).str.replace(",", "", regex=False).str.replace("+", "", regex=False)
    return pd.to_numeric(raw, errors="coerce").fillna(default)


def text(frame: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=str)
    return frame[col].astype("string").fillna(default).astype(str).str.strip()


def clean_race_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d+)", expand=False).fillna("").str.zfill(16)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if pd.isna(value):
        return None
    return value


def surface_simple(series: pd.Series) -> pd.Series:
    raw = series.astype("string").fillna("").astype(str)
    return pd.Series(
        np.select(
            [
                raw.str.contains("ダ", regex=False, na=False),
                raw.str.contains("芝", regex=False, na=False),
            ],
            ["dirt", "turf"],
            default="other",
        ),
        index=series.index,
    )


def read_runner_features(paths: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for raw in paths:
        path = project_path(raw)
        if not path.exists():
            raise FileNotFoundError(path)
        header = read_csv(path, nrows=0).columns.tolist()
        usecols = [col for col in RUNNER_COLS if col in header]
        frame = read_csv(path, usecols=usecols)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(
        columns={
            "日付S": "date_s",
            "場所": "venue",
            "レース名": "race_name",
            "性別": "sex",
            "斤量": "carried_weight",
            "頭数": "field_size",
            "出走頭数": "field_size_alt",
            "枠番": "frame_no",
            "馬番": "horse_no",
            "人気": "popularity",
            "単勝オッズ": "win_odds",
            "芝・ダ": "surface",
            "単勝配当": "win_pay",
            "複勝配当": "place_pay",
            "レースID(新/馬番無)": "race_id",
            "血統登録番号": "horse_id",
            "確定着順": "finish",
        }
    )
    df["race_id"] = clean_race_id(df["race_id"])
    df["horse_no"] = num(df, "horse_no").astype("Int64")
    df["frame_no"] = num(df, "frame_no")
    df["field_size"] = num(df, "field_size").where(num(df, "field_size").gt(0), num(df, "field_size_alt"))
    df["year"] = pd.to_numeric(df["race_id"].str.slice(0, 4), errors="coerce")
    df["month"] = pd.to_numeric(df["race_id"].str.slice(4, 6), errors="coerce")
    df["is_summer"] = df["month"].between(6, 9, inclusive="both")
    df["is_female"] = text(df, "sex").str.contains("牝", regex=False, na=False)
    df["surface_simple"] = surface_simple(df["surface"])
    df["current_turf"] = df["surface_simple"].eq("turf")
    df["current_dirt"] = df["surface_simple"].eq("dirt")
    df["carried_weight_num"] = num(df, "carried_weight")
    df["popularity_num"] = num(df, "popularity")
    df["win_odds_num"] = num(df, "win_odds")
    df["finish_num"] = num(df, "finish", 99)
    df["win_pay_num"] = num(df, "win_pay", 0.0)
    df["place_pay_num"] = num(df, "place_pay", 0.0)
    race_weight_median = df.groupby("race_id")["carried_weight_num"].transform("median")
    df["weight_advantage_kg"] = (race_weight_median - df["carried_weight_num"]).fillna(0.0)
    df["lightweight_advantage"] = df["weight_advantage_kg"].ge(1.0)
    df["outer_frame"] = df["frame_no"].ge(7)
    df["outer_third_gate"] = (num(df, "horse_no") / df["field_size"].replace(0, np.nan)).ge(2 / 3)
    df["outer_quarter_gate"] = (num(df, "horse_no") / df["field_size"].replace(0, np.nan)).ge(0.75)

    factor_defs = {
        "outer_frame": df["outer_frame"],
        "outer_third_gate": df["outer_third_gate"],
        "outer_quarter_gate": df["outer_quarter_gate"],
        "summer_female": df["is_summer"] & df["is_female"],
        "summer_female_turf": df["is_summer"] & df["is_female"] & df["current_turf"],
        "summer_female_dirt": df["is_summer"] & df["is_female"] & df["current_dirt"],
        "summer_female_lightweight": df["is_summer"] & df["is_female"] & df["lightweight_advantage"],
        "outer_frame_female": df["outer_frame"] & df["is_female"],
        "outer_frame_summer_female": df["outer_frame"] & df["is_summer"] & df["is_female"],
        "outer_frame_summer_female_turf": df["outer_frame"] & df["is_summer"] & df["is_female"] & df["current_turf"],
        "outer_frame_summer_female_dirt": df["outer_frame"] & df["is_summer"] & df["is_female"] & df["current_dirt"],
        "outer_frame_summer_female_lightweight": (
            df["outer_frame"] & df["is_summer"] & df["is_female"] & df["lightweight_advantage"]
        ),
        "outer_third_summer_female_turf": (
            df["outer_third_gate"] & df["is_summer"] & df["is_female"] & df["current_turf"]
        ),
        "outer_third_summer_female_lightweight": (
            df["outer_third_gate"] & df["is_summer"] & df["is_female"] & df["lightweight_advantage"]
        ),
    }
    for name, mask in factor_defs.items():
        df[name] = mask.fillna(False).astype(int)
    return df.drop_duplicates(["race_id", "horse_no"], keep="last")


def stake_return(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    stake = None
    ret = None
    for col in ["eval_stake_yen", "scaled_stake_yen", "runtime_stake_yen", "stake_yen"]:
        if col in df.columns:
            stake = num(df, col, 0.0)
            break
    for col in ["eval_return_yen", "scaled_return_yen", "runtime_return_yen", "return_yen"]:
        if col in df.columns:
            ret = num(df, col, 0.0)
            break
    if stake is None:
        stake = pd.Series(100.0, index=df.index, dtype=float)
    if ret is None:
        ret = pd.Series(0.0, index=df.index, dtype=float)
    return stake.fillna(0.0), ret.fillna(0.0)


def metrics(rows: pd.DataFrame, label: str, pool_rows: int | None = None) -> dict[str, Any]:
    if rows.empty:
        return {
            "label": label,
            "pool_rows": int(pool_rows or 0),
            "tickets": 0,
            "races": 0,
            "days": 0,
            "hits": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi_pct": None,
            "hit_rate_pct": None,
            "max_return_yen": 0.0,
            "top1_removed_roi_pct": None,
            "top3_removed_roi_pct": None,
        }
    stake, ret = stake_return(rows)
    days = rows["race_id"].astype(str).str.slice(0, 8)
    base_stake = float(stake.sum())
    base_return = float(ret.sum())
    sorted_ret = ret.sort_values(ascending=False).reset_index(drop=True)
    top1_ret = float(sorted_ret.iloc[1:].sum()) if len(sorted_ret) > 1 else 0.0
    top3_ret = float(sorted_ret.iloc[3:].sum()) if len(sorted_ret) > 3 else 0.0
    return {
        "label": label,
        "pool_rows": int(pool_rows if pool_rows is not None else len(rows)),
        "tickets": int(len(rows)),
        "races": int(rows["race_id"].nunique()),
        "days": int(days.nunique()),
        "hits": int(ret.gt(0).sum()),
        "stake_yen": base_stake,
        "return_yen": base_return,
        "profit_yen": base_return - base_stake,
        "roi_pct": base_return / base_stake * 100.0 if base_stake > 0 else None,
        "hit_rate_pct": float(ret.gt(0).mean() * 100.0),
        "avg_return_per_hit_yen": float(ret[ret.gt(0)].mean()) if ret.gt(0).any() else 0.0,
        "max_return_yen": float(ret.max()) if len(ret) else 0.0,
        "top1_return_concentration_pct": float(ret.max() / base_return * 100.0) if base_return > 0 else 0.0,
        "top1_removed_roi_pct": top1_ret / base_stake * 100.0 if base_stake > 0 else None,
        "top3_removed_roi_pct": top3_ret / base_stake * 100.0 if base_stake > 0 else None,
    }


def runner_metrics(rows: pd.DataFrame, label: str, pool_rows: int | None = None) -> dict[str, Any]:
    if rows.empty:
        return {
            "label": label,
            "pool_rows": int(pool_rows or 0),
            "starts": 0,
            "races": 0,
            "win_rate_pct": None,
            "top3_rate_pct": None,
            "win_roi_pct": None,
            "place_roi_pct": None,
        }
    finish = num(rows, "finish_num", 99)
    win_return = num(rows, "win_pay_num", 0.0).where(finish.eq(1), 0.0)
    place_return = num(rows, "place_pay_num", 0.0).where(finish.le(3), 0.0)
    stake = len(rows) * 100.0
    return {
        "label": label,
        "pool_rows": int(pool_rows if pool_rows is not None else len(rows)),
        "starts": int(len(rows)),
        "races": int(rows["race_id"].nunique()),
        "win_rate_pct": float(finish.eq(1).mean() * 100.0),
        "top3_rate_pct": float(finish.le(3).mean() * 100.0),
        "win_roi_pct": float(win_return.sum() / stake * 100.0) if stake > 0 else None,
        "place_roi_pct": float(place_return.sum() / stake * 100.0) if stake > 0 else None,
        "avg_popularity": float(num(rows, "popularity_num", np.nan).mean()),
        "avg_win_odds": float(num(rows, "win_odds_num", np.nan).mean()),
    }


def enrich_tickets(tickets: pd.DataFrame, runners: pd.DataFrame) -> pd.DataFrame:
    out = tickets.copy()
    out["race_id"] = clean_race_id(out["race_id"])
    out["anchor_no"] = num(out, "anchor_no").astype("Int64")
    out["partner_no"] = num(out, "partner_no").astype("Int64")
    factor_cols = [c for c in runners.columns if c.startswith(("outer_", "summer_"))]
    keep = [
        "race_id",
        "horse_no",
        "sex",
        "surface_simple",
        "frame_no",
        "field_size",
        "weight_advantage_kg",
        "popularity_num",
        "win_odds_num",
        *factor_cols,
    ]
    lookup = runners[keep].drop_duplicates(["race_id", "horse_no"], keep="last")
    anchor = lookup.rename(columns={c: f"anchor_{c}" for c in lookup.columns if c not in {"race_id", "horse_no"}})
    partner = lookup.rename(columns={c: f"partner_{c}" for c in lookup.columns if c not in {"race_id", "horse_no"}})
    out = out.merge(anchor, left_on=["race_id", "anchor_no"], right_on=["race_id", "horse_no"], how="left")
    out = out.drop(columns=["horse_no"], errors="ignore")
    out = out.merge(partner, left_on=["race_id", "partner_no"], right_on=["race_id", "horse_no"], how="left")
    out = out.drop(columns=["horse_no"], errors="ignore")
    for factor in factor_cols:
        out[f"ticket_any_{factor}"] = (
            num(out, f"anchor_{factor}", 0.0).eq(1) | num(out, f"partner_{factor}", 0.0).eq(1)
        ).astype(int)
        out[f"ticket_both_{factor}"] = (
            num(out, f"anchor_{factor}", 0.0).eq(1) & num(out, f"partner_{factor}", 0.0).eq(1)
        ).astype(int)
    return out


def segment_tables(enriched: pd.DataFrame, source: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    factor_cols = sorted(
        col.replace("ticket_any_", "")
        for col in enriched.columns
        if col.startswith("ticket_any_")
    )
    policy_rows: list[dict[str, Any]] = [
        {"source": source, "policy": "baseline_all", **metrics(enriched, "baseline_all", len(enriched))}
    ]
    segment_rows: list[dict[str, Any]] = []
    year_rows: list[dict[str, Any]] = []
    for factor in factor_cols:
        flagged = enriched[enriched[f"ticket_any_{factor}"].eq(1)].copy()
        unflagged = enriched[enriched[f"ticket_any_{factor}"].ne(1)].copy()
        policy_rows.append(
            {"source": source, "policy": f"require_{factor}", **metrics(flagged, f"require_{factor}", len(enriched))}
        )
        policy_rows.append(
            {"source": source, "policy": f"exclude_{factor}", **metrics(unflagged, f"exclude_{factor}", len(enriched))}
        )
        segment_rows.append(
            {"source": source, "factor": factor, "segment": "has_factor", **metrics(flagged, f"has_{factor}", len(enriched))}
        )
        segment_rows.append(
            {
                "source": source,
                "factor": factor,
                "segment": "without_factor",
                **metrics(unflagged, f"without_{factor}", len(enriched)),
            }
        )
        if not flagged.empty:
            for year, group in flagged.groupby(flagged["race_id"].astype(str).str.slice(0, 4), dropna=False):
                year_rows.append(
                    {
                        "source": source,
                        "factor": factor,
                        "year": year,
                        **metrics(group, f"{factor}_{year}", len(flagged)),
                    }
                )
    return pd.DataFrame(policy_rows), pd.DataFrame(segment_rows), pd.DataFrame(year_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="*", default=DEFAULT_FEATURES)
    parser.add_argument("--tickets", nargs="*", default=DEFAULT_TICKETS)
    parser.add_argument("--output-dir", default="outputs/analysis/outer_draw_summer_female_roi_v1")
    args = parser.parse_args()

    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runners = read_runner_features(args.features)
    factor_cols = sorted([c for c in runners.columns if c.startswith(("outer_", "summer_"))])

    runner_rows = [{"factor": "baseline_all", **runner_metrics(runners, "baseline_all", len(runners))}]
    for factor in factor_cols:
        runner_rows.append(
            {
                "factor": factor,
                "segment": "has_factor",
                **runner_metrics(runners[runners[factor].eq(1)], f"has_{factor}", len(runners)),
            }
        )
        runner_rows.append(
            {
                "factor": factor,
                "segment": "without_factor",
                **runner_metrics(runners[runners[factor].ne(1)], f"without_{factor}", len(runners)),
            }
        )
    runner_summary = pd.DataFrame(runner_rows)
    runner_summary.to_csv(out_dir / "runner_segments.csv", index=False, encoding="utf-8-sig")

    all_policy: list[pd.DataFrame] = []
    all_segments: list[pd.DataFrame] = []
    all_years: list[pd.DataFrame] = []
    enriched_files: dict[str, str] = {}
    for raw in args.tickets:
        path = project_path(raw)
        if not path.exists():
            continue
        tickets = read_csv(path)
        source = path.stem
        enriched = enrich_tickets(tickets, runners)
        enriched_path = out_dir / f"{source}_enriched.csv"
        enriched.to_csv(enriched_path, index=False, encoding="utf-8-sig")
        enriched_files[source] = str(enriched_path)
        policy, segments, years = segment_tables(enriched, source)
        all_policy.append(policy)
        all_segments.append(segments)
        all_years.append(years)

    policy_df = pd.concat(all_policy, ignore_index=True) if all_policy else pd.DataFrame()
    segments_df = pd.concat(all_segments, ignore_index=True) if all_segments else pd.DataFrame()
    years_df = pd.concat(all_years, ignore_index=True) if all_years else pd.DataFrame()
    policy_df.to_csv(out_dir / "ticket_policy_summary.csv", index=False, encoding="utf-8-sig")
    segments_df.to_csv(out_dir / "ticket_factor_segments.csv", index=False, encoding="utf-8-sig")
    years_df.to_csv(out_dir / "ticket_factor_yearly.csv", index=False, encoding="utf-8-sig")

    preferred = [
        "require_summer_female_turf",
        "require_summer_female_lightweight",
        "require_outer_frame",
        "require_outer_frame_female",
        "require_outer_frame_summer_female",
        "require_outer_frame_summer_female_turf",
        "require_outer_frame_summer_female_lightweight",
        "require_outer_third_summer_female_turf",
        "require_outer_third_summer_female_lightweight",
        "exclude_summer_female_dirt",
    ]
    snapshot = (
        policy_df[policy_df["policy"].isin(preferred)]
        .sort_values(["source", "roi_pct"], ascending=[True, False])
        .to_dict("records")
        if not policy_df.empty
        else []
    )
    summary = {
        "output_dir": str(out_dir),
        "features": args.features,
        "tickets": [str(project_path(p)) for p in args.tickets if project_path(p).exists()],
        "runner_rows": int(len(runners)),
        "runner_races": int(runners["race_id"].nunique()),
        "runner_year_min": int(pd.to_numeric(runners["year"], errors="coerce").min()),
        "runner_year_max": int(pd.to_numeric(runners["year"], errors="coerce").max()),
        "factor_columns": factor_cols,
        "enriched_ticket_files": enriched_files,
        "preferred_policy_snapshot": snapshot,
        "notes": [
            "Static pre-race factors only: frame, horse number, sex, carried weight, season, and surface.",
            "This diagnostic does not change Champion tickets or dashboard logic.",
            "Outer frame is frame number >= 7. Outer third/quarter use horse number percentile within field size.",
        ],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# 外枠・夏牝馬・斤量利 ROI検証",
        "",
        f"- 出力先: `{out_dir}`",
        f"- 対象走数: {summary['runner_rows']:,}走 / {summary['runner_races']:,}R",
        f"- 年度範囲: {summary['runner_year_min']} - {summary['runner_year_max']}",
        "",
        "## 主要ポリシー",
        "",
    ]
    for row in snapshot:
        lines.append(
            "- {source} / {policy}: {tickets}点, {races}R, 的中{hits}, ROI {roi:.1f}%, "
            "top1除外 {top1:.1f}%, top3除外 {top3:.1f}%".format(
                source=row.get("source"),
                policy=row.get("policy"),
                tickets=int(row.get("tickets") or 0),
                races=int(row.get("races") or 0),
                hits=int(row.get("hits") or 0),
                roi=float(row.get("roi_pct") or 0.0),
                top1=float(row.get("top1_removed_roi_pct") or 0.0),
                top3=float(row.get("top3_removed_roi_pct") or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## 判定メモ",
            "",
            "- 夏牝馬×芝、夏牝馬×斤量利は、単独フィルターではなく準候補昇格スコア向き。",
            "- 外枠を足すと点数がさらに薄くなるため、年別安定性とtop3除外ROIが残る場合のみ採用候補。",
            "- 夏牝馬×ダートはプラスではなく警戒・減点側の確認対象。",
        ]
    )
    (out_dir / "review.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
