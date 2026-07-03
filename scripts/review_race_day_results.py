from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def strip_tags(fragment: str) -> str:
    text = re.sub(r"<[^>]+>", "", fragment)
    return html.unescape(text).replace("\xa0", " ").strip()


def safe_num(value: Any) -> float:
    try:
        if value is None or value == "":
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def read_dashboard_payload(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    marker = "const payload = "
    start = text.index(marker) + len(marker)
    payload, _ = json.JSONDecoder().raw_decode(text[start:])
    return payload


def parse_result_html(path: Path, race_id: str) -> list[dict[str, Any]]:
    text = path.read_bytes().decode("shift_jis", errors="ignore")
    m = re.search(r'<div id="race_result"[\s\S]*?<tbody>([\s\S]*?)</tbody>', text)
    if not m:
        return []
    rows: list[dict[str, Any]] = []
    for row in re.findall(r"<tr[\s\S]*?</tr>", m.group(1)):
        def td(cls: str) -> str:
            found = re.search(rf'<td class="{cls}">([\s\S]*?)</td>', row)
            return strip_tags(found.group(1)) if found else ""

        place = safe_num(td("place"))
        horse_no = safe_num(td("num"))
        horse_name = td("horse")
        pop = safe_num(td("pop"))
        f_time = safe_num(td("f_time"))
        corner_html = ""
        corner_m = re.search(r'<td class="corner">([\s\S]*?)</td>', row)
        if corner_m:
            corner_html = corner_m.group(1)
        corners = [safe_num(strip_tags(x)) for x in re.findall(r"<li[^>]*>([\s\S]*?)</li>", corner_html)]
        corners = [int(x) for x in corners if pd.notna(x)]
        rows.append(
            {
                "race_id": str(race_id),
                "finish": int(place) if pd.notna(place) else np.nan,
                "horse_no": int(horse_no) if pd.notna(horse_no) else np.nan,
                "horse_name_result": horse_name,
                "actual_pop": int(pop) if pd.notna(pop) else np.nan,
                "corner3": corners[-2] if len(corners) >= 2 else (corners[0] if corners else np.nan),
                "corner4": corners[-1] if corners else np.nan,
                "estimated_final3f": f_time,
            }
        )
    return rows


def style_from_corner(corner4: float, field_size: float) -> str:
    if pd.isna(corner4) or pd.isna(field_size):
        return "unknown"
    if corner4 <= 3:
        return "front"
    if corner4 <= max(5, field_size * 0.35):
        return "stalker"
    if corner4 >= field_size * 0.65:
        return "closer"
    return "mid"


def actual_bias_for_race(part: pd.DataFrame, field_size: float) -> dict[str, Any]:
    top3 = part[part["finish"].between(1, 3)].copy()
    if top3.empty:
        return {}
    top3["actual_style"] = top3["corner4"].map(lambda x: style_from_corner(x, field_size))
    winner = top3[top3["finish"].eq(1)].head(1)
    front_like = top3["actual_style"].isin(["front", "stalker"]).sum()
    closer_like = top3["actual_style"].eq("closer").sum()
    avg_corner4 = top3["corner4"].mean()
    if front_like >= 2 and avg_corner4 <= max(5, field_size * 0.4):
        shape = "front_stalker"
    elif closer_like >= 2 or avg_corner4 >= field_size * 0.55:
        shape = "closer"
    else:
        shape = "mixed"
    return {
        "actual_bias_shape": shape,
        "top3_numbers": "-".join(str(int(x)) for x in top3["horse_no"].dropna().tolist()),
        "top3_corner4": "-".join(str(int(x)) for x in top3["corner4"].dropna().tolist()),
        "top3_avg_corner4": float(avg_corner4) if pd.notna(avg_corner4) else np.nan,
        "top3_front_stalker_count": int(front_like),
        "top3_closer_count": int(closer_like),
        "winner_no": int(winner["horse_no"].iloc[0]) if not winner.empty else np.nan,
        "winner_corner4": float(winner["corner4"].iloc[0]) if not winner.empty else np.nan,
        "winner_pop": float(winner["actual_pop"].iloc[0]) if not winner.empty else np.nan,
    }


def predicted_bias_from_decision(row: pd.Series) -> str:
    key = str(row.get("decision_key") or row.get("key") or "")
    reason = str(row.get("decision_reason") or row.get("reason") or "")
    front5 = safe_num(row.get("projected_front5_prob"))
    pace_fit = safe_num(row.get("pace_fit_pair_score"))
    collapse = safe_num(row.get("race_difficulty_score"))
    if "差し" in reason or key == "closer_watch":
        return "closer_watch"
    if pd.notna(front5) and front5 >= 0.78 and (pd.isna(pace_fit) or pace_fit >= 0.55):
        return "front_stalker"
    if pd.notna(collapse) and collapse >= 0.65:
        return "unstable"
    return "mixed"


def mismatch_label(predicted: str, actual: str) -> str:
    if predicted == "closer_watch" and actual == "front_stalker":
        return "差し想定→実際は前残り"
    if predicted == "front_stalker" and actual == "closer":
        return "前目想定→実際は差し"
    if predicted == "front_stalker" and actual == "front_stalker":
        return "前目一致"
    if predicted == "closer_watch" and actual == "closer":
        return "差し一致"
    if actual == "mixed":
        return "混戦/判定難"
    return "大ズレなし"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard-html", default="outputs/ui/live_odds_dashboard.html")
    parser.add_argument("--result-track-csv", default="outputs/analysis/current_live_pnl/current_result_track_conditions.csv")
    parser.add_argument("--decision-csv", default="outputs/analysis/current_strongest_runtime_v1/decision_snapshot_latest.csv")
    parser.add_argument(
        "--decision-history-csv",
        default="data/processed/live_decision_snapshots/current_strongest_decision_snapshots.csv",
    )
    parser.add_argument("--pnl-detail-csv", default="outputs/analysis/current_live_pnl/current_live_pnl_detail.csv")
    parser.add_argument("--out-dir", default="outputs/analysis/race_day_review_20260627")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = read_dashboard_payload(ROOT / args.dashboard_html)
    races = pd.DataFrame(payload.get("races", []))
    singles = pd.DataFrame(payload.get("singleRows", []))
    if not singles.empty:
        singles = singles.rename(
            columns={
                "raceId": "race_id",
                "horseNo": "horse_no",
                "horseName": "horse_name",
                "aiRank": "ai_rank",
                "aiScore": "ai_score",
                "winOdds": "win_odds",
            }
        )
        for c in ["horse_no", "ai_rank", "ai_score", "frontRunning", "closing", "win_odds", "popularity"]:
            if c in singles.columns:
                singles[c] = pd.to_numeric(singles[c], errors="coerce")

    result_track = pd.read_csv(ROOT / args.result_track_csv, encoding="utf-8-sig", low_memory=False)
    result_rows: list[dict[str, Any]] = []
    for _, row in result_track.iterrows():
        result_path = Path(str(row.get("result_path") or ""))
        if not result_path.exists():
            continue
        result_rows.extend(parse_result_html(result_path, str(row["race_id"])))
    results = pd.DataFrame(result_rows)
    results.to_csv(out_dir / "parsed_results_all_horses.csv", index=False, encoding="utf-8-sig")

    merged = singles.merge(results, on=["race_id", "horse_no"], how="left")
    if "raceId" in races.columns:
        race_meta = races.rename(columns={"raceId": "race_id", "raceNo": "race_no", "fieldSize": "field_size"})
        race_meta = race_meta.rename(
            columns={
                "raceName": "race_name",
                "startTime": "start_time",
                "expectedPace": "race_expected_pace",
                "frontRunnerCount": "front_runner_count",
                "pressureScore": "pressure_score",
                "runtimeGoing": "runtime_going",
                "turfGoing": "turf_going",
                "dirtGoing": "dirt_going",
                "upsetLabel": "upset_label",
                "upsetNote": "upset_note",
            }
        )
        keep = [
            c
            for c in [
                "race_id",
                "venue",
                "race_no",
                "race_name",
                "start_time",
                "surface",
                "distance",
                "field_size",
                "race_expected_pace",
                "front_runner_count",
                "pressure_score",
                "runtime_going",
                "turf_going",
                "dirt_going",
                "upset_label",
                "upset_note",
            ]
            if c in race_meta.columns
        ]
        merged = merged.merge(race_meta[keep], on="race_id", how="left")
    merged["is_win"] = merged["finish"].eq(1)
    merged["is_top2"] = merged["finish"].between(1, 2)
    merged["is_top3"] = merged["finish"].between(1, 3)
    merged["actual_style"] = merged.apply(lambda r: style_from_corner(r.get("corner4"), safe_num(r.get("field_size"))), axis=1)
    merged.to_csv(out_dir / "ai_rank_results_all_horses.csv", index=False, encoding="utf-8-sig")

    rank_rows = []
    for label, mask in {
        "AI1": merged["ai_rank"].eq(1),
        "AI2": merged["ai_rank"].eq(2),
        "AI3": merged["ai_rank"].eq(3),
        "AI1-3": merged["ai_rank"].between(1, 3),
        "AI1-5": merged["ai_rank"].between(1, 5),
    }.items():
        part = merged[mask].copy()
        rank_rows.append(
            {
                "bucket": label,
                "horses": int(len(part)),
                "races": int(part["race_id"].nunique()),
                "win_rate": float(part["is_win"].mean()) if len(part) else np.nan,
                "top2_rate": float(part["is_top2"].mean()) if len(part) else np.nan,
                "top3_rate": float(part["is_top3"].mean()) if len(part) else np.nan,
                "avg_finish": float(part["finish"].mean()) if len(part) else np.nan,
                "avg_actual_pop": float(part["actual_pop"].mean()) if len(part) else np.nan,
                "avg_win_odds": float(part["win_odds"].mean()) if len(part) else np.nan,
            }
        )
    rank_summary = pd.DataFrame(rank_rows)
    rank_summary.to_csv(out_dir / "ai_rank_summary.csv", index=False, encoding="utf-8-sig")

    venue_rows = []
    for keys, part in merged.groupby(["venue", "surface"], dropna=False):
        venue, surface = keys
        ai1 = part[part["ai_rank"].eq(1)]
        top3 = part[part["ai_rank"].between(1, 3)]
        venue_rows.append(
            {
                "venue": venue,
                "surface": surface,
                "races": int(part["race_id"].nunique()),
                "ai1_win_rate": float(ai1["is_win"].mean()) if len(ai1) else np.nan,
                "ai1_top3_rate": float(ai1["is_top3"].mean()) if len(ai1) else np.nan,
                "ai_top3_top3_rate": float(top3["is_top3"].mean()) if len(top3) else np.nan,
                "top3_actual_front_stalker_rate": float(part[part["finish"].between(1, 3)]["actual_style"].isin(["front", "stalker"]).mean()),
                "top3_actual_closer_rate": float(part[part["finish"].between(1, 3)]["actual_style"].eq("closer").mean()),
                "avg_top3_corner4": float(part[part["finish"].between(1, 3)]["corner4"].mean()),
            }
        )
    venue_summary = pd.DataFrame(venue_rows)
    venue_summary.to_csv(out_dir / "venue_surface_review.csv", index=False, encoding="utf-8-sig")

    decision = pd.read_csv(ROOT / args.decision_csv, encoding="utf-8-sig", low_memory=False)
    if "race_id" not in decision.columns and "raceId" in decision.columns:
        decision = decision.rename(columns={"raceId": "race_id"})
    history_path = ROOT / args.decision_history_csv
    if history_path.exists():
        history = pd.read_csv(history_path, encoding="utf-8-sig", low_memory=False)
        if "race_id" in history.columns and "minutes_to_post" in history.columns:
            history["minutes_to_post_num"] = pd.to_numeric(history["minutes_to_post"], errors="coerce")
            target_races = set(races.get("raceId", pd.Series(dtype=str)).astype(str))
            prepost = history[
                history["race_id"].astype(str).isin(target_races)
                & history["minutes_to_post_num"].ge(0)
            ].copy()
            if not prepost.empty:
                decision = (
                    prepost.sort_values(["race_id", "minutes_to_post_num"], kind="mergesort")
                    .groupby("race_id", as_index=False)
                    .head(1)
                    .drop(columns=["minutes_to_post_num"], errors="ignore")
                )
    bias_rows = []
    for race_id, part in merged.groupby("race_id"):
        race = part.iloc[0]
        field_size = safe_num(race.get("field_size"))
        actual = actual_bias_for_race(part, field_size)
        dec = decision[decision["race_id"].astype(str).eq(str(race_id))]
        dec_row = dec.iloc[0] if not dec.empty else pd.Series(dtype=object)
        predicted = predicted_bias_from_decision(dec_row)
        actual_shape = actual.get("actual_bias_shape", "")
        bias_rows.append(
            {
                "race_id": race_id,
                "venue": race.get("venue"),
                "race_no": race.get("race_no"),
                "race_name": race.get("race_name"),
                "surface": race.get("surface"),
                "distance": race.get("distance"),
                "runtime_going": race.get("runtime_going"),
                "expected_pace": race.get("race_expected_pace", race.get("expectedPace")),
                "front_runner_count": race.get("front_runner_count"),
                "pressure_score": race.get("pressure_score"),
                "decision_label": dec_row.get("decision_label_ui", dec_row.get("label", "")),
                "decision_key": dec_row.get("decision_key", dec_row.get("key", "")),
                "predicted_bias_shape": predicted,
                **actual,
                "mismatch": mismatch_label(predicted, str(actual_shape)),
            }
        )
    bias_review = pd.DataFrame(bias_rows).sort_values(["venue", "race_no"])
    bias_review.to_csv(out_dir / "pace_bias_review_by_race.csv", index=False, encoding="utf-8-sig")

    pnl = pd.read_csv(ROOT / args.pnl_detail_csv, encoding="utf-8-sig", low_memory=False)
    pnl_summary = (
        pnl.groupby(["decisionGroup", "decisionLabel"], dropna=False)
        .agg(
            tickets=("raceId", "size"),
            races=("raceId", "nunique"),
            stake=("stakeYen", "sum"),
            payout=("payoutYen", "sum"),
            hits=("hit", "sum"),
        )
        .reset_index()
    )
    pnl_summary["profit"] = pnl_summary["payout"] - pnl_summary["stake"]
    pnl_summary["roi_pct"] = np.where(pnl_summary["stake"] > 0, pnl_summary["payout"] / pnl_summary["stake"] * 100.0, np.nan)
    pnl_summary.to_csv(out_dir / "ticket_pnl_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "out_dir": str(out_dir),
        "races": int(merged["race_id"].nunique()),
        "horses": int(len(merged)),
        "ai_rank_summary": rank_summary.to_dict(orient="records"),
        "ticket_pnl_summary": pnl_summary.to_dict(orient="records"),
        "bias_mismatch_counts": bias_review["mismatch"].value_counts(dropna=False).to_dict(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
