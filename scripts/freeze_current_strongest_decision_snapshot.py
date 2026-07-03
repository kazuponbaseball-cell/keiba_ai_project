from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, dtype=str, low_memory=False)


def text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def num(value: Any, default: float = np.nan) -> float:
    try:
        raw = text(value).replace(",", "")
        if raw == "":
            return default
        return float(raw)
    except Exception:
        return default


def boolish(value: Any) -> bool:
    raw = text(value).lower()
    if raw in {"1", "1.0", "true", "yes", "y"}:
        return True
    try:
        return float(raw) > 0
    except Exception:
        return False


def read_dashboard_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    body = path.read_text(encoding="utf-8", errors="ignore")
    marker = "const payload = "
    if marker not in body:
        return {}
    start = body.index(marker) + len(marker)
    payload, _ = json.JSONDecoder().raw_decode(body[start:])
    return payload


def parse_dt(value: Any) -> datetime | None:
    raw = text(value)
    if not raw:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y%m%d_%H%M%S",
    ):
        try:
            return datetime.strptime(raw, fmt)
        except Exception:
            pass
    parsed = pd.to_datetime(raw, errors="coerce")
    if pd.notna(parsed):
        return parsed.to_pydatetime()
    return None


def race_post_at(race: dict[str, Any]) -> datetime | None:
    date_key = text(race.get("dateKey"))
    start_time = text(race.get("startTime"))
    if len(date_key) != 8 or ":" not in start_time:
        return None
    try:
        hour, minute = [int(part) for part in start_time.split(":", 1)]
        return datetime(int(date_key[:4]), int(date_key[4:6]), int(date_key[6:8]), hour, minute)
    except Exception:
        return None


def snapshot_id(captured_at: str, decision_label: str, race_id: str, decision_key: str) -> str:
    raw = f"{captured_at}|{decision_label}|{race_id}|{decision_key}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"{race_id}_{decision_label}_{captured_at}_{digest}"


def race_id_col(frame: pd.DataFrame) -> str | None:
    for col in ("race_id", "raceId", "レースID"):
        if col in frame.columns:
            return col
    return None


def pick_top_candidate(candidates: pd.DataFrame, race_id: str) -> dict[str, Any]:
    if candidates.empty:
        return {}
    rid_col = race_id_col(candidates)
    if not rid_col:
        return {}
    race = candidates[candidates[rid_col].astype(str).eq(str(race_id))].copy()
    if race.empty:
        return {}
    sort_cols = [c for c in ("strongest_current_score", "min_odds_margin_ratio", "runtime_expected_roi") if c in race.columns]
    for col in sort_cols:
        race[f"_{col}"] = pd.to_numeric(race[col], errors="coerce").fillna(-9999)
    if sort_cols:
        race = race.sort_values([f"_{c}" for c in sort_cols], ascending=[False] * len(sort_cols))
    return race.iloc[0].to_dict()


def selected_rows_for_race(tickets: pd.DataFrame, race_id: str) -> pd.DataFrame:
    if tickets.empty:
        return pd.DataFrame()
    rid_col = race_id_col(tickets)
    if not rid_col:
        return pd.DataFrame()
    return tickets[tickets[rid_col].astype(str).eq(str(race_id))].copy()


def selected_stake(rows: pd.DataFrame) -> float:
    for col in ("runtime_stake_yen", "scaled_stake_yen", "eval_stake_yen", "stake_yen"):
        if col in rows.columns:
            return float(pd.to_numeric(rows[col], errors="coerce").fillna(0).sum())
    return 0.0


def track_rows_by_key(track: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    if track.empty:
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for _, row in track.iterrows():
        date_key = text(row.get("effective_date")) or text(row.get("observed_date"))
        venue = text(row.get("venue"))
        if date_key and venue:
            out[(date_key, venue)] = row.to_dict()
    return out


def gate_failures(candidate: dict[str, Any], decision_key: str) -> list[str]:
    if not candidate:
        return ["no_candidate_row"]
    failures: list[str] = []
    anchor_rank = num(candidate.get("anchor_ai_rank_num"), 99)
    partner_rank = num(candidate.get("partner_ai_rank_num"), 99)
    margin = num(candidate.get("min_odds_margin_ratio"), 0)
    expected_roi = num(candidate.get("runtime_expected_roi"), 0)
    skip = num(candidate.get("skip_risk_score"), 1)
    danger = num(candidate.get("ticket_danger_popular_score"), 1)
    danger_pair = num(candidate.get("ticket_danger_popular_in_pair_score"), danger)
    gelding = num(candidate.get("gelding_pair_risk_score"), 0)
    quinella = num(candidate.get("pair_quinella_score"), 0)
    score = num(candidate.get("strongest_current_score"), 0)
    front5 = num(candidate.get("projected_front5_prob"), 0)
    pace = num(candidate.get("pace_fit_pair_score"), 0)
    workout = num(candidate.get("workout_pair_score"), 0)
    live_odds = num(candidate.get("live_odds"), 9999)
    first_unc = num(candidate.get("first_condition_pair_uncertainty_score"), 0)
    difficulty = num(candidate.get("race_difficulty_score"), 0)
    parity_anchor = num(candidate.get("anchor_strongest_feature_parity_ready"), 0)
    parity_partner = num(candidate.get("partner_strongest_feature_parity_ready"), 0)
    soft_heavy = boolish(candidate.get("anchor_runtime_soft_heavy_flag")) or boolish(candidate.get("partner_runtime_soft_heavy_flag"))
    hakodate = boolish(candidate.get("anchor_runtime_hakodate_flag")) or boolish(candidate.get("partner_runtime_hakodate_flag"))
    venue = text(candidate.get("venue_eval")) or text(candidate.get("anchor_場所")) or text(candidate.get("anchor_蝣ｴ謇"))
    going = text(candidate.get("anchor_runtime_going_class")) or text(candidate.get("anchor_runtime_going"))
    tokyo_wet = ("Tokyo" in venue or "東京" in venue or "譚ｱ莠ｬ" in venue) and going in {"Yielding", "Soft", "Heavy", "稍重", "重", "不良", "遞埼㍾", "驥・", "荳崎憶"}

    base_checks = [
        ("anchor_rank_out", anchor_rank <= 3),
        ("partner_rank_out", partner_rank <= 8),
        ("feature_parity_missing", parity_anchor >= 1 and parity_partner >= 1),
        ("margin_below_base", margin >= 0.95),
        ("expected_roi_below_base", expected_roi >= 1.35),
        ("skip_risk_high_base", skip <= 0.52),
        ("danger_popular_high_base", danger <= 0.70),
        ("gelding_risk_high", gelding <= 0.35),
        ("pair_quinella_low", quinella >= 0.52),
    ]
    strict_checks = [
        ("score_below_strict", score >= 0.86),
        ("margin_below_strict", margin >= 2.50),
        ("skip_risk_high_strict", skip <= 0.45),
        ("front5_low", front5 >= 0.60),
        ("pace_fit_low", pace >= 0.35),
        ("workout_score_low", workout >= 0.20),
        ("odds_too_high", live_odds <= 120),
        ("first_condition_uncertainty", first_unc < 0.45 or (margin >= 3.0 and expected_roi >= 1.6)),
        ("danger_popular_in_pair", danger_pair < 0.42 or (margin >= 3.0 and expected_roi >= 1.6)),
        ("race_difficulty_high", difficulty <= 0.58 or (margin >= 3.2 and expected_roi >= 1.75)),
    ]
    for label, ok in base_checks + strict_checks:
        if not ok:
            failures.append(label)
    if hakodate:
        failures.append("hakodate_guard")
    if soft_heavy:
        failures.append("soft_heavy_guard")
    if tokyo_wet:
        failures.append("tokyo_wet_guard")
    if decision_key == "buy":
        return []
    return failures[:20]


def track_uncertainty(minutes_since_update: float, changed: bool, available: bool) -> float:
    if not available:
        return 1.0
    score = 0.0
    if not np.isfinite(minutes_since_update):
        score += 0.35
    elif minutes_since_update > 90:
        score += 0.45
    elif minutes_since_update > 45:
        score += 0.25
    elif minutes_since_update > 20:
        score += 0.10
    if changed:
        score += 0.45
    return min(1.0, score)


def append_csv(path: Path, rows: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path, dtype=str, low_memory=False)
        combined = pd.concat([existing, rows.astype(str)], ignore_index=True, sort=False)
    else:
        combined = rows.astype(str)
    combined.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze immutable current-strongest decision snapshots for shadow validation.")
    parser.add_argument("--dashboard-html", default="outputs/ui/live_odds_dashboard.html")
    parser.add_argument("--candidates-csv", default="outputs/analysis/current_strongest_runtime_v1/current_strongest_all_candidates.csv")
    parser.add_argument("--tickets-csv", default="outputs/analysis/current_strongest_runtime_v1/selected_after_live_safety.csv")
    parser.add_argument("--track-condition-csv", default="data/processed/live_track_conditions/current_track_conditions.csv")
    parser.add_argument("--track-change-summary-json", default="outputs/analysis/live_track_conditions/track_condition_change_summary.json")
    parser.add_argument("--decision-label", default="manual")
    parser.add_argument("--race-ids", nargs="*", default=[])
    parser.add_argument("--captured-at", default="")
    parser.add_argument("--output-csv", default="data/processed/live_decision_snapshots/current_strongest_decision_snapshots.csv")
    parser.add_argument("--latest-csv", default="outputs/analysis/current_strongest_runtime_v1/decision_snapshot_latest.csv")
    parser.add_argument("--summary-json", default="outputs/analysis/current_strongest_runtime_v1/decision_snapshot_summary.json")
    args = parser.parse_args()

    captured_dt = parse_dt(args.captured_at) or datetime.now()
    captured_at = captured_dt.isoformat(timespec="seconds")
    captured_stamp = captured_dt.strftime("%Y%m%d_%H%M%S")
    payload = read_dashboard_payload(project_path(args.dashboard_html))
    candidates = read_csv_safe(project_path(args.candidates_csv))
    tickets = read_csv_safe(project_path(args.tickets_csv))
    track = read_csv_safe(project_path(args.track_condition_csv))
    track_map = track_rows_by_key(track)
    changed_keys: set[tuple[str, str]] = set()
    change_path = project_path(args.track_change_summary_json)
    if change_path.exists():
        try:
            changes = json.loads(change_path.read_text(encoding="utf-8")).get("changes", [])
            for item in changes:
                changed_keys.add((text(item.get("effective_date")), text(item.get("venue"))))
        except Exception:
            changed_keys = set()

    race_filter = {text(rid) for rid in args.race_ids if text(rid)}
    decision_by_race = {text(row.get("raceId")): row for row in payload.get("raceDecisionRows", [])}
    races = payload.get("races", [])
    if race_filter:
        races = [race for race in races if text(race.get("raceId")) in race_filter]

    rows: list[dict[str, Any]] = []
    for race in races:
        race_id = text(race.get("raceId"))
        if not race_id:
            continue
        decision = decision_by_race.get(race_id, {})
        selected = selected_rows_for_race(tickets, race_id)
        has_final_buy = not selected.empty
        decision_key = "buy" if has_final_buy else text(decision.get("key")) or "skip"
        candidate = pick_top_candidate(candidates, race_id)
        failures = gate_failures(candidate, decision_key)
        post_at = race_post_at(race)
        minutes_to_post = ((post_at - captured_dt).total_seconds() / 60.0) if post_at else np.nan
        date_key = text(race.get("dateKey")) or race_id[:8]
        venue = text(race.get("venue"))
        track_row = track_map.get((date_key, venue), {})
        track_fetched_at = text(track_row.get("fetched_at"))
        track_dt = parse_dt(track_fetched_at)
        minutes_since_track = ((captured_dt - track_dt).total_seconds() / 60.0) if track_dt else np.nan
        track_available = bool(track_row)
        changed = (date_key, venue) in changed_keys
        top_pair = decision.get("topPair") if isinstance(decision.get("topPair"), dict) else {}
        row = {
            "decision_snapshot_id": snapshot_id(captured_stamp, args.decision_label, race_id, decision_key),
            "captured_at": captured_at,
            "captured_stamp": captured_stamp,
            "decision_label": args.decision_label,
            "model_version": "current_strongest_mcs_pbo_strict",
            "race_id": race_id,
            "date_key": date_key,
            "race_label": text(race.get("raceLabel")),
            "venue": venue,
            "race_no": text(race.get("raceNo")),
            "race_name": text(race.get("raceName")),
            "post_time": text(race.get("startTime")),
            "minutes_to_post": round(minutes_to_post, 2) if np.isfinite(minutes_to_post) else "",
            "decision_key": decision_key,
            "decision_label_ui": text(decision.get("label")) if not has_final_buy else "買い目あり",
            "decision_reason": text(decision.get("reason")),
            "priority_score": num(decision.get("priorityScore"), np.nan),
            "dashboard_top_score": num(decision.get("topScore"), np.nan),
            "dashboard_margin": num(decision.get("margin"), np.nan),
            "dashboard_expected_roi": num(decision.get("expectedRoi"), np.nan),
            "dashboard_skip_risk": num(decision.get("skipRisk"), np.nan),
            "dashboard_front5": num(decision.get("front5"), np.nan),
            "top_pair_a_no": text(top_pair.get("aNo")),
            "top_pair_b_no": text(top_pair.get("bNo")),
            "top_pair_odds": num(top_pair.get("odds"), np.nan),
            "final_buy_tickets": int(len(selected)),
            "final_buy_stake_yen": round(selected_stake(selected), 0),
            "falloff_reason_count": len(failures),
            "falloff_reasons": "|".join(failures),
            "single_gate_failure": 1 if len(failures) == 1 else 0,
            "multi_gate_failure": 1 if len(failures) > 1 else 0,
            "track_available": int(track_available),
            "track_weather": text(track_row.get("weather")),
            "track_turf_going": text(track_row.get("turf_going")),
            "track_dirt_going": text(track_row.get("dirt_going")),
            "going_observed_at": track_fetched_at,
            "minutes_since_going_update": round(minutes_since_track, 2) if np.isfinite(minutes_since_track) else "",
            "going_changed_since_previous_observation": int(changed),
            "track_state_uncertainty": track_uncertainty(minutes_since_track, changed, track_available),
        }
        for col in [
            "strongest_current_score",
            "min_odds_margin_ratio",
            "runtime_expected_roi",
            "expected_roi_after_slippage",
            "skip_risk_score",
            "projected_front5_prob",
            "pace_fit_pair_score",
            "workout_pair_score",
            "live_odds",
            "ticket_hit_prob",
            "pair_quinella_score",
            "ticket_danger_popular_score",
            "ticket_danger_popular_in_pair_score",
            "race_difficulty_score",
            "late_value_survives_score",
            "odds_timeline_ready",
            "late_odds_drop_rate",
            "late_odds_drift_rate",
            "track_lap_regime",
            "lap_track_shadow_label",
            "lap_track_shadow_score",
            "lap_track_shadow_note",
            "lap_signal_strict_gap_low",
            "lap_signal_goodrun_min_high",
            "lap_signal_role_low_collision",
            "first_condition_pair_uncertainty_score",
            "first_condition_pair_impressive_prev_score",
            "anchor_ai_rank_num",
            "partner_ai_rank_num",
            "anchor_runtime_going_class",
            "partner_runtime_going_class",
            "anchor_runtime_soft_heavy_flag",
            "partner_runtime_soft_heavy_flag",
            "anchor_runtime_hakodate_flag",
            "partner_runtime_hakodate_flag",
            "anchor_strongest_feature_parity_ready",
            "partner_strongest_feature_parity_ready",
        ]:
            row[col] = candidate.get(col, "")
        rows.append(row)

    latest = pd.DataFrame(rows)
    latest_path = project_path(args.latest_csv)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest.to_csv(latest_path, index=False, encoding="utf-8-sig")
    if not latest.empty:
        append_csv(project_path(args.output_csv), latest)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "decision_label": args.decision_label,
        "captured_at": captured_at,
        "races": int(len(latest)),
        "output_csv": str(project_path(args.output_csv)),
        "latest_csv": str(latest_path),
        "decision_counts": latest["decision_key"].value_counts(dropna=False).to_dict() if not latest.empty else {},
        "single_gate_failures": int(latest["single_gate_failure"].sum()) if "single_gate_failure" in latest else 0,
        "track_uncertain_races": int((pd.to_numeric(latest.get("track_state_uncertainty", pd.Series(dtype=float)), errors="coerce").fillna(0) >= 0.45).sum()) if not latest.empty else 0,
    }
    summary_path = project_path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
