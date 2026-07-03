from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs/analysis/lap_track_condition_interactions_v1"

TRACK_CSV = ROOT / "data/raw/track_condition_metrics.csv"
RACE_CONTEXT_CSV = ROOT / "outputs/analysis/front_survival_context_v1/race_front_survival_context.csv"
ROLE_TICKETS_CSV = ROOT / "outputs/analysis/lap_waveform_role_goodrun_v1/lap_waveform_role_goodrun_enriched_tickets.csv"
LOAD_TICKETS_CSV = ROOT / "outputs/analysis/lap_waveform_load_tolerance_v1/lap_waveform_load_enriched_tickets.csv"
CURRENT_TICKETS_CSV = ROOT / "outputs/analysis/front_survival_context_v1/tickets_with_front_survival_context.csv"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def text_series(frame: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="string")
    return frame[col].astype("string").fillna(default)


def num_series(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").fillna(default)


def normalize_venue(value: Any) -> str:
    return str(value).strip().replace(" ", "").replace("　", "")


def race_date_from_id(value: Any) -> str:
    text = str(value).strip()
    return text[:8] if len(text) >= 8 else ""


def is_turf(surface: Any) -> bool:
    text = str(surface)
    return "芝" in text or "ѓ_" in text


def is_dirt(surface: Any) -> bool:
    text = str(surface)
    return "ダ" in text or "ЋЕ" in text


def qcut_bucket_by_group(values: pd.Series, groups: pd.Series, labels: tuple[str, str, str]) -> pd.Series:
    out = pd.Series("unknown", index=values.index, dtype="object")
    for _, idx in groups.groupby(groups).groups.items():
        s = pd.to_numeric(values.loc[idx], errors="coerce")
        valid = s.notna()
        if valid.sum() < 20 or s.nunique(dropna=True) < 3:
            out.loc[idx[valid]] = "mid"
            continue
        q1 = s.loc[valid].quantile(1 / 3)
        q2 = s.loc[valid].quantile(2 / 3)
        out.loc[idx[valid & s.le(q1)]] = labels[0]
        out.loc[idx[valid & s.gt(q1) & s.le(q2)]] = labels[1]
        out.loc[idx[valid & s.gt(q2)]] = labels[2]
    return out


def build_race_track_context() -> pd.DataFrame:
    races = read_csv(RACE_CONTEXT_CSV)
    races["race_id"] = races["race_id"].astype(str)
    races["date_yyyymmdd"] = races["race_id"].map(race_date_from_id)
    races["venue_norm"] = races["venue"].map(normalize_venue)

    track = read_csv(TRACK_CSV)
    track["date_yyyymmdd"] = track["date"].astype(str).str.replace(r"\D", "", regex=True).str.slice(0, 8)
    track["venue_norm"] = track["venue"].map(normalize_venue)
    for col in [
        "cushion_value",
        "moisture_turf_goal",
        "moisture_turf_back",
        "moisture_dirt_goal",
        "moisture_dirt_back",
    ]:
        track[col] = pd.to_numeric(track[col], errors="coerce")
    track["turf_moisture_avg"] = track[["moisture_turf_goal", "moisture_turf_back"]].mean(axis=1)
    track["dirt_moisture_avg"] = track[["moisture_dirt_goal", "moisture_dirt_back"]].mean(axis=1)
    track["course_setting"] = track.get("course", "").astype("string").fillna("").str.strip()

    tcols = [
        "date_yyyymmdd",
        "venue_norm",
        "course_setting",
        "cushion_value",
        "turf_moisture_avg",
        "dirt_moisture_avg",
        "moisture_turf_goal",
        "moisture_turf_back",
        "moisture_dirt_goal",
        "moisture_dirt_back",
    ]
    out = races.merge(track[tcols].drop_duplicates(["date_yyyymmdd", "venue_norm"]), on=["date_yyyymmdd", "venue_norm"], how="left")
    out["surface_norm"] = np.select(
        [out["surface"].map(is_turf), out["surface"].map(is_dirt)],
        ["turf", "dirt"],
        default="unknown",
    )
    out["moisture_surface_avg"] = np.where(
        out["surface_norm"].eq("turf"),
        out["turf_moisture_avg"],
        np.where(out["surface_norm"].eq("dirt"), out["dirt_moisture_avg"], np.nan),
    )
    out["venue_surface"] = out["venue_norm"] + "_" + out["surface_norm"].astype(str)
    out["cushion_bucket"] = qcut_bucket_by_group(out["cushion_value"], out["venue_norm"], ("low_cushion", "mid_cushion", "high_cushion"))
    out["moisture_bucket"] = qcut_bucket_by_group(
        out["moisture_surface_avg"], out["venue_surface"], ("dry_moisture", "mid_moisture", "wet_moisture")
    )
    out["track_lap_regime"] = np.select(
        [
            out["surface_norm"].eq("turf") & out["cushion_bucket"].eq("high_cushion") & out["moisture_bucket"].eq("dry_moisture"),
            out["surface_norm"].eq("turf") & out["cushion_bucket"].eq("high_cushion"),
            out["surface_norm"].eq("turf") & out["moisture_bucket"].eq("wet_moisture"),
            out["surface_norm"].eq("turf") & out["cushion_bucket"].eq("low_cushion"),
            out["surface_norm"].eq("dirt") & out["moisture_bucket"].eq("wet_moisture"),
            out["surface_norm"].eq("dirt") & out["moisture_bucket"].eq("dry_moisture"),
        ],
        [
            "turf_fast_high_cushion_dry",
            "turf_high_cushion",
            "turf_wet_moisture",
            "turf_low_cushion",
            "dirt_wet_moisture",
            "dirt_dry_moisture",
        ],
        default="track_mid_or_unknown",
    )
    return out


def standardize_ticket_frame(frame: pd.DataFrame, universe: str) -> pd.DataFrame:
    out = frame.copy()
    out["universe"] = universe
    out["race_id"] = out["race_id"].astype(str)
    if "stake_yen_eval" in out.columns:
        out["stake_eval"] = num_series(out, "stake_yen_eval", 100.0)
    else:
        out["stake_eval"] = num_series(out, "stake_yen", 100.0)
    if "return_yen_eval" in out.columns:
        out["return_eval"] = num_series(out, "return_yen_eval", 0.0)
    else:
        out["return_eval"] = num_series(out, "return_yen", 0.0)
    if "hit_eval" in out.columns:
        out["hit_eval_std"] = num_series(out, "hit_eval", 0.0).gt(0)
    elif "hit" in out.columns:
        out["hit_eval_std"] = num_series(out, "hit", 0.0).gt(0)
    else:
        out["hit_eval_std"] = out["return_eval"].gt(0)
    out["ticket_type"] = text_series(out, "ticket_type", "unknown").astype(str)
    return out


def add_lap_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    idx = out.index
    strict_gap = num_series(out, "strict_waveform_pair_gap_score", np.nan)
    strict_avg = num_series(out, "strict_waveform_pair_avg_score", np.nan)
    goodrun_min = num_series(out, "goodrun_lap_pair_min_score", np.nan)
    goodrun_avg = num_series(out, "goodrun_lap_pair_avg_score", np.nan)
    combo = num_series(out, "lap_advanced_combo_score", np.nan)
    role_proxy = num_series(out, "lap_role_pair_probability_proxy", np.nan)
    collision = num_series(out, "lap_role_front_front_collision_risk", np.nan)

    def q(s: pd.Series, value: float, default: float) -> float:
        valid = s.dropna()
        return float(valid.quantile(value)) if len(valid) else default

    thresholds = {
        "strict_gap_q20": q(strict_gap, 0.20, 0.02),
        "strict_avg_q70": q(strict_avg, 0.70, 0.70),
        "goodrun_min_q80": q(goodrun_min, 0.80, 0.80),
        "goodrun_avg_q70": q(goodrun_avg, 0.70, 0.80),
        "combo_q70": q(combo, 0.70, 0.60),
        "role_q70": q(role_proxy, 0.70, 0.60),
        "collision_q40": q(collision, 0.40, 0.30),
    }
    out["lap_signal_strict_gap_low"] = strict_gap.le(thresholds["strict_gap_q20"])
    out["lap_signal_goodrun_min_high"] = goodrun_min.ge(thresholds["goodrun_min_q80"])
    out["lap_signal_combo_high"] = combo.ge(thresholds["combo_q70"])
    out["lap_signal_role_low_collision"] = role_proxy.ge(thresholds["role_q70"]) & collision.le(thresholds["collision_q40"])
    out["lap_signal_any_strong"] = (
        out["lap_signal_strict_gap_low"]
        | out["lap_signal_goodrun_min_high"]
        | out["lap_signal_combo_high"]
        | out["lap_signal_role_low_collision"]
    )
    out.attrs["thresholds"] = thresholds
    return out


def add_lap_track_shadow_labels(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    regime = text_series(out, "track_lap_regime", "track_mid_or_unknown").astype(str)
    ticket = text_series(out, "ticket_type", "unknown").astype(str)
    strict_gap = out.get("lap_signal_strict_gap_low", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    goodrun_min = out.get("lap_signal_goodrun_min_high", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    role_low_collision = out.get("lap_signal_role_low_collision", pd.Series(False, index=out.index)).fillna(False).astype(bool)

    label = pd.Series("neutral", index=out.index, dtype="object")
    score = pd.Series(0.0, index=out.index, dtype=float)
    note = pd.Series("", index=out.index, dtype="object")

    positive_wet_strict = regime.eq("turf_wet_moisture") & strict_gap & ticket.eq("umaren")
    positive_fast_role = regime.eq("turf_fast_high_cushion_dry") & role_low_collision
    positive_fast_goodrun_umaren = regime.eq("turf_fast_high_cushion_dry") & goodrun_min & ticket.eq("umaren")
    positive_dirt_wet_role_wide = regime.eq("dirt_wet_moisture") & role_low_collision & ticket.eq("wide")

    caution_wet_role_umaren = regime.eq("turf_wet_moisture") & role_low_collision & ticket.eq("umaren")
    caution_low_cushion_strict_umaren = regime.eq("turf_low_cushion") & strict_gap & ticket.eq("umaren")
    caution_dirt_dry_strict_umaren = regime.eq("dirt_dry_moisture") & strict_gap & ticket.eq("umaren")

    label.loc[positive_wet_strict] = "positive"
    score.loc[positive_wet_strict] = 0.90
    note.loc[positive_wet_strict] = "turf wet/moisture + strict waveform gap + umaren was reproducible in 2024-2025"

    label.loc[positive_fast_role] = "positive"
    score.loc[positive_fast_role] = np.maximum(score.loc[positive_fast_role], 0.82)
    note.loc[positive_fast_role] = "fast/high-cushion/dry turf + low-collision role fit was positive across years"

    label.loc[positive_fast_goodrun_umaren] = "positive"
    score.loc[positive_fast_goodrun_umaren] = np.maximum(score.loc[positive_fast_goodrun_umaren], 0.78)
    note.loc[positive_fast_goodrun_umaren] = "fast/high-cushion/dry turf + goodrun min high + umaren stayed positive across years"

    label.loc[positive_dirt_wet_role_wide] = "positive_soft"
    score.loc[positive_dirt_wet_role_wide] = np.maximum(score.loc[positive_dirt_wet_role_wide], 0.74)
    note.loc[positive_dirt_wet_role_wide] = "wet dirt + low-collision role fit + wide was promising but sample is small"

    caution = caution_wet_role_umaren | caution_low_cushion_strict_umaren | caution_dirt_dry_strict_umaren
    downgrade = caution & label.eq("neutral")
    label.loc[downgrade] = "caution"
    score.loc[downgrade] = 0.35
    note.loc[caution_wet_role_umaren & downgrade] = "high overall ROI but year-skewed; keep shadow only"
    note.loc[caution_low_cushion_strict_umaren & downgrade] = "low-cushion strict gap was year-skewed; keep shadow only"
    note.loc[caution_dirt_dry_strict_umaren & downgrade] = "dry dirt strict gap was year-skewed; keep shadow only"

    out["lap_track_shadow_label"] = label
    out["lap_track_shadow_score"] = score
    out["lap_track_shadow_note"] = note
    return out


def metrics(frame: pd.DataFrame, segment: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "segment": segment,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "roi_pct": np.nan,
            "hit_rate_pct": np.nan,
            "top_return_share_pct": np.nan,
            "roi_ex_top1_pct": np.nan,
        }
    stake = pd.to_numeric(frame["stake_eval"], errors="coerce").fillna(100.0)
    ret = pd.to_numeric(frame["return_eval"], errors="coerce").fillna(0.0)
    top = float(ret.max()) if len(ret) else 0.0
    top_idx = ret.idxmax() if len(ret) else None
    total_stake = float(stake.sum())
    total_ret = float(ret.sum())
    ex_stake = total_stake - (float(stake.loc[top_idx]) if top_idx is not None else 0.0)
    return {
        "segment": segment,
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()),
        "stake_yen": total_stake,
        "return_yen": total_ret,
        "roi_pct": total_ret / total_stake * 100 if total_stake else np.nan,
        "hit_rate_pct": float(frame["hit_eval_std"].mean() * 100) if len(frame) else np.nan,
        "top_return_share_pct": top / total_ret * 100 if total_ret else np.nan,
        "roi_ex_top1_pct": (total_ret - top) / ex_stake * 100 if ex_stake > 0 else np.nan,
    }


def yearly_metrics(frame: pd.DataFrame, segment: str) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    out = frame.copy()
    out["year"] = out["race_id"].astype(str).str.slice(0, 4)
    rows: list[dict[str, Any]] = []
    for year, g in out.groupby("year", dropna=False):
        row = metrics(g, segment)
        row["year"] = year
        rows.append(row)
    return rows


def summarize_universe(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    policy_year_rows: list[dict[str, Any]] = []

    rows.append(metrics(frame, "all"))
    for col in ["track_lap_regime", "going_group", "cushion_bucket", "moisture_bucket", "course_setting"]:
        if col not in frame.columns:
            continue
        for value, g in frame.groupby(col, dropna=False):
            rows.append(metrics(g, f"{col}:{value}"))
            for ticket_type, gt in g.groupby("ticket_type", dropna=False):
                rows.append(metrics(gt, f"{col}:{value}:{ticket_type}"))

    signal_cols = [
        "lap_signal_strict_gap_low",
        "lap_signal_goodrun_min_high",
        "lap_signal_combo_high",
        "lap_signal_role_low_collision",
        "lap_signal_any_strong",
    ]
    for regime, gr in frame.groupby("track_lap_regime", dropna=False):
        for sig in signal_cols:
            if sig not in gr.columns:
                continue
            sg = gr.loc[gr[sig].fillna(False)]
            if len(sg):
                segment_name = f"{regime}:{sig}"
                policy_rows.append(metrics(sg, segment_name))
                policy_year_rows.extend(yearly_metrics(sg, segment_name))
                for ticket_type, gt in sg.groupby("ticket_type", dropna=False):
                    segment_name = f"{regime}:{sig}:{ticket_type}"
                    policy_rows.append(metrics(gt, segment_name))
                    policy_year_rows.extend(yearly_metrics(gt, segment_name))

    for sig in signal_cols:
        if sig in frame.columns:
            sg = frame.loc[frame[sig].fillna(False)]
            segment_name = f"all:{sig}"
            policy_rows.append(metrics(sg, segment_name))
            policy_year_rows.extend(yearly_metrics(sg, segment_name))
            for ticket_type, gt in sg.groupby("ticket_type", dropna=False):
                segment_name = f"all:{sig}:{ticket_type}"
                policy_rows.append(metrics(gt, segment_name))
                policy_year_rows.extend(yearly_metrics(gt, segment_name))

    return pd.DataFrame(rows), pd.DataFrame(policy_rows), pd.DataFrame(policy_year_rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    race_context = build_race_track_context()
    context_cols = [
        "race_id",
        "date_yyyymmdd",
        "venue",
        "surface",
        "surface_norm",
        "distance_bin",
        "class_group",
        "going_group",
        "course_setting",
        "cushion_value",
        "turf_moisture_avg",
        "dirt_moisture_avg",
        "moisture_surface_avg",
        "cushion_bucket",
        "moisture_bucket",
        "track_lap_regime",
        "pre_high_pressure_signal",
        "actual_front_survival",
        "actual_front_collapse",
    ]
    race_context[context_cols].to_csv(OUT_DIR / "race_lap_track_context.csv", index=False, encoding="utf-8-sig")

    universes = {
        "lap_role_goodrun": ROLE_TICKETS_CSV,
        "lap_load_tolerance": LOAD_TICKETS_CSV,
        "current_front_context": CURRENT_TICKETS_CSV,
    }
    all_summary: list[pd.DataFrame] = []
    all_policy: list[pd.DataFrame] = []
    all_policy_yearly: list[pd.DataFrame] = []
    summary_json: dict[str, Any] = {
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
        "race_context_rows": int(len(race_context)),
    }
    for name, path in universes.items():
        raw = read_csv(path)
        tickets = standardize_ticket_frame(raw, name)
        tickets = tickets.merge(race_context[context_cols].drop_duplicates("race_id"), on="race_id", how="left")
        tickets = add_lap_labels(tickets)
        tickets = add_lap_track_shadow_labels(tickets)
        seg, pol, pol_year = summarize_universe(tickets)
        seg.insert(0, "universe", name)
        pol.insert(0, "universe", name)
        pol_year.insert(0, "universe", name)
        all_summary.append(seg)
        all_policy.append(pol)
        all_policy_yearly.append(pol_year)
        tickets.to_csv(OUT_DIR / f"{name}_tickets_with_lap_track.csv", index=False, encoding="utf-8-sig")
        summary_json[name] = {
            "tickets": int(len(tickets)),
            "races": int(tickets["race_id"].nunique()),
            "track_context_coverage": float(tickets["track_lap_regime"].notna().mean()),
            "lap_thresholds": tickets.attrs.get("thresholds", {}),
        }

    summary = pd.concat(all_summary, ignore_index=True)
    policy = pd.concat(all_policy, ignore_index=True)
    policy_yearly = pd.concat(all_policy_yearly, ignore_index=True)
    summary.to_csv(OUT_DIR / "lap_track_segment_summary.csv", index=False, encoding="utf-8-sig")
    policy.to_csv(OUT_DIR / "lap_track_policy_summary.csv", index=False, encoding="utf-8-sig")
    policy_yearly.to_csv(OUT_DIR / "lap_track_policy_yearly_summary.csv", index=False, encoding="utf-8-sig")

    practical = policy[
        policy["tickets"].ge(30)
        & policy["top_return_share_pct"].fillna(100).le(60)
        & policy["roi_ex_top1_pct"].fillna(0).ge(100)
    ].sort_values(["roi_pct", "tickets"], ascending=[False, False])
    practical.to_csv(OUT_DIR / "lap_track_practical_candidates.csv", index=False, encoding="utf-8-sig")
    practical_keys = practical[["universe", "segment"]].drop_duplicates()
    practical_yearly = policy_yearly.merge(practical_keys, on=["universe", "segment"], how="inner")
    practical_yearly.to_csv(OUT_DIR / "lap_track_practical_candidates_yearly.csv", index=False, encoding="utf-8-sig")

    summary_json["top_segments"] = summary.sort_values(["roi_pct", "tickets"], ascending=[False, False]).head(20).replace({np.nan: None}).to_dict(orient="records")
    summary_json["top_practical_policies"] = practical.head(20).replace({np.nan: None}).to_dict(orient="records")
    summary_json["note"] = (
        "Shadow validation only. Uses JRA track-condition cushion/moisture values and historical lap-ticket universes. "
        "Do not promote formal BUY without T-5/T-3 OOS confirmation."
    )
    (OUT_DIR / "summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary_json, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
