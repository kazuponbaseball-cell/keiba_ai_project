from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACK = ROOT / "data/raw/track_condition_metrics.csv"
DEFAULT_RACES = ROOT / "outputs/analysis/front_survival_context_v1/race_front_survival_context.csv"
DEFAULT_TICKETS = ROOT / "outputs/analysis/front_survival_context_v1/tickets_with_front_survival_context.csv"
DEFAULT_OUT = ROOT / "outputs/analysis/opening_week_course_change_context_v1"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def yymmdd_to_yyyymmdd(value: Any) -> int | float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    text = text.zfill(6)
    year = int(text[:2])
    return int(f"{2000 + year:04d}{text[2:]}")


def normalize_venue(value: Any) -> str:
    return str(value).strip().replace(" ", "").replace("　", "")


def pdf_block(value: Any) -> str:
    text = str(value)
    name = Path(text.replace("\\", "/")).stem
    m = re.search(r"([a-z]+)(\d+)$", name, flags=re.I)
    if not m:
        return ""
    return f"{m.group(1).lower()}{m.group(2)}"


def pdf_block_no(value: Any) -> float:
    block = pdf_block(value)
    m = re.search(r"(\d+)$", block)
    return float(m.group(1)) if m else np.nan


def add_meeting_context(races: pd.DataFrame, track: pd.DataFrame) -> pd.DataFrame:
    out = races.copy()
    out["date_yyyymmdd"] = out["date_key"].map(yymmdd_to_yyyymmdd)
    out["venue_norm"] = out["venue"].map(normalize_venue)

    t = track.copy()
    t["date_yyyymmdd"] = pd.to_numeric(t["date"], errors="coerce")
    t["venue_norm"] = t["venue"].map(normalize_venue)
    t["track_pdf_block"] = t["source_pdf"].map(pdf_block)
    t["track_pdf_block_no"] = t["source_pdf"].map(pdf_block_no)
    t["track_source_url_block"] = t["source_url"].map(pdf_block)
    t["course_setting"] = t["course"].astype(str).str.strip().replace({"nan": "", "None": ""})
    course_days = (
        t.loc[t["course_setting"].ne(""), ["date_yyyymmdd", "venue_norm", "course_setting"]]
        .drop_duplicates()
        .sort_values(["venue_norm", "date_yyyymmdd"])
        .copy()
    )
    course_days["previous_course_setting"] = course_days.groupby("venue_norm")["course_setting"].shift(1)
    course_days["course_setting_changed"] = (
        course_days["previous_course_setting"].notna()
        & course_days["course_setting"].ne(course_days["previous_course_setting"])
    ).astype(int)
    course_days["course_change_group"] = course_days.groupby("venue_norm")["course_setting_changed"].cumsum()
    course_days["course_setting_day_index"] = course_days.groupby(["venue_norm", "course_change_group"]).cumcount() + 1
    course_days = course_days.drop(columns=["course_change_group"])
    t = t[
        [
            "date_yyyymmdd",
            "venue_norm",
            "course",
            "course_setting",
            "cushion_value",
            "moisture_turf_goal",
            "moisture_turf_back",
            "moisture_dirt_goal",
            "moisture_dirt_back",
            "source_url",
            "source_pdf",
            "track_pdf_block",
            "track_pdf_block_no",
            "track_source_url_block",
        ]
    ].drop_duplicates(["date_yyyymmdd", "venue_norm"])

    out = out.merge(t, on=["date_yyyymmdd", "venue_norm"], how="left")
    out = out.merge(course_days, on=["date_yyyymmdd", "venue_norm", "course_setting"], how="left")

    # When the official PDF parser cannot read A/B/C course directly,
    # the archive PDF block (tokyo01, tokyo02, ...) is still a useful proxy
    # for a new meeting block / possible course-setting change.
    out["meeting_block"] = out["track_pdf_block"].fillna("")
    missing_block = out["meeting_block"].eq("")
    fallback = out["venue_norm"] + "_" + out["date_yyyymmdd"].astype("Int64").astype(str).str.slice(0, 6)
    out.loc[missing_block, "meeting_block"] = fallback[missing_block]

    day_map = (
        out[["venue_norm", "meeting_block", "date_yyyymmdd"]]
        .drop_duplicates()
        .sort_values(["venue_norm", "meeting_block", "date_yyyymmdd"])
    )
    day_map["meeting_day_index"] = day_map.groupby(["venue_norm", "meeting_block"]).cumcount() + 1
    day_map["meeting_days_total"] = day_map.groupby(["venue_norm", "meeting_block"])["date_yyyymmdd"].transform("count")
    first_date = day_map.groupby(["venue_norm", "meeting_block"])["date_yyyymmdd"].transform("min")
    day_map["days_since_meeting_open"] = pd.to_datetime(day_map["date_yyyymmdd"].astype(str)) - pd.to_datetime(first_date.astype(str))
    day_map["days_since_meeting_open"] = day_map["days_since_meeting_open"].dt.days
    out = out.merge(day_map, on=["venue_norm", "meeting_block", "date_yyyymmdd"], how="left")

    idx = pd.to_numeric(out["meeting_day_index"], errors="coerce")
    total = pd.to_numeric(out["meeting_days_total"], errors="coerce")
    out["meeting_stage"] = np.select(
        [
            idx.le(1),
            idx.le(2),
            idx.le(4),
            idx.ge(total.sub(1)),
        ],
        ["opening_day", "opening_2days", "early_3_4days", "final_2days"],
        default="middle",
    )
    out["is_opening_2days"] = idx.le(2).astype(int)
    out["is_early_4days"] = idx.le(4).astype(int)
    out["is_final_2days"] = idx.ge(total.sub(1)).astype(int)
    out["course_setting_available"] = out["course"].notna() & out["course"].astype(str).str.strip().ne("")
    out["course_setting"] = out["course_setting"].fillna("")
    out["course_setting_changed"] = pd.to_numeric(out["course_setting_changed"], errors="coerce").fillna(0).astype(int)
    out["course_setting_day_index"] = pd.to_numeric(out["course_setting_day_index"], errors="coerce")
    out["course_change_stage"] = np.select(
        [
            out["course_setting"].eq(""),
            out["course_setting_changed"].eq(1),
            out["course_setting_day_index"].le(2),
            out["course_setting_day_index"].le(4),
        ],
        ["unknown", "change_day", "course_opening_2days", "course_early_4days"],
        default="course_middle_late",
    )
    return out


def race_stage_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, g in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: key[i] for i, col in enumerate(group_cols)}
        high = pd.to_numeric(g["pre_high_pressure_signal"], errors="coerce").fillna(0).ge(
            pd.to_numeric(df["pre_high_pressure_signal"], errors="coerce").fillna(0).quantile(0.60)
        )
        row.update(
            {
                "races": int(len(g)),
                "front_survival_rate": float(pd.to_numeric(g["actual_front_survival"], errors="coerce").mean()),
                "front_collapse_rate": float(pd.to_numeric(g["actual_front_collapse"], errors="coerce").mean()),
                "winner_front5_rate": float(pd.to_numeric(g["winner_front5"], errors="coerce").mean()),
                "top3_front5_share": float(pd.to_numeric(g["actual_top3_front5_share"], errors="coerce").mean()),
                "avg_high_pressure_signal": float(pd.to_numeric(g["pre_high_pressure_signal"], errors="coerce").mean()),
                "high_pressure_races": int(high.sum()),
                "high_pressure_survival_rate": float(pd.to_numeric(g.loc[high, "actual_front_survival"], errors="coerce").mean())
                if high.any()
                else np.nan,
                "high_pressure_collapse_rate": float(pd.to_numeric(g.loc[high, "actual_front_collapse"], errors="coerce").mean())
                if high.any()
                else np.nan,
                "readability_score_avg": float(pd.to_numeric(g["front_context_readability_score"], errors="coerce").mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols)


def ticket_return(frame: pd.DataFrame) -> pd.Series:
    if "return_yen" in frame.columns:
        existing = pd.to_numeric(frame["return_yen"], errors="coerce")
        if existing.notna().any():
            return existing.fillna(0.0)
    typ = frame["ticket_type"].astype(str)
    stake = pd.to_numeric(frame.get("stake_yen"), errors="coerce").fillna(100.0)
    values = np.select(
        [typ.eq("wide"), typ.eq("umaren"), typ.eq("umatan") & frame.get("umatan_anchor_hit", 0).astype(bool)],
        [
            pd.to_numeric(frame.get("wide_pay"), errors="coerce").fillna(0.0) * stake / 100.0,
            pd.to_numeric(frame.get("umaren_pay"), errors="coerce").fillna(0.0) * stake / 100.0,
            pd.to_numeric(frame.get("umatan_pay"), errors="coerce").fillna(0.0) * stake / 100.0,
        ],
        default=0.0,
    )
    return pd.Series(values, index=frame.index, dtype=float)


def ticket_hit(frame: pd.DataFrame) -> pd.Series:
    typ = frame["ticket_type"].astype(str)
    values = np.select(
        [typ.eq("wide"), typ.eq("umaren"), typ.eq("umatan")],
        [
            pd.to_numeric(frame.get("wide_hit"), errors="coerce").fillna(0).gt(0),
            pd.to_numeric(frame.get("umaren_hit"), errors="coerce").fillna(0).gt(0),
            pd.to_numeric(frame.get("umatan_anchor_hit"), errors="coerce").fillna(0).gt(0),
        ],
        default=pd.to_numeric(frame.get("hit"), errors="coerce").fillna(0).gt(0),
    )
    return pd.Series(values, index=frame.index, dtype=bool)


def num_series(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in frame.columns:
        return pd.to_numeric(frame[col], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def metrics(frame: pd.DataFrame, segment: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "segment": segment,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi": np.nan,
            "hit_rate": np.nan,
            "top_return_share": np.nan,
            "roi_ex_top1": np.nan,
        }
    stake = pd.to_numeric(frame.get("stake_yen"), errors="coerce").fillna(100.0)
    ret = ticket_return(frame)
    hit = ticket_hit(frame)
    top = float(ret.max()) if len(ret) else 0.0
    total_ret = float(ret.sum())
    total_stake = float(stake.sum())
    return {
        "segment": segment,
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()),
        "stake_yen": total_stake,
        "return_yen": total_ret,
        "profit_yen": total_ret - total_stake,
        "roi": total_ret / total_stake if total_stake else np.nan,
        "hit_rate": float(hit.mean()) if len(hit) else np.nan,
        "top_return_share": top / total_ret if total_ret else np.nan,
        "roi_ex_top1": (total_ret - top) / (total_stake - float(stake.loc[ret.idxmax()])) if total_stake > float(stake.loc[ret.idxmax()]) else np.nan,
    }


def ticket_stage_metrics(tickets: pd.DataFrame, race_context: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ctx_cols = [
        "race_id",
        "meeting_block",
        "meeting_day_index",
        "meeting_days_total",
        "meeting_stage",
        "is_opening_2days",
        "is_early_4days",
        "is_final_2days",
        "course_setting_available",
        "course_setting",
        "course_setting_changed",
        "course_setting_day_index",
        "course_change_stage",
        "track_pdf_block_no",
    ]
    t = tickets.merge(race_context[ctx_cols].drop_duplicates("race_id"), on="race_id", how="left")
    t["front_context_collapse_risk_score"] = num_series(
        t, "front_collapse_reinforced_score", np.nan
    ).fillna(num_series(t, "front_context_collapse_risk_score", 0.0))
    rows = [metrics(t, "all")]
    for stage, g in t.groupby("meeting_stage", dropna=False):
        rows.append(metrics(g, f"stage:{stage}"))
    for typ, g in t.groupby("ticket_type", dropna=False):
        rows.append(metrics(g, f"ticket:{typ}"))
    for stage, gs in t.groupby("meeting_stage", dropna=False):
        for typ, g in gs.groupby("ticket_type", dropna=False):
            rows.append(metrics(g, f"stage_ticket:{stage}:{typ}"))

    # Existing high-performing context-style filters, stage-aware.
    survival = num_series(t, "front_survival_despite_pressure_score", 0.0)
    collapse = num_series(t, "front_collapse_reinforced_score", 0.0)
    readability = num_series(t, "front_context_readability_score", 0.0)
    masks = {
        "opening_2days_only": t["is_opening_2days"].eq(1),
        "non_opening_2days": t["is_opening_2days"].ne(1),
        "early_4days_only": t["is_early_4days"].eq(1),
        "final_2days_only": t["is_final_2days"].eq(1),
        "avoid_collapse_context_q80": collapse.le(collapse.quantile(0.80)),
        "readable_survival_support": readability.ge(readability.quantile(0.50)) & survival.ge(survival.quantile(0.60)),
        "opening_2days_and_avoid_collapse_q80": t["is_opening_2days"].eq(1) & collapse.le(collapse.quantile(0.80)),
        "non_opening_and_avoid_collapse_q80": t["is_opening_2days"].ne(1) & collapse.le(collapse.quantile(0.80)),
        "opening_2days_readable_survival": t["is_opening_2days"].eq(1)
        & readability.ge(readability.quantile(0.50))
        & survival.ge(survival.quantile(0.60)),
        "non_opening_readable_survival": t["is_opening_2days"].ne(1)
        & readability.ge(readability.quantile(0.50))
        & survival.ge(survival.quantile(0.60)),
    }
    policy_rows = []
    for name, mask in masks.items():
        policy_rows.append(metrics(t.loc[mask], name))
        for typ, g in t.loc[mask].groupby("ticket_type", dropna=False):
            policy_rows.append(metrics(g, f"{name}:{typ}"))
    return pd.DataFrame(rows), pd.DataFrame(policy_rows), t


def main() -> None:
    DEFAULT_OUT.mkdir(parents=True, exist_ok=True)
    track = read_csv(DEFAULT_TRACK)
    races = read_csv(DEFAULT_RACES)
    tickets = read_csv(DEFAULT_TICKETS)

    race_context = add_meeting_context(races, track)
    race_stage = race_stage_metrics(race_context, ["meeting_stage"])
    race_stage_surface = race_stage_metrics(race_context, ["surface", "meeting_stage"])
    race_stage_venue = race_stage_metrics(race_context, ["venue", "meeting_stage"])
    race_course = race_stage_metrics(race_context.loc[race_context["course_setting"].ne("")], ["course_setting"])
    race_course_change = race_stage_metrics(race_context, ["course_change_stage"])
    race_course_surface = race_stage_metrics(
        race_context.loc[race_context["course_setting"].ne("")],
        ["course_setting", "surface", "meeting_stage"],
    )
    ticket_stage, ticket_policy, tickets_enriched = ticket_stage_metrics(tickets, race_context)

    race_context.to_csv(DEFAULT_OUT / "race_opening_week_context.csv", index=False, encoding="utf-8-sig")
    race_stage.to_csv(DEFAULT_OUT / "race_stage_summary.csv", index=False, encoding="utf-8-sig")
    race_stage_surface.to_csv(DEFAULT_OUT / "race_stage_surface_summary.csv", index=False, encoding="utf-8-sig")
    race_stage_venue.to_csv(DEFAULT_OUT / "race_stage_venue_summary.csv", index=False, encoding="utf-8-sig")
    race_course.to_csv(DEFAULT_OUT / "race_course_setting_summary.csv", index=False, encoding="utf-8-sig")
    race_course_change.to_csv(DEFAULT_OUT / "race_course_change_stage_summary.csv", index=False, encoding="utf-8-sig")
    race_course_surface.to_csv(DEFAULT_OUT / "race_course_surface_stage_summary.csv", index=False, encoding="utf-8-sig")
    ticket_stage.to_csv(DEFAULT_OUT / "ticket_stage_summary.csv", index=False, encoding="utf-8-sig")
    ticket_policy.to_csv(DEFAULT_OUT / "ticket_stage_policy_summary.csv", index=False, encoding="utf-8-sig")
    tickets_enriched.to_csv(DEFAULT_OUT / "tickets_with_opening_week_context.csv", index=False, encoding="utf-8-sig")

    summary = {
        "output_dir": str(DEFAULT_OUT.relative_to(ROOT)),
        "races": int(len(race_context)),
        "tickets": int(len(tickets_enriched)),
        "course_setting_available_rate": float(race_context["course_setting_available"].mean()),
        "course_setting_counts": race_context["course_setting"].replace("", "unknown").value_counts().to_dict(),
        "race_stage_top": race_stage.sort_values("front_survival_rate", ascending=False).head(5).replace({np.nan: None}).to_dict(orient="records"),
        "race_course_top": race_course.sort_values("front_survival_rate", ascending=False).head(5).replace({np.nan: None}).to_dict(orient="records"),
        "race_course_change_top": race_course_change.sort_values("front_survival_rate", ascending=False).head(5).replace({np.nan: None}).to_dict(orient="records"),
        "ticket_policy_top": ticket_policy.sort_values("roi", ascending=False).head(10).replace({np.nan: None}).to_dict(orient="records"),
        "note": "Opening week is derived from JRA track-condition PDF blocks and race dates. A/B/C course setting is not available in the current CSV parser, so meeting block is a proxy for course-setting changes.",
    }
    (DEFAULT_OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
