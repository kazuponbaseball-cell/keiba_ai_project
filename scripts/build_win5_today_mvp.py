from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_win5_candidates as win5  # noqa: E402

DEFAULT_LEGS_BY_DATE = {
    "20260823": ["中京6R", "札幌10R", "新潟7R", "中京7R", "札幌11R"],
}

RACE_ID_COL = "レースID(新/馬番無)"
HORSE_ID_COL = "血統登録番号"
HORSE_NAME_COL = "馬名"
RANK_COL = "確定着順"
DATE_COL = "日付"
ABNORMAL_COL = "異常コード"


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_leg(label: str) -> tuple[str, int]:
    match = re.fullmatch(r"\s*(.+?)\s*(\d{1,2})\s*[RＲ]\s*", label)
    if not match:
        raise ValueError(f"Invalid leg label: {label!r}; expected e.g. 中京6R")
    return match.group(1).strip(), int(match.group(2))


def latest_prediction() -> Path:
    candidates = sorted(
        project_path("outputs/predictions").glob("baseline_predictions_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No baseline_predictions_*.csv found under outputs/predictions")
    return candidates[0]


def parse_date_num(value: object) -> int | None:
    digits = re.sub(r"\D", "", "" if value is None or pd.isna(value) else str(value))
    if len(digits) == 8 and digits.startswith("20"):
        return int(digits[2:])
    if len(digits) == 6:
        return int(digits)
    return None


def find_col(columns: list[str], names: list[str]) -> str | None:
    for name in names:
        if name in columns:
            return name
    return None


def safe_num(value: object, default: float = np.nan) -> float:
    try:
        if value is None or pd.isna(value) or str(value).strip() == "":
            return default
        return float(str(value).replace(",", ""))
    except Exception:
        return default


def pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{100.0 * value:.1f}%"


def resolve_target_races(pred: pd.DataFrame, date: str, leg_labels: list[str]) -> tuple[pd.DataFrame, list[str]]:
    frame = pred.copy()
    frame["race_id"] = frame.apply(win5.official_race_id, axis=1)
    frame = frame[frame["race_id"].str.startswith(date)].copy()
    frame["venue"] = win5.col(frame, ["場所", "venue"], "")
    frame["race_no"] = pd.to_numeric(win5.col(frame, ["Ｒ", "race_no"], 0), errors="coerce").fillna(0).astype(int)

    rows = []
    race_ids = []
    for leg_no, label in enumerate(leg_labels, start=1):
        venue, race_no = parse_leg(label)
        matched = frame[(frame["venue"].astype(str).str.strip() == venue) & (frame["race_no"] == race_no)]
        ids = [x for x in matched["race_id"].dropna().astype(str).unique().tolist() if x]
        if len(ids) != 1:
            raise ValueError(f"Could not uniquely resolve {label}: ids={ids}")
        race_id = ids[0]
        race_ids.append(race_id)
        rows.append({"leg_no": leg_no, "label": label, "venue": venue, "race_no": race_no, "race_id": race_id})
    return pd.DataFrame(rows), race_ids


def prepare_entry(entry: pd.DataFrame) -> pd.DataFrame:
    out = entry.copy()
    out["race_id"] = out.apply(win5.official_race_id, axis=1)
    out["horse_no"] = pd.to_numeric(win5.col(out, ["馬番", "horse_no"], 0), errors="coerce").fillna(0).astype(int)
    out["horse_name"] = win5.col(out, ["馬名", "horse_name"], "").astype(str).str.strip()
    if HORSE_ID_COL in out.columns:
        out["horse_id"] = out[HORSE_ID_COL].astype("string").fillna("").str.strip()
    else:
        out["horse_id"] = ""
    return out


def history_config() -> tuple[Path, str]:
    cfg = json.loads(project_path("config/baseline_features.json").read_text(encoding="utf-8"))
    data = cfg["data"]
    return project_path(data["historical_csv"]), data.get("encoding", "cp932")


def wanted_course_keys(target_entry: pd.DataFrame) -> set[tuple[str, str, int]]:
    keys: set[tuple[str, str, int]] = set()
    for _, row in target_entry.iterrows():
        venue = str(row.get("場所", "")).strip()
        surface = str(row.get("芝・ダ", "")).strip()
        dist = int(safe_num(row.get("距離"), 0) or 0)
        if venue and surface and dist > 0:
            keys.add((venue, surface, dist))
    return keys


def load_relevant_history(history_path: Path, encoding: str, target_entry: pd.DataFrame, date: str) -> pd.DataFrame:
    header = pd.read_csv(history_path, encoding=encoding, nrows=0).columns.tolist()
    horse_ids = set(target_entry["horse_id"].astype(str).str.strip()) - {""}
    horse_names = set(target_entry["horse_name"].astype(str).str.strip()) - {""}
    course_keys = wanted_course_keys(target_entry)
    current_num = int(date[2:])

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(history_path, encoding=encoding, chunksize=200_000, low_memory=False):
        if DATE_COL in chunk.columns:
            dates = chunk[DATE_COL].map(parse_date_num)
            chunk = chunk[pd.Series(dates, index=chunk.index).fillna(99999999).astype(int) < current_num]
        if chunk.empty:
            continue

        horse_mask = pd.Series(False, index=chunk.index)
        if horse_ids and HORSE_ID_COL in chunk.columns:
            horse_mask |= chunk[HORSE_ID_COL].astype("string").fillna("").str.strip().isin(horse_ids)
        if horse_names and HORSE_NAME_COL in chunk.columns:
            horse_mask |= chunk[HORSE_NAME_COL].astype("string").fillna("").str.strip().isin(horse_names)

        course_mask = pd.Series(False, index=chunk.index)
        if {"場所", "芝・ダ", "距離"}.issubset(chunk.columns) and course_keys:
            venue = chunk["場所"].astype("string").fillna("").str.strip()
            surface = chunk["芝・ダ"].astype("string").fillna("").str.strip()
            dist = pd.to_numeric(chunk["距離"], errors="coerce").fillna(0).astype(int)
            for key in course_keys:
                course_mask |= venue.eq(key[0]) & surface.eq(key[1]) & dist.eq(key[2])

        matched = chunk[horse_mask | course_mask]
        if not matched.empty:
            chunks.append(matched.copy())

    if not chunks:
        return pd.DataFrame(columns=header)
    history = pd.concat(chunks, ignore_index=True, sort=False)
    if ABNORMAL_COL in history.columns:
        abnormal = pd.to_numeric(history[ABNORMAL_COL], errors="coerce").fillna(0)
        history = history[abnormal.eq(0)].copy()
    if RANK_COL in history.columns:
        history[RANK_COL] = pd.to_numeric(history[RANK_COL], errors="coerce")
        history = history[history[RANK_COL].notna()].copy()
    return history


def perf_series(frame: pd.DataFrame) -> pd.Series:
    rank = pd.to_numeric(frame.get(RANK_COL), errors="coerce")
    field_col = find_col(frame.columns.tolist(), ["出走頭数", "頭数"])
    if field_col:
        field = pd.to_numeric(frame[field_col], errors="coerce")
    elif RACE_ID_COL in frame.columns:
        field = frame.groupby(RACE_ID_COL)[RANK_COL].transform("count")
    else:
        field = pd.Series(np.nan, index=frame.index)
    field = field.where(field >= 2)
    return ((field + 1 - rank) / field).clip(lower=0, upper=1)


def stats(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty or RANK_COL not in frame.columns:
        return {"starts": 0, "win": np.nan, "top3": np.nan, "perf": np.nan}
    rank = pd.to_numeric(frame[RANK_COL], errors="coerce")
    valid = frame[rank.notna()].copy()
    rank = pd.to_numeric(valid[RANK_COL], errors="coerce")
    if valid.empty:
        return {"starts": 0, "win": np.nan, "top3": np.nan, "perf": np.nan}
    perf = perf_series(valid)
    return {
        "starts": int(len(valid)),
        "win": float((rank == 1).mean()),
        "top3": float((rank <= 3).mean()),
        "perf": float(perf.mean()) if perf.notna().any() else np.nan,
    }


def slice_row(label: str, frame: pd.DataFrame, career_perf: float) -> dict[str, Any]:
    summary = stats(frame)
    delta = summary["perf"] - career_perf if np.isfinite(summary["perf"]) and np.isfinite(career_perf) else np.nan
    shrunk = delta * (summary["starts"] / (summary["starts"] + 3.0)) if np.isfinite(delta) else np.nan
    return {"label": label, **summary, "delta": delta, "shrunk": shrunk}


def horse_history(history: pd.DataFrame, entry_row: pd.Series) -> pd.DataFrame:
    horse_id = str(entry_row.get("horse_id", "")).strip()
    horse_name = str(entry_row.get("horse_name", "")).strip()
    if horse_id and HORSE_ID_COL in history.columns:
        matched = history[history[HORSE_ID_COL].astype("string").fillna("").str.strip().eq(horse_id)]
        if not matched.empty:
            return matched.copy()
    if horse_name and HORSE_NAME_COL in history.columns:
        return history[history[HORSE_NAME_COL].astype("string").fillna("").str.strip().eq(horse_name)].copy()
    return history.iloc[0:0].copy()


def build_horse_profile(history: pd.DataFrame, entry_row: pd.Series) -> dict[str, Any]:
    horse = horse_history(history, entry_row)
    career = stats(horse)
    career_perf = career["perf"]
    venue = str(entry_row.get("場所", "")).strip()
    surface = str(entry_row.get("芝・ダ", "")).strip()
    distance = int(safe_num(entry_row.get("距離"), 0) or 0)
    going = str(entry_row.get("馬場状態", "")).strip()

    slices = []
    if not horse.empty:
        if "芝・ダ" in horse.columns and surface:
            slices.append(slice_row(surface, horse[horse["芝・ダ"].astype(str).str.strip().eq(surface)], career_perf))
        if "場所" in horse.columns and venue:
            slices.append(slice_row(venue, horse[horse["場所"].astype(str).str.strip().eq(venue)], career_perf))
        if "距離" in horse.columns and distance:
            hd = pd.to_numeric(horse["距離"], errors="coerce")
            slices.append(slice_row(f"{distance}m±200", horse[(hd - distance).abs() <= 200], career_perf))
        if {"場所", "芝・ダ", "距離"}.issubset(horse.columns) and venue and surface and distance:
            hd = pd.to_numeric(horse["距離"], errors="coerce")
            mask = (
                horse["場所"].astype(str).str.strip().eq(venue)
                & horse["芝・ダ"].astype(str).str.strip().eq(surface)
                & ((hd - distance).abs() <= 200)
            )
            slices.append(slice_row("今回類似条件", horse[mask], career_perf))
        if "馬場状態" in horse.columns and going:
            slices.append(slice_row(f"馬場:{going}", horse[horse["馬場状態"].astype(str).str.strip().eq(going)], career_perf))

    strengths = [x for x in slices if x["starts"] >= 2 and np.isfinite(x["shrunk"]) and x["shrunk"] >= 0.035]
    weaknesses = [x for x in slices if x["starts"] >= 2 and np.isfinite(x["shrunk"]) and x["shrunk"] <= -0.035]
    strengths.sort(key=lambda x: x["shrunk"], reverse=True)
    weaknesses.sort(key=lambda x: x["shrunk"])

    flags = []
    if career["starts"] < 4:
        flags.append("経験不足")
    similar = next((x for x in slices if x["label"] == "今回類似条件"), None)
    if similar and similar["starts"] == 0:
        flags.append("今回類似条件未経験")
    if career["starts"] >= 5 and np.isfinite(career["win"]) and career["win"] == 0 and career["top3"] >= 0.30:
        flags.append("好走するが勝ち切り実績薄い")

    return {
        "career": career,
        "slices": slices,
        "strengths": strengths[:3],
        "weaknesses": weaknesses[:3],
        "flags": flags,
    }


def course_profile(history: pd.DataFrame, entry_row: pd.Series) -> dict[str, Any]:
    if history.empty or RANK_COL not in history.columns:
        return {"races": 0, "runners": 0}
    venue = str(entry_row.get("場所", "")).strip()
    surface = str(entry_row.get("芝・ダ", "")).strip()
    distance = int(safe_num(entry_row.get("距離"), 0) or 0)
    if not {"場所", "芝・ダ", "距離"}.issubset(history.columns):
        return {"races": 0, "runners": 0}
    hd = pd.to_numeric(history["距離"], errors="coerce")
    course = history[
        history["場所"].astype(str).str.strip().eq(venue)
        & history["芝・ダ"].astype(str).str.strip().eq(surface)
        & hd.eq(distance)
    ].copy()
    winners = course[pd.to_numeric(course[RANK_COL], errors="coerce").eq(1)].copy()
    race_col = RACE_ID_COL if RACE_ID_COL in course.columns else None
    out: dict[str, Any] = {
        "races": int(course[race_col].nunique()) if race_col else int(len(winners)),
        "runners": int(len(course)),
        "winner_count": int(len(winners)),
    }

    corner4_col = find_col(course.columns.tolist(), ["４角位置", "4角位置", "４角", "4角", "コーナー順位4"])
    if corner4_col and not winners.empty:
        pos = pd.to_numeric(winners[corner4_col], errors="coerce").dropna()
        if not pos.empty:
            out["winner_4c_median"] = float(pos.median())
            out["winner_4c_front5_rate"] = float((pos <= 5).mean())

    popularity_col = find_col(course.columns.tolist(), ["人気", "単勝人気"])
    if popularity_col and not winners.empty:
        pop = pd.to_numeric(winners[popularity_col], errors="coerce").dropna()
        if not pop.empty:
            out["winner_pop_median"] = float(pop.median())
            out["winner_top3pop_rate"] = float((pop <= 3).mean())
    return out


def scenario_risk(pred_row: pd.Series) -> tuple[str, float]:
    values = []
    for column in ("slow_ai_score", "middle_ai_score", "fast_ai_score"):
        if column in pred_row.index:
            value = safe_num(pred_row.get(column))
            if np.isfinite(value):
                values.append(value)
    if len(values) < 2:
        return "不明", np.nan
    spread = max(values) - min(values)
    if spread >= 0.12:
        return "高", spread
    if spread >= 0.06:
        return "中", spread
    return "低", spread


def format_slice(item: dict[str, Any]) -> str:
    return f"{item['label']} {item['starts']}走 / 勝{pct(item['win'])} / 複{pct(item['top3'])}"


def render_report(plan: dict[str, Any], pred: pd.DataFrame, entry: pd.DataFrame, history: pd.DataFrame, resolved: pd.DataFrame, output: Path) -> None:
    pred2 = pred.copy()
    pred2["race_id"] = pred2.apply(win5.official_race_id, axis=1)
    pred2["horse_no"] = pd.to_numeric(win5.col(pred2, ["馬番", "horse_no"], 0), errors="coerce").fillna(0).astype(int)
    entry2 = prepare_entry(entry)

    lines = [
        f"# WIN5 当日MVPレポート {plan['date']}",
        "",
        "用途: 本日購入前の候補精査。既存AI順位を置き換えず、TARGET全過去走からFIX可否・展開依存・条件適性を補助診断する。",
        "",
        "## 対象レース",
        "",
    ]
    for row in resolved.itertuples(index=False):
        lines.append(f"- WIN{row.leg_no}: {row.label} / race_id={row.race_id}")
    lines += ["", "## レース別診断", ""]

    for leg in plan["legs"]:
        race_id = str(leg["raceId"])
        leg_no = int(leg["legNo"])
        label = f"{leg['venue']}{leg['raceNo']}R"
        race_entry = entry2[entry2["race_id"].eq(race_id)].copy()
        if race_entry.empty:
            venue_series = entry2["場所"] if "場所" in entry2.columns else pd.Series("", index=entry2.index)
            race_series = entry2["Ｒ"] if "Ｒ" in entry2.columns else pd.Series(0, index=entry2.index)
            race_entry = entry2[
                venue_series.astype(str).str.strip().eq(str(leg["venue"]))
                & pd.to_numeric(race_series, errors="coerce").fillna(0).astype(int).eq(int(leg["raceNo"]))
            ].copy()
        race_pred = pred2[pred2["race_id"].eq(race_id)].copy()
        first_entry = race_entry.iloc[0] if not race_entry.empty else pd.Series(dtype=object)
        course = course_profile(history, first_entry) if not race_entry.empty else {"races": 0, "runners": 0}

        lines.append(f"### WIN{leg_no} {label} {leg.get('raceName','')}")
        lines.append("")
        surface = str(first_entry.get("芝・ダ", "")).strip()
        distance = int(safe_num(first_entry.get("距離"), 0) or 0)
        going = str(first_entry.get("馬場状態", "")).strip()
        course_bits = [x for x in [surface, f"{distance}m" if distance else "", going] if x]
        lines.append(f"- 条件: {' / '.join(course_bits) if course_bits else '-'}")
        lines.append(f"- 現行難易度: **{leg.get('difficulty_label')}** / base {leg.get('base_count')}頭")
        if course.get("races", 0):
            message = f"- TARGET同コース集計: {course['races']}R / {course['runners']}頭"
            if "winner_4c_median" in course:
                message += f" / 勝馬4角中央値 {course['winner_4c_median']:.1f}"
            if "winner_4c_front5_rate" in course:
                message += f" / 勝馬4角5番手以内 {pct(course['winner_4c_front5_rate'])}"
            if "winner_top3pop_rate" in course:
                message += f" / 勝馬3人気以内率 {pct(course['winner_top3pop_rate'])}"
            lines.append(message)

        top_candidates = leg["candidates"][: min(6, len(leg["candidates"]))]
        top_prob = float(top_candidates[0]["prob"]) if top_candidates else 0.0
        second_prob = float(top_candidates[1]["prob"]) if len(top_candidates) > 1 else 0.0
        lines += ["", "|順位|馬|WIN5比重|展開依存|過去走診断|", "|---:|---|---:|---|---|"]

        horse_profiles: list[tuple[dict[str, Any], dict[str, Any], str, float]] = []
        for rank, candidate in enumerate(top_candidates, start=1):
            horse_no = int(candidate["horseNo"])
            horse_name = str(candidate["horseName"])
            entry_rows = race_entry[race_entry["horse_no"].eq(horse_no)]
            entry_row = entry_rows.iloc[0] if not entry_rows.empty else pd.Series({"horse_name": horse_name, "horse_id": ""})
            pred_rows = race_pred[race_pred["horse_no"].eq(horse_no)]
            pred_row = pred_rows.iloc[0] if not pred_rows.empty else pd.Series(dtype=object)
            profile = build_horse_profile(history, entry_row)
            risk, spread = scenario_risk(pred_row)
            diagnosis = []
            if profile["strengths"]:
                diagnosis.append("強:" + "、".join(x["label"] for x in profile["strengths"][:2]))
            if profile["weaknesses"]:
                diagnosis.append("弱:" + "、".join(x["label"] for x in profile["weaknesses"][:2]))
            if profile["flags"]:
                diagnosis.append("注意:" + "、".join(profile["flags"][:2]))
            if not diagnosis:
                diagnosis.append(f"全{profile['career']['starts']}走")
            lines.append(f"|{rank}|{horse_no}.{horse_name}|{pct(float(candidate['prob']))}|{risk}|{' / '.join(diagnosis)}|")
            horse_profiles.append((candidate, profile, risk, spread))

        fix = "複数残し"
        reasons = []
        if top_candidates:
            first_profile = horse_profiles[0][1]
            first_risk = horse_profiles[0][2]
            gap = top_prob - second_prob
            severe_flags = {"経験不足", "今回類似条件未経験"}
            has_severe = any(flag in severe_flags for flag in first_profile["flags"])
            if (
                leg.get("difficulty_label") in {"堅め", "やや堅め"}
                and top_prob >= 0.34
                and gap >= 0.10
                and first_risk != "高"
                and not has_severe
            ):
                fix = f"暫定FIX候補: {top_candidates[0]['horseNo']}.{top_candidates[0]['horseName']}"
                reasons = ["AI集中", "2位差あり", "展開依存が極端でない"]
            else:
                if gap < 0.10:
                    reasons.append("上位差小")
                if first_risk == "高":
                    reasons.append("展開依存高")
                if has_severe:
                    reasons.append("条件/経験不確実")
                if leg.get("difficulty_label") not in {"堅め", "やや堅め"}:
                    reasons.append("レース難度高")

        lines += ["", f"**判断: {fix}**" + (f" — {', '.join(reasons)}" if reasons else ""), ""]
        for candidate, profile, risk, spread in horse_profiles[:4]:
            lines.append(f"#### {candidate['horseNo']}.{candidate['horseName']}")
            career = profile["career"]
            lines.append(f"- 全成績: {career['starts']}走 / 勝{pct(career['win'])} / 複{pct(career['top3'])}")
            if profile["slices"]:
                lines.append("- 条件別: " + " | ".join(format_slice(item) for item in profile["slices"]))
            if np.isfinite(spread):
                lines.append(f"- slow/middle/fast AIスコア差: {spread:.3f}（展開依存 {risk}）")
            lines.append("")

    lines += ["## 予算別フォーメーション", ""]
    for budget_plan in plan.get("plans", []):
        selections = []
        for selection in budget_plan["selections"]:
            horses = ",".join(str(horse["horseNo"]) for horse in selection["horses"])
            selections.append(f"{selection['raceLabel']}[{horses}]")
        lines.append(
            f"- {budget_plan['budgetYen']:,}円枠: {budget_plan['combos']}点 / {budget_plan['stakeYen']:,}円 / "
            + " → ".join(selections)
        )

    lines += [
        "",
        "## 注意",
        "",
        "- `win5_prob` は現行スコアから作る候補配分用の相対値で、厳密に校正済みの1着確率ではない。",
        "- 当日MVPではTARGET構造化データを優先し、映像による不利補正は未実装。",
        "- 購入前の最終判断では当日馬場・馬体重・取消変更を反映する。",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a practical same-day WIN5 report using existing predictions + TARGET history.")
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--prediction-csv", default="")
    parser.add_argument("--entry-csv", default="data/datasets/inference/weekly/entry_snapshot.csv")
    parser.add_argument("--history-csv", default="")
    parser.add_argument("--single-odds-csv", default="data/processed/live_odds/realtime_single_odds_latest.csv")
    parser.add_argument("--legs", nargs="*", default=[])
    parser.add_argument("--budgets", nargs="+", type=int, default=[5000, 10000, 25000])
    parser.add_argument("--output-dir", default="outputs/analysis/win5_runtime")
    args = parser.parse_args()

    date = re.sub(r"\D", "", args.date)
    if len(date) != 8:
        raise ValueError("--date must be YYYYMMDD")

    leg_labels = args.legs or DEFAULT_LEGS_BY_DATE.get(date, [])
    if len(leg_labels) != 5:
        raise ValueError("Exactly five --legs are required for this date, e.g. 中京6R 札幌10R 新潟7R 中京7R 札幌11R")

    prediction_csv = project_path(args.prediction_csv) if args.prediction_csv else latest_prediction()
    entry_csv = project_path(args.entry_csv)
    history_path, history_encoding = history_config()
    if args.history_csv:
        history_path = project_path(args.history_csv)
    odds_csv = project_path(args.single_odds_csv) if args.single_odds_csv else None

    pred = win5.read_csv(prediction_csv)
    resolved, race_ids = resolve_target_races(pred, date, leg_labels)
    entry = win5.read_csv(entry_csv)
    entry2 = prepare_entry(entry)
    target_entry = entry2[entry2["race_id"].isin(race_ids)].copy()
    if target_entry.empty:
        pieces = []
        for label in leg_labels:
            venue, race_no = parse_leg(label)
            venue_series = entry2["場所"] if "場所" in entry2.columns else pd.Series("", index=entry2.index)
            race_series = entry2["Ｒ"] if "Ｒ" in entry2.columns else pd.Series(0, index=entry2.index)
            mask = venue_series.astype(str).str.strip().eq(venue) & pd.to_numeric(race_series, errors="coerce").fillna(0).astype(int).eq(race_no)
            pieces.append(entry2[mask])
        target_entry = pd.concat(pieces, ignore_index=True) if pieces else entry2.iloc[0:0]

    if target_entry.empty:
        raise ValueError(f"No target entry rows found in {entry_csv}")

    history = load_relevant_history(history_path, history_encoding, target_entry, date)

    out_dir = project_path(args.output_dir)
    plan_json = out_dir / f"win5_plan_{date}.json"
    plan_csv = out_dir / f"win5_candidates_{date}.csv"
    report_md = out_dir / f"win5_today_report_{date}.md"

    plan = win5.build_plan(prediction_csv, odds_csv, date, args.budgets, race_ids)
    plan_json.parent.mkdir(parents=True, exist_ok=True)
    plan_json.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    win5.write_leg_csv(plan, plan_csv)
    render_report(plan, pred, entry, history, resolved, report_md)

    print(
        json.dumps(
            {
                "date": date,
                "prediction_csv": str(prediction_csv),
                "history_csv": str(history_path),
                "target_legs": leg_labels,
                "race_ids": race_ids,
                "plan_json": str(plan_json),
                "candidate_csv": str(plan_csv),
                "report_md": str(report_md),
                "history_rows_loaded": int(len(history)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
