from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RACE_ID_COL = "レースID(新/馬番無)"
VENUE_CODE_BY_NAME = {
    "札幌": "01",
    "函館": "02",
    "福島": "03",
    "新潟": "04",
    "東京": "05",
    "中山": "06",
    "中京": "07",
    "京都": "08",
    "阪神": "09",
    "小倉": "10",
}
VENUE_ORDER = {name: i for i, name in enumerate(VENUE_CODE_BY_NAME, start=1)}


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, **kwargs)


def text(value: object, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    return str(value).strip()


def num(value: object, default: float = np.nan) -> float:
    try:
        if value is None or pd.isna(value) or text(value) == "":
            return default
        return float(str(value).replace(",", ""))
    except Exception:
        return default


def col(frame: pd.DataFrame, names: list[str], default: object = "") -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series(default, index=frame.index)


def parse_date_key(value: object) -> str:
    raw = re.sub(r"\D", "", text(value))
    if len(raw) == 8 and raw.startswith("20"):
        return raw
    if len(raw) == 6:
        return "20" + raw
    return ""


def official_race_id(row: pd.Series) -> str:
    raw = text(row.get(RACE_ID_COL) if RACE_ID_COL in row.index else row.get("race_id"))
    digits = re.sub(r"\D", "", raw)
    if len(digits) == 16:
        return digits
    source_url = text(row.get("source_url"))
    match = re.search(r"race_id=(\d{12})", source_url)
    netkeiba_id = match.group(1) if match else ""
    date_key = parse_date_key(row.get("日付") if "日付" in row.index else row.get("date"))
    if date_key and len(netkeiba_id) == 12:
        return date_key + netkeiba_id[4:12]
    venue = text(row.get("場所"))
    race_no = int(num(row.get("Ｒ"), 0) or 0)
    if date_key and venue in VENUE_CODE_BY_NAME and race_no:
        # Last-resort synthetic id. Kaiji/nichiji are unknown, so this path is
        # only for grouping before official ids are present.
        return f"{date_key}{VENUE_CODE_BY_NAME[venue]}0000{race_no:02d}"
    return ""


def softmax_by_race(frame: pd.DataFrame, score_col: str, temperature: float = 0.085) -> pd.Series:
    out = pd.Series(0.0, index=frame.index, dtype=float)
    for race_id, idx in frame.groupby("race_id").groups.items():
        scores = pd.to_numeric(frame.loc[idx, score_col], errors="coerce").fillna(0.0)
        z = ((scores - scores.max()) / max(temperature, 0.001)).clip(-50, 50)
        exp = np.exp(z)
        total = float(exp.sum())
        out.loc[idx] = exp / total if total > 0 else 1.0 / len(idx)
    return out


def normalize_by_race(frame: pd.DataFrame, value_col: str) -> pd.Series:
    values = pd.to_numeric(frame[value_col], errors="coerce").fillna(0.0).clip(lower=0.0)
    sums = values.groupby(frame["race_id"]).transform("sum")
    field = frame.groupby("race_id")["race_id"].transform("size").clip(lower=1)
    return np.where(sums > 0, values / sums, 1.0 / field)


def class_score(value: object, race_name: object) -> float:
    raw = f"{text(value)} {text(race_name)}"
    if any(x in raw for x in ("Ｇ１", "G1", "GⅠ")):
        return 50
    if any(x in raw for x in ("Ｇ２", "G2", "GⅡ")):
        return 42
    if any(x in raw for x in ("Ｇ３", "G3", "GⅢ")):
        return 36
    if any(x in raw for x in ("OP", "オープン", "L", "リステッド", "S")):
        return 26
    if "3勝" in raw or "３勝" in raw:
        return 20
    if "2勝" in raw or "２勝" in raw:
        return 14
    if "1勝" in raw or "１勝" in raw:
        return 6
    return 0


def target_race_score(row: pd.Series) -> float:
    race_no = int(num(row.get("race_no"), 0) or 0)
    score = class_score(row.get("class_name"), row.get("race_name"))
    if race_no == 11:
        score += 80
    elif race_no == 10:
        score += 62
    elif race_no == 9:
        score += 35
    elif race_no == 12:
        score += 20
    score += max(0, race_no - 8)
    return float(score)


def select_target_races(races: pd.DataFrame, explicit_ids: list[str]) -> pd.DataFrame:
    if explicit_ids:
        wanted = [re.sub(r"\D", "", x) for x in explicit_ids]
        out = races[races["race_id"].isin(wanted)].copy()
        out["target_source"] = "explicit"
        return out.sort_values(["sort_time", "venue_order", "race_no"]).head(5)

    candidates = races[races["race_no"].isin([10, 11])].copy()
    if len(candidates) < 5:
        candidates = races[races["race_no"].isin([9, 10, 11, 12])].copy()
    candidates["target_score"] = candidates.apply(target_race_score, axis=1)
    out = candidates.sort_values(["target_score", "race_no"], ascending=[False, False]).head(5).copy()
    out["target_source"] = "auto_estimated"
    return out.sort_values(["sort_time", "venue_order", "race_no"])


def race_difficulty(group: pd.DataFrame) -> dict[str, Any]:
    probs = pd.to_numeric(group["win5_prob"], errors="coerce").fillna(0.0).sort_values(ascending=False).to_numpy()
    top = float(probs[0]) if len(probs) else 0.0
    second = float(probs[1]) if len(probs) > 1 else 0.0
    top3 = float(probs[:3].sum()) if len(probs) else 0.0
    entropy = -float(np.sum([p * math.log(max(p, 1e-12)) for p in probs])) / math.log(max(len(probs), 2))
    pressure = float(pd.to_numeric(group.get("race_early_pressure_score"), errors="coerce").fillna(0.0).max())
    confidence = float(pd.to_numeric(group.get("ai_confidence_score"), errors="coerce").fillna(0.0).max())
    field_size = int(len(group))
    score = (
        0.38 * top
        + 0.28 * max(top - second, 0.0)
        + 0.16 * top3
        + 0.12 * confidence
        - 0.18 * entropy
        - 0.06 * pressure
        - 0.025 * max(field_size - 14, 0)
    )
    if score >= 0.17:
        label = "堅め"
        base_count = 1
    elif score >= 0.10:
        label = "やや堅め"
        base_count = 2
    elif score >= 0.03:
        label = "中荒れ"
        base_count = 3
    else:
        label = "荒れ注意"
        base_count = 4
    if pressure >= 0.65 and base_count < 4:
        base_count += 1
    if field_size <= 9 and base_count > 2:
        base_count -= 1
    return {
        "top_prob": top,
        "second_prob": second,
        "top3_prob": top3,
        "entropy": entropy,
        "pressure": pressure,
        "confidence": confidence,
        "field_size": field_size,
        "difficulty_score": score,
        "difficulty_label": label,
        "base_count": int(max(1, min(base_count, min(field_size, 6)))),
    }


def fit_counts_to_budget(legs: list[dict[str, Any]], budget_yen: int) -> list[int]:
    max_combos = max(1, budget_yen // 100)
    counts = [int(leg["base_count"]) for leg in legs]

    def product(values: list[int]) -> int:
        out = 1
        for v in values:
            out *= max(1, int(v))
        return out

    while product(counts) > max_combos:
        reducible = [(counts[i], legs[i]["difficulty_score"], i) for i in range(len(counts)) if counts[i] > 1]
        if not reducible:
            break
        _, _, idx = sorted(reducible, key=lambda x: (-x[0], -x[1]))[0]
        counts[idx] -= 1

    expanded = True
    while expanded:
        expanded = False
        options: list[tuple[float, int]] = []
        for i, leg in enumerate(legs):
            if counts[i] >= min(int(leg["field_size"]), 6):
                continue
            next_count = counts[i] + 1
            next_prob = leg["candidates"][next_count - 1]["prob"] if len(leg["candidates"]) >= next_count else 0.0
            if product(counts[:i] + [next_count] + counts[i + 1 :]) <= max_combos:
                options.append((next_prob + 0.03 * max(0.0, -float(leg["difficulty_score"])), i))
        if options:
            _, idx = max(options)
            counts[idx] += 1
            expanded = True
    return counts


def build_plan(
    prediction_csv: Path,
    single_odds_csv: Path | None,
    date: str,
    budgets: list[int],
    race_ids: list[str],
) -> dict[str, Any]:
    pred = read_csv(prediction_csv)
    if pred.empty:
        raise ValueError(f"Prediction CSV is empty: {prediction_csv}")
    pred["race_id"] = pred.apply(official_race_id, axis=1)
    pred = pred[pred["race_id"].str.startswith(date)].copy()
    if pred.empty:
        raise ValueError(f"No prediction rows for date={date}: {prediction_csv}")

    pred["horse_no"] = pd.to_numeric(col(pred, ["馬番", "horse_no"], 0), errors="coerce").fillna(0).astype(int)
    pred = pred[pred["horse_no"].gt(0)].copy()
    pred["venue"] = col(pred, ["場所", "venue"], "")
    pred["venue_order"] = pred["venue"].map(VENUE_ORDER).fillna(99).astype(int)
    pred["race_no"] = pd.to_numeric(col(pred, ["Ｒ", "race_no"], 0), errors="coerce").fillna(0).astype(int)
    pred["race_name"] = col(pred, ["レース名", "race_name"], "")
    pred["class_name"] = col(pred, ["クラス名", "class_name"], "")
    pred["horse_name"] = col(pred, ["馬名", "horse_name"], "")
    pred["ai_score_num"] = pd.to_numeric(col(pred, ["ai_score"], 0.0), errors="coerce").fillna(0.0)
    pred["ai_rank_num"] = pd.to_numeric(col(pred, ["ai_rank"], 999), errors="coerce").fillna(999).astype(int)
    pred["confidence"] = pd.to_numeric(col(pred, ["ai_confidence_score"], 0.0), errors="coerce").fillna(0.0)
    pred["front5_hint"] = pd.to_numeric(col(pred, ["projected_front5_prob", "front_running_tendency"], 0.0), errors="coerce").fillna(0.0)
    pred["overlay_hint"] = pd.to_numeric(col(pred, ["market_overlay_score"], 0.0), errors="coerce").fillna(0.0)
    pred["sort_time"] = pred["venue_order"] * 100 + pred["race_no"]

    if single_odds_csv and single_odds_csv.exists():
        odds = read_csv(single_odds_csv, dtype={"race_id": str})
        if {"race_id", "horse_no", "live_win_odds"}.issubset(odds.columns):
            odds["horse_no"] = pd.to_numeric(odds["horse_no"], errors="coerce").fillna(0).astype(int)
            odds["live_win_odds"] = pd.to_numeric(odds["live_win_odds"], errors="coerce")
            odds["live_popularity"] = pd.to_numeric(odds.get("live_popularity"), errors="coerce")
            odds = odds.sort_values("snapshot_at" if "snapshot_at" in odds.columns else "race_id")
            odds = odds.drop_duplicates(["race_id", "horse_no"], keep="last")
            pred = pred.merge(
                odds[["race_id", "horse_no", "live_win_odds", "live_popularity"]],
                on=["race_id", "horse_no"],
                how="left",
            )
    if "live_win_odds" not in pred.columns:
        pred["live_win_odds"] = np.nan
    if "live_popularity" not in pred.columns:
        pred["live_popularity"] = np.nan

    pred["model_prob"] = softmax_by_race(pred, "ai_score_num")
    implied = 1.0 / pd.to_numeric(pred["live_win_odds"], errors="coerce").replace(0, np.nan)
    pred["market_prob"] = implied.groupby(pred["race_id"]).transform(lambda s: s / s.sum() if s.notna().any() and s.sum() > 0 else np.nan)
    blend = np.where(pred["market_prob"].notna(), 0.68 * pred["model_prob"] + 0.32 * pred["market_prob"], pred["model_prob"])
    pred["win5_raw"] = (
        blend
        * (0.92 + 0.12 * pred["confidence"].clip(0, 1))
        * (0.96 + 0.04 * pred["front5_hint"].clip(0, 1))
        + 0.012 * pred["overlay_hint"].clip(0, 1)
    )
    pred["win5_prob"] = normalize_by_race(pred, "win5_raw")
    pred["edge_note"] = np.where(
        pred["market_prob"].notna() & (pred["model_prob"] > pred["market_prob"] * 1.25),
        "AI>市場",
        np.where(pred["ai_rank_num"].le(2), "AI上位", "押さえ")
    )

    races = (
        pred.groupby("race_id", as_index=False)
        .agg(
            venue=("venue", "first"),
            venue_order=("venue_order", "first"),
            race_no=("race_no", "first"),
            race_name=("race_name", "first"),
            class_name=("class_name", "first"),
            sort_time=("sort_time", "first"),
        )
    )
    targets = select_target_races(races, race_ids)
    if len(targets) < 5:
        raise ValueError(f"Could not infer 5 WIN5 target races. inferred={len(targets)}")

    legs: list[dict[str, Any]] = []
    for leg_no, race in enumerate(targets.itertuples(index=False), start=1):
        group = pred[pred["race_id"].eq(race.race_id)].copy()
        group = group.sort_values(["win5_prob", "ai_rank_num"], ascending=[False, True])
        diff = race_difficulty(group)
        candidates = []
        for _, row in group.head(8).iterrows():
            candidates.append(
                {
                    "horseNo": int(row["horse_no"]),
                    "horseName": text(row["horse_name"]),
                    "prob": round(float(row["win5_prob"]), 5),
                    "modelProb": round(float(row["model_prob"]), 5),
                    "marketProb": None if pd.isna(row["market_prob"]) else round(float(row["market_prob"]), 5),
                    "aiRank": int(row["ai_rank_num"]),
                    "aiScore": round(float(row["ai_score_num"]), 5),
                    "odds": None if pd.isna(row["live_win_odds"]) else round(float(row["live_win_odds"]), 1),
                    "popularity": None if pd.isna(row["live_popularity"]) else int(row["live_popularity"]),
                    "note": text(row["edge_note"]),
                }
            )
        leg = {
            "legNo": leg_no,
            "raceId": race.race_id,
            "venue": race.venue,
            "raceNo": int(race.race_no),
            "raceName": text(race.race_name),
            "className": text(race.class_name),
            **diff,
            "candidates": candidates,
        }
        legs.append(leg)

    plans = []
    for budget in budgets:
        counts = fit_counts_to_budget(legs, budget)
        selections = []
        hit_prob = 1.0
        combo_count = 1
        for leg, count in zip(legs, counts):
            selected = leg["candidates"][:count]
            leg_prob = float(sum(c["prob"] for c in selected))
            hit_prob *= leg_prob
            combo_count *= count
            selections.append(
                {
                    "legNo": leg["legNo"],
                    "raceId": leg["raceId"],
                    "raceLabel": f"{leg['venue']}{leg['raceNo']}R",
                    "count": count,
                    "coverageProb": round(leg_prob, 5),
                    "horses": selected,
                }
            )
        plans.append(
            {
                "budgetYen": int(budget),
                "combos": int(combo_count),
                "stakeYen": int(combo_count * 100),
                "unusedYen": int(max(0, budget - combo_count * 100)),
                "estimatedHitProb": round(float(hit_prob), 7),
                "selections": selections,
            }
        )

    combo_preview = []
    main_plan = plans[-1] if plans else {}
    if main_plan:
        horse_lists = [[(h["horseNo"], h["horseName"]) for h in leg["horses"]] for leg in main_plan["selections"]]
        for combo in itertools.islice(itertools.product(*horse_lists), 20):
            combo_preview.append(" / ".join(f"{no}.{name}" for no, name in combo))

    return {
        "generatedAt": datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
        "date": date,
        "targetSource": "explicit" if race_ids else "auto_estimated",
        "note": "WIN5 target races are auto-estimated unless --race-ids is supplied.",
        "legs": legs,
        "plans": plans,
        "combinationPreview": combo_preview,
    }


def write_leg_csv(plan: dict[str, Any], path: Path) -> None:
    rows = []
    for leg in plan.get("legs", []):
        for c in leg.get("candidates", []):
            rows.append(
                {
                    "date": plan.get("date"),
                    "leg_no": leg.get("legNo"),
                    "race_id": leg.get("raceId"),
                    "race_label": f"{leg.get('venue')}{leg.get('raceNo')}R",
                    "race_name": leg.get("raceName"),
                    "difficulty_label": leg.get("difficulty_label"),
                    "base_count": leg.get("base_count"),
                    "horse_no": c.get("horseNo"),
                    "horse_name": c.get("horseName"),
                    "win5_prob": c.get("prob"),
                    "ai_rank": c.get("aiRank"),
                    "odds": c.get("odds"),
                    "popularity": c.get("popularity"),
                    "note": c.get("note"),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a WIN5-specific candidate plan.")
    parser.add_argument("--prediction-csv", required=True)
    parser.add_argument("--single-odds-csv", default="data/processed/live_odds/realtime_single_odds_latest.csv")
    parser.add_argument("--date", required=True)
    parser.add_argument("--budgets", nargs="+", type=int, default=[1000, 3000, 5000, 10000])
    parser.add_argument("--race-ids", nargs="*", default=[])
    parser.add_argument("--output-json", default="outputs/analysis/win5_runtime/win5_plan.json")
    parser.add_argument("--output-csv", default="outputs/analysis/win5_runtime/win5_candidates.csv")
    args = parser.parse_args()

    prediction_csv = project_path(args.prediction_csv)
    single_odds_csv = project_path(args.single_odds_csv) if args.single_odds_csv else None
    plan = build_plan(prediction_csv, single_odds_csv, args.date, args.budgets, args.race_ids)

    output_json = project_path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    write_leg_csv(plan, project_path(args.output_csv))
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "legs": len(plan.get("legs", [])),
                "plans": [
                    {"budgetYen": p["budgetYen"], "combos": p["combos"], "stakeYen": p["stakeYen"]}
                    for p in plan.get("plans", [])
                ],
                "targetSource": plan.get("targetSource"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
