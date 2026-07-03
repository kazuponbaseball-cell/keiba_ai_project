from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluate_pedigree_ancestor_tag_overlay import TAG_PATTERNS


DEFAULT_TRAIN = "outputs/analysis/content_bridge_member_features_v1/train_features_with_content_bridge.csv"
DEFAULT_TEST = "outputs/analysis/content_bridge_member_features_v1/test_features_with_content_bridge.csv"
DEFAULT_DEEP_PEDIGREE_MASTER = "data/processed/target/deep_pedigree_master.csv"
DEFAULT_OUT = "outputs/analysis/pedigree_rotation_runner_value_v1"


CONTENT_COLS = [
    "レースID(新/馬番無)",
    "馬番",
    "血統登録番号",
    "date",
    "場所",
    "venue",
    "surface",
    "distance",
    "馬場状態",
    "race_class_name",
    "クラス名",
    "枠番",
    "確定着順",
    "人気",
    "単勝オッズ",
    "単勝配当",
    "複勝配当",
    "rotation_distance_up_flag",
    "rotation_distance_down_flag",
    "rotation_big_distance_change_flag",
    "rotation_surface_switch_flag",
    "rotation_class_up_flag",
    "rotation_class_down_flag",
    "distance_diff",
    "斤量",
    "前走斤量",
    "horse_fast_lap_score_past5",
    "fast_clock_aptitude_score",
]


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False, **kwargs)


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def text(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series("", index=frame.index, dtype=str)
    return frame[col].fillna("").astype(str)


def boolish(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    s = frame[col]
    if s.dtype == bool:
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).ne(0)
    return s.fillna("").astype(str).str.lower().isin(["true", "1", "yes"])


def tag_count(frame: pd.DataFrame, tag: str) -> pd.Series:
    return num(frame, f"tag_{tag}_count", 0.0)


def tag_flag(frame: pd.DataFrame, tag: str) -> pd.Series:
    if f"tag_{tag}_count" in frame.columns:
        return tag_count(frame, tag).gt(0)
    return boolish(frame, f"tag_{tag}")


def side_count(frame: pd.DataFrame, tag: str, side: str) -> pd.Series:
    return num(frame, f"tag_{tag}_{side}_side_count", 0.0)


def cross_flag(frame: pd.DataFrame, tag: str) -> pd.Series:
    if f"tag_{tag}_cross_flag" in frame.columns:
        return boolish(frame, f"tag_{tag}_cross_flag")
    return tag_count(frame, tag).ge(2)


def surface(frame: pd.DataFrame, value: str) -> pd.Series:
    return text(frame, "surface").eq(value)


def bad_going(frame: pd.DataFrame) -> pd.Series:
    return text(frame, "馬場状態").isin(["稍", "稍重", "重", "不", "不良"])


def local(frame: pd.DataFrame) -> pd.Series:
    venue = text(frame, "venue")
    if venue.eq("").all():
        venue = text(frame, "場所")
    return venue.isin(["札幌", "函館", "福島", "新潟", "小倉"])


def small_turn(frame: pd.DataFrame) -> pd.Series:
    venue = text(frame, "venue")
    if venue.eq("").all():
        venue = text(frame, "場所")
    return venue.isin(["札幌", "函館", "福島", "中山", "小倉"])


def distance(frame: pd.DataFrame) -> pd.Series:
    return num(frame, "distance", np.nan)


def race_class(frame: pd.DataFrame) -> pd.Series:
    if "race_class_name" in frame.columns:
        return text(frame, "race_class_name")
    return text(frame, "クラス名")


def frame_no(frame: pd.DataFrame) -> pd.Series:
    if "frame_no" not in frame.columns and "枠番" in frame.columns:
        return num(frame, "枠番", np.nan)
    return num(frame, "frame_no", np.nan)


def fast_lap(frame: pd.DataFrame) -> pd.Series:
    return num(frame, "horse_fast_lap_score_past5", 0.0)


def fast_clock(frame: pd.DataFrame) -> pd.Series:
    return num(frame, "fast_clock_aptitude_score", 0.0)


def us_power(frame: pd.DataFrame) -> pd.Series:
    return tag_flag(frame, "storm_cat") | tag_flag(frame, "mr_prospector") | tag_flag(frame, "danzig") | tag_flag(frame, "deputy_minister")


def euro_stamina(frame: pd.DataFrame) -> pd.Series:
    return tag_flag(frame, "roberto") | tag_flag(frame, "sadlers_wells") | tag_flag(frame, "nureyev") | tag_flag(frame, "blushing_groom")


def halo_family(frame: pd.DataFrame) -> pd.Series:
    return tag_flag(frame, "halo") | tag_flag(frame, "hail_to_reason")


def hypotheses() -> list[dict[str, Any]]:
    return [
        {"name": "us_power_dirt_sprint", "label": "米国スピード/パワー血統 × ダート短距離", "mask": lambda f: us_power(f) & surface(f, "ダ") & distance(f).le(1400)},
        {"name": "us_power_dirt_distance_down", "label": "米国スピード/パワー血統 × ダート距離短縮", "mask": lambda f: us_power(f) & surface(f, "ダ") & boolish(f, "rotation_distance_down_flag")},
        {"name": "us_power_dirt_surface_switch", "label": "米国スピード/パワー血統 × 芝ダ替わりダート", "mask": lambda f: us_power(f) & surface(f, "ダ") & boolish(f, "rotation_surface_switch_flag")},
        {"name": "mrp_dam_dirt_surface_switch", "label": "母系Mr. Prospector系 × 芝ダ替わりダート", "mask": lambda f: side_count(f, "mr_prospector", "dam").gt(0) & surface(f, "ダ") & boolish(f, "rotation_surface_switch_flag")},
        {"name": "deputy_minister_dirt_surface_switch", "label": "Deputy Minister系 × 芝ダ替わりダート", "mask": lambda f: tag_flag(f, "deputy_minister") & surface(f, "ダ") & boolish(f, "rotation_surface_switch_flag")},
        {"name": "danzig_turf_distance_down", "label": "Danzig系 × 芝距離短縮", "mask": lambda f: tag_flag(f, "danzig") & surface(f, "芝") & boolish(f, "rotation_distance_down_flag")},
        {"name": "danzig_maiden_new", "label": "Danzig系 × 新馬/未勝利", "mask": lambda f: tag_flag(f, "danzig") & race_class(f).str.contains("新馬|未勝", regex=True, na=False)},
        {"name": "kingmambo_fast_lap", "label": "Kingmambo系 × 速いラップ適性", "mask": lambda f: tag_flag(f, "kingmambo") & fast_lap(f).ge(0.55)},
        {"name": "kingmambo_fast_lap_class_up", "label": "Kingmambo系 × 速いラップ適性 × 昇級", "mask": lambda f: tag_flag(f, "kingmambo") & fast_lap(f).ge(0.55) & boolish(f, "rotation_class_up_flag")},
        {"name": "kingmambo_bad_going", "label": "Kingmambo系 × 道悪", "mask": lambda f: tag_flag(f, "kingmambo") & bad_going(f)},
        {"name": "roberto_bad_going", "label": "Roberto系 × 道悪", "mask": lambda f: tag_flag(f, "roberto") & bad_going(f)},
        {"name": "roberto_distance_up_midlong", "label": "Roberto系 × 距離延長中長距離", "mask": lambda f: tag_flag(f, "roberto") & boolish(f, "rotation_distance_up_flag") & distance(f).ge(1800)},
        {"name": "euro_stamina_turf_distance_up", "label": "欧州/スタミナ血統 × 芝距離延長", "mask": lambda f: euro_stamina(f) & surface(f, "芝") & boolish(f, "rotation_distance_up_flag")},
        {"name": "sadlers_midlong_turf_class_up", "label": "Sadler's Wells系 × 芝中長距離昇級", "mask": lambda f: tag_flag(f, "sadlers_wells") & surface(f, "芝") & distance(f).ge(1800) & boolish(f, "rotation_class_up_flag")},
        {"name": "nureyev_smallturn_outer", "label": "Nureyev系 × 小回り外枠", "mask": lambda f: tag_flag(f, "nureyev") & small_turn(f) & frame_no(f).ge(7)},
        {"name": "halo_fast_clock_turf_inner", "label": "Halo/Sunday系 × 高速芝内枠", "mask": lambda f: halo_family(f) & surface(f, "芝") & fast_clock(f).ge(0.62) & frame_no(f).le(2)},
        {"name": "northern_dancer_cross_bad", "label": "Northern Dancerクロス × 道悪", "mask": lambda f: cross_flag(f, "northern_dancer") & bad_going(f)},
    ]


def metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"horses": 0, "races": 0, "win_roi": 0.0, "place_roi": 0.0, "win_rate": 0.0, "top3_rate": 0.0, "avg_popularity": 0.0, "avg_odds": 0.0}
    finish = num(frame, "確定着順", 99)
    stake = len(frame) * 100.0
    win_ret = np.where(finish.eq(1), num(frame, "単勝配当", 0.0), 0.0)
    place_ret = np.where(finish.le(3), num(frame, "複勝配当", 0.0), 0.0)
    return {
        "horses": int(len(frame)),
        "races": int(frame["race_id"].nunique()) if "race_id" in frame.columns else 0,
        "win_roi": float(np.sum(win_ret) / stake) if stake else 0.0,
        "place_roi": float(np.sum(place_ret) / stake) if stake else 0.0,
        "win_rate": float(finish.eq(1).mean()),
        "top3_rate": float(finish.le(3).mean()),
        "avg_popularity": float(num(frame, "人気", np.nan).mean()),
        "avg_odds": float(num(frame, "単勝オッズ", np.nan).mean()),
    }


def evaluate(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    filters = {
        "all": pd.Series(True, index=frame.index),
        "pop4plus": num(frame, "人気", 99).ge(4),
        "pop6plus": num(frame, "人気", 99).ge(6),
        "odds10plus": num(frame, "単勝オッズ", 0).ge(10),
    }
    rows = []
    for hypo in hypotheses():
        base = hypo["mask"](frame).fillna(False)
        for filter_name, filter_mask in filters.items():
            part = frame[base & filter_mask]
            rows.append({"source": source, "filter": filter_name, "hypothesis": hypo["name"], "label": hypo["label"], **metrics(part)})
    return pd.DataFrame(rows)


def term_hit(frame: pd.DataFrame, columns: list[str], terms: list[str]) -> pd.DataFrame:
    if not columns:
        return pd.DataFrame(False, index=frame.index, columns=[])
    text_frame = frame[columns].fillna("").astype(str)
    out = pd.DataFrame(False, index=frame.index, columns=columns)
    for term in terms:
        if not term:
            continue
        out |= text_frame.apply(lambda s: s.str.contains(term, case=False, regex=False, na=False))
    return out


def load_master_tags(path: Path) -> pd.DataFrame:
    master = read_csv(path)
    reg_col = "血統登録番号"
    ancestor_cols = [
        c
        for c in master.columns
        if c.endswith("馬") and c not in [reg_col]
    ]
    sire_cols = [c for c in ancestor_cols if c.startswith("父")]
    dam_cols = [c for c in ancestor_cols if c.startswith("母")]
    out = master[[reg_col]].copy()
    out[reg_col] = out[reg_col].astype(str)
    for tag, terms in TAG_PATTERNS.items():
        hits = term_hit(master, ancestor_cols, terms)
        out[f"tag_{tag}_count"] = hits.sum(axis=1).astype(int)
        out[f"tag_{tag}"] = out[f"tag_{tag}_count"].gt(0)
        out[f"tag_{tag}_sire_side_count"] = hits[sire_cols].sum(axis=1).astype(int) if sire_cols else 0
        out[f"tag_{tag}_dam_side_count"] = hits[dam_cols].sum(axis=1).astype(int) if dam_cols else 0
        out[f"tag_{tag}_cross_flag"] = out[f"tag_{tag}_count"].ge(2)
    out["tag_total_ancestor_hits"] = out[[f"tag_{tag}_count" for tag in TAG_PATTERNS]].sum(axis=1)
    return out.drop_duplicates(subset=[reg_col])


def load_content(path: Path, tags: pd.DataFrame) -> pd.DataFrame:
    sample = read_csv(path, nrows=1)
    usecols = [c for c in CONTENT_COLS if c in sample.columns]
    if "distance" not in sample.columns:
        distance_values = {1000, 1150, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 3000, 3200, 3400, 3600}
        for col in sample.columns:
            if "distance" in col.lower() or col in usecols:
                continue
            value = pd.to_numeric(sample[col], errors="coerce").iloc[0]
            if pd.notna(value) and int(value) in distance_values:
                usecols.append(col)
                break
    frame = read_csv(path, usecols=usecols)
    if "distance" not in frame.columns:
        distance_values = {1000, 1150, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 3000, 3200, 3400, 3600}
        for col in frame.columns:
            if "distance" in col.lower():
                continue
            values = pd.to_numeric(frame[col], errors="coerce")
            if values.dropna().isin(distance_values).mean() > 0.90:
                frame["distance"] = values
                break
    frame["race_id"] = frame["レースID(新/馬番無)"].astype(str)
    frame["horse_no"] = pd.to_numeric(frame["馬番"], errors="coerce").astype("Int64")
    frame["血統登録番号"] = frame["血統登録番号"].astype(str)
    return frame.merge(tags, on="血統登録番号", how="left")


def judgement(row: pd.Series) -> str:
    horses = float(row.get("horses", 0) or 0)
    win_roi = float(row.get("win_roi", 0) or 0)
    place_roi = float(row.get("place_roi", 0) or 0)
    if horses >= 500 and win_roi >= 1.05 and place_roi >= 0.90:
        return "market_value_candidate"
    if horses >= 250 and win_roi >= 1.10:
        return "win_value_watch"
    if horses >= 250 and place_roi >= 0.95:
        return "place_stability_watch"
    if horses < 250 and (win_roi >= 1.2 or place_roi >= 1.0):
        return "too_thin"
    return "reject"


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                val = f"{val:.4f}" if math.isfinite(val) else ""
            vals.append(str(val).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def render_review(summary: dict[str, Any], result: pd.DataFrame) -> str:
    cols = ["source", "filter", "hypothesis", "label", "horses", "races", "win_roi", "place_roi", "win_rate", "top3_rate", "avg_popularity", "avg_odds", "judgement"]
    watch = result[result["judgement"].ne("reject")].sort_values(["judgement", "win_roi"], ascending=[True, False])
    test_top = result[result["source"].eq("test")].sort_values("win_roi", ascending=False).head(25)
    lines = [
        "# 血統×ローテ/条件 単馬市場価値検証",
        "",
        "## 入力",
        "",
        f"- train: `{summary['train_csv']}`",
        f"- test: `{summary['test_csv']}`",
        f"- deep_pedigree_master: `{summary['deep_pedigree_master']}`",
        "",
        "## 監視候補",
        "",
        markdown_table(watch[cols].head(40).to_dict("records")),
        "",
        "## test単勝ROI上位",
        "",
        markdown_table(test_top[cols].to_dict("records")),
        "",
        "## 判断",
        "",
        "- これはペア買い目の直接検証ではなく、市場が血統条件を過小評価しているかの単馬検証。",
        "- testで残り、かつ人気薄フィルタでも残るものだけを、ペア相手候補の補助特徴量にする。",
        "- 血統ベタ買いは弱いので、BUY条件ではなくシャドー特徴量/相手選びの説明変数として扱う。",
    ]
    return "\n".join(lines) + "\n"


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", default=DEFAULT_TRAIN)
    parser.add_argument("--test-csv", default=DEFAULT_TEST)
    parser.add_argument("--deep-pedigree-master", default=DEFAULT_DEEP_PEDIGREE_MASTER)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tags = load_master_tags(Path(args.deep_pedigree_master))
    frames = []
    for source, path_str in [("train", args.train_csv), ("test", args.test_csv)]:
        frame = load_content(Path(path_str), tags)
        coverage = float(frame["tag_total_ancestor_hits"].notna().mean()) if "tag_total_ancestor_hits" in frame.columns else 0.0
        result = evaluate(frame, source)
        result["tag_coverage"] = coverage
        frames.append(result)
    result_all = pd.concat(frames, ignore_index=True)
    result_all["judgement"] = result_all.apply(judgement, axis=1)
    result_all.to_csv(out_dir / "runner_value_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"hypothesis": h["name"], "label": h["label"]} for h in hypotheses()]).to_csv(out_dir / "hypothesis_catalog.csv", index=False, encoding="utf-8-sig")
    summary = {
        "train_csv": args.train_csv,
        "test_csv": args.test_csv,
        "deep_pedigree_master": args.deep_pedigree_master,
        "output_dir": str(out_dir),
        "hypotheses": len(hypotheses()),
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "review.md").write_text(render_review(summary, result_all), encoding="utf-8")
    print(json.dumps(json_ready({"ok": True, **summary}), ensure_ascii=False))


if __name__ == "__main__":
    main()
