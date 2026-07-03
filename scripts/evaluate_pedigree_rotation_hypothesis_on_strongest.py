from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


DEFAULT_ENRICHED = "outputs/analysis/optimized_stack_ab_s_gate_v1/s_priority_enriched_tickets.csv"
DEFAULT_SELECTED = "outputs/analysis/optimized_stack_ab_s_gate_v1/s_priority_selected_tickets.csv"
DEFAULT_RUNNER_TAGS = "outputs/analysis/deep_pedigree_fine_context_overlay_v1/runner_pedigree_tag_profile.csv"
DEFAULT_OUT = "outputs/analysis/pedigree_rotation_hypothesis_strongest_v1"


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


def pcol(role: str, col: str) -> str:
    return f"{role}_{col}"


def tag_count(frame: pd.DataFrame, role: str, tag: str) -> pd.Series:
    return num(frame, pcol(role, f"tag_{tag}_count"), 0.0)


def tag_flag(frame: pd.DataFrame, role: str, tag: str) -> pd.Series:
    count_col = pcol(role, f"tag_{tag}_count")
    flag_col = pcol(role, f"tag_{tag}")
    if count_col in frame.columns:
        return tag_count(frame, role, tag).gt(0)
    return boolish(frame, flag_col)


def side_count(frame: pd.DataFrame, role: str, tag: str, side: str) -> pd.Series:
    return num(frame, pcol(role, f"tag_{tag}_{side}_side_count"), 0.0)


def cross_flag(frame: pd.DataFrame, role: str, tag: str) -> pd.Series:
    col = pcol(role, f"tag_{tag}_cross_flag")
    if col in frame.columns:
        return boolish(frame, col)
    return tag_count(frame, role, tag).ge(2)


def surface(frame: pd.DataFrame, value: str) -> pd.Series:
    return text(frame, "surface").eq(value)


def bad_going(frame: pd.DataFrame) -> pd.Series:
    return text(frame, "馬場状態").isin(["稍", "稍重", "重", "不", "不良"])


def local(frame: pd.DataFrame) -> pd.Series:
    return text(frame, "venue").isin(["札幌", "函館", "福島", "新潟", "小倉"])


def small_turn(frame: pd.DataFrame) -> pd.Series:
    return text(frame, "venue").isin(["札幌", "函館", "福島", "中山", "小倉"])


def distance(frame: pd.DataFrame) -> pd.Series:
    return num(frame, "distance", np.nan)


def race_class(frame: pd.DataFrame) -> pd.Series:
    return text(frame, "race_class_name")


def frame_no(frame: pd.DataFrame, role: str) -> pd.Series:
    return num(frame, pcol(role, "frame_no"), np.nan)


def fast_lap(frame: pd.DataFrame, role: str) -> pd.Series:
    return num(frame, pcol(role, "horse_fast_lap_score_past5"), 0.0)


def fast_clock(frame: pd.DataFrame, role: str) -> pd.Series:
    return num(frame, pcol(role, "fast_clock_aptitude_score"), 0.0)


def distance_down(frame: pd.DataFrame, role: str) -> pd.Series:
    return boolish(frame, pcol(role, "rotation_distance_down_flag"))


def distance_up(frame: pd.DataFrame, role: str) -> pd.Series:
    return boolish(frame, pcol(role, "rotation_distance_up_flag"))


def surface_switch(frame: pd.DataFrame, role: str) -> pd.Series:
    return boolish(frame, pcol(role, "rotation_surface_switch_flag"))


def class_up(frame: pd.DataFrame, role: str) -> pd.Series:
    return boolish(frame, pcol(role, "rotation_class_up_flag"))


def class_down(frame: pd.DataFrame, role: str) -> pd.Series:
    return boolish(frame, pcol(role, "rotation_class_down_flag"))


def us_power(frame: pd.DataFrame, role: str) -> pd.Series:
    return (
        tag_flag(frame, role, "storm_cat")
        | tag_flag(frame, role, "mr_prospector")
        | tag_flag(frame, role, "danzig")
        | tag_flag(frame, role, "deputy_minister")
    )


def euro_stamina(frame: pd.DataFrame, role: str) -> pd.Series:
    return (
        tag_flag(frame, role, "roberto")
        | tag_flag(frame, role, "sadlers_wells")
        | tag_flag(frame, role, "nureyev")
        | tag_flag(frame, role, "blushing_groom")
    )


def halo_family(frame: pd.DataFrame, role: str) -> pd.Series:
    return tag_flag(frame, role, "halo") | tag_flag(frame, role, "hail_to_reason")


MaskFunc = Callable[[pd.DataFrame, str], pd.Series]


def hypotheses() -> list[dict[str, Any]]:
    return [
        {
            "name": "us_power_dirt_distance_down",
            "label": "米国スピード/パワー血統 × ダート距離短縮",
            "mask": lambda f, r: us_power(f, r) & surface(f, "ダ") & distance_down(f, r),
        },
        {
            "name": "us_power_dirt_surface_switch",
            "label": "米国スピード/パワー血統 × 芝ダ替わりダート",
            "mask": lambda f, r: us_power(f, r) & surface(f, "ダ") & surface_switch(f, r),
        },
        {
            "name": "mrp_dam_dirt_surface_switch",
            "label": "母系Mr. Prospector系 × 芝ダ替わりダート",
            "mask": lambda f, r: side_count(f, r, "mr_prospector", "dam").gt(0) & surface(f, "ダ") & surface_switch(f, r),
        },
        {
            "name": "deputy_minister_dirt_surface_switch",
            "label": "Deputy Minister系 × 芝ダ替わりダート",
            "mask": lambda f, r: tag_flag(f, r, "deputy_minister") & surface(f, "ダ") & surface_switch(f, r),
        },
        {
            "name": "danzig_turf_distance_down",
            "label": "Danzig系 × 芝距離短縮",
            "mask": lambda f, r: tag_flag(f, r, "danzig") & surface(f, "芝") & distance_down(f, r),
        },
        {
            "name": "danzig_maiden_new",
            "label": "Danzig系 × 新馬/未勝利",
            "mask": lambda f, r: tag_flag(f, r, "danzig") & race_class(f).str.contains("新馬|未勝", regex=True, na=False),
        },
        {
            "name": "kingmambo_fast_lap",
            "label": "Kingmambo系 × 速いラップ適性",
            "mask": lambda f, r: tag_flag(f, r, "kingmambo") & fast_lap(f, r).ge(0.55),
        },
        {
            "name": "kingmambo_fast_lap_class_up",
            "label": "Kingmambo系 × 速いラップ適性 × 昇級",
            "mask": lambda f, r: tag_flag(f, r, "kingmambo") & fast_lap(f, r).ge(0.55) & class_up(f, r),
        },
        {
            "name": "kingmambo_bad_going",
            "label": "Kingmambo系 × 道悪",
            "mask": lambda f, r: tag_flag(f, r, "kingmambo") & bad_going(f),
        },
        {
            "name": "roberto_bad_going",
            "label": "Roberto系 × 道悪",
            "mask": lambda f, r: tag_flag(f, r, "roberto") & bad_going(f),
        },
        {
            "name": "roberto_distance_up_midlong",
            "label": "Roberto系 × 距離延長中長距離",
            "mask": lambda f, r: tag_flag(f, r, "roberto") & distance_up(f, r) & distance(f).ge(1800),
        },
        {
            "name": "euro_stamina_turf_distance_up",
            "label": "欧州/スタミナ血統 × 芝距離延長",
            "mask": lambda f, r: euro_stamina(f, r) & surface(f, "芝") & distance_up(f, r),
        },
        {
            "name": "sadlers_midlong_turf_class_up",
            "label": "Sadler's Wells系 × 芝中長距離昇級",
            "mask": lambda f, r: tag_flag(f, r, "sadlers_wells") & surface(f, "芝") & distance(f).ge(1800) & class_up(f, r),
        },
        {
            "name": "nureyev_smallturn_outer",
            "label": "Nureyev系 × 小回り外枠",
            "mask": lambda f, r: tag_flag(f, r, "nureyev") & small_turn(f) & frame_no(f, r).ge(7),
        },
        {
            "name": "halo_fast_clock_turf_inner",
            "label": "Halo/Sunday系 × 高速芝内枠",
            "mask": lambda f, r: halo_family(f, r) & surface(f, "芝") & fast_clock(f, r).ge(0.62) & frame_no(f, r).le(2),
        },
        {
            "name": "northern_dancer_cross_bad",
            "label": "Northern Dancerクロス × 道悪",
            "mask": lambda f, r: cross_flag(f, r, "northern_dancer") & bad_going(f),
        },
        {
            "name": "local_dirt_us_power_class_down",
            "label": "米国パワー血統 × ローカルダート降級/相手弱化",
            "mask": lambda f, r: us_power(f, r) & local(f) & surface(f, "ダ") & class_down(f, r),
        },
    ]


def metrics(frame: pd.DataFrame, stake_col: str = "stake_yen", return_col: str = "return_yen") -> dict[str, Any]:
    if frame.empty:
        return {
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi": 0.0,
            "hit_rate": 0.0,
            "top5_removed_roi": 0.0,
            "top10_removed_roi": 0.0,
        }
    stake = num(frame, stake_col, 0.0)
    ret = num(frame, return_col, 0.0)
    out = {
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()) if "race_id" in frame.columns else 0,
        "stake_yen": float(stake.sum()),
        "return_yen": float(ret.sum()),
        "profit_yen": float(ret.sum() - stake.sum()),
        "roi": float(ret.sum() / stake.sum()) if stake.sum() > 0 else 0.0,
        "hit_rate": float(ret.gt(0).mean()) if len(frame) else 0.0,
    }
    for n in (5, 10):
        if len(frame) <= n:
            out[f"top{n}_removed_roi"] = 0.0
            continue
        drop = ret.sort_values(ascending=False).index[:n]
        stake2 = stake.drop(index=drop)
        ret2 = ret.drop(index=drop)
        out[f"top{n}_removed_roi"] = float(ret2.sum() / stake2.sum()) if stake2.sum() > 0 else 0.0
    return out


def judgement(row: pd.Series) -> str:
    races = float(row.get("races", 0) or 0)
    roi = float(row.get("roi", 0) or 0)
    top5 = float(row.get("top5_removed_roi", 0) or 0)
    if races >= 150 and roi >= 1.20 and top5 >= 1.0:
        return "strong_shadow"
    if races >= 80 and roi >= 1.20 and top5 >= 0.75:
        return "shadow_candidate"
    if races >= 40 and roi >= 1.50:
        return "thin_high_roi"
    if races >= 80 and roi >= 1.0:
        return "watch"
    if races < 40 and roi >= 1.2:
        return "too_thin"
    if roi >= 1.1 and top5 < 0.5:
        return "payout_dependent"
    return "reject"


def load_runner_tags(path: Path) -> pd.DataFrame:
    sample = read_csv(path, nrows=1)
    tag_cols = [c for c in sample.columns if c.startswith("tag_")]
    usecols = ["race_id", "horse_no", "frame_no", "horse_fast_lap_score_past5", "fast_clock_aptitude_score", *tag_cols]
    frame = read_csv(path, usecols=lambda c: c in usecols)
    frame["race_id"] = frame["race_id"].astype(str)
    frame["horse_no"] = pd.to_numeric(frame["horse_no"], errors="coerce").astype("Int64")
    return frame


def attach_role_tags(tickets: pd.DataFrame, runner_tags: pd.DataFrame, role: str, no_col: str) -> pd.DataFrame:
    tickets = tickets.copy()
    tickets["race_id"] = tickets["race_id"].astype(str)
    tickets[no_col] = pd.to_numeric(tickets[no_col], errors="coerce").astype("Int64")
    tag_frame = runner_tags.rename(columns={c: pcol(role, c) for c in runner_tags.columns if c not in ["race_id", "horse_no"]})
    merged = tickets.merge(tag_frame, left_on=["race_id", no_col], right_on=["race_id", "horse_no"], how="left")
    return merged.drop(columns=["horse_no"], errors="ignore")


def evaluate(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for role in ["anchor", "partner"]:
        for hypo in hypotheses():
            mask = hypo["mask"](frame, role).fillna(False)
            part = frame[mask]
            rows.append(
                {
                    "source": source,
                    "role": role,
                    "hypothesis": hypo["name"],
                    "label": hypo["label"],
                    **metrics(part),
                }
            )
    for hypo in hypotheses():
        mask = (hypo["mask"](frame, "anchor").fillna(False) | hypo["mask"](frame, "partner").fillna(False))
        part = frame[mask]
        rows.append(
            {
                "source": source,
                "role": "either",
                "hypothesis": hypo["name"],
                "label": hypo["label"],
                **metrics(part),
            }
        )
    out = pd.DataFrame(rows)
    out["judgement"] = out.apply(judgement, axis=1)
    return out


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
    cols = ["source", "role", "hypothesis", "label", "tickets", "races", "roi", "hit_rate", "top5_removed_roi", "top10_removed_roi", "profit_yen", "judgement"]
    usable = result[result["judgement"].isin(["strong_shadow", "shadow_candidate", "thin_high_roi", "watch"])].sort_values(["judgement", "roi"], ascending=[True, False])
    partner = result[(result["role"].eq("partner")) & (result["source"].eq("selected"))].sort_values("roi", ascending=False).head(20)
    lines = [
        "# 最強版チケット 血統×ローテ/条件 仮説検証",
        "",
        "## 入力",
        "",
        f"- enriched: `{summary['enriched_csv']}`",
        f"- selected: `{summary['selected_csv']}`",
        f"- runner_tags: `{summary['runner_tags_csv']}`",
        "",
        "## 採用/監視候補",
        "",
        markdown_table(usable[cols].head(40).to_dict("records")),
        "",
        "## partner側の上位",
        "",
        markdown_table(partner[cols].to_dict("records")),
        "",
        "## 判断",
        "",
        "- 最強版チケットは母数が小さいため、直接BUY条件ではなくシャドー加点/減点候補として扱う。",
        "- partner側で残る仮説は、相手選びの説明変数として価値がある可能性が高い。",
        "- top5_removed_roiが低いものは、少数の高配当に依存しているので本採用しない。",
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
    parser.add_argument("--enriched-csv", default=DEFAULT_ENRICHED)
    parser.add_argument("--selected-csv", default=DEFAULT_SELECTED)
    parser.add_argument("--runner-tags-csv", default=DEFAULT_RUNNER_TAGS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runner_tags = load_runner_tags(Path(args.runner_tags_csv))
    frames = []
    for source, path_str in [("enriched", args.enriched_csv), ("selected", args.selected_csv)]:
        tickets = read_csv(Path(path_str))
        merged = attach_role_tags(tickets, runner_tags, "anchor", "anchor_no")
        merged = attach_role_tags(merged, runner_tags, "partner", "partner_no")
        coverage = {
            "source": source,
            "tickets": int(len(merged)),
            "anchor_tag_coverage": float(merged["anchor_tag_total_ancestor_hits"].notna().mean()) if "anchor_tag_total_ancestor_hits" in merged.columns else 0.0,
            "partner_tag_coverage": float(merged["partner_tag_total_ancestor_hits"].notna().mean()) if "partner_tag_total_ancestor_hits" in merged.columns else 0.0,
        }
        merged.to_csv(out_dir / f"{source}_with_pedigree_rotation_tags.csv", index=False, encoding="utf-8-sig")
        result = evaluate(merged, source)
        result["anchor_tag_coverage"] = coverage["anchor_tag_coverage"]
        result["partner_tag_coverage"] = coverage["partner_tag_coverage"]
        frames.append(result)
    all_result = pd.concat(frames, ignore_index=True)
    all_result.to_csv(out_dir / "hypothesis_summary.csv", index=False, encoding="utf-8-sig")
    catalog = [{"hypothesis": h["name"], "label": h["label"]} for h in hypotheses()]
    pd.DataFrame(catalog).to_csv(out_dir / "hypothesis_catalog.csv", index=False, encoding="utf-8-sig")
    summary = {
        "enriched_csv": args.enriched_csv,
        "selected_csv": args.selected_csv,
        "runner_tags_csv": args.runner_tags_csv,
        "output_dir": str(out_dir),
        "hypotheses": len(catalog),
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "review.md").write_text(render_review(summary, all_result), encoding="utf-8")
    print(json.dumps(json_ready({"ok": True, **summary}), ensure_ascii=False))


if __name__ == "__main__":
    main()
