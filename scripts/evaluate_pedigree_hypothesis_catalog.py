from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


DEFAULT_DYNAMIC = "outputs/analysis/deep_pedigree_fine_context_overlay_v1/dynamic_selected_with_pedigree_tags.csv"
DEFAULT_PURGED = "outputs/analysis/deep_pedigree_fine_context_overlay_v1/purged_selected_with_pedigree_tags.csv"
DEFAULT_RUNNER = "outputs/analysis/deep_pedigree_fine_context_overlay_v1/runner_pedigree_tag_profile.csv"
DEFAULT_OUT = "outputs/analysis/pedigree_hypothesis_catalog_v1"


MaskFunc = Callable[[pd.DataFrame, str], pd.Series]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def bool_col(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    s = frame[col]
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def prefixed(prefix: str, col: str) -> str:
    return f"{prefix}_{col}" if prefix else col


def tag_count(frame: pd.DataFrame, prefix: str, tag: str, suffix: str = "count") -> pd.Series:
    return num(frame, prefixed(prefix, f"tag_{tag}_{suffix}"), 0.0)


def tag_flag(frame: pd.DataFrame, prefix: str, tag: str) -> pd.Series:
    count_col = prefixed(prefix, f"tag_{tag}_count")
    flag_col = prefixed(prefix, f"tag_{tag}")
    if count_col in frame.columns:
        return tag_count(frame, prefix, tag).gt(0)
    return bool_col(frame, flag_col)


def side_count(frame: pd.DataFrame, prefix: str, tag: str, side: str) -> pd.Series:
    return tag_count(frame, prefix, tag, f"{side}_side_count")


def cross_flag(frame: pd.DataFrame, prefix: str, tag: str) -> pd.Series:
    col = prefixed(prefix, f"tag_{tag}_cross_flag")
    if col in frame.columns:
        return bool_col(frame, col)
    return tag_count(frame, prefix, tag).ge(2)


def surface(frame: pd.DataFrame, value: str) -> pd.Series:
    return frame.get("surface", pd.Series("", index=frame.index)).astype(str).eq(value)


def bad_going(frame: pd.DataFrame) -> pd.Series:
    return frame.get("going", pd.Series("", index=frame.index)).astype(str).isin(["稍", "重", "不", "稍重", "不良"])


def local(frame: pd.DataFrame) -> pd.Series:
    return frame.get("venue", pd.Series("", index=frame.index)).astype(str).isin(["札幌", "函館", "福島", "新潟", "小倉"])


def small_turn(frame: pd.DataFrame) -> pd.Series:
    return frame.get("venue", pd.Series("", index=frame.index)).astype(str).isin(["札幌", "函館", "福島", "中山", "小倉"])


def local_small_turn(frame: pd.DataFrame) -> pd.Series:
    return frame.get("venue", pd.Series("", index=frame.index)).astype(str).isin(["札幌", "函館", "福島", "小倉"])


def distance(frame: pd.DataFrame) -> pd.Series:
    return num(frame, "distance", np.nan)


def frame_no(frame: pd.DataFrame, prefix: str) -> pd.Series:
    col = prefixed(prefix, "frame_no")
    if col in frame.columns:
        return num(frame, col, np.nan)
    return num(frame, "frame_no", np.nan)


def fast_lap(frame: pd.DataFrame, prefix: str) -> pd.Series:
    col = prefixed(prefix, "horse_fast_lap_score_past5")
    if col in frame.columns:
        return num(frame, col, 0.0)
    return num(frame, "horse_fast_lap_score_past5", 0.0)


def fast_clock(frame: pd.DataFrame, prefix: str) -> pd.Series:
    col = prefixed(prefix, "fast_clock_aptitude_score")
    if col in frame.columns:
        return num(frame, col, 0.0)
    return num(frame, "fast_clock_aptitude_score", 0.0)


def race_class(frame: pd.DataFrame) -> pd.Series:
    return frame.get("race_class", pd.Series("", index=frame.index)).astype(str)


def us_speed_power(frame: pd.DataFrame, prefix: str) -> pd.Series:
    return (
        tag_flag(frame, prefix, "storm_cat")
        | tag_flag(frame, prefix, "mr_prospector")
        | tag_flag(frame, prefix, "danzig")
        | tag_flag(frame, prefix, "deputy_minister")
    )


def euro_stamina(frame: pd.DataFrame, prefix: str) -> pd.Series:
    return (
        tag_flag(frame, prefix, "sadlers_wells")
        | tag_flag(frame, prefix, "nureyev")
        | tag_flag(frame, prefix, "blushing_groom")
        | tag_flag(frame, prefix, "roberto")
    )


def halo_family(frame: pd.DataFrame, prefix: str) -> pd.Series:
    return tag_flag(frame, prefix, "halo") | tag_flag(frame, prefix, "hail_to_reason")


def metrics(frame: pd.DataFrame, stake_col: str, return_col: str) -> dict[str, Any]:
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
    hit = ret.gt(0)
    top_removed = {}
    for n in (5, 10):
        if len(frame) <= n:
            top_removed[f"top{n}_removed_roi"] = 0.0
            continue
        drop = ret.sort_values(ascending=False).index[:n]
        stake2 = stake.drop(index=drop)
        ret2 = ret.drop(index=drop)
        top_removed[f"top{n}_removed_roi"] = float(ret2.sum() / stake2.sum()) if stake2.sum() > 0 else 0.0
    return {
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()) if "race_id" in frame.columns else 0,
        "stake_yen": float(stake.sum()),
        "return_yen": float(ret.sum()),
        "profit_yen": float(ret.sum() - stake.sum()),
        "roi": float(ret.sum() / stake.sum()) if stake.sum() > 0 else 0.0,
        "hit_rate": float(hit.mean()) if len(hit) else 0.0,
        **top_removed,
    }


def hypothesis_catalog() -> list[dict[str, Any]]:
    return [
        {
            "name": "us_speed_power_dirt_sprint",
            "label": "米国スピード/パワー血統 × ダート短距離",
            "rationale": "Storm Cat / Mr. Prospector / Danzig / Deputy Minister 系はダート短距離の速力・パワー条件で拾う仮説。",
            "mask": lambda f, p: us_speed_power(f, p) & surface(f, "ダ") & distance(f).le(1400),
        },
        {
            "name": "us_speed_power_outer_dirt_sprint",
            "label": "米国スピード/パワー血統 × 外枠ダート短距離",
            "rationale": "外から揉まれず先行しやすい条件で米国型スピードを拾う仮説。",
            "mask": lambda f, p: us_speed_power(f, p) & surface(f, "ダ") & distance(f).le(1400) & frame_no(f, p).ge(7),
        },
        {
            "name": "storm_cat_outer_small_turn",
            "label": "Storm Cat系 × 外枠小回り",
            "rationale": "スピードを外から押し切る小回り条件を拾う仮説。",
            "mask": lambda f, p: tag_flag(f, p, "storm_cat") & small_turn(f) & frame_no(f, p).ge(7),
        },
        {
            "name": "storm_cat_local_dirt",
            "label": "Storm Cat系 × ローカルダート",
            "rationale": "ローカルの前受け/機動力寄りダートでStorm Cat的スピードを拾う仮説。",
            "mask": lambda f, p: tag_flag(f, p, "storm_cat") & local(f) & surface(f, "ダ"),
        },
        {
            "name": "mrp_dam_side_dirt",
            "label": "母系Mr. Prospector系 × ダート",
            "rationale": "母系に入る米国型パワー/スピードをダート適性補完として拾う仮説。",
            "mask": lambda f, p: side_count(f, p, "mr_prospector", "dam").gt(0) & surface(f, "ダ"),
        },
        {
            "name": "mrp_fast_lap",
            "label": "Mr. Prospector系 × 速いラップ適性",
            "rationale": "米国型スピード血統を速いラップへの実走対応で絞る仮説。",
            "mask": lambda f, p: tag_flag(f, p, "mr_prospector") & fast_lap(f, p).ge(0.55),
        },
        {
            "name": "deputy_minister_dirt",
            "label": "Deputy Minister系 × ダート",
            "rationale": "Deputy Minister/French Deputy/クロフネ系のダートパワー仮説。",
            "mask": lambda f, p: tag_flag(f, p, "deputy_minister") & surface(f, "ダ"),
        },
        {
            "name": "deputy_minister_bad_dirt",
            "label": "Deputy Minister系 × 道悪ダート",
            "rationale": "パワー型が脚抜きや湿ったダートで浮上するかを確認する仮説。",
            "mask": lambda f, p: tag_flag(f, p, "deputy_minister") & surface(f, "ダ") & bad_going(f),
        },
        {
            "name": "danzig_turf_sprint",
            "label": "Danzig系 × 芝短距離",
            "rationale": "Danzig/Danehill/Green Desert系の芝スピード仮説。",
            "mask": lambda f, p: tag_flag(f, p, "danzig") & surface(f, "芝") & distance(f).le(1400),
        },
        {
            "name": "danzig_fast_lap",
            "label": "Danzig系 × 速いラップ適性",
            "rationale": "スピード血統を実走の速いラップ適性で絞る仮説。",
            "mask": lambda f, p: tag_flag(f, p, "danzig") & fast_lap(f, p).ge(0.55),
        },
        {
            "name": "danzig_maiden_new",
            "label": "Danzig系 × 新馬/未勝利",
            "rationale": "若駒・未勝利でスピードの完成度が市場評価より効くかを見る仮説。",
            "mask": lambda f, p: tag_flag(f, p, "danzig") & race_class(f).str.contains("新馬|未勝", regex=True, na=False),
        },
        {
            "name": "kingmambo_turf_fast_lap",
            "label": "Kingmambo系 × 芝 × 速いラップ適性",
            "rationale": "Kingmambo/キンカメ系を芝の速い流れ・高速持続で拾う仮説。",
            "mask": lambda f, p: tag_flag(f, p, "kingmambo") & surface(f, "芝") & fast_lap(f, p).ge(0.55),
        },
        {
            "name": "kingmambo_fast_lap",
            "label": "Kingmambo系 × 速いラップ適性",
            "rationale": "前回検証で最も筋が良かった血統×ラップ適性仮説。",
            "mask": lambda f, p: tag_flag(f, p, "kingmambo") & fast_lap(f, p).ge(0.55),
        },
        {
            "name": "kingmambo_bad_going",
            "label": "Kingmambo系 × 道悪",
            "rationale": "Kingmambo/キンカメ系のパワー・持続性が道悪で効くかを見る仮説。",
            "mask": lambda f, p: tag_flag(f, p, "kingmambo") & bad_going(f),
        },
        {
            "name": "roberto_bad_going",
            "label": "Roberto系 × 道悪",
            "rationale": "Roberto系は道悪/タフ条件に強いという一般論を検証する仮説。",
            "mask": lambda f, p: tag_flag(f, p, "roberto") & bad_going(f),
        },
        {
            "name": "roberto_small_turn_dirt",
            "label": "Roberto系 × 小回りダート",
            "rationale": "パワー・持続力を小回りダートの機動力条件で拾う仮説。",
            "mask": lambda f, p: tag_flag(f, p, "roberto") & small_turn(f) & surface(f, "ダ"),
        },
        {
            "name": "roberto_midlong",
            "label": "Roberto系 × 中長距離",
            "rationale": "Roberto系の持続力/スタミナが距離延長寄りで効くかを見る仮説。",
            "mask": lambda f, p: tag_flag(f, p, "roberto") & distance(f).ge(1800),
        },
        {
            "name": "sadlers_wells_bad_turf",
            "label": "Sadler's Wells系 × 道悪芝",
            "rationale": "欧州型スタミナ/タフさが道悪芝で効くかを見る仮説。",
            "mask": lambda f, p: tag_flag(f, p, "sadlers_wells") & surface(f, "芝") & bad_going(f),
        },
        {
            "name": "sadlers_wells_midlong_turf",
            "label": "Sadler's Wells系 × 芝中長距離",
            "rationale": "欧州型スタミナが芝1800m以上で効くかを見る仮説。",
            "mask": lambda f, p: tag_flag(f, p, "sadlers_wells") & surface(f, "芝") & distance(f).ge(1800),
        },
        {
            "name": "nureyev_bad_or_smallturn",
            "label": "Nureyev系 × 道悪/小回り",
            "rationale": "Nureyev/Special的な機動力・持続力をタフ/小回り条件で拾う仮説。",
            "mask": lambda f, p: tag_flag(f, p, "nureyev") & (bad_going(f) | small_turn(f)),
        },
        {
            "name": "northern_dancer_cross_bad",
            "label": "Northern Dancerクロス × 道悪",
            "rationale": "Northern Dancer濃度がタフ条件で効くかを見る仮説。",
            "mask": lambda f, p: cross_flag(f, p, "northern_dancer") & bad_going(f),
        },
        {
            "name": "northern_dancer_cross_turf",
            "label": "Northern Dancerクロス × 芝",
            "rationale": "芝の総合力/持続力としてNorthern Dancerクロスを検証する仮説。",
            "mask": lambda f, p: cross_flag(f, p, "northern_dancer") & surface(f, "芝"),
        },
        {
            "name": "halo_fast_clock_turf",
            "label": "Halo/Sunday系 × 高速芝",
            "rationale": "日本芝の速い時計への対応力としてHalo/Sunday系を絞る仮説。",
            "mask": lambda f, p: halo_family(f, p) & surface(f, "芝") & fast_clock(f, p).ge(0.62),
        },
        {
            "name": "halo_inner_draw",
            "label": "Halo/Sunday系 × 内枠",
            "rationale": "器用さ/瞬発力を内枠でロスなく使えるかを見る仮説。",
            "mask": lambda f, p: halo_family(f, p) & frame_no(f, p).le(2),
        },
        {
            "name": "blushing_groom_bad_going",
            "label": "Blushing Groom系 × 道悪",
            "rationale": "欧州/持続力色のある血を道悪で拾う仮説。",
            "mask": lambda f, p: tag_flag(f, p, "blushing_groom") & bad_going(f),
        },
        {
            "name": "euro_tough_bad_going",
            "label": "欧州/タフ血統 × 道悪",
            "rationale": "Roberto/Sadler's Wells/Nureyev/Blushing Groomをまとめ、道悪で効くかを見る仮説。",
            "mask": lambda f, p: euro_stamina(f, p) & bad_going(f),
        },
        {
            "name": "euro_stamina_midlong_turf",
            "label": "欧州/スタミナ血統 × 芝中長距離",
            "rationale": "欧州型持続力を芝1800m以上で拾う仮説。",
            "mask": lambda f, p: euro_stamina(f, p) & surface(f, "芝") & distance(f).ge(1800),
        },
        {
            "name": "us_power_local_dirt",
            "label": "米国パワー血統 × ローカルダート",
            "rationale": "ローカルダートの前受け・パワー・機動力を米国型で拾う仮説。",
            "mask": lambda f, p: us_speed_power(f, p) & local(f) & surface(f, "ダ"),
        },
        {
            "name": "local_smallturn_power",
            "label": "パワー血統 × ローカル小回り",
            "rationale": "小回りローカルで機動力/パワー型血統が相手として残るかを見る仮説。",
            "mask": lambda f, p: (us_speed_power(f, p) | tag_flag(f, p, "roberto")) & local_small_turn(f),
        },
    ]


def evaluate_tickets(frame: pd.DataFrame, source: str, prefix: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for hypo in hypothesis_catalog():
        mask = hypo["mask"](frame, prefix).fillna(False)
        part = frame[mask]
        rows.append({"source": source, "hypothesis": hypo["name"], "label": hypo["label"], "rationale": hypo["rationale"], **metrics(part, "stake_yen", "return_yen")})
        for year, year_part in part.groupby("year", sort=True):
            rows.append(
                {
                    "source": f"{source}_by_year",
                    "hypothesis": hypo["name"],
                    "label": hypo["label"],
                    "rationale": hypo["rationale"],
                    "year": int(year),
                    **metrics(year_part, "stake_yen", "return_yen"),
                }
            )
    return pd.DataFrame(rows)


def evaluate_runners(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["stake_100"] = 100.0
    out["win_return_100"] = np.where(num(out, "finish", 99).eq(1), num(out, "win_pay_100", 0.0), 0.0)
    out["place_return_100"] = np.where(num(out, "finish", 99).le(3), num(out, "place_pay_100", 0.0), 0.0)
    rows: list[dict[str, Any]] = []
    ai_top5 = num(out, "ai_rank", 99).le(5)
    for hypo in hypothesis_catalog():
        mask = hypo["mask"](out, "").fillna(False) & ai_top5
        part = out[mask]
        for bet, ret_col in [("win", "win_return_100"), ("place", "place_return_100")]:
            rows.append(
                {
                    "source": "runner_ai_top5",
                    "hypothesis": hypo["name"],
                    "label": hypo["label"],
                    "rationale": hypo["rationale"],
                    "bet_type": bet,
                    **metrics(part, "stake_100", ret_col),
                }
            )
    return pd.DataFrame(rows)


def judgement(row: pd.Series) -> str:
    races = float(row.get("races", 0) or 0)
    roi = float(row.get("roi", 0) or 0)
    top5 = float(row.get("top5_removed_roi", 0) or 0)
    top10 = float(row.get("top10_removed_roi", 0) or 0)
    if races >= 250 and roi >= 1.10 and top5 >= 0.80:
        return "shadow_plus"
    if races >= 120 and roi >= 1.15 and top5 >= 0.75:
        return "watch_plus"
    if races >= 80 and roi >= 1.25:
        return "high_roi_thin"
    if races >= 100 and roi >= 1.00:
        return "watch"
    if races < 80 and roi >= 1.20:
        return "too_thin"
    if top10 < 0.50 and roi >= 1.10:
        return "payout_dependent"
    return "reject"


def markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}" if math.isfinite(value) else ""
            vals.append(str(value).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def render_review(summary: dict[str, Any], ticket_summary: pd.DataFrame, runner_summary: pd.DataFrame) -> str:
    main = ticket_summary[~ticket_summary["source"].str.endswith("_by_year")].copy()
    main["judgement"] = main.apply(judgement, axis=1)
    cols = ["source", "hypothesis", "label", "tickets", "races", "roi", "hit_rate", "top5_removed_roi", "top10_removed_roi", "profit_yen", "judgement"]
    best = main[main["judgement"].isin(["shadow_plus", "watch_plus", "high_roi_thin", "watch"])].sort_values(["judgement", "roi"], ascending=[True, False]).head(30)
    stable = main[(main["races"].ge(200)) & (main["roi"].ge(1.05)) & (main["top5_removed_roi"].ge(0.80))].sort_values("roi", ascending=False).head(20)
    runner_best = runner_summary[(runner_summary["races"].ge(100)) & (runner_summary["roi"].ge(1.0))].sort_values("roi", ascending=False).head(20)
    runner_cols = ["hypothesis", "label", "bet_type", "tickets", "races", "roi", "hit_rate", "top5_removed_roi", "profit_yen"]
    lines = [
        "# 血統×条件 仮説カタログ検証",
        "",
        "## 目的",
        "",
        "一般的に語られやすい血統適性を、固定観念ではなく仮説カタログとして特徴化し、既存BUY/厳格purged/単馬AI上位に重ねてROIを確認した。",
        "",
        "## 入力",
        "",
        f"- dynamic: `{summary['dynamic_csv']}`",
        f"- purged: `{summary['purged_csv']}`",
        f"- runner: `{summary['runner_csv']}`",
        "",
        "## 採用候補",
        "",
        markdown_table(best[cols].to_dict("records")),
        "",
        "## 比較的安定して見えるもの",
        "",
        markdown_table(stable[cols].to_dict("records")),
        "",
        "## 単馬AI上位での参考",
        "",
        markdown_table(runner_best[runner_cols].to_dict("records")),
        "",
        "## 判断",
        "",
        "- 直接BUY条件にするにはまだ早い。",
        "- `shadow_plus` はシャドー加点または理由表示に使える候補。",
        "- `high_roi_thin` は高回収に見えても母数/上位払戻依存が残るので、週次シャドー蓄積が必要。",
        "- 血統は単独主因より、ラップ適性・枠・小回り・馬場と合わせた補助レイヤーが最も筋が良い。",
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
    parser.add_argument("--dynamic-csv", default=DEFAULT_DYNAMIC)
    parser.add_argument("--purged-csv", default=DEFAULT_PURGED)
    parser.add_argument("--runner-csv", default=DEFAULT_RUNNER)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dynamic_path = Path(args.dynamic_csv)
    purged_path = Path(args.purged_csv)
    runner_path = Path(args.runner_csv)
    dynamic = read_csv(dynamic_path)
    purged = read_csv(purged_path)
    runner = read_csv(runner_path)

    ticket_summary = pd.concat(
        [
            evaluate_tickets(dynamic, "dynamic_selected_tickets", "partner"),
            evaluate_tickets(purged, "purged_selected_tickets", "partner"),
        ],
        ignore_index=True,
    )
    ticket_summary["judgement"] = ticket_summary.apply(judgement, axis=1)
    runner_summary = evaluate_runners(runner)
    runner_summary["judgement"] = runner_summary.apply(judgement, axis=1)

    ticket_summary.to_csv(out_dir / "hypothesis_ticket_summary.csv", index=False, encoding="utf-8-sig")
    runner_summary.to_csv(out_dir / "hypothesis_runner_summary.csv", index=False, encoding="utf-8-sig")
    catalog_rows = [
        {"hypothesis": h["name"], "label": h["label"], "rationale": h["rationale"]}
        for h in hypothesis_catalog()
    ]
    pd.DataFrame(catalog_rows).to_csv(out_dir / "hypothesis_catalog.csv", index=False, encoding="utf-8-sig")
    summary = {
        "dynamic_csv": str(dynamic_path),
        "purged_csv": str(purged_path),
        "runner_csv": str(runner_path),
        "hypotheses": len(catalog_rows),
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "review.md").write_text(render_review(summary, ticket_summary, runner_summary), encoding="utf-8")
    print(json.dumps(json_ready({"ok": True, **summary}), ensure_ascii=False))


if __name__ == "__main__":
    main()
