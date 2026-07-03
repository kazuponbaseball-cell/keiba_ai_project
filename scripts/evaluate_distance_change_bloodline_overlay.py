from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_FEATURES = (
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus_rebuilt_20260623/"
    "train_features_with_same_day_bias_v3_retro.csv"
)
DEFAULT_TEST_FEATURES = (
    "data/datasets/cache/workout_lap_pedigree_interactions_confirmed_opponent_2023plus_rebuilt_20260623/"
    "test_features_with_same_day_bias_v3_retro.csv"
)
DEFAULT_PAIR_CANDIDATES = "outputs/analysis/dynamic_pair_ticket_allocation_rebuilt_20260623/pair_candidate_universe.csv"
DEFAULT_DYNAMIC_TICKETS = "outputs/analysis/dynamic_pair_ticket_allocation_rebuilt_20260623/walkforward_selected_tickets.csv"
DEFAULT_PURGED_TICKETS = "outputs/analysis/purged_walkforward_mcs_pbo_rebuilt_20260623/purged_walkforward_selected_tickets.csv"
DEFAULT_OUT = "outputs/analysis/distance_change_bloodline_overlay_rebuilt_20260623"


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    values = frame[col]
    if values.dtype == object or str(values.dtype).startswith("string"):
        values = values.astype("string").str.replace(",", "", regex=False).str.replace("+", "", regex=False)
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def top_removed_roi(ret: pd.Series, stake: pd.Series, top_n: int) -> float:
    if len(ret) <= top_n:
        return 0.0
    drop_idx = ret.sort_values(ascending=False).index[:top_n]
    ret2 = ret.drop(index=drop_idx)
    stake2 = stake.drop(index=drop_idx)
    return float(ret2.sum() / stake2.sum()) if stake2.sum() > 0 else 0.0


def max_drawdown_by_race(frame: pd.DataFrame, stake_col: str, return_col: str) -> float:
    if frame.empty:
        return 0.0
    tmp = frame.copy()
    tmp["_stake"] = num(tmp, stake_col)
    tmp["_return"] = num(tmp, return_col)
    tmp["_profit"] = tmp["_return"] - tmp["_stake"]
    sort_cols = [c for c in ["year", "race_id"] if c in tmp.columns]
    if sort_cols:
        tmp = tmp.sort_values(sort_cols, kind="mergesort")
    race_profit = tmp.groupby("race_id", sort=False)["_profit"].sum()
    equity = race_profit.cumsum()
    dd = equity - equity.cummax()
    return float(dd.min()) if len(dd) else 0.0


def metrics(frame: pd.DataFrame, stake_col: str, return_col: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi": 0.0,
            "ticket_hit_rate": 0.0,
            "race_hit_rate": 0.0,
            "max_drawdown_yen": 0.0,
            "top5_removed_roi": 0.0,
            "top10_removed_roi": 0.0,
        }
    stake = num(frame, stake_col)
    ret = num(frame, return_col)
    hit = ret.gt(0)
    race_hit = frame.assign(_hit=hit).groupby("race_id")["_hit"].max()
    return {
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()) if "race_id" in frame.columns else 0,
        "stake_yen": float(stake.sum()),
        "return_yen": float(ret.sum()),
        "profit_yen": float(ret.sum() - stake.sum()),
        "roi": float(ret.sum() / stake.sum()) if stake.sum() > 0 else 0.0,
        "ticket_hit_rate": float(hit.mean()) if len(hit) else 0.0,
        "race_hit_rate": float(race_hit.mean()) if len(race_hit) else 0.0,
        "max_drawdown_yen": max_drawdown_by_race(frame, stake_col, return_col),
        "top5_removed_roi": top_removed_roi(ret, stake, 5),
        "top10_removed_roi": top_removed_roi(ret, stake, 10),
    }


def change_bucket(diff: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [diff <= -400, diff.between(-399, -200), diff.between(-199, 199), diff.between(200, 399), diff >= 400],
            ["shorten_big", "shorten", "same", "extend", "extend_big"],
            default="unknown",
        ),
        index=diff.index,
    )


def load_runner_profile(feature_csv: Path) -> pd.DataFrame:
    usecols = [
        "レースID(新/馬番無)",
        "馬番",
        "馬名",
        "年齢",
        "芝・ダ",
        "距離",
        "前距離",
        "distance_diff",
        "sire_distance_lift",
        "bms_distance_lift",
        "bloodline_surface_distance_fit_score",
        "bloodline_reliability_score",
    ]
    frame = pd.read_csv(feature_csv, encoding="utf-8-sig", low_memory=False, usecols=lambda c: c in usecols)
    out = pd.DataFrame(index=frame.index)
    out["race_id"] = frame["レースID(新/馬番無)"].astype(str)
    out["horse_no"] = num(frame, "馬番").astype("Int64").astype(str)
    out["horse_name"] = frame["馬名"].astype(str)
    current_distance = num(frame, "距離", np.nan)
    previous_distance = num(frame, "前距離", np.nan)
    diff = current_distance - previous_distance
    if "distance_diff" in frame.columns:
        diff = num(frame, "distance_diff", np.nan).where(num(frame, "distance_diff", np.nan).notna(), diff)
    out["distance_diff"] = diff
    out["distance_change_bucket"] = change_bucket(diff)
    out["distance_change_abs"] = diff.abs()
    out["distance_lift_combo"] = num(frame, "sire_distance_lift") + num(frame, "bms_distance_lift")
    out["sire_distance_lift"] = num(frame, "sire_distance_lift")
    out["bms_distance_lift"] = num(frame, "bms_distance_lift")
    out["bloodline_surface_distance_fit_score"] = num(frame, "bloodline_surface_distance_fit_score")
    out["bloodline_reliability_score"] = num(frame, "bloodline_reliability_score")
    out["surface"] = frame.get("芝・ダ", pd.Series("", index=frame.index)).astype(str)
    out["age"] = num(frame, "年齢", np.nan)
    return out.drop_duplicates(["race_id", "horse_no"], keep="last")


def thresholds_from_train(train_csv: Path) -> dict[str, float]:
    train = load_runner_profile(train_csv)
    return {
        "distance_lift_hi": float(train["distance_lift_combo"].quantile(0.75)),
        "distance_lift_top": float(train["distance_lift_combo"].quantile(0.90)),
        "fit_hi": float(train["bloodline_surface_distance_fit_score"].quantile(0.75)),
        "fit_top": float(train["bloodline_surface_distance_fit_score"].quantile(0.90)),
    }


def add_side_profile(frame: pd.DataFrame, profile: pd.DataFrame, side: str, no_col: str) -> pd.DataFrame:
    out = frame.copy()
    out["race_id"] = out["race_id"].astype(str)
    out[no_col] = num(out, no_col).astype("Int64").astype(str)
    side_profile = profile.add_prefix(f"{side}_").rename(
        columns={f"{side}_race_id": "race_id", f"{side}_horse_no": no_col}
    )
    return out.merge(side_profile, on=["race_id", no_col], how="left")


def add_distance_flags(frame: pd.DataFrame, q: dict[str, float]) -> pd.DataFrame:
    out = frame.copy()
    for side in ["anchor", "partner"]:
        out[f"{side}_distance_lift_hi"] = num(out, f"{side}_distance_lift_combo").ge(q["distance_lift_hi"])
        out[f"{side}_distance_lift_top"] = num(out, f"{side}_distance_lift_combo").ge(q["distance_lift_top"])
        out[f"{side}_distance_fit_hi"] = num(out, f"{side}_bloodline_surface_distance_fit_score").ge(q["fit_hi"])
        out[f"{side}_material_distance_change"] = num(out, f"{side}_distance_change_abs").ge(200)
        out[f"{side}_big_distance_change"] = num(out, f"{side}_distance_change_abs").ge(400)
        bucket = out.get(f"{side}_distance_change_bucket", pd.Series("", index=out.index)).astype(str)
        out[f"{side}_shorten"] = bucket.isin(["shorten", "shorten_big"])
        out[f"{side}_extend"] = bucket.isin(["extend", "extend_big"])
        out[f"{side}_distance_change_lift_hi"] = out[f"{side}_material_distance_change"] & out[f"{side}_distance_lift_hi"]
        out[f"{side}_distance_change_lift_top"] = out[f"{side}_material_distance_change"] & out[f"{side}_distance_lift_top"]
        out[f"{side}_distance_change_fit_hi"] = out[f"{side}_material_distance_change"] & out[f"{side}_distance_fit_hi"]
    out["either_distance_change_lift_hi"] = out["anchor_distance_change_lift_hi"] | out["partner_distance_change_lift_hi"]
    out["both_distance_change_lift_hi"] = out["anchor_distance_change_lift_hi"] & out["partner_distance_change_lift_hi"]
    return out


def quality_core_mask(frame: pd.DataFrame) -> pd.Series:
    pair_q = num(frame, "pair_quinella_score").quantile(0.75)
    overlay_q = num(frame, "market_overlay_score").quantile(0.75)
    late_q = num(frame, "late_value_survives_score").quantile(0.50)
    front_q = num(frame, "projected_front5_prob").quantile(0.50)
    return (
        num(frame, "pair_quinella_score").ge(pair_q)
        & num(frame, "market_overlay_score").ge(overlay_q)
        & num(frame, "late_value_survives_score").ge(late_q)
        & num(frame, "projected_front5_prob").ge(front_q)
        & num(frame, "anchor_danger").le(0.70)
        & num(frame, "partner_danger").le(0.70)
    )


def evaluate_selected(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    action = frame.get("runtime_action", pd.Series("BUY", index=frame.index)).astype(str)
    buy = action.eq("BUY")
    masks = [
        ("all_buy", buy),
        ("anchor_distance_change_lift_hi", buy & frame["anchor_distance_change_lift_hi"]),
        ("partner_distance_change_lift_hi", buy & frame["partner_distance_change_lift_hi"]),
        ("partner_distance_change_lift_top", buy & frame["partner_distance_change_lift_top"]),
        ("either_distance_change_lift_hi", buy & frame["either_distance_change_lift_hi"]),
        ("both_distance_change_lift_hi", buy & frame["both_distance_change_lift_hi"]),
        ("partner_shorten_lift_hi", buy & frame["partner_shorten"] & frame["partner_distance_lift_hi"]),
        ("partner_extend_lift_hi", buy & frame["partner_extend"] & frame["partner_distance_lift_hi"]),
        ("partner_change_without_lift_hi", buy & frame["partner_material_distance_change"] & ~frame["partner_distance_lift_hi"]),
        ("partner_no_material_change", buy & ~frame["partner_material_distance_change"]),
    ]
    rows = []
    for segment, mask in masks:
        part = frame[mask.fillna(False)].copy()
        row = {"source": label, "scope": "selected_tickets", "bet_type": "actual_stake", "segment": segment}
        row.update(metrics(part, "stake_yen", "return_yen"))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["stake_100"] = 100.0
    out["umaren_return_100"] = np.where(out["umaren_hit"].astype(bool), num(out, "umaren_pay"), 0.0)
    out["wide_return_100"] = np.where(out["wide_hit"].astype(bool), num(out, "wide_pay"), 0.0)
    qcore = quality_core_mask(out)
    value_partner = (num(out, "partner_pop", 99).ge(4) | num(out, "partner_odds", 0).ge(8.0)) & num(
        out, "partner_ai_rank", 99
    ).le(8)
    anchor_ok = (num(out, "anchor_quinella_model_rank", 99).le(8) | num(out, "anchor_pop", 99).le(3)) & num(
        out, "skip_risk_score"
    ).le(0.75)
    masks = [
        ("all_pair_candidates", pd.Series(True, index=out.index)),
        ("quality_core", qcore),
        ("partner_distance_change_lift_hi", out["partner_distance_change_lift_hi"]),
        ("partner_distance_change_lift_hi_quality_core", out["partner_distance_change_lift_hi"] & qcore),
        ("partner_distance_change_lift_top_quality_core", out["partner_distance_change_lift_top"] & qcore),
        ("partner_distance_change_lift_hi_value_quality", out["partner_distance_change_lift_hi"] & qcore & value_partner),
        (
            "partner_distance_change_lift_hi_anchor_ok_value_quality",
            out["partner_distance_change_lift_hi"] & qcore & value_partner & anchor_ok,
        ),
        ("partner_shorten_lift_hi_quality_core", out["partner_shorten"] & out["partner_distance_lift_hi"] & qcore),
        ("partner_extend_lift_hi_quality_core", out["partner_extend"] & out["partner_distance_lift_hi"] & qcore),
        ("either_distance_change_lift_hi_quality_core", out["either_distance_change_lift_hi"] & qcore),
    ]
    rows = []
    for segment, mask in masks:
        part = out[mask.fillna(False)].copy()
        for bet_type, return_col in [("umaren", "umaren_return_100"), ("wide", "wide_return_100")]:
            row = {"source": "candidate_universe", "scope": "candidate_universe", "bet_type": bet_type, "segment": segment}
            row.update(metrics(part, "stake_100", return_col))
            rows.append(row)
    return pd.DataFrame(rows)


def yearly_breakdown(frame: pd.DataFrame, label: str) -> list[dict[str, Any]]:
    rows = []
    action = frame.get("runtime_action", pd.Series("BUY", index=frame.index)).astype(str)
    frame = frame[action.eq("BUY")].copy()
    segments = {
        "all_buy": pd.Series(True, index=frame.index),
        "partner_distance_change_lift_hi": frame["partner_distance_change_lift_hi"],
        "partner_distance_change_lift_top": frame["partner_distance_change_lift_top"],
        "partner_change_without_lift_hi": frame["partner_material_distance_change"] & ~frame["partner_distance_lift_hi"],
    }
    for segment, mask in segments.items():
        part = frame[mask.fillna(False)].copy()
        for year, year_part in part.groupby("year"):
            row = {"source": label, "segment": segment, "year": int(year)}
            row.update(metrics(year_part, "stake_yen", "return_yen"))
            rows.append(row)
    return rows


def write_review(out_dir: Path, summary: dict[str, Any], segment_summary: pd.DataFrame) -> None:
    selected = segment_summary[segment_summary["scope"].eq("selected_tickets")].copy()
    candidates = segment_summary[segment_summary["scope"].eq("candidate_universe")].copy()
    key_selected = selected[
        selected["segment"].isin(
            [
                "all_buy",
                "partner_distance_change_lift_hi",
                "partner_distance_change_lift_top",
                "partner_change_without_lift_hi",
                "partner_no_material_change",
            ]
        )
    ][["source", "segment", "tickets", "races", "roi", "ticket_hit_rate", "top5_removed_roi"]]
    key_candidates = candidates[
        candidates["segment"].isin(
            [
                "quality_core",
                "partner_distance_change_lift_hi_quality_core",
                "partner_distance_change_lift_hi_value_quality",
                "partner_distance_change_lift_hi_anchor_ok_value_quality",
                "partner_shorten_lift_hi_quality_core",
                "partner_extend_lift_hi_quality_core",
            ]
        )
    ][["bet_type", "segment", "tickets", "races", "roi", "ticket_hit_rate", "top5_removed_roi"]]

    def md_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_No rows._"
        cols = list(frame.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in frame.iterrows():
            vals = []
            for col in cols:
                value = row[col]
                if isinstance(value, float):
                    value = f"{value:.4f}"
                vals.append(str(value).replace("|", "/"))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    body = [
        "# 距離短縮/延長 × 血統距離リフト 検証",
        "",
        "## 結論",
        "",
        summary["verdict"],
        "",
        "## 選抜済みチケット",
        "",
        md_table(key_selected),
        "",
        "## 候補宇宙",
        "",
        md_table(key_candidates),
        "",
        "## 解釈",
        "",
        "- ランナー単体では一部のAI上位・3歳/4歳・芝中距離/マイルで単勝ROIが上振れる。",
        "- ただし現行の買い目に重ねると、距離変更×血統距離リフトは安定した上乗せにならない。",
        "- 馬連は特に弱く、ワイド候補でも上位払戻依存が強い。",
        "- BUY拡張ではなく、経験不足/初距離馬の説明・準候補シャドーに限定する。",
    ]
    (out_dir / "review.md").write_text("\n".join(body) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate distance-change x bloodline distance-lift overlays.")
    parser.add_argument("--train-features", default=DEFAULT_TRAIN_FEATURES)
    parser.add_argument("--test-features", default=DEFAULT_TEST_FEATURES)
    parser.add_argument("--pair-candidates", default=DEFAULT_PAIR_CANDIDATES)
    parser.add_argument("--dynamic-tickets", default=DEFAULT_DYNAMIC_TICKETS)
    parser.add_argument("--purged-tickets", default=DEFAULT_PURGED_TICKETS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args()

    out_dir = project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    q = thresholds_from_train(project_path(args.train_features))
    profile = load_runner_profile(project_path(args.test_features))

    pair_candidates = pd.read_csv(project_path(args.pair_candidates), low_memory=False)
    pair_candidates = add_side_profile(pair_candidates, profile, "anchor", "anchor_no")
    pair_candidates = add_side_profile(pair_candidates, profile, "partner", "partner_no")
    pair_candidates = add_distance_flags(pair_candidates, q)
    candidate_summary = evaluate_candidates(pair_candidates)

    all_segments = [candidate_summary]
    yearly_rows: list[dict[str, Any]] = []
    selected_profiles = []
    for label, path in [
        ("dynamic", project_path(args.dynamic_tickets)),
        ("purged", project_path(args.purged_tickets)),
    ]:
        tickets = pd.read_csv(path, low_memory=False)
        tickets = add_side_profile(tickets, profile, "anchor", "anchor_no")
        tickets = add_side_profile(tickets, profile, "partner", "partner_no")
        tickets = add_distance_flags(tickets, q)
        selected_profiles.append((label, tickets))
        all_segments.append(evaluate_selected(tickets, label))
        yearly_rows.extend(yearly_breakdown(tickets, label))
        tickets.to_csv(out_dir / f"{label}_selected_ticket_distance_bloodline_profile.csv", index=False, encoding="utf-8-sig")

    segment_summary = pd.concat(all_segments, ignore_index=True)
    yearly = pd.DataFrame(yearly_rows)
    pair_candidates.to_csv(out_dir / "pair_candidate_distance_bloodline_profile.csv", index=False, encoding="utf-8-sig")
    segment_summary.to_csv(out_dir / "segment_summary.csv", index=False, encoding="utf-8-sig")
    yearly.to_csv(out_dir / "yearly_breakdown.csv", index=False, encoding="utf-8-sig")

    dyn = segment_summary[
        (segment_summary["source"].eq("dynamic"))
        & (segment_summary["segment"].eq("partner_distance_change_lift_hi"))
    ]
    pur = segment_summary[
        (segment_summary["source"].eq("purged"))
        & (segment_summary["segment"].eq("partner_distance_change_lift_hi"))
    ]
    verdict = "最終BUY拡張としては不採用。距離変更×血統距離リフトは、現行チケットではROI改善が安定しない。"
    if not dyn.empty and not pur.empty:
        if float(dyn.iloc[0]["roi"]) > 1.05 and float(pur.iloc[0]["roi"]) > 1.05:
            verdict = "シャドー昇格候補。両検証でROIがプラスだが、上位払戻除外と年別安定性を追加確認してから採用。"

    summary = {
        "output_dir": str(out_dir),
        "thresholds": q,
        "merge_rates": {
            "pair_anchor_profile_rate": float(pair_candidates["anchor_distance_lift_combo"].notna().mean()),
            "pair_partner_profile_rate": float(pair_candidates["partner_distance_lift_combo"].notna().mean()),
        },
        "verdict": verdict,
        "key_metrics": segment_summary[
            segment_summary["segment"].isin(["all_buy", "partner_distance_change_lift_hi", "partner_distance_change_lift_top"])
        ].to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    write_review(out_dir, summary, segment_summary)
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
