from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONTENT_DIR = ROOT / "outputs/analysis/content_bridge_member_features_v1"
DEFAULT_TICKETS = ROOT / "outputs/analysis/optimized_stack_ab_s_gate_v1/s_priority_selected_tickets.csv"
DEFAULT_OUT = ROOT / "outputs/analysis/lap_pair_refinement_candidates_v1"

LAP_MODES = ["fast", "slow", "instant", "sustain", "long_spurt"]


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def available_usecols(path: Path, wanted: list[str]) -> list[str]:
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    return [c for c in wanted if c in header.columns]


def num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def roc_auc(y_true: pd.Series, score: pd.Series) -> float:
    y = pd.to_numeric(y_true, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    mask = y.notna() & s.notna()
    y = y.loc[mask].astype(int)
    s = s.loc[mask]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = s.rank(method="average")
    pos_rank_sum = float(ranks.loc[y == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def normalize_rows(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    total = values.sum(axis=1).replace(0.0, np.nan)
    out = values.div(total, axis=0).fillna(1.0 / len(values.columns))
    return out.clip(0.0, 1.0)


def pct_rank_in_race(df: pd.DataFrame, col: str, higher_is_better: bool = True) -> pd.Series:
    values = num(df.get(col), df.index)
    counts = values.notna().groupby(df["race_id"]).transform("sum")
    ranks = values.groupby(df["race_id"]).rank(ascending=not higher_is_better, method="average")
    return ((counts - ranks) / (counts - 1)).replace([np.inf, -np.inf], np.nan).fillna(0.5).clip(0.0, 1.0)


def load_runner_features() -> pd.DataFrame:
    wanted = [
        "レースID(新/馬番無)",
        "日付S",
        "日付",
        "場所",
        "Ｒ",
        "馬番",
        "馬名",
        "人気",
        "単勝オッズ",
        "単勝配当",
        "複勝配当",
        "確定着順",
        "出走頭数",
        "頭数",
        "芝・ダ",
        "距離",
        "クラス名",
        "馬場状態",
        "expected_pace",
        "slow_ai_score",
        "middle_ai_score",
        "fast_ai_score",
        "race_front_runner_count",
        "race_early_pressure_score",
        "race_pace_collapse_risk",
        "race_slow_pace_risk",
        "front_advantage_score",
        "closer_advantage_score",
        "positioning_advantage_score",
        "pace_fit_score",
        "draw_pace_fit_score",
        "prev_lap_pace_index",
        "prev_lap_finish_index",
        "prev_lap_sustain_index",
        "prev_lap_long_spurt_index",
        "horse_fast_lap_count_past5",
        "horse_fast_lap_score_past5",
        "horse_slow_lap_count_past5",
        "horse_slow_lap_score_past5",
        "horse_instant_lap_count_past5",
        "horse_instant_lap_score_past5",
        "horse_sustain_lap_count_past5",
        "horse_sustain_lap_score_past5",
        "horse_long_spurt_lap_count_past5",
        "horse_long_spurt_lap_score_past5",
        "lap_pace_versatility_score",
        "lap_aptitude_fit_score",
        "lap_aptitude_reliability_score",
        "前PCI",
        "前走PCI3",
        "前走RPCI",
        "前走Ave-3F",
        "前走平均1Fタイム",
        "PCI",
        "PCI3",
        "RPCI",
        "Ave-3F",
    ]
    frames: list[pd.DataFrame] = []
    for source in ["train", "test"]:
        path = CONTENT_DIR / f"{source}_features_with_content_bridge.csv"
        usecols = available_usecols(path, wanted)
        df = read_csv(path, dtype=str, usecols=usecols, low_memory=False)
        df["source"] = source
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(columns={"レースID(新/馬番無)": "race_id"})
    out["race_id"] = out["race_id"].astype(str)
    out["horse_no"] = num(out.get("馬番"), out.index)
    out["race_no"] = num(out.get("Ｒ"), out.index)
    out["popularity"] = num(out.get("人気"), out.index)
    out["odds"] = num(out.get("単勝オッズ"), out.index)
    out["finish"] = num(out.get("確定着順"), out.index)
    out["field_size"] = num(out.get("出走頭数"), out.index).fillna(num(out.get("頭数"), out.index))
    out["is_win"] = out["finish"].eq(1)
    out["is_top3"] = out["finish"].le(3)
    out["win_return_100"] = np.where(out["finish"].eq(1), num(out.get("単勝配当"), out.index, 0.0).fillna(0.0), 0.0)
    out["place_return_100"] = np.where(out["finish"].le(3), num(out.get("複勝配当"), out.index, 0.0).fillna(0.0), 0.0)
    return out


def add_lap_pair_refinement_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    expected = out.get("expected_pace", pd.Series("", index=idx)).fillna("").astype(str).str.lower()
    expected_fast = expected.str.contains("fast|ハイ", regex=True).astype(float)
    expected_slow = expected.str.contains("slow|スロー", regex=True).astype(float)
    expected_middle = expected.str.contains("middle|mid|ミドル", regex=True).astype(float)

    pressure = num(out.get("race_early_pressure_score"), idx).fillna(0.5).clip(0.0, 1.0)
    collapse = num(out.get("race_pace_collapse_risk"), idx).fillna(0.5).clip(0.0, 1.0)
    slow_risk = num(out.get("race_slow_pace_risk"), idx).fillna(0.5).clip(0.0, 1.0)
    front_adv = num(out.get("front_advantage_score"), idx).fillna(0.5).clip(0.0, 1.0)
    closer_adv = num(out.get("closer_advantage_score"), idx).fillna(0.5).clip(0.0, 1.0)
    pos_adv = num(out.get("positioning_advantage_score"), idx).fillna(0.5).clip(0.0, 1.0)

    race_profile_raw = pd.DataFrame(
        {
            "race_need_fast": 0.38 * pressure + 0.30 * collapse + 0.20 * expected_fast + 0.12 * (1.0 - slow_risk),
            "race_need_slow": 0.42 * slow_risk + 0.20 * expected_slow + 0.18 * (1.0 - pressure) + 0.20 * front_adv,
            "race_need_instant": 0.46 * slow_risk + 0.20 * expected_slow + 0.18 * closer_adv + 0.16 * (1.0 - pressure),
            "race_need_sustain": 0.34 * pressure + 0.26 * collapse + 0.18 * expected_middle + 0.12 * pos_adv + 0.10 * (1.0 - slow_risk),
            "race_need_long_spurt": 0.24 * pressure + 0.22 * collapse + 0.18 * closer_adv + 0.18 * pos_adv + 0.18 * expected_middle,
        },
        index=idx,
    )
    race_profile = normalize_rows(race_profile_raw)
    out = pd.concat([out, race_profile], axis=1)
    mode_cols = [f"race_need_{m}" for m in LAP_MODES]
    out["predicted_lap_mode"] = race_profile[mode_cols].idxmax(axis=1).str.replace("race_need_", "", regex=False)
    out["race_lap_profile_concentration"] = race_profile[mode_cols].max(axis=1)
    out["race_lap_prediction_confidence"] = (
        0.58 * ((out["race_lap_profile_concentration"] - 0.20) / 0.80).clip(0.0, 1.0)
        + 0.22 * (pressure - slow_risk).abs().clip(0.0, 1.0)
        + 0.20 * (1.0 - (collapse - slow_risk).abs().rsub(1.0).clip(0.0, 1.0) * 0.35)
    ).clip(0.0, 1.0)

    horse_profile_raw = pd.DataFrame(
        {
            "horse_lap_fast": num(out.get("horse_fast_lap_score_past5"), idx).fillna(0.0).clip(0.0, 1.0),
            "horse_lap_slow": num(out.get("horse_slow_lap_score_past5"), idx).fillna(0.0).clip(0.0, 1.0),
            "horse_lap_instant": num(out.get("horse_instant_lap_score_past5"), idx).fillna(0.0).clip(0.0, 1.0),
            "horse_lap_sustain": num(out.get("horse_sustain_lap_score_past5"), idx).fillna(0.0).clip(0.0, 1.0),
            "horse_lap_long_spurt": num(out.get("horse_long_spurt_lap_score_past5"), idx).fillna(0.0).clip(0.0, 1.0),
        },
        index=idx,
    )
    horse_profile = normalize_rows(horse_profile_raw)
    out = pd.concat([out, horse_profile], axis=1)
    horse_cols = [f"horse_lap_{m}" for m in LAP_MODES]
    out["horse_lap_profile_top_mode"] = horse_profile[horse_cols].idxmax(axis=1).str.replace("horse_lap_", "", regex=False)
    out["horse_lap_profile_sharpness"] = (
        horse_profile[horse_cols].max(axis=1) - horse_profile[horse_cols].mean(axis=1)
    ).clip(0.0, 1.0)

    fit_terms = []
    for mode in LAP_MODES:
        fit_terms.append(race_profile[f"race_need_{mode}"] * horse_profile_raw[f"horse_lap_{mode}"])
    out["lap_profile_fit_score"] = pd.concat(fit_terms, axis=1).sum(axis=1).clip(0.0, 1.0)
    out["lap_profile_fit_rank_in_race"] = pct_rank_in_race(out, "lap_profile_fit_score", True)
    reliability = num(out.get("lap_aptitude_reliability_score"), idx).fillna(0.5).clip(0.0, 1.0)
    out["lap_fit_confident_score"] = (
        out["lap_profile_fit_score"] * (0.55 + 0.45 * reliability) * (0.60 + 0.40 * out["race_lap_prediction_confidence"])
    ).clip(0.0, 1.0)
    versatility = num(out.get("lap_pace_versatility_score"), idx).fillna(0.5).clip(0.0, 1.0)
    out["lap_axis_candidate_score"] = (
        0.48 * out["lap_fit_confident_score"]
        + 0.28 * versatility
        + 0.24 * out["lap_profile_fit_rank_in_race"]
    ).clip(0.0, 1.0)
    out["lap_partner_specialist_score"] = (
        out["lap_fit_confident_score"]
        * (0.55 + 0.45 * out["horse_lap_profile_sharpness"])
        * (0.65 + 0.35 * (1.0 - out["popularity"].fillna(9).sub(1).clip(0, 8) / 8))
    ).clip(0.0, 1.0)
    out["lap_mismatch_popular_risk"] = (
        out["popularity"].le(3).astype(float)
        * (1.0 - out["lap_profile_fit_rank_in_race"])
        * (0.65 + 0.35 * out["race_lap_prediction_confidence"])
    ).clip(0.0, 1.0)

    # Actual race type is diagnostic only. It is never merged into ticket policies.
    actual_rpci = num(out.get("RPCI"), idx)
    actual_pci3 = num(out.get("PCI3"), idx)
    out["actual_lap_mode_diagnostic"] = np.select(
        [
            actual_rpci.le(47),
            actual_rpci.ge(53) & actual_pci3.ge(53),
            actual_pci3.ge(53),
            actual_rpci.between(47, 53, inclusive="both"),
        ],
        ["fast", "slow", "instant", "sustain"],
        default="unknown",
    )
    out["lap_mode_prediction_hit_diagnostic"] = out["predicted_lap_mode"].eq(out["actual_lap_mode_diagnostic"])
    return out


RUNNER_FEATURES = [
    "lap_profile_fit_score",
    "lap_profile_fit_rank_in_race",
    "lap_fit_confident_score",
    "lap_axis_candidate_score",
    "lap_partner_specialist_score",
    "lap_mismatch_popular_risk",
    "race_lap_prediction_confidence",
    "race_lap_profile_concentration",
    "horse_lap_profile_sharpness",
]


def summarize_runner(df: pd.DataFrame, mask: pd.Series, segment: str) -> dict[str, object]:
    seg = df.loc[mask.fillna(False)].copy()
    if seg.empty:
        return {"segment": segment, "races": 0, "horses": 0, "win_rate": np.nan, "top3_rate": np.nan, "win_roi": np.nan, "place_roi": np.nan}
    stake = len(seg) * 100.0
    return {
        "segment": segment,
        "races": int(seg["race_id"].nunique()),
        "horses": int(len(seg)),
        "win_rate": float(seg["is_win"].mean()),
        "top3_rate": float(seg["is_top3"].mean()),
        "win_roi": safe_div(float(seg["win_return_100"].sum()), stake),
        "place_roi": safe_div(float(seg["place_return_100"].sum()), stake),
        "avg_popularity": float(seg["popularity"].mean()),
        "avg_odds": float(seg["odds"].mean()),
    }


def evaluate_runner_features(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source, group in df.groupby("source", sort=False):
        group = group[group["finish"].notna()].copy()
        for feature in RUNNER_FEATURES:
            values = num(group.get(feature), group.index)
            valid = values.notna()
            if valid.sum() < 300:
                continue
            q80 = values.loc[valid].quantile(0.80)
            q20 = values.loc[valid].quantile(0.20)
            base = {
                "source": source,
                "feature": feature,
                "valid_rows": int(valid.sum()),
                "auc_top3": roc_auc(group.loc[valid, "is_top3"], values.loc[valid]),
                "auc_win": roc_auc(group.loc[valid, "is_win"], values.loc[valid]),
            }
            for label, mask in [
                ("top20pct", values.ge(q80)),
                ("bottom20pct", values.le(q20)),
                ("top20pct_pop4plus", values.ge(q80) & group["popularity"].ge(4)),
            ]:
                rows.append({**base, **summarize_runner(group, mask & valid, label)})
    return pd.DataFrame(rows)


def ticket_metrics(df: pd.DataFrame, name: str) -> dict[str, object]:
    if df.empty:
        return {"policy": name, "tickets": 0, "races": 0, "stake_yen": 0.0, "return_yen": 0.0, "roi": np.nan, "hit_rate": np.nan}
    stake = num(df.get("runtime_stake_yen"), df.index).fillna(num(df.get("stake_yen"), df.index, 0.0)).fillna(0.0)
    ret = num(df.get("runtime_return_yen"), df.index).fillna(num(df.get("return_yen"), df.index, 0.0)).fillna(0.0)
    return {
        "policy": name,
        "tickets": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "stake_yen": float(stake.sum()),
        "return_yen": float(ret.sum()),
        "roi": safe_div(float(ret.sum()), float(stake.sum())),
        "hit_rate": float(ret.gt(0).mean()),
    }


def load_tickets_with_runner_features(runners: pd.DataFrame, tickets_path: Path) -> pd.DataFrame:
    tickets = read_csv(tickets_path, dtype=str, low_memory=False)
    tickets["race_id"] = tickets["race_id"].astype(str)
    tickets["anchor_no"] = num(tickets.get("anchor_no"), tickets.index)
    tickets["partner_no"] = num(tickets.get("partner_no"), tickets.index)
    cols = [
        "race_id",
        "horse_no",
        "source",
        "predicted_lap_mode",
        "horse_lap_profile_top_mode",
        *RUNNER_FEATURES,
    ]
    r = runners[cols].copy()
    r["_source_order"] = r["source"].map({"test": 0, "train": 1}).fillna(9)
    r = r.sort_values("_source_order").drop_duplicates(["race_id", "horse_no"], keep="first")
    anchor = r.add_prefix("anchor_").rename(columns={"anchor_race_id": "race_id", "anchor_horse_no": "anchor_no"})
    partner = r.add_prefix("partner_").rename(columns={"partner_race_id": "race_id", "partner_horse_no": "partner_no"})
    out = tickets.merge(anchor, on=["race_id", "anchor_no"], how="left").merge(
        partner, on=["race_id", "partner_no"], how="left"
    )
    for feature in RUNNER_FEATURES:
        a = num(out.get(f"anchor_{feature}"), out.index)
        p = num(out.get(f"partner_{feature}"), out.index)
        out[f"pair_min_{feature}"] = pd.concat([a, p], axis=1).min(axis=1)
        out[f"pair_avg_{feature}"] = pd.concat([a, p], axis=1).mean(axis=1)
        out[f"pair_max_{feature}"] = pd.concat([a, p], axis=1).max(axis=1)
        out[f"pair_gap_{feature}"] = (a - p).abs()
    anchor_modes = out.get("anchor_horse_lap_profile_top_mode", pd.Series("", index=out.index)).fillna("")
    partner_modes = out.get("partner_horse_lap_profile_top_mode", pd.Series("", index=out.index)).fillna("")
    predicted_modes = out.get("anchor_predicted_lap_mode", pd.Series("", index=out.index)).fillna("")
    out["pair_lap_mode_same_flag"] = anchor_modes.eq(partner_modes).astype(float)
    out["pair_both_match_predicted_lap_flag"] = (anchor_modes.eq(predicted_modes) & partner_modes.eq(predicted_modes)).astype(float)
    out["pair_lap_contradiction_score"] = (
        0.48 * (1.0 - out["pair_lap_mode_same_flag"])
        + 0.32 * out["pair_gap_lap_profile_fit_score"].fillna(0.0).clip(0.0, 1.0)
        + 0.20 * out["pair_gap_horse_lap_profile_sharpness"].fillna(0.0).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    out["pair_lap_same_race_fit_score"] = (
        0.52 * out["pair_min_lap_fit_confident_score"].fillna(0.0)
        + 0.28 * out["pair_avg_lap_fit_confident_score"].fillna(0.0)
        + 0.20 * (1.0 - out["pair_lap_contradiction_score"].fillna(0.5))
    ).clip(0.0, 1.0)
    out["pair_lap_race_confidence"] = out["pair_avg_race_lap_prediction_confidence"].fillna(0.5).clip(0.0, 1.0)
    out["pair_lap_mismatch_popular_max"] = out["pair_max_lap_mismatch_popular_risk"].fillna(0.0).clip(0.0, 1.0)
    return out


def evaluate_ticket_policies(tickets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = [ticket_metrics(tickets, "base_all")]
    candidates = [
        ("pair_lap_same_race_fit_score", "skip_low_pair_lap_fit_q20", "low"),
        ("pair_lap_same_race_fit_score", "skip_low_pair_lap_fit_q30", "low30"),
        ("pair_lap_same_race_fit_score", "keep_high_pair_lap_fit_top20", "high"),
        ("pair_lap_contradiction_score", "skip_high_lap_contradiction_q80", "high_bad"),
        ("pair_lap_mismatch_popular_max", "skip_lap_mismatch_popular_q80", "high_bad"),
        ("pair_lap_race_confidence", "keep_high_race_lap_confidence_top50", "top50"),
    ]
    for col, policy, mode in candidates:
        values = num(tickets.get(col), tickets.index)
        valid = values.notna()
        if valid.sum() < 20:
            continue
        if mode == "low":
            threshold = values.loc[valid].quantile(0.20)
            mask = values.ge(threshold) | values.isna()
        elif mode == "low30":
            threshold = values.loc[valid].quantile(0.30)
            mask = values.ge(threshold) | values.isna()
        elif mode == "high":
            threshold = values.loc[valid].quantile(0.80)
            mask = values.ge(threshold)
        elif mode == "top50":
            threshold = values.loc[valid].quantile(0.50)
            mask = values.ge(threshold)
        else:
            threshold = values.loc[valid].quantile(0.80)
            mask = values.le(threshold) | values.isna()
        rows.append({**ticket_metrics(tickets.loc[mask.fillna(False)], policy), "feature": col, "threshold": float(threshold)})
    combo_fit = num(tickets.get("pair_lap_same_race_fit_score"), tickets.index)
    combo_conf = num(tickets.get("pair_lap_race_confidence"), tickets.index)
    combo_contra = num(tickets.get("pair_lap_contradiction_score"), tickets.index)
    fit_q30 = combo_fit.quantile(0.30)
    conf_q40 = combo_conf.quantile(0.40)
    contra_q80 = combo_contra.quantile(0.80)
    combo = combo_fit.ge(fit_q30) & combo_conf.ge(conf_q40) & combo_contra.le(contra_q80)
    rows.append(
        {
            **ticket_metrics(tickets.loc[combo.fillna(False)], "combo_lap_fit_confidence_no_contradiction"),
            "feature": "pair_lap_same_race_fit_score + pair_lap_race_confidence + pair_lap_contradiction_score",
            "threshold": float(fit_q30),
            "confidence_threshold": float(conf_q40),
            "contradiction_max": float(contra_q80),
        }
    )
    return pd.DataFrame(rows)


def write_readme(out_dir: Path, runner_eval: pd.DataFrame, ticket_eval: pd.DataFrame, runners: pd.DataFrame) -> None:
    def fmt_pct(x: float) -> str:
        return "" if pd.isna(x) else f"{x * 100:.1f}%"

    top = runner_eval[(runner_eval["source"] == "test") & (runner_eval["segment"] == "top20pct")].copy()
    top = top.sort_values(["place_roi", "auc_top3"], ascending=False).head(10)
    diag = runners[runners["source"].eq("test")].drop_duplicates("race_id")
    mode_hit = float(diag["lap_mode_prediction_hit_diagnostic"].mean()) if not diag.empty else float("nan")
    lines = [
        "# Lap Pair Refinement Candidate Check",
        "",
        "External advice: use lap/PCI/RPCI as race-quality and pair-simultaneity signals, not only runner-level additives.",
        "",
        "## Coverage",
        f"- runner rows: {len(runners):,}",
        f"- test rows: {(runners['source'] == 'test').sum():,}",
        f"- races: {runners['race_id'].nunique():,}",
        f"- diagnostic predicted lap mode hit rate: {fmt_pct(mode_hit)}",
        "",
        "## Runner Segments (test top20%)",
        "| feature | horses | races | top3 | win ROI | place ROI | AUC top3 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in top.iterrows():
        lines.append(
            f"| {r['feature']} | {int(r['horses'])} | {int(r['races'])} | {fmt_pct(r['top3_rate'])} | "
            f"{fmt_pct(r['win_roi'])} | {fmt_pct(r['place_roi'])} | {r['auc_top3']:.3f} |"
        )
    lines += [
        "",
        "## Strongest Ticket Overlay Policies",
        "| policy | tickets | races | ROI | hit | stake | return |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in ticket_eval.iterrows():
        lines.append(
            f"| {r['policy']} | {int(r['tickets'])} | {int(r['races'])} | {fmt_pct(r['roi'])} | "
            f"{fmt_pct(r['hit_rate'])} | {r['stake_yen']:.0f} | {r['return_yen']:.0f} |"
        )
    lines += [
        "",
        "## Recommendation",
        "- If pair-level lap fit improves ROI without collapsing ticket count, promote it first as a shadow label.",
        "- Do not use actual PCI/RPCI/PCI3 of the target race for betting. Actual mode is diagnostic only.",
        "- Popular-horse lap mismatch is a candidate danger filter only if it improves existing BUY overlays.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate lap/race-quality pair refinement features.")
    parser.add_argument("--tickets-csv", default=str(DEFAULT_TICKETS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runners = add_lap_pair_refinement_features(load_runner_features())
    runner_eval = evaluate_runner_features(runners)
    tickets = load_tickets_with_runner_features(runners, Path(args.tickets_csv))
    ticket_eval = evaluate_ticket_policies(tickets)

    keep_cols = [
        "source",
        "race_id",
        "horse_no",
        "馬名",
        "popularity",
        "odds",
        "finish",
        "is_win",
        "is_top3",
        "predicted_lap_mode",
        "actual_lap_mode_diagnostic",
        "lap_mode_prediction_hit_diagnostic",
        "horse_lap_profile_top_mode",
        *[f"race_need_{m}" for m in LAP_MODES],
        *[f"horse_lap_{m}" for m in LAP_MODES],
        *RUNNER_FEATURES,
    ]
    runners[keep_cols].to_csv(out_dir / "runner_lap_pair_refinement_features.csv", index=False, encoding="utf-8-sig")
    runner_eval.to_csv(out_dir / "runner_lap_feature_segments.csv", index=False, encoding="utf-8-sig")
    tickets.to_csv(out_dir / "s_priority_tickets_with_lap_pair_refinement.csv", index=False, encoding="utf-8-sig")
    ticket_eval.to_csv(out_dir / "ticket_lap_policy_metrics.csv", index=False, encoding="utf-8-sig")

    diag = runners[runners["source"].eq("test")].drop_duplicates("race_id")
    summary = {
        "output_dir": str(out_dir),
        "coverage": {
            "runner_rows": int(len(runners)),
            "test_rows": int((runners["source"] == "test").sum()),
            "train_rows": int((runners["source"] == "train").sum()),
            "races": int(runners["race_id"].nunique()),
            "diagnostic_lap_mode_hit_rate": float(diag["lap_mode_prediction_hit_diagnostic"].mean())
            if not diag.empty
            else None,
        },
        "runner_test_top20_by_place_roi": runner_eval[
            (runner_eval["source"] == "test") & (runner_eval["segment"] == "top20pct")
        ]
        .sort_values(["place_roi", "auc_top3"], ascending=False)
        .head(10)
        .to_dict(orient="records"),
        "ticket_policy_metrics": ticket_eval.to_dict(orient="records"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(out_dir, runner_eval, ticket_eval, runners)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
