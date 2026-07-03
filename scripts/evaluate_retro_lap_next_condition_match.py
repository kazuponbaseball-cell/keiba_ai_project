from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_historical_condition_lap_context import DEFAULT_FEATURES, DEFAULT_FRONT3F  # noqa: E402
from scripts.evaluate_historical_race_quality_overlay import (  # noqa: E402
    add_race_quality_scores,
    build_prior_context_by_race,
    load_race_base,
    load_runner_features,
)
from scripts.evaluate_retro_lap_adversity import merge_pair_features, ncol, select_top_per_race  # noqa: E402
from scripts.evaluate_shape_adjusted_pair_selection import (  # noqa: E402
    DEFAULT_RACE_SHAPE,
    DEFAULT_UNIVERSE,
    add_shape_scores,
    gate_mask,
    load_universe,
    ticket_return,
    hit_flag,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "outputs" / "analysis" / "retro_lap_adversity_v1"
OUT_DIR = ROOT / "outputs" / "analysis" / "retro_lap_next_condition_match_v1"


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_csv_any(path: Path, **kwargs: Any) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False, **kwargs)


def metric_row(frame: pd.DataFrame, policy: str) -> dict[str, Any]:
    stake = float(frame["stake_yen"].sum()) if not frame.empty else 0.0
    ret = float(frame["return_yen"].sum()) if not frame.empty else 0.0
    top = float(frame["return_yen"].max()) if not frame.empty else 0.0
    ex = frame.drop(frame["return_yen"].idxmax()) if not frame.empty and top > 0 else frame.iloc[0:0]
    ex_stake = float(ex["stake_yen"].sum()) if not ex.empty else 0.0
    ex_ret = float(ex["return_yen"].sum()) if not ex.empty else 0.0
    return {
        "policy": policy,
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()) if not frame.empty else 0,
        "stake_yen": round(stake, 1),
        "return_yen": round(ret, 1),
        "profit_yen": round(ret - stake, 1),
        "roi_pct": round(ret / stake * 100, 1) if stake > 0 else 0.0,
        "hit_rate_pct": round(float(frame["hit"].mean() * 100), 1) if not frame.empty else 0.0,
        "roi_ex_top1_pct": round(ex_ret / ex_stake * 100, 1) if ex_stake > 0 else 0.0,
        "top_return_share_pct": round(top / ret * 100, 1) if ret > 0 else 0.0,
        "avg_next_match": round(float(ncol(frame, "retro_next_pair_match_score", 0.0).mean()), 3) if not frame.empty else np.nan,
        "avg_next_mismatch": round(float(ncol(frame, "retro_next_pair_mismatch_score", 0.0).mean()), 3) if not frame.empty else np.nan,
        "avg_next_edge": round(float(ncol(frame, "retro_next_pair_edge_score", 0.0).mean()), 3) if not frame.empty else np.nan,
    }


def add_next_condition_match_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    fast_need = ncol(out, "race_quality_fast_need_score", 0.0).clip(0.0, 1.0)
    slow_need = ncol(out, "race_quality_slow_need_score", 0.0).clip(0.0, 1.0)
    sustain_need = ncol(out, "race_quality_sustain_need_score", 0.5).clip(0.0, 1.0)
    queue_front = ncol(out, "queue_front_load_score", 0.0).clip(0.0, 1.0)
    queue_duel = ncol(out, "queue_duel_risk_score", 0.0).clip(0.0, 1.0)
    queue_clarity = ncol(out, "queue_clarity_score", 0.0).clip(0.0, 1.0)
    label = out.get("queue_shape_label", pd.Series("unknown", index=out.index)).astype(str)

    front_loaded_need = np.maximum(fast_need.to_numpy(), (0.55 * queue_front + 0.30 * queue_duel).clip(0.0, 1.0).to_numpy())
    slow_repeat_need = np.maximum(slow_need.to_numpy(), (0.45 * queue_clarity * label.eq("single_leader_clear").astype(float)).to_numpy())
    collapse_or_sustain_need = np.maximum.reduce(
        [
            fast_need.to_numpy(),
            sustain_need.to_numpy() * 0.80,
            (0.45 * queue_duel + 0.25 * queue_front).clip(0.0, 1.0).to_numpy(),
        ]
    )
    standard_need = (1.0 - np.maximum(front_loaded_need, slow_repeat_need)).clip(0.0, 1.0)

    out["next_front_loaded_need_score"] = pd.Series(front_loaded_need, index=out.index).clip(0.0, 1.0)
    out["next_slow_repeat_need_score"] = pd.Series(slow_repeat_need, index=out.index).clip(0.0, 1.0)
    out["next_collapse_or_sustain_need_score"] = pd.Series(collapse_or_sustain_need, index=out.index).clip(0.0, 1.0)
    out["next_standard_need_score"] = pd.Series(standard_need, index=out.index).clip(0.0, 1.0)

    def side_score(side: str) -> tuple[pd.Series, pd.Series, pd.Series]:
        front_res = ncol(out, f"{side}_past3_avg_retro_lap_front_load_resistance", 0.0).clip(0.0, 1.0)
        slow_exc = ncol(out, f"{side}_past3_avg_retro_lap_slow_closer_excuse", 0.0).clip(0.0, 1.0)
        long_res = ncol(out, f"{side}_past3_avg_retro_lap_long_spurt_resistance", 0.0).clip(0.0, 1.0)
        pos_avg = ncol(out, f"{side}_past3_avg_retro_lap_positive_score", 0.0).clip(0.0, 1.0)
        overhelp = ncol(out, f"{side}_past3_avg_retro_lap_overhelped_score", 0.0).clip(0.0, 1.0)
        neg = ncol(out, f"{side}_past3_avg_retro_lap_negative_score", 0.0).clip(0.0, 1.0)
        evidence = ncol(out, f"{side}_retro_lap_prior_count", 0.0).clip(0.0, 5.0) / 5.0

        match = (
            0.36 * front_res * out["next_front_loaded_need_score"]
            + 0.32 * slow_exc * out["next_collapse_or_sustain_need_score"]
            + 0.24 * long_res * (0.55 * sustain_need + 0.45 * out["next_standard_need_score"])
            + 0.08 * pos_avg * (0.35 + 0.65 * evidence)
        ).clip(0.0, 1.0)

        mismatch = (
            0.38 * slow_exc * out["next_slow_repeat_need_score"]
            + 0.24 * front_res * slow_need
            + 0.22 * overhelp
            + 0.16 * neg
            + 0.08 * (1.0 - evidence)
        ).clip(0.0, 1.0)
        edge = (match - 0.70 * mismatch).clip(-1.0, 1.0)
        out[f"{side}_retro_next_match_score"] = match
        out[f"{side}_retro_next_mismatch_score"] = mismatch
        out[f"{side}_retro_next_edge_score"] = edge
        return match, mismatch, edge

    a_match, a_mismatch, a_edge = side_score("anchor")
    p_match, p_mismatch, p_edge = side_score("partner")
    out["retro_next_pair_match_score"] = (0.58 * np.maximum(a_match, p_match) + 0.42 * ((a_match + p_match) / 2.0)).clip(0.0, 1.0)
    out["retro_next_pair_mismatch_score"] = np.maximum(a_mismatch, p_mismatch).clip(0.0, 1.0)
    out["retro_next_pair_min_edge_score"] = np.minimum(a_edge, p_edge).clip(-1.0, 1.0)
    out["retro_next_pair_edge_score"] = (
        0.54 * out["retro_next_pair_match_score"]
        - 0.36 * out["retro_next_pair_mismatch_score"]
        + 0.10 * out["retro_next_pair_min_edge_score"]
    ).clip(-1.0, 1.0)
    out["retro_next_pair_context_ready"] = ncol(out, "race_quality_context_ready", 0.0).gt(0).astype(float)
    out["retro_next_pair_label"] = np.select(
        [
            out["retro_next_pair_edge_score"].ge(0.12),
            out["retro_next_pair_edge_score"].ge(0.06),
            out["retro_next_pair_mismatch_score"].ge(0.22),
        ],
        ["match_strong", "match_watch", "mismatch_caution"],
        default="neutral",
    )
    return out


def base_score(frame: pd.DataFrame, shape_weight: float = 0.85) -> pd.Series:
    risk_weight = min(0.12, shape_weight * 0.75)
    return (
        (1.0 - shape_weight) * frame["shape_base_rank_score"]
        + shape_weight * frame["shape_pair_fit_score"]
        + 0.06 * frame["shape_value_score"]
        - risk_weight * frame["shape_pair_risk_score"]
    )


def select(frame: pd.DataFrame, score_col: str, ticket_type: str, policy: str, gate: str) -> pd.DataFrame:
    selected = select_top_per_race(frame, score_col, ticket_type, policy, gate)
    selected["stake_yen"] = 100.0
    selected["return_yen"] = ticket_return(selected, ticket_type)
    selected["hit"] = hit_flag(selected, ticket_type)
    return selected


def run_overlay(df: pd.DataFrame, gates: list[str], weights: list[float], ticket_types: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    selections: list[pd.DataFrame] = []
    for gate in gates:
        gated = df[gate_mask(df, gate)].copy()
        if gated.empty:
            continue
        gated["base_score"] = base_score(gated, 0.85)
        valid = gated[ncol(gated, "retro_lap_pair_evidence_ready", 0.0).gt(0)]
        risk_q60 = float(valid["retro_lap_pair_risk_score"].quantile(0.60)) if not valid.empty else 1.0
        low_risk = gated[ncol(gated, "retro_lap_pair_risk_score", 1.0).le(risk_q60)].copy()

        pools = {
            "base_gate": gated,
            "retro_low_risk": low_risk,
            "retro_low_risk_match_top30": low_risk[ncol(low_risk, "retro_next_pair_edge_score", -1.0).ge(float(low_risk["retro_next_pair_edge_score"].quantile(0.70)))],
            "retro_low_risk_match_top20": low_risk[ncol(low_risk, "retro_next_pair_edge_score", -1.0).ge(float(low_risk["retro_next_pair_edge_score"].quantile(0.80)))],
            "retro_low_risk_match_positive": low_risk[ncol(low_risk, "retro_next_pair_edge_score", -1.0).ge(0.06)],
            "retro_low_risk_no_mismatch": low_risk[ncol(low_risk, "retro_next_pair_mismatch_score", 0.0).le(0.18)],
        }

        for pool_name, pool in pools.items():
            if pool.empty:
                continue
            for ticket_type in ticket_types:
                for weight in weights:
                    score_col = f"score_{pool_name}_w{weight:.2f}"
                    if weight == 0:
                        pool[score_col] = pool["base_score"]
                    else:
                        pool[score_col] = (
                            pool["base_score"]
                            + weight * ncol(pool, "retro_next_pair_edge_score", 0.0)
                            + 0.35 * weight * (ncol(pool, "retro_next_pair_match_score", 0.0) - 0.5)
                            - 0.25 * weight * ncol(pool, "retro_next_pair_mismatch_score", 0.0)
                        )
                    policy = f"{ticket_type}_{gate}_{pool_name}_nextmatch_w{weight:.2f}"
                    selected = select(pool, score_col, ticket_type, policy, gate)
                    selected["pool"] = pool_name
                    selected["next_match_weight"] = weight
                    selections.append(selected)
                    row = metric_row(selected, policy)
                    row["gate"] = gate
                    row["pool"] = pool_name
                    row["ticket_type"] = ticket_type
                    row["next_match_weight"] = weight
                    rows.append(row)
                    for year, gy in selected.groupby("year"):
                        yr = metric_row(gy, policy)
                        yr["year"] = int(year)
                        yr["gate"] = gate
                        yr["pool"] = pool_name
                        yr["ticket_type"] = ticket_type
                        yr["next_match_weight"] = weight
                        rows.append(yr)

    summary = pd.DataFrame(rows)
    detail = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    if not summary.empty:
        summary = summary.sort_values(["year", "roi_pct", "tickets"], ascending=[True, False, False], na_position="first")
    return summary, detail


def main() -> int:
    parser = argparse.ArgumentParser(description="Backtest whether retro lap adversity is useful when the next race condition matches.")
    parser.add_argument("--universe", default=str(DEFAULT_UNIVERSE))
    parser.add_argument("--race-shape", default=str(DEFAULT_RACE_SHAPE))
    parser.add_argument("--source-dir", default=str(SOURCE_DIR))
    parser.add_argument("--feature-csv", action="append", default=[])
    parser.add_argument("--front3f-csv", default=str(DEFAULT_FRONT3F))
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--gates", default="price_sane_strong,value_mid")
    parser.add_argument("--weights", default="0,0.04,0.08,0.12,0.18,0.24")
    parser.add_argument("--ticket-types", default="umaren,wide")
    args = parser.parse_args()

    universe_path = project_path(args.universe)
    race_shape_path = project_path(args.race_shape)
    source_dir = project_path(args.source_dir)
    out_dir = project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_paths = [project_path(p) for p in (args.feature_csv or DEFAULT_FEATURES)]
    front3f_path = project_path(args.front3f_csv)
    gates = [x.strip() for x in args.gates.split(",") if x.strip()]
    weights = [float(x.strip()) for x in args.weights.split(",") if x.strip()]
    ticket_types = [x.strip() for x in args.ticket_types.split(",") if x.strip()]

    prior = read_csv_any(source_dir / "runner_retro_lap_prior_features.csv", dtype={"race_id": str})
    universe = add_shape_scores(load_universe(universe_path, race_shape_path))
    race_ids = set(universe["race_id"].astype(str))
    race_source = load_race_base(feature_paths, front3f_path)
    context = build_prior_context_by_race(race_source, race_ids, years=args.years)
    runner_features = load_runner_features(feature_paths)

    scored = add_race_quality_scores(universe, context, runner_features)
    scored = merge_pair_features(scored, prior)
    scored = add_next_condition_match_scores(scored)

    summary, detail = run_overlay(scored, gates, weights, ticket_types)

    score_cols = [
        "race_id",
        "year",
        "pair_key_norm",
        "anchor_no",
        "anchor_name",
        "partner_no",
        "partner_name",
        "race_quality_label",
        "race_quality_fast_need_score",
        "race_quality_slow_need_score",
        "race_quality_sustain_need_score",
        "queue_shape_label",
        "retro_lap_pair_fit_score",
        "retro_lap_pair_risk_score",
        "retro_next_pair_match_score",
        "retro_next_pair_mismatch_score",
        "retro_next_pair_edge_score",
        "retro_next_pair_label",
        "wide_hit",
        "umaren_hit",
        "wide_pay",
        "umaren_pay",
    ]
    scored[[c for c in score_cols if c in scored.columns]].to_csv(out_dir / "tickets_with_next_condition_match.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "next_condition_match_summary.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(out_dir / "next_condition_match_detail.csv", index=False, encoding="utf-8-sig")

    total_summary = summary[summary["year"].isna()].copy() if "year" in summary.columns else summary
    best = total_summary.head(30).replace({np.nan: None}).to_dict(orient="records") if not total_summary.empty else []
    report = {
        "universe": str(universe_path.relative_to(ROOT)),
        "rows": int(len(scored)),
        "races": int(scored["race_id"].nunique()) if not scored.empty else 0,
        "context_ready_rate_pct": round(float(ncol(scored, "race_quality_context_ready", 0.0).mean() * 100), 1),
        "top_policies": best,
        "note": "Shadow validation. Prior retro-lap adversity is shifted per horse; next-race condition uses pre-race race-quality and queue-shape estimates, not the actual result lap.",
    }
    (out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = """# Retro Lap Next Condition Match v1

目的: 過去走で受けたラップ不利/利得が、今回の想定ラップ質と噛み合う場合だけ評価を上げる検証。

リーク対策:
- 過去走ラップ不利は `runner_retro_lap_prior_features.csv` の shift 済み特徴を使用。
- 今回のレース質は `race_quality_*` と `queue_shape_label` の事前推定のみ使用。
- 実際の当該レースRPCI/ラップ/結果は選定スコアに使わない。

出力:
- `tickets_with_next_condition_match.csv`
- `next_condition_match_summary.csv`
- `next_condition_match_detail.csv`
- `summary.json`
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
