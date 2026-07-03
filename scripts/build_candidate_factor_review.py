from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "analysis" / "candidate_factor_review_v1"


def read_json(path: str) -> dict[str, Any]:
    p = ROOT / path
    if not p.exists():
        return {"missing": str(p)}
    return json.loads(p.read_text(encoding="utf-8"))


def read_csv(path: str) -> pd.DataFrame:
    p = ROOT / path
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, encoding="utf-8-sig", low_memory=False)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if pd.isna(value):
        return None
    return value


def pct(value: float | None, multiplier: float = 1.0) -> str:
    if value is None:
        return "NA"
    return f"{value * multiplier:.1f}%"


def get_metric(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return None


def pick_policy(rows: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    for row in rows:
        if row.get("policy") == policy:
            return row
    return {}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    female = read_json("outputs/analysis/female_dirt_summer_roi_v1/summary.json")
    first_impressive = read_json("outputs/analysis/first_condition_impressive_rescue_mcs_v4_v1/summary.json")
    first_uncertainty = read_json("outputs/analysis/first_condition_uncertainty_overlay_mcs_v4_v1/summary.json")
    late_value = read_json("outputs/analysis/late_value_survival_gate_v1/summary.json")
    front5_ticket = read_json("outputs/analysis/front5_ticket_overlay_v1/summary.json")
    front5_model = read_json("outputs/analysis/front5_position_model_v1/summary.json")
    shadow = read_json("outputs/analysis/shadow_promotion_readiness_v1/summary.json")
    risk = read_json("outputs/analysis/risk_models_v1/summary.json")
    danger_gate = read_json("outputs/analysis/ticket_type_danger_gate_v1/summary.json")

    female_policy_csv = read_csv("outputs/analysis/female_dirt_summer_roi_v1/ticket_policy_summary.csv")
    female_stress_csv = read_csv("outputs/analysis/female_dirt_summer_roi_v1/ticket_policy_stress_summary.csv")

    def csv_policy(source: str, policy: str) -> dict[str, Any]:
        if female_policy_csv.empty:
            return {}
        rows = female_policy_csv[
            female_policy_csv["source"].eq(source) & female_policy_csv["policy"].eq(policy)
        ]
        return rows.iloc[0].to_dict() if len(rows) else {}

    def csv_stress(source: str, policy: str, remove_top_returns: int) -> dict[str, Any]:
        if female_stress_csv.empty:
            return {}
        rows = female_stress_csv[
            female_stress_csv["source"].eq(source)
            & female_stress_csv["policy"].eq(policy)
            & female_stress_csv["remove_top_returns"].eq(remove_top_returns)
        ]
        return rows.iloc[0].to_dict() if len(rows) else {}

    female_runtime = female.get("ticket_policy_top_recommended_runtime", [])
    female_stress = female.get("ticket_policy_stress_recommended_runtime", [])
    source = "recommended_runtime_tickets"
    summer_turf = csv_policy(source, "require_summer_female_turf") or pick_policy(
        female_runtime, "require_summer_female_turf"
    )
    summer_light = csv_policy(source, "require_summer_female_lightweight") or pick_policy(
        female_runtime, "require_summer_female_lightweight"
    )
    summer_dirt_excl = csv_policy(source, "exclude_summer_female_dirt") or pick_policy(
        female_runtime, "exclude_summer_female_dirt"
    )
    female_base = csv_policy(source, "baseline_all") or pick_policy(female_runtime, "baseline_all")
    summer_turf_stress3 = csv_stress(source, "require_summer_female_turf", 3) or next(
        (
            r
            for r in female_stress
            if r.get("policy") == "require_summer_female_turf" and r.get("remove_top_returns") == 3
        ),
        {},
    )
    summer_light_stress3 = csv_stress(source, "require_summer_female_lightweight", 3) or next(
        (
            r
            for r in female_stress
            if r.get("policy") == "require_summer_female_lightweight" and r.get("remove_top_returns") == 3
        ),
        {},
    )

    factor_rows = [
        {
            "factor": "夏牝馬×芝",
            "status": "加点候補",
            "use": "準候補昇格スコアに薄く加点。ハードフィルターにはしない。",
            "evidence": {
                "tickets": summer_turf.get("tickets"),
                "races": summer_turf.get("races"),
                "roi_pct": summer_turf.get("roi_pct"),
                "hit_rate_pct": summer_turf.get("hit_rate_pct"),
                "top3_removed_roi_pct": summer_turf_stress3.get("roi_pct"),
            },
            "risk": "サンプルは29R。BUY直結ではなくシャドー昇格の補助から。",
            "decision": "candidate_promote_score",
        },
        {
            "factor": "夏牝馬×斤量利",
            "status": "強い加点候補",
            "use": "芝・軽斤量・既存AI評価が揃う準候補を上げる材料。",
            "evidence": {
                "tickets": summer_light.get("tickets"),
                "races": summer_light.get("races"),
                "roi_pct": summer_light.get("roi_pct"),
                "hit_rate_pct": summer_light.get("hit_rate_pct"),
                "top3_removed_roi_pct": summer_light_stress3.get("roi_pct"),
            },
            "risk": "16Rと薄い。採用はスコア加点まで。",
            "decision": "candidate_promote_score",
        },
        {
            "factor": "夏牝馬×ダート",
            "status": "減点候補",
            "use": "即切りではなく、昇格を止める警戒フラグ。",
            "evidence": {
                "exclude_roi_pct": summer_dirt_excl.get("roi_pct"),
                "baseline_roi_pct": female_base.get("roi_pct"),
            },
            "risk": "一律除外は点数を減らす。強い他要素があれば残す。",
            "decision": "risk_flag",
        },
        {
            "factor": "牝馬限定勝ち上がり×ダート",
            "status": "加点非推奨",
            "use": "市場に織り込まれている可能性。軽い警戒のみ。",
            "evidence": {
                "runner_win_roi_pct": 63.9,
                "runner_place_roi_pct": 62.0,
                "runtime_ticket_hits": 0,
            },
            "risk": "勝率は高めでも妙味が出ていない。",
            "decision": "do_not_add_as_positive",
        },
        {
            "factor": "初条件レスキュー",
            "status": "採用寄りだが拡張用ではない",
            "use": "経験不足馬を盲目的に救済せず、不確実ペアには追加margin/EVを要求。",
            "evidence": {
                "baseline_tickets": first_impressive.get("baseline", {}).get("tickets"),
                "baseline_roi": first_impressive.get("baseline", {}).get("roi"),
                "best_tickets": first_impressive.get("best", {}).get("tickets"),
                "best_roi": first_impressive.get("best", {}).get("roi"),
                "best_hit_rate": first_impressive.get("best", {}).get("ticket_hit_rate"),
            },
            "risk": "点数は297→251に減る。買い増しではなく品質ゲート。",
            "decision": "quality_gate_keep",
        },
        {
            "factor": "初条件不確実性",
            "status": "採用寄り",
            "use": "人気・前走内容の裏付けがない初条件馬を危険視。",
            "evidence": {
                "baseline_roi": first_uncertainty.get("baseline", {}).get("roi"),
                "best_roi": first_uncertainty.get("best", {}).get("roi"),
                "best_tickets": first_uncertainty.get("best", {}).get("tickets"),
            },
            "risk": "これも点数を減らす方向。昇格用途には別検証が必要。",
            "decision": "quality_gate_keep",
        },
        {
            "factor": "T-5/T-3妙味残存",
            "status": "最優先",
            "use": "直前オッズで妙味が残るかをBUY/準候補昇格の中核にする。",
            "evidence": {
                "ungated_roi": late_value.get("ungated", {}).get("roi"),
                "gated_roi": late_value.get("gated", {}).get("roi"),
                "ungated_tickets": late_value.get("ungated", {}).get("tickets"),
                "gated_tickets": late_value.get("gated", {}).get("tickets"),
                "delta_roi": late_value.get("delta_roi"),
            },
            "risk": "現状はproxy色が強い。実T-5/T-3固定スナップショットで継続検証必須。",
            "decision": "highest_priority_runtime",
        },
        {
            "factor": "前に行ける人気薄/Front5",
            "status": "補助",
            "use": "前目確率は当たるが、強く絞るとROIは下がりやすい。相手選びの説明/安定化に使う。",
            "evidence": {
                "front5_auc": front5_model.get("overall_metrics", [{}])[0].get("auc"),
                "top10pct_front5_rate": front5_model.get("overall_metrics", [{}])[0].get("top10pct_front5_rate"),
                "base_roi": front5_ticket.get("comparison", [{}])[0].get("roi"),
                "front_ge050_roi": front5_ticket.get("comparison", [{}, {}])[1].get("roi"),
                "base_top10_removed_roi": front5_ticket.get("comparison", [{}])[0].get("top10_removed_roi"),
            },
            "risk": "前目だけで買うとROI低下。ペア確率・展開利得とセット。",
            "decision": "supporting_feature_not_hard_gate",
        },
        {
            "factor": "危険人気馬",
            "status": "警戒フラグ",
            "use": "券種別/ペア単位の警戒に留める。",
            "evidence": {
                "ungated_roi": danger_gate.get("ungated", {}).get("roi"),
                "danger_gated_roi": danger_gate.get("danger_gated", {}).get("roi"),
                "delta_roi": danger_gate.get("delta_roi"),
            },
            "risk": "単純ゲートはROI悪化。広く切ると旨味も切る。",
            "decision": "warning_only_pair_level",
        },
        {
            "factor": "レース難易度",
            "status": "採用済み/継続",
            "use": "見送り・減額・アラートに使う。買い増しには使わない。",
            "evidence": {
                "difficulty_bins": risk.get("diagnostics", {}).get("race_difficulty_bins", []),
            },
            "risk": "低難易度=高ROIではない。買う条件ではなく事故回避。",
            "decision": "skip_stake_guard",
        },
        {
            "factor": "準候補昇格スコア",
            "status": "シャドー継続",
            "use": "単一条件落ち候補をシャドーで蓄積。BUY昇格はまだ禁止。",
            "evidence": {
                "pools": shadow.get("pool_summary", []),
                "promotion_allowed_now": shadow.get("policy", {}).get("promotion_allowed_now"),
            },
            "risk": "1日分・上位配当依存。formal BUYの全チェックは未通過。",
            "decision": "shadow_only",
        },
    ]

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(OUT_DIR),
        "factor_rows": factor_rows,
        "ranking": [
            "T-5/T-3妙味残存",
            "夏牝馬×芝/斤量利を準候補昇格スコアへ薄く加点",
            "初条件不確実性ゲートは維持",
            "Front5は説明・相手品質の補助",
            "危険人気馬はハード除外せず警戒",
            "準候補昇格はシャドー継続",
        ],
        "adoption_policy": {
            "avoid_more_hard_filters": True,
            "prefer_score_bonus_for_expansion": True,
            "do_not_change_champion_from_this_review_alone": True,
        },
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Candidate Factor Review v1",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## 結論",
        "",
        "- 点数をさらに減らすハードフィルターは増やさない。",
        "- 買い目を増やす方向で使えるのは、夏牝馬×芝/斤量利のスコア加点と、T-5/T-3妙味残存。",
        "- 初条件・危険人気・レース難易度・Front5は、昇格よりも品質管理と説明に向く。",
        "",
        "## ファクター別",
        "",
    ]
    for row in factor_rows:
        evidence = row["evidence"]
        lines.extend(
            [
                f"### {row['factor']}",
                f"- status: {row['status']}",
                f"- use: {row['use']}",
                f"- decision: {row['decision']}",
                f"- risk: {row['risk']}",
                f"- evidence: `{json.dumps(json_ready(evidence), ensure_ascii=False)}`",
                "",
            ]
        )
    (OUT_DIR / "review.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
