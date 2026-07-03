from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTED = ROOT / "outputs/analysis/lap_positive_expansion_v1/lap_positive_expansion_selected_tickets.csv"
DEFAULT_META = ROOT / "data/datasets/cache/lap_pedigree_interactions_confirmed_opponent/test_features.csv"
DEFAULT_OUT = ROOT / "outputs/analysis/lap_positive_expansion_segments_v1"

VENUE_CODE_MAP = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}

SELECTED_COLS = [
    "race_id",
    "year",
    "policy",
    "ticket_type",
    "horse_a",
    "horse_b",
    "stake_yen",
    "return_yen",
    "hit",
    "anchor_pop",
    "partner_pop",
    "anchor_odds",
    "partner_odds",
    "wide_quote_proxy",
    "umaren_quote_proxy",
    "odds_geom",
    "market_overlay_score",
    "late_value_survives_score",
    "pair_quinella_score",
    "shape_pair_fit_score",
    "shape_pair_risk_score",
    "danger_sum",
    "danger_max",
    "queue_shape_label",
    "queue_clarity_score",
    "queue_duel_risk_score",
    "queue_front_load_score",
    "v2_predicted_lap_mode",
    "v2_confidence",
    "v2_margin",
    "pair_lap_profile_fit_avg",
    "pair_lap_confident_min",
    "pair_lap_axis_avg",
    "pair_lap_partner_specialist_max",
    "pair_lap_mismatch_popular_max",
    "lap_axis_specialist_role_score",
    "lap_positive_score",
    "lap_expansion_select_score",
    "lap_expansion_candidate_label",
    "lap_role_expansion_select_score",
]

META_COLS = [
    "race_id",
    "レースID(新/馬番無)",
    "場所",
    "race_class_name",
    "レース名",
    "surface",
    "distance",
    "芝・ダ",
    "距離",
    "distance_category_eval",
    "distance_bin",
    "クラス名",
    "馬場状態",
    "class_group",
]

DEFAULT_POLICIES = [
    "wide_price_sane_strong_base",
    "umaren_price_sane_strong_base",
    "wide_price_sane_strong_plus_value_mid_role_q90_extra_only",
    "wide_price_sane_strong_plus_value_mid_lap_q60_extra_only",
    "umaren_price_sane_strong_plus_value_mid_lap_q70_extra_only",
    "umaren_price_sane_strong_plus_value_mid_lap_q60_extra_only",
    "wide_price_sane_strong_plus_value_mid_role_q90",
    "wide_price_sane_strong_plus_value_mid_lap_q60",
    "umaren_price_sane_strong_plus_value_mid_lap_q70",
    "umaren_price_sane_strong_plus_value_mid_lap_q60",
]


def available_usecols(path: Path, wanted: list[str]) -> list[str]:
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    return [c for c in wanted if c in header.columns]


def read_selected(path: Path, policies: list[str]) -> pd.DataFrame:
    usecols = available_usecols(path, SELECTED_COLS)
    df = pd.read_csv(path, usecols=usecols, encoding="utf-8-sig", low_memory=False)
    df["race_id"] = clean_race_id(df["race_id"])
    if policies:
        df = df[df["policy"].astype(str).isin(policies)].copy()
    return df


def read_meta(path: Path) -> pd.DataFrame:
    usecols = available_usecols(path, META_COLS)
    if "race_id" not in usecols and "レースID(新/馬番無)" not in usecols:
        return pd.DataFrame(columns=["race_id"])
    meta = pd.read_csv(path, usecols=usecols, encoding="utf-8-sig", low_memory=False)
    if "race_id" not in meta.columns:
        meta["race_id"] = clean_race_id(meta["レースID(新/馬番無)"])
    else:
        meta["race_id"] = clean_race_id(meta["race_id"])
    meta = meta.drop_duplicates("race_id", keep="last")
    return meta


def clean_race_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True).str.strip().str.zfill(16)


def num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def text_col(df: pd.DataFrame, name: str, default: str = "") -> pd.Series:
    if name not in df.columns:
        return pd.Series(default, index=df.index, dtype="string")
    return df[name].astype("string").fillna(default)


def max_drawdown(profit: pd.Series) -> float:
    if profit.empty:
        return 0.0
    curve = pd.to_numeric(profit, errors="coerce").fillna(0.0).cumsum()
    return float((curve - curve.cummax()).min())


def roi_without_top_returns(frame: pd.DataFrame, n: int) -> float:
    if frame.empty:
        return 0.0
    kept = frame.sort_values("return_yen", ascending=False).iloc[n:]
    stake = num(kept.get("stake_yen"), kept.index, 0.0).fillna(0.0).sum()
    ret = num(kept.get("return_yen"), kept.index, 0.0).fillna(0.0).sum()
    return float(ret / stake) if stake > 0 else 0.0


def metric(frame: pd.DataFrame, label: str, segment_value: str = "") -> dict[str, Any]:
    if frame.empty:
        return {
            "policy": label,
            "segment_value": segment_value,
            "tickets": 0,
            "races": 0,
            "stake_yen": 0.0,
            "return_yen": 0.0,
            "profit_yen": 0.0,
            "roi": 0.0,
            "hit_rate": 0.0,
            "race_hit_rate": 0.0,
            "max_drawdown_yen": 0.0,
            "top1_removed_roi": 0.0,
            "top3_removed_roi": 0.0,
            "top5_removed_roi": 0.0,
            "avg_pair_odds_proxy": np.nan,
            "avg_lap_positive_score": np.nan,
            "avg_lap_role_score": np.nan,
            "avg_v2_confidence": np.nan,
        }
    stake = num(frame.get("stake_yen"), frame.index, 0.0).fillna(0.0)
    ret = num(frame.get("return_yen"), frame.index, 0.0).fillna(0.0)
    profit = ret - stake
    hit = frame.get("hit", False).astype(bool)
    ordered = frame.assign(_profit=profit).sort_values(["race_id", "ticket_type", "horse_a", "horse_b"], kind="mergesort")
    race_hit_rate = float(frame.groupby("race_id")["hit"].max().mean()) if "race_id" in frame.columns else float(hit.mean())
    return {
        "policy": label,
        "segment_value": segment_value,
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()),
        "stake_yen": float(stake.sum()),
        "return_yen": float(ret.sum()),
        "profit_yen": float(profit.sum()),
        "roi": float(ret.sum() / stake.sum()) if float(stake.sum()) > 0 else 0.0,
        "hit_rate": float(hit.mean()),
        "race_hit_rate": race_hit_rate,
        "max_drawdown_yen": max_drawdown(ordered["_profit"]),
        "top1_removed_roi": roi_without_top_returns(frame, 1),
        "top3_removed_roi": roi_without_top_returns(frame, 3),
        "top5_removed_roi": roi_without_top_returns(frame, 5),
        "avg_pair_odds_proxy": float(num(frame.get("pair_odds_proxy"), frame.index, np.nan).mean()),
        "avg_lap_positive_score": float(num(frame.get("lap_positive_score"), frame.index, np.nan).mean()),
        "avg_lap_role_score": float(num(frame.get("lap_axis_specialist_role_score"), frame.index, np.nan).mean()),
        "avg_v2_confidence": float(num(frame.get("v2_confidence"), frame.index, np.nan).mean()),
    }


def add_context(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(meta, on="race_id", how="left") if not meta.empty else df.copy()
    inferred_venue = out["race_id"].str[8:10].map(VENUE_CODE_MAP).fillna("不明")
    raw_venue = text_col(out, "場所")
    out["venue"] = raw_venue.mask(raw_venue.eq(""), inferred_venue)
    out["date_key"] = pd.to_datetime(out["race_id"].str[:8], format="%Y%m%d", errors="coerce").dt.strftime("%Y-%m-%d")
    out["race_no"] = pd.to_numeric(out["race_id"].str[14:16], errors="coerce")
    if "year" not in out.columns:
        out["year"] = pd.to_numeric(out["race_id"].str[:4], errors="coerce")
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")

    surface = text_col(out, "surface")
    fallback_surface = text_col(out, "芝・ダ")
    out["surface_eval"] = surface.mask(surface.eq(""), fallback_surface).fillna("").replace({"T": "芝", "D": "ダ"})
    out["surface_eval"] = out["surface_eval"].mask(out["surface_eval"].eq(""), "不明")

    distance = num(out.get("distance"), out.index, np.nan)
    fallback_distance = num(out.get("距離"), out.index, np.nan)
    out["distance_eval"] = distance.fillna(fallback_distance)
    out["distance_band"] = pd.cut(
        out["distance_eval"],
        bins=[0, 1400, 1700, 2000, 2300, 10000],
        labels=["短距離", "マイル", "中距離", "中長距離", "長距離"],
        right=True,
    ).astype("string").fillna("不明")

    class_name = text_col(out, "class_group")
    for col in ["race_class_name", "クラス名"]:
        class_name = class_name.mask(class_name.eq(""), text_col(out, col))
    out["class_eval"] = class_name.mask(class_name.eq(""), "不明")

    going = text_col(out, "馬場状態")
    out["going_eval"] = going.mask(going.eq(""), "不明")

    wide_quote = num(out.get("wide_quote_proxy"), out.index, np.nan)
    umaren_quote = num(out.get("umaren_quote_proxy"), out.index, np.nan)
    odds_geom = num(out.get("odds_geom"), out.index, np.nan)
    out["pair_odds_proxy"] = np.where(out["ticket_type"].astype(str).eq("wide"), wide_quote, umaren_quote)
    out["pair_odds_proxy"] = pd.Series(out["pair_odds_proxy"], index=out.index).fillna(odds_geom)
    out["pair_odds_proxy"] = out["pair_odds_proxy"].where(out["pair_odds_proxy"].le(100), out["pair_odds_proxy"] / 100.0)
    out["pair_odds_band"] = pd.cut(
        out["pair_odds_proxy"],
        bins=[0, 3, 5, 10, 20, 50, np.inf],
        labels=["<=3", "3-5", "5-10", "10-20", "20-50", "50+"],
        right=True,
    ).astype("string").fillna("不明")

    out["pop_sum"] = num(out.get("anchor_pop"), out.index, np.nan) + num(out.get("partner_pop"), out.index, np.nan)
    out["pop_sum_band"] = pd.cut(
        out["pop_sum"],
        bins=[0, 5, 10, 18, 30, np.inf],
        labels=["人気上位ペア", "中位人気ペア", "中穴ペア", "穴ペア", "大穴ペア"],
        right=True,
    ).astype("string").fillna("不明")

    out["lap_score_band"] = pd.cut(
        num(out.get("lap_positive_score"), out.index, np.nan),
        bins=[-0.01, 0.35, 0.44, 0.50, 1.0],
        labels=["低", "中", "高", "最上位"],
        right=True,
    ).astype("string").fillna("不明")
    out["lap_role_band"] = pd.cut(
        num(out.get("lap_axis_specialist_role_score"), out.index, np.nan),
        bins=[-0.01, 0.35, 0.50, 0.65, 1.0],
        labels=["低", "中", "高", "最上位"],
        right=True,
    ).astype("string").fillna("不明")
    out["v2_conf_band"] = pd.cut(
        num(out.get("v2_confidence"), out.index, np.nan),
        bins=[-0.01, 0.24, 0.30, 0.40, 1.0],
        labels=["低", "最低限", "中", "高"],
        right=True,
    ).astype("string").fillna("不明")
    out["mismatch_band"] = pd.cut(
        num(out.get("pair_lap_mismatch_popular_max"), out.index, np.nan),
        bins=[-0.01, 0.20, 0.30, 0.40, 1.0],
        labels=["低", "許容", "注意", "高"],
        right=True,
    ).astype("string").fillna("不明")
    out["lap_mode"] = text_col(out, "v2_predicted_lap_mode").mask(text_col(out, "v2_predicted_lap_mode").eq(""), "unknown")
    out["queue_eval"] = text_col(out, "queue_shape_label").mask(text_col(out, "queue_shape_label").eq(""), "unknown")
    return out


def segment_metrics(df: pd.DataFrame, segment_cols: list[str], *, min_tickets: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in segment_cols:
        if col not in df.columns:
            continue
        for (policy, value), group in df.groupby(["policy", col], dropna=False, sort=False):
            if len(group) < min_tickets:
                continue
            row = metric(group, str(policy), str(value))
            row["segment"] = col
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["roi", "tickets"], ascending=[False, False])


def year_robustness(df: pd.DataFrame, segment_cols: list[str], *, min_tickets: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in segment_cols:
        if col not in df.columns:
            continue
        for (policy, value), group in df.groupby(["policy", col], dropna=False, sort=False):
            if len(group) < min_tickets:
                continue
            base = metric(group, str(policy), str(value))
            year_rows = []
            for year, yg in group.groupby("year", dropna=False):
                if len(yg) < 8:
                    continue
                yr = metric(yg, str(policy), str(value))
                year_rows.append((str(year), yr["tickets"], yr["roi"], yr["hit_rate"], yr["profit_yen"]))
            row = {**base, "segment": col}
            row["year_buckets"] = len(year_rows)
            roi_values = [x[2] for x in year_rows]
            row["min_year_roi_ge8"] = float(min(roi_values)) if roi_values else np.nan
            row["profitable_year_buckets"] = int(sum(x[2] >= 1.0 for x in year_rows))
            for year, tickets, roi, hit_rate, profit in year_rows:
                row[f"tickets_{year}"] = tickets
                row[f"roi_{year}"] = roi
                row[f"hit_rate_{year}"] = hit_rate
                row[f"profit_{year}"] = profit
            row["robustness_label"] = classify_robustness(row)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["robustness_rank", "roi", "tickets"], ascending=[False, False, False])


def combo_rule_metrics(df: pd.DataFrame, combo_cols: list[str], *, min_tickets: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    extra = df[df["policy"].astype(str).str.contains("extra_only", na=False)].copy()
    base_wide = df[df["policy"].astype(str).eq("wide_price_sane_strong_base")].copy()
    base_umaren = df[df["policy"].astype(str).eq("umaren_price_sane_strong_base")].copy()
    for policy, part in extra.groupby("policy", sort=False):
        base = base_wide if str(policy).startswith("wide_") else base_umaren
        for size in [2, 3]:
            for cols in combinations(combo_cols, size):
                if any(c not in part.columns for c in cols):
                    continue
                group_cols = list(cols)
                for values, group in part.groupby(group_cols, dropna=False, sort=False):
                    values_tuple = values if isinstance(values, tuple) else (values,)
                    if any(str(v) in {"", "不明", "unknown", "nan", "<NA>"} for v in values_tuple):
                        continue
                    if len(group) < min_tickets:
                        continue
                    row = metric(group, str(policy), " & ".join(f"{c}={v}" for c, v in zip(group_cols, values_tuple)))
                    row["rule_cols"] = "+".join(group_cols)
                    row["rule_size"] = size
                    year_rows = []
                    for year, yg in group.groupby("year", dropna=False):
                        if len(yg) < 6:
                            continue
                        yr = metric(yg, str(policy), row["segment_value"])
                        year_rows.append((str(year), yr["tickets"], yr["roi"], yr["hit_rate"], yr["profit_yen"]))
                    row["year_buckets"] = len(year_rows)
                    roi_values = [x[2] for x in year_rows]
                    row["min_year_roi_ge6"] = float(min(roi_values)) if roi_values else np.nan
                    row["profitable_year_buckets"] = int(sum(x[2] >= 1.0 for x in year_rows))
                    for year, tickets, roi, hit_rate, profit in year_rows:
                        row[f"tickets_{year}"] = tickets
                        row[f"roi_{year}"] = roi
                        row[f"hit_rate_{year}"] = hit_rate
                        row[f"profit_{year}"] = profit
                    row["rule_action"] = classify_combo(row)
                    if not base.empty and int(row.get("rule_rank", 0)) >= 3:
                        portfolio = pd.concat([base, group], ignore_index=True)
                        portfolio["_ticket_key"] = (
                            portfolio["race_id"].astype(str)
                            + ":"
                            + portfolio["ticket_type"].astype(str)
                            + ":"
                            + pd.to_numeric(portfolio["horse_a"], errors="coerce").astype("Int64").astype(str)
                            + "-"
                            + pd.to_numeric(portfolio["horse_b"], errors="coerce").astype("Int64").astype(str)
                        )
                        portfolio = portfolio.drop_duplicates("_ticket_key", keep="last").drop(columns=["_ticket_key"])
                        port = metric(portfolio, f"{policy}_plus_rule", row["segment_value"])
                        row["base_policy"] = "wide_price_sane_strong_base" if str(policy).startswith("wide_") else "umaren_price_sane_strong_base"
                        row["portfolio_tickets"] = port["tickets"]
                        row["portfolio_races"] = port["races"]
                        row["portfolio_roi"] = port["roi"]
                        row["portfolio_hit_rate"] = port["hit_rate"]
                        row["portfolio_top3_removed_roi"] = port["top3_removed_roi"]
                        row["portfolio_profit_yen"] = port["profit_yen"]
                    rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["rule_rank", "roi", "tickets"], ascending=[False, False, False])


def classify_combo(row: dict[str, Any]) -> str:
    tickets = float(row.get("tickets", 0))
    roi = float(row.get("roi", 0))
    top3 = float(row.get("top3_removed_roi", 0))
    top5 = float(row.get("top5_removed_roi", 0))
    years = int(row.get("year_buckets", 0))
    profitable = int(row.get("profitable_year_buckets", 0))
    min_year = row.get("min_year_roi_ge6", np.nan)
    min_year_ok = pd.notna(min_year) and float(min_year) >= 0.75
    if tickets >= 35 and roi >= 1.20 and top3 >= 0.90 and top5 >= 0.75 and years >= 2 and profitable >= 2 and min_year_ok:
        row["rule_rank"] = 4
        return "小口昇格候補"
    if tickets >= 25 and roi >= 1.15 and top3 >= 0.75 and years >= 2 and profitable >= 1:
        row["rule_rank"] = 3
        return "シャドー強"
    if roi >= 1.25 and top3 < 0.75:
        row["rule_rank"] = 2
        return "配当依存"
    row["rule_rank"] = 1 if roi >= 1.0 else 0
    return "保留" if roi >= 1.0 else "不可"


def classify_robustness(row: dict[str, Any]) -> str:
    tickets = float(row.get("tickets", 0))
    roi = float(row.get("roi", 0))
    top3 = float(row.get("top3_removed_roi", 0))
    years = int(row.get("year_buckets", 0))
    profitable = int(row.get("profitable_year_buckets", 0))
    min_year = row.get("min_year_roi_ge8", np.nan)
    min_year_ok = pd.notna(min_year) and float(min_year) >= 0.85
    if tickets >= 80 and roi >= 1.12 and top3 >= 0.90 and years >= 2 and profitable >= 2 and min_year_ok:
        row["robustness_rank"] = 4
        return "昇格検討"
    if tickets >= 50 and roi >= 1.08 and top3 >= 0.75 and years >= 2 and profitable >= 1:
        row["robustness_rank"] = 3
        return "シャドー継続"
    if roi >= 1.15 and top3 < 0.75:
        row["robustness_rank"] = 2
        return "配当依存"
    if roi < 1.0:
        row["robustness_rank"] = 0
        return "拡張不可"
    row["robustness_rank"] = 1
    return "保留"


def write_readme(out_dir: Path, overall: pd.DataFrame, robustness: pd.DataFrame, combos: pd.DataFrame) -> None:
    top_extra = overall[overall["policy"].str.contains("extra_only", na=False)].head(8)
    promote = robustness[robustness.get("robustness_label", pd.Series(dtype=str)).isin(["昇格検討", "シャドー継続"])].head(12)
    lines = [
        "# Lap Positive Expansion Segment Analysis",
        "",
        "ラップ適性を正式BUY拡張に使える条件があるかを、競馬場・芝ダート・距離・クラス・オッズ帯・ラップ型で分解した検証です。",
        "",
        "## Extra-only Top",
        "",
    ]
    if top_extra.empty:
        lines.append("- 該当なし")
    else:
        for _, row in top_extra.iterrows():
            lines.append(
                f"- {row['policy']}: tickets={int(row['tickets'])}, races={int(row['races'])}, "
                f"ROI={row['roi']:.3f}, hit={row['hit_rate']:.3f}, top3_removed_roi={row['top3_removed_roi']:.3f}"
            )
    lines += ["", "## Robust Segment Candidates", ""]
    if promote.empty:
        lines.append("- 正式昇格まで到達したセグメントはなし。シャドーで継続確認が必要。")
    else:
        for _, row in promote.iterrows():
            lines.append(
                f"- {row['robustness_label']} / {row['policy']} / {row['segment']}={row['segment_value']}: "
                f"tickets={int(row['tickets'])}, ROI={row['roi']:.3f}, "
                f"top3_removed_roi={row['top3_removed_roi']:.3f}, min_year_roi={row['min_year_roi_ge8']:.3f}"
            )
    lines += ["", "## Combo Rule Candidates", ""]
    if combos.empty:
        lines.append("- 複合ルール候補なし。")
    else:
        keep = combos[combos["rule_action"].isin(["小口昇格候補", "シャドー強", "配当依存"])].head(12)
        if keep.empty:
            lines.append("- 小口昇格候補はなし。")
        else:
            for _, row in keep.iterrows():
                lines.append(
                    f"- {row['rule_action']} / {row['policy']} / {row['segment_value']}: "
                    f"tickets={int(row['tickets'])}, ROI={row['roi']:.3f}, "
                    f"hit={row['hit_rate']:.3f}, top3_removed_roi={row['top3_removed_roi']:.3f}"
                )
    lines += [
        "",
        "## Interpretation",
        "",
        "- ROIだけ高いが top3_removed_roi が低いものは、上位配当に依存しているため正式BUYにはしない。",
        "- 2026年が弱いセグメントは、現行オッズ環境との相性が悪い可能性があるためシャドー継続。",
        "- 正式BUYを増やすなら、少なくとも年別・配当除外後の両方で崩れにくいセグメントに限定する。",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze where lap-positive expansion can safely add tickets.")
    parser.add_argument("--selected-csv", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--meta-csv", type=Path, default=DEFAULT_META)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--policy", action="append", default=None, help="Policy to include. Can be repeated.")
    parser.add_argument("--min-segment-tickets", type=int, default=25)
    args = parser.parse_args()

    policies = args.policy or DEFAULT_POLICIES
    args.out_dir.mkdir(parents=True, exist_ok=True)

    selected = read_selected(args.selected_csv, policies)
    meta = read_meta(args.meta_csv) if args.meta_csv.exists() else pd.DataFrame(columns=["race_id"])
    df = add_context(selected, meta)

    overall_rows = [metric(group, policy) for policy, group in df.groupby("policy", sort=False)]
    overall = pd.DataFrame(overall_rows).sort_values(["roi", "tickets"], ascending=[False, False])

    segment_cols = [
        "ticket_type",
        "venue",
        "surface_eval",
        "distance_band",
        "class_eval",
        "going_eval",
        "pair_odds_band",
        "pop_sum_band",
        "lap_mode",
        "queue_eval",
        "lap_expansion_candidate_label",
        "lap_score_band",
        "lap_role_band",
        "v2_conf_band",
        "mismatch_band",
        "year",
    ]
    segments = segment_metrics(df, segment_cols, min_tickets=args.min_segment_tickets)
    robustness = year_robustness(df, segment_cols, min_tickets=max(args.min_segment_tickets, 40))

    safe = robustness[
        robustness.get("robustness_label", pd.Series(dtype=str)).isin(["昇格検討", "シャドー継続", "配当依存"])
    ].copy()
    combo_cols = [
        "venue",
        "surface_eval",
        "distance_band",
        "class_eval",
        "going_eval",
        "pair_odds_band",
        "pop_sum_band",
        "lap_mode",
        "queue_eval",
        "lap_expansion_candidate_label",
        "lap_score_band",
        "lap_role_band",
        "v2_conf_band",
        "mismatch_band",
    ]
    combos = combo_rule_metrics(df, combo_cols, min_tickets=20)

    overall.to_csv(args.out_dir / "lap_positive_policy_overall.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(args.out_dir / "lap_positive_segment_metrics.csv", index=False, encoding="utf-8-sig")
    robustness.to_csv(args.out_dir / "lap_positive_segment_robustness.csv", index=False, encoding="utf-8-sig")
    safe.to_csv(args.out_dir / "lap_positive_segment_action_candidates.csv", index=False, encoding="utf-8-sig")
    combos.to_csv(args.out_dir / "lap_positive_combo_rule_candidates.csv", index=False, encoding="utf-8-sig")

    summary = {
        "selected_csv": str(args.selected_csv),
        "meta_csv": str(args.meta_csv),
        "rows": int(len(df)),
        "races": int(df["race_id"].nunique()),
        "policies": policies,
        "overall_top": overall.head(12).to_dict(orient="records"),
        "action_candidates_top": safe.head(20).to_dict(orient="records"),
        "combo_rule_candidates_top": combos.head(20).to_dict(orient="records"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(args.out_dir, overall, robustness, combos)

    print(f"Wrote {args.out_dir}")
    print(overall.head(10).to_string(index=False))
    if not safe.empty:
        print("\nAction candidates")
        print(safe.head(12)[["robustness_label", "policy", "segment", "segment_value", "tickets", "roi", "top3_removed_roi", "min_year_roi_ge8"]].to_string(index=False))
    if not combos.empty:
        print("\nCombo candidates")
        print(
            combos.head(12)[
                ["rule_action", "policy", "segment_value", "tickets", "roi", "hit_rate", "top3_removed_roi", "min_year_roi_ge6"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
