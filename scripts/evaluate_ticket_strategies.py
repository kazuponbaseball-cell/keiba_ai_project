from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_json_config
from src.utils.paths import ensure_dir, project_path


def _num(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        series = series.astype(str).str.replace(",", "", regex=False).str.replace("(", "", regex=False).str.replace(")", "", regex=False)
    return pd.to_numeric(series, errors="coerce")


def _col(df: pd.DataFrame, names: list[str]) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f"Missing any of {names}")


def _add_model_columns(df: pd.DataFrame, model_path: Path, race_col: str) -> pd.DataFrame:
    with model_path.open("rb") as f:
        model = pickle.load(f)
    out = df.copy()
    out["ai_score"] = model.predict(out)
    out["ai_rank"] = out.groupby(race_col)["ai_score"].rank(ascending=False, method="first").astype(int)
    out["popularity_num"] = _num(out[_col(out, ["人気", "莠ｺ豌・"])])
    out["odds_decimal"] = _num(out[_col(out, ["単勝オッズ", "蜊伜享繧ｪ繝・ぜ"])])
    out["pop_rank"] = out.groupby(race_col)["popularity_num"].rank(ascending=True, method="first").astype(int)
    out["ai_pop_gap"] = out["ai_rank"] - out["pop_rank"]
    second = out.groupby(race_col)["ai_score"].transform(lambda s: s.sort_values(ascending=False).iloc[1] if len(s) > 1 else np.nan)
    out["ai_score_gap_to_second"] = (out["ai_score"] - second).where(out["ai_rank"] == 1, 0.0).fillna(0.0)
    out["rank_num"] = _num(out[_col(out, ["確定着順", "遒ｺ螳夂捩鬆・"])])
    out["horse_name_for_ticket"] = out[_col(out, ["馬名", "鬥ｬ蜷・"])].astype(str)
    return out


def _load_pair_payoffs(raw_csv: Path, race_col: str, encoding: str) -> pd.DataFrame:
    usecols = [race_col, "確定着順", "馬連", "馬単"]
    raw = pd.read_csv(raw_csv, encoding=encoding, usecols=lambda c: c in usecols, low_memory=False)
    raw["rank_num"] = _num(raw["確定着順"])
    raw["umaren_pay"] = _num(raw["馬連"]).fillna(0.0)
    raw["umatan_pay"] = _num(raw["馬単"]).fillna(0.0)
    pay = (
        raw[raw["rank_num"].isin([1, 2])]
        .groupby(race_col, as_index=False)
        .agg(umaren_pay=("umaren_pay", "max"), umatan_pay=("umatan_pay", "max"))
    )
    return pay


def _strategy_masks(df: pd.DataFrame) -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    return {
        "anchor_ai1": lambda x: x["ai_rank"] == 1,
        "anchor_ai1_gap005": lambda x: (x["ai_rank"] == 1) & (x["ai_score_gap_to_second"] >= 0.05),
        "anchor_ai1_gap010": lambda x: (x["ai_rank"] == 1) & (x["ai_score_gap_to_second"] >= 0.10),
        "anchor_ai1_pop3": lambda x: (x["ai_rank"] == 1) & (x["pop_rank"] <= 3),
        "anchor_ai1_pop3_gap005": lambda x: (x["ai_rank"] == 1) & (x["pop_rank"] <= 3) & (x["ai_score_gap_to_second"] >= 0.05),
        "partner_ai_top5_pop10plus": lambda x: (x["ai_rank"] <= 5) & (x["popularity_num"] >= 10),
        "partner_ai_top5_market_gap5": lambda x: (x["ai_rank"] <= 5) & (x["ai_pop_gap"] <= -5),
        "partner_ai_top3_market_gap5": lambda x: (x["ai_rank"] <= 3) & (x["ai_pop_gap"] <= -5),
        "partner_market_gap5": lambda x: x["ai_pop_gap"] <= -5,
        "partner_ai_top5_pop7plus": lambda x: (x["ai_rank"] <= 5) & (x["popularity_num"] >= 7),
    }


def _build_tickets(df: pd.DataFrame, race_col: str, anchor_name: str, partner_name: str, top_partners: int) -> pd.DataFrame:
    masks = _strategy_masks(df)
    anchor_mask = masks[anchor_name](df)
    partner_mask = masks[partner_name](df)
    anchor_cols = [
        race_col,
        "horse_name_for_ticket",
        "ai_rank",
        "popularity_num",
        "target_top3",
        "rank_num",
    ]
    partner_cols = [
        race_col,
        "horse_name_for_ticket",
        "ai_rank",
        "popularity_num",
        "odds_decimal",
        "target_top3",
        "rank_num",
    ]
    anchors = (
        df[anchor_mask]
        .sort_values([race_col, "ai_rank", "popularity_num"])
        .groupby(race_col, as_index=False)
        .head(1)[anchor_cols]
        .rename(
            columns={
                "horse_name_for_ticket": "anchor_horse",
                "ai_rank": "anchor_ai_rank",
                "popularity_num": "anchor_popularity",
                "target_top3": "anchor_top3",
                "rank_num": "anchor_rank_num",
            }
        )
    )
    partners = df[partner_mask].sort_values([race_col, "ai_rank", "popularity_num"]).copy()
    if top_partners > 0:
        partners = partners.groupby(race_col, as_index=False).head(top_partners)
    partners = partners[partner_cols].rename(
        columns={
            "horse_name_for_ticket": "partner_horse",
            "ai_rank": "partner_ai_rank",
            "popularity_num": "partner_popularity",
            "odds_decimal": "partner_odds",
            "target_top3": "partner_top3",
            "rank_num": "partner_rank_num",
        }
    )
    tickets = anchors.merge(partners, on=race_col, how="inner")
    tickets = tickets[tickets["anchor_horse"] != tickets["partner_horse"]].copy()
    tickets["wide_like_hit"] = ((tickets["anchor_top3"] == 1) & (tickets["partner_top3"] == 1)).astype(int)
    tickets["top2_hit"] = ((tickets["anchor_rank_num"] <= 2) & (tickets["partner_rank_num"] <= 2)).astype(int)
    tickets["umaren_hit"] = tickets["top2_hit"]
    return tickets[
        [
            race_col,
            "anchor_horse",
            "partner_horse",
            "anchor_ai_rank",
            "partner_ai_rank",
            "anchor_popularity",
            "partner_popularity",
            "partner_odds",
            "wide_like_hit",
            "top2_hit",
            "umaren_hit",
        ]
    ]


def _metrics(tickets: pd.DataFrame, payoffs: pd.DataFrame, race_col: str, label: str) -> dict[str, float | int | str | None]:
    if tickets.empty:
        return {
            "strategy": label,
            "tickets": 0,
            "races": 0,
            "wide_like_hit_rate": 0.0,
            "wide_break_even_avg_pay": None,
            "umaren_hit_rate": 0.0,
            "umaren_roi": 0.0,
            "avg_partner_popularity": None,
            "avg_partner_odds": None,
        }
    merged = tickets.merge(payoffs, on=race_col, how="left")
    stake = len(merged) * 100.0
    wide_hit_rate = float(merged["wide_like_hit"].mean())
    umaren_pay = merged["umaren_pay"].fillna(0.0).where(merged["umaren_hit"] == 1, 0.0)
    return {
        "strategy": label,
        "tickets": int(len(merged)),
        "races": int(merged[race_col].nunique()),
        "wide_like_hit_rate": wide_hit_rate,
        "wide_break_even_avg_pay": float(100.0 / wide_hit_rate) if wide_hit_rate > 0 else None,
        "umaren_hit_rate": float(merged["umaren_hit"].mean()),
        "umaren_roi": float(umaren_pay.sum() / stake),
        "umaren_profit_flat100": float(umaren_pay.sum() - stake),
        "avg_partner_popularity": float(merged["partner_popularity"].mean()),
        "avg_partner_odds": float(merged["partner_odds"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate anchor-value pair ticket strategies.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--raw-csv", default="date/raw/全競走馬成績.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/ticket_strategies")
    parser.add_argument("--top-partners", type=int, default=3)
    args = parser.parse_args()

    config = load_json_config(args.config)
    race_col = config["data"]["race_id_column"]
    encoding = config["data"].get("encoding", "cp932")
    df = pd.read_csv(project_path(args.test_csv), low_memory=False)
    scored = _add_model_columns(df, project_path(args.model), race_col)
    payoffs = _load_pair_payoffs(project_path(args.raw_csv), race_col, encoding)

    anchors = ["anchor_ai1", "anchor_ai1_gap005", "anchor_ai1_gap010", "anchor_ai1_pop3", "anchor_ai1_pop3_gap005"]
    partners = [
        "partner_ai_top5_pop10plus",
        "partner_ai_top5_market_gap5",
        "partner_ai_top3_market_gap5",
        "partner_market_gap5",
        "partner_ai_top5_pop7plus",
    ]
    rows = []
    ticket_frames = []
    for anchor in anchors:
        for partner in partners:
            label = f"{anchor}__{partner}__top{args.top_partners}"
            tickets = _build_tickets(scored, race_col, anchor, partner, args.top_partners)
            rows.append(_metrics(tickets, payoffs, race_col, label))
            if not tickets.empty:
                tickets = tickets.copy()
                tickets["strategy"] = label
                ticket_frames.append(tickets)

    summary = pd.DataFrame(rows).sort_values(["umaren_roi", "wide_like_hit_rate", "tickets"], ascending=[False, False, False])
    output_dir = ensure_dir(project_path(args.output_dir))
    summary_path = output_dir / "ticket_strategy_summary.csv"
    tickets_path = output_dir / "ticket_details.csv"
    json_path = output_dir / "summary.json"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    if ticket_frames:
        pd.concat(ticket_frames, ignore_index=True).to_csv(tickets_path, index=False, encoding="utf-8-sig")
    payload = {
        "summary_csv": str(summary_path),
        "ticket_details_csv": str(tickets_path),
        "top_strategies": summary.head(20).to_dict(orient="records"),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
