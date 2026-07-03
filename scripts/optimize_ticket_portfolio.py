from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_ticket_strategies import _add_model_columns, _load_pair_payoffs, _num
from src.data.loaders import load_json_config
from src.utils.paths import ensure_dir, project_path


def _role_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "core_anchor": (df["ai_rank"] == 1) & (df["pop_rank"] <= 3) & (df["ai_score_gap_to_second"] >= 0.05),
        "clear_head": (df["ai_rank"] == 1) & (df["ai_score_gap_to_second"] >= 0.10),
        "head_or_anchor": (df["ai_rank"] == 1) & (df["ai_score_gap_to_second"] >= 0.05),
        "value_longshot": (df["ai_rank"] <= 5) & (df["popularity_num"] >= 10),
        "value_market_gap": (df["ai_rank"] <= 5) & (df["ai_pop_gap"] <= -5),
        "value_top3_gap": (df["ai_rank"] <= 3) & (df["ai_pop_gap"] <= -5),
        "solid_fill": (df["ai_rank"] <= 5) & (df["popularity_num"] <= 7),
    }


def _one_per_race(df: pd.DataFrame, mask: pd.Series, race_col: str) -> pd.DataFrame:
    return (
        df[mask]
        .sort_values([race_col, "ai_rank", "popularity_num", "odds_decimal"], ascending=[True, True, True, False])
        .groupby(race_col, as_index=False)
        .head(1)
    )


def _topn_per_race(df: pd.DataFrame, mask: pd.Series, race_col: str, n: int) -> pd.DataFrame:
    return (
        df[mask]
        .sort_values([race_col, "ai_rank", "popularity_num", "odds_decimal"], ascending=[True, True, False, False])
        .groupby(race_col, as_index=False)
        .head(n)
    )


def _base_horse_cols(prefix: str) -> dict[str, str]:
    return {
        "horse_name_for_ticket": f"{prefix}_horse",
        "ai_rank": f"{prefix}_ai_rank",
        "popularity_num": f"{prefix}_popularity",
        "odds_decimal": f"{prefix}_odds",
        "rank_num": f"{prefix}_finish",
        "target_top3": f"{prefix}_top3",
    }


def _select_cols(df: pd.DataFrame, race_col: str, prefix: str) -> pd.DataFrame:
    cols = [race_col, "horse_name_for_ticket", "ai_rank", "popularity_num", "odds_decimal", "rank_num", "target_top3"]
    return df[cols].rename(columns=_base_horse_cols(prefix))


def _pair_tickets(scored: pd.DataFrame, race_col: str, anchor_role: str, partner_role: str, *, partner_topn: int) -> pd.DataFrame:
    masks = _role_masks(scored)
    anchors = _select_cols(_one_per_race(scored, masks[anchor_role], race_col), race_col, "a")
    partners = _select_cols(_topn_per_race(scored, masks[partner_role], race_col, partner_topn), race_col, "b")
    tickets = anchors.merge(partners, on=race_col, how="inner")
    tickets = tickets[tickets["a_horse"] != tickets["b_horse"]].copy()
    tickets["wide_hit"] = ((tickets["a_top3"] == 1) & (tickets["b_top3"] == 1)).astype(int)
    tickets["umaren_hit"] = ((tickets["a_finish"] <= 2) & (tickets["b_finish"] <= 2)).astype(int)
    tickets["umatan_a_to_b_hit"] = ((tickets["a_finish"] == 1) & (tickets["b_finish"] == 2)).astype(int)
    tickets["umatan_b_to_a_hit"] = ((tickets["b_finish"] == 1) & (tickets["a_finish"] == 2)).astype(int)
    tickets["strategy_detail"] = f"{anchor_role}->{partner_role}_top{partner_topn}"
    return tickets


def _trio_tickets(scored: pd.DataFrame, race_col: str, *, partner_topn: int, fill_topn: int) -> pd.DataFrame:
    masks = _role_masks(scored)
    anchors = _select_cols(_one_per_race(scored, masks["core_anchor"], race_col), race_col, "a")
    partners = _select_cols(_topn_per_race(scored, masks["value_market_gap"] | masks["value_longshot"], race_col, partner_topn), race_col, "b")
    fills = _select_cols(_topn_per_race(scored, masks["solid_fill"] | masks["value_top3_gap"], race_col, fill_topn), race_col, "c")
    tickets = anchors.merge(partners, on=race_col, how="inner").merge(fills, on=race_col, how="inner")
    tickets = tickets[
        (tickets["a_horse"] != tickets["b_horse"])
        & (tickets["a_horse"] != tickets["c_horse"])
        & (tickets["b_horse"] != tickets["c_horse"])
    ].copy()
    key = tickets.apply(lambda r: tuple(sorted([r["a_horse"], r["b_horse"], r["c_horse"]])), axis=1)
    tickets = tickets.loc[~key.duplicated()].copy()
    tickets["trio_hit"] = ((tickets["a_finish"] <= 3) & (tickets["b_finish"] <= 3) & (tickets["c_finish"] <= 3)).astype(int)
    tickets["strategy_detail"] = f"core_anchor-value_top{partner_topn}-fill_top{fill_topn}"
    return tickets


def _trifecta_tickets(scored: pd.DataFrame, race_col: str, *, value_first: bool, partner_topn: int, fill_topn: int) -> pd.DataFrame:
    masks = _role_masks(scored)
    first_role = "value_top3_gap" if value_first else "clear_head"
    firsts = _select_cols(_one_per_race(scored, masks[first_role], race_col), race_col, "a")
    seconds = _select_cols(_topn_per_race(scored, masks["core_anchor"] | masks["value_market_gap"], race_col, partner_topn), race_col, "b")
    thirds = _select_cols(_topn_per_race(scored, masks["solid_fill"] | masks["value_longshot"] | masks["value_market_gap"], race_col, fill_topn), race_col, "c")
    tickets = firsts.merge(seconds, on=race_col, how="inner").merge(thirds, on=race_col, how="inner")
    tickets = tickets[
        (tickets["a_horse"] != tickets["b_horse"])
        & (tickets["a_horse"] != tickets["c_horse"])
        & (tickets["b_horse"] != tickets["c_horse"])
    ].copy()
    tickets["trifecta_hit"] = ((tickets["a_finish"] == 1) & (tickets["b_finish"] == 2) & (tickets["c_finish"] == 3)).astype(int)
    tickets["strategy_detail"] = (
        f"{'value' if value_first else 'clear_head'}_1st-second_top{partner_topn}-third_top{fill_topn}"
    )
    return tickets


def _break_even(hit_rate: float) -> float | None:
    return float(100.0 / hit_rate) if hit_rate > 0 else None


def _load_exotic_payoffs(raw_csv: Path, race_col: str, encoding: str) -> pd.DataFrame:
    usecols = [race_col, "確定着順", "３連複", "３連単"]
    raw = pd.read_csv(raw_csv, encoding=encoding, usecols=lambda c: c in usecols, low_memory=False)
    raw["rank_num"] = _num(raw["確定着順"])
    raw["trio_pay"] = _num(raw["３連複"]).fillna(0.0)
    raw["trifecta_pay"] = _num(raw["３連単"]).fillna(0.0)
    return (
        raw[raw["rank_num"].isin([1, 2, 3])]
        .groupby(race_col, as_index=False)
        .agg(trio_pay=("trio_pay", "max"), trifecta_pay=("trifecta_pay", "max"))
    )


def _pair_metrics(tickets: pd.DataFrame, payoffs: pd.DataFrame, race_col: str, strategy: str, stake: int) -> dict[str, object]:
    if tickets.empty:
        return {"strategy": strategy, "tickets": 0, "races": 0}
    merged = tickets.merge(payoffs, on=race_col, how="left")
    ticket_stake = len(merged) * stake
    umaren_pay = merged["umaren_pay"].fillna(0.0).where(merged["umaren_hit"] == 1, 0.0) * (stake / 100.0)
    umatan_a_pay = merged["umatan_pay"].fillna(0.0).where(merged["umatan_a_to_b_hit"] == 1, 0.0) * (stake / 100.0)
    umatan_b_pay = merged["umatan_pay"].fillna(0.0).where(merged["umatan_b_to_a_hit"] == 1, 0.0) * (stake / 100.0)
    wide_hit_rate = float(merged["wide_hit"].mean())
    return {
        "strategy": strategy,
        "ticket_type": "pair",
        "stake_yen": stake,
        "tickets": int(len(merged)),
        "races": int(merged[race_col].nunique()),
        "avg_partner_popularity": float(merged["b_popularity"].mean()),
        "avg_partner_odds": float(merged["b_odds"].mean()),
        "wide_hit_rate": wide_hit_rate,
        "wide_break_even_avg_pay": _break_even(wide_hit_rate),
        "umaren_hit_rate": float(merged["umaren_hit"].mean()),
        "umaren_roi": float(umaren_pay.sum() / ticket_stake),
        "umaren_profit_yen": float(umaren_pay.sum() - ticket_stake),
        "umatan_a_to_b_hit_rate": float(merged["umatan_a_to_b_hit"].mean()),
        "umatan_a_to_b_roi": float(umatan_a_pay.sum() / ticket_stake),
        "umatan_b_to_a_hit_rate": float(merged["umatan_b_to_a_hit"].mean()),
        "umatan_b_to_a_roi": float(umatan_b_pay.sum() / ticket_stake),
    }


def _combo_metrics(
    tickets: pd.DataFrame,
    payoffs: pd.DataFrame,
    race_col: str,
    strategy: str,
    stake: int,
    hit_col: str,
    ticket_type: str,
) -> dict[str, object]:
    if tickets.empty:
        return {"strategy": strategy, "ticket_type": ticket_type, "tickets": 0, "races": 0}
    merged = tickets.merge(payoffs, on=race_col, how="left")
    hit_rate = float(tickets[hit_col].mean())
    payoff_col = "trio_pay" if ticket_type == "trio" else "trifecta_pay"
    ticket_stake = len(merged) * stake
    actual_pay = merged[payoff_col].fillna(0.0).where(merged[hit_col] == 1, 0.0) * (stake / 100.0)
    return {
        "strategy": strategy,
        "ticket_type": ticket_type,
        "stake_yen": stake,
        "tickets": int(len(merged)),
        "races": int(merged[race_col].nunique()),
        "hit_rate": hit_rate,
        "break_even_avg_pay": _break_even(hit_rate),
        "actual_roi": float(actual_pay.sum() / ticket_stake),
        "actual_profit_yen": float(actual_pay.sum() - ticket_stake),
        "avg_hit_pay": float(merged.loc[merged[hit_col] == 1, payoff_col].mean()) if int(merged[hit_col].sum()) > 0 else None,
        "avg_b_popularity": float(merged["b_popularity"].mean()) if "b_popularity" in merged else None,
        "avg_b_odds": float(merged["b_odds"].mean()) if "b_odds" in merged else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize flexible ticket portfolio from AI roles.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--raw-csv", default="date/raw/全競走馬成績.csv")
    parser.add_argument("--output-dir", default="outputs/analysis/ticket_portfolio_optimizer")
    args = parser.parse_args()

    config = load_json_config(args.config)
    race_col = config["data"]["race_id_column"]
    encoding = config["data"].get("encoding", "cp932")
    df = pd.read_csv(project_path(args.test_csv), low_memory=False)
    scored = _add_model_columns(df, project_path(args.model), race_col)
    payoffs = _load_pair_payoffs(project_path(args.raw_csv), race_col, encoding)
    exotic_payoffs = _load_exotic_payoffs(project_path(args.raw_csv), race_col, encoding)

    rows: list[dict[str, object]] = []
    ticket_frames: list[pd.DataFrame] = []
    pair_specs = [
        ("umaren_core_anchor_to_value_longshot_1pt", "core_anchor", "value_longshot", 1, 300),
        ("umaren_head_anchor_to_value_longshot_1pt", "head_or_anchor", "value_longshot", 1, 200),
        ("umaren_core_anchor_to_value_gap_1pt", "core_anchor", "value_top3_gap", 1, 200),
        ("umaren_core_anchor_to_value_gap_2pt", "core_anchor", "value_market_gap", 2, 100),
        ("umaren_head_anchor_to_pop7_2pt", "head_or_anchor", "value_market_gap", 2, 100),
    ]
    for label, anchor_role, partner_role, topn, stake in pair_specs:
        tickets = _pair_tickets(scored, race_col, anchor_role, partner_role, partner_topn=topn)
        rows.append(_pair_metrics(tickets, payoffs, race_col, label, stake))
        if not tickets.empty:
            frame = tickets.copy()
            frame["strategy"] = label
            frame["ticket_type"] = "pair"
            frame["stake_yen"] = stake
            ticket_frames.append(frame)

    trio_specs = [
        ("trio_core_anchor_value_fill_conservative", 2, 3, 100),
        ("trio_core_anchor_value_fill_wide", 3, 5, 100),
    ]
    for label, partner_topn, fill_topn, stake in trio_specs:
        tickets = _trio_tickets(scored, race_col, partner_topn=partner_topn, fill_topn=fill_topn)
        rows.append(_combo_metrics(tickets, exotic_payoffs, race_col, label, stake, "trio_hit", "trio"))
        if not tickets.empty:
            frame = tickets.copy()
            frame["strategy"] = label
            frame["ticket_type"] = "trio"
            frame["stake_yen"] = stake
            ticket_frames.append(frame)

    trifecta_specs = [
        ("trifecta_clear_head_value_fill", False, 2, 4, 100),
        ("trifecta_value_first_anchor_fill", True, 2, 4, 100),
    ]
    for label, value_first, partner_topn, fill_topn, stake in trifecta_specs:
        tickets = _trifecta_tickets(scored, race_col, value_first=value_first, partner_topn=partner_topn, fill_topn=fill_topn)
        rows.append(_combo_metrics(tickets, exotic_payoffs, race_col, label, stake, "trifecta_hit", "trifecta"))
        if not tickets.empty:
            frame = tickets.copy()
            frame["strategy"] = label
            frame["ticket_type"] = "trifecta"
            frame["stake_yen"] = stake
            ticket_frames.append(frame)

    summary = pd.DataFrame(rows)
    sort_cols = [c for c in ["umaren_roi", "hit_rate", "wide_hit_rate", "tickets"] if c in summary.columns]
    summary = summary.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")

    output_dir = ensure_dir(project_path(args.output_dir))
    summary_path = output_dir / "portfolio_strategy_summary.csv"
    tickets_path = output_dir / "portfolio_ticket_details.csv"
    json_path = output_dir / "summary.json"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    if ticket_frames:
        pd.concat(ticket_frames, ignore_index=True, sort=False).to_csv(tickets_path, index=False, encoding="utf-8-sig")
    payload = {
        "summary_csv": str(summary_path),
        "ticket_details_csv": str(tickets_path),
        "note": "馬連・馬単・三連複・三連単は実配当ROI。ワイドは現CSVに券種配当がないため、的中率と損益分岐平均配当で評価。",
        "top": summary.head(20).to_dict(orient="records"),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
