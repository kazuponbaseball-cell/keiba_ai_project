from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTED = ROOT / "outputs/analysis/lap_positive_expansion_v1/lap_positive_expansion_selected_tickets.csv"
DEFAULT_RUNNERS = ROOT / "data/datasets/cache/lap_pedigree_interactions_confirmed_opponent/test_features.csv"
DEFAULT_OUT = ROOT / "outputs/analysis/lap_1win_transition_segments_v1"

COL_DATE = "\u65e5\u4ed8"
COL_RACE_ID = "\u30ec\u30fc\u30b9ID(\u65b0/\u99ac\u756a\u7121)"
COL_HORSE_NO = "\u99ac\u756a"
COL_HORSE_ID = "\u8840\u7d71\u767b\u9332\u756a\u53f7"
COL_HORSE_NAME = "\u99ac\u540d"
COL_CLASS = "\u30af\u30e9\u30b9\u540d"
COL_PREV_CLASS = "\u524d\u30af\u30e9\u30b9\u540d"
COL_AGE = "\u5e74\u9f62"
COL_SEX = "\u6027\u5225"
COL_CAREER = "\u30ad\u30e3\u30ea\u30a2"
COL_SURFACE = "\u829d\u30fb\u30c0"
COL_DISTANCE = "\u8ddd\u96e2"
COL_MARGIN = "\u7740\u5dee"
COL_POP = "\u4eba\u6c17"
COL_4C = "4\u89d2"
COL_AGARI_RANK = "\u4e0a\u308a3F\u9806"
COL_PCI = "PCI"
COL_PCI3 = "PCI3"
COL_RPCI = "RPCI"
COL_PREV_SURFACE = "\u524d\u829d\u30fb\u30c0"
COL_PREV_DISTANCE = "\u524d\u8ddd\u96e2"
COL_PREV_MARGIN = "\u524d\u8d70\u7740\u5dee\u30bf\u30a4\u30e0"
COL_PREV_POP = "\u524d\u8d70\u4eba\u6c17"
COL_PREV_4C = "\u524d4\u89d2"
COL_PREV_AGARI_RANK = "\u524d\u8d70\u4e0a\u308a3F\u9806"
COL_PREV_PCI = "\u524dPCI"
COL_PREV_PCI3 = "\u524d\u8d70PCI3"
COL_PREV_RPCI = "\u524d\u8d70RPCI"
COL_PREV_RACE_ID = "\u524d\u8d70\u30ec\u30fc\u30b9ID(\u65b0/\u99ac\u756a\u7121)"

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
    "wide_quote_proxy",
    "umaren_quote_proxy",
    "odds_geom",
    "queue_shape_label",
    "v2_predicted_lap_mode",
    "v2_confidence",
    "pair_lap_mismatch_popular_max",
    "lap_positive_score",
    "lap_axis_specialist_role_score",
    "lap_expansion_candidate_label",
]

RUNNER_COLS = [
    COL_DATE,
    COL_RACE_ID,
    COL_HORSE_NO,
    COL_HORSE_ID,
    COL_HORSE_NAME,
    COL_CLASS,
    COL_PREV_CLASS,
    COL_AGE,
    COL_SEX,
    COL_CAREER,
    COL_SURFACE,
    COL_DISTANCE,
    COL_MARGIN,
    COL_POP,
    COL_4C,
    COL_AGARI_RANK,
    COL_PCI,
    COL_PCI3,
    COL_RPCI,
    COL_PREV_SURFACE,
    COL_PREV_DISTANCE,
    COL_PREV_MARGIN,
    COL_PREV_POP,
    COL_PREV_4C,
    COL_PREV_AGARI_RANK,
    COL_PREV_PCI,
    COL_PREV_PCI3,
    COL_PREV_RPCI,
    COL_PREV_RACE_ID,
]


def available_usecols(path: Path, wanted: list[str]) -> list[str]:
    header = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    return [c for c in wanted if c in header.columns]


def clean_race_id(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True).str.strip().str.zfill(16)


def num(series: pd.Series | None, index: pd.Index, default: float = np.nan) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce")


def txt(series: pd.Series | None, index: pd.Index, default: str = "") -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype="string")
    return series.astype("string").fillna(default).str.strip()


def class_rank(value: Any) -> float:
    s = str(value).strip()
    mapping = {
        "\u65b0\u99ac": 0,
        "\u672a\u52dd\u5229": 1,
        "1\u52dd": 2,
        "2\u52dd": 3,
        "3\u52dd": 4,
        "\uff75\uff70\uff8c\uff9f\uff9d": 5,
        "OP(L)": 5,
        "\uff27\uff13": 6,
        "\uff27\uff12": 7,
        "\uff27\uff11": 8,
    }
    return float(mapping.get(s, np.nan))


def is_blank_text(series: pd.Series) -> pd.Series:
    text = series.astype("string")
    lowered = text.str.strip().str.lower()
    return text.isna() | lowered.isin(["", "nan", "<na>", "none"])


def side_transition(curr: Any, prev: Any) -> str:
    curr_s = str(curr).strip()
    prev_s = str(prev).strip()
    if curr_s != "1\u52dd":
        return "not_1win"
    if prev_s in {"\u672a\u52dd\u5229", "\u65b0\u99ac"}:
        return "up_from_maiden"
    if prev_s == "1\u52dd":
        return "stay_1win"
    cr = class_rank(curr_s)
    pr = class_rank(prev_s)
    if np.isfinite(pr) and np.isfinite(cr):
        if pr > cr:
            return "down_from_higher"
        if pr < cr:
            return "up_other"
    return "unknown_prev_class"


def pair_label(a: pd.Series, b: pd.Series, name: str) -> pd.Series:
    av = a.astype("string").fillna("")
    bv = b.astype("string").fillna("")
    return (name + ":" + av + "+" + bv).astype("string")


def pair_transition_label(a: pd.Series, b: pd.Series) -> pd.Series:
    av = a.astype("string").fillna("unknown")
    bv = b.astype("string").fillna("unknown")
    labels = pd.Series("other", index=a.index, dtype="string")
    labels = labels.mask(av.eq("stay_1win") & bv.eq("stay_1win"), "both_stay_1win")
    labels = labels.mask(av.eq("up_from_maiden") & bv.eq("up_from_maiden"), "both_up_from_maiden")
    labels = labels.mask(
        (av.eq("up_from_maiden") & bv.eq("stay_1win")) | (av.eq("stay_1win") & bv.eq("up_from_maiden")),
        "up_from_maiden_plus_stay",
    )
    labels = labels.mask(av.eq("up_from_maiden") | bv.eq("up_from_maiden"), "has_up_from_maiden")
    labels = labels.mask(av.eq("down_from_higher") | bv.eq("down_from_higher"), "has_down_from_higher")
    labels = labels.mask(av.eq("unknown_prev_class") | bv.eq("unknown_prev_class"), "has_unknown_prev")
    return labels


def class_move_bucket(curr: pd.Series, prev: pd.Series) -> pd.Series:
    cr = curr.map(class_rank)
    pr = prev.map(class_rank)
    diff = cr - pr
    out = pd.Series("unknown", index=curr.index, dtype="string")
    out = out.mask(diff.eq(0), "same_class")
    out = out.mask(diff.gt(0), "class_up")
    out = out.mask(diff.lt(0), "class_down")
    return out


def age_pair_bucket(a: pd.Series, b: pd.Series) -> pd.Series:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    out = pd.Series("unknown", index=a.index, dtype="string")
    out = out.mask(aa.eq(3) & bb.eq(3), "both_3yo")
    out = out.mask((aa.eq(3) | bb.eq(3)) & ~out.eq("both_3yo"), "has_3yo")
    out = out.mask(aa.ge(4) & bb.ge(4), "older_only")
    return out


def career_pair_bucket(a: pd.Series, b: pd.Series) -> pd.Series:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    mn = pd.concat([aa, bb], axis=1).min(axis=1)
    mx = pd.concat([aa, bb], axis=1).max(axis=1)
    out = pd.Series("unknown", index=a.index, dtype="string")
    out = out.mask(mx.le(5), "both_lightly_raced")
    out = out.mask(mn.le(5) & ~out.eq("both_lightly_raced"), "has_lightly_raced")
    out = out.mask(mn.ge(10), "both_exposed")
    out = out.mask(mx.ge(10) & mn.lt(10) & ~out.isin(["both_lightly_raced", "has_lightly_raced"]), "mixed_experience")
    return out


def surface_switch_pair(curr_a: pd.Series, prev_a: pd.Series, curr_b: pd.Series, prev_b: pd.Series) -> pd.Series:
    ca = curr_a.astype("string").fillna("").str.strip()
    pa = prev_a.astype("string").fillna("").str.strip()
    cb = curr_b.astype("string").fillna("").str.strip()
    pb = prev_b.astype("string").fillna("").str.strip()
    a_switch = ca.ne(pa)
    b_switch = cb.ne(pb)
    a_known = ca.ne("") & pa.ne("")
    b_known = cb.ne("") & pb.ne("")
    out = pd.Series("unknown", index=curr_a.index, dtype="string")
    out = out.mask(a_known & b_known & ~a_switch & ~b_switch, "both_same_surface")
    out = out.mask(a_known & b_known & a_switch & b_switch, "both_surface_switch")
    out = out.mask(a_known & b_known & (a_switch | b_switch) & ~out.eq("both_surface_switch"), "has_surface_switch")
    return out


def distance_change_bucket(curr_a: pd.Series, prev_a: pd.Series, curr_b: pd.Series, prev_b: pd.Series) -> pd.Series:
    da = pd.to_numeric(curr_a, errors="coerce") - pd.to_numeric(prev_a, errors="coerce")
    db = pd.to_numeric(curr_b, errors="coerce") - pd.to_numeric(prev_b, errors="coerce")
    out = pd.Series("unknown", index=curr_a.index, dtype="string")
    both_known = da.notna() & db.notna()
    shortening = da.le(-200) | db.le(-200)
    extension = da.ge(200) | db.ge(200)
    same = da.abs().le(100) & db.abs().le(100)
    out = out.mask(both_known & same, "both_similar_distance")
    out = out.mask(both_known & shortening & extension, "mixed_shortening_extension")
    out = out.mask(both_known & shortening & ~extension, "has_shortening")
    out = out.mask(both_known & extension & ~shortening, "has_extension")
    return out


def prev_lap_bucket(pci_a: pd.Series, pci_b: pd.Series, rpci_a: pd.Series, rpci_b: pd.Series) -> pd.Series:
    pci_mean = pd.concat([pd.to_numeric(pci_a, errors="coerce"), pd.to_numeric(pci_b, errors="coerce")], axis=1).mean(axis=1)
    rpci_mean = pd.concat([pd.to_numeric(rpci_a, errors="coerce"), pd.to_numeric(rpci_b, errors="coerce")], axis=1).mean(axis=1)
    out = pd.Series("unknown", index=pci_a.index, dtype="string")
    out = out.mask(rpci_mean.le(45), "prev_fast_or_tough")
    out = out.mask(rpci_mean.ge(50), "prev_slow_or_instant")
    out = out.mask(pci_mean.between(45, 50, inclusive="both") & out.eq("unknown"), "prev_middle")
    return out


def read_selected(path: Path) -> pd.DataFrame:
    usecols = available_usecols(path, SELECTED_COLS)
    df = pd.read_csv(path, usecols=usecols, encoding="utf-8-sig", low_memory=False)
    df["race_id"] = clean_race_id(df["race_id"])
    return df


def runner_source_paths(path: Path) -> list[Path]:
    names = ["train_features.csv", "test_features.csv"]
    paths = [path]
    for name in names:
        candidate = path.parent / name
        if candidate.exists():
            paths.append(candidate)
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen and p.exists():
            seen.add(rp)
            out.append(p)
    return out


def fill_prev_from_horse_history(df: pd.DataFrame) -> pd.DataFrame:
    if COL_HORSE_ID not in df.columns or COL_DATE not in df.columns:
        return df
    out = df.copy()
    out["_horse_sort"] = pd.to_numeric(out[COL_HORSE_ID], errors="coerce")
    out["_date_sort"] = pd.to_numeric(out[COL_DATE], errors="coerce")
    out["_race_sort"] = pd.to_numeric(out[COL_RACE_ID], errors="coerce")
    out["_horse_no_sort"] = pd.to_numeric(out[COL_HORSE_NO], errors="coerce")
    out = out.sort_values(["_horse_sort", "_date_sort", "_race_sort", "_horse_no_sort"], kind="mergesort")
    prev_pairs = [
        (COL_PREV_CLASS, COL_CLASS),
        (COL_PREV_SURFACE, COL_SURFACE),
        (COL_PREV_DISTANCE, COL_DISTANCE),
        (COL_PREV_MARGIN, COL_MARGIN),
        (COL_PREV_POP, COL_POP),
        (COL_PREV_4C, COL_4C),
        (COL_PREV_AGARI_RANK, COL_AGARI_RANK),
        (COL_PREV_PCI, COL_PCI),
        (COL_PREV_PCI3, COL_PCI3),
        (COL_PREV_RPCI, COL_RPCI),
        (COL_PREV_RACE_ID, COL_RACE_ID),
    ]
    grouped = out.groupby(COL_HORSE_ID, dropna=False, sort=False)
    for prev_col, current_col in prev_pairs:
        if current_col not in out.columns:
            continue
        derived = grouped[current_col].shift(1)
        if prev_col not in out.columns:
            out[prev_col] = derived
            continue
        existing = out[prev_col]
        if existing.dtype == object or str(existing.dtype).startswith("string"):
            blank = is_blank_text(existing)
        else:
            blank = existing.isna()
        out[prev_col] = existing.where(~blank, derived)
    return out.drop(columns=["_horse_sort", "_date_sort", "_race_sort", "_horse_no_sort"], errors="ignore")


def read_runners(path: Path) -> pd.DataFrame:
    frames = []
    for source in runner_source_paths(path):
        usecols = available_usecols(source, RUNNER_COLS)
        if not usecols:
            continue
        frames.append(pd.read_csv(source, usecols=usecols, encoding="utf-8-sig", low_memory=False))
    if not frames:
        return pd.DataFrame(columns=["race_id", "horse_no"])
    df = pd.concat(frames, ignore_index=True)
    df["race_id"] = clean_race_id(df[COL_RACE_ID])
    df["horse_no"] = pd.to_numeric(df[COL_HORSE_NO], errors="coerce").astype("Int64")
    df = fill_prev_from_horse_history(df)
    return df.drop_duplicates(["race_id", "horse_no"], keep="last")


def merge_side(pairs: pd.DataFrame, runners: pd.DataFrame, side: str, horse_col: str) -> pd.DataFrame:
    rename = {
        c: f"{side}_{c}"
        for c in runners.columns
        if c not in {"race_id", "horse_no", COL_RACE_ID, COL_HORSE_NO}
    }
    side_df = runners.rename(columns=rename)
    side_df[f"{side}_horse_no"] = side_df["horse_no"]
    out = pairs.copy()
    out[horse_col] = pd.to_numeric(out[horse_col], errors="coerce").astype("Int64")
    out = out.merge(side_df.drop(columns=[COL_RACE_ID, COL_HORSE_NO], errors="ignore"), left_on=["race_id", horse_col], right_on=["race_id", "horse_no"], how="left")
    out = out.drop(columns=["horse_no"], errors="ignore")
    return out


def enrich(pairs: pd.DataFrame, runners: pd.DataFrame) -> pd.DataFrame:
    out = merge_side(pairs, runners, "a", "horse_a")
    out = merge_side(out, runners, "b", "horse_b")
    idx = out.index
    out["class_eval"] = txt(out.get(f"a_{COL_CLASS}"), idx).mask(txt(out.get(f"a_{COL_CLASS}"), idx).eq(""), txt(out.get(f"b_{COL_CLASS}"), idx))
    out["a_transition"] = [side_transition(c, p) for c, p in zip(txt(out.get(f"a_{COL_CLASS}"), idx), txt(out.get(f"a_{COL_PREV_CLASS}"), idx))]
    out["b_transition"] = [side_transition(c, p) for c, p in zip(txt(out.get(f"b_{COL_CLASS}"), idx), txt(out.get(f"b_{COL_PREV_CLASS}"), idx))]
    out["pair_1win_transition"] = pair_transition_label(out["a_transition"], out["b_transition"])
    out["a_class_move"] = class_move_bucket(txt(out.get(f"a_{COL_CLASS}"), idx), txt(out.get(f"a_{COL_PREV_CLASS}"), idx))
    out["b_class_move"] = class_move_bucket(txt(out.get(f"b_{COL_CLASS}"), idx), txt(out.get(f"b_{COL_PREV_CLASS}"), idx))
    out["pair_class_move"] = pair_label(out["a_class_move"], out["b_class_move"], "move")
    out["pair_age_bucket"] = age_pair_bucket(out.get(f"a_{COL_AGE}"), out.get(f"b_{COL_AGE}"))
    out["pair_career_bucket"] = career_pair_bucket(out.get(f"a_{COL_CAREER}"), out.get(f"b_{COL_CAREER}"))
    out["pair_surface_switch"] = surface_switch_pair(
        txt(out.get(f"a_{COL_SURFACE}"), idx),
        txt(out.get(f"a_{COL_PREV_SURFACE}"), idx),
        txt(out.get(f"b_{COL_SURFACE}"), idx),
        txt(out.get(f"b_{COL_PREV_SURFACE}"), idx),
    )
    out["pair_distance_change"] = distance_change_bucket(
        out.get(f"a_{COL_DISTANCE}"),
        out.get(f"a_{COL_PREV_DISTANCE}"),
        out.get(f"b_{COL_DISTANCE}"),
        out.get(f"b_{COL_PREV_DISTANCE}"),
    )
    out["prev_lap_bucket"] = prev_lap_bucket(
        out.get(f"a_{COL_PREV_PCI}"),
        out.get(f"b_{COL_PREV_PCI}"),
        out.get(f"a_{COL_PREV_RPCI}"),
        out.get(f"b_{COL_PREV_RPCI}"),
    )
    out["prev_margin_best"] = pd.concat(
        [pd.to_numeric(out.get(f"a_{COL_PREV_MARGIN}"), errors="coerce"), pd.to_numeric(out.get(f"b_{COL_PREV_MARGIN}"), errors="coerce")],
        axis=1,
    ).min(axis=1)
    out["prev_margin_pair_bucket"] = pd.cut(
        out["prev_margin_best"],
        bins=[-np.inf, 0.0, 0.3, 0.8, np.inf],
        labels=["has_prev_win_or_equal", "has_close_prev", "has_ok_prev", "no_recent_content"],
        right=True,
    ).astype("string").fillna("unknown")
    out["lap_mode"] = txt(out.get("v2_predicted_lap_mode"), idx, "unknown")
    out["mismatch_band"] = pd.cut(
        pd.to_numeric(out.get("pair_lap_mismatch_popular_max"), errors="coerce"),
        bins=[-0.01, 0.20, 0.30, 0.40, 1.0],
        labels=["low", "ok", "warn", "high"],
        right=True,
    ).astype("string").fillna("unknown")
    wide_quote = pd.to_numeric(out.get("wide_quote_proxy"), errors="coerce")
    umaren_quote = pd.to_numeric(out.get("umaren_quote_proxy"), errors="coerce")
    odds_geom = pd.to_numeric(out.get("odds_geom"), errors="coerce")
    out["pair_odds_proxy"] = np.where(out["ticket_type"].astype(str).eq("wide"), wide_quote, umaren_quote)
    out["pair_odds_proxy"] = pd.Series(out["pair_odds_proxy"], index=idx).fillna(odds_geom)
    out["pair_odds_proxy"] = out["pair_odds_proxy"].where(out["pair_odds_proxy"].le(100), out["pair_odds_proxy"] / 100.0)
    out["pair_odds_band"] = pd.cut(
        out["pair_odds_proxy"],
        bins=[0, 3, 5, 10, 20, 50, np.inf],
        labels=["<=3", "3-5", "5-10", "10-20", "20-50", "50+"],
        right=True,
    ).astype("string").fillna("unknown")
    return out


def max_drawdown(profit: pd.Series) -> float:
    if profit.empty:
        return 0.0
    curve = pd.to_numeric(profit, errors="coerce").fillna(0.0).cumsum()
    return float((curve - curve.cummax()).min())


def roi_without(frame: pd.DataFrame, n: int) -> float:
    kept = frame.sort_values("return_yen", ascending=False).iloc[n:]
    stake = pd.to_numeric(kept["stake_yen"], errors="coerce").fillna(0.0).sum()
    ret = pd.to_numeric(kept["return_yen"], errors="coerce").fillna(0.0).sum()
    return float(ret / stake) if stake > 0 else 0.0


def metrics(frame: pd.DataFrame, policy: str, segment: str, value: str) -> dict[str, Any]:
    if frame.empty:
        return {}
    stake = pd.to_numeric(frame["stake_yen"], errors="coerce").fillna(0.0)
    ret = pd.to_numeric(frame["return_yen"], errors="coerce").fillna(0.0)
    hit = frame["hit"].astype(bool)
    profit = ret - stake
    ordered = frame.assign(_profit=profit).sort_values(["race_id", "ticket_type", "horse_a", "horse_b"], kind="mergesort")
    year_parts = []
    for year, yg in frame.groupby("year", dropna=False):
        if len(yg) < 5:
            continue
        st = pd.to_numeric(yg["stake_yen"], errors="coerce").fillna(0.0).sum()
        rt = pd.to_numeric(yg["return_yen"], errors="coerce").fillna(0.0).sum()
        year_parts.append((str(year), len(yg), float(rt / st) if st else 0.0))
    return {
        "policy": policy,
        "segment": segment,
        "segment_value": value,
        "tickets": int(len(frame)),
        "races": int(frame["race_id"].nunique()),
        "stake_yen": float(stake.sum()),
        "return_yen": float(ret.sum()),
        "profit_yen": float(profit.sum()),
        "roi": float(ret.sum() / stake.sum()) if float(stake.sum()) > 0 else 0.0,
        "hit_rate": float(hit.mean()),
        "max_drawdown_yen": max_drawdown(ordered["_profit"]),
        "top1_removed_roi": roi_without(frame, 1),
        "top3_removed_roi": roi_without(frame, 3),
        "top5_removed_roi": roi_without(frame, 5),
        "year_buckets_ge5": len(year_parts),
        "min_year_roi_ge5": float(min([x[2] for x in year_parts])) if year_parts else np.nan,
        "profitable_year_buckets": int(sum(x[2] >= 1.0 for x in year_parts)),
        **{f"roi_{year}": roi for year, _, roi in year_parts},
        **{f"tickets_{year}": tickets for year, tickets, _ in year_parts},
    }


def segment_table(df: pd.DataFrame, cols: list[str], min_tickets: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy, part in df.groupby("policy", sort=False):
        for col in cols:
            for value, group in part.groupby(col, dropna=False, sort=False):
                if len(group) < min_tickets:
                    continue
                rows.append(metrics(group, str(policy), col, str(value)))
        for a, b in combinations(cols, 2):
            for values, group in part.groupby([a, b], dropna=False, sort=False):
                if len(group) < min_tickets:
                    continue
                value = f"{a}={values[0]} & {b}={values[1]}"
                rows.append(metrics(group, str(policy), f"{a}+{b}", value))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["roi", "tickets"], ascending=[False, False])


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze why lap-positive expansion works in 1-win class.")
    parser.add_argument("--selected-csv", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--runner-csv", type=Path, default=DEFAULT_RUNNERS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-tickets", type=int, default=20)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected = read_selected(args.selected_csv)
    selected = selected[selected["policy"].astype(str).str.contains("extra_only", na=False)].copy()
    runners = read_runners(args.runner_csv)
    enriched = enrich(selected, runners)
    one_win = enriched[enriched["class_eval"].astype(str).eq("1\u52dd")].copy()

    cols = [
        "pair_1win_transition",
        "pair_age_bucket",
        "pair_career_bucket",
        "pair_surface_switch",
        "pair_distance_change",
        "prev_lap_bucket",
        "prev_margin_pair_bucket",
        "lap_mode",
        "mismatch_band",
        "pair_odds_band",
    ]
    segments = segment_table(one_win, cols, args.min_tickets)
    enriched.to_csv(args.out_dir / "lap_1win_transition_enriched_tickets.csv", index=False, encoding="utf-8-sig")
    one_win.to_csv(args.out_dir / "lap_1win_only_enriched_tickets.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(args.out_dir / "lap_1win_transition_segment_metrics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "selected_rows": int(len(selected)),
        "enriched_rows": int(len(enriched)),
        "one_win_rows": int(len(one_win)),
        "one_win_races": int(one_win["race_id"].nunique()),
        "top_segments": segments.head(30).to_dict(orient="records") if not segments.empty else [],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out_dir}")
    print(json.dumps({k: v for k, v in summary.items() if k != "top_segments"}, ensure_ascii=False, indent=2))
    if not segments.empty:
        print(segments.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
