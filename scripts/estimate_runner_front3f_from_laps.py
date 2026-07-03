from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


RACE_ID_ALIASES = ["race_id", "レースID", "レースID(新/馬番無)", "レースID(新/馬番無)", "レースID(新/馬番有)"]
HORSE_NO_ALIASES = ["horse_no", "馬番", "馬No", "番"]
HORSE_NAME_ALIASES = ["horse_name", "馬名"]
DISTANCE_ALIASES = ["distance", "距離"]
LAP_ALIASES = ["race_laps", "レースラップタイム", "レースラップ", "lap_string"]
FINISH_TIME_ALIASES = ["finish_time", "走破タイム", "タイム", "前走走破タイム"]
FINAL3F_ALIASES = ["final_3f", "上り3F", "上がり3F", "上3F", "前走上り3F"]
FINISH_GAP_ALIASES = ["finish_gap_sec", "着差タイム", "着差秒", "前走着差タイム"]
FINISH_MARGIN_ALIASES = ["着差", "margin"]
FINISH_POS_ALIASES = ["finish_position", "確定着順", "着順"]
ACTUAL_FRONT3F_ALIASES = ["actual_front_3f", "前3F", "テン3F", "前半3F"]
CORNER_ALIASES = {
    "corner1": ["corner1", "1角", "1角通過順", "1C", "前1角"],
    "corner2": ["corner2", "2角", "2角通過順", "2C", "前2角"],
    "corner3": ["corner3", "3角", "3角通過順", "3C", "前3角"],
    "corner4": ["corner4", "4角", "4角通過順", "4C", "前4角", "4角.1", "前4角.1"],
}


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


def pick_col(frame: pd.DataFrame, aliases: list[str]) -> str | None:
    for col in aliases:
        if col in frame.columns:
            return col
    return None


def text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def num(value: object, default: float = np.nan) -> float:
    if value is None or pd.isna(value):
        return default
    raw = str(value).strip()
    if not raw:
        return default
    try:
        return float(raw.replace(",", ""))
    except Exception:
        return default


def parse_laps(value: object) -> list[float]:
    raw = text(value)
    if not raw:
        return []
    values = re.findall(r"\d+(?:\.\d+)?", raw)
    return [float(v) for v in values]


def parse_time_to_seconds(value: object) -> float:
    raw = text(value)
    if not raw:
        return np.nan
    raw = raw.replace(",", "")
    if ":" in raw:
        parts = raw.split(":")
        try:
            return float(parts[-2]) * 60.0 + float(parts[-1])
        except Exception:
            return np.nan
    if re.fullmatch(r"\d{3,4}", raw):
        # TARGET-style compact time, e.g. 1098 => 1:09.8
        try:
            x = int(raw)
            return (x // 1000) * 60.0 + ((x % 1000) / 10.0)
        except Exception:
            return np.nan
    return num(raw)


def parse_margin_to_seconds(value: object, length_seconds: float = 0.17) -> float:
    raw = text(value)
    if not raw:
        return np.nan
    if raw in {"0", "0.0", "同着", "アタマ差なし"}:
        return 0.0
    direct = num(raw)
    if np.isfinite(direct):
        return direct
    table = {
        "ハナ": 0.05,
        "鼻": 0.05,
        "アタマ": 0.10,
        "頭": 0.10,
        "クビ": 0.20,
        "首": 0.20,
        "大差": 3.5,
    }
    if raw in table:
        return table[raw]
    m = re.search(r"(\d+(?:\.\d+)?)", raw)
    if m:
        return float(m.group(1)) * length_seconds
    return np.nan


def parse_int_bytes(raw: bytes) -> int | None:
    value = raw.decode("ascii", errors="ignore").strip()
    if not value or not value.lstrip("+-").isdigit():
        return None
    return int(value)


def decode_cp932(raw: bytes) -> str:
    return raw.decode("cp932", errors="replace").strip().replace("\u3000", "")


def parse_position_bytes(raw: bytes) -> dict[str, int | None]:
    values = {}
    for key, start in (("corner1", 351), ("corner2", 353), ("corner3", 355), ("corner4", 357)):
        values[key] = parse_int_bytes(raw[start : start + 2])
    return values


def parse_su_record(raw: bytes) -> dict[str, object] | None:
    if not raw.startswith(b"SE") or len(raw) < 535:
        return None
    race_id = raw[11:27].decode("ascii", errors="ignore")
    if len(race_id) != 16 or not race_id[:8].isdigit():
        return None
    finish = parse_int_bytes(raw[331:334])
    final3f_raw = parse_int_bytes(raw[390:393])
    margin_raw = parse_int_bytes(raw[531:535])
    row: dict[str, object] = {
        "race_id": race_id,
        "horse_id": raw[30:40].decode("ascii", errors="ignore").strip(),
        "horse_name": decode_cp932(raw[40:76]),
        "finish_position": finish,
        "popularity": parse_int_bytes(raw[363:365]),
        "final_3f": final3f_raw / 10.0 if final3f_raw is not None else np.nan,
        "finish_gap_sec": (margin_raw or 0) / 10.0,
    }
    row.update(parse_position_bytes(raw))
    return row


def extract_su_runners_for_races(target_se_root: Path, race_ids: set[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    race_id_bytes = {race_id.encode("ascii") for race_id in race_ids if len(race_id) == 16}
    years = sorted({race_id[:4] for race_id in race_ids if race_id[:4].isdigit()})
    for year in years:
        year_dir = target_se_root / year
        if not year_dir.exists():
            continue
        for path in sorted(year_dir.glob("SU*.DAT")):
            try:
                records = path.read_bytes().splitlines()
            except OSError:
                continue
            for raw in records:
                if raw[11:27] not in race_id_bytes:
                    continue
                parsed = parse_su_record(raw)
                if parsed:
                    rows.append(parsed)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(["race_id", "horse_id"])


def lap_segment_lengths(distance_m: float | int, laps: list[float]) -> list[float]:
    if not laps:
        return []
    distance = float(distance_m)
    first = distance - 200.0 * (len(laps) - 1)
    if first <= 0 or first > 200:
        first = 200.0
    return [first] + [200.0] * (len(laps) - 1)


def sum_lap_distance(laps: list[float], distance_m: float | int, meters: float, *, from_start: bool) -> float:
    lengths = lap_segment_lengths(distance_m, laps)
    if not lengths:
        return float("nan")
    paired = list(zip(lengths, laps))
    if not from_start:
        paired = list(reversed(paired))
    remaining = float(meters)
    total = 0.0
    for length, lap in paired:
        if remaining <= 0:
            break
        take = min(length, remaining)
        total += float(lap) * (take / length)
        remaining -= take
    return float(total)


@dataclass(frozen=True)
class RaceLapContext:
    race_id: str
    distance_m: int
    laps_200m: list[float]

    @property
    def first3f(self) -> float:
        return sum_lap_distance(self.laps_200m, self.distance_m, 600.0, from_start=True)

    @property
    def last3f(self) -> float:
        return sum_lap_distance(self.laps_200m, self.distance_m, 600.0, from_start=False)

    @property
    def total_time(self) -> float:
        return float(sum(self.laps_200m))

    @property
    def last3f_start_m(self) -> int:
        return max(600, self.distance_m - 600)


def approximate_corner_distance(distance_m: int, corner_name: str) -> float:
    # This is intentionally coarse. Exact corner passage distances differ by
    # venue/course, so these are only priors for the optimizer.
    if corner_name == "corner4":
        return max(600.0, distance_m - 400.0)
    if corner_name == "corner3":
        return max(600.0, distance_m - 700.0)
    if corner_name == "corner2":
        return max(400.0, min(distance_m - 900.0, distance_m * 0.45))
    if corner_name == "corner1":
        return max(300.0, min(distance_m - 1200.0, distance_m * 0.25))
    return np.nan


class RunnerFront3FEstimator:
    def __init__(
        self,
        *,
        length_seconds: float = 0.17,
        rank_gap_seconds: float = 0.15,
        rank_weight: float = 1.4,
        prior_weight: float = 1.0,
        monotonic_weight: float = 4.0,
        use_optimizer: bool = True,
    ) -> None:
        self.length_seconds = length_seconds
        self.rank_gap_seconds = rank_gap_seconds
        self.rank_weight = rank_weight
        self.prior_weight = prior_weight
        self.monotonic_weight = monotonic_weight
        self.use_optimizer = use_optimizer

    def estimate_race(self, race: RaceLapContext, runners: pd.DataFrame) -> pd.DataFrame:
        work = runners.copy()
        finish_times = pd.to_numeric(work["finish_time_sec"], errors="coerce")
        if finish_times.notna().any():
            winner_time = float(finish_times.min())
            finish_gap = finish_times - winner_time
        else:
            finish_gap = pd.to_numeric(work["finish_gap_sec"], errors="coerce")
            winner_time = race.total_time
            finish_times = winner_time + finish_gap
        final3f = pd.to_numeric(work["final_3f_sec"], errors="coerce")
        finish_position = pd.to_numeric(work.get("finish_position"), errors="coerce")
        valid = (
            finish_gap.notna()
            & final3f.notna()
            & finish_gap.between(0.0, 30.0)
            & (finish_position.isna() | finish_position.between(1, 99))
        )
        work = work[valid].copy()
        if work.empty:
            return work
        finish_gap = finish_gap[valid].astype(float).to_numpy()
        final3f = final3f[valid].astype(float).to_numpy()

        # Derived from:
        # runner_final3f = race_last3f + finish_gap(D) - gap(D-600)
        gap_last3_start = finish_gap + race.last3f - final3f
        gap_last3_start = np.maximum(gap_last3_start, 0.0)

        if race.distance_m <= 1200:
            gap_600 = gap_last3_start.copy()
            method = "direct_1200_from_final3f"
        else:
            prior = self._build_gap600_prior(race, work, gap_last3_start)
            if self.use_optimizer:
                gap_600 = self._optimize_gap600(prior, work)
                method = "constrained_rank_prior"
            else:
                gap_600 = prior
                method = "fast_rank_prior"

        front3f = race.first3f + gap_600
        middle_base = race.total_time - race.first3f - race.last3f
        middle_time = middle_base + (gap_last3_start - gap_600)
        reconstructed = front3f + middle_time + final3f

        work["race_first3f_sec"] = race.first3f
        work["race_last3f_sec"] = race.last3f
        work["finish_gap_sec_used"] = finish_gap
        work["gap_last3f_start_sec"] = gap_last3_start
        work["estimated_gap_600m_sec"] = gap_600
        work["estimated_front3f_sec"] = front3f
        work["estimated_middle_sec"] = middle_time
        work["reconstructed_finish_time_sec"] = reconstructed
        work["finish_time_sec_used"] = finish_times[valid].astype(float).to_numpy()
        work["reconstruction_error_sec"] = work["reconstructed_finish_time_sec"] - work["finish_time_sec_used"]
        work["front3f_method"] = method
        work["front3f_confidence"] = self._confidence_label(race, work)
        if "actual_front_3f_sec" in work.columns:
            actual = pd.to_numeric(work["actual_front_3f_sec"], errors="coerce")
            work["front3f_error_vs_actual_sec"] = work["estimated_front3f_sec"] - actual
        return work

    def _build_gap600_prior(self, race: RaceLapContext, runners: pd.DataFrame, gap_last3_start: np.ndarray) -> np.ndarray:
        s_last = float(race.last3f_start_m)
        linear_prior = gap_last3_start * min(1.0, 600.0 / max(600.0, s_last))

        corner_prior = np.full(len(runners), np.nan)
        best_dist = float("inf")
        for corner in ("corner1", "corner2", "corner3", "corner4"):
            if corner not in runners.columns:
                continue
            dist = approximate_corner_distance(race.distance_m, corner)
            if not np.isfinite(dist):
                continue
            d = abs(dist - 600.0)
            if d < best_dist:
                ranks = pd.to_numeric(runners[corner], errors="coerce")
                if ranks.notna().any():
                    corner_prior = ((ranks - 1.0).clip(lower=0.0) * self.rank_gap_seconds).to_numpy(dtype=float)
                    best_dist = d

        if np.isfinite(corner_prior).any():
            closeness = max(0.15, 1.0 - min(best_dist, 900.0) / 900.0)
            prior = (1.0 - closeness) * linear_prior + closeness * np.nan_to_num(corner_prior, nan=np.nanmedian(linear_prior))
        else:
            prior = linear_prior
        return np.maximum(prior, 0.0)

    def _optimize_gap600(self, prior: np.ndarray, runners: pd.DataFrame) -> np.ndarray:
        x0 = np.maximum(prior.astype(float), 0.0)
        early_rank = self._early_rank(runners)
        if len(x0) <= 1:
            return x0
        try:
            from scipy.optimize import minimize
        except Exception:
            return x0

        pairs: list[tuple[int, int]] = []
        if early_rank.notna().sum() >= 2:
            ranks = early_rank.to_numpy(dtype=float)
            order = np.argsort(np.nan_to_num(ranks, nan=999.0))
            for a, b in zip(order[:-1], order[1:]):
                if np.isfinite(ranks[a]) and np.isfinite(ranks[b]) and ranks[a] < ranks[b]:
                    pairs.append((int(a), int(b)))

        def objective(x: np.ndarray) -> float:
            prior_loss = self.prior_weight * float(np.sum((x - x0) ** 2))
            rank_loss = 0.0
            if early_rank.notna().any():
                rank_prior = ((early_rank.fillna(early_rank.median()) - 1.0).clip(lower=0.0) * self.rank_gap_seconds).to_numpy()
                rank_loss = self.rank_weight * float(np.sum((x - rank_prior) ** 2))
            mono_loss = 0.0
            for a, b in pairs:
                mono_loss += max(0.0, x[a] - x[b] + 0.03) ** 2
            return prior_loss + rank_loss + self.monotonic_weight * mono_loss

        bounds = [(0.0, max(6.0, float(v) + 3.0)) for v in x0]
        result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 400})
        if not result.success:
            return x0
        return np.maximum(result.x, 0.0)

    def _early_rank(self, runners: pd.DataFrame) -> pd.Series:
        for col in ("corner1", "corner2", "corner3", "corner4"):
            if col in runners.columns:
                s = pd.to_numeric(runners[col], errors="coerce")
                if s.notna().any():
                    return s
        return pd.Series(np.nan, index=runners.index, dtype=float)

    def _confidence_label(self, race: RaceLapContext, runners: pd.DataFrame) -> str:
        if race.distance_m <= 1200:
            return "high"
        if race.distance_m <= 1400:
            return "medium_high"
        if any(col in runners.columns and pd.to_numeric(runners[col], errors="coerce").notna().any() for col in ("corner1", "corner2")):
            return "medium"
        return "low"


def normalize_runners(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    mapping = {
        "race_id": RACE_ID_ALIASES,
        "horse_no": HORSE_NO_ALIASES,
        "horse_name": HORSE_NAME_ALIASES,
        "finish_position": FINISH_POS_ALIASES,
    }
    for target, aliases in mapping.items():
        col = pick_col(frame, aliases)
        out[target] = frame[col] if col else pd.NA
    finish_time_col = pick_col(frame, FINISH_TIME_ALIASES)
    final3f_col = pick_col(frame, FINAL3F_ALIASES)
    finish_gap_col = pick_col(frame, FINISH_GAP_ALIASES)
    finish_margin_col = pick_col(frame, FINISH_MARGIN_ALIASES)
    actual_front_col = pick_col(frame, ACTUAL_FRONT3F_ALIASES)
    out["finish_time_sec"] = frame[finish_time_col].map(parse_time_to_seconds) if finish_time_col else np.nan
    out["final_3f_sec"] = pd.to_numeric(frame[final3f_col], errors="coerce") if final3f_col else np.nan
    if finish_gap_col:
        out["finish_gap_sec"] = pd.to_numeric(frame[finish_gap_col], errors="coerce")
    elif finish_margin_col:
        out["finish_gap_sec"] = frame[finish_margin_col].map(parse_margin_to_seconds)
    else:
        out["finish_gap_sec"] = np.nan
    if actual_front_col:
        out["actual_front_3f_sec"] = pd.to_numeric(frame[actual_front_col], errors="coerce")
    for target, aliases in CORNER_ALIASES.items():
        col = pick_col(frame, aliases)
        if col:
            out[target] = pd.to_numeric(frame[col], errors="coerce")
    return out


def normalize_laps(frame: pd.DataFrame) -> pd.DataFrame:
    race_col = pick_col(frame, RACE_ID_ALIASES)
    lap_col = pick_col(frame, LAP_ALIASES)
    dist_col = pick_col(frame, DISTANCE_ALIASES)
    if race_col is None or lap_col is None:
        raise ValueError("lap CSV requires race_id and race_laps/レースラップタイム columns")
    out = pd.DataFrame()
    out["race_id"] = frame[race_col].astype(str)
    out["laps_200m"] = frame[lap_col].map(parse_laps)
    if dist_col:
        out["distance_m"] = pd.to_numeric(frame[dist_col], errors="coerce")
    else:
        out["distance_m"] = out["laps_200m"].map(lambda laps: len(laps) * 200)
    out = out[out["laps_200m"].map(len).ge(3) & out["distance_m"].notna()].copy()
    out["distance_m"] = out["distance_m"].astype(int)
    return out


def estimate_front3f(
    lap_csv: Path,
    runner_csv: Path | None,
    output_csv: Path,
    *,
    target_se_root: Path | None = None,
) -> dict[str, Any]:
    laps = normalize_laps(read_csv_any(lap_csv))
    if runner_csv is not None:
        runners_source = read_csv_any(runner_csv)
        runner_source_label = str(runner_csv)
    elif target_se_root is not None:
        race_ids = set(laps["race_id"].astype(str))
        runners_source = extract_su_runners_for_races(target_se_root, race_ids)
        runner_source_label = str(target_se_root)
    else:
        raise ValueError("Either runner_csv or target_se_root is required.")
    runners = normalize_runners(runners_source)
    runners["race_id"] = runners["race_id"].astype(str)
    estimator = RunnerFront3FEstimator()

    frames: list[pd.DataFrame] = []
    for _, race_row in laps.iterrows():
        race_id = str(race_row["race_id"])
        part = runners[runners["race_id"].eq(race_id)].copy()
        if part.empty:
            continue
        race = RaceLapContext(
            race_id=race_id,
            distance_m=int(race_row["distance_m"]),
            laps_200m=list(race_row["laps_200m"]),
        )
        estimated = estimator.estimate_race(race, part)
        if not estimated.empty:
            estimated["race_id"] = race_id
            estimated["distance_m"] = race.distance_m
            first_cols = ["race_id", "distance_m"]
            estimated = estimated[first_cols + [c for c in estimated.columns if c not in first_cols]]
            frames.append(estimated)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    summary: dict[str, Any] = {
        "lap_csv": str(lap_csv),
        "runner_source": runner_source_label,
        "output_csv": str(output_csv),
        "races": int(out["race_id"].nunique()) if not out.empty and "race_id" in out else 0,
        "rows": int(len(out)),
        "runner_source_rows": int(len(runners_source)),
        "mean_abs_reconstruction_error": float(out["reconstruction_error_sec"].abs().mean()) if not out.empty else None,
    }
    if "front3f_error_vs_actual_sec" in out.columns and out["front3f_error_vs_actual_sec"].notna().any():
        err = out["front3f_error_vs_actual_sec"].dropna()
        summary["actual_front3f_compare_rows"] = int(len(err))
        summary["actual_front3f_mae"] = float(err.abs().mean())
        summary["actual_front3f_bias"] = float(err.mean())
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Estimate each runner's front 3F from official race laps, finish time/gap, final 3F, and corner ranks."
    )
    parser.add_argument("--lap-csv", required=True, help="CSV with race_id and official 200m lap string.")
    parser.add_argument("--runner-csv", default="", help="CSV with race_id, finish time or margin, final 3F, and optional corners.")
    parser.add_argument(
        "--target-se-root",
        default="",
        help="Optional TARGET Data Lab SE_DATA root. If set and --runner-csv is omitted, SU*.DAT is used for runner results.",
    )
    parser.add_argument("--output-csv", default="outputs/analysis/front3f_estimation/estimated_runner_front3f.csv")
    args = parser.parse_args()

    runner_csv = project_path(args.runner_csv) if args.runner_csv else None
    target_se_root = project_path(args.target_se_root) if args.target_se_root else None
    summary = estimate_front3f(
        project_path(args.lap_csv),
        runner_csv,
        project_path(args.output_csv),
        target_se_root=target_se_root,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
