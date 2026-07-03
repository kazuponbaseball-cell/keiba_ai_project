from __future__ import annotations

import argparse
import glob
import html
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def read_csv_safe(path: Path) -> pd.DataFrame:
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def text(value: object, default: str = "") -> str:
    if value is None or pd.isna(value):
        return default
    return str(value)


def number(value: object, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value) or text(value).strip() == "":
            return default
        return float(value)
    except Exception:
        return default


def escape(value: object) -> str:
    return html.escape(text(value))


def fmt_int(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        return str(int(float(value)))
    except Exception:
        return text(value)


def fmt_date(value: object) -> str:
    raw = text(value).strip()
    if not raw:
        return ""
    for fmt in ("%Y.%m.%d", "%Y-%m-%d", "%Y%m%d"):
        try:
            dt = datetime.strptime(raw, fmt)
            return f"{dt.year}年{dt.month}月{dt.day}日"
        except Exception:
            pass
    if raw.isdigit() and len(raw) == 6:
        try:
            dt = datetime.strptime("20" + raw, "%Y%m%d")
            return f"{dt.year}年{dt.month}月{dt.day}日"
        except Exception:
            pass
    return raw


def date_key(value: object) -> str:
    raw = text(value).strip()
    if not raw:
        return ""
    for fmt in ("%Y.%m.%d", "%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y%m%d")
        except Exception:
            pass
    if raw.isdigit() and len(raw) == 6:
        return "20" + raw
    return raw


def pick_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def norm_id(value: object) -> str:
    raw = text(value).strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    try:
        return str(int(float(raw)))
    except Exception:
        return raw


def has_value(value: object) -> bool:
    raw = text(value).strip()
    return bool(raw and raw.lower() not in {"nan", "none", "<na>"})


def valid_odds(value: object) -> float | None:
    try:
        if value is None or pd.isna(value) or text(value).strip() == "":
            return None
        parsed = float(str(value).replace(",", "").strip())
    except Exception:
        return None
    if 1.0 <= parsed < 999.0:
        return parsed
    return None


def first_valid_odds(row: pd.Series, names: list[str]) -> float | None:
    for name in names:
        if name in row.index:
            odds = valid_odds(row.get(name))
            if odds is not None:
                return odds
    return None


def target_dates(today: datetime) -> list[str]:
    return [(today + timedelta(days=1)).strftime("%Y%m%d"), (today + timedelta(days=2)).strftime("%Y%m%d")]


def date_col(df: pd.DataFrame) -> str | None:
    return pick_col(df, ["日付S", "日付", "date", "年月日"])


def race_id_col(df: pd.DataFrame) -> str | None:
    return pick_col(df, ["レースID(新/馬番無)", "race_id"])


def horse_no_col(df: pd.DataFrame) -> str | None:
    return pick_col(df, ["馬番", "horse_no"])


def frame_col(df: pd.DataFrame) -> str | None:
    return pick_col(df, ["枠番", "frame_no"])


def snapshot_dates(df: pd.DataFrame) -> list[str]:
    col = date_col(df)
    if df.empty or not col:
        return []
    return sorted({date_key(v) for v in df[col].dropna().unique() if date_key(v)})


def has_target_dates(df: pd.DataFrame, today: datetime) -> bool:
    return bool(set(snapshot_dates(df)).intersection(target_dates(today)))


def classify_source(path: Path, snapshot: pd.DataFrame) -> str:
    name = path.name.lower()
    if "netkeiba" in name or ("source_url" in snapshot.columns and snapshot["source_url"].notna().any()):
        return "external"
    return "target"


def choose_entry_source(
    *,
    explicit_entry_csv: str | None,
    target_entry_csv: str,
    fallback_entry_glob: str,
    today: datetime,
    allow_external_fallback: bool = True,
) -> tuple[Path, pd.DataFrame, str, str]:
    if explicit_entry_csv:
        path = project_path(explicit_entry_csv)
        snapshot = read_csv_safe(path) if path.exists() else pd.DataFrame()
        return path, snapshot, classify_source(path, snapshot), "明示指定"

    target_path = project_path(target_entry_csv)
    if target_path.exists():
        target_snapshot = read_csv_safe(target_path)
        if has_target_dates(target_snapshot, today):
            return target_path, target_snapshot, "target", "TARGET公式データを使用"

    if not allow_external_fallback:
        if target_path.exists():
            return target_path, read_csv_safe(target_path), "target", "TARGET official-only mode: target dates not ready"
        return target_path, pd.DataFrame(), "missing", "TARGET official-only mode: entry snapshot missing"

    fallback_candidates = sorted(
        [Path(p) for p in glob.glob(str(project_path(fallback_entry_glob)))],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in fallback_candidates:
        snapshot = read_csv_safe(candidate)
        if has_target_dates(snapshot, today):
            return candidate, snapshot, classify_source(candidate, snapshot), "TARGET未検出のため外部暫定データを使用"

    if target_path.exists():
        return target_path, read_csv_safe(target_path), "target", "TARGET公式データはあるが対象日外"
    if fallback_candidates:
        candidate = fallback_candidates[0]
        snapshot = read_csv_safe(candidate)
        return candidate, snapshot, classify_source(candidate, snapshot), "最新の外部暫定データを使用"
    return target_path, pd.DataFrame(), "missing", "出馬表データ未検出"


def latest_prediction(predictions_dir: Path) -> Path | None:
    files = sorted(predictions_dir.glob("baseline_predictions_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def choose_predictions_dir(explicit_predictions_dir: str | None, source_kind: str) -> Path:
    if explicit_predictions_dir:
        return project_path(explicit_predictions_dir)
    if source_kind == "external" and project_path("outputs/predictions/preday_netkeiba_enriched").exists():
        return project_path("outputs/predictions/preday_netkeiba_enriched")
    return project_path("outputs/predictions")


def candidate_files(dirs: list[Path], max_files: int = 12) -> list[dict[str, object]]:
    candidates: list[Path] = []
    for directory in dirs:
        if not directory.exists():
            continue
        for pattern in ("*.csv", "*.html", "*.htm"):
            candidates.extend(path for path in directory.glob(pattern) if path.is_file())
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        {
            "path": str(path),
            "name": path.name,
            "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            "size": path.stat().st_size,
        }
        for path in candidates[:max_files]
    ]


def status_from_snapshot(snapshot: pd.DataFrame, today: datetime, source_kind: str, source_reason: str) -> dict[str, object]:
    targets = target_dates(today)
    if snapshot.empty:
        return {
            "state": "missing",
            "label": "出馬表未検出",
            "detail": "前日版に使える出馬表スナップショットがありません。",
            "source_kind": source_kind,
            "source_reason": source_reason,
            "target_dates": targets,
        }

    keys = snapshot_dates(snapshot)
    has_target = bool(set(keys).intersection(targets))
    if has_target and source_kind == "target":
        return {
            "state": "ready",
            "label": "TARGET公式データ",
            "detail": "今週対象日のTARGET出馬表が入っています。通常ロジックの形式で最終更新できます。",
            "dates": keys,
            "source_kind": source_kind,
            "source_reason": source_reason,
            "target_dates": targets,
        }
    if has_target and source_kind == "external":
        return {
            "state": "ready",
            "label": "外部暫定データ",
            "detail": "TARGET公式が未反映のため外部出馬表で暫定表示しています。最終判断はTARGET・馬場・馬体重・直前オッズ更新後です。",
            "dates": keys,
            "source_kind": source_kind,
            "source_reason": source_reason,
            "target_dates": targets,
        }
    return {
        "state": "stale",
        "label": "対象日未検出",
        "detail": "現在のスナップショットは対象日外です。TARGET出馬表CSVまたは外部暫定CSVの更新が必要です。",
        "dates": keys,
        "source_kind": source_kind,
        "source_reason": source_reason,
        "target_dates": targets,
    }


def merge_prediction(snapshot: pd.DataFrame, prediction: pd.DataFrame | None) -> pd.DataFrame:
    if snapshot.empty:
        return snapshot.copy()
    if prediction is None or prediction.empty:
        out = snapshot.copy()
        for col in ("ai_rank", "ai_score", "expected_pace"):
            out[col] = pd.NA
        return out

    left = snapshot.copy()
    right = prediction.copy()
    left_race = race_id_col(left)
    right_race = race_id_col(right)
    left_horse = horse_no_col(left)
    right_horse = horse_no_col(right)
    if left_race and right_race and left_horse and right_horse:
        left["_race_key"] = left[left_race].map(norm_id)
        right["_race_key"] = right[right_race].map(norm_id)
        left["_horse_key"] = left[left_horse].map(norm_id)
        right["_horse_key"] = right[right_horse].map(norm_id)
        right_cols = [c for c in right.columns if c not in left.columns or c in {"_race_key", "_horse_key"}]
        merged = left.merge(right[right_cols], on=["_race_key", "_horse_key"], how="left")
        return merged.drop(columns=["_race_key", "_horse_key"], errors="ignore")

    out = snapshot.copy()
    for col in ("ai_rank", "ai_score", "expected_pace"):
        if col not in out.columns:
            out[col] = pd.NA
    return out


def race_value(race: pd.DataFrame, names: list[str], default: object = "") -> object:
    col = pick_col(race, names)
    if not col:
        return default
    for value in race[col].tolist():
        if has_value(value):
            return value
    return default


def non_empty_rate(race: pd.DataFrame, names: list[str]) -> float:
    col = pick_col(race, names)
    if not col or race.empty:
        return 0.0
    return float(race[col].map(has_value).mean())


def valid_odds_rate(race: pd.DataFrame, names: list[str]) -> float:
    if race.empty:
        return 0.0
    return float(race.apply(lambda row: first_valid_odds(row, names) is not None, axis=1).mean())


def top_runner(race: pd.DataFrame) -> pd.Series:
    if "ai_rank" in race.columns and race["ai_rank"].notna().any():
        work = race.copy()
        work["_rank"] = pd.to_numeric(work["ai_rank"], errors="coerce").fillna(999)
        return work.sort_values(["_rank"]).iloc[0]
    return race.iloc[0]


def race_decision(race: pd.DataFrame, source_kind: str) -> dict[str, Any]:
    top = top_runner(race)
    field = max(number(race_value(race, ["頭数", "出走頭数"]), len(race)), len(race))
    horse_no_ready = non_empty_rate(race, ["馬番", "horse_no"]) >= 0.98
    frame_ready = non_empty_rate(race, ["枠番", "frame_no"]) >= 0.98
    odds_ready = valid_odds_rate(race, ["odds_latest_win", "単勝オッズ"]) >= 0.5
    body_ready = non_empty_rate(race, ["馬体重", "body_weight"]) >= 0.5
    track_ready = non_empty_rate(race, ["馬場状態", "track_condition"]) >= 0.5
    previous_rate = max(
        non_empty_rate(race, ["前走レースID(新/馬番無)"]),
        non_empty_rate(race, ["前走確定着順"]),
        non_empty_rate(race, ["前走走破タイム"]),
    )

    ai_score = number(top.get("ai_score"), 0.0)
    gap = number(top.get("ai_score_gap_to_second"), 0.0)
    top_vs_median = number(top.get("ai_top_score_vs_median"), 0.0)
    confidence = number(top.get("ai_confidence_score"), 0.0)
    pressure = number(top.get("race_early_pressure_score"), 0.0)
    front = number(top.get("front_running_tendency"), 0.0)
    close = number(top.get("closing_tendency"), 0.0)

    ai_pass = confidence >= 0.04 or gap >= 0.006 or top_vs_median >= 0.025 or ai_score >= 0.44
    data_pass = horse_no_ready and frame_ready and previous_rate >= 0.45
    final_ready = odds_ready and body_ready and track_ready and source_kind == "target"

    positives: list[str] = []
    waits: list[str] = []
    reasons: list[str] = []

    if ai_pass:
        positives.append("AI上位差または基礎スコアが一定以上")
    if gap >= 0.006:
        positives.append("AI1位と2位の差あり")
    if previous_rate >= 0.75:
        positives.append("前走履歴の補完率が高い")
    if front >= 0.55:
        positives.append("前に行ける軸候補")
    if close >= 0.55:
        positives.append("差し脚の裏付けあり")

    if source_kind == "external":
        waits.append("TARGET公式データ待ち")
    if not horse_no_ready or not frame_ready:
        waits.append("枠順・馬番未確定")
    if not odds_ready:
        waits.append("当日オッズ未取得")
    if not body_ready:
        waits.append("当日馬体重待ち")
    if not track_ready:
        waits.append("馬場状態待ち")

    if not ai_pass:
        reasons.append("AI上位差が小さく信頼度不足")
    if previous_rate < 0.45:
        reasons.append("前走履歴の補完不足")
    if field >= 16:
        reasons.append("多頭数で不確実性高め")
    if pressure >= 0.75:
        reasons.append("先行負荷が高く展開崩れ注意")

    if not data_pass:
        action = "保留"
        action_class = "wait"
        note = "データ未確定。最終判定不可"
    elif not ai_pass:
        action = "見送り候補"
        action_class = "skip"
        note = "AI優位が薄い。買うなら当日材料の上積み必須"
    elif final_ready:
        action = "買い候補"
        action_class = "buy"
        note = "最低条件は通過。直前妙味で最終金額を決定"
    else:
        action = "暫定候補"
        action_class = "candidate"
        note = "構造条件は通過。当日オッズ・馬体重・馬場で最終確認"

    return {
        "action": action,
        "class": action_class,
        "note": note,
        "positives": positives[:4] or ["明確な強調材料は控えめ"],
        "waits": waits[:5],
        "reasons": reasons[:5] or ["大きな見送り理由は薄い"],
        "metrics": {
            "confidence": confidence,
            "gap": gap,
            "top_vs_median": top_vs_median,
            "previous_rate": previous_rate,
            "field": field,
        },
    }


def runner_points(row: pd.Series) -> tuple[list[str], list[str]]:
    points: list[str] = []
    concerns: list[str] = []
    row_odds = first_valid_odds(row, ["odds_latest_win", "単勝オッズ"])
    if number(row.get("ai_rank"), 99) <= 3:
        points.append("AI上位評価")
    if number(row.get("front_running_tendency"), 0.0) >= 0.55:
        points.append("前に行ける形")
    if number(row.get("closing_tendency"), 0.0) >= 0.55:
        points.append("差し脚の裏付け")
    if number(row.get("ai_score"), 0.0) >= 0.44:
        points.append("基礎スコア高め")
    if number(row.get("ai_score_gap_to_second"), 0.0) >= 0.006:
        points.append("AI差あり")
    if row_odds is not None and row_odds <= 3.0:
        concerns.append("市場評価が高く妙味確認")
    if number(row.get("人気"), 0.0) >= 8:
        concerns.append("人気薄で再現性確認")
    if row_odds is None:
        concerns.append("当日オッズ待ち")
    if not has_value(row.get("馬体重")):
        concerns.append("馬体重待ち")
    return points[:3] or ["前日材料で暫定評価"], concerns[:3] or ["大きな減点は未検出"]


def decision_summary(df: pd.DataFrame, source_kind: str) -> dict[str, int]:
    race_col = race_id_col(df)
    if df.empty or not race_col:
        return {"buy": 0, "candidate": 0, "wait": 0, "skip": 0}
    counts = {"buy": 0, "candidate": 0, "wait": 0, "skip": 0}
    for _, race in df.groupby(race_col, sort=False):
        dec = race_decision(race, source_kind)
        counts[dec["class"]] += 1
    return counts


def race_cards(df: pd.DataFrame, source_kind: str, max_races: int) -> str:
    if df.empty:
        return """
        <section class="empty">
          <h2>出馬表がまだありません</h2>
          <p>TARGET出馬表CSVを data/inbox/target/entries に置くか、外部暫定CSVを生成すると、この画面が更新されます。</p>
        </section>
        """

    race_col = race_id_col(df)
    horse_col = horse_no_col(df)
    sort_cols = [c for c in [date_col(df), pick_col(df, ["場所"]), pick_col(df, ["Ｒ"]), race_col, "ai_rank", horse_col] if c]
    shown = df.sort_values(sort_cols) if sort_cols else df
    groups = list(shown.groupby(race_col, sort=False)) if race_col else [("race", shown)]
    html_parts: list[str] = []

    for _, race in groups[:max_races]:
        first = race.iloc[0]
        decision = race_decision(race, source_kind)
        date_label = fmt_date(race_value(race, ["日付S", "日付", "date"]))
        venue = escape(race_value(race, ["場所"]))
        race_no = escape(race_value(race, ["Ｒ", "R"]))
        race_name = escape(race_value(race, ["レース名"]))
        surface = escape(race_value(race, ["芝・ダ"]))
        distance = escape(race_value(race, ["距離"]))
        field = escape(race_value(race, ["頭数", "出走頭数"], len(race)))
        pace = escape(first.get("expected_pace", ""))
        metrics = decision["metrics"]
        gate_chips = "".join(f"<span>{escape(v)}</span>" for v in decision["positives"])
        wait_chips = "".join(f"<span>{escape(v)}</span>" for v in decision["waits"])
        reason_chips = "".join(f"<span>{escape(v)}</span>" for v in decision["reasons"])

        if "ai_rank" in race.columns and race["ai_rank"].notna().any():
            race_sorted = race.assign(_rank=pd.to_numeric(race["ai_rank"], errors="coerce").fillna(999)).sort_values("_rank")
        elif horse_col:
            race_sorted = race.assign(_horse=pd.to_numeric(race[horse_col], errors="coerce").fillna(999)).sort_values("_horse")
        else:
            race_sorted = race

        table_rows: list[str] = []
        for _, row in race_sorted.head(18).iterrows():
            points, concerns = runner_points(row)
            rank = fmt_int(row.get("ai_rank")) or "-"
            score = number(row.get("ai_score"), float("nan"))
            score_text = "" if pd.isna(score) else f"{score:.3f}"
            odds_value = first_valid_odds(row, ["odds_latest_win", "単勝オッズ"])
            odds = "-" if odds_value is None else f"{odds_value:.1f}"
            popularity = text(row.get("odds_latest_popularity")) or text(row.get("人気"), "-")
            points_html = "".join(f"<span>{escape(v)}</span>" for v in points)
            concerns_html = "".join(f"<span>{escape(v)}</span>" for v in concerns)
            table_rows.append(
                f"""
                <tr>
                  <td class="rank">{escape(rank)}</td>
                  <td>{escape(fmt_int(row.get("枠番")))}</td>
                  <td>{escape(fmt_int(row.get("馬番")))}</td>
                  <td class="horse">{escape(row.get("馬名"))}<small>{escape(row.get("騎手"))} / {escape(row.get("斤量"))}</small></td>
                  <td>{score_text}</td>
                  <td>{escape(popularity or "-")}</td>
                  <td>{escape(odds or "-")}</td>
                  <td><div class="chips good">{points_html}</div></td>
                  <td><div class="chips bad">{concerns_html}</div></td>
                </tr>
                """
            )

        html_parts.append(
            f"""
            <section class="race">
              <header class="race-head">
                <div>
                  <div class="meta">{date_label} {venue}{race_no}R</div>
                  <h2>{race_name}</h2>
                </div>
                <div class="race-facts">
                  <span>{surface}{distance}m</span>
                  <span>{field}頭</span>
                  <span>展開 {pace or "暫定"}</span>
                </div>
              </header>
              <div class="decision {decision["class"]}">
                <div>
                  <strong>{escape(decision["action"])}</strong>
                  <span>{escape(decision["note"])}</span>
                </div>
                <div class="mini-metrics">
                  <span>信頼 {metrics["confidence"]:.3f}</span>
                  <span>AI差 {metrics["gap"]:.3f}</span>
                  <span>履歴 {metrics["previous_rate"]:.0%}</span>
                </div>
                <div class="decision-grid">
                  <div><b>通過材料</b><div class="chips good">{gate_chips}</div></div>
                  <div><b>当日待ち</b><div class="chips warn">{wait_chips or "<span>なし</span>"}</div></div>
                  <div><b>見送り理由</b><div class="chips bad">{reason_chips}</div></div>
                </div>
              </div>
              <div class="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>AI</th><th>枠</th><th>馬番</th><th>馬</th><th>Score</th><th>人気</th><th>単勝</th><th>評価ポイント</th><th>懸念点</th>
                    </tr>
                  </thead>
                  <tbody>{''.join(table_rows)}</tbody>
                </table>
              </div>
            </section>
            """
        )
    return "\n".join(html_parts)


def build_html(
    *,
    snapshot: pd.DataFrame,
    combined: pd.DataFrame,
    status: dict[str, object],
    source_kind: str,
    candidates: list[dict[str, object]],
    entry_path: Path,
    prediction_path: Path | None,
    predictions_dir: Path,
    today: datetime,
    max_races: int,
) -> str:
    state = text(status.get("state"))
    state_class = {"ready": "ready", "stale": "stale"}.get(state, "missing")
    dates = ", ".join(fmt_date(v) for v in snapshot_dates(snapshot)) if not snapshot.empty else "-"
    targets = ", ".join(datetime.strptime(d, "%Y%m%d").strftime("%Y年%m月%d日") for d in status.get("target_dates", []))
    primary_target_date = next(iter(status.get("target_dates", [])), today.strftime("%Y%m%d"))
    race_col = race_id_col(snapshot)
    venue_col = pick_col(snapshot, ["場所"])
    race_count = snapshot[race_col].nunique() if race_col else 0
    venue_count = snapshot[venue_col].nunique() if venue_col else 0
    source_label = "TARGET公式" if source_kind == "target" else "外部暫定" if source_kind == "external" else "未検出"
    counts = decision_summary(combined, source_kind)

    candidate_rows = "\n".join(
        f"<tr><td>{escape(item['name'])}</td><td>{escape(item['modified'])}</td><td>{int(item['size']):,}</td><td class=\"path\">{escape(item['path'])}</td></tr>"
        for item in candidates
    ) or "<tr><td colspan=\"4\">候補ファイルは見つかっていません。</td></tr>"
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    pred_label = str(prediction_path.relative_to(ROOT)) if prediction_path and prediction_path.exists() else "未生成"
    entry_label = str(entry_path.relative_to(ROOT)) if entry_path.exists() and entry_path.is_relative_to(ROOT) else str(entry_path)
    predictions_label = str(predictions_dir.relative_to(ROOT)) if predictions_dir.exists() and predictions_dir.is_relative_to(ROOT) else str(predictions_dir)

    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Keiba AI 前日版</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f6f8;
      --panel: #ffffff;
      --line: #d7dee8;
      --text: #17202a;
      --muted: #64748b;
      --accent: #0f766e;
      --blue: #1d4ed8;
      --warn: #b45309;
      --danger: #b91c1c;
      --good-bg: #e8f7f1;
      --warn-bg: #fff7ed;
      --bad-bg: #fff1f2;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; line-height: 1.5; }}
    header.hero {{ padding: 18px clamp(14px, 3vw, 28px); background: #fff; border-bottom: 1px solid var(--line); position: sticky; top: 0; z-index: 5; }}
    .hero-row {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; }}
    h1 {{ margin: 0; font-size: 22px; letter-spacing: 0; }}
    h2 {{ margin: 2px 0 0; font-size: 18px; letter-spacing: 0; }}
    .sub {{ color: var(--muted); font-size: 13px; margin-top: 3px; }}
    .badge {{ display: inline-flex; align-items: center; white-space: nowrap; border: 1px solid var(--line); border-radius: 999px; padding: 6px 10px; font-weight: 700; font-size: 13px; background: #f8fafc; }}
    .badge.ready {{ color: var(--accent); border-color: #99d6ca; background: #eefbf7; }}
    .badge.stale {{ color: var(--warn); border-color: #f0c37d; background: #fff8ea; }}
    .badge.missing {{ color: var(--danger); border-color: #f3a6aa; background: #fff1f2; }}
    main {{ padding: 18px clamp(12px, 3vw, 28px) 36px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; margin-bottom: 14px; }}
    .metric, .note, .race, .source {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }}
    .metric {{ padding: 12px; min-height: 78px; }}
    .metric small {{ display:block; color: var(--muted); font-size: 12px; }}
    .metric strong {{ display:block; font-size: 19px; margin-top: 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 100%; }}
    .note {{ padding: 13px 14px; margin-bottom: 14px; display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, .9fr); gap: 14px; }}
    code {{ display: block; padding: 9px 10px; background: #0f172a; color: #e2e8f0; border-radius: 6px; overflow-x: auto; font-size: 12px; white-space: nowrap; }}
    .workflow {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin: 10px 0 2px; }}
    .step {{ border: 1px solid var(--line); border-radius: 6px; padding: 8px; background: #f8fafc; font-size: 13px; }}
    .step strong {{ display:block; font-size: 13px; }}
    .rules {{ margin-top: 10px; padding-left: 20px; color: #334155; font-size: 13px; }}
    .race {{ margin-top: 12px; overflow: hidden; }}
    .race-head {{ padding: 12px 14px; border-bottom: 1px solid var(--line); display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }}
    .meta {{ color: var(--muted); font-size: 12px; }}
    .race-facts {{ display: flex; gap: 7px; flex-wrap: wrap; justify-content: flex-end; }}
    .race-facts span, .mini-metrics span {{ border: 1px solid var(--line); border-radius: 999px; padding: 4px 8px; font-size: 12px; color: #334155; background: #f8fafc; }}
    .decision {{ padding: 10px 14px; border-bottom: 1px solid var(--line); }}
    .decision > div:first-child {{ display: flex; gap: 8px; align-items: baseline; flex-wrap: wrap; }}
    .decision strong {{ font-size: 17px; }}
    .decision.buy strong {{ color: #047857; }}
    .decision.candidate strong {{ color: #0f766e; }}
    .decision.wait strong {{ color: #b45309; }}
    .decision.skip strong {{ color: #b91c1c; }}
    .mini-metrics {{ display: flex; gap: 6px; flex-wrap: wrap; margin: 8px 0; }}
    .decision-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }}
    .decision-grid b {{ display:block; font-size: 12px; color: var(--muted); margin-bottom: 4px; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 900px; }}
    th, td {{ border-bottom: 1px solid #e6ebf2; padding: 9px 8px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ color: #475569; background: #f8fafc; font-size: 12px; font-weight: 700; }}
    td.rank {{ font-weight: 800; color: var(--blue); }}
    td.horse {{ font-weight: 800; min-width: 170px; }}
    td.horse small {{ display: block; color: var(--muted); font-weight: 500; margin-top: 2px; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 4px; min-width: 130px; }}
    .chips span {{ border-radius: 999px; padding: 3px 7px; font-size: 12px; white-space: nowrap; }}
    .chips.good span {{ color: #047857; background: var(--good-bg); border: 1px solid #b4e2d0; }}
    .chips.warn span {{ color: #b45309; background: var(--warn-bg); border: 1px solid #fed7aa; }}
    .chips.bad span {{ color: #b91c1c; background: var(--bad-bg); border: 1px solid #fecdd3; }}
    .source {{ margin-top: 14px; padding: 12px 14px; }}
    .source h2 {{ margin-bottom: 8px; }}
    .source table {{ min-width: 820px; }}
    .path {{ color: var(--muted); font-size: 12px; }}
    .empty {{ background: var(--panel); border: 1px dashed #b8c2cf; border-radius: 8px; padding: 22px; text-align: center; }}
    @media (max-width: 860px) {{
      .hero-row, .race-head, .note {{ display: block; }}
      .badge {{ margin-top: 9px; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .workflow, .decision-grid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 20px; }}
      table {{ min-width: 760px; }}
    }}
  </style>
</head>
<body>
  <header class="hero">
    <div class="hero-row">
      <div>
        <h1>Keiba AI 前日版</h1>
        <div class="sub">生成 {generated_at} / 対象目安 {targets}</div>
      </div>
      <span class="badge {state_class}">{escape(status.get("label"))}</span>
    </div>
  </header>
  <main>
    <section class="metrics">
      <div class="metric"><small>データソース</small><strong>{escape(source_label)}</strong></div>
      <div class="metric"><small>現在の出馬表</small><strong>{len(snapshot):,}頭</strong></div>
      <div class="metric"><small>レース数</small><strong>{int(race_count):,}R</strong></div>
      <div class="metric"><small>暫定候補</small><strong>{counts["candidate"] + counts["buy"]}R</strong></div>
      <div class="metric"><small>見送り候補</small><strong>{counts["skip"]}R</strong></div>
    </section>

    <section class="note">
      <div>
        <h2>ステータス</h2>
        <p>{escape(status.get("detail"))}</p>
        <p class="sub">現在入っている日付: {escape(dates)} / entry: {escape(entry_label)}</p>
        <p class="sub">選択理由: {escape(status.get("source_reason"))} / prediction dir: {escape(predictions_label)}</p>
        <div class="workflow">
          <div class="step"><strong>1. 入力選択</strong>TARGET公式を優先。未検出なら外部暫定へ自動フォールバック。</div>
          <div class="step"><strong>2. 買い判定</strong>AI差・履歴補完率・枠順確定・当日情報でゲート判定。</div>
          <div class="step"><strong>3. 見送り理由</strong>足りない条件をレースごとに表示。</div>
          <div class="step"><strong>4. 最終更新</strong>当日オッズ・馬体重・馬場で買い目と金額を確定。</div>
        </div>
      </div>
      <div>
        <h2>買い判定ルール</h2>
        <ol class="rules">
          <li>AI優位: 信頼スコア0.040以上、AI差0.006以上、または基礎スコア0.440以上。</li>
          <li>データ条件: 枠順・馬番確定、前走履歴補完率45%以上。</li>
          <li>最終条件: TARGET公式、当日オッズ、馬体重、馬場状態が揃うこと。</li>
          <li>前日版は「暫定候補」まで。当日の直前妙味が残ったものだけ買い。</li>
        </ol>
        <code>powershell -ExecutionPolicy Bypass -File scripts\\run_target_weekly_update.ps1 -TargetDate {primary_target_date}</code>
      </div>
    </section>

    {race_cards(combined, source_kind, max_races)}

    <section class="source">
      <h2>最近見つかった出馬表候補</h2>
      <div class="table-wrap">
        <table>
          <thead><tr><th>ファイル</th><th>更新</th><th>サイズ</th><th>場所</th></tr></thead>
          <tbody>{candidate_rows}</tbody>
        </table>
      </div>
    </section>
  </main>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a provisional pre-race dashboard for the day before racing.")
    parser.add_argument("--entry-csv", default=None, help="Explicit entry snapshot. If omitted, TARGET is preferred and external snapshots are fallback.")
    parser.add_argument("--target-entry-csv", default="data/datasets/inference/weekly/entry_snapshot.csv")
    parser.add_argument("--fallback-entry-glob", default="data/datasets/inference/weekly/entry_snapshot_netkeiba_*_enriched.csv")
    parser.add_argument("--official-only", action="store_true", help="Disable external/netkeiba fallback and show TARGET status only.")
    parser.add_argument("--predictions-dir", default=None, help="Explicit prediction directory. If omitted, selected from the entry source.")
    parser.add_argument("--output-html", default="outputs/ui/keiba_preday_dashboard.html")
    parser.add_argument("--today", default=None, help="YYYY-MM-DD. Defaults to local date.")
    parser.add_argument("--max-races", type=int, default=12)
    parser.add_argument(
        "--candidate-dir",
        action="append",
        default=[
            "data/inbox/target/entries",
            "data/datasets/inference/weekly",
            "data/raw/external/netkeiba_entries",
            "C:/Users/kazup/Data Lab/TXT",
            "C:/Users/kazup/Downloads",
        ],
    )
    args = parser.parse_args()

    today = datetime.strptime(args.today, "%Y-%m-%d") if args.today else datetime.now()
    entry_path, snapshot, source_kind, source_reason = choose_entry_source(
        explicit_entry_csv=args.entry_csv,
        target_entry_csv=args.target_entry_csv,
        fallback_entry_glob=args.fallback_entry_glob,
        today=today,
        allow_external_fallback=not args.official_only,
    )
    predictions_dir = choose_predictions_dir(args.predictions_dir, source_kind)
    pred_path = latest_prediction(predictions_dir)
    prediction = read_csv_safe(pred_path) if pred_path else None
    combined = merge_prediction(snapshot, prediction)
    status = status_from_snapshot(snapshot, today, source_kind, source_reason)
    candidate_dirs = [project_path(p) for p in args.candidate_dir]
    if args.official_only:
        candidate_dirs = [path for path in candidate_dirs if "netkeiba" not in str(path).lower()]
    candidates = candidate_files(candidate_dirs)

    output_path = project_path(args.output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_html(
            snapshot=snapshot,
            combined=combined,
            status=status,
            source_kind=source_kind,
            candidates=candidates,
            entry_path=entry_path,
            prediction_path=pred_path,
            predictions_dir=predictions_dir,
            today=today,
            max_races=args.max_races,
        ),
        encoding="utf-8",
    )
    summary = {
        "output_html": str(output_path),
        "status": status,
        "source_kind": source_kind,
        "entry_csv": str(entry_path),
        "rows": int(len(snapshot)),
        "races": int(snapshot[race_id_col(snapshot)].nunique()) if race_id_col(snapshot) else 0,
        "decision_counts": decision_summary(combined, source_kind),
        "prediction_csv": str(pred_path) if pred_path else "",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
