from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from lxml import html as lxml_html


ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, low_memory=False)


def text_from(node) -> str:
    return " ".join(part.strip() for part in node.xpath(".//text()") if part and part.strip())


def decode_html(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("cp932", "shift_jis", "utf-8"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("cp932", errors="replace")


def race_meta(path: Path) -> dict[str, str]:
    body = decode_html(path)
    tree = lxml_html.fromstring(body)
    race_name = ""
    race_name_nodes = tree.xpath("//*[contains(concat(' ', normalize-space(@class), ' '), ' race_name ')]")
    if race_name_nodes:
        race_name = text_from(race_name_nodes[0])
    title = text_from(tree.xpath("//title")[0]) if tree.xpath("//title") else ""
    course = ""
    course_nodes = tree.xpath("//*[contains(concat(' ', normalize-space(@class), ' '), ' race_data ')]")
    if course_nodes:
        course = text_from(course_nodes[0])
    horses: list[dict[str, str | int]] = []
    for tr in tree.xpath("//table[contains(concat(' ', normalize-space(@class), ' '), ' tanpuku ')]//tbody/tr"):
        num_nodes = tr.xpath(".//td[contains(concat(' ', normalize-space(@class), ' '), ' num ')]")
        horse_nodes = tr.xpath(".//td[contains(concat(' ', normalize-space(@class), ' '), ' horse ')]")
        if not num_nodes or not horse_nodes:
            continue
        num_text = text_from(num_nodes[0])
        match = re.search(r"\d+", num_text)
        if not match:
            continue
        horses.append({"horse_no": int(match.group(0)), "horse_name": text_from(horse_nodes[0])})
    race_id_match = re.search(r"(20\d{14})", str(path))
    race_id = race_id_match.group(1) if race_id_match else path.parent.name
    return {"race_id": race_id, "race_name": race_name, "course": course, "title": title, "horses": horses}


def race_label(race_id: str) -> str:
    venue_code = race_id[8:10] if len(race_id) >= 10 else ""
    race_no = str(int(race_id[-2:])) if len(race_id) >= 2 and race_id[-2:].isdigit() else race_id[-2:]
    venues = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京", "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}
    return f"{venues.get(venue_code, venue_code)} {race_no}R"


def fmt_odds(value) -> str:
    try:
        if pd.isna(value):
            return ""
        return f"{float(value):.1f}"
    except Exception:
        return ""


def h(value) -> str:
    return html.escape("" if value is None else str(value))


def render(args: argparse.Namespace) -> dict:
    date_key = args.date
    raw_root = project_path(args.raw_root)
    single = read_csv(project_path(args.single_odds_csv))
    pair = read_csv(project_path(args.pair_odds_csv))
    race_ids = sorted(set(single.get("race_id", pd.Series(dtype=str)).astype(str)))
    cards: list[dict] = []
    for race_id in race_ids:
        html_files = sorted((raw_root / race_id).glob("*_win_place_frame.html"), key=lambda p: p.stat().st_mtime, reverse=True)
        meta = race_meta(html_files[0]) if html_files else {"race_id": race_id, "race_name": "", "course": "", "title": "", "horses": []}
        horses = pd.DataFrame(meta["horses"])
        race_single = single[single["race_id"].astype(str).eq(race_id)].copy() if not single.empty else pd.DataFrame()
        if not horses.empty and not race_single.empty:
            race_single["horse_no"] = pd.to_numeric(race_single["horse_no"], errors="coerce").astype("Int64")
            horses["horse_no"] = pd.to_numeric(horses["horse_no"], errors="coerce").astype("Int64")
            runners = horses.merge(race_single, on="horse_no", how="left")
        else:
            runners = horses if not horses.empty else race_single
        if "live_popularity" in runners.columns:
            runners = runners.sort_values(["live_popularity", "horse_no"], na_position="last")
        cards.append({"race_id": race_id, "meta": meta, "runners": runners})

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections = []
    for card in cards:
        race_id = card["race_id"]
        meta = card["meta"]
        runners = card["runners"]
        rows = []
        for _, row in runners.iterrows():
            rows.append(
                "<tr>"
                f"<td>{h(int(row.get('horse_no')) if pd.notna(row.get('horse_no')) else '')}</td>"
                f"<td class='horse'>{h(row.get('horse_name', ''))}</td>"
                f"<td>{h(int(row.get('live_popularity')) if pd.notna(row.get('live_popularity')) else '')}</td>"
                f"<td>{h(fmt_odds(row.get('live_win_odds')))}</td>"
                f"<td>{h(fmt_odds(row.get('live_place_odds_min')))}-{h(fmt_odds(row.get('live_place_odds_max')))}</td>"
                "<td><span class='badge wait'>暫定</span></td>"
                "</tr>"
            )
        race_pair = pair[pair["race_id"].astype(str).eq(race_id)].copy() if not pair.empty else pd.DataFrame()
        pair_blocks = []
        for ticket_type, label in [("umaren", "馬連"), ("wide", "ワイド")]:
            sub = race_pair[race_pair["ticket_type"].eq(ticket_type)].copy() if not race_pair.empty else pd.DataFrame()
            if sub.empty:
                continue
            sub = sub.sort_values("live_odds").head(8)
            items = []
            name_by_no = {int(r["horse_no"]): str(r.get("horse_name", "")) for _, r in runners.iterrows() if pd.notna(r.get("horse_no"))}
            for _, row in sub.iterrows():
                a = int(row["a_no"])
                b = int(row["b_no"])
                items.append(f"<li>{a}. {h(name_by_no.get(a, ''))} - {b}. {h(name_by_no.get(b, ''))}<b>{fmt_odds(row.get('live_odds'))}</b></li>")
            pair_blocks.append(f"<div class='pair'><h4>{label} 人気順</h4><ul>{''.join(items)}</ul></div>")
        sections.append(
            "<section class='race-card'>"
            f"<div class='race-head'><div><p>{h(race_label(race_id))}</p><h2>{h(meta.get('race_name') or race_id)}</h2><span>{h(meta.get('course'))}</span></div><b>判定: TARGET待ち</b></div>"
            "<div class='note'>TARGET出馬表・過去走・調教・馬場実測が入るまでは、AI順位と最終買い目は確定しません。ここでは前日公式オッズの歪み確認だけ行います。</div>"
            "<table><thead><tr><th>馬番</th><th>馬名</th><th>人気</th><th>単勝</th><th>複勝</th><th>状態</th></tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
            + "<div class='pairs'>"
            + "".join(pair_blocks)
            + "</div></section>"
        )

    if not sections:
        sections.append("<section class='race-card'><h2>取得できた明日レースはありません</h2><p>JRA公式オッズまたはTARGET出馬表の更新待ちです。</p></section>")

    html_text = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{date_key} 明日前日 暫定ダッシュボード</title>
<style>
  body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:#f4f6f8; color:#111827; }}
  header {{ padding:20px clamp(16px,4vw,40px); background:#111827; color:white; }}
  header h1 {{ margin:0 0 8px; font-size:24px; letter-spacing:0; }}
  header p {{ margin:0; color:#cbd5e1; }}
  main {{ padding:18px clamp(12px,3vw,32px) 40px; max-width:1180px; margin:auto; }}
  .status {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:10px; margin-bottom:16px; }}
  .status div,.race-card {{ background:white; border:1px solid #d9e1ea; border-radius:8px; }}
  .status div {{ padding:12px 14px; }}
  .status b {{ display:block; font-size:13px; color:#475569; margin-bottom:4px; }}
  .race-card {{ margin:0 0 16px; overflow:hidden; }}
  .race-head {{ display:flex; justify-content:space-between; gap:12px; padding:14px 16px; border-bottom:1px solid #e5e7eb; align-items:start; }}
  .race-head p {{ margin:0 0 2px; color:#475569; font-weight:700; }}
  .race-head h2 {{ margin:0 0 4px; font-size:20px; }}
  .race-head span {{ color:#64748b; font-size:13px; }}
  .race-head b {{ color:#92400e; background:#fef3c7; border:1px solid #fde68a; border-radius:999px; padding:6px 10px; white-space:nowrap; }}
  .note {{ margin:12px 16px; padding:10px 12px; background:#f8fafc; border-left:4px solid #64748b; color:#475569; font-size:14px; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; }}
  th,td {{ padding:9px 10px; border-top:1px solid #edf2f7; text-align:right; }}
  th {{ color:#475569; background:#f8fafc; font-weight:700; }}
  th:nth-child(2),td.horse {{ text-align:left; }}
  .badge {{ display:inline-block; padding:3px 8px; border-radius:999px; font-size:12px; font-weight:700; }}
  .wait {{ background:#e0f2fe; color:#075985; }}
  .pairs {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; padding:14px 16px 16px; }}
  .pair {{ border:1px solid #e5e7eb; border-radius:8px; padding:10px 12px; }}
  .pair h4 {{ margin:0 0 8px; }}
  .pair ul {{ margin:0; padding:0; list-style:none; }}
  .pair li {{ display:flex; justify-content:space-between; gap:10px; padding:6px 0; border-top:1px solid #f1f5f9; }}
  .pair li:first-child {{ border-top:0; }}
  .pair b {{ color:#047857; }}
  @media (max-width: 720px) {{
    header h1 {{ font-size:20px; }}
    .race-head {{ display:block; }}
    .race-head b {{ display:inline-block; margin-top:8px; }}
    table {{ font-size:13px; }}
    th,td {{ padding:8px 6px; }}
  }}
</style>
</head>
<body>
<header>
  <h1>{date_key} 明日前日 暫定ダッシュボード</h1>
  <p>今日の反省を反映し、明日は「TARGET確定前は暫定」「馬場変化・馬体重・直前オッズで最終判定」に分けます。</p>
</header>
<main>
  <div class="status">
    <div><b>生成時刻</b>{h(generated_at)}</div>
    <div><b>取得レース</b>{len(cards)}R（JRA公式オッズで確認できた分）</div>
    <div><b>AI判定</b>TARGET出馬表待ち</div>
    <div><b>明日の重点</b>道悪ガード、馬場変化検知、直前オッズ、馬体重を最終反映</div>
  </div>
  {''.join(sections)}
</main>
</body>
</html>
"""
    out = project_path(args.output_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    summary = {
        "date": date_key,
        "output_html": str(out),
        "generated_at": generated_at,
        "races": len(cards),
        "race_ids": race_ids,
        "note": "Official JRA preday probe dashboard. AI rankings require TARGET entry snapshot for the target date.",
    }
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact tomorrow dashboard from JRA official odds HTML.")
    parser.add_argument("--date", required=True)
    parser.add_argument("--raw-root", default="data/raw/jra_official_odds")
    parser.add_argument("--single-odds-csv", required=True)
    parser.add_argument("--pair-odds-csv", required=True)
    parser.add_argument("--output-html", default="outputs/ui/keiba_tomorrow_probe_dashboard.html")
    args = parser.parse_args()
    print(json.dumps(render(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
