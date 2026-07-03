import argparse
import csv
import datetime as dt
import html
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_dashboard_payload(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    marker = "const payload = "
    start = text.index(marker) + len(marker)
    payload, _ = json.JSONDecoder().raw_decode(text[start:])
    return payload


def safe_int(value, default=None):
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def race_datetime(race: dict) -> dt.datetime | None:
    date_key = str(race.get("dateKey") or "")
    start_time = str(race.get("startTime") or "")
    if len(date_key) != 8 or ":" not in start_time:
        return None
    try:
        hh, mm = [int(x) for x in start_time.split(":", 1)]
        return dt.datetime(int(date_key[:4]), int(date_key[4:6]), int(date_key[6:8]), hh, mm)
    except Exception:
        return None


def pair_key(a, b) -> str:
    aa = safe_int(a)
    bb = safe_int(b)
    if aa is None or bb is None:
        return ""
    lo, hi = sorted([aa, bb])
    return f"{lo}-{hi}"


def find_pair(pair_rows, race_id, ticket_type, a_no, b_no):
    key = pair_key(a_no, b_no)
    for row in pair_rows:
        if row.get("raceId") != race_id:
            continue
        if row.get("ticketType") != ticket_type:
            continue
        if pair_key(row.get("aNo"), row.get("bNo")) == key:
            return row
    return None


def reference_group_for_decision(decision_key: str) -> str:
    key = str(decision_key or "skip")
    if key == "candidate":
        return "reference_candidate"
    if key == "watch":
        return "reference_watch"
    if key == "weak":
        return "reference_weak"
    return "reference_skip"


def reference_action_for_decision(decision_key: str) -> str:
    key = str(decision_key or "skip")
    if key == "candidate":
        return "非購入・準候補"
    if key == "watch":
        return "非購入・注視"
    if key == "weak":
        return "非購入・参考弱"
    return "非購入・見送り"


def build_reference_tickets(race_id, singles_by_race, pair_rows, decision_key="skip", decision_label=""):
    rows = [x for x in singles_by_race.get(race_id, []) if safe_int(x.get("aiRank")) is not None]
    rows.sort(key=lambda x: safe_int(x.get("aiRank"), 999))
    if len(rows) < 2:
        return []
    anchor = rows[0]
    partners = rows[1:4]
    out = []

    def add(ticket_type, partner, stake_yen):
        pair = find_pair(pair_rows, race_id, ticket_type, anchor.get("horseNo"), partner.get("horseNo")) or {}
        out.append(
            {
                "raceId": race_id,
                "ticketType": ticket_type,
                "ticketLabel": "ワイド" if ticket_type == "wide" else "馬連",
                "aNo": safe_int(anchor.get("horseNo")),
                "bNo": safe_int(partner.get("horseNo")),
                "aName": anchor.get("horseName", ""),
                "bName": partner.get("horseName", ""),
                "stakeYen": float(stake_yen),
                "action": reference_action_for_decision(decision_key),
                "decisionGroup": reference_group_for_decision(decision_key),
                "decisionLabel": decision_label,
                "reason": decision_label or reference_action_for_decision(decision_key),
                "liveOdds": pair.get("odds"),
                "livePay": pair.get("payPer100"),
                "reference": True,
            }
        )

    if len(partners) >= 1:
        add("wide", partners[0], 100)
        add("umaren", partners[0], 100)
    if len(partners) >= 2:
        add("wide", partners[1], 100)
    return out


def build_displayed_tickets(payload: dict) -> list[dict]:
    singles_by_race = {}
    for row in payload.get("singleRows", []):
        singles_by_race.setdefault(row.get("raceId"), []).append(row)
    pair_rows = payload.get("pairRows", [])
    final_by_race = {}
    for row in payload.get("ticketRows", []):
        item = dict(row)
        item["decisionGroup"] = "final_buy"
        item["reference"] = False
        final_by_race.setdefault(item.get("raceId"), []).append(item)
    decision_by_race = {
        str(row.get("raceId")): row
        for row in payload.get("raceDecisionRows", [])
        if row.get("raceId")
    }

    tickets = []
    for race in payload.get("races", []):
        race_id = race.get("raceId")
        finals = final_by_race.get(race_id, [])
        if finals:
            race_tickets = finals
        else:
            # Keep non-buy reference tickets, but preserve how close the race was to BUY.
            decision = decision_by_race.get(str(race_id), {})
            race_tickets = build_reference_tickets(
                race_id,
                singles_by_race,
                pair_rows,
                decision_key=decision.get("key", "skip"),
                decision_label=decision.get("label", ""),
            )
        for ticket in race_tickets:
            enriched = dict(ticket)
            enriched["raceLabel"] = race.get("raceLabel", "")
            enriched["venue"] = race.get("venue", "")
            enriched["raceNo"] = race.get("raceNo", "")
            enriched["startTime"] = race.get("startTime", "")
            enriched["raceName"] = race.get("raceName", "")
            enriched["dateKey"] = race.get("dateKey", "")
            tickets.append(enriched)
    return tickets


def latest_odds_html(race_id: str) -> Path | None:
    base = ROOT / "data" / "raw" / "jra_official_odds" / race_id
    if not base.exists():
        return None
    files = sorted(base.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


RESULT_CNAME_RE = re.compile(
    r"pw01sde01(?P<jyo>\d{2})(?P<year>\d{4})(?P<kaiji>\d{2})(?P<nichiji>\d{2})(?P<race>\d{2})(?P<date>\d{8})/(?P<check>[0-9A-F]{2})"
)


def result_cname_race_id(cname: str) -> str | None:
    m = RESULT_CNAME_RE.search(str(cname or ""))
    if not m:
        return None
    return f"{m.group('date')}{m.group('jyo')}{m.group('kaiji')}{m.group('nichiji')}{m.group('race')}"


def result_cname_parts(cname: str) -> dict | None:
    m = RESULT_CNAME_RE.search(str(cname or ""))
    return m.groupdict() if m else None


def extract_result_cnames_from_text(text: str) -> list[str]:
    candidates = re.findall(r'accessS\.html\?CNAME=([^"\']+)', text)
    candidates.extend(re.findall(r"doAction\('/JRADB/accessS\.html'\s*,\s*'([^']+)'", text))
    candidates.extend(re.findall(r'doAction\("/JRADB/accessS\.html"\s*,\s*"([^"]+)"', text))
    return [html.unescape(cname) for cname in candidates]


def next_result_check(prev_check: str, prev_race_no: int) -> str:
    # JRA result CNAME checks for adjacent race numbers move predictably.
    # The 09->10 boundary changes both race-number digits, hence the smaller step.
    step = 0x0B if prev_race_no == 9 else 0x4B
    return f"{(int(prev_check, 16) - step) % 256:02X}"


def previous_result_check(next_check: str, prev_race_no: int) -> str:
    step = 0x0B if prev_race_no == 9 else 0x4B
    return f"{(int(next_check, 16) + step) % 256:02X}"


def build_result_cname_from_known(known_cname: str, target_race_id: str) -> str | None:
    parts = result_cname_parts(known_cname)
    if not parts or len(target_race_id) < 16:
        return None
    target_date = target_race_id[:8]
    target_jyo = target_race_id[8:10]
    target_kaiji = target_race_id[10:12]
    target_nichiji = target_race_id[12:14]
    target_race_no = safe_int(target_race_id[14:16])
    known_race_no = safe_int(parts.get("race"))
    if (
        target_race_no is None
        or known_race_no is None
        or parts["date"] != target_date
        or parts["jyo"] != target_jyo
        or parts["kaiji"] != target_kaiji
        or parts["nichiji"] != target_nichiji
    ):
        return None
    check = parts["check"]
    if known_race_no < target_race_no:
        for race_no in range(known_race_no, target_race_no):
            check = next_result_check(check, race_no)
    elif known_race_no > target_race_no:
        for race_no in range(known_race_no - 1, target_race_no - 1, -1):
            check = previous_result_check(check, race_no)
    return f"pw01sde01{target_jyo}{target_race_id[:4]}{target_kaiji}{target_nichiji}{target_race_id[14:16]}{target_date}/{check}"


def fallback_result_cname_from_neighbor(race_id: str) -> str | None:
    if len(str(race_id)) < 16:
        return None
    prefix = str(race_id)[:14]
    root = ROOT / "data" / "raw" / "jra_official_odds"
    known: list[tuple[int, str]] = []
    for race_no in range(1, 13):
        sibling_id = f"{prefix}{race_no:02d}"
        base = root / sibling_id
        if not base.exists():
            continue
        for path in sorted(base.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True):
            text = path.read_bytes().decode("shift_jis", errors="ignore")
            for cname in extract_result_cnames_from_text(text):
                if result_cname_race_id(cname) == sibling_id:
                    known.append((abs(race_no - safe_int(str(race_id)[14:16], 99)), cname))
                    break
            if known and result_cname_race_id(known[-1][1]) == sibling_id:
                break
    if not known:
        return None
    known.sort(key=lambda item: item[0])
    for _, cname in known:
        built = build_result_cname_from_known(cname, str(race_id))
        if built:
            return built
    return None


def extract_result_cname_from_odds(race_id: str) -> str | None:
    base = ROOT / "data" / "raw" / "jra_official_odds" / str(race_id)
    if base.exists():
        for path in sorted(base.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True):
            text = path.read_bytes().decode("shift_jis", errors="ignore")
            for cname in extract_result_cnames_from_text(text):
                if result_cname_race_id(cname) == str(race_id):
                    return cname
    fallback = fallback_result_cname_from_neighbor(str(race_id))
    if fallback:
        return fallback
    path = latest_odds_html(str(race_id))
    if not path:
        return None
    text = path.read_bytes().decode("shift_jis", errors="ignore")
    candidates = extract_result_cnames_from_text(text)
    for cname in candidates:
        if "sde" in cname and result_cname_race_id(cname):
            return cname
    return None


def fetch_result_html(race_id: str, cname: str, out_dir: Path, sleep_sec: float = 0.2) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{stamp}_result.html"
    url = "https://www.jra.go.jp/JRADB/accessS.html?CNAME=" + cname
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        out_path.write_bytes(resp.read())
    if sleep_sec:
        time.sleep(sleep_sec)
    return out_path


def latest_result_html(race_id: str) -> Path | None:
    base = ROOT / "data" / "raw" / "jra_official_results" / race_id
    if not base.exists():
        return None
    files = sorted(base.glob("*_result.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files:
        text = path.read_bytes().decode("shift_jis", errors="ignore")
        if 'id="race_result"' in text and "refund_area" in text:
            return path
    return files[0] if files else None


def strip_tags(fragment: str) -> str:
    text = re.sub(r"<[^>]+>", "", fragment)
    return html.unescape(text).replace("\xa0", " ").strip()


def parse_result_page(path: Path) -> dict:
    text = path.read_bytes().decode("shift_jis", errors="ignore")
    result = {
        "path": str(path),
        "isFinal": False,
        "weather": "",
        "turfGoing": "",
        "dirtGoing": "",
        "finishOrder": [],
        "top2": [],
        "top3": [],
        "refunds": {},
    }
    for css_classes, key in [(["weather"], "weather"), (["turf"], "turfGoing"), (["dirt", "durt"], "dirtGoing")]:
        for css_class in css_classes:
            state_m = re.search(rf'<li class="{css_class}"[\s\S]*?<span class="txt">([\s\S]*?)</span>', text)
            if state_m:
                result[key] = strip_tags(state_m.group(1))
                break
    m = re.search(r'<div id="race_result"[\s\S]*?<tbody>([\s\S]*?)</tbody>', text)
    if m:
        body = m.group(1)
        for row in re.findall(r"<tr[\s\S]*?</tr>", body):
            place_m = re.search(r'<td class="place">([\s\S]*?)</td>', row)
            num_m = re.search(r'<td class="num">([\s\S]*?)</td>', row)
            horse_m = re.search(r'<td class="horse">([\s\S]*?)</td>', row)
            place = safe_int(strip_tags(place_m.group(1)) if place_m else None)
            horse_no = safe_int(strip_tags(num_m.group(1)) if num_m else None)
            horse_name = strip_tags(horse_m.group(1)) if horse_m else ""
            if place is not None and horse_no is not None:
                result["finishOrder"].append({"place": place, "horseNo": horse_no, "horseName": horse_name})
    result["finishOrder"].sort(key=lambda x: x["place"])
    result["top2"] = [x["horseNo"] for x in result["finishOrder"][:2]]
    result["top3"] = [x["horseNo"] for x in result["finishOrder"][:3]]
    result["isFinal"] = len(result["top3"]) >= 3

    area_m = re.search(r'<div class="refund_area[\s\S]*?</div>\s*</div>\s*</div>\s*<div class="horse_prof_area', text)
    area = area_m.group(0) if area_m else text
    class_to_type = {
        "win": "win",
        "place": "place",
        "wide": "wide",
        "umaren": "umaren",
        "wakuren": "wakuren",
        "umatan": "umatan",
        "trio": "trio",
        "tierce": "tierce",
    }
    for cls, ticket_type in class_to_type.items():
        li_m = re.search(rf'<li class="{cls}">([\s\S]*?)</li>', area)
        if not li_m:
            continue
        refunds = {}
        for line in re.findall(r'<div class="line">([\s\S]*?)</div>\s*</div>', li_m.group(1)):
            num_m = re.search(r'<div class="num">([\s\S]*?)</div>', line)
            yen_m = re.search(r'<div class="yen">([\s\S]*?)<span', line)
            if not num_m or not yen_m:
                continue
            combo = strip_tags(num_m.group(1)).replace(" ", "")
            yen_text = strip_tags(yen_m.group(1)).replace(",", "")
            try:
                yen = float(yen_text)
            except Exception:
                continue
            if ticket_type in {"wide", "umaren", "wakuren", "umatan"} and "-" in combo:
                combo = pair_key(*combo.split("-", 1))
            refunds[combo] = yen
        if refunds:
            result["refunds"][ticket_type] = refunds
    return result


def settle_ticket(ticket: dict, race_result: dict) -> dict:
    stake = float(ticket.get("stakeYen") or 0)
    ticket_type = ticket.get("ticketType")
    a_no = safe_int(ticket.get("aNo"))
    b_no = safe_int(ticket.get("bNo"))
    hit = False
    payout = 0.0
    official_pay100 = None

    if not race_result.get("isFinal"):
        status = "pending_result"
    else:
        status = "settled"
        if ticket_type in {"wide", "umaren"}:
            key = pair_key(a_no, b_no)
            official_pay100 = race_result.get("refunds", {}).get(ticket_type, {}).get(key)
            hit = official_pay100 is not None
            payout = (official_pay100 or 0.0) * stake / 100.0
        elif ticket_type == "win":
            official_pay100 = race_result.get("refunds", {}).get("win", {}).get(str(a_no))
            hit = official_pay100 is not None
            payout = (official_pay100 or 0.0) * stake / 100.0
        elif ticket_type == "place":
            official_pay100 = race_result.get("refunds", {}).get("place", {}).get(str(a_no))
            hit = official_pay100 is not None
            payout = (official_pay100 or 0.0) * stake / 100.0

    out = dict(ticket)
    out["resultStatus"] = status
    out["finishTop3"] = "-".join(str(x) for x in race_result.get("top3", []))
    out["officialPayPer100"] = official_pay100
    out["hit"] = hit
    out["payoutYen"] = round(payout, 1)
    out["profitYen"] = round(payout - stake if status == "settled" else 0.0, 1)
    return out


def summarize(rows: list[dict]) -> dict:
    settled = [r for r in rows if r.get("resultStatus") == "settled"]
    stake = sum(float(r.get("stakeYen") or 0) for r in settled)
    payout = sum(float(r.get("payoutYen") or 0) for r in settled)
    hits = sum(1 for r in settled if r.get("hit"))
    return {
        "tickets": len(settled),
        "races": len({r.get("raceId") for r in settled}),
        "stakeYen": round(stake, 1),
        "payoutYen": round(payout, 1),
        "profitYen": round(payout - stake, 1),
        "roiPct": round((payout / stake * 100.0), 1) if stake else None,
        "hitTickets": hits,
        "hitRatePct": round(hits / len(settled) * 100.0, 1) if settled else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard", default="outputs/ui/live_odds_dashboard.html")
    parser.add_argument("--out-dir", default="outputs/analysis/current_live_pnl")
    parser.add_argument("--now", default=None, help="YYYY-mm-dd HH:MM. Default: local now.")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--grace-minutes", type=int, default=8)
    parser.add_argument(
        "--update-official-laps",
        action="store_true",
        help="After fetching/saving JRA result HTML, refresh official race-lap CSVs and coverage audit.",
    )
    args = parser.parse_args()

    payload = read_dashboard_payload(ROOT / args.dashboard)
    tickets = build_displayed_tickets(payload)
    now = dt.datetime.strptime(args.now, "%Y-%m-%d %H:%M") if args.now else dt.datetime.now()
    cutoff = now - dt.timedelta(minutes=args.grace_minutes)

    race_map = {r.get("raceId"): r for r in payload.get("races", [])}
    active_races = []
    for race_id, race in race_map.items():
        rd = race_datetime(race)
        if rd and rd <= cutoff:
            active_races.append(race_id)

    result_by_race = {}
    for race_id in active_races:
        result_path = latest_result_html(race_id)
        if args.fetch:
            cname = extract_result_cname_from_odds(race_id)
            if cname:
                try:
                    result_path = fetch_result_html(
                        race_id,
                        cname,
                        ROOT / "data" / "raw" / "jra_official_results" / race_id,
                    )
                except Exception as exc:
                    print(f"WARN fetch failed {race_id}: {exc}")
        if result_path:
            result_by_race[race_id] = parse_result_page(result_path)

    settled_rows = []
    for ticket in tickets:
        race_id = ticket.get("raceId")
        rd = race_datetime(race_map.get(race_id, {}))
        if not rd or rd > cutoff:
            continue
        race_result = result_by_race.get(race_id, {"isFinal": False, "top2": [], "top3": [], "refunds": {}})
        settled_rows.append(settle_ticket(ticket, race_result))

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "current_live_pnl_detail.csv"
    result_track_path = out_dir / "current_result_track_conditions.csv"
    fields = [
        "dateKey",
        "raceId",
        "raceLabel",
        "startTime",
        "raceName",
        "decisionGroup",
        "decisionLabel",
        "action",
        "ticketType",
        "ticketLabel",
        "aNo",
        "bNo",
        "aName",
        "bName",
        "stakeYen",
        "liveOdds",
        "livePay",
        "resultStatus",
        "finishTop3",
        "officialPayPer100",
        "hit",
        "payoutYen",
        "profitYen",
        "reason",
    ]
    with detail_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(settled_rows)

    result_track_fields = [
        "race_id",
        "race_label",
        "date_key",
        "venue",
        "surface",
        "start_time",
        "weather",
        "turf_going",
        "dirt_going",
        "runtime_going",
        "source",
        "result_path",
        "fetched_at",
    ]
    result_track_rows = []
    for race_id in active_races:
        race = race_map.get(race_id, {})
        race_result = result_by_race.get(race_id, {})
        if not race_result.get("isFinal"):
            continue
        surface = str(race.get("surface") or "")
        turf_going = race_result.get("turfGoing", "")
        dirt_going = race_result.get("dirtGoing", "")
        if surface.startswith("芝"):
            runtime_going = turf_going
        elif surface.startswith("ダ"):
            runtime_going = dirt_going
        else:
            runtime_going = turf_going or dirt_going
        result_track_rows.append(
            {
                "race_id": race_id,
                "race_label": race.get("raceLabel", ""),
                "date_key": race.get("dateKey", ""),
                "venue": race.get("venue", ""),
                "surface": surface,
                "start_time": race.get("startTime", ""),
                "weather": race_result.get("weather", ""),
                "turf_going": turf_going,
                "dirt_going": dirt_going,
                "runtime_going": runtime_going,
                "source": "jra_result",
                "result_path": race_result.get("path", ""),
                "fetched_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
    with result_track_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result_track_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result_track_rows)

    official_lap_update = None
    if args.update_official_laps:
        cmd = [sys.executable, "scripts/update_official_lap_store.py"]
        result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        official_lap_update = {
            "command": cmd,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }

    groups = {
        "all_displayed": settled_rows,
        "final_buy_only": [r for r in settled_rows if r.get("decisionGroup") == "final_buy"],
        "reference_only": [r for r in settled_rows if str(r.get("decisionGroup", "")).startswith("reference")],
        "reference_candidate": [r for r in settled_rows if r.get("decisionGroup") == "reference_candidate"],
        "reference_watch": [r for r in settled_rows if r.get("decisionGroup") == "reference_watch"],
        "reference_weak": [r for r in settled_rows if r.get("decisionGroup") == "reference_weak"],
        "reference_skip": [r for r in settled_rows if r.get("decisionGroup") == "reference_skip"],
    }
    summary = {
        "generatedAt": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dashboardGeneratedAt": payload.get("generatedAt"),
        "latestSnapshotLabel": payload.get("latestSnapshotLabel"),
        "now": now.strftime("%Y-%m-%d %H:%M:%S"),
        "cutoff": cutoff.strftime("%Y-%m-%d %H:%M:%S"),
        "activeRaces": len(active_races),
        "resultFetchedRaces": sum(1 for r in result_by_race.values() if r.get("isFinal")),
        "pendingResultRaces": len(active_races) - sum(1 for r in result_by_race.values() if r.get("isFinal")),
        "resultTrackCsv": str(result_track_path),
        "officialLapUpdate": official_lap_update,
        "groups": {name: summarize(rows) for name, rows in groups.items()},
    }
    summary_path = out_dir / "current_live_pnl_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(str(detail_path))


if __name__ == "__main__":
    main()
