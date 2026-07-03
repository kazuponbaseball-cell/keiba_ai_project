from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


TRAINER_NAMES = {
    "427": "森秀行",
    "1055": "藤原英昭",
    "1061": "友道康夫",
    "1070": "堀宣行",
    "1075": "矢作芳人",
    "1098": "野中賢二",
    "1102": "大竹正博",
    "1117": "高野友和",
    "1126": "木村哲也",
    "1130": "吉村圭司",
    "1137": "中内田充正",
    "1144": "池添学",
    "1151": "斉藤崇史",
    "1152": "竹内正洋",
    "1157": "杉山晴紀",
    "1158": "寺島良",
    "1164": "安田翔伍",
    "1168": "上村洋行",
    "1176": "吉岡辰弥",
    "1183": "辻野泰之",
    "1039": "中竹和也",
    "1058": "大久保龍志",
    "1062": "藤岡健一",
    "1067": "久保田貴士",
    "1071": "池江泰寿",
    "1086": "斎藤誠",
    "1092": "松永幹夫",
    "1097": "鹿戸雄一",
    "1105": "須貝尚介",
    "1110": "清水久詞",
    "1113": "牧浦充徳",
    "1115": "菊沢隆徳",
    "1124": "大和田成",
    "1127": "栗田徹",
    "1140": "石橋守",
    "1146": "奥村豊",
    "1149": "松下武士",
    "1154": "橋口慎介",
    "1160": "武幸四郎",
    "1161": "武英智",
    "1162": "田中博康",
    "1169": "加藤士津八",
    "1172": "新谷功一",
    "1184": "四位洋文",
    "1186": "中村直也",
    "1191": "蛯名正義",
    "1195": "嘉藤貴行",
    "1203": "福永祐一",
}


GRADE_ORDER = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}


@dataclass
class WorkoutContext:
    entry: pd.Series
    workouts: pd.DataFrame
    trainer_code: str
    trainer_name: str
    horse_name: str
    surface: str
    jockey: str
    career: float
    interval: float
    class_name: str
    venue: str
    age: float
    popularity: float
    race_type: str
    track_code: str
    owner: str


def evaluate_workout_knowledge(entry: pd.Series, workouts: pd.DataFrame) -> dict[str, Any]:
    ctx = _context(entry, workouts)
    if ctx.trainer_code not in TRAINER_NAMES:
        return _result(ctx, "C", "対象外厩舎", [], ["ナレッジベース未登録の厩舎"], "登録済みの厩舎別勝負パターンがないため中立評価。")

    rule = {
        "1075": _eval_yahagi,
        "1157": _eval_sugiyama_haruki,
        "1061": _eval_tomomichi,
        "427": _eval_mori_hideyuki,
        "1070": _eval_hori,
        "1168": _eval_uemura,
        "1126": _eval_kimura,
        "1137": _eval_nakauchida,
        "1176": _eval_yoshioka,
        "1098": _eval_nonaka,
        "1144": _eval_ikezoe,
        "1117": _eval_takano,
        "1183": _eval_tsujino,
        "1152": _eval_takeuchi,
        "1151": _eval_saito_takashi,
        "1130": _eval_yoshimura,
        "1158": _eval_terashima,
        "1164": _eval_yasuda_shogo,
        "1055": _eval_fujiwara,
        "1102": _eval_otake,
        "1039": _eval_nakatake,
        "1058": _eval_okubo_ryuji,
        "1062": _eval_fujioka_kenichi,
        "1067": _eval_kubota,
        "1071": _eval_ikee,
        "1086": _eval_saito_makoto,
        "1092": _eval_matsunaga_mikio,
        "1097": _eval_shikato,
        "1105": _eval_sugai,
        "1110": _eval_shimizu,
        "1113": _eval_makiura,
        "1115": _eval_kikuzawa,
        "1124": _eval_owada,
        "1127": _eval_kurita,
        "1140": _eval_ishibashi,
        "1146": _eval_okumura_yutaka,
        "1149": _eval_matsushita,
        "1154": _eval_hashiguchi,
        "1160": _eval_take_koshiro,
        "1161": _eval_take_hidenori,
        "1162": _eval_tanaka_hiroyasu,
        "1169": _eval_kato_shizuya,
        "1172": _eval_shintani,
        "1184": _eval_shii,
        "1186": _eval_nakamura_naoya,
        "1191": _eval_ebina,
        "1195": _eval_kato_takayuki,
        "1203": _eval_fukunaga,
    }[ctx.trainer_code]
    return rule(ctx)


def prepare_workouts_for_knowledge(workouts: pd.DataFrame) -> pd.DataFrame:
    out = workouts.copy()
    out["workout_date_dt"] = _to_datetime(out["workout_date"])
    out["course_bucket"] = out.get("course", pd.Series("", index=out.index)).astype("string")
    out["penultimate_1f_sec"] = pd.to_numeric(out["final_2f_sec"], errors="coerce") - pd.to_numeric(
        out["final_1f_sec"], errors="coerce"
    )
    out["lap_group"] = _lap_group(out["penultimate_1f_sec"], out["final_1f_sec"])
    return out


def select_entry_workouts(entry: pd.Series, workouts: pd.DataFrame, *, lookback_days: int = 21) -> pd.DataFrame:
    horse_id = str(entry.get("血統登録番号", entry.get("horse_id", ""))).replace(".0", "")
    race_date = _to_datetime(pd.Series([entry.get("日付", entry.get("date"))])).iloc[0]
    if not horse_id or pd.isna(race_date):
        return workouts.iloc[0:0].copy()
    w = workouts[workouts["horse_id"].astype("string") == horse_id].copy()
    if "workout_date_dt" not in w.columns:
        w = prepare_workouts_for_knowledge(w)
    days = (race_date - w["workout_date_dt"]).dt.days
    return w[(days >= 0) & (days <= lookback_days)].sort_values("workout_date_dt", kind="mergesort")


def _context(entry: pd.Series, workouts: pd.DataFrame) -> WorkoutContext:
    trainer_code = _code(entry.get("調教師コード", entry.get("trainer_code", "")))
    return WorkoutContext(
        entry=entry,
        workouts=workouts,
        trainer_code=trainer_code,
        trainer_name=TRAINER_NAMES.get(trainer_code, str(trainer_code)),
        horse_name=str(entry.get("馬名", entry.get("horse_name", ""))),
        surface=str(entry.get("芝・ダ", entry.get("surface", ""))),
        jockey=str(entry.get("騎手", entry.get("jockey", ""))),
        career=_float(entry.get("キャリア", np.nan)),
        interval=_float(entry.get("間隔", np.nan)),
        class_name=str(entry.get("クラス名", "")),
        venue=str(entry.get("場所", "")),
        age=_float(entry.get("年齢", np.nan)),
        popularity=_float(entry.get("人気", np.nan)),
        race_type=str(entry.get("競走種別", "")),
        track_code=str(entry.get("トラックコード", "")),
        owner=str(entry.get("馬主名", entry.get("馬主", ""))),
    )


def _eval_yahagi(ctx: WorkoutContext) -> dict[str, Any]:
    main = _any_lap(ctx.workouts, ["A2"])
    hill_fast = _any(ctx.workouts, course="hill", total_max=53.9, lap=["A2"])
    career_ok = ctx.career >= 7
    grade = "S" if main and career_ok else "A" if main else "C"
    adds = _adds([(main, "A2（2Fとも12秒台加速）に該当"), (hill_fast, "坂路53秒台以下のA2"), (career_ok, "キャリア7戦目以降")])
    return _result(ctx, grade, "矢作芳人厩舎 A2勝負パターン", adds, [], "A2をここぞで使う厩舎。キャリア条件まで揃えば強く評価。")


def _eval_sugiyama_haruki(ctx: WorkoutContext) -> dict[str, Any]:
    a3 = _any_lap(ctx.workouts, ["A3"])
    jockey = "西村" in ctx.jockey
    return _result(ctx, "S" if a3 and jockey else "A" if a3 else "C", "杉山晴紀厩舎 A3", _adds([(a3, "A3（終い11秒台加速）"), (jockey, "西村淳也騎手")]), [], "終い11秒台加速を勝負サインとして評価。")


def _eval_tomomichi(ctx: WorkoutContext) -> dict[str, Any]:
    a1 = _any_lap(ctx.workouts, ["A1"])
    seq = _prev_weekend(ctx, lap=["A1"]) and _current_week(ctx, course="wood")
    a2 = _any_lap(ctx.workouts, ["A2"])
    return _result(ctx, "S" if seq else "A" if a1 else "C", "友道康夫厩舎 A1", _adds([(a1, "A1（終いだけ伸ばす）"), (seq, "前週土曜A1から当週ウッド")]), _adds([(a2, "A2は過大評価しない")]), "派手な時計ではなく地味な終い加速を重視。")


def _eval_mori_hideyuki(ctx: WorkoutContext) -> dict[str, Any]:
    lap = _any_lap(ctx.workouts, ["A3", "A2", "B3"])
    ideal_clock = _any(ctx.workouts, course="hill", total_min=52.0, total_max=53.9, lap=["A3", "A2", "B3"])
    return _result(ctx, "A" if lap else "C", "森秀行厩舎 ラップ重視", _adds([(lap, "A3/A2/B3の好ラップ"), (ideal_clock, "坂路52〜53秒台の好ラップ")]), [], "49秒台の速さより、52〜53秒台でラップがまとまる形を評価。")


def _eval_hori(ctx: WorkoutContext) -> dict[str, Any]:
    wood53 = _any(ctx.workouts, course="wood", total_max=53.9)
    prev_hill = _prev_weekend(ctx, course="hill", total_max=55.9)
    return _result(ctx, "S" if wood53 and prev_hill else "A" if wood53 else "C", "堀宣行厩舎 美浦W4F53秒台以下", _adds([(wood53, "ウッド4F相当53秒台以下"), (prev_hill, "前週土日坂路あり")]), [], "美浦ウッド4F時計を最優先。")


def _eval_uemura(ctx: WorkoutContext) -> dict[str, Any]:
    hill_a2 = _any(ctx.workouts, course="hill", total_max=53.9, lap=["A2"])
    wood_11 = _any(ctx.workouts, course="wood", final1_max=11.9)
    prev_hill = _prev_weekend(ctx, course="hill")
    return _result(ctx, "S" if (hill_a2 or wood_11) and prev_hill else "A" if (hill_a2 or wood_11) else "C", "上村洋行厩舎 坂路A2/ウッド終い11秒台", _adds([(hill_a2, "坂路A2かつ53秒台以下"), (wood_11, "ウッド終い11秒台"), (prev_hill, "前週土日坂路あり")]), [], "坂路とウッドを併用する厩舎。ウッド終い11秒台を高評価。")


def _eval_kimura(ctx: WorkoutContext) -> dict[str, Any]:
    seq = _prev_weekend(ctx, course="hill", total_max=55.9) and _current_week(ctx, course="wood") and _any_lap(ctx.workouts, ["A3", "A2", "A1"])
    lemaire = "ルメール" in ctx.jockey
    return _result(ctx, "S" if seq and lemaire else "A" if seq else "C", "木村哲也厩舎 前週坂路→当週ウッド", _adds([(seq, "前週坂路55秒台以下から当週ウッド加速ラップ"), (lemaire, "ルメール騎手")]), [], "前週で負荷をかけ、当週ウッドで整える形を評価。")


def _eval_nakauchida(ctx: WorkoutContext) -> dict[str, Any]:
    prev_hill = _prev_weekend(ctx, course="hill", total_max=55.9)
    kawada = "川田" in ctx.jockey
    debut = ctx.career <= 1 and _any(ctx.workouts, course="wood", total_max=67.0)
    return _result(ctx, "S" if prev_hill and kawada else "A" if prev_hill or debut else "C", "中内田充正厩舎 前週土日坂路", _adds([(prev_hill, "前週土日坂路55秒台以下"), (kawada, "川田将雅騎手"), (debut, "新馬戦想定の栗東W5F67秒以下")]), [], "前週土日の坂路負荷を最重視。")


def _eval_yoshioka(ctx: WorkoutContext) -> dict[str, Any]:
    prev_fast = _prev_weekend(ctx, course="hill", total_min=51.0, total_max=52.9)
    return _result(ctx, "A" if prev_fast else "C", "吉岡辰弥厩舎 前週51〜52秒台", _adds([(prev_fast, "前週土日坂路51〜52秒台")]), [], "前週土日に仕上げる厩舎。当週軽くても割り引かない。")


def _eval_nonaka(ctx: WorkoutContext) -> dict[str, Any]:
    dirt = "ダ" in ctx.surface
    a = _any_lap(ctx.workouts, ["A1", "A2"])
    slow_prev = _prev_weekend(ctx, course="hill", total_min=60.0)
    return _result(ctx, "D" if not dirt else "S" if a and slow_prev else "A" if a else "C", "野中賢二厩舎 ダートA1/A2", _adds([(dirt, "ダート"), (a, "A1/A2"), (slow_prev, "前週土日坂路60秒以上")]), _adds([(not dirt, "芝は大幅割引")]), "ダート専用評価。芝では強く割り引く。")


def _eval_ikezoe(ctx: WorkoutContext) -> dict[str, Any]:
    a3 = _any_lap(ctx.workouts, ["A3"])
    return _result(ctx, "A" if a3 else "C", "池添学厩舎 A3", _adds([(a3, "終い11秒台加速")]), [], "A3を出した時を本気サインとして評価。")


def _eval_takano(ctx: WorkoutContext) -> dict[str, Any]:
    turf = "芝" in ctx.surface
    pattern = _prev_weekend(ctx, course="hill", total_max=58.9) and _any_lap(ctx.workouts, ["A1"])
    return _result(ctx, "S" if turf and pattern else "A" if pattern else "C", "高野友和厩舎 前週坂路→A1", _adds([(turf, "芝"), (pattern, "前週土日58秒以下からA1")]), [], "派手な時計より地味な時計とA1を評価。")


def _eval_tsujino(ctx: WorkoutContext) -> dict[str, Any]:
    turf = "芝" in ctx.surface
    accel = _any_lap(ctx.workouts, ["A1", "A2", "A3"])
    b1 = _any_lap(ctx.workouts, ["B1"])
    return _result(ctx, "D" if b1 else "A" if turf and accel else "C", "辻野泰之厩舎 芝加速ラップ", _adds([(turf, "芝"), (accel, "A1/A2/A3")]), _adds([(b1, "B1は消し")]), "芝の加速ラップなら買い、B1は消し。")


def _eval_takeuchi(ctx: WorkoutContext) -> dict[str, Any]:
    wood67 = _any(ctx.workouts, course="wood", total_max=67.0)
    best = _any(ctx.workouts, course="wood", total_max=65.9, final1_max=11.9)
    return _result(ctx, "S" if best else "A" if wood67 else "C", "竹内正洋厩舎 美浦W5F時計", _adds([(wood67, "美浦W5F相当67秒以下"), (best, "65秒台＋終い11秒台")]), [], "ラップ分類より時計を優先。")


def _eval_saito_takashi(ctx: WorkoutContext) -> dict[str, Any]:
    turf = "芝" in ctx.surface
    dirt = "ダ" in ctx.surface
    turf_pattern = turf and _any(ctx.workouts, course="wood", total_max=67.0) and _current_week(ctx, course="hill")
    dirt_pattern = dirt and _any_lap(ctx.workouts, ["A1", "A2"])
    return _result(ctx, "A" if turf_pattern or dirt_pattern else "C", "斉藤崇史厩舎 芝ダート別判定", _adds([(turf_pattern, "芝: W5F67秒以下＋水〜金坂路調整"), (dirt_pattern, "ダート: A1/A2")]), [], "芝とダートで調教の意味を分ける。")


def _eval_yoshimura(ctx: WorkoutContext) -> dict[str, Any]:
    a2 = _any_lap(ctx.workouts, ["A2"])
    return _result(ctx, "A" if a2 else "C", "吉村圭司厩舎 A2", _adds([(a2, "A2")]), [], "A2だけを強く見る厩舎。併せ先着は現データでは未判定。")


def _eval_terashima(ctx: WorkoutContext) -> dict[str, Any]:
    dirt = "ダ" in ctx.surface
    main = _prev_weekend(ctx, lap=["A1"]) and _current_week(ctx, course="wood") and dirt
    derived = _prev_weekend(ctx, lap=["A2"])
    return _result(ctx, "S" if main else "B" if derived else "C", "寺島良厩舎 前週本番", _adds([(main, "前週A1→当週ウッド→ダート"), (derived, "前週A2")]), [], "前週情報が必須の厩舎。")


def _eval_yasuda_shogo(ctx: WorkoutContext) -> dict[str, Any]:
    wood52 = _any(ctx.workouts, course="wood", total_max=52.9)
    wood51 = _any(ctx.workouts, course="wood", total_max=51.9)
    exception = "横山典" in ctx.jockey
    return _result(ctx, "S" if wood51 else "A" if wood52 else "C", "安田翔伍厩舎 ウッド時計", _adds([(wood52, "栗東W4F相当52秒以下"), (wood51, "51秒台"), (exception, "横山典弘騎手は例外扱い")]), _adds([(_any(ctx.workouts, course="hill") and not exception, "坂路は過信しない")]), "坂路ではなくウッド時計を最優先。")


def _eval_fujiwara(ctx: WorkoutContext) -> dict[str, Any]:
    eleven = _any_lap(ctx.workouts, ["A3", "B3"]) and _any(ctx.workouts, course="wood")
    prev_day_hill = _prev_day(ctx, course="hill") and any(v in ctx.venue for v in ["京都", "阪神", "中京"])
    return _result(ctx, "D" if prev_day_hill else "A" if eleven else "C", "藤原英昭厩舎 11秒台ウッド", _adds([(eleven, "ウッドでA3/B3の11秒台")]), _adds([(prev_day_hill, "京都/阪神/中京で前日坂路追い")]), "11秒台が出たら勝負。前日坂路追いは割引。")


def _eval_otake(ctx: WorkoutContext) -> dict[str, Any]:
    turf = "芝" in ctx.surface
    pattern = _prev_weekend(ctx, course="hill", total_max=57.9) and _current_week(ctx, course="wood") and turf
    prev_day = _prev_day(ctx, course="hill")
    layoff = ctx.interval >= 12
    return _result(ctx, "D" if not turf else "S" if pattern and prev_day and layoff else "A" if pattern else "C", "大竹正博厩舎 芝専用", _adds([(pattern, "前週土日坂路57秒台以下→当週ウッド→芝"), (prev_day, "前日坂路"), (layoff, "長期休養明け")]), _adds([(not turf, "ダートは大幅割引")]), "芝専用で評価。ダートは強く割り引く。")


def _eval_ikee(ctx: WorkoutContext) -> dict[str, Any]:
    main = _any_lap(ctx.workouts, ["A2", "A3"])
    weekend_load = _prev_weekend(ctx, course="hill", total_max=59.9)
    key_jockey = _jockey_has(ctx, ["川田", "松山"])
    pop1 = ctx.popularity == 1
    decel = _any_lap(ctx.workouts, ["B1", "B2", "B3"])
    grade = "S" if main and (weekend_load or key_jockey or pop1) else "A" if main else "D" if decel and not weekend_load else "C"
    return _result(ctx, grade, "池江泰寿厩舎 坂路A2/A3", _adds([(main, "A2/A3"), (weekend_load, "前週土日坂路60秒未満"), (key_jockey, "川田/松山騎乗"), (pop1, "1番人気")]), _adds([(decel, "減速ラップ"), (not weekend_load, "前週土日の負荷不足")]), "栗東坂路でA2/A3を重視。人気馬なら素直に評価する。")


def _eval_shimizu(ctx: WorkoutContext) -> dict[str, Any]:
    obstacle = _is_obstacle(ctx)
    wood66_67 = _any(ctx.workouts, course="wood", total_min=66.0, total_max=67.9)
    return _result(ctx, "A" if obstacle and wood66_67 else "C", "清水久詞厩舎 障害戦+栗東ウッド", _adds([(obstacle, "障害戦"), (wood66_67, "ウッド5F66-67秒台")]), [], "平地戦は調教評価を過信せず中立。障害戦のみ加点する。")


def _eval_saito_makoto(ctx: WorkoutContext) -> dict[str, Any]:
    prev = _prev_weekend(ctx, course="hill", total_min=56.0, total_max=59.9)
    current_fast = _any(_current_week_frame(ctx), course="hill", total_max=53.9)
    return _result(ctx, "S" if prev and current_fast else "A" if current_fast else "C", "斎藤誠厩舎 前週負荷+当週坂路速時計", _adds([(prev, "前週土日坂路56-59秒台"), (current_fast, "当週坂路53秒台以下")]), [], "土日に負荷を掛け、当週も速い時計を出した時を高評価。")


def _eval_nakatake(ctx: WorkoutContext) -> dict[str, Any]:
    prev_hill = _prev_weekend(ctx, course="hill")
    hill51 = _any(ctx.workouts, course="hill", total_max=51.9)
    wood65 = _any(ctx.workouts, course="wood", total_max=65.9)
    prev_day = _prev_day(ctx)
    layoff = ctx.interval >= 12
    strong = hill51 or wood65
    grade = "S" if prev_hill and strong and prev_day else "A" if prev_hill and strong else "D" if layoff and not strong else "C"
    return _result(ctx, grade, "中竹和也厩舎 前週土日坂路+好時計", _adds([(prev_hill, "前週土日坂路あり"), (hill51, "栗東坂路51秒台以下"), (wood65, "栗東W5F65秒台以下"), (prev_day, "前日追い")]), _adds([(layoff, "長期休養明けは割引")]), "前週土日調教を重視。当週だけで判断しない。")


def _eval_take_hidenori(ctx: WorkoutContext) -> dict[str, Any]:
    wood53 = _any(ctx.workouts, course="wood", total_max=53.9)
    wood51 = _any(ctx.workouts, course="wood", total_max=51.9)
    no_prev_wood = not _prev_weekend(ctx, course="wood")
    local_hill = any(v in ctx.venue for v in ["札幌", "函館"]) and _any(ctx.workouts, course="hill")
    hill_only = _any(ctx.workouts, course="hill") and not _any(ctx.workouts, course="wood") and not local_hill
    grade = "S" if wood51 else "A" if wood53 or local_hill else "D" if hill_only else "C"
    return _result(ctx, grade, "武英智厩舎 ウッド時計重視", _adds([(wood53, "栗東W4F相当53秒台以下"), (wood51, "W4F51秒台"), (no_prev_wood and wood53, "前週土日ウッド未使用"), (local_hill, "札幌/函館の坂路追い")]), _adds([(hill_only, "坂路主体は過信しない")]), "栗東では珍しいウッド重視。坂路だけの評価は控えめにする。")


def _eval_owada(ctx: WorkoutContext) -> dict[str, Any]:
    wood = _any(ctx.workouts, course="wood")
    prev_hill = _prev_weekend(ctx, course="hill")
    wood67 = _any(ctx.workouts, course="wood", total_max=67.0)
    hara_owner = "原" in ctx.owner
    return _result(ctx, "S" if hara_owner and wood67 else "A" if wood and prev_hill else "C", "大和田成厩舎 美浦ウッド+前週坂路", _adds([(wood, "ウッド追い"), (prev_hill, "前週土日坂路あり"), (wood67, "W5F67秒以下"), (hara_owner, "原オーナー所有馬")]), [], "美浦ウッドを重視。馬主情報が入る場合は原オーナー馬を追加加点する。")


def _eval_matsunaga_mikio(ctx: WorkoutContext) -> dict[str, Any]:
    hill_accel = _any(ctx.workouts, course="hill", lap=["A2", "A1", "A3"])
    wood_under70 = _any(ctx.workouts, course="wood", total_max=69.9)
    wood_slow = _any(ctx.workouts, course="wood", total_min=70.0)
    return _result(ctx, "A" if hill_accel or wood_under70 else "D" if wood_slow else "C", "松永幹夫厩舎 坂路/ウッド万能型", _adds([(hill_accel, "坂路A2/A1/A3"), (wood_under70, "ウッド5F70秒未満")]), _adds([(wood_slow, "W5F70秒以上は割引")]), "加速ラップを高評価。ウッドも70秒未満なら評価対象にする。")


def _eval_sugai(ctx: WorkoutContext) -> dict[str, Any]:
    a2 = _any_lap(ctx.workouts, ["A2"])
    popular = ctx.popularity <= 3
    young_special = ctx.age <= 3 and _prev_weekend(ctx, course="hill") and _current_week(ctx, course="wood") and _any(ctx.workouts, final1_max=11.9)
    return _result(ctx, "S" if young_special else "A" if a2 and popular else "B" if a2 else "C", "須貝尚介厩舎 人気馬A2/若駒特殊", _adds([(a2, "A2"), (popular, "人気上位"), (young_special, "2-3歳+前週坂路+当週ウッド+終い11秒台")]), [], "人気馬で買う厩舎。若駒の特殊条件は高評価する。")


def _eval_fujioka_kenichi(ctx: WorkoutContext) -> dict[str, Any]:
    prev57 = _prev_weekend(ctx, course="hill", total_max=57.9)
    return _result(ctx, "A" if prev57 else "C", "藤岡健一厩舎 前週土日坂路", _adds([(prev57, "前週土日坂路57秒台以下")]), [], "前週土日の坂路負荷を重視。併せ遅れだけでは減点しない。")


def _eval_take_koshiro(ctx: WorkoutContext) -> dict[str, Any]:
    a2 = _any(ctx.workouts, course="hill", lap=["A2"])
    wood51 = _any(ctx.workouts, course="wood", total_max=51.9)
    overpop = _jockey_has(ctx, ["武豊", "ルメール"]) and ctx.popularity <= 3
    return _result(ctx, "S" if wood51 else "A" if a2 else "C", "武幸四郎厩舎 坂路A2/ウッド好時計", _adds([(a2, "坂路A2"), (wood51, "W4F51秒台")]), _adds([(overpop, "武豊/ルメール騎乗時は過剰人気注意")]), "坂路ならA2、ウッドなら51秒台を大きく評価する。")


def _eval_shii(ctx: WorkoutContext) -> dict[str, Any]:
    main = _any_lap(ctx.workouts, ["A2", "A3"])
    prev57 = _prev_weekend(ctx, course="hill", total_max=57.0)
    decel = _any_lap(ctx.workouts, ["B1", "B2", "B3"])
    return _result(ctx, "S" if main and prev57 else "A" if main else "D" if decel else "C", "四位洋文厩舎 A2/A3", _adds([(main, "A2/A3"), (prev57, "前週土日坂路57秒以下")]), _adds([(decel, "減速ラップは評価しない")]), "加速ラップ重視。前週坂路が速ければ追加加点。")


def _eval_ebina(ctx: WorkoutContext) -> dict[str, Any]:
    prev55 = _prev_weekend(ctx, course="hill", total_max=55.9)
    slow = _prev_weekend(ctx, course="hill", total_min=60.0)
    return _result(ctx, "A" if prev55 else "D" if slow else "C", "蛯名正義厩舎 前週土日坂路", _adds([(prev55, "前週土日坂路55秒台以下")]), _adds([(slow, "前週土日坂路60秒以上は割引")]), "美浦坂路改修後の上昇を反映し、時計が速いほど加点する。")


def _eval_tanaka_hiroyasu(ctx: WorkoutContext) -> dict[str, Any]:
    prev55 = _prev_weekend(ctx, course="hill", total_max=55.0)
    tosaki = _jockey_has(ctx, ["戸崎"])
    return _result(ctx, "S" if prev55 and tosaki else "A" if prev55 else "C", "田中博康厩舎 前週坂路+戸崎", _adds([(prev55, "前週土日坂路55秒以下"), (tosaki, "戸崎騎乗")]), _adds([(not _is_dirt(ctx), "ダート寄り厩舎のため芝では過信注意")]), "ダートで強く、前週坂路55秒以下を高評価。")


def _eval_nakamura_naoya(ctx: WorkoutContext) -> dict[str, Any]:
    decel = _any_lap(ctx.workouts, ["B2", "B3"])
    turf_fast = _is_turf(ctx) and _any(ctx.workouts, course="hill", total_max=51.0)
    return _result(ctx, "S" if decel and turf_fast else "A" if decel else "C", "中村直也厩舎 減速ラップ型", _adds([(decel, "B2/B3"), (turf_fast, "芝+栗東坂路51秒以下")]), [], "珍しい減速ラップ型としてB2/B3を高評価する。")


def _eval_okubo_ryuji(ctx: WorkoutContext) -> dict[str, Any]:
    main = _any_lap(ctx.workouts, ["A3", "B3"])
    wood51 = _any(ctx.workouts, course="wood", total_max=51.0)
    popular = ctx.popularity <= 3
    return _result(ctx, "S" if popular and main else "A" if main or wood51 else "C", "大久保龍志厩舎 人気上位A3/B3", _adds([(main, "A3/B3"), (wood51, "W4F51秒以下"), (popular, "人気上位")]), [], "人気上位馬をしっかり勝たせる厩舎として評価する。")


def _eval_hashiguchi(ctx: WorkoutContext) -> dict[str, Any]:
    main = _any_lap(ctx.workouts, ["A2", "B2"])
    fast = _any(ctx.workouts, total_max=53.0)
    return _result(ctx, "S" if main and fast else "A" if main else "C", "橋口慎介厩舎 A2/B2", _adds([(main, "A2/B2"), (fast, "全体時計53秒以下")]), [], "A2とB2を使い分ける厩舎として評価する。")


def _eval_kurita(ctx: WorkoutContext) -> dict[str, Any]:
    prev_day_hill = _prev_day(ctx, course="hill")
    venue = any(v in ctx.venue for v in ["東京", "中山"])
    return _result(ctx, "S" if prev_day_hill and venue else "A" if prev_day_hill else "C", "栗田徹厩舎 前日美浦坂路", _adds([(prev_day_hill, "前日坂路追い"), (venue, "東京/中山")]), [], "前日坂路追いを最重要視。東京・中山で追加加点する。")


def _eval_okumura_yutaka(ctx: WorkoutContext) -> dict[str, Any]:
    slow_wood = _any(ctx.workouts, course="wood", total_min=70.0)
    prev_day = _prev_day(ctx, course="hill")
    return _result(ctx, "S" if slow_wood and prev_day else "A" if slow_wood else "C", "奥村豊厩舎 遅いウッド時計", _adds([(slow_wood, "W5F70秒以上"), (prev_day, "前日坂路追い")]), [], "遅いウッド時計で走る珍しい厩舎として扱う。")


def _eval_kato_takayuki(ctx: WorkoutContext) -> dict[str, Any]:
    slow_hill = _prev_weekend(ctx, course="hill", total_min=60.0)
    prev_wood = _prev_weekend(ctx, course="wood")
    fast_hill = _prev_weekend(ctx, course="hill", total_max=55.9)
    return _result(ctx, "A" if slow_hill or prev_wood else "D" if fast_hill else "C", "嘉藤貴行厩舎 美浦ウッド/土日ゆったり", _adds([(slow_hill, "前週土日坂路60秒以上"), (prev_wood, "前週土日ウッド")]), _adds([(fast_hill, "速すぎる土日坂路は割引")]), "美浦ウッド主体。土日の速すぎる坂路は過信しない。")


def _eval_ishibashi(ctx: WorkoutContext) -> dict[str, Any]:
    a3 = _any_lap(ctx.workouts, ["A3"])
    a2 = _any_lap(ctx.workouts, ["A2"])
    decel = _any_lap(ctx.workouts, ["B1", "B2", "B3"])
    return _result(ctx, "A" if a3 else "B" if a2 else "D" if decel else "C", "石橋守厩舎 A3/A2", _adds([(a3, "A3"), (a2, "A2")]), _adds([(decel, "減速ラップは消し")]), "加速ラップ重視。A3を最優先する。")


def _eval_makiura(ctx: WorkoutContext) -> dict[str, Any]:
    a1 = _any_lap(ctx.workouts, ["A1"])
    derived = _any_lap(ctx.workouts, ["A2", "B2"])
    return _result(ctx, "A" if a1 else "B" if derived else "C", "牧浦充徳厩舎 終い重点A1", _adds([(a1, "A1"), (derived, "A2/B2派生")]), [], "終い重点型としてA1を最優先する。")


def _eval_matsushita(ctx: WorkoutContext) -> dict[str, Any]:
    a3 = _any_lap(ctx.workouts, ["A3"])
    a2 = _any_lap(ctx.workouts, ["A2"])
    switch = _older_course(ctx, "hill") and _latest_course(ctx) == "wood"
    return _result(ctx, "S" if a3 and switch else "A" if a3 or a2 else "C", "松下武士厩舎 A3/A2", _adds([(a3, "A3"), (a2, "A2"), (switch, "過去坂路→今回ウッドのコース変更")]), [], "矢作厩舎系統としてA3を最優先。調教コース変更も見る。")


def _eval_shintani(ctx: WorkoutContext) -> dict[str, Any]:
    b2 = _any_lap(ctx.workouts, ["B2"])
    b3 = _any_lap(ctx.workouts, ["B3"])
    fast = _any(ctx.workouts, course="hill", total_max=51.0)
    return _result(ctx, "S" if b2 and fast else "A" if b2 or b3 else "C", "新谷功一厩舎 減速ラップ型", _adds([(b2, "B2"), (b3, "B3"), (fast, "坂路51秒以下")]), [], "B2を最優先する減速ラップ型として評価する。")


def _eval_kikuzawa(ctx: WorkoutContext) -> dict[str, Any]:
    main = _any_lap(ctx.workouts, ["A2", "A1"])
    b = _any_lap(ctx.workouts, ["B1", "B2", "B3"])
    return _result(ctx, "A" if main else "D" if b else "C", "菊沢隆徳厩舎 A2/A1", _adds([(main, "A2/A1")]), _adds([(b, "B系は消し")]), "加速ラップ重視。A2を最優先する。")


def _eval_shikato(ctx: WorkoutContext) -> dict[str, Any]:
    prev55 = _prev_weekend(ctx, course="hill", total_max=55.0)
    prev_accel = _prev_weekend(ctx, lap=["A2", "A3"])
    current_wood = _current_week(ctx, course="wood")
    return _result(ctx, "S" if prev55 and prev_accel and current_wood else "A" if prev55 else "C", "鹿戸雄一厩舎 前週土日坂路", _adds([(prev55, "前週土日55秒以下"), (prev_accel, "前週A2/A3"), (current_wood, "当週ウッド")]), [], "前週土日坂路を重視。前週加速ラップから当週ウッドなら強く評価する。")


def _eval_kubota(ctx: WorkoutContext) -> dict[str, Any]:
    prev_day_hill = _prev_day(ctx, course="hill")
    tanabe = _jockey_has(ctx, ["田辺"])
    return _result(ctx, "S" if prev_day_hill and tanabe else "A" if prev_day_hill else "C", "久保田貴士厩舎 前日坂路", _adds([(prev_day_hill, "前日美浦坂路追い"), (tanabe, "田辺騎乗")]), [], "前日坂路追いを最重要視。田辺騎乗なら大幅加点。")


def _eval_kato_shizuya(ctx: WorkoutContext) -> dict[str, Any]:
    hill = _any(ctx.workouts, course="hill")
    return _result(ctx, "A" if hill else "C", "加藤士津八厩舎 美浦坂路主体", _adds([(hill, "坂路主体")]), [], "美浦坂路主体なら加点する。")


def _eval_fukunaga(ctx: WorkoutContext) -> dict[str, Any]:
    accel = _any_lap(ctx.workouts, ["A2", "A1"])
    decel = _any_lap(ctx.workouts, ["B1", "B2", "B3"])
    return _result(ctx, "A" if accel else "D" if decel else "C", "福永祐一厩舎 加速ラップ型", _adds([(accel, "A2/A1")]), _adds([(decel, "減速ラップは割引")]), "加速ラップを高評価し、減速ラップは割り引く。")


def _result(ctx: WorkoutContext, grade: str, pattern: str, adds: list[str], minuses: list[str], comment: str) -> dict[str, Any]:
    content = _content_summary(ctx.workouts)
    return {
        "horse_name": ctx.horse_name,
        "trainer": ctx.trainer_name,
        "trainer_code": ctx.trainer_code,
        "surface": ctx.surface,
        "workout_content": content,
        "matched_pattern": pattern,
        "plus_factors": adds,
        "minus_factors": minuses,
        "grade": grade,
        "grade_score": GRADE_ORDER.get(grade, 0),
        "comment": comment,
    }


def _adds(items: list[tuple[bool, str]]) -> list[str]:
    return [text for ok, text in items if ok]


def _any_lap(w: pd.DataFrame, laps: list[str]) -> bool:
    return bool(len(w) and w["lap_group"].isin(laps).any())


def _any(
    w: pd.DataFrame,
    *,
    course: str | None = None,
    total_min: float | None = None,
    total_max: float | None = None,
    final1_max: float | None = None,
    lap: list[str] | None = None,
) -> bool:
    part = w
    if course:
        part = part[part["course_bucket"].astype(str).str.contains(course, case=False, na=False)]
    if total_min is not None:
        part = part[pd.to_numeric(part["total_time_sec"], errors="coerce") >= total_min]
    if total_max is not None:
        part = part[pd.to_numeric(part["total_time_sec"], errors="coerce") <= total_max]
    if final1_max is not None:
        part = part[pd.to_numeric(part["final_1f_sec"], errors="coerce") <= final1_max]
    if lap:
        part = part[part["lap_group"].isin(lap)]
    return bool(len(part))


def _prev_weekend(ctx: WorkoutContext, *, course: str | None = None, total_min: float | None = None, total_max: float | None = None, lap: list[str] | None = None) -> bool:
    part = _prev_weekend_frame(ctx)
    return _any(part, course=course, total_min=total_min, total_max=total_max, lap=lap)


def _current_week(ctx: WorkoutContext, *, course: str | None = None) -> bool:
    return _any(_current_week_frame(ctx), course=course)


def _current_week_frame(ctx: WorkoutContext) -> pd.DataFrame:
    race_date = _race_date(ctx)
    if pd.isna(race_date):
        return ctx.workouts.iloc[0:0]
    start = race_date - pd.Timedelta(days=int(race_date.weekday()))
    return ctx.workouts[(ctx.workouts["workout_date_dt"] >= start) & (ctx.workouts["workout_date_dt"] <= race_date)]


def _prev_day(ctx: WorkoutContext, *, course: str | None = None) -> bool:
    race_date = _race_date(ctx)
    part = ctx.workouts[ctx.workouts["workout_date_dt"] == race_date - pd.Timedelta(days=1)]
    return _any(part, course=course)


def _prev_weekend_frame(ctx: WorkoutContext) -> pd.DataFrame:
    race_date = _race_date(ctx)
    if pd.isna(race_date):
        return ctx.workouts.iloc[0:0]
    week_start = race_date - pd.Timedelta(days=int(race_date.weekday()))
    prev_sat = week_start - pd.Timedelta(days=2)
    prev_sun = week_start - pd.Timedelta(days=1)
    return ctx.workouts[ctx.workouts["workout_date_dt"].isin([prev_sat, prev_sun])]


def _latest_course(ctx: WorkoutContext) -> str:
    if ctx.workouts.empty:
        return ""
    latest = ctx.workouts.sort_values("workout_date_dt").iloc[-1]
    return str(latest.get("course_bucket", ""))


def _older_course(ctx: WorkoutContext, course: str) -> bool:
    if len(ctx.workouts) <= 1:
        return False
    ordered = ctx.workouts.sort_values("workout_date_dt")
    older = ordered.iloc[:-1]
    return _any(older, course=course)


def _jockey_has(ctx: WorkoutContext, names: list[str]) -> bool:
    return any(name in ctx.jockey for name in names)


def _is_turf(ctx: WorkoutContext) -> bool:
    return "芝" in ctx.surface or str(ctx.track_code).startswith("1")


def _is_dirt(ctx: WorkoutContext) -> bool:
    return "ダ" in ctx.surface or str(ctx.track_code).startswith("2")


def _is_obstacle(ctx: WorkoutContext) -> bool:
    return "障" in ctx.race_type or str(ctx.track_code).startswith("3")


def _race_date(ctx: WorkoutContext) -> pd.Timestamp:
    return _to_datetime(pd.Series([ctx.entry.get("日付", ctx.entry.get("date"))])).iloc[0]


def _content_summary(w: pd.DataFrame) -> str:
    if w.empty:
        return "該当調教なし"
    latest = w.sort_values("workout_date_dt").iloc[-1]
    return (
        f"{latest.get('workout_date')} {latest.get('course_bucket')} "
        f"{latest.get('total_time_sec')}秒 "
        f"終い2F {latest.get('final_2f_sec')} / 1F {latest.get('final_1f_sec')} "
        f"{latest.get('lap_group')}（本数{len(w)}）"
    )


def _lap_group(penultimate_1f: pd.Series, final_1f: pd.Series) -> pd.Series:
    second = pd.to_numeric(penultimate_1f, errors="coerce")
    last = pd.to_numeric(final_1f, errors="coerce")
    accel = last < second
    decel = last > second
    second_12 = second.between(12.0, 12.999, inclusive="both")
    last_12 = last.between(12.0, 12.999, inclusive="both")
    second_11 = second.between(11.0, 11.999, inclusive="both")
    last_11 = last.between(11.0, 11.999, inclusive="both")
    return pd.Series(
        np.select(
            [
                accel & last_11,
                decel & second_11 & last_11,
                accel & second_12 & last_12,
                decel & second_12 & last_12,
                accel & ~second_12 & ~second_11 & last_12,
                decel & second_12 & ~last_12 & ~last_11,
            ],
            ["A3", "B3", "A2", "B2", "A1", "B1"],
            default="other",
        ),
        index=final_1f.index,
    )


def _to_datetime(values: pd.Series) -> pd.Series:
    text = values.astype("string").str.replace(r"\.0$", "", regex=True)
    text = text.where(~text.str.fullmatch(r"\d{6}", na=False), "20" + text)
    return pd.to_datetime(text, errors="coerce", format="mixed")


def _code(value: Any) -> str:
    text = str(value).replace(".0", "").strip()
    return text.lstrip("0") or text


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan
