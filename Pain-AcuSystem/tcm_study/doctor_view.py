from __future__ import annotations


from difflib import SequenceMatcher
import re

from tcm_study import schema as S
from tcm_study.needling import PREG_CUE, _age_sex

_DIFF_RE = re.compile(r"需鉴别[：:]|需(?:优先)?排除[：:]")
_DIFF_BLOCK_RE = re.compile(r"(?P<label>需鉴别|需(?:优先)?排除)[：:](?P<body>[^。\n]*)(?P<end>。?)")
_NUM_PREFIX = re.compile(r"^\s*(?:[-*•·]|\d+[\.．、)）]|[①②③④⑤⑥⑦⑧⑨])\s*")
_OLD_HEADER = re.compile(r"^(?:禁忌(?:与特殊人群|与风险警示)?|建议完善检查|操作前后注意事项|患者教育)[：:]\s*")
_EXAM_RE = re.compile(r"检查|复查|影像|化验|X\s*线|MRI|CT|超声|心电图|血常规|尿常规|CRP|ESR|尿酸|HCG|电生理|肌电图|血糖|肝功能|肾功能|评估")
_OP_RE = re.compile(r"针刺|进针|出针|针孔|消毒|无菌|按压|晕针|滞针|断针|提插|捻转|得气|深刺|血管|神经|关节腔|电针|艾灸|拔针|起针")
_EDU_RE = re.compile(r"保暖|避风|受凉|休息|负重|活动|运动|锻炼|牵伸|拉伸|饮食|睡眠|复诊|调护|夹板|姿势|患者|自我管理|热敷|康复")
_RISK_RE = re.compile(r"禁忌|禁用|孕妇|妊娠|怀孕|慎用|禁针|不宜|避免|危险|感染|破溃|出血|抗凝|起搏器|过敏|急危|红旗|转诊|急诊|发热|醉酒|空腹|饥饿|疲劳|紧张")


_FALSE_TP_MISSING = re.compile(
    r"(?:"
    r"舌象[、，,]?脉象|舌象、脉象|舌脉|舌象及脉象|舌象和脉象|舌、脉"
    r")\s*(?:均)?(?:未记录|未提供|缺失|不详|资料不足)"
    r"(?:\s*[，,]\s*(?:暂不强行辨证|暂无法明确辨证分型|暂难明确辨证|无法据此辨证|故暂不辨证))?"
    r"\s*[；;。]?"
    r"|未记录舌脉(?:资料)?(?:\s*[，,]\s*(?:暂不强行辨证|暂无法明确辨证分型))?\s*[；;。]?"
)

_GENERIC_RISK = "明显疲劳、空腹或过饱、醉酒、情绪高度紧张时宜先纠正状态后再针刺；局部皮肤明显破损或感染处不直接进针，有晕针史者应提前告知并取安全体位。"
_GENERIC_OP = "操作前核对患者、穴位、侧别及禁忌并常规皮肤消毒；针刺过程中观察面色、出汗、恶心、头晕等晕针先兆。出针后清点针数并按压针孔；发生晕针时立即停止刺激并起针、平卧保暖、监测生命体征，持续不缓解者按急救流程处理；发生滞针时先嘱患者放松，轻柔解除肌肉紧张后退针，不强行拔针。"
_GENERIC_EDU = "针后短暂休息并保持针孔清洁；根据病情注意保暖，避免诱发或加重症状的过度活动或负荷；按疗程复诊，若症状明显加重或出现新的异常体征及时就医。"
_GENERIC_EXAM = "现有资料未提示必须追加的特异性检查；若症状持续、加重或出现新的红旗征象，应根据病情补充相应检查。"


def _clean_line(x: str) -> str:
    x = _NUM_PREFIX.sub("", str(x or "").strip())
    x = _OLD_HEADER.sub("", x).strip()
    x = re.sub(r"^（规则核查）\s*", "", x)
    return x.strip(" \t；;")


def _dedupe(items: list[str]) -> list[str]:
    
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        x = re.sub(r"\s+", " ", str(x or "")).strip()
        if not x:
            continue
        key = re.sub(r"[\s，,。；;：:]", "", x)
        if key and key not in seen:
            seen.add(key)
            out.append(x if x.endswith(("。", "！", "？")) else x + "。")
    return out


def _norm_clause(x: str) -> str:
    return re.sub(r"[\s，,。；;：:！？!?、（）()]", "", x or "")


def _split_clauses(x: str) -> list[str]:
   
    return [p.strip(" \t。；;！!？?") for p in re.split(r"[。；;！!？?]+", str(x or "")) if p.strip()]


def _dedupe_category(items: list[str], near_ratio: float = 0.90) -> list[str]:
    
    out: list[str] = []
    keys: list[str] = []
    for item in items:
        for clause in _split_clauses(item):
            key = _norm_clause(clause)
            if not key:
                continue
            matched = None
            for i, old in enumerate(keys):
                if key == old:
                    matched = i
                    break
                if min(len(key), len(old)) >= 12:
                    ratio = SequenceMatcher(None, key, old).ratio()
                    same_frame = (
                        min(len(key), len(old)) >= 14
                        and key[:6] == old[:6]
                        and key[-4:] == old[-4:]
                        and ratio >= 0.75
                    )
                    if ratio >= near_ratio or same_frame:
                        matched = i
                        break
            if matched is None:
                out.append(clause)
                keys.append(key)
            elif len(key) > len(keys[matched]):
                
                out[matched] = clause
                keys[matched] = key
    return [x + "。" for x in out]


def _remove_false_tp_missing(
    reasoning: str,
    input_text: str,
) -> tuple[str, list[str]]:
    
    reason = str(reasoning or "").strip()
    source = str(input_text or "")

    if not reason:
        return reason, []

    has_tongue = bool(S.TONGUE_RE.search(source))
    has_pulse = bool(S.PULSE_RE.search(source))

 
    if not (has_tongue or has_pulse):
        return reason, []

    blanket_missing = re.compile(
        r"(?:"
        r"舌象[、，,及和与]?脉象"
        r"|舌脉"
        r"|舌、脉"
        r")"
        r"(?:资料|信息|情况)?"
        r"(?:均)?"
        r"(?:未记录|未提供|缺失|不详|资料不足)"
        r"(?:\s*[，,]\s*"
        r"(?:暂不强行辨证|暂无法明确辨证分型|暂不辨证型)"
        r")?"
        r"\s*[；;。]?"
    )

    removed = [
        m.group(0).strip()
        for m in blanket_missing.finditer(reason)
    ]

    if not removed:
        return reason, []

    reason = blanket_missing.sub("", reason)

    reason = re.sub(r"\s+([，,。；;])", r"\1", reason)
    reason = re.sub(r"[；;]\s*[；;]", "；", reason)
    reason = re.sub(r"[；;]\s*。", "。", reason)
    reason = re.sub(r"。\s*。", "。", reason)
    reason = re.sub(r"\s{2,}", " ", reason).strip(" ；;")

    return reason, removed


def _split_top_level_diff_items(body: str) -> list[str]:
    
    out: list[str] = []
    buf: list[str] = []
    depth = 0
    pairs_open = "（([【["
    pairs_close = "）)]】]"
    for ch in str(body or ""):
        if ch in pairs_open:
            depth += 1
        elif ch in pairs_close and depth > 0:
            depth -= 1
        if ch in "；;" and depth == 0:
            item = "".join(buf).strip(" ；;")
            if item:
                out.append(item)
            buf = []
        else:
            buf.append(ch)
    item = "".join(buf).strip(" ；;")
    if item:
        out.append(item)
    return out




def _normalize_diff_item_v19(item: str) -> str:
    
    x = str(item or "").strip()
    m = re.match(r"^(?P<name>[^（(]{2,40})[（(](?P<body>.+)[）)]$", x)
    if not m:
        return x
    body = m.group("body").strip()
    if body.startswith("关键区分点：") or body.startswith("关键区分点:"):
        return f"{m.group('name').strip()}（{body.replace('关键区分点:', '关键区分点：', 1)}）"
    if body.startswith("危险信号："):
        body = "危险信号为" + body[len("危险信号："):]
    return f"{m.group('name').strip()}（关键区分点：{body}）"

def _limit_differentials(reasoning: str, max_dd: int = 3, max_red: int = 1) -> tuple[str, dict]:
   
    reason = str(reasoning or "")
    budget = {"dd": max(0, int(max_dd)), "red": max(0, int(max_red))}
    before = after = 0
    removed: list[str] = []

    def repl(m: re.Match) -> str:
        nonlocal before, after
        label = m.group("label")
        body = m.group("body").strip(" ；;")
        end = m.group("end") or ""
        items = _split_top_level_diff_items(body)
        before += len(items)
        kind = "red" if "排除" in label else "dd"
        keep = [_normalize_diff_item_v19(x) for x in items[: budget[kind]]]
        budget[kind] -= len(keep)
        after += len(keep)
        removed.extend(items[len(keep):])
        if not keep:
            return ""
        return f"{label}：" + "；".join(keep) + (end or "。")

    out = _DIFF_BLOCK_RE.sub(repl, reason)
    out = re.sub(r"。\s*。", "。", out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out, {
        "applied": bool(removed),
        "before_items": before,
        "after_items": after,
        "removed_items": removed,
        "max_dd": max_dd,
        "max_priority_exclusion": max_red,
    }


def _high_risk_point_note(lex, points_text: str) -> str:
    if lex is None or not points_text:
        return ""
    high: list[str] = []
    try:
        for it in lex.extract(points_text):
            if it.get("status") not in ("standard", "alias"):
                continue
            name = it.get("canonical")
            row = lex.rows.get(name, {})
            if row.get("禁深刺") == "是" or row.get("危险等级") == "高":
                if name and name not in high:
                    high.append(name)
    except Exception:
        return ""
    if not high:
        return ""
    return "所选穴位中" + "、".join(high[:6]) + "属于禁深刺或高风险穴位，须严格按【操作】中的安全深度、方向和解剖边界执行。"


def _derived_exams(western_dx: str, input_text: str) -> list[str]:
    inp = S.parse_input(input_text)
    dx = re.sub(r"（待查）|\(待查\)", "", str(western_dx or "")).split("；")[0].strip()
    if not dx:
        return []
    try:
        exams, _ = S.pending_exams(dx, bool(inp.get("_has_aux")), "moderate")
    except Exception:
        return []
    out = []
    for e in exams:
        e = S.sex_consistent(e, input_text)[0].strip()
        if e and e != S.GENERIC_EXAM and e not in out:
            out.append(e)
    return out


def standardize_doctor_view(
    lex,
    reasoning: str,
    notes: str,
    input_text: str,
    points_text: str = "",
    western_dx: str = "",
) -> tuple[str, str, dict]:
    """Return (reasoning, notes, audit)."""
    raw = str(notes or "").strip()
    emergency: list[str] = []
    differentials: list[str] = []
    risk: list[str] = []
    exams: list[str] = []
    op: list[str] = []
    edu: list[str] = []

   
    current: str | None = None
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(S.EMERGENCY_TAG):
            if not S.NULL_WARNING_RE.search(line + "\n"):
                emergency.append(line)
            current = None
            continue
        if _DIFF_RE.search(line):
            differentials.append(_clean_line(line))
            current = None
            continue
        hm = re.match(r"^\s*([1-4])[\.．、)）]\s*([^：:]{1,16})[：:]\s*(.*)$", line)
        if hm and hm.group(2).strip() in {"禁忌", "禁忌与特殊人群", "禁忌与风险警示", "建议完善检查", "操作前后注意事项", "患者教育"}:
            idx = hm.group(1)
            body = hm.group(3).strip()
            current = idx
            if body:
                {"1": risk, "2": exams, "3": op, "4": edu}[idx].append(body)
            continue
        current = None
        line = _clean_line(line)
        if not line:
            continue
        if current in {"1", "2", "3", "4"}:
            {"1": risk, "2": exams, "3": op, "4": edu}[current].append(line)
            continue
        if _EXAM_RE.search(line):
            exams.append(line)
        elif _RISK_RE.search(line):
            risk.append(line)
        elif _OP_RE.search(line):
            op.append(line)
        elif _EDU_RE.search(line):
            edu.append(line)
        else:
            risk.append(line)

    sex, age = _age_sex(input_text)
    pregnant = bool(PREG_CUE.search(input_text))
    if pregnant and not any("妊娠" in x for x in risk):
        risk.append("患者存在妊娠线索，针刺前须再次核对孕周及本方案穴位的妊娠禁忌，避免使用禁针或强刺激穴位。")
    high_note = _high_risk_point_note(lex, points_text)
    if high_note:
        risk.append(high_note)
    risk.append(_GENERIC_RISK)

    if not exams:
        for e in _derived_exams(western_dx, input_text):
            exams.append(e)
    if not exams:
        exams.append(_GENERIC_EXAM)

    op.append(_GENERIC_OP)
    edu.append(_GENERIC_EDU)

    exams = [re.sub(r"^(?:建议|必要时)?完善(?:检查)?[：:]\s*", "", x).strip() for x in exams]
    risk, exams, op, edu = map(_dedupe_category, (risk, exams, op, edu))

   
    reason = str(reasoning or "").strip()
    moved = []
    for d in _dedupe(differentials):
        d0 = d.rstrip("。") + "。"
        if d0 and d0 not in reason:
            moved.append(d0)
    if moved:
        reason = (reason.rstrip() + (" " if reason else "") + "".join(moved)).strip()

    reason, tp_removed = _remove_false_tp_missing(reason, input_text)
    reason, diff_audit = _limit_differentials(reason, max_dd=3, max_red=1)

    note_lines: list[str] = []
    if emergency:
        note_lines.append(_dedupe(emergency)[0])
    note_lines.extend([
        "1. 禁忌与特殊人群：" + "".join(risk),
        "2. 建议完善检查：" + "".join(exams),
        "3. 操作前后注意事项：" + "".join(op),
        "4. 患者教育：" + "".join(edu),
    ])
    audit = {
        "applied": True,
        "differential_moved_to_reasoning": moved,
        "tongue_pulse_false_missing_removed": tp_removed,
        "differential_cap": diff_audit,
        "emergency_kept": bool(emergency),
        "risk_items": len(risk),
        "exam_items": len(exams),
        "operation_items": len(op),
        "education_items": len(edu),
    }
    return reason, "\n".join(note_lines), audit
