from __future__ import annotations

import re

from tcm_study import schema as S
from tcm_study.needling import PREG_CUE, SPEC_LABEL, _age_sex, _depth_given_for, correct_pregnancy_lists, point_spec, pregnancy_rules, standardize_operation_v18, standardize_point_laterality
from tcm_study.doctor_view import standardize_doctor_view
from tcm_study.content_cleanup import final_content_cleanup

_WARN_PRESENT = re.compile(r"急危重症|急诊|转诊|立即(?:就医|就诊|停止|转|送)|生命体征|危急")
LAYER_TAG = "（规则核查）"
FALLBACK_NOTE = "疗程内症状无改善或加重，应重新评估诊断。"


def _norm(s: str) -> str:
    return re.sub(r"[\s，,。；;：:、（）()]", "", s or "")


def ensure_operation_depths(lex, text: str, with_location: bool = True) -> tuple[str, dict]:

    sec = S.parse_bracket_sections(text, top_level_only=True)
    pts_text = (sec.get("取穴") or "").strip()
    op = (sec.get("操作") or "").strip()
    if not pts_text:
        return text, {"applied": False, "reason": "no_points", "added": []}
    names = []
    for it in lex.extract(pts_text):
        if it["status"] in ("standard", "alias"):
            n = it["canonical"]
        elif it["status"] == "whitelist":
            n = it["token"]
        else:
            continue
        if n not in names:
            names.append(n)
    missing = []
    for n in names:
        if not _depth_given_for(op, n):
            sp = point_spec(lex, n, with_location=with_location)
            if sp:
                missing.append(sp)
    if not missing:
        return text, {"applied": False, "reason": "complete", "added": []}
    if SPEC_LABEL in op:
     
        op = op.rstrip()
        if op.endswith("。"):
            op = op[:-1]
        op += "；" + "；".join(missing) + "。"
    else:
        op = op.rstrip() + ("\n" if op else "") + SPEC_LABEL + "；".join(missing) + "。"
    sec["操作"] = op
    out = S.render_sections({k: v for k, v in sec.items() if not k.startswith("_")}, extra=())
    return out, {"applied": True, "reason": "fill_missing_depth", "added": missing}


def apply(lex, text: str, input_text: str, strict_redflags: bool = False, operation_v18: bool = True, doctor_view_v181: bool = True, point_laterality_v19: bool = True) -> tuple[str, dict]:

    sec = S.parse_bracket_sections(text, top_level_only=True)
    notes = (sec.get("注意事项") or "").strip()
    if notes in ("无", "（未给出）"):
        notes = ""
    rep = {"added": [], "removed": []}
    add = []

   
    hits = S.red_flag_warnings(input_text)
    em = [h for h in hits if h["level"] == "E"]
    if strict_redflags:
        old_notes = notes
        notes = re.sub(r"(?m)^\s*【急危重症警示】[^\n]*(?:\n|$)", "", notes).strip()
        if notes != old_notes:
            rep["removed"].append("model_emergency_warning")
        if em:
            line = S.EMERGENCY_TAG + "".join(f"{h['reason']}，{h['action']}。" for h in em)
            add.append(line)
            rep["added"] += [f"red_flag:{h['rule']}" for h in em]
    elif em and not _WARN_PRESENT.search(text):
        line = S.EMERGENCY_TAG + "".join(f"{h['reason']}，{h['action']}。" for h in em)
        add.append(line)
        rep["added"] += [f"red_flag:{h['rule']}" for h in em]
    for h in hits:
        if h["level"] != "E" and _norm(h["action"])[:12] not in _norm(text):
            add.append(f"{LAYER_TAG}{h['reason']}：{h['action']}。")
            rep["added"].append(f"red_flag_note:{h['rule']}")

   
    latrep = {"applied": False, "reason": "disabled"}
    if point_laterality_v19 and (sec.get("取穴") or "").strip():
        old_pts = (sec.get("取穴") or "").strip()
        new_pts, latrep = standardize_point_laterality(lex, old_pts, input_text)
        if new_pts and new_pts != old_pts:
            sec["取穴"] = new_pts
            rep["added"].append("point_laterality_v19")
        rep["point_laterality_v19"] = latrep

    sex, age = _age_sex(input_text)
    pregnant = bool(PREG_CUE.search(input_text))
    if not S.pregnancy_text_applies(input_text):
        notes, rm = S.sex_consistent(notes, input_text)
        rep["removed"] += rm
        notes = notes.strip()
    
    notes, rm = correct_pregnancy_lists(lex, notes)
    rep["removed"] += rm
    pr = pregnancy_rules(lex, sec.get("取穴", ""), input_text)
    if pr["childbearing"] and pr["notes"] and "妊娠" not in notes:
        add += [f"{LAYER_TAG}{n}" for n in pr["notes"]]
        rep["added"].append("pregnancy_note")

    old_notes = notes
    notes = re.sub(r"(?m)^\s*（规则核查）[^\n]*(?:所写深度|超过穴位库上限|禁深刺穴)[^\n]*(?:\n|$)", "", notes).strip()
    if notes != old_notes:
        rep["removed"].append("legacy_depth_rule_note")

    add = [x for x in (S.sex_consistent(a_, input_text)[0].strip() for a_ in add) if x]
    if not notes.strip() and not add:
        notes = FALLBACK_NOTE
        rep["added"].append("fallback_note")
    em_lines = [x for x in add if x.startswith(S.EMERGENCY_TAG)]
    sec["注意事项"] = "\n".join(x for x in em_lines + [notes] + [x for x in add if x not in em_lines] if x)
    
    oprep = {"applied": False, "reason": "disabled"}
    if operation_v18:
        op0 = (sec.get("操作") or "").strip()
        op18, oprep = standardize_operation_v18(lex, sec.get("取穴", ""), op0, input_text)
        if op18 and op18 != op0:
            sec["操作"] = op18
            rep["added"].append("operation_v18")
            rep["operation_v18"] = oprep
        elif oprep.get("depth_corrections"):
            rep["operation_v18"] = oprep

    if doctor_view_v181:
        r0 = (sec.get("诊疗思路") or "").strip()
        n0 = (sec.get("注意事项") or "").strip()
        r1, n1, dvrep = standardize_doctor_view(
            lex, r0, n0, input_text,
            points_text=(sec.get("取穴") or ""),
            western_dx=(sec.get("西医诊断") or ""),
        )
        sec["诊疗思路"], sec["注意事项"] = r1, n1
        if r1 != r0 or n1 != n0:
            rep["doctor_view_v181"] = dvrep
            rep["added"].append("doctor_view_v181")

  
    reason0 = (sec.get("诊疗思路") or "").strip()
    notes0 = (sec.get("注意事项") or "").strip()

    reason1, notes1, content_rep = final_content_cleanup(
        reason0,
        notes0,
        input_text,
    )

    sec["诊疗思路"] = reason1
    sec["注意事项"] = notes1

    if content_rep.get("applied"):
        rep["v19_content_cleanup"] = content_rep
        rep["added"].append("v19_content_cleanup")

    out = S.render_sections({k: v for k, v in sec.items() if not k.startswith("_")}, extra=())
    if not add and not rep["removed"] and "operation_v18" not in rep and "doctor_view_v181" not in rep and "v19_content_cleanup" not in rep and notes == (sec.get("注意事项") or "").strip():
        return text, rep
    return out, rep



SAFETY_CARD_TAG = "【病例安全核查】"
GUIDELINE_SAFETY_CARD_TAG = "【指南安全提示卡】"

def guideline_safety_card(evidence: str, max_chars: int = 1800, include_red: bool = False) -> str:

    lines = (evidence or "").splitlines()
    chunks: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        wanted = (
            include_red and line.startswith("▪ 危险信号"),
            line.startswith("▪ 禁忌与护理"),
            line.startswith("▪ 针灸适应证与安全边界"),
            line.startswith("▪ 指南通用注意"),
        )
        if not any(wanted):
            i += 1
            continue
        buf = [line]
        i += 1
       
        while i < len(lines):
            nxt = lines[i].strip()
            if nxt.startswith("▪ ") or nxt.startswith("【西医证据") or nxt.startswith("【中医证据") or nxt.startswith("【穴位参考"):
                break
            if nxt:
                buf.append(nxt)
            i += 1
        chunks.append("\n".join(buf))
    if not chunks:
        return ""
    head = (
        GUIDELINE_SAFETY_CARD_TAG
        + "（来自本次检索到的病种指南/知识库；用于治疗禁忌与安全边界，不代表患者已经出现急危征象）\n"
        + "- 使用原则：W_RED 已由【鉴别诊断卡】承载；本卡主要用于禁忌、护理与针灸治疗边界，不得自行升级为【急危重症警示】。\n"
    )
    body = "\n".join(chunks)
    return (head + body)[: max(200, int(max_chars))]


def _span(h: dict) -> str:
    t = re.sub(r"【[^】]{1,8}】", " ", h.get("span", ""))
    t = re.sub(r"^[^【]{0,8}】|【[^】]{0,8}$", " ", t)
    return re.sub(r"\s+", " ", t).strip(" ，,。；;：:")


def safety_check_card(lex, input_text: str, points: list[str] | None = None) -> str:
    
    hits = S.red_flag_warnings(input_text)
    em = [h for h in hits if h["level"] == "E"]
    other = [h for h in hits if h["level"] != "E"]
    lines = [SAFETY_CARD_TAG + "（规则依据病例记录自动生成）"]
    if not S.pregnancy_text_applies(input_text):
        hits = [{**h, "action": S.sex_consistent(h["action"], input_text)[0]} for h in hits]
        em = [h for h in hits if h["level"] == "E"]
        other = [h for h in hits if h["level"] != "E"]
    if em:
        for h in em:
            lines.append(f"- 已记录的急危征象：{h['reason']}（病例原文：{_span(h)}）→ {h['action']}")
    else:
        lines.append("- 病例未记录急危重症征象：【急危重症警示】只写\"无\"，不写假设性警示。")
    for h in other:
        lines.append(f"- 需注意：{h['reason']}（病例原文：{_span(h)}）→ {h['action']}")
    sex, age = _age_sex(input_text)
    pregnant = bool(PREG_CUE.search(input_text))
    a_pts, b_pts = [], []
    for p in points or []:
        tag = str(lex.rows.get(p, {}).get("妊娠禁忌") or "")
        (a_pts if tag.startswith("A-") else b_pts if tag.startswith("B-") else []).append(p)
    who = f"{sex or '性别未记录'}{'' if age is None else f'，{age}岁'}"
    # 与训练目标规则（needling.pregnancy_rules）口径一致：育龄女性无妊娠线索时，只有 A 类穴才要求写"针前需排除妊娠"
    if pregnant:
        lines.append(f"- 妊娠：病例有妊娠线索（{who}）。" + (f"所选穴位中{'、'.join(a_pts)}为妊娠绝对禁忌穴，须在注意事项中以【妊娠禁忌】开头写明；" if a_pts else "")
                     + (f"{'、'.join(b_pts)}位于腰骶/下腹部，妊娠期禁针、不宜灸；" if b_pts else "") + "须在注意事项中写明妊娠期注意。")
    elif not S.pregnancy_text_applies(input_text):
        lines.append(f"- 妊娠：患者为{who}，妊娠相关禁忌不适用，注意事项中不写妊娠、育龄女性、排除妊娠、查HCG等内容。")
    elif a_pts and sex == "女" and age is not None and 15 <= age <= 50:
        lines.append(f"- 妊娠：患者为{who}（育龄女性，无妊娠线索）。所选穴位中{'、'.join(a_pts)}为妊娠禁用穴，须写明针前需排除妊娠。只列这些穴，不要增列。")
    else:
        lines.append(f"- 妊娠：患者为{who}（无妊娠线索），" + ("所选穴位无妊娠绝对禁忌穴，" if not a_pts else "") + "不必写妊娠相关内容。")
    return "\n".join(lines)
