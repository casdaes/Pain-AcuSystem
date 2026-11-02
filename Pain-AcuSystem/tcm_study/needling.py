from __future__ import annotations

import re

PREG_CUE = re.compile(r"怀孕|孕\s*\d+\s*周|孕\s*\d+\s*个?月|(?<![无非])妊娠(?!史|试验阴性|禁忌|期禁)|孕妇|早孕|晚孕|先兆流产|孕期|胎位|臀位|宫内孕|停经\s*\d{1,2}\s*周")
_DEPTH_CLAUSE = re.compile(r"[^，。；,;]*?\d+(?:\.\d+)?\s*(?:[~～\-—至]\s*\d+(?:\.\d+)?)?\s*寸")
_NUM_DEPTH = re.compile(r"(\d+(?:\.\d+)?)\s*(?:[~～\-—至]\s*(\d+(?:\.\d+)?))?\s*(寸|mm|毫米|cm|厘米)")
_HAS_DEPTH_NEAR = re.compile(r"\d+(?:\.\d+)?\s*(?:[~～\-—至]\s*\d+(?:\.\d+)?)?\s*(?:寸|mm|毫米|cm|厘米)")
SPEC_LABEL = "逐穴针刺规范："
GENERIC_OP = "毫针刺，得气为度，平补平泻，留针20～30分钟，每日或隔日1次，10次为1疗程。"
GENERIC_OP_V18 = "采用毫针刺法，得气为度，平补平泻。留针20～30分钟，每日或隔日1次，10次为1个疗程。"

_NEEDLE_DEPTH_RE = re.compile(
    r"(?P<prefix>(?:直刺|斜刺|平刺|横刺|沿皮(?:平)?刺|透刺|刺入|进针|针刺深度|进针深度|深度|长针至|针至)"
    r"[^，。；;\n]{0,18}?)(?P<lo>\d+(?:\.\d+)?)\s*"
    r"(?:(?P<sep>[~～\-—至])\s*(?P<hi>\d+(?:\.\d+)?))?\s*(?P<unit>寸|mm|毫米|cm|厘米)"
)

_WHITELIST_SPEC = {
    "夹脊": "根据部位的不同直刺0.3~1寸，不得深刺（胸段深层为胸腔、腰段为腰动脉）",
    "阿是": "据局部解剖酌情取深度，胸背部、腹部不得深刺",
    "八髎": "直刺1~1.5寸（骶后孔内）",
    "内外膝眼": "向膝中斜刺0.5~1寸，或透刺犊鼻",
}


def _cun(v: str) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def depth_clause(row: dict) -> str:
    txt = str(row.get("深度（针灸学教科书）") or "").replace(",", "，")
    lim = _cun(row.get("深度上限_寸"))
    m = _DEPTH_CLAUSE.search(txt)
    if m:
        clause = m.group(0).strip("，。； ")
        if lim is not None:
            dm = re.search(r"(\d+(?:\.\d+)?)\s*(?:[~～\-—至]\s*(\d+(?:\.\d+)?))?\s*寸", clause)
            if dm:
                lo = float(dm.group(1)); hi = float(dm.group(2)) if dm.group(2) else lo
                if hi > lim + 1e-9:
                    if dm.group(2) and lo < lim - 1e-9:
                        rep = f"{lo:g}~{lim:g}寸"
                    elif abs(lo - lim) <= 1e-9:
                        rep = f"{lim:g}寸"
                    else:
                        rep = f"不超过{lim:g}寸"
                    clause = clause[:dm.start()] + rep + clause[dm.end():]
        return clause
    method = str(row.get("主要针法") or "").replace("/", "或")
    if lim is not None:
        return f"{method or '刺'}不超过{lim:g}寸"
    return txt.split("。")[0][:40] if txt else ""


_LOC_CODE = re.compile(r"[（(]\s*[A-Z]{1,3}\s*\d{1,3}\s*[)）]")
_LOC_SPAN = re.compile(r"[（(]定位[：:][^）)]*[）)]")
_LOC_TYPO = {"胭": "腘", "外毗": "外眦", "内毗": "内眦"}

def short_location(row: dict, max_chars: int = 40) -> str:

    t = str(row.get("定位") or "").strip()
    if not t or t in ("nan", "无记载"):
        return ""
    t = re.split(r"注\s*\d*\s*[:：]", t)[0]
    t = _LOC_CODE.sub("", t).replace(",", "，").replace("(", "").replace(")", "").replace("（", "").replace("）", "")
    for a, b in _LOC_TYPO.items():
        t = t.replace(a, b)
    t = t.split("。")[0].strip("，；。 ")
    if len(t) > max_chars:
        cut = t.rfind("，", 0, max_chars)
        t = t[:cut] if cut >= 8 else t[:max_chars]
    return t


def mask_locations(text: str) -> str:
   
    return _LOC_SPAN.sub("", text or "")


def point_spec(lex, name: str, with_location: bool = False) -> str:
  
    canon = name if name in lex.rows else lex.alias.get(name)
    if not canon:
        for k, v in _WHITELIST_SPEC.items():
            if k in name:
                return f"{name}：{v}"
        return ""
    r = lex.rows[canon]
    parts = [depth_clause(r)]
    danger = str(r.get("解剖危险区") or "").replace("|", "/")
    if r.get("禁深刺") == "是":
        parts.append(f"禁深刺（{danger}）" if danger else "禁深刺")
    elif r.get("危险等级") == "高" and danger:
        parts.append(f"不宜深刺（{danger}）")
    elif r.get("危险等级") == "中" and danger:
        parts.append(f"注意{danger}")
    if r.get("限提插捻转") == "是":
        parts.append("不宜大幅提插捻转")
    if r.get("出针按压") == "是":
        parts.append("出针后按压针孔")
    loc = short_location(r) if with_location else ""
    return f"{canon}" + (f"（定位：{loc}）" if loc else "") + "：" + "，".join(p for p in parts if p)


def _names(lex, points_text: str) -> list[tuple[str, str]]:

    out, seen = [], set()
    for it in lex.extract(points_text):
        if it["status"] in ("standard", "alias"):
            key = it["canonical"]
        elif it["status"] == "whitelist":
            key = it["token"]
        else:
            continue
        if key not in seen:
            seen.add(key)
            out.append((key, it["status"]))
    return out


def needling_spec(lex, points_text: str, doctor_op: str = "", with_location: bool = False) -> list[str]:
 
    lines = []
    for name, _ in _names(lex, points_text):
        if doctor_op and _depth_given_for(doctor_op, name):
            continue
        s = point_spec(lex, name, with_location)
        if s:
            lines.append(s)
    return lines


def _depth_given_for(op_text: str, name: str) -> bool:
   
    for m in re.finditer(re.escape(name), op_text or ""):
        if _NEEDLE_DEPTH_RE.search((op_text or "")[m.end(): m.end() + 160]):
            return True
    return False


def compose_operation(doctor_op: str, spec_lines: list[str]) -> str:
    """【操作】= 医师原手法（原样，不改） + 换行 + 逐穴针刺规范。医师未写操作时用保守常规操作。"""
    head = (doctor_op or "").strip() or GENERIC_OP
    if not spec_lines:
        return head
    return head.rstrip() + "\n" + SPEC_LABEL + "；".join(spec_lines) + "。"


def strip_spec(op_text: str) -> str:
    """去掉逐穴针刺规范/逐穴操作块（构造"第一步输出"样本时用）。"""
    t = re.sub(r"\n?" + re.escape(SPEC_LABEL) + r"[^\n]*", "", op_text or "").strip()
  
    m = re.search(r"1[\.．、]\s*手法与疗程[：:]\s*(.*?)(?=\n\s*2[\.．、]\s*逐穴操作|$)", t, re.S)
    if m:
        return m.group(1).strip()
    return t



_POINT_SPLIT = re.compile(r"[、，,；;\n]+")


def _split_point_items(text: str) -> list[str]:
   
    out, buf = [], []
    depth = 0

    for ch in str(text or ""):
        if ch in "（(":
            depth += 1
        elif ch in "）)" and depth > 0:
            depth -= 1

        if ch in "、，,；;\n" and depth == 0:
            item = "".join(buf).strip()
            if item:
                out.append(item)
            buf = []
        else:
            buf.append(ch)

    item = "".join(buf).strip()
    if item:
        out.append(item)

    return out

_SIDE_RE = re.compile(r"(双侧|双|左侧|右侧|患侧|健侧|对侧|左|右)")
_SIDE_CANON = {"左": "左侧", "左侧": "左侧", "右": "右侧", "右侧": "右侧", "双": "双侧", "双侧": "双侧"}
_MIDLINE_MERIDIANS = {"任脉", "督脉"}
_MIDLINE_EXTRA = {"印堂", "腰奇"}  # 经外奇穴中解剖学明确位于正中者；成对奇穴（如腰眼）不得放入此表
_MIDLINE_LOC = re.compile(r"(?:前|后)?正中线(?:上|中|处)?|两眉(?:毛)?内侧端中间|两眉.*连线的中点")
_BILATERAL_CUE = re.compile(r"双侧|两侧|双膝|双肩|双足|双踝|双腕|双手|双髋|双下肢|双上肢|两膝|两肩|两足|两踝|两腕|两手|两髋")
_PAREN_TEXT = re.compile(r"[（(]([^）)]{1,30})[）)]")
_EXTRA_MANIP = re.compile(r"提插|捻转|平补平泻|补法|泻法|强刺激|得气后|行针|活动|运动|温针|电针|艾灸|透刺")
_REGION_RULES = [
    ("足跟", r"足跟|脚跟|跟骨"), ("足底", r"足底|跖筋膜"), ("踝部", r"踝(?:关节)?|外踝|内踝"),
    ("膝部", r"膝(?:关节)?|膝前|膝后"), ("肩部", r"肩(?:关节)?|肩峰|肩周|肩胛"),
    ("肘部", r"肘(?:关节)?"), ("腕部", r"腕(?:关节)?"), ("手部", r"手掌|手背|手指|指间"),
    ("髋臀部", r"髋(?:关节)?|臀部|臀"), ("大腿", r"大腿|股部"), ("小腿", r"小腿|腓肠肌"),
    ("腰部", r"腰(?:部|痛|椎)?|腰骶"), ("颈项部", r"颈(?:部|椎)?|项部|落枕"),
    ("胸胁部", r"胸(?:部|痛)?|胁肋|肋间"), ("背部", r"背部|胸背"), ("腹部", r"腹(?:部|痛)?|胃脘|下腹"),
    ("头面部", r"头部|头痛|额部|颞部|面部|面痛|眶周"),
]


def _selected_point_items(lex, points_text: str) -> list[dict]:
    """保留取穴顺序、侧别/括号限定词；同名左右侧分别保留。"""
    out, seen = [], set()
    for raw in _split_point_items(points_text):
        raw = raw.strip()
        if not raw:
            continue
      
        bare_side = ""
        bm = re.search(r"(?<!侧)(左|右)$", raw)
        classify_raw = raw
        if bm:
            bare_side = bm.group(1)
            classify_raw = raw[:bm.start()].strip()
        it = lex.classify(classify_raw)

        if it.get("status") not in ("standard", "alias", "whitelist"):
            qm = _PAREN_TEXT.search(classify_raw)
            if qm:
                base_candidate = (
                    classify_raw[:qm.start()] +
                    classify_raw[qm.end():]
                ).strip()
                it2 = lex.classify(base_candidate)
                if it2.get("status") in ("standard", "alias", "whitelist"):
                    it = it2
        if it.get("status") in ("empty", "param"):
            continue
        if it.get("status") in ("standard", "alias"):
            canon = it.get("canonical")
        elif it.get("status") == "whitelist":
    
            canon = it.get("token") or it.get("canonical")
        else:
            canon = it.get("token")
        if not canon:
            continue
        pm = _PAREN_TEXT.search(raw)
        qualifier = pm.group(1).strip() if pm else ""
        sm = _SIDE_RE.search(qualifier or raw)
        side = sm.group(1) if sm else bare_side
        side = _SIDE_CANON.get(side, side)
        key = (canon, qualifier or side)
        if key in seen:
            continue
        seen.add(key)
        out.append({"raw": raw, "name": canon, "status": it.get("status"), "category": it.get("category", ""), "side": side, "qualifier": qualifier})
    return out


def _case_region(input_text: str, qualifier: str = "") -> str:
    q = re.sub(r"^(?:左|右|双侧|患侧|健侧|对侧)", "", qualifier or "").strip()
    if q and not re.fullmatch(r"[左右双侧患健对]+", q) and len(q) <= 12:
      
        for label, pat in _REGION_RULES:
            if re.search(pat, q):
                return label
        return q
    text = str(input_text or "")
    chief = re.search(r"【主诉】([^\n]+)", text)
    hay = (chief.group(1) if chief else "") + "\n" + text
    for label, pat in _REGION_RULES:
        if re.search(pat, hay):
            return label
    return "患处"


def _normalize_side(side: str) -> str:
    return _SIDE_CANON.get(str(side or "").strip(), str(side or "").strip())


def _case_side(input_text: str, side: str = "", region: str = "") -> str:
   
    side = _normalize_side(side)
    if side in ("左侧", "右侧", "双侧"):
        return side
    text = str(input_text or "")
    chief = re.search(r"【主诉】([^\n]+)", text)
    chief_text = chief.group(1) if chief else ""
    hay = chief_text + "\n" + text
    region_pat = next((pat for lab, pat in _REGION_RULES if lab == region), re.escape(region)) if region else r".{0,1}"
    
    if region and (re.search(r"(?:双侧|两侧|双|两)(?:[^，。；;\n]{0,8})(?:" + region_pat + r")", hay)
                   or re.search(r"(?:" + region_pat + r")[^，。；;\n]{0,8}(?:双侧|两侧)", hay)):
        return "双侧"
    if _BILATERAL_CUE.search(chief_text):
        return "双侧"
    hits = []
    if region:
        pats = [
            r"(左|右)(?:侧)?[^，。；;\n]{0,12}(?:" + region_pat + r")",
            r"(?:" + region_pat + r")[^，。；;\n]{0,12}(左|右)(?:侧)?",
        ]
        for pat in pats:
            for m in re.finditer(pat, hay):
                hits.append(m.group(1))
    if "左" in hits and "右" in hits:
        return "双侧"
    if hits:
        return "左侧" if hits[0] == "左" else "右侧"
    m = re.search(r"(左|右)(?:侧)?", chief_text)
    return ("左侧" if m and m.group(1) == "左" else "右侧" if m else "")


def _point_is_midline(lex, name: str) -> bool:
    row = lex.rows.get(name, {}) if lex is not None else {}
    meridian = str(row.get("归经") or "").strip()
    loc = str(row.get("定位") or "")
    if meridian in _MIDLINE_MERIDIANS:
        return True
  
    code = str(row.get("WHO code") or "").strip().upper()
    if code.startswith(("CV", "GV")):
        return True
    if _MIDLINE_LOC.search(loc) and not re.search(r"正中线旁开(?:约)?\s*\d", loc):
        return True
    if name in _MIDLINE_EXTRA:
        return True
    return False


def _point_region(lex, name: str) -> str:
    row = lex.rows.get(name, {}) if lex is not None else {}
    loc = str(row.get("定位") or "")
    rules = [
        ("头面部", r"头部|面部|额部|颞部|眉|眼|眶|鼻|口|下颌|耳"),
        ("颈项部", r"颈|项部"), ("肩部", r"肩|肩胛"), ("肘部", r"肘"), ("腕部", r"腕"),
        ("手部", r"手背|手掌|掌骨|指|虎口"), ("胸胁部", r"胸|肋|胁"), ("背部", r"背部|胸椎"),
        ("腰部", r"腰|腰椎"), ("腹部", r"腹|脐|耻骨"), ("髋臀部", r"髋|臀"),
        ("膝部", r"膝|髌"), ("踝部", r"踝"), ("足跟", r"足跟|跟骨"), ("足底", r"足底|跖"),
        ("大腿", r"大腿|股部"), ("小腿", r"小腿|胫骨|腓骨"),
    ]
    for label, pat in rules:
        if re.search(pat, loc):
            return label
    return ""


def _region_family(region: str) -> str:
    return {
        "足跟": "足踝部", "足底": "足踝部", "踝部": "足踝部", "膝部": "膝部", "小腿": "小腿", "大腿": "大腿",
        "肩部": "肩部", "肘部": "肘部", "腕部": "腕部", "手部": "手部", "髋臀部": "髋臀部",
        "腰部": "腰部", "背部": "背部", "颈项部": "颈项部", "头面部": "头面部", "胸胁部": "胸胁部", "腹部": "腹部",
    }.get(region, region)


def _same_limb_chain(case_region: str, point_region: str) -> bool:
    
    upper = {"肩部", "肘部", "腕部", "手部"}
    lower = {"髋臀部", "大腿", "膝部", "小腿", "踝部", "足跟", "足底"}
    return (case_region in upper and point_region in upper) or (case_region in lower and point_region in lower)


def resolve_point_laterality(lex, name: str, explicit_side: str, input_text: str, qualifier: str = "", category: str = "") -> tuple[str, str]:
    
    if _point_is_midline(lex, name):
        return "", "midline"
    explicit = _normalize_side(explicit_side)
    case_region = _case_region(input_text, qualifier)
    case_side = _case_side(input_text, "", case_region)
    if explicit in ("左侧", "右侧", "双侧"):
        return explicit, "explicit"
    if explicit in ("患侧", "健侧", "对侧"):
        if case_side in ("左侧", "右侧"):
            if explicit == "患侧":
                return case_side, "explicit_affected"
            other = "右侧" if case_side == "左侧" else "左侧"
            return other, "explicit_contralateral"
        return "双侧", "unresolved_relative_default_bilateral"
 
    if category == "阿是类" or "阿是" in name:
        return (case_side if case_side else "双侧"), "ashi_case_side" if case_side else "ashi_default_bilateral"
    point_region = _point_region(lex, name)
    local = bool(point_region and _region_family(point_region) == _region_family(case_region))
    if any(k in name for k in ("夹脊", "膝眼", "肩三针")):
        local = True
    if case_side == "双侧":
        return "双侧", "bilateral_case"
    if case_side in ("左侧", "右侧"):
        if local:
            return case_side, "local_affected_side"
        if _same_limb_chain(case_region, point_region):
            return case_side, "same_limb_affected_side"
  
    return "双侧", "remote_or_unspecified_bilateral"


def standardize_point_laterality(lex, points_text: str, input_text: str) -> tuple[str, dict]:
   
    items = _selected_point_items(lex, points_text)
    if not items:
        return str(points_text or "").strip(), {"applied": False, "items": [], "reason": "no_points"}
    out, audit, seen = [], [], set()
    for it in items:
        name = it["name"]
        if it.get("status") == "unknown":
            display = it.get("raw") or name
            side, reason = "", "unknown_preserved"
        else:
            side, reason = resolve_point_laterality(lex, name, it.get("side", ""), input_text, it.get("qualifier", ""), it.get("category", ""))
            qualifier = str(it.get("qualifier") or "").strip()

            if qualifier:
       
                q_parts = [
                    x.strip()
                    for x in re.split(r"[、，,；;]+", qualifier)
                    if x.strip()
                ]

                side_tokens = {
                    "左", "左侧",
                    "右", "右侧",
                    "双", "双侧",
                    "患侧", "健侧", "对侧",
                }

                q_keep = [x for x in q_parts if x not in side_tokens]
                has_standalone_side = any(x in side_tokens for x in q_parts)

                has_embedded_side = (
                    bool(_SIDE_RE.search(qualifier))
                    and not has_standalone_side
                )

                display_name = name

                if (
                    "未指节段" in display_name
                    or any(x and x in display_name for x in q_keep)
                ):
                    display_name = re.sub(
                        r"[（(][^（）()]{1,40}[）)]$",
                        "",
                        display_name,
                    ).strip() or name

                inner_parts = list(q_keep)

                if side and not has_embedded_side:
                    inner_parts.append(side)

                display = (
                    f"{display_name}（{'，'.join(inner_parts)}）"
                    if inner_parts
                    else display_name
                )
            else:
                display = name if not side else f"{name}（{side}）"
        if display in seen:
            continue
        seen.add(display)
        out.append(display)
        audit.append({"point": name, "raw": it.get("raw"), "side": side, "reason": reason, "midline": bool(reason == "midline")})
    new = "、".join(out)
    return new, {"applied": new != str(points_text or "").strip(), "items": audit, "n_points": len(out)}


def standardize_point_laterality_free_text(lex, text: str, input_text: str) -> tuple[str, dict]:
    
    src = str(text or "")
    if not src.strip() or lex is None:
        return src.strip(), {"applied": False, "items": [], "reason": "empty_or_no_lexicon"}

    names = sorted((n for n in lex.rows if n and n in src), key=len, reverse=True)
    # Composite/extra labels not necessarily represented as an exact lexicon row.
    for n in ("阿是穴", "夹脊穴", "华佗夹脊"):
        if n in src and n not in names:
            names.append(n)
    if not names:
        return src.strip(), {"applied": False, "items": [], "reason": "no_points_in_free_text"}

    # Select non-overlapping longest point-name matches first.
    candidates = []
    for name in names:
        for m in re.finditer(re.escape(name), src):
            # “十宣或十二井” is an alternative group prescription, not two independent
            # single-point selections; preserve it verbatim.
            near = src[max(0, m.start()-1): min(len(src), m.end()+2)]
            if name in ("十宣", "十二井") and "或" in near:
                continue
            # Do not rewrite inside the whitelist audit label “夹脊(未指节段)”.
            if name in ("夹脊", "夹脊穴", "华佗夹脊") and src[m.end():].startswith("(未指节段)"):
                continue
            candidates.append((m.start(), m.end(), -len(name), name))
    chosen = []
    for st, en, neglen, name in sorted(candidates, key=lambda x: (x[0], x[2])):
        if any(not (en <= a or st >= b) for a, b, *_ in chosen):
            continue
        chosen.append((st, en, name))
    chosen.sort()

    replacements, audit = [], []
    side_suffix = re.compile(r"^[（(]\s*(双侧|双|左侧|右侧|患侧|健侧|对侧|左|右)\s*[）)]")
    for st, en, name in chosen:
        tail = src[en: en + 16]
        sm = side_suffix.match(tail)
        explicit = sm.group(1) if sm else ""
        old_end = en + (sm.end() if sm else 0)
        if name in ("阿是穴",):
            side, reason = resolve_point_laterality(lex, name, explicit, input_text, category="阿是类")
        elif name in ("夹脊穴", "华佗夹脊"):
            side = _case_side(input_text, explicit, _case_region(input_text)) or "双侧"
            reason = "composite_case_side" if side != "双侧" else "composite_default_bilateral"
        else:
            side, reason = resolve_point_laterality(lex, name, explicit, input_text)
        new = name if not side else f"{name}（{side}）"
        old = src[st:old_end]
        if new != old:
            replacements.append((st, old_end, new))
        audit.append({"point": name, "raw": old, "new": new, "side": side, "reason": reason})

    out = src
    for st, en, new in sorted(replacements, reverse=True):
        out = out[:st] + new + out[en:]

    out = re.sub(r"（(左侧|右侧|双侧)）\s*（\1）", r"（\1）", out)
    return out.strip(), {"applied": out.strip() != src.strip(), "items": audit, "n_points": len(audit), "mode": "free_text"}


def _operation_point_display(
    name: str,
    side: str = "",
    qualifier: str = "",
    raw: str = "",
) -> str:
 
    name = str(name or "").strip()
    side = _normalize_side(side)

    q = str(qualifier or "").strip()


    if not q and raw:
        pm = _PAREN_TEXT.search(str(raw))
        if pm:
            q = pm.group(1).strip()

    q = re.sub(
        r"^(?:左侧|右侧|双侧|患侧|健侧|对侧|左|右|双)\s*",
        "",
        q,
    ).strip()

    side_tokens = {
        "左", "左侧",
        "右", "右侧",
        "双", "双侧",
        "患侧", "健侧", "对侧",
    }

    parts = [
        x.strip()
        for x in re.split(r"[、，,；;]+", q)
        if x.strip()
    ]

    keep = [
        x
        for x in parts
        if x not in side_tokens
    ]

    if side:
        keep.append(side)

    if not keep:
        return name

    return f"{name}（{'，'.join(keep)}）"


_ASHI_MANUAL_SITE_RE = re.compile(
    r"咽后壁|扁桃体|舌根|咽喉|咽部|喉部|口腔|口咽|"
    r"眼球|眼眶内|鼻腔|耳道|气管|食管|"
    r"阴道|宫颈|子宫|直肠|肛管|"
    r"胸腔|腹腔|关节腔|椎管|"
    r"脱垂"
)


def _ashi_safe_region(input_text: str, qualifier: str = "") -> str:
   
    q = str(qualifier or "").strip()

    q = re.sub(
        r"^(?:左侧|右侧|双侧|患侧|健侧|对侧|左|右|双)\s*",
        "",
        q,
    ).strip()

    for label, pat in _REGION_RULES:
        if q and re.search(pat, q):
            return label

    return _case_region(input_text, "")


def ashi_case_spec(
    input_text: str,
    side: str = "",
    qualifier: str = "",
    display: str = "",
) -> str:
    
    q = str(qualifier or "").strip()

    region = _ashi_safe_region(input_text, q)
    side = _case_side(input_text, side, region)

    loc_side = {
        "左侧": "左",
        "右侧": "右",
        "双侧": "双侧",
    }.get(side, side)

    loc = f"{loc_side}{region}" if loc_side else region

  
    label = _operation_point_display(
        "阿是穴",
        side=side,
        qualifier=q,
        raw=display,
    )

    # --------------------------------------------------------
    # 高风险 / 内部 / 黏膜位置：
    # 不自动提供普通阿是穴刺法与深度。
    # --------------------------------------------------------
    if q and _ASHI_MANUAL_SITE_RE.search(q):
        return (
            f"{label}：该限定位置涉及黏膜、腔道、内脏或其他重要解剖结构，"
            "不能套用通用阿是穴针刺深度模板；"
            "仅保留为待现场核对的候选痛点，由执业医师确认是否适宜针刺，"
            "并现场确定具体定位、方向和深度。"
        )

    text = str(input_text or "")

    # --------------------------------------------------------
    # 皮损病例
    # --------------------------------------------------------
    skin = bool(
        re.search(
            r"水疱|疱疹|皮损|破溃|渗出|皮疹",
            text
        )
    )

    if skin:
        return (
            f"{label}：定位在{loc}皮损边缘疼痛最明显处，"
            "先避开水疱、破溃面及感染灶后以指压确认敏感点。"
            "沿皮损边缘平刺或围刺0.3~0.5寸，以局部轻度得气为度；"
            "严格无菌操作，不穿刺破溃皮损。"
        )

    # --------------------------------------------------------
    # 胸胁 / 背部
    # --------------------------------------------------------
    if region in ("胸胁部", "背部"):
        return (
            f"{label}：定位在{loc}疼痛或压痛最明显处，"
            "先以指压复现主要疼痛并确认局部解剖。"
            "沿皮下平刺或斜刺0.3~0.5寸，严禁向胸腔方向深刺，"
            "以局部得气为度。"
        )

    # --------------------------------------------------------
    # 腹部
    # --------------------------------------------------------
    if region == "腹部":
        return (
            f"{label}：定位在{loc}疼痛或压痛最明显处，"
            "先以轻压确认敏感点并避开明显血管。"
            "平刺或浅斜刺0.3~0.5寸，避免深刺进入腹腔，"
            "以局部得气为度。"
        )

    # --------------------------------------------------------
    # 头面 / 颈项
    # --------------------------------------------------------
    if region in ("头面部", "颈项部"):
        return (
            f"{label}：定位在{loc}疼痛或压痛最明显处，"
            "避开眼球、明显血管及重要神经血管结构。"
            "以浅平刺或斜刺0.3~0.5寸为主，"
            "以局部得气为度，不作深刺。"
        )

    # --------------------------------------------------------
    # 足跟 / 足底
    # --------------------------------------------------------
    if region in ("足跟", "足底"):
        return (
            f"{label}：定位在{loc}疼痛或压痛最明显处，"
            "先以指压确认痛点并避开皮肤破损及足底明显血管神经走行区。"
            "根据局部软组织厚度围刺或斜刺0.3~0.5寸，"
            "以局部酸胀为度，避免强行深刺至骨膜。"
        )

    # --------------------------------------------------------
    # 关节附近
    # --------------------------------------------------------
    if region in (
        "肩部",
        "膝部",
        "踝部",
        "肘部",
        "腕部",
    ):
        return (
            f"{label}：定位在{loc}疼痛或压痛最明显处，"
            "先以指压确认痛点并避开关节间隙内的危险方向。"
            "视局部软组织厚度直刺或斜刺0.5~1.0寸，"
            "以局部酸胀为度，避免深刺进入关节腔或刺激骨膜。"
        )

    # --------------------------------------------------------
    # 连病例都不能确认安全体表区域时：
    # 不再机械套用 0.5~1.0 寸。
    # --------------------------------------------------------
    if region == "患处":
        return (
            f"{label}：病例现有信息不足以将该阿是穴确定到"
            "可安全套用通用针刺模板的体表区域；"
            "需由执业医师现场确认具体压痛点、局部解剖、"
            "进针方向和深度后再操作。"
        )

    # --------------------------------------------------------
    # 其他明确体表区域
    # --------------------------------------------------------
    return (
        f"{label}：定位在{loc}疼痛、压痛或触发点最明显处，"
        "先以指压复现主要疼痛并避开明显血管神经。"
        "视局部软组织厚度直刺或斜刺0.5~1.0寸，"
        "以局部酸胀为度；局部组织较薄时取浅刺，不盲目深刺。"
    )


def _point_positions(op_text: str, names: list[str]) -> list[tuple[int, int, str]]:
    pos = []
    t = op_text or ""
    loc_spans = [(m.start(), m.end()) for m in _LOC_SPAN.finditer(t)]
    for name in dict.fromkeys(n for n in names if n):
        for m in re.finditer(re.escape(name), t):
            # 穴位定位文字常会引用另一个穴名（如“足三里：犊鼻下3寸”）；那不是该穴操作的起点。
            if any(a <= m.start() < b for a, b in loc_spans):
                continue
            line_start = max(t.rfind("\n", 0, m.start()), t.rfind("；", 0, m.start()), t.rfind(";", 0, m.start())) + 1
            if "定位" in t[line_start:m.start()]:
                continue
            pos.append((m.start(), m.end(), name))
    return sorted(pos)


def _segment_stop(t: str, start: int, fallback: int) -> int:
    """Point-operation segment ends at the next selected point OR hard text delimiter, whichever comes first."""
    stops = [fallback]
    for sep in ("\n", "；", ";"):
        k = t.find(sep, start)
        if k >= 0:
            stops.append(k)
    return min(stops)


def _depth_values_in_segment(seg: str) -> list[tuple[re.Match, float]]:
    out = []
    for m in _NEEDLE_DEPTH_RE.finditer(seg or ""):
        vals = [float(m.group("lo"))]
        if m.group("hi"):
            vals.append(float(m.group("hi")))
        out.append((m, max(_to_cun(v, m.group("unit")) for v in vals)))
    return out


def _clamp_depth_match(m: re.Match, lim: float) -> tuple[str, bool, str, str]:
    lo = _to_cun(float(m.group("lo")), m.group("unit"))
    hi = _to_cun(float(m.group("hi")), m.group("unit")) if m.group("hi") else lo
    old = m.group(0)
    if hi <= lim + 1e-9:
        return old, False, old, old
    if m.group("hi") and lo < lim - 1e-9:
        depth = f"{lo:g}~{lim:g}寸"
    elif abs(lo - lim) <= 1e-9:
        depth = f"{lim:g}寸"
    else:
        depth = f"不超过{lim:g}寸"
    new = m.group("prefix") + depth
    return new, True, old, new


def correct_operation_depths(lex, op_text: str, points_text: str) -> tuple[str, dict]:

    op = str(op_text or "")
    items = [x for x in _selected_point_items(lex, points_text) if x["status"] in ("standard", "alias")]
    names = [x["name"] for x in items]
    positions = _point_positions(op, names)
    if not positions:
        return op, {"applied": False, "corrections": []}
    corrections = []

    replacements: list[tuple[int, int, str]] = []
    for i, (st, en, name) in enumerate(positions):
        r = lex.rows.get(name, {})
        lim = _cun(r.get("深度上限_寸"))
        if lim is None:
            continue
        next_pos = positions[i + 1][0] if i + 1 < len(positions) else min(len(op), en + 260)
        stop = _segment_stop(op, en, next_pos)
        seg = op[en:stop]
        for m, maxd in _depth_values_in_segment(seg):
            if maxd <= lim + 1e-9:
                continue
            new, changed, old, new_full = _clamp_depth_match(m, lim)
            if changed:
                a, b = en + m.start(), en + m.end()
                replacements.append((a, b, new))
                corrections.append({"point": name, "old": old, "new": new_full, "limit": lim})
    for a, b, new in sorted(replacements, reverse=True):
        op = op[:a] + new + op[b:]
    return op, {"applied": bool(corrections), "corrections": corrections}


def _general_method_and_course(op_text: str, selected_names: list[str]) -> str:
    t = strip_spec(op_text or "")
    
    m = re.search(r"1[\.．、]\s*手法与疗程[：:]\s*(.*?)(?=\n\s*2[\.．、]\s*逐穴操作|$)", t, re.S)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    pieces = re.split(r"[\n；;]", t)
    keep = []
    for x in pieces:
        x = re.sub(r"^\s*(?:[-*•·]|\d+[\.．、)）])\s*", "", x).strip()
        if not x or x.startswith("逐穴"):
            continue
        if any(n and n in x for n in selected_names):
            continue
        if re.search(r"毫针|针刺|手法|平补平泻|补法|泻法|留针|每日|隔日|疗程|每周|每次", x):
            keep.append(x.rstrip("。"))
    if not keep:
        return GENERIC_OP
    out = "；".join(dict.fromkeys(keep))
    out = re.sub(r"^(?:手法|手法与疗程)[：:]", "", out).strip()
    return out[:260] + ("。" if out and not out.endswith("。") else "")


def _model_point_extra(op_text: str, name: str, selected_names: list[str]) -> str:
    """仅保留模型原文中该穴深度之后的特殊手法/运动配合；定位和深度由穴位库重建。"""
    pos = _point_positions(op_text or "", selected_names)
    for i, (st, en, nm) in enumerate(pos):
        if nm != name:
            continue
        next_pos = pos[i + 1][0] if i + 1 < len(pos) else min(len(op_text), en + 260)
        stop = _segment_stop(op_text, en, next_pos)
        seg = op_text[en:stop]
        dms = list(_NEEDLE_DEPTH_RE.finditer(seg))
        rem = seg[dms[-1].end():] if dms else seg
        rem = re.split(r"\n|；;", rem, 1)[0].strip(" ，,。；;：:")
        if rem and _EXTRA_MANIP.search(rem) and len(rem) <= 120:
            return rem.rstrip("。")
    return ""


def _standard_point_line(lex, name: str, side: str, op_text: str, selected_names: list[str], display: str = "") -> str:
    r = lex.rows.get(name)
    display = _operation_point_display(
        name,
        side=side,
        raw=display,
    )
    if not r:
        sp = point_spec(lex, name, with_location=True)
        return (sp if sp else f"{display}：未在标准穴位库中匹配，定位与刺法需由执业医师人工核对。").replace(name + "：", display + "：", 1)
    loc = short_location(r, max_chars=70)
    loc_txt = ("定位" + loc) if loc.startswith("在") else (f"定位于{loc}" if loc else "按标准定位取穴")
    parts = [f"{display}：{loc_txt}。", depth_clause(r).rstrip("，。； ") + "。"]
    danger = str(r.get("解剖危险区") or "").replace("|", "/").strip()
    if r.get("禁深刺") == "是":
        parts.append((f"禁深刺，注意{danger}。" if danger else "禁深刺。"))
    elif r.get("危险等级") == "高" and danger:
        parts.append(f"不宜深刺，注意{danger}。")
    elif r.get("危险等级") == "中" and danger:
        parts.append(f"注意避开{danger}。")
    if r.get("限提插捻转") == "是":
        parts.append("不宜大幅提插捻转。")
    if r.get("出针按压") == "是":
        parts.append("出针后按压针孔。")
    extra = _model_point_extra(op_text, name, selected_names)
    if extra and extra not in "".join(parts):
        parts.append(extra + "。")
    return "".join(parts)


def standardize_operation_v18(lex, points_text: str, op_text: str, input_text: str) -> tuple[str, dict]:
   
    corrected, crep = correct_operation_depths(lex, op_text, points_text)
    items = _selected_point_items(lex, points_text)
    names = [x["name"] for x in items]
    general = _general_method_and_course(corrected, names)
    lines = []
    ashi = []
    for it in items:
        name, side = it["name"], it["side"]
        if it.get("category") == "阿是类" or "阿是" in name:
            line = ashi_case_spec(
                input_text,
                side=side,
                qualifier=it.get("qualifier", ""),
                display=it.get("raw", ""),
            )
            ashi.append(line)
        elif it["status"] in ("standard", "alias"):
            line = _standard_point_line(
                lex,
                name,
                side,
                corrected,
                names,
                display=it.get("raw", ""),
            )
        elif it["status"] == "whitelist":
            sp = point_spec(lex, name, with_location=True)
            display = _operation_point_display(
                name,
                side=side,
                qualifier=it.get("qualifier", ""),
                raw=it.get("raw", ""),
            )
            line = (sp.replace(name + "：", display + "：", 1) if sp else f"{display}：按该非单穴规则的标准定位与安全边界操作。")
        else:
            display = _operation_point_display(
                name,
                side=side,
                qualifier=it.get("qualifier", ""),
                raw=it.get("raw", ""),
            )
            line = f"{display}：未在标准穴位库中匹配，定位与刺法需由执业医师人工核对。"
        lines.append("- " + line.strip())
    if not lines:
        return corrected, {"applied": bool(crep.get("applied")), "depth_corrections": crep.get("corrections", []), "ashi": [], "reason": "no_points"}
    out = "1. 手法与疗程：" + general.strip() + "\n2. 逐穴操作：\n" + "\n".join(lines)
    return out, {"applied": True, "depth_corrections": crep.get("corrections", []), "ashi": ashi, "n_points": len(lines)}


# ------------------------------------------------------------------------------------------------
# 妊娠
# ------------------------------------------------------------------------------------------------
def _age_sex(input_text: str) -> tuple[str, int | None]:
    sex = "女" if re.search(r"性别[:：]?\s*女", input_text) else ("男" if re.search(r"性别[:：]?\s*男", input_text) else "")
    m = re.search(r"年龄[:：]?\s*(\d{1,3})", input_text)
    return sex, (int(m.group(1)) if m else None)


def pregnancy_rules(lex, points_text: str, input_text: str) -> dict:

    sex, age = _age_sex(input_text)
    pregnant = bool(PREG_CUE.search(input_text))
    childbearing = sex == "女" and age is not None and 15 <= age <= 50
    a_pts, b_pts = [], []
    for name, st in _names(lex, points_text):
        if st == "whitelist":
            continue
        tag = str(lex.rows.get(name, {}).get("妊娠禁忌") or "")
        if tag.startswith("A-"):
            a_pts.append(name)
        elif tag.startswith("B-"):
            b_pts.append(name)
    notes, remove = [], []
    if pregnant:
      
        if a_pts:
            notes.append("【妊娠禁忌】患者妊娠，" + "、".join(a_pts) + "属具有通经活血功能、孕妇应禁行针刺的腧穴；如按本方案使用，须经产科评估并取得知情同意，由有经验的医师施行，手法宜轻，不得强刺激，一旦出现腹痛或宫缩立即停针。")
        if b_pts:
            notes.append("妊娠期" + "、".join(b_pts) + "所在的腰骶部/下腹部禁针刺、不宜施灸、禁电针，仅可极浅刺或改用远端取穴。")
        if not a_pts and not b_pts:
            notes.append("患者妊娠，下腹、腰骶部及三阴交、合谷、昆仑、至阴等具有通经活血功能的腧穴应禁行针刺，腹部和腰骶部不宜施灸，手法宜轻。")
    elif childbearing and a_pts:
        notes.append("育龄女性，" + "、".join(a_pts) + "为妊娠禁用穴，针前需排除妊娠。")
    return {"pregnant": pregnant, "childbearing": childbearing, "remove": remove, "notes": notes, "a_points": a_pts, "b_points": b_pts}


def remove_points(points_text: str, names: list[str]) -> str:
    if not names:
        return points_text
    parts = [p for p in re.split(r"[、，,；;]", points_text) if p.strip()]
    keep = [p for p in parts if not any(n in re.sub(r"[（(].*?[)）]", "", p) for n in names)]
    return "、".join(keep)


# ------------------------------------------------------------------------------------------------
# 自动指标：深度超穴位库上限
# ------------------------------------------------------------------------------------------------
def _to_cun(v: float, unit: str) -> float:
    if unit in ("mm", "毫米"):
        return v / 25.0            # 教科书换算 1 寸 ≈ 25 mm（同身寸，近似）
    if unit in ("cm", "厘米"):
        return v / 2.5
    return v


def depth_violations(lex, text: str, tol: float = 1e-9) -> dict:
    
    try:
        from tcm_study.normalize import parse_sections
        sec = parse_sections(text)
        points_text = sec.get("取穴", "")
        op = sec.get("操作", "") or text
        sec_pts = _names(lex, points_text)
    except Exception:
        op, sec_pts = str(text or ""), []
    names = [n for n, st in sec_pts if st != "whitelist"] or [n for n in lex.rows if n in op]
    positions = _point_positions(op, names)
    by_name: dict[str, list[str]] = {n: [] for n in names}
    for i, (st, en, name) in enumerate(positions):
        next_pos = positions[i + 1][0] if i + 1 < len(positions) else min(len(op), en + 260)
        stop = _segment_stop(op, en, next_pos)
        by_name.setdefault(name, []).append(op[en:stop])
    viol, deep_forbidden, with_depth = [], [], 0
    for name in dict.fromkeys(names):
        r = lex.rows.get(name)
        if not r:
            continue
        lim = _cun(r.get("深度上限_寸"))
        vals = []
        for seg in by_name.get(name, []):
            vals.extend(maxd for _, maxd in _depth_values_in_segment(seg))
            if r.get("禁深刺") == "是" and re.search(r"(?<!不)(?<!禁)(?<!避免)深刺", seg) and not re.search(r"不(?:可|宜|得|能)深刺|禁深刺|忌深刺|避免深刺", seg):
                deep_forbidden.append(name)
        if vals:
            with_depth += 1
            maxd = max(vals)
            if lim is not None and maxd > lim + tol:
                viol.append({"point": name, "depth": round(maxd, 2), "limit": lim})
    return {"n_points": len(list(dict.fromkeys(names))), "n_with_depth": with_depth, "violations": viol, "deep_on_forbidden": sorted(set(deep_forbidden))}



_PREG_LIST_SENT = re.compile(r"[^。；;\n【】]*妊娠(?:绝对)?禁(?:用|忌)穴[^。；;\n【】]*[。；;]?")
_PREG_KEY = re.compile(r"妊娠(?:绝对)?禁(?:用|忌)穴")


def correct_pregnancy_lists(lex, text: str) -> tuple[str, list[str]]:
    """在含"妊娠禁用穴/妊娠禁忌穴"的分句里，逐个核对穴名：穴位库标为 A（绝对禁忌）或 B（腰骶/下腹部位禁忌）的保留，
    库里没有妊娠禁忌记载的删掉（v10：A3/A4 曾把内关、中脘、天枢、地机列为妊娠禁用穴）；一个都不剩时删掉整句。
    只核对关键词之前的穴名（"X、Y为妊娠禁用穴"），关键词之后的内容不动。返回 (文本, 删除记录)。"""
    t = str(text or "")
    removed: list[str] = []

    def _fix(m):
        sent = m.group(0)
        k = _PREG_KEY.search(sent)
        head, tail = sent[: k.start()], sent[k.start():]
        
        core = re.sub(r"(?:等)?(?:均为|均属|属于|为|属|是)?\s*$", "", head)
        core = re.sub(r"^\s*(?:\d+[\.．、)）]|[①②③④⑤⑥⑦⑧⑨])\s*", "", core)
        items = [i for i in lex.extract(core) if i["status"] in ("standard", "alias")]
        if not items:
            return sent
        drop = [i for i in items if not str(lex.rows.get(i["canonical"], {}).get("妊娠禁忌") or "").startswith(("A-", "B-"))]
        if not drop:
            return sent
        keep = [i for i in items if i not in drop]
        removed.append("非妊娠禁忌穴：" + "、".join(i["token"] for i in drop))
        if not keep:
            return ""
        for i in drop:
            head = re.sub(re.escape(i["token"]) + r"(?:[（(][^）)]{0,12}[）)])?\s*[、，,和及与]?", "", head, count=1)
        head = re.sub(r"[、，,和及与]\s*(?=(?:等)?(?:为|属于?|是|均为|属)?\s*$)", "", head)
        return head + tail

    t2 = _PREG_LIST_SENT.sub(_fix, t)
    if not removed:
        return t, []
    t2 = re.sub(r"\n\s*(?:\d+[\.．、)）]|[①②③④⑤⑥⑦⑧⑨])\s*(?=\n|$)", "", t2)
    t2 = re.sub(r"\n\s*\n+", "\n", t2).strip()
    from tcm_study.schema import _renumber
    return _renumber(t2), removed
