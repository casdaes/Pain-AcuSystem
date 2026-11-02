from __future__ import annotations


import re

# ------------------------------------------------------------------------------------------------
# 1. 固定栏目与系统提示词
# ------------------------------------------------------------------------------------------------
SECTIONS_V3 = ["诊疗思路", "西医诊断", "中医诊断", "证型", "病机", "治法", "取穴", "操作", "注意事项"]
UNCERTAINTY_NOTE = "舌象、脉象未记录，暂无法明确辨证分型；建议补充舌脉，四诊合参后再定证型。"
EMERGENCY_TAG = "【急危重症警示】"
PENDING = "（待查）"

SYSTEM_PROMPT_V3 = (
    '你是针灸科临床辅助诊疗系统，输出供执业医师审核，不是最终医嘱。请严格按下列九个栏目输出，栏目名用【】标出，'
    '顺序固定、一个都不能省略；每个栏目独占一行或一段。\n'
    '【诊疗思路】写成一个连贯、临床化的诊断推理段落，建议5–7句，按以下顺序组织：'
    '①先写主诉及病程；病例原文有明确发作/加重年月日或时长时原样引用，没有则只写已有病程，不得补造具体日期；'
    '②概括现病史中的主要症状及伴随症状；③引用查体中有诊断意义的阳性体征和阴性体征、已有辅助检查，以及病例实际记录的舌象、苔象、脉象；舌脉未记录时必须明确说明未记录，不得臆造。若某一已记录体征与当前拟诊的典型表现明显一致或不一致，应紧随该体征用全角括号补充临床意义，例如“局部无压痛（此体征与典型跖筋膜炎的固定压痛点不符，需结合重复查体及影像学进一步鉴别）”；括号内容必须由病例事实或检索证据支持，不得臆造病理机制；'
    '④综合判断西医诊断与中医病位/相关经络、病机；病位与经络只能依据症状部位、经络循行或病例原记录，不得强行指定脏腑；'
    '⑤明确写出支持诊断的病例依据，并说明证据不足之处；⑥写具体鉴别诊断及关键区分点/所需检查。'
    '鉴别诊断固定优先使用“需鉴别：疾病A（关键区分点：……）；疾病B（关键区分点：……）”；若为严重疾病候选则写“需优先排除：疾病C（关键区分点：……）”。关键区分点应具体到症状、体征、影像/化验或专科检查，不得只写“结合临床鉴别”。只有病例已有阴性检查或明确反证时才能写“可排除/基本排除”，'
    '否则必须写“尚需排除/需鉴别”，不得把未证实疾病写成已确诊。全段只引用病例和检索证据中已有信息，不简单复述病例，不写套话。\n'
    '【西医诊断】只写病名，可多条并用“；”分隔；依据不足（如缺少必要的影像或化验）时在病名后标注“（待查）”。本栏不写证型、症状描述或解释。\n'
    '【中医诊断】只写中医病名，不写证型，不加括号说明。\n'
    '【证型】只写证型名。舌象、脉象未记录且无法据现有信息辨证时，写“' + UNCERTAINTY_NOTE + '”；不得臆测未提供的舌脉。\n'
    '【病机】简要病机分析。\n'
    '【治法】治疗原则。\n'
    '【取穴】主穴在前、配穴在后，用“、”分隔。所有可分左右的单穴必须明确侧别，统一写“穴名（左侧）”“穴名（右侧）”或“穴名（双侧）”；任脉、督脉及其他明确位于人体正中线的中线穴不加左右/双侧。病例为单侧肢体病变时，病变局部穴及同一患肢上的远端经穴优先取患侧；跨肢体/全身调节穴若病例或原医嘱未明确单侧，默认双侧；双侧症状取双侧；阿是穴按病例实际病侧；对侧取穴只有病例或原医嘱明确时才写，不自行推断。不得把关元、中极、腰阳关等中线穴写成“双侧”。\n'
    '【操作】固定写成两部分：第1部分“1. 手法与疗程：”概括毫针/补泻、留针时间、频次与疗程；第2部分“2. 逐穴操作：”每个穴位单独一行，以“- 穴位（侧别）：定位…。刺法与深度…。必要安全提示…。”书写。'
    '标准穴位的定位、刺法与深度必须服从穴位安全库，任何深度不得超过安全上限；禁深刺或邻近解剖危险区者必须写明。'
    '阿是穴必须结合本病例实际疼痛/压痛部位与侧别给出定位和操作，不得仅写“无固定位置”。不得另起“逐穴针刺规范”重复一遍。\n'
    '【注意事项】是给执行针灸的医生看的，不再重复诊疗思路中的鉴别诊断。若病例当前存在已记录的红旗征象，第一行必须以“' + EMERGENCY_TAG +
    '”写明具体征象及急诊/转诊处理；若无当前红旗，中间生成时可写“' + EMERGENCY_TAG + '无。”，最终展示层会删除空警示。'
    '其余内容固定写成四项：'
    '1. 禁忌与特殊人群：写与本患者有关的妊娠、抗凝/出血风险、感染或皮肤破损、高风险穴位/禁深刺等；并提示明显疲劳、空腹或过饱、醉酒、情绪高度紧张时宜先纠正状态后再针刺。'
    '2. 建议完善检查：诊断标注“（待查）”者必须列出具体检查；需要进一步鉴别时写相应影像、化验或专科评估，不得泛写“完善相关检查”。'
    '3. 操作前后注意事项：包括针前核对与消毒、针中观察晕针先兆、出针后清点针数与按压针孔；如发生晕针应立即停止刺激并起针、平卧保暖并监测生命体征，'
    '滞针时先放松局部并轻柔处理，禁止强行拔针。'
    '4. 患者教育：结合病情写针后休息、局部清洁、保暖、避免诱发或加重症状的活动/负荷及按时复诊等。'
    '安全信息仍分三层：病例当前红旗 > 穴位/手法硬安全约束 > 病种指南背景；指南危险信号只有病例事实同时满足时才可升级为当前急危。'
    '穴位安全卡和手法卡中的深度上限、禁深刺、解剖危险区、妊娠禁忌均属硬约束；妊娠内容只写与该患者性别、年龄及所选穴位相符的内容。\n'
    '不得编造病例中没有的信息；检索证据与病例事实冲突时以病例事实为准。'
)


# ------------------------------------------------------------------------------------------------
# 2. 解析 / 渲染 【栏目】 文本
# ------------------------------------------------------------------------------------------------
_BRACKET = re.compile(r"[【〔\[]([^】〕\]\n]{1,14})[】〕\]]")
_ALIASES = {"治则": "治法", "选穴": "取穴", "穴位": "取穴", "处方": "取穴", "手法": "操作", "针刺操作": "操作", "方解": "方义",
            "辨证": "证型", "辨证分型": "证型", "中医证型": "证型", "中医病机": "病机", "中医治法": "治法", "风险提示": "注意事项",
            "安全提示": "注意事项", "调护": "调护", "诊疗分析": "诊疗思路", "辨证分析": "诊疗思路", "分析": "诊疗思路"}


TOP_LEVEL = frozenset(SECTIONS_V3) | frozenset(_ALIASES) | frozenset(_ALIASES.values()) | {"方义", "调护"}
DROP_SECTIONS = frozenset({"穴位安全卡", "手法卡", "第一步输出", "检索证据", "穴位参考", "病例安全核查", "鉴别诊断卡", "指南安全提示卡"})
_TOP_MARK = re.compile(r"^[ \t#>*]*[【〔\[]([^】〕\]\n]{1,14})[】〕\]]", re.M)


# v14.3：训练目标每例都以"【急危重症警示】…"开头，没有急危征象时写"【急危重症警示】无。"——"写不写"变成标签后面接"无"还是接内容，
# A4 可以直接照抄核查卡；该行是格式占位，生成后（所有臂同一规则）删除，不进入评分稿。
NULL_WARNING_RE = re.compile(r"[ \t]*【急危重症警示】[ \t]*(?:无|暂无|未见|未记录(?:急危重症?征象)?)[。.；;，,]?[ \t]*(?:\n|$)")


_NULL_ITEM_RE = re.compile(
    r"(?:(?<=\n)|(?<=^)|(?<=[。；;]))[ \t]*(?:[①②③④⑤⑥⑦⑧⑨]|\d+[\.．、)）])?[ \t]*"
    r"(?:【?急危重症警示】?[ \t]*[:：]?[ \t]*(?:无|暂无)(?:急危重症征象|明显急危征象)?|无急危重症(?:警示|征象)|未见急危重症征象|病例(?:中)?未记录急危重症征象)"
    r"[^。；;\n]{0,80}?(?:[。；;]|(?=\n)|$)[ \t]*\n?"
    r"|(?:(?<=\n)|(?<=^)|(?<=[。；;]))[ \t]*(?:[①②③④⑤⑥⑦⑧⑨]|\d+[\.．、)）])[ \t]*无[ \t]*(?:[。；;]|(?=\n)|$)[ \t]*\n?")


def strip_null_warning(text: str) -> tuple[str, bool]:
    t = str(text or "")
    t2 = NULL_WARNING_RE.sub("", t)
    m = re.search(r"【注意事项】", t2)
    if m:
        end = re.search(r"\n【(?!急危重症警示|妊娠禁忌)[^】\n]{1,10}】", t2[m.end():])
        stop = m.end() + end.start() if end else len(t2)
        body = _NULL_ITEM_RE.sub("", t2[m.end():stop]).rstrip()
        body = re.sub(r"\n{2,}", "\n", body)
        t2 = t2[:m.end()] + body + ("\n" if stop < len(t2) and not t2[stop:].startswith("\n") else "") + t2[stop:]
    return (t2, True) if t2 != t else (t, False)


def has_real_warning(text: str) -> bool:
    return EMERGENCY_TAG in strip_null_warning(text)[0]


def relocate_emergency(out: dict[str, str]) -> dict[str, str]:
    """把出现在其他栏目里的"【急危重症警示】…"段（标签起至该栏目末）移到注意事项首行。"""
    for k in list(out):
        if k in ("注意事项", "_lead"):
            continue
        idx = (out[k] or "").find(EMERGENCY_TAG)
        if idx >= 0:
            seg, out[k] = out[k][idx:].strip(), out[k][:idx].rstrip()
            notes = (out.get("注意事项") or "").strip()
            out["注意事项"] = seg + ("\n" + notes if notes and notes != "无" else "")
    return out


def parse_bracket_sections(text: str, top_level_only: bool = False) -> dict[str, str]:
    """把 '【西医诊断】…\\n【中医诊断】…' 解析成 dict。

    top_level_only=False（默认，旧行为）：任何【…】都开启新栏目，未知栏目名原样保留。
    top_level_only=True（v12）：见上方注释——只认行首顶层栏目；复述的输入块丢弃；急危重症警示归位到注意事项。"""
    t = str(text or "").replace("\r", "")
    out: dict[str, str] = {}
    if top_level_only:
        marks = []
        for m in _TOP_MARK.finditer(t):
            raw = m.group(1).strip()
            if raw in DROP_SECTIONS:
                marks.append((m, None))
            elif raw in TOP_LEVEL:
                marks.append((m, _ALIASES.get(raw, raw)))
    else:
        marks = [(m, _ALIASES.get(m.group(1).strip(), m.group(1).strip())) for m in _BRACKET.finditer(t)]
    for i, (m, name) in enumerate(marks):
        body = t[m.end(): marks[i + 1][0].start() if i + 1 < len(marks) else len(t)].strip()
        if name is None:
            continue                                     # 复述的输入块，丢弃
        out[name] = (out[name] + "\n" + body) if name in out and body else out.get(name, body)
    lead = t[: marks[0][0].start()].strip() if marks else t.strip()
    if lead:
        out["_lead"] = lead
    return relocate_emergency(out) if top_level_only else out



_LIB_CODE_RE = re.compile(r"(?<![A-Za-z0-9])(?:GL\d{3}|[DGB]\d{4}|BK\d)(?:\s*/\s*(?:GL\d{3}|[DGB]\d{4}|BK\d))*(?![A-Za-z0-9型号])")


def strip_library_codes(text: str) -> str:
    t = _LIB_CODE_RE.sub("", str(text or ""))
    t = re.sub(r"[（(]\s*(?:见|参见|详见)?\s*[：:/、，,·]*\s*[)）]", "", t)
    t = re.sub(r"([（(《「])\s+", r"\1", t)
    t = re.sub(r"\s+([)）》」])", r"\1", t)
    t = re.sub(r"《\s*》|[\[［]\s*[\]］]", "", t)
    return re.sub(r"[ \t]{2,}", " ", t)

_HCG_ITEM = re.compile(r"[，,、；;]?\s*(?:育龄(?:期)?女性|育龄期?妇女)?\s*(?:查|行)?\s*(?:尿|血)\s*HCG(?:检查|检测)?", re.I)

_CHILDBEARING_CLAUSE = re.compile(r"[^。；;\n【】]*(?:育龄(?:期)?女性|育龄期?妇女|针前需排除妊娠|排除妊娠后)[^。；;\n【】]*[。；;]?")

_PREG_NOTE_CLAUSE = re.compile(r"[^。；;\n【】]*(?:妊娠相关(?:禁忌)?|妊娠(?:期|绝对)?禁(?:用|忌|针)|孕妇|孕期|妊娠期|妊妇|落胎|堕胎|诱发宫缩|宫缩)[^。；;\n【】]*[。；;]?")


def patient_sex_age(input_text: str) -> tuple[str, int | None]:
    sex = "女" if re.search(r"性别[:：]?\s*女", input_text or "") else ("男" if re.search(r"性别[:：]?\s*男", input_text or "") else "")
    m = re.search(r"年龄[:：]?\s*(\d{1,3})", input_text or "")
    return sex, (int(m.group(1)) if m else None)


def pregnancy_text_applies(input_text: str) -> bool:

    from tcm_study.needling import PREG_CUE
    if PREG_CUE.search(input_text or ""):
        return True
    sex, age = patient_sex_age(input_text)
    if sex == "男":
        return False
    if sex == "女" and age is not None and not (15 <= age <= 50):
        return False
    return True


def sex_consistent(text: str, input_text: str) -> tuple[str, list[str]]:

    t = str(text or "")
    if pregnancy_text_applies(input_text):
        return t, []
    removed = [m.group(0).strip(" ，,、；;") for m in _HCG_ITEM.finditer(t) if m.group(0).strip(" ，,、；;")]
    t = _HCG_ITEM.sub("", t)
    removed += [m.group(0).strip() for m in _CHILDBEARING_CLAUSE.finditer(t) if m.group(0).strip()]
    t = _CHILDBEARING_CLAUSE.sub("", t)
    m = re.search(r"【注意事项】", t)
    if m:
        end = re.search(r"\n【(?!急危重症警示|妊娠禁忌)[^】\n]{1,10}】", t[m.end():])
        stop = m.end() + end.start() if end else len(t)
        body = t[m.end():stop]
        removed += [x.group(0).strip() for x in _PREG_NOTE_CLAUSE.finditer(body) if x.group(0).strip()]
        t = t[:m.end()] + _PREG_NOTE_CLAUSE.sub("", body) + t[stop:]
    t = re.sub(r"[，,、]\s*([；;。])", r"\1", t)
    t = re.sub(r"\n\s*(?:\d+[\.．、)）]|[①②③④⑤⑥⑦⑧⑨])\s*(?=\n|$)", "", t)
    if not removed:
        return str(text or ""), []
    t = re.sub(r"\n\s*\n+", "\n", t).strip()
    return _renumber(t), removed


def _renumber(t: str) -> str:
    out, k = [], 0
    for line in t.split("\n"):
        m = re.match(r"^(\s*(?:[【〔][^】〕\n]{1,14}[】〕])?\s*)(\d+)([\.．、])(\s*)", line)
        if m:
            k = 1 if re.match(r"^\s*[【〔]", line) else k + 1
            line = f"{m.group(1)}{k}{m.group(3)}{m.group(4)}" + line[m.end():]
        else:
            k = 0 if not re.match(r"^\s*$", line) else k
        out.append(line)
    return "\n".join(out)


def render_sections(sec: dict[str, str], order: list[str] = SECTIONS_V3, extra: tuple[str, ...] = ("方义", "调护")) -> str:
    lines = [f"【{k}】{strip_library_codes((sec.get(k) or '').strip()) or '无'}" for k in order]
    for k in extra:
        if (sec.get(k) or "").strip():
            lines.append(f"【{k}】{strip_library_codes(sec[k].strip())}")
    return "\n".join(lines)



_FIELD = re.compile(r"【(基本信息|主诉|现病史|查体|辅助检查|舌脉)】(.*?)(?=\n【|\Z)", re.S)
TONGUE_RE = re.compile(r"舌[质体苔]?[淡红暗紫绛胖瘦边尖]|舌苔|苔[白黄薄厚腻燥少剥润滑灰黑]")
PULSE_RE = re.compile(r"脉[弦细滑沉数缓紧濡涩浮迟虚实结代洪微弱]")


def parse_input(user_text: str) -> dict[str, str]:
    d = {k: v.strip() for k, v in _FIELD.findall(str(user_text or ""))}
    d["_has_tp"] = bool(d.get("舌脉") and d["舌脉"] != "未记录") or bool(TONGUE_RE.search(user_text) or PULSE_RE.search(user_text))
    aux = d.get("辅助检查", "")
    d["_has_aux"] = bool(aux) and aux not in ("未提供", "未记录", "无")
    return d



_PAREN = re.compile(r"[（(]([^（）()]*)[）)]")
_PATTERN_WORD = re.compile(r"(?:证|型|夹杂|不足|亏虚|亏损|阻络|痹阻|瘀阻|内蕴|上扰|下注|失养|不和|不通|亢盛|气滞|血瘀|痰湿|湿热|寒湿|风寒|风热|阴虚|阳虚|气虚|血虚|肝郁|肾虚|脾虚)$")
_TCM_HINT = re.compile(r"^(?:中医(?:病名|诊断)?|对应中医|中医称)[:：]?\s*(.+)$")
_WM_HINT = re.compile(r"^(?:西医(?:病名|诊断)?|对应西医|西医对应)[:：]?\s*(.+)$")
_STRIP_HEAD = re.compile(r"^(?:\d+[\.．、)]\s*|[-•·]\s*|初步诊断为|诊断为|考虑为|考虑|诊断[:：])")


def _split_multi(s: str) -> list[str]:
    parts = re.split(r"[；;\n]|(?<=[^\d])\d+[．.、)]\s*", s)
    return [p.strip(" ，,。") for p in parts if p and p.strip(" ，,。")]


def clean_dx(raw: str, kind: str) -> tuple[str, dict[str, str]]:
    """返回 (清洗后的诊断, 被挪走的内容{'证型':..., '中医诊断':..., '西医诊断':...})。
    kind = '西医' | '中医'。规则：括号内若是证型 → 挪到证型；若是另一体系的病名 → 挪到对应诊断栏；其他括号说明删除，
    但保留"（待查）"、分型/分期/侧别（如 Ⅱ型、左侧、神经根型）。"""
    moved: dict[str, str] = {}
    s = str(raw or "").strip()
    if not s:
        return "", moved
    keep_paren = re.compile(r"^(?:待查|[ⅠⅡⅢⅣIV]+型?|[左右双]侧?|急性期?|慢性期?|恢复期|后遗症期|早期|晚期|[A-Za-z]{1,8}|\d+[级期度]|神经根型|椎动脉型|脊髓型|交感型|混合型)$")

    def repl(m):
        inner = m.group(1).strip()
        if keep_paren.match(inner):
            return f"（{inner}）"
        mt, mw = _TCM_HINT.match(inner), _WM_HINT.match(inner)
        if mt and kind == "西医":
            moved["中医诊断"] = mt.group(1).strip(); return ""
        if mw and kind == "中医":
            moved["西医诊断"] = mw.group(1).strip(); return ""
        if _PATTERN_WORD.search(inner) or inner.endswith(("证", "型")):
            moved["证型"] = inner.rstrip("型") if inner.endswith("型") and not inner.endswith("证型") else inner
            return ""
        return ""
    s = _PAREN.sub(repl, s)
    parts = []
    for p in _split_multi(s):
        p = _STRIP_HEAD.sub("", p).strip(" ，,。")
        if not p:
            continue
    
        if kind == "中医":
            m = re.match(r"^(气滞血瘀|瘀血|寒湿|湿热|肾虚|风寒|风热|血瘀|气血亏虚|肝肾亏虚|肝肾不足|痰湿|阴虚|阳虚)(腰痛|头痛|痹证|痹症|眩晕|胃痛|胁痛|痛经|不寐)$", p)
            if m:
                moved.setdefault("证型", m.group(1)); p = m.group(2)
        parts.append(p)
    return "；".join(dict.fromkeys(parts)), moved



EXAM_REQUIRED: list[tuple[str, list[str], bool]] = [
    (r"颈椎病|颈椎", ["颈椎X线正侧位（必要时过伸过屈位）", "有上肢放射痛、麻木或肌力下降者行颈椎MRI"], True),
    (r"颈痛|落枕|颈肩|项痹|项强|颈部", ["颈椎X线正侧位（排除骨性病变）", "有上肢放射痛、麻木或肌力下降者行颈椎MRI"], False),
    (r"^腹痛|下腹痛|腹痛", ["腹部及盆腔超声", "血常规、尿常规，育龄女性查尿HCG"], False),
    (r"腰椎间盘|椎间盘|腰突|腰椎管|坐骨神经", ["腰椎MRI（或CT）", "神经系统查体（直腿抬高试验、下肢肌力/感觉/反射）"], True),
    (r"腰肌劳损|腰痛|腰痹|腰扭伤|腰背痛|腰部", ["腰椎X线正侧位（排除骨性病变）", "疼痛持续>6周、夜间痛或伴下肢症状时行腰椎MRI"], False),
    (r"肩袖|肩关节撞击|肩峰下", ["肩关节MRI或超声", "肩关节X线"], True),
    (r"肩周炎|肩痹|肩凝|冻结肩|肩关节周围炎|肩痛", ["肩关节X线", "肩关节MRI或超声（鉴别肩袖损伤与粘连性关节囊炎）"], False),
    (r"骨关节炎|骨性关节炎|滑膜炎|关节炎|痛风|类风湿", ["受累关节X线", "血常规、ESR、CRP、尿酸、RF（排除炎性/痛风性/感染性关节炎）"], True),
    (r"膝痹|膝关节痛|关节痛|痹证|痹症|痛痹|着痹|行痹", ["受累关节X线", "血常规、ESR、CRP、尿酸（排除炎性/痛风性关节炎）"], False),
    (r"踝|扭伤|筋伤|伤筋|软组织损伤", ["受伤部位X线（排除骨折、撕脱骨折）", "必要时超声/MRI评估韧带损伤"], False),
    (r"腕管|尺神经|神经卡压|肘管|胸廓出口", ["神经电生理（肌电图、神经传导速度）", "必要时超声"], True),
    (r"网球肘|肱骨外上髁|腱鞘炎|狭窄性|肌腱炎", ["局部超声（必要时）"], False),
    (r"三叉神经痛|面痛", ["头颅MRI（排除继发性三叉神经痛：肿瘤、血管压迫、多发性硬化）"], False),
    (r"面瘫|面神经|口眼歪斜|口僻|贝尔", ["神经系统查体区分周围性/中枢性面瘫", "伴耳后疱疹、听力改变或其他脑神经受累时行头颅影像"], False),
    (r"头痛|偏头痛|紧张性头痛|头风", ["血压", "首发剧烈头痛、进行性加重、伴神经系统体征或年龄>50岁者行头颅CT/MRI"], False),
    (r"眩晕|头晕", ["血压、血糖", "前庭功能及神经系统查体，必要时头颅MRI"], False),
    (r"带状疱疹|蛇串疮|疱疹后", ["血糖", "皮损处病毒学检查（必要时）"], False),
    (r"胃炎|胃溃疡|十二指肠|消化性溃疡", ["胃镜及幽门螺杆菌检测", "心电图（排除心源性上腹痛）"], True),
    (r"胃脘痛|胃痛|胃痉挛|上腹痛|胃肠功能", ["胃镜及幽门螺杆菌检测", "心电图（排除心源性上腹痛）", "腹部超声、血尿淀粉酶（排除胆胰疾病）"], False),
    (r"胆囊炎|胆石|胆道", ["腹部超声", "血常规、肝功能、淀粉酶"], True),
    (r"胁痛|胁肋", ["腹部超声", "血常规、肝功能"], False),
    (r"盆腔炎|子宫内膜异位|附件", ["盆腔超声", "育龄女性查尿HCG", "妇科检查"], True),
    (r"痛经|月经", ["盆腔超声（排除子宫内膜异位症、盆腔炎、附件包块）", "育龄女性查尿HCG"], False),
    (r"肾结石|肾绞痛|输尿管|泌尿系结石", ["泌尿系超声或CT", "尿常规"], True),
    (r"心绞痛|冠心病|心肌", ["心电图、心肌酶/肌钙蛋白", "心脏超声或冠脉评估"], True),
    (r"胸痹|胸痛", ["心电图、心肌酶/肌钙蛋白", "必要时心脏超声或冠脉评估"], False),
    (r"脑梗|脑出血|卒中|脑血栓|脑卒中", ["头颅CT/MRI", "血压、血糖、血脂、凝血功能"], True),
    (r"中风|中经络|中脏腑", ["头颅CT/MRI", "血压、血糖、血脂、凝血功能"], True),
    (r"周围神经病|神经损伤|神经炎", ["神经电生理检查", "血糖（排除糖尿病性周围神经病）"], False),
    (r"神经痛", ["血糖", "必要时神经电生理检查"], False),
    (r"强直性脊柱炎|骶髂关节炎", ["骶髂关节X线/MRI", "HLA-B27、ESR、CRP"], True),
    (r"骨折", ["X线，必要时CT"], True),
    (r"肿瘤|癌|转移", ["影像学及肿瘤专科评估"], True),
    (r"骨质疏松", ["骨密度（DXA）", "脊柱X线"], True),
    (r"糖尿病|消渴", ["空腹血糖、糖化血红蛋白", "尿常规、肾功能"], True),
    (r"高血压", ["多次血压测量/动态血压", "血脂、血糖、肾功能、心电图"], True),
    (r"甲状腺", ["甲状腺功能、甲状腺超声"], True),
    (r"癫痫|痫", ["脑电图", "头颅MRI"], True),
    (r"帕金森|震颤", ["神经科评估", "头颅MRI"], False),
    (r"贫血", ["血常规、铁代谢"], True),
]
_GENERIC_STRUCTURAL = re.compile(r"痛|痹|伤|炎|病|综合征")
GENERIC_EXAM = "与诊断相应的影像学或实验室检查"


def pending_exams(wm_dx: str, has_aux: bool, policy: str = "moderate") -> tuple[list[str], bool]:

    if has_aux or not wm_dx:
        return [], False
    exams: list[str] = []
    hard = False
    for pat, ex, h in EXAM_REQUIRED:
        if re.search(pat, wm_dx):
            exams += ex
            hard = hard or h
    if not exams and _GENERIC_STRUCTURAL.search(wm_dx):
        exams, hard = [GENERIC_EXAM], False
    pend = bool(exams) and (policy == "strict" or hard)
    return list(dict.fromkeys(exams)), pend


def mark_pending(wm_dx: str) -> str:
    if not wm_dx or PENDING in wm_dx or "待查" in wm_dx:
        return wm_dx
    return "；".join(p + PENDING for p in wm_dx.split("；") if p)

TCM_TO_SYMPTOM_LABEL: list[tuple[str, str]] = [
    (r"中风|中经络|中脏腑", "脑卒中"), (r"面瘫|口僻|口眼歪斜|口喎", "周围性面瘫"), (r"面痛", "面部疼痛"), (r"头痛|头风", "头痛"), (r"眩晕", "眩晕"),
    (r"落枕|项痹|项强|颈", "颈痛"), (r"漏肩风|肩痹|肩凝|肩", "肩痛"), (r"肘劳", "肱骨外上髁炎"), (r"膝痹|膝", "膝关节痛"), (r"足跟痛", "跟痛症"),
    (r"腰痛|腰痹|腰", "腰痛"), (r"扭伤|伤筋|筋伤", "软组织损伤"), (r"腱鞘囊肿", "腱鞘囊肿"), (r"痹证|痹症|痛痹|行痹|着痹|热痹|痹病", "关节痛"),
    (r"痿证|痿症|痿病", "肢体无力"), (r"颤证|颤病", "震颤"), (r"痉证|抽搐|惊风", "抽搐"), (r"厥证|晕厥", "晕厥"), (r"脱证", "休克"),
    (r"胃脘痛|胃痛|胃痞|痞满", "上腹痛"), (r"胁痛", "胁肋痛"), (r"腹痛", "腹痛"), (r"呕吐", "呕吐"), (r"呃逆", "呃逆"), (r"泄泻", "腹泻"), (r"便秘", "便秘"),
    (r"痢疾", "细菌性痢疾"), (r"黄疸", "黄疸"), (r"鼓胀", "腹水"), (r"肠痈", "急性阑尾炎"), (r"痔", "痔疮"), (r"脱肛", "直肠脱垂"),
    (r"胸痹|心痛", "冠心病心绞痛"), (r"心悸|怔忡", "心悸"), (r"不寐|失眠", "失眠"), (r"多寐|嗜睡", "嗜睡"), (r"郁证|郁病", "抑郁状态"),
    (r"癫证|狂证|癫狂|癫病", "精神障碍"), (r"痫证|痫病", "癫痫"), (r"健忘|痴呆|呆病", "认知功能减退"), (r"消渴", "糖尿病"), (r"水肿", "水肿"),
    (r"淋证|石淋|热淋", "尿路感染/泌尿系结石"), (r"癃闭", "尿潴留"), (r"遗尿", "遗尿"), (r"遗精|阳痿|早泄", "男性性功能障碍"), (r"不育", "男性不育"),
    (r"哮喘|哮病|喘证|哮证", "支气管哮喘"), (r"肺胀", "慢性阻塞性肺疾病"), (r"咳嗽", "咳嗽"), (r"感冒|伤风", "上呼吸道感染"), (r"发热|壮热", "发热"),
    (r"汗证|盗汗|自汗", "多汗症"), (r"虚劳|疲劳", "慢性疲劳"), (r"肥胖", "单纯性肥胖"), (r"中暑", "中暑"), (r"血证|衄血|鼻衄", "出血"), (r"紫癜", "紫癜"),
    (r"耳鸣|耳聋|暴聋", "耳鸣/听力下降"), (r"鼻渊", "鼻窦炎"), (r"鼻鼽", "变应性鼻炎"), (r"牙痛", "牙痛"), (r"咽喉|喉痹|乳蛾", "咽痛/扁桃体炎"), (r"口疮|口糜", "口腔溃疡"),
    (r"目赤|天行赤眼", "急性结膜炎"), (r"针眼|麦粒肿", "睑腺炎"), (r"近视", "近视"), (r"目痒|眼疳", "眼部炎症"),
    (r"蛇串疮|缠腰火丹", "带状疱疹"), (r"瘾疹", "荨麻疹"), (r"湿疮|湿疹", "湿疹"), (r"粉刺|痤疮", "痤疮"), (r"油风|斑秃|脱发", "斑秃"), (r"疣", "疣"),
    (r"丹毒", "丹毒"), (r"疔|疖|痈", "皮肤软组织感染"), (r"冻疮", "冻疮"), (r"牛皮癣|白疕", "银屑病/神经性皮炎"), (r"乳痈", "急性乳腺炎"), (r"乳癖", "乳腺增生"),
    (r"瘿", "甲状腺肿"), (r"痄腮", "流行性腮腺炎"), (r"疳", "小儿营养不良"), (r"脑瘫|五迟|五软", "脑性瘫痪"), (r"抽动", "抽动障碍"), (r"厌食", "小儿厌食"),
    (r"痛经", "痛经"), (r"崩漏", "异常子宫出血"), (r"闭经|经闭", "闭经"), (r"带下", "阴道炎/带下病"), (r"月经|经行|经期", "月经失调"),
    (r"绝经前后诸证|经断前后", "围绝经期综合征"), (r"不孕", "不孕症"), (r"胎位", "胎位异常"), (r"滞产|难产", "产程异常"), (r"缺乳", "产后乳汁不足"),
    (r"阴挺", "子宫脱垂"), (r"产后", "产后疾病"), (r"乳", "乳腺相关疾病"),
]


def symptom_label_from_tcm(tcm_dx: str, chief: str) -> str:
    for pat, lab in TCM_TO_SYMPTOM_LABEL:
        if re.search(pat, tcm_dx or ""):
            return lab
    c = re.sub(r"^(?:主诉|现病史)[:：]\s*", "", (chief or "").strip())
    c = re.split(r"[。；，,;]", c, 1)[0]
    c = re.sub(r"[\d一二三四五六七八九十半两]+[年月周日天小时余]+.*$", "", c).strip("，。 ")
    if c and len(c) <= 12 and not re.search(r"患者|平素|未婚|已婚|某|岁|因|后", c):
        return c
    return (tcm_dx.strip() + "对应疾病") if tcm_dx else "病因"


def is_symptom_label(wm_dx: str) -> bool:
    """判断西医诊断是否为症状性标签（腰痛/颈痛/关节痛…或"X对应疾病"）——此时应标（待查）。"""
    d = re.sub(r"（待查）|\(待查\)", "", wm_dx or "").strip()
    if not d or "；" in d:
        return False
    sym = {"腰痛", "颈痛", "肩痛", "膝关节痛", "关节痛", "头痛", "眩晕", "上腹痛", "腹痛", "胁肋痛", "面部疼痛", "肢体无力", "震颤", "抽搐", "晕厥",
           "呕吐", "呃逆", "腹泻", "便秘", "心悸", "失眠", "嗜睡", "咳嗽", "发热", "出血", "耳鸣/听力下降", "牙痛", "咽痛/扁桃体炎", "水肿", "黄疸", "腹水",
           "病因", "抑郁状态", "认知功能减退", "精神障碍", "慢性疲劳", "多汗症", "月经失调", "异常子宫出血", "闭经", "产程异常", "软组织损伤", "跟痛症"}
    if d in sym or d.endswith("对应疾病"):
        return True

    return bool(re.search(r"痛|晕|鸣|麻木|乏力|无力|减退|失调|不适|障碍|不利|胀|酸", d)) and not re.search(r"炎|病|症|综合征|损伤|突出|骨折|狭窄|结石|癌|瘤|梗|疹|疮|痹$", d)


_AUX_SYN = [(r"椎间盘突出|间盘突出|膨出", "椎间盘突出|椎间盘膨出|膨出|颈椎病|腰椎病|坐骨神经"), (r"骨质增生|退行性|退变|骨赘", "骨关节炎|退行性|骨质增生|颈椎病|腰椎"),
            (r"椎管狭窄|狭窄", "狭窄|椎间盘突出|颈椎病"), (r"出血", "出血|卒中|中风"), (r"梗死|梗塞|缺血灶", "梗死|梗塞|卒中|脑血栓|中风"),
            (r"溶骨|骨质破坏|转移", "癌|肿瘤|转移"), (r"肩袖", "肩袖|肩关节"), (r"骨折", "骨折"), (r"积液", "积液|滑膜炎|关节炎|盆腔炎"),
            (r"结石", "结石|绞痛"), (r"胆囊壁增厚|胆囊", "胆囊炎|胆石"), (r"神经损伤|神经断裂|传导", "神经"), (r"疱疹", "疱疹")]


def aux_consistent(aux_text: str, wm_dx: str) -> bool | None:
    """辅助检查里有明确病变描述时，西医诊断是否与之相容（同义表）。无可比信息 → None。"""
    if not aux_text or not wm_dx:
        return None
    checks = [(m, b) for m, b in ((_positive_match(a, aux_text), b) for a, b in _AUX_SYN) if m]
    if not checks:
        return None
    return any(re.search(b, wm_dx) for _, b in checks)



RED_FLAGS: list[tuple[str, str, str, str, str]] = [
    ("RF-01", r"面色苍白[^。；\n]{0,30}(?:冷汗|汗出|大汗)|(?:冷汗|大汗)[^。；\n]{0,30}面色苍白|口唇(?:发绀|青紫)|意识(?:蒙眬|模糊|不清|障碍)|神志(?:不清|模糊)|昏迷|休克|手足厥冷",
     "E", "存在意识改变、面色苍白冷汗、口唇发绀或循环衰竭等危象表现", "立即急诊处理并监测生命体征，排除急腹症、心血管事件、休克；针灸仅可作为辅助"),
    ("RF-02", r"(?:胃|上腹|脘腹|心前区|胸(?!椎|背|锁|廓))[^。；，\n]{0,10}(?:剧烈|剧痛|难以忍受|绞痛|突发)|心前区[^。；\n]{0,10}(?:痛|压榨|憋闷)|胸痛[^。；\n]{0,15}(?:胸闷|气促|大汗|放射|压榨|憋)|(?:突发|阵发性|发作性|活动后|劳累后)[^。；，\n]{0,6}胸痛|胸闷[^。；，\n]{0,6}(?:憋气|气促|大汗)|压榨",
     "E", "急性剧烈上腹/胸部疼痛或心前区症状，需除外心肌梗死、主动脉夹层、消化道穿孔、胰腺炎、胆道急症", "急查心电图、心肌酶/肌钙蛋白、淀粉酶、腹部超声/CT；症状不缓解或伴生命体征异常即转急诊"),
    ("RF-03", r"右下腹[^。；，\n]{0,6}(?:痛|坠胀|压痛)|麦氏点[^。；，\n]{0,4}(?:压痛|阳性|\(\+\)|（\+）)|转移性[^。；，\n]{0,4}腹痛|反跳痛|板状腹|腹肌紧张|腹部?[^。；，\n]{0,4}肌紧张",
     "E", "存在右下腹痛/腹膜刺激征相关描述，需除外急性阑尾炎、异位妊娠、卵巢囊肿蒂扭转等急腹症", "外科/妇科会诊，查血常规、腹部及盆腔超声，育龄女性查尿HCG；未排除急腹症前不宜以针灸为主"),
    ("RF-04", r"大小便失禁|二便失禁|鞍区[^。；，\n]{0,4}(?:麻木|感觉)|会阴[^。；，\n]{0,6}(?:麻木|感觉减退)|马尾|(?:腰痛|腰部疼痛|腰腿痛|下肢(?:麻木|无力|放射))[^。；\n]{0,40}(?:尿潴留|排尿困难)|(?:尿潴留|排尿困难)[^。；\n]{0,40}(?:腰痛|腰部疼痛|腰腿痛|下肢(?:麻木|无力|放射))",
     "E", "疑似马尾综合征（二便功能障碍/鞍区感觉异常伴腰腿症状）", "急诊腰椎MRI并请脊柱外科急会诊，属外科急症，禁止延误"),
    ("RF-05", r"进行性[^。；，\n]{0,6}(?:无力|肌力下降|肌肉萎缩|瘫)|肌力\s*[0-3Ⅰ-Ⅲ]\s*级|肌力\s*(?:Ⅰ|Ⅱ|Ⅲ)级|足下垂|下肢[^。；，\n]{0,6}(?:瘫|无力明显)|截瘫|(?:双|单)?下肢(?:软弱)?无力[^。；\n]{0,20}(?:加重|萎缩|进行性)|(?:肢体|下肢|上肢|手部?)[^。；，\n]{0,6}肌肉萎缩|肌肉萎缩[^。；，\n]{0,4}(?:明显|进行性)|全身肌肉萎缩",
     "E", "存在进行性/明显神经功能缺损（肌力下降、肌肉萎缩、足下垂等）", "尽快完善脊柱或头颅MRI及神经科/脊柱外科评估，明确有无脊髓、神经根受压或其他神经系统疾病"),
    ("RF-06", r"(?:突发|突然|骤然|霹雳样|一生中最|首次)[^。；，\n]{0,8}(?:剧烈)?头痛|头痛[^。；\n]{0,20}(?:喷射|复视|颈项强直|颈强|意识|抽搐)|头痛[^。；，\n]{0,12}(?:呕吐)|头痛[^。；，\n]{0,12}(?:进行性加重|逐渐加重)|蛛网膜下腔出血",
     "E", "头痛伴警示征象（突发剧烈/伴呕吐、复视、颈强、意识改变或进行性加重），需除外蛛网膜下腔出血、颅内占位、脑膜炎", "急诊头颅CT/MRI，必要时腰穿；神经科急会诊"),
    ("RF-07", r"(?:一侧|右侧|左侧|偏身)(?:肢体)?(?:活动不利|无力|麻木)[^。；\n]{0,30}(?:小时|急性|突然|突发)|(?:突然|突发|急性)[^。；\n]{0,20}(?:肢体活动不利|偏瘫|半身不遂|口角歪斜|言语不利|失语)",
     "E", "急性起病的偏侧肢体活动障碍/言语不利，需首先除外急性脑卒中（时间窗内可溶栓/取栓）", "立即急诊，头颅CT/MRI，启动卒中绿色通道；针灸在急性期仅作辅助并需在神经科监护下进行"),
    ("RF-08", r"体重(?:下降|减轻|明显减轻)|消瘦[^。；\n]{0,20}(?:乏力|盗汗|低热)|夜间痛[^。；\n]{0,20}(?:持续|不缓解|体重)|骨质破坏|溶骨|病理性骨折|骨转移|转移(?:癌|瘤)",
     "E", "存在提示肿瘤或感染的全身警示征象（不明原因体重下降、持续夜间痛、影像骨质破坏/转移等）", "完善血常规、ESR、CRP、肿瘤标志物及相应影像（MRI/CT/骨扫描），肿瘤科/骨科评估后再决定针灸方案"),
    ("RF-09", r"(?:发热|高热|寒战)[^。；\n]{0,20}(?:腰痛|背痛|关节(?:肿)?痛|颈痛|脊柱)|(?:腰痛|背痛|关节(?:肿)?痛|颈痛|脊柱)[^。；\n]{0,20}(?:发热|高热|寒战)",
     "E", "脊柱/关节疼痛伴发热，需除外化脓性或结核性脊柱炎、化脓性关节炎、椎旁脓肿等感染", "血常规、ESR、CRP、血培养，相应部位MRI；感染未排除前不宜针刺局部"),
    ("RF-10", r"(?:外伤|摔|跌|坠|车祸|撞|扭)[^。；\n]{0,30}(?:畸形|骨擦|异常活动|不能负重|无法行走|不能站立)|(?:怀疑|疑似|考虑)骨折|压缩性?骨折|骨折",
     "E", "外伤后出现畸形、不能负重/无法行走或已提示骨折", "先行X线（必要时CT），骨科处理后再考虑针灸；已知骨折者针刺避开骨折部位"),
    ("RF-11", r"孕\d*周|妊娠\d*(?:周|个?月)|怀孕|孕妇|停经[^。；，\n]{0,10}(?:周|月)|早孕|臀位|胎位",
     "N", "妊娠状态", "腰骶部、下腹部及合谷、三阴交、至阴、昆仑等孕妇慎用/禁用穴位须回避，强刺激手法禁用；须在妇产科评估下进行"),
    ("RF-12", r"抗凝|华法林|利伐沙班|阿司匹林[^。；，\n]{0,6}氯吡格雷|血小板\s*(?:[1-4]\d|[1-9])\s*[×x]|凝血功能异常|血友病|出血倾向",
     "N", "存在出血倾向或抗凝治疗", "深刺、放血、拔罐慎用，压迫止血时间延长，必要时查凝血功能"),
    ("RF-13", r"起搏器|心脏起搏|除颤器", "N", "植入心脏起搏器/除颤器", "禁用电针，避免电刺激类治疗"),
    ("RF-14", r"糖尿病[^。；\n]{0,30}(?:足|溃疡|感觉减退|周围神经)|糖尿病足", "N", "糖尿病伴末梢病变", "下肢及足部针刺严格无菌，慎用艾灸以防烫伤，注意伤口愈合"),
    ("RF-15", r"(?:癌|肿瘤|恶性)(?:术后|病史|根治术)|(?:直肠|胃|肺|乳腺|肝|卵巢|宫颈|前列腺|甲状腺)癌", "N", "有肿瘤病史", "疼痛需先排除肿瘤复发或转移（影像学、肿瘤标志物）；针刺避开肿瘤及放疗区域，注意血象"),
    ("RF-16", r"发现[^。；，\n]{0,6}肿瘤|肿瘤[^。；，\n]{0,4}(?:\d+年|术)", "N", "存在肿瘤相关病史", "疼痛需先排除肿瘤进展相关原因；治疗方案须与肿瘤专科协同"),
]
_NEG_BEFORE = re.compile(r"(?:(?:无|未见|否认|不伴|未发现|排除|未出现|未诉|无明显|未闻及)(?:[^，。；\n无]{0,10}[、及和/与])?|\d+\s*年前[^。；\n]{0,10}|多年[^。；\n]{0,6})\s*$")
_NEG_AFTER = re.compile(r"^\s*(?:阴性|\(-\)|（-）|\(−\)|正常|未见异常|愈合|已愈合|已愈|术后|后遗)")
_EMERGENCY_CV = {"CV-01", "CV-03", "CV-04", "CV-07"}

_CRIT = [
    (re.compile(r"血糖[:：]?\s*(\d+(?:\.\d+)?)\s*mmol"), lambda v: v >= 33.3 or v <= 2.8, "血糖{v} mmol/L 为危急值，需除外高血糖高渗状态/酮症酸中毒或低血糖", "立即复测血糖、急查血酮/血气/电解质，内分泌或急诊处理；同时核对该数值是否录入错误"),
    (re.compile(r"血糖[:：]?\s*(\d+(?:\.\d+)?)\s*mmol"), lambda v: 11.1 <= v < 33.3, "随机血糖{v} mmol/L 明显升高", "复查空腹血糖及HbA1c，评估糖尿病及其并发症"),
    (re.compile(r"血压[:：]?\s*(\d{2,3})\s*/\s*(\d{2,3})"), lambda v: v[0] >= 180 or v[1] >= 110, "血压{v0}/{v1} mmHg 达高血压急症/亚急症范围", "复测血压并评估靶器官损害（头痛、胸痛、神经系统体征）；先降压处理，再考虑针灸"),
    (re.compile(r"血压[:：]?\s*(\d{2,3})\s*/\s*(\d{2,3})"), lambda v: v[0] < 90, "血压{v0}/{v1} mmHg 偏低", "警惕休克/血容量不足，监测生命体征"),
    (re.compile(r"(?:心率|脉搏)[:：]?\s*(\d{2,3})\s*次"), lambda v: v >= 120 or v <= 45, "心率{v}次/分 明显异常", "心电图及心内科评估"),
    (re.compile(r"体温[:：]?\s*(\d{2}(?:\.\d)?)\s*[℃°]"), lambda v: v >= 38.5, "体温{v}℃ 发热", "查找感染灶，血常规、CRP；感染未排除前不宜针刺局部"),
    (re.compile(r"(?:SpO2|血氧饱和度|指氧)[:：]?\s*(\d{2})\s*%"), lambda v: v < 93, "血氧饱和度{v}% 偏低", "吸氧并急查血气，除外肺栓塞、肺部感染等"),
]


def _positive_match(rx: str, t: str):
    for m in re.finditer(rx, t):
        if _NEG_BEFORE.search(t[max(0, m.start() - 14): m.start()]):
            continue
        if _NEG_AFTER.match(t[m.end(): m.end() + 6]):
            continue
        return m
    return None


def red_flag_warnings(input_text: str) -> list[dict]:
    
    t = str(input_text or "")
    t = re.sub(r"^请根据以下病例给出针灸诊疗方案。\s*", "", t)
    hits: list[dict] = []
    for rid, pat, level, reason, action in RED_FLAGS:
        m = _positive_match(pat, t)
        if m:
            hits.append({"rule": rid, "level": level, "reason": reason, "action": action, "span": t[max(0, m.start() - 12): m.end() + 12].replace("\n", " ")})
    for j, (rx, cond, reason, action) in enumerate(_CRIT):
        for m in rx.finditer(t):
            vals = tuple(float(x) for x in m.groups())
            v = vals[0] if len(vals) == 1 else vals
            try:
                ok = cond(v)
            except Exception:
                ok = False
            if ok:
                rid = f"CV-{j + 1:02d}"
                fmt = {"v": vals[0]} if len(vals) == 1 else {"v0": int(vals[0]), "v1": int(vals[1])}
                hits.append({"rule": rid, "level": "E" if rid in _EMERGENCY_CV else "N", "reason": reason.format(**fmt), "action": action, "span": m.group(0)})
                break
    return hits


def is_emergency_level(hits: list[dict]) -> bool:
    return any(h["level"] == "E" for h in hits)


def format_notes(hits: list[dict]) -> tuple[str, list[str]]:
  
    em = [h for h in hits if h["level"] == "E"]
    others = [f"{h['reason']}：{h['action']}。" for h in hits if h["level"] != "E"]
    line = (EMERGENCY_TAG + "".join(f"{h['reason']}，{h['action']}。" for h in em)) if em else ""
    return line, others


PATTERN_KB: dict[str, tuple[str, str]] = {
    "气滞血瘀": ("气机郁滞，血行不畅，瘀阻经络，不通则痛。", "行气活血，化瘀通络止痛"),
    "瘀血阻滞": ("瘀血内停，经脉痹阻，不通则痛。", "活血化瘀，通络止痛"),
    "气血瘀滞": ("气血运行不畅，瘀滞经络，不通则痛。", "活血行气，通络止痛"),
    "寒湿痹阻": ("寒湿之邪侵袭经络，气血凝滞，经脉痹阻。", "散寒除湿，温经通络"),
    "风寒袭络": ("风寒之邪外袭，客于经络，气血痹阻。", "祛风散寒，温经通络"),
    "风寒湿痹": ("风寒湿邪杂至，痹阻经络关节，气血运行不畅。", "祛风散寒除湿，通络止痛"),
    "湿热痹阻": ("湿热蕴结，痹阻经络关节。", "清热利湿，通络止痛"),
    "肝肾亏虚": ("肝肾精血不足，筋骨失于濡养。", "补益肝肾，强筋壮骨"),
    "肝肾不足": ("肝肾亏虚，筋骨失养，兼经络不畅。", "补益肝肾，舒筋通络"),
    "肾虚": ("肾精亏虚，腰府失养。", "补肾益精，强腰止痛"),
    "气血亏虚": ("气血不足，经脉失养，不荣则痛。", "益气养血，濡养经脉"),
    "肝阳上亢": ("肝肾阴虚，肝阳上亢，上扰清窍。", "平肝潜阳，滋阴息风"),
    "肝郁气滞": ("情志不遂，肝失疏泄，气机郁滞。", "疏肝理气，通络止痛"),
    "痰湿阻络": ("痰湿内生，阻滞经络。", "化痰祛湿，通络止痛"),
    "阴虚火旺": ("阴液亏虚，虚火内扰。", "滋阴降火"),
    "风痰阻络": ("肝风夹痰，上扰清窍，脑脉痹阻。", "息风化痰，活血通络"),
}


def kb_pathogenesis_and_principle(pattern: str) -> tuple[str, str]:
    for k, v in PATTERN_KB.items():
        if k in (pattern or ""):
            return v
    return "", ""
