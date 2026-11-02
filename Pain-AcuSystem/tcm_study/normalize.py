from __future__ import annotations

import re

SECTIONS = ["诊疗思路", "西医诊断", "中医诊断", "证型", "病机", "治法", "取穴", "操作", "注意事项", "方义"]
EXTRA_KEY = "其他内容"
_ALIAS = {
    "治则": "治法", "治疗原则": "治法", "治疗法则": "治法",
    "选穴": "取穴", "穴位": "取穴", "处方": "取穴", "针灸处方": "取穴", "针灸取穴": "取穴", "取穴与操作": "取穴", "针灸取穴与操作": "取穴",
    "针灸取穴及操作": "取穴", "针灸方案": "取穴", "穴位处方": "取穴", "主穴": "取穴", "配穴": "取穴", "辅助穴": "取穴", "取穴方案": "取穴",
    "手法": "操作", "针刺操作": "操作", "操作方法": "操作", "针刺方法": "操作", "刺法": "操作", "针法": "操作", "操作要点": "操作",
    "方解": "方义", "方义分析": "方义", "选穴依据": "方义", "配穴分析": "方义",
    "辨证": "证型", "辨证分型": "证型", "中医辨证": "证型", "中医证型": "证型", "证候": "证型",
    "辨证分析": "病机", "病因病机": "病机", "中医病机": "病机", "病机分析": "病机",
    "诊疗分析": "诊疗思路", "诊断分析": "诊疗思路", "分析": "诊疗思路", "临床分析": "诊疗思路", "诊断依据": "诊疗思路", "思路": "诊疗思路",
    "注意": "注意事项", "风险提示": "注意事项", "安全提示": "注意事项", "急危重症警示": "注意事项", "警示": "注意事项", "转诊建议": "注意事项",
    "随访": "注意事项", "跟踪随访": "注意事项", "医嘱": "注意事项", "调护": "注意事项", "预防调护": "注意事项", "建议": "注意事项",
    "西医病名": "西医诊断", "西医对应": "西医诊断", "对应西医": "西医诊断", "西医": "西医诊断",
    "中医病名": "中医诊断", "中医": "中医诊断", "中医病": "中医诊断",
}
_HEAD_WORDS = "|".join(sorted({*SECTIONS, *_ALIAS, "诊断", "初步诊断", "诊断建议", "诊断结果", "西医诊断", "中医诊断"}, key=len, reverse=True))

_HEAD = re.compile(
    r"^[ \t]*(?:[【〔\[](?P<b>" + _HEAD_WORDS + r")[】〕\]]"
    r"|#{1,6}[ \t]*\**[ \t]*(?P<h1>" + _HEAD_WORDS + r")[ \t]*\**[ \t]*[:：]?[ \t]*$"
    r"|\*\*[ \t]*(?P<h2>" + _HEAD_WORDS + r")[ \t]*[:：]?[ \t]*\*\*[ \t]*[:：]?[ \t]*$"
    r"|(?:[-*•·][ \t]*|\d+[\.．、)）][ \t]*)?\**[ \t]*(?P<h>" + _HEAD_WORDS + r")[ \t]*\**[ \t]*[:：])[ \t]*",
    re.M)
_NOISE_SENT = re.compile(
    r"[^。！？!?\n]*(?:仅供参考|请(?:及时)?咨询|遵医嘱|不能替代|不可替代|免责声明|如有不适请|作为AI|作为一个AI|我是AI|语言模型)[^。！？!?\n]*[。！？!?]?")
_GREETING = re.compile(r"^\s*(?:您好|你好|好的|当然|以下是|根据(?:您|你)?提供的(?:病例|信息)[^。\n]{0,20}[，,:：])[^\n]*\n?", re.M)
_TCM_IN_DX = re.compile(r"[（(]\s*(?:中医(?:病名|诊断)?|对应中医|中医称为?)\s*[:：]\s*([^）)]+)[）)]")
_WM_IN_DX = re.compile(r"[（(]\s*(?:西医(?:病名|诊断)?|对应西医|西医对应)\s*[:：]\s*([^）)]+)[）)]")
_SENTENCE_LINE = re.compile(r"应|如有|注意|避免|确保|建议|防止|嘱|以免|需要|可以|可结合|采用")
_SUBLINE = re.compile(r"^[ \t]*(?:[-*•·]\s*|\d+[\.．、)）]\s*)?\**\s*(西医诊断|中医诊断|西医病名|中医病名|西医对应|对应西医|中医|西医|证型|辨证)\s*\**\s*[:：]\s*(.+)$", re.M)


def strip_noise(text: str) -> str:
    t = str(text or "").replace("\r", "")
    t = re.sub(r"```[a-zA-Z]*", "", t)
    t = re.sub(r"\*\*|__", "", t)
    t = _GREETING.sub("", t)
    t = _NOISE_SENT.sub("", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def _clean_md(s: str) -> str:
    s = re.sub(r"\*\*|__", "", s)
    s = re.sub(r"^\s*#{1,6}\s*", "", s, flags=re.M)
    return s.strip()


def _points_from_list(body: str) -> str:
  
    names = []
    for line in body.split("\n"):
        line = _clean_md(line)
        line = re.sub(r"^\s*(?:[-*•·]|\d+[\.．、)）])\s*", "", line)
        line = re.sub(r"^(?:主穴|配穴|辅助穴|加减|随症配穴|操作|操作方法|注意事项)\s*[:：]?\s*", "", line)
        if not line or re.match(r"^(?:操作|手法|针刺|每日|留针|得气|先行|采用|注意)", line):
            continue
        head = re.split(r"[:：]", line, 1)[0]
        head = re.sub(r"[（(][^）)]*[）)]", "", head).strip(" 。；;，,")
        if 1 < len(head) <= 40 and not re.search(r"寸|分钟|留针|直刺|斜刺|平刺", head) and not (len(head) > 12 and _SENTENCE_LINE.search(head)):
            names.append(head)
    return "、".join(dict.fromkeys(names))


def _split_dx_block(body: str, out: dict) -> None:
   
    found = False
    for m in _SUBLINE.finditer(body):
        k = _ALIAS.get(m.group(1), m.group(1))
        k = "证型" if k in ("证型", "辨证") else k
        out[k] = (out[k] + "；" + m.group(2).strip()) if out.get(k) else m.group(2).strip()
        found = True
    rest = _SUBLINE.sub("", body).strip()
    if not found:
        rest = body.strip()
    if rest:
        mt, mw = _TCM_IN_DX.search(rest), _WM_IN_DX.search(rest)
        if mt and "中医诊断" not in out:
            out["中医诊断"] = mt.group(1).strip()
        if mw and "西医诊断" not in out:
            out["西医诊断"] = mw.group(1).strip()
        rest2 = _TCM_IN_DX.sub("", _WM_IN_DX.sub("", rest)).strip()
        if rest2:
            key = "西医诊断" if "西医诊断" not in out else "诊疗思路"
            out[key] = (out[key] + "\n" + rest2) if out.get(key) else rest2


def parse_sections(text: str) -> dict[str, str]:
    t = strip_noise(text)
    out: dict[str, str] = {}
    heads = list(_HEAD.finditer(t))
    if not heads:
        return {EXTRA_KEY: _clean_md(t)}
    lead = _clean_md(t[: heads[0].start()])
    if lead:
        out["诊疗思路"] = lead
    for i, m in enumerate(heads):
        raw = (m.group("b") or m.group("h1") or m.group("h2") or m.group("h")).strip()
        name = _ALIAS.get(raw, raw)
        body = t[m.end(): heads[i + 1].start() if i + 1 < len(heads) else len(t)]
        body = _clean_md(body)
        if name in ("诊断", "初步诊断", "诊断建议", "诊断结果"):
            _split_dx_block(body, out)
            continue
        if name == "取穴" and raw in ("主穴", "配穴", "辅助穴") or (name == "取穴" and "\n" in body and re.search(r"[:：]", body)):
            pts = _points_from_list(body) if "\n" in body else body
            ops = "\n".join(l for l in body.split("\n") if re.search(r"寸|留针|得气|分钟|手法|捻转|提插|补泻|每日|每次|疗程", l))
            if pts:
                out["取穴"] = (out["取穴"] + "、" + pts) if out.get("取穴") else pts
            if ops and raw not in ("主穴", "配穴", "辅助穴"):
                out["操作"] = (out["操作"] + "\n" + ops) if out.get("操作") else ops
            continue
        if name == "西医诊断" or name == "中医诊断":
            
            mt, mw = _TCM_IN_DX.search(body), _WM_IN_DX.search(body)
            if mt and name == "西医诊断":
                out["中医诊断"] = out.get("中医诊断") or mt.group(1).strip(); body = _TCM_IN_DX.sub("", body).strip()
            if mw and name == "中医诊断":
                out["西医诊断"] = out.get("西医诊断") or mw.group(1).strip(); body = _WM_IN_DX.sub("", body).strip()
        key = name if name in SECTIONS else EXTRA_KEY
        if key == EXTRA_KEY:
            body = f"{raw}：{body}"
        out[key] = (out[key] + ("\n" if key in (EXTRA_KEY, "注意事项", "操作", "诊疗思路") else "；") + body) if out.get(key) else body
    return out


RENDERED = SECTIONS[:9]


def _strip_codes(t: str) -> str:
    from tcm_study.schema import strip_library_codes
    return strip_library_codes(t)


def render(sections: dict[str, str]) -> str:
    lines = []
    for name in RENDERED:
        lines.append(f"【{name}】{_strip_codes(sections.get(name, '').strip()) or '（未给出）'}")
    return "\n".join(lines)


def normalize(text: str) -> tuple[dict[str, str], str]:
    from tcm_study.schema import strip_null_warning
    text = strip_null_warning(text)[0]
    sec = parse_sections(text)
   
    from tcm_study.schema import relocate_emergency
    sec = relocate_emergency({k: v for k, v in sec.items() if k != EXTRA_KEY} | ({EXTRA_KEY: sec[EXTRA_KEY]} if EXTRA_KEY in sec else {}))
    return sec, render(sec)
