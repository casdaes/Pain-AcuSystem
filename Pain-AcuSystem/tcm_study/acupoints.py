from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

_SPLIT = re.compile(r"[、，,；;\n/]+|以及|及|和|配合|＋|\+")
_PAREN = re.compile(r"[（(][^）)]*[）)]")
_PREFIX = re.compile(r"^(?:[甲乙丙丁]组穴?[:：]?|(?:选取|另取|再取|首选|主穴|配穴|取|加|配|针刺|针灸|艾灸|温针|电针|双侧|双|左侧|右侧|左|右|患侧|健侧|对侧|同侧)[:：]?)+")
_SUFFIX = re.compile(r"(?:双侧|双|左侧|右侧|患侧|健侧)$")
_JIAJI = re.compile(r"夹脊")
_EXTRA_DROP = re.compile(r"^(?:灸|温针?|针刺?\d*次?|操作|手法|局部伤口.*)$")
_EXTRA_SETS = {"十宣": "经外奇穴组", "八邪": "经外奇穴组", "八风": "经外奇穴组", "四缝": "经外奇穴组"}
_CODE = re.compile(r"\b([A-Z]{1,3}\d{1,2}(?:-[A-Z]+\d+)?)\b")
from tcm_study.needling import PREG_CUE
TONGUE_OUT = re.compile(r"舌[质体苔]?[淡红暗紫绛胖瘦边尖]|舌苔|苔[白黄薄厚腻燥少剥润滑灰黑]")
PULSE_OUT = re.compile(r"脉[弦细滑沉数缓紧濡涩浮迟虚实结代洪微弱]")
MISSING_ID = re.compile(r"(?:舌|脉|舌脉|舌象|脉象|四诊)[^。；\n]{0,14}(?:未记录|未提供|缺失|欠缺|不详|未见|待补充|需补充|补充|合参)|无法[^。；\n]{0,8}(?:辨证|定证)|不能[^。；\n]{0,8}(?:明确|确定)[^。；\n]{0,4}(?:证型|辨证)")
HEDGE = re.compile(r"建议[^。；\n]{0,8}(?:进一步|完善|四诊合参)|仅供参考|需结合")
REFERRAL = re.compile(r"转诊|急诊|立即就医|尽快就医|及时就医|排除[^。；\n]{0,10}(?:急腹症|心梗|心肌梗死|马尾|夹层|骨折|肿瘤|感染|卒中|脑梗)|进一步检查|完善[^。；\n]{0,6}检查")
_DEPTH = re.compile(r"(\d+(?:\.\d+)?)\s*(?:[-~—至]\s*(\d+(?:\.\d+)?))?\s*寸")
_SEG = re.compile(r"(\d{1,2})\s*[-~—至]\s*[CTLS]?(\d{1,2})")
_JIAJI_DEFAULT = {"颈": ("C", 2, 7), "胸": ("T", 1, 12), "腰": ("L", 1, 5), "骶": ("S", 1, 4)}
_COMBO = {"八髎": ["上髎", "次髎", "中髎", "下髎"], "内外膝眼": ["内膝眼", "犊鼻"], "颈三针": ["天柱", "百劳", "大杼"],
          "肩三针": ["肩髃", "肩前", "肩后"]}


class AcupointLexicon:
    def __init__(self, xlsx: str | Path):
        lib = pd.read_excel(xlsx, sheet_name="穴位标准库", dtype=str, keep_default_na=False)
        self.lib = lib
        self.rows = {r["穴位名称"].strip(): r for r in lib.to_dict("records")}
        self.alias: dict[str, str] = {}
        for name, r in self.rows.items():
            for a in re.split(r"[|、,，;；/]+", r.get("别名", "")):
                a = a.strip()
                if a and a not in self.rows:
                    self.alias[a] = name
        self.code_to_name = {r["WHO code"].strip(): n for n, r in self.rows.items() if r.get("WHO code", "").strip()}
        wl = pd.read_excel(xlsx, sheet_name="非单穴白名单", header=3, dtype=str, keep_default_na=False)
        self.rules = []
        for r in wl.to_dict("records"):
            pat = r.get("匹配正则", "").strip()
            if pat and r.get("类别", "").strip() and not r["类别"].startswith("统计口径"):
                try:
                    self.rules.append({"category": r["类别"].strip(), "canonical": r["规范表述"].strip(), "re": re.compile(pat),
                                       "counted": r.get("计入穴位指标", "").startswith(("是", "部分")), "drop": r.get("计入穴位指标", "").startswith("否(剔除)"),
                                       "note": r.get("展开/映射", "")})
                except re.error:
                    pass

    # ---- token classification ------------------------------------------------------------
    def classify(self, raw: str) -> dict:
        tok = _PAREN.sub("", raw).strip(" 　。.")
        tok = _PREFIX.sub("", tok)
        tok = _SUFFIX.sub("", tok) if len(tok) > 2 else tok
        base = {"raw": raw.strip(), "token": tok}
        if not tok or _EXTRA_DROP.match(tok):
            return {**base, "status": "empty"}
        for r in self.rules:
            if r["re"].search(tok):
                if r["drop"]:
                    return {**base, "status": "param"}
                return {**base, "status": "whitelist", "canonical": r["canonical"], "category": r["category"], "counted": r["counted"]}
        if _JIAJI.search(tok) and not any(r["re"].search(tok) for r in self.rules):
            lead = tok[0] if tok[0] in "颈胸腰骶" else ({"C": "颈", "T": "胸", "L": "腰", "S": "骶"}.get(tok[0], ""))
            canon = f"{lead}夹脊" if lead else "夹脊(未指节段)"
            return {**base, "status": "whitelist", "canonical": canon, "category": "夹脊节段", "counted": True}
        for k, cat in _EXTRA_SETS.items():
            if tok == k or tok.endswith(k):
                return {**base, "status": "whitelist", "canonical": k, "category": cat, "counted": False}
        t = tok
        t = t[:-1] if t.endswith("穴") and len(t) > 2 else t
        for cand in (tok, t):
            if cand in self.rows:
                return {**base, "status": "standard", "canonical": cand}
            if cand in self.alias:
                return {**base, "status": "alias", "canonical": self.alias[cand]}
        m = _CODE.fullmatch(tok)
        if m and m.group(1) in self.code_to_name:
            return {**base, "status": "standard", "canonical": self.code_to_name[m.group(1)]}
        return {**base, "status": "unknown"}

    def extract(self, points_text: str) -> list[dict]:
        out = []
        text = re.split(r"(?:手法|操作)\s*[:：]", str(points_text or ""), maxsplit=1)[0].replace("。", "、")
        text = re.sub(r"或", "、", text)
        for raw in _SPLIT.split(_PAREN.sub("", text)):
            raw = raw.strip()
            if raw:
                c = self.classify(raw)
                if c["status"] not in ("empty", "param"):
                    out.append(c)
        return out

    def expand(self, items: list[dict]) -> list[str]:
        """Canonical, comparable point labels; whitelisted composite/segment points are expanded."""
        out: list[str] = []
        for it in items:
            if it["status"] in ("standard", "alias"):
                out.append(it["canonical"])
            elif it["status"] == "whitelist" and it["counted"]:
                c, tok = it["canonical"], it["token"]
                if "夹脊" in c:
                    key = c[0] if c[0] in _JIAJI_DEFAULT else None
                    if key:
                        p, lo, hi = _JIAJI_DEFAULT[key]
                        m = _SEG.search(tok)
                        if m:
                            lo, hi = int(m.group(1)), int(m.group(2))
                        out += [f"{p}{n}夹脊" for n in range(lo, hi + 1)]
                    else:
                        out.append("夹脊(未指节段)")
                elif c in _COMBO:
                    out += _COMBO[c]
                else:
                    out.append(c)
        return list(dict.fromkeys(out))

    # ---- safety / integrity pre-screens ------------------------------------------------------------
    def code_check(self, text: str) -> dict:
        pairs, bad = 0, []
        for m in re.finditer(r"([一-鿿]{2,5})[（(]\s*([A-Z]{1,3}\d{1,2}(?:-[A-Z]+\d+)?)\s*[）)]", text):
            name, code = m.group(1), m.group(2)
            pairs += 1
            canon = self.alias.get(name, name)
            if self.code_to_name.get(code) != canon:
                bad.append(f"{name}({code})")
        return {"code_pairs": pairs, "code_mismatch": bad}

    def safety_flags(self, points: list[str], out_text: str, in_text: str) -> list[dict]:
        flags = []
        preg = bool(PREG_CUE.search(in_text))
       
        from tcm_study.needling import depth_violations
        dv = depth_violations(self, out_text)
        over = {x["point"]: x for x in dv.get("violations", [])}
        deep = set(dv.get("deep_on_forbidden", []))
        for p in points:
            r = self.rows.get(p)
            if not r:
                continue
            if preg and r.get("妊娠禁忌", "").startswith(("A-", "B-")):
                flags.append({"point": p, "candidate": "SE-1" if r["妊娠禁忌"].startswith("A-") else "SE-2", "why": r["妊娠禁忌"]})
            if p in over or p in deep:
                se = [x for x in re.split(r"[|/]", r.get("对应SE条目", "")) if x.startswith("SE-") and x not in ("SE-1", "SE-2")]
                why = f"depth {over[p]['depth']}>{over[p]['limit']}" if p in over else "forbidden deep insertion"
                flags.append({"point": p, "candidate": "|".join(se) or "SE-?", "why": why})
        return flags



# ---------------------------------------------------------------------------------------------------------
def point_metrics(out_pts: list[str], ref_pts: list[str], out_main: set[str] | None = None, ref_main: set[str] | None = None) -> dict:
    a, b = set(out_pts), set(ref_pts)
    inter = a & b
    p = len(inter) / len(a) if a else 0.0
    r = len(inter) / len(b) if b else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    jac = len(inter) / len(a | b) if (a | b) else 0.0
    w = lambda s, main: sum(2 if x in (main or set()) else 1 for x in s)
    wi = sum(min(2 if x in (out_main or set()) else 1, 2 if x in (ref_main or set()) else 1) for x in inter)
    wp = wi / w(a, out_main) if a else 0.0
    wr = wi / w(b, ref_main) if b else 0.0
    wf1 = 2 * wp * wr / (wp + wr) if wp + wr else 0.0
    return {"jaccard": jac, "precision": p, "recall": r, "f1": f1, "weighted_f1": wf1, "n_out": len(a), "n_ref": len(b)}


def main_points(ordered: list[str], frac: float = 0.5) -> set[str]:
    """Handbook rule: points listed first are principal points (weight 2).  Default: first half (at least 1)."""
    k = max(1, round(len(ordered) * frac)) if ordered else 0
    return set(ordered[:k])


def output_autometrics(lex: AcupointLexicon, sections: dict[str, str], full_text: str, input_text: str, task: str) -> dict:
    items = lex.extract(sections.get("取穴", ""))
    pts = lex.expand(items)
    n_named = sum(1 for i in items if i["status"] in ("standard", "alias", "unknown"))
    unknown = [i["token"] for i in items if i["status"] == "unknown"]
    tongue_fab = bool(TONGUE_OUT.search(full_text)) and not TONGUE_OUT.search(input_text)
    pulse_fab = bool(PULSE_OUT.search(full_text)) and not PULSE_OUT.search(input_text)
    if not (TONGUE_OUT.search(input_text) or PULSE_OUT.search(input_text)):
        miss = "identified" if MISSING_ID.search(full_text) else ("partial" if HEDGE.search(full_text) else "silent")
    else:
        miss = "n/a"
    if tongue_fab or pulse_fab:
        miss = "fabricated" if miss != "n/a" else miss
    code = lex.code_check(full_text)
    from tcm_study.needling import depth_violations
    dv = depth_violations(lex, full_text)
    return {
        "n_items": len(items), "n_named_points": n_named, "n_unknown": len(unknown), "unknown_rate": (len(unknown) / n_named) if n_named else 0.0,
        "unknown_tokens": unknown, "ashi_used": any(i.get("category") == "阿是类" for i in items if i["status"] == "whitelist"),
        "points": pts, "code_pairs": code["code_pairs"], "code_mismatch": code["code_mismatch"],
        "tongue_fabricated": tongue_fab, "pulse_fabricated": pulse_fab, "missing_info": miss,
        "mentions_referral": bool(REFERRAL.search(full_text)), "n_chars": len(full_text),
        "n_points_checkable": dv["n_points"], "n_points_with_depth": dv["n_with_depth"],
        "depth_specified_rate": (dv["n_with_depth"] / dv["n_points"]) if dv["n_points"] else None,
        "n_depth_over_limit": len(dv["violations"]), "depth_over_limit": ";".join(f"{v['point']}{v['depth']}>{v['limit']}" for v in dv["violations"]),
        "deep_on_forbidden": ";".join(dv["deep_on_forbidden"]),
        "safety_flags": lex.safety_flags([i["canonical"] for i in items if i["status"] in ("standard", "alias")], full_text, input_text),
    }
