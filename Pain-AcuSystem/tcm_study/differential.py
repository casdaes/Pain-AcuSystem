from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from painrag.retriever import strip_codes
from painrag.textutil import norm_key

_ITEM_SPLIT = re.compile(r"(?:^|\n|\s)-\s+")
_RED_NAME = re.compile(r"警惕【([^】]{2,30})】")
_PENDING = re.compile(r"[（(]\s*待查\s*[)）]|待查")
_BAD_KEY = re.compile(r"未给出鉴别|仅列出需鉴别|症状、体征、实验室|见上文|同上|「|表\s*\d|图\s*\d")
_BAD_RED = re.compile(r"停用|药物|因素|依从|过度|共病|复发|控制|延误|[，,]|但|风险|较重|致残")
_PREG = re.compile(r"妊娠|子痫|孕|产后|产褥|胎")


def _bigrams(t: str) -> set:
    k = norm_key(t)
    return {k[i:i + 2] for i in range(len(k) - 1)}


def _overlap(a: set, text: str) -> float:
    b = _bigrams(text)
    return len(a & b) / max(1, len(b)) if b else 0.0


def _short(s: str, n: int) -> str:
    s = re.sub(r"\s+", "", s).strip("；;，,。 ")
    if len(s) <= n:
        return s
    cut = max(s.rfind(x, 0, n) for x in "，,、；;")
    return s[:cut] if cut >= 8 else s[:n]



SYMPTOM_PROXY: list[tuple[str, str | None, str]] = [
    (r"头痛|头风", None, "紧张型头痛"),
    (r"面部疼痛|面痛", None, "三叉神经痛"),
    (r"颈痛|颈部|落枕|项痹", None, "颈椎病"),
    (r"颞颌|颞下颌|下颌关节", None, "颞下颌关节紊乱病"),
    (r"肩(?:关节)?(?:痛|疼痛|炎)|肩痹|肩部", None, "粘连性肩关节囊炎"),
    (r"肘", None, "肱骨外上髁炎"),
    (r"腕|手腕", r"扭|挫|外伤|跌|摔", "腕关节扭伤"),
    (r"腕|手腕", None, "腕骨关节炎"),
    (r"髋", None, "髋骨关节炎"),
    (r"膝", None, "膝骨关节炎"),
    (r"踝", r"扭|崴|外伤|跌|摔", "踝关节扭伤"),
    (r"踝", None, "踝骨关节炎"),
    (r"足跟|跟痛", None, "跖筋膜炎"),
    (r"滑膜炎", r"髋", "髋骨关节炎"),
    (r"滑膜炎", r"踝", "踝关节滑膜炎"),
    (r"滑膜炎", r"膝", "膝骨关节炎"),
    (r"滑膜炎", None, "膝骨关节炎"),
    (r"腰椎劳损|腰肌劳损|腰背", None, "腰背肌筋膜疼痛综合征"),
    (r"肌肉劳损|劳损|肌痉挛|筋膜", r"腰", "腰背肌筋膜疼痛综合征"),
    (r"肌肉劳损|劳损|肌痉挛|筋膜", r"颈|项", "颈部劳损"),
    (r"肌肉劳损|劳损|肌痉挛|筋膜", r"背", "胸背肌筋膜疼痛综合征"),
    (r"坐骨神经", None, "腰椎间盘突出症"),
    (r"椎间盘|椎体退|椎退行|椎退变", r"颈", "颈椎病"),
    (r"椎间盘|腰椎退行|腰椎退变|腰椎病", None, "腰椎间盘突出症"),
 
    (r"关节痛|关节炎|痛痹|痹证|痹症|行痹|着痹", r"多发|多关节|对称|晨僵|掌指|近端指间|腕关节", "类风湿关节炎"),
    (r"上腹痛|胃痛|胃脘", None, "功能性消化不良"),
    (r"腹痛", None, "功能性腹痛"),
    (r"胁肋痛|胁痛", None, "肋间神经痛"),
    (r"失眠|不寐", None, "非器质性失眠症"),
]
_SUPPORT_BAD = re.compile(r"排除|除外|X线|MRI|CT|超声|实验室|化验|检查|阳性率|符合\d|——|WBC|ESR|CRP|RF|HLA|抗体|\d+\s*/\s*ml")
_AGE_ITEM = re.compile(r"(?:年龄)?\s*(?:[≥>]|多在|多见于)?\s*(\d{2})\s*岁(?:以上|及以上)?")

_CLINICAL_FEATURES = [
    ("压痛", r"压痛|触痛"), ("挤压痛", r"挤压痛|侧方挤压"), ("肿胀", r"肿胀|肿大"),
    ("活动受限", r"活动受限|活动度受限|活动障碍"), ("Tinel征", r"Tinel"), ("Windlass试验", r"Windlass"),
    ("直腿抬高", r"直腿抬高"), ("反跳痛", r"反跳痛"), ("肌紧张", r"肌紧张"), ("叩击痛", r"叩击痛|叩痛"),
    ("红肿热痛", r"红肿热痛|红肿|皮温升高"), ("水疱", r"水疱|疱疹"), ("麻木", r"麻木|感觉减退"),
    ("放射痛", r"放射痛|放射至"), ("夜间痛", r"夜间痛|夜痛"), ("晨僵", r"晨僵"),
]
_NEGATED = re.compile(r"(?:无|未见|未触及|未引出|阴性|无明显|不明显|否认)[^，。；;]{0,16}")


def _dx_key_body(body: str) -> str:
    m = re.search(r"\*\*最简诊断要点\*\*[：:](.+?)(?:\n\*\*|$)", body or "", re.S)
    return re.sub(r"\*\*", "", m.group(1)).strip() if m else ""


def _criterion_clause(body: str, feature_pat: str) -> str:
    key = _dx_key_body(body)
    if not key:
        return ""
    for seg in re.split(r"[；;]", key):
        if re.search(feature_pat, seg, re.I):
            seg = re.sub(r"[（(][^）)]{0,60}[）)]", "", seg).strip(" ，,。；;")
            return _short(seg, 42)
    return ""


class DxKB:
    def __init__(self, kb_dir: str | Path):
        kb = Path(kb_dir)
        self.concepts = json.load(open(kb / "concepts.json", encoding="utf-8"))
        self.units = defaultdict(lambda: {"W_DD": [], "W_RED": [], "W_DX": []})
        for line in open(kb / "west.jsonl", encoding="utf-8"):
            d = json.loads(line)
            if d.get("section") in ("W_DD", "W_RED", "W_DX"):
                self.units[d["concept_id"]][d["section"]].append(d)
        self.names = defaultdict(list)
        for cid, c in self.concepts.items():
            if cid.startswith("AP-") or re.match(r"^GL\d+$", cid) or not (self.units[cid]["W_DD"] or self.units[cid]["W_RED"]):
                continue
            for n in [c["name"]] + list(c.get("aliases", [])):
                k = norm_key(n)
                if len(k) >= 2:
                    self.names[k].append(cid)
        self._keys = sorted(self.names, key=len, reverse=True)
        self.by_name = {}
        for cid, c in self.concepts.items():
            if cid in self.units and (self.units[cid]["W_DD"] or self.units[cid]["W_RED"] or self.units[cid]["W_DX"]):
                self.by_name.setdefault(c["name"], cid)

    def resolve(self, wm: str, case_text: str = "") -> tuple[str | None, str | None]:
    
        cid = self.concept_for(wm)
        if cid:
            return cid, None
        first = re.split(r"[、，,；;/]|\s+伴|合并", _PENDING.sub("", wm or "").strip())[0]
        for pat, ctx, name in SYMPTOM_PROXY:
            if re.search(pat, first) and (ctx is None or re.search(ctx, first + case_text)):
                c = self.by_name.get(name)
                if c:
                    return c, self.concepts[c]["name"]
        return None, None

    def _dx_body(self, cid: str) -> str:
        for u in self.units[cid]["W_DX"]:
            b = strip_codes(u["body"])
            if "最简诊断要点" in b or "首选检查" in b:
                return b
        return ""

    def first_exams(self, cid: str, max_n: int = 2) -> list[str]:
        
        m = re.search(r"\*\*首选检查\*\*[：:](.+?)(?:\n\*\*|$)", self._dx_body(cid), re.S)
        if not m:
            return []
        out = []
        for it in re.split(r"[；;]", m.group(1)):
            name = re.split(r"[：:]", it.strip(), 1)[0].strip(" 。")
            name = re.sub(r"\*+", "", name).strip()
            name = re.sub(r"[（(][^）)]*[）)]", "", name).strip()
            if 2 <= len(name) <= 16 and not re.search(r"者|均应|所有|必要时|如|若", name) and name not in out:
                out.append(name)
        return out[:max_n]

    def diagnostic_keypoint(self, cid: str, max_chars: int = 52) -> str:
        
        body = _dx_key_body(self._dx_body(cid))
        if not body:
            return ""
        parts = [x.strip(" ，,。；;") for x in re.split(r"[；;]", body) if x.strip()]
        if not parts:
            return ""
       
        parts.sort(key=lambda x: (bool(re.search(r"X线|MRI|CT|超声|实验室|化验", x)), len(x)))
        return _short(parts[0], max_chars)

    def missing_keypoints(self, cid: str, case_text: str, max_n: int = 2) -> list[str]:
       
        body = self._dx_body(cid)
        key = _dx_key_body(body)
        if not key:
            return []
        out = []
        for label, pat in _CLINICAL_FEATURES:
            if len(out) >= max_n:
                break
            if not re.search(pat, key, re.I):
                continue
           
            if re.search(pat, case_text or "", re.I):
                continue
            criterion = _criterion_clause(body, pat)
            if criterion and criterion not in out:
                out.append(criterion)
        return out[:max_n]

    def annotate_findings(self, wm: str, case_text: str, findings: list[str], max_n: int = 2) -> tuple[list[str], list[dict]]:
       
        cid, proxy = self.resolve(wm, case_text)
        if not cid:
            return findings, []
        anchor_name = proxy or self.concepts[cid]["name"]
        dx_body = self._dx_body(cid)
        key_body = _dx_key_body(dx_body)
        if not key_body:
            return findings, []
        first_exam = self.first_exams(cid, 1)
        exam_tail = (f"并结合{first_exam[0]}进一步鉴别" if first_exam else "并结合重复查体及必要检查进一步鉴别")
        out = list(findings)
        audit: list[dict] = []
        used_idx: set[int] = set()

        def _match(fr: str):
            for label, pat in _CLINICAL_FEATURES:
                if re.search(pat, fr, re.I) and re.search(pat, key_body, re.I):
                    criterion = _criterion_clause(dx_body, pat)
                    if criterion:
                        return label, pat, criterion
            return None

        # Pass 1: negative/discordant findings first.
        for i, fr in enumerate(findings):
            if len(audit) >= max_n:
                break
            if not re.search(r"无|未见|未触及|未引出|阴性|无明显|不明显|否认", fr):
                continue
            hit = _match(fr)
            if not hit:
                continue
            _, _, criterion = hit
            note = f"此体征与典型{anchor_name}的“{criterion}”不一致，需复核检查手法{exam_tail}"
            out[i] = fr.rstrip("。；;") + f"（{note}）"
            audit.append({"finding": fr, "diagnosis": anchor_name, "expected": criterion, "relation": "discordant", "annotation": note})
            used_idx.add(i)

        # Pass 2: if capacity remains, annotate concordant positive findings as support.
        for i, fr in enumerate(findings):
            if len(audit) >= max_n:
                break
            if i in used_idx or re.search(r"无|未见|未触及|未引出|阴性|无明显|不明显|否认", fr):
                continue
            hit = _match(fr)
            if not hit:
                continue
            _, _, criterion = hit
            note = f"此体征与{anchor_name}的典型表现“{criterion}”一致，可作为支持依据之一"
            out[i] = fr.rstrip("。；;") + f"（{note}）"
            audit.append({"finding": fr, "diagnosis": anchor_name, "expected": criterion, "relation": "supportive", "annotation": note})
            used_idx.add(i)

        return out, audit

    def support(self, cid: str, case_text: str, max_n: int = 4) -> list[str]:
        
        m = re.search(r"\*\*最简诊断要点\*\*[：:](.+?)(?:\n\*\*|$)", self._dx_body(cid), re.S)
        if not m:
            return []
        body = re.sub(r"\*\*", "", m.group(1).split("——")[0])
        age_m = re.search(r"年龄[:：]?\s*(\d{1,3})", case_text)
        age = int(age_m.group(1)) if age_m else None
        case_k = norm_key(case_text)
        qb = _bigrams(case_text)
        out = []
        for it in re.split(r"[；;]", body):
            if _SUPPORT_BAD.search(it) and not _AGE_ITEM.search(it):
                continue
            am = re.search(r"(?:年龄|多在|多见于|发病)\D{0,4}?[≥>]?\s*(\d{2})\s*岁(?:以上|及以上)", it) or re.search(r"年龄\s*[≥>]\s*(\d{2})", it)
            if am:
                if age is not None and age >= int(am.group(1)) and f"年龄{age}岁" not in out:
                    out.append(f"年龄{age}岁")
                continue
            for fr in re.split(r"[、，,/±（）()]|或|及|伴|和", it):
                fr = fr.strip(" 。：:")
                if len(fr) < 3 or len(fr) > 14 or _SUPPORT_BAD.search(fr) or re.search(r"\d|^(?:常用|可|多|常|需|应)|试验$|^阳性$", fr):
                    continue
                k = norm_key(fr)
                b = _bigrams(fr)
                if (k and k in case_k) or (len(b) >= 3 and len(b & qb) / len(b) >= 0.6):
                    if fr not in out:
                        out.append(fr)
                if len(out) >= max_n:
                    break
            if len(out) >= max_n:
                break
        return out if len(out) >= 2 else []

    def concept_for(self, wm: str) -> str | None:
        """按西医诊断名找概念：先整名/别名精确匹配，再找诊断里包含的最长病名（≥3 字）。"""
        first = re.split(r"[、，,；;/]|\s+伴|合并", _PENDING.sub("", wm or "").strip())[0]
        k = norm_key(first)
        if not k:
            return None
        if k in self.names:
            return self.names[k][0]
        for name in self._keys:
            if len(name) >= 3 and name in k:
                return self.names[name][0]
        return None

    def items(self, cid: str):
        dd, red = [], []
        for u in self.units[cid]["W_DD"]:
            for it in _ITEM_SPLIT.split(strip_codes(u["body"].replace("**", ""))):
                it = it.strip()
                m = re.match(r"([^：:\n]{2,24})[：:](.+)", it)
                if m:
                    dd.append((m.group(1).strip(" -"), m.group(2).strip(), u.get("source_id", ""), u.get("source_title", "")))
        for u in self.units[cid]["W_RED"]:
            for it in _ITEM_SPLIT.split(strip_codes(u["body"].replace("**", ""))):
                m = _RED_NAME.search(it)
                if m:
                    red.append((m.group(1).strip(), it.split("→")[0].strip(), u.get("source_id", ""), u.get("source_title", "")))
        return dd, red

    def note(self, wm: str, case_text: str, max_dd: int = 3, max_red: int = 2, max_chars: int = 280) -> tuple[str, dict]:
        cid, proxy = self.resolve(wm, case_text)
        if not cid:
            return "", {"concept": None}
        from tcm_study.schema import is_symptom_label
        if proxy and not is_symptom_label(wm):
            proxy = None                                   # "踝扭伤"→踝关节扭伤：是同一病，不当作鉴别对象
        dd, red = self.items(cid)
        from tcm_study.schema import pregnancy_text_applies
        self_names = {norm_key(x) for x in [self.concepts[cid]["name"], _PENDING.sub("", wm)] + list(self.concepts[cid].get("aliases", []))}
        keys = [k for _, k, _, _ in dd]
        def _clean_key(n, k):
            k = re.sub(r"[（(][^（）()]*[）)]", "", k)                      # 去掉要点里的嵌套括注
            k = k[len(n):].lstrip("：:，, ") if k.startswith(n) else k        # 要点开头重复病名时去掉
            return "" if (_BAD_KEY.search(k) or keys.count(k) > 1 or len(k) < 4) else k
        dd = [(n, _clean_key(n, k), sid, st) for n, k, sid, st in dd if norm_key(n) not in self_names]
        preg_ok = pregnancy_text_applies(case_text)
        red = [(n, t, sid, st) for n, t, sid, st in red if len(n) <= 14 and not _BAD_RED.search(n) and (preg_ok or not _PREG.search(n)) and norm_key(n) not in self_names]
        dd = [(n, k, sid, st) for n, k, sid, st in dd if preg_ok or not _PREG.search(n)]
        if not dd and not red and not proxy:
            return "", {"concept": cid, "empty": True}
        qb = _bigrams(case_text)
        
        def _aff_norm(s: str) -> str:
            # “急性/慢性/原发性/继发性”本身不代表同病种；否则“慢性胃炎”会因“慢性”错误贴近“慢性胆囊炎”。
            return re.sub(r"急性|慢性|亚急性|原发性|继发性|特发性|复发性", "", s or "")
        dx_text = " ".join(_aff_norm(x) for x in [wm, self.concepts[cid]["name"]] + list(self.concepts[cid].get("aliases", [])))
        dx_b = _bigrams(dx_text)
        def _dx_aff(name: str) -> float:
            nb = _bigrams(_aff_norm(name))
            return len(nb & dx_b) / max(1, len(nb)) if nb else 0.0
        group_aff: dict[str, float] = {}
        for n, _, sid, _ in dd:
            if sid:
                group_aff[sid] = max(group_aff.get(sid, 0.0), _dx_aff(n))
        best_group_aff = max(group_aff.values(), default=0.0)
        preferred_sources = {sid for sid, sc in group_aff.items() if best_group_aff >= 0.40 and sc >= max(0.25, best_group_aff - 0.20)}
        dd_rank_pool = [(i, x) for i, x in enumerate(dd) if not preferred_sources or x[2] in preferred_sources]
        if len(dd_rank_pool) < max_dd:
            dd_rank_pool = list(enumerate(dd))
        def _dd_score(item):
            i, (name, key, sid, st) = item
            dx = _dx_aff(name)
            case = _overlap(qb, name + key)
            source_bonus = 0.15 if sid in preferred_sources else 0.0
            return (-(4.0 * dx + 3.0 * case + source_bonus), i)
        dd_s = sorted(dd_rank_pool, key=_dd_score)[:max_dd]
        def _red_score(item):
            i, (name, trigger, sid, st) = item
            return (-(2.0 * _dx_aff(name) + 3.0 * _overlap(qb, name + trigger)), i)
        red_s = sorted(enumerate(red), key=_red_score)[:max_red]
        seen, parts_dd = set(), []
        if proxy:
            seen.add(proxy)
            parts_dd.append(f"{proxy}（关键区分点：{self.diagnostic_keypoint(cid) or '结合典型症状、体征及必要检查判断'}）")
            dd_s = dd_s[: max(1, max_dd - 1)]
        for _, (name, key, _, _) in dd_s:
            if name in seen:
                continue
            seen.add(name)
            parts_dd.append(f"{name}（关键区分点：{_short(key, 42)}）" if key else f"{name}（关键区分点：结合其特异症状、体征及必要检查判断）")
        parts_red = []
        for _, (name, trigger, _, _) in red_s:
            if name not in seen and not any(x.startswith(name + "（") or x == name for x in parts_red):
                clue = _short(re.sub(r"警惕【[^】]+】", "", trigger).strip(" -：:"), 32)
                parts_red.append(f"{name}（关键区分点：危险信号为{clue}）" if clue else f"{name}（关键区分点：结合相应危险信号及专科检查排除）")
        segs = []
        if parts_dd:
            segs.append("需鉴别：" + "；".join(parts_dd))
        if parts_red:
            segs.append("需优先排除：" + "；".join(parts_red))
        line = "；".join(segs)
        if len(line) > max_chars:
            line = line[:max_chars].rstrip("、；，（")
            line = re.sub(r"（[^）]*$", "", line)
        return (line + "。") if line else "", {
            "concept": cid,
            "concept_name": self.concepts[cid]["name"],
            "proxy": bool(proxy),
            "dd": parts_dd,
            "red": parts_red,
            "preferred_sources": sorted(preferred_sources),
        }

    def card(self, wm: str, case_text: str, max_dd: int = 3, max_red: int = 2, max_chars: int = 2200) -> tuple[str, dict]:
    
        line, info = self.note(wm, case_text, max_dd=max_dd, max_red=max_red, max_chars=min(800, max_chars))
        if not line:
            return "", info
        anchor = info.get("concept_name") or re.sub(r"（待查）|\(待查\)", "", wm or "").strip()
        card = (
            "【鉴别诊断卡】（按第一阶段西医诊断锚定 W_DD/W_RED；均为候选疾病，不等于已确诊，也不等于患者当前急危）\n"
            f"- 诊断锚点：{anchor}\n"
            f"- 候选：{line}\n"
            "- 使用规则：结合病例症状、体征和辅助检查，优先保留最相关的具体疾病名；每病必须保留括号内“关键区分点：……”并在必要时写所需检查。"
            "W_RED 中的严重疾病只能写‘需优先排除’，除非【病例安全核查】同时命中相应当前急危征象，否则不得写成【急危重症警示】。"
        )
        return card[: max(400, int(max_chars))], info
