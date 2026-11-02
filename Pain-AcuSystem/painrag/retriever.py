# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import re
import json
import hashlib
from collections import defaultdict, Counter

from . import config as C
from .bm25 import BM25
from .textutil import norm_key, val
from . import findings as F

GENERIC = {"疼痛", "痛", "痛证", "痛症", "痹证", "痹病", "痹", "综合征", "损伤", "炎", "疾病", "病"}

# 体区词典：用于「病例主诉的部位」与「概念名称的部位」冲突惩罚 / 一致加分
REGIONS = {
    "头": ["头", "颅", "偏头", "额", "颞", "枕"], "面口": ["面", "颜面", "口", "舌", "牙", "颌", "三叉", "咽", "喉", "耳", "鼻", "眼", "目"],
    "颈": ["颈", "项", "落枕"], "肩": ["肩"], "肘臂": ["肘", "臂", "肱骨", "上肢"], "腕手": ["腕", "手", "指", "掌", "腱鞘"],
    "胸": ["胸", "肋", "乳", "心"], "背": ["背", "胸椎"], "腰": ["腰"], "骶臀髋": ["骶", "臀", "髋", "股", "梨状肌", "坐骨"],
    "膝": ["膝", "髌", "半月板"], "踝足": ["踝", "足", "跟", "跖", "趾"], "腹胃": ["腹", "胃", "脘", "肠", "胆", "肝", "胰"],
    "盆会阴": ["盆", "会阴", "阴", "膀胱", "尿", "经", "痛经", "月经", "宫"], "全身": ["全身", "多关节", "纤维肌痛", "风湿"],
}
_REGION_OF = {w: r for r, ws in REGIONS.items() for w in ws}


def regions_of(text: str) -> set:
    t = text or ""
    out = set()
    for w, r in _REGION_OF.items():
        if w in t:
            out.add(r)
    return out

DEFAULT_PLAN = {
    # 板块 → 每概念最多取几条；顺序即优先级
    "west": [("W_DX", 2), ("W_SX", 1), ("W_RED", 1), ("W_DD", 2), ("W_PE", 1), ("W_STAGE", 1), ("W_DEF", 1), ("W_BOOK", 1), ("W_TX", 0)],
    "tcm": [("T_NAME", 1), ("T_SYN", 3), ("T_PATH", 1), ("T_ACU", 4), ("T_STAGE", 1), ("T_BOUND", 1), ("T_CAUTION", 2), ("T_DD", 1), ("T_HERB", 0)],
}
DEFAULT_BUDGET = {"west": 2200, "tcm": 3200, "acupoint": 900}
SEC_LABEL = {"W_DEF": "定义与流行病学", "W_SX": "疼痛特点与症状体征", "W_DX": "诊断要点与辅助检查", "W_PE": "体格检查试验",
             "W_RED": "危险信号", "W_DD": "鉴别诊断", "W_STAGE": "病程分期", "W_TX": "西医治疗与预后", "W_BOOK": "书籍原文",
             "T_NAME": "中医病名与诊断", "T_PATH": "病因病机", "T_SYN": "辨证分型", "T_HERB": "中医内治", "T_ACU": "针灸与外治",
             "T_STAGE": "治疗分期", "T_CAUTION": "禁忌与护理", "T_DD": "中医鉴别", "T_BOUND": "针灸适应证与安全边界"}


def _src(title: str) -> str:
    return re.sub(r"\s*\[[^\]]*\]", "", str(title or "")).strip() or "来源"
_LIB_CODE = r"(?:GL\d{3}|[DGB]\d{4}|BK\d)"
_CODE_PATTERNS = [
    re.compile(r"[（(]\s*(?:见|参见|详见)\s*" + _LIB_CODE + r"(?:\s*[/、，,]\s*" + _LIB_CODE + r")*\s*[)）]"),
    re.compile(r"[\[［]\s*" + _LIB_CODE + r"(?:\s*[/、，,]\s*" + _LIB_CODE + r")*\s*[\]］]"),
    re.compile(r"(?<![A-Za-z0-9])" + _LIB_CODE + r"(?:\s*/\s*" + _LIB_CODE + r")*(?![A-Za-z0-9型号])"),
]
_CODE_CLEANUP = [
    (re.compile(r"[（(]\s*[：:/、，,·]+\s*"), "（"),      # （：针刺 → （针刺
    (re.compile(r"\s*[：:/、，,·]+\s*[)）]"), "）"),      # 第21章、） → 第21章）
    (re.compile(r"([（(《「])\s+"), r"\1"),                # （ 循证针灸… → （循证针灸…
    (re.compile(r"\s+([)）》」])"), r"\1"),
    (re.compile(r"[（(]\s*[)）]|《\s*》"), ""),            # （） 《》
    (re.compile(r"[ \t]{2,}"), " "),
]


def strip_codes(text: str) -> str:
   
    t = str(text or "")
    for p in _CODE_PATTERNS:
        t = p.sub("", t)
    for p, rep in _CODE_CLEANUP:
        t = p.sub(rep, t)
    return t


def _bigrams(text: str, limit: int = 1500) -> set:
    t = norm_key(text)[:limit]
    return {t[i:i + 2] for i in range(len(t) - 1)}


def _lex(qb: set, text: str) -> float:
    b = _bigrams(text, 900)
    if not qb or not b:
        return 0.0
    return min(1.0, 3.0 * 2 * len(qb & b) / (len(qb) + len(b)))


def _rec_bonus(u: dict) -> float:
    rec, ev = val(u.get("recommendation")), val(u.get("evidence_level")).upper()
    s = 0.0
    if "强推荐" in rec or rec.startswith("1"):
        s += 0.15
    elif "弱推荐" in rec or "建议" in rec or rec.startswith("2"):
        s += 0.05
    if ev.startswith("A") or "高" in ev:
        s += 0.10
    elif ev.startswith("B") or "中" in ev:
        s += 0.05
    if u.get("point_based"):
        s += 0.10
    return s


class PainRAG:
    def __init__(self, kb_dir: str = None, use_dense: str = "auto", device: str = None, verbose: bool = True):
        self.kb = kb_dir or C.KB_DIR
        self.docs, self.bm25, self.vec = {}, {}, {}
        for cat in C.COLLECTIONS:
            p = os.path.join(self.kb, f"{cat}.jsonl")
            if not os.path.exists(p):
                continue
            self.docs[cat] = [json.loads(l) for l in open(p, encoding="utf-8")]
            pb = os.path.join(self.kb, f"{cat}.bm25")
            if os.path.exists(pb):
                self.bm25[cat] = BM25.load(pb)
            pv = os.path.join(self.kb, f"{cat}.npy")
            if use_dense != "off" and os.path.exists(pv):
                import numpy as np
                self.vec[cat] = np.load(pv)
        self.encoder = None
        if self.vec:
            try:
                from sentence_transformers import SentenceTransformer
                model = open(os.path.join(self.kb, "embedding_model.txt"), encoding="utf-8").read().strip()
                self.encoder = SentenceTransformer(model, device=device or C.EMBED_DEVICE)
            except Exception as e:
                if use_dense == "on":
                    raise
                if verbose:
                    print(f"[PainRAG] 稠密向量不可用（{e}），退回 BM25")
                self.vec = {}
        self.concepts = json.load(open(os.path.join(self.kb, "concepts.json"), encoding="utf-8"))
        # 概念 → 单元索引
        self.by_concept = {cat: defaultdict(list) for cat in self.docs}
        for cat, ds in self.docs.items():
            for i, d in enumerate(ds):
                self.by_concept[cat][d["concept_id"]].append(i)
        # 名称词典：归一化名 -> [(cid, 权重)]
        self.names = defaultdict(list)
        for cid, c in self.concepts.items():
            if cid.startswith("AP-") or re.match(r"^GL\d+$", cid):
                continue
            for n, w in [(c["name"], 1.0)] + [(a, 1.0) for a in c.get("aliases", [])] + [(t, 0.7) for t in c.get("tcm_names", [])]:
                k = norm_key(n)
                if len(k) >= 2 and k not in GENERIC:
                    self.names[k].append((cid, w))
        self._name_list = sorted(self.names, key=len, reverse=True)
        self._creg = {cid: regions_of(" ".join([c["name"]] + c.get("aliases", [])[:6])) for cid, c in self.concepts.items()}
        # 穴位查找
        self.points = {}
        for d in self.docs.get("acupoint", []):
            self.points[d["point"]] = d
            for a in d.get("aliases", []):
                self.points.setdefault(a, d)
        # 手法卡：按名称查找，不入索引；文件缺失则为空
        self.manips, self._manip_alias = {}, {}
        pm = os.path.join(self.kb, "manipulation_cards.json")
        if os.path.exists(pm):
            for card in json.load(open(pm, encoding="utf-8")).get("cards", []):
                self.manips[card["name"]] = card
                for a in [card["name"]] + list(card.get("aliases") or []):
                    self._manip_alias[a] = card["name"]
        self._manip_names = sorted(self._manip_alias, key=len, reverse=True)
        self._qcache = {}
        if verbose:
            print(f"[PainRAG] 集合 {{{', '.join(f'{k}:{len(v)}' for k, v in self.docs.items())}}}｜稠密向量 {'开' if self.vec else '关'}")

    # ------------------------------------------------------------ 基础检索
    def _qvec(self, q: str):
        if q not in self._qcache:
            self._qcache[q] = self.encoder.encode([q], normalize_embeddings=True).astype("float32")[0]
            if len(self._qcache) > 512:
                self._qcache.pop(next(iter(self._qcache)))
        return self._qcache[q]

    def search(self, query: str, cat: str = "west", k: int = 8, filt=None) -> list:
       
        if cat not in self.docs:
            return []
        n = max(k * 20, 300) if filt else k * 2
        ranks = defaultdict(float)
        if cat in self.bm25:
            for r, (s, i) in enumerate(self.bm25[cat].search(query, n)):
                ranks[i] += 1.0 / (60 + r)
        if cat in self.vec and self.encoder is not None:
            import numpy as np
            sims = self.vec[cat] @ self._qvec(query)
            top = np.argpartition(-sims, min(n, len(sims) - 1))[:n]
            for r, i in enumerate(sorted(top, key=lambda j: -sims[j])):
                ranks[int(i)] += 1.0 / (60 + r)
        out = []
        for i, s in sorted(ranks.items(), key=lambda x: -x[1]):
            d = self.docs[cat][i]
            if filt and not filt(d):
                continue
            out.append((round(s * 60, 4), d))
            if len(out) >= k:
                break
        return out

    def sim(self, query: str, cat: str, idxs: list) -> dict:
        
        qb = _bigrams(query)
        out = {}
        if cat in self.vec and self.encoder is not None:
            qv = self._qvec(query)
            for i in idxs:
                out[i] = 0.7 * float(self.vec[cat][i] @ qv) + 0.3 * _lex(qb, self.docs[cat][i]["body"])
        else:
            bs = self.bm25[cat].scores(query, only=set(idxs)) if cat in self.bm25 else {}
            mx = max(bs.values()) if bs else 1.0
            for i in idxs:
                out[i] = 0.5 * (bs.get(i, 0.0) / mx if mx else 0.0) + 0.5 * _lex(qb, self.docs[cat][i]["body"])
        return out

    # ------------------------------------------------------------ 概念路由
    def route(self, case_text: str, k: int = 3, findings: dict = None) -> list:
       
        q = case_text.strip()
        nk = norm_key(q)
        st = F.usable(findings)
        aux_k = hx_k = neg_k = ""
        q2 = q
        
        chief = " ".join(re.findall(r"【(?:主诉|诊断|西医诊断|中医诊断)】([^\n【]{0,80})", q)) + " " + \
            " ".join(re.findall(r"诊断为[“\"]?([^\n，。；”\"]{2,30})", q))
        exam = " ".join(re.findall(r"【(?:查体|体格检查)】([^\n【]{0,200})", q))
        hpi = " ".join(re.findall(r"【现病史】([^\n【]{0,160})", q))
        focus = norm_key(chief)
        fq = (chief + " " + hpi + " " + exam).strip() or q[:300]
        case_reg = regions_of(chief) or regions_of(q[:200])
        if st:
            wq, tq = F.compose_queries(findings)
            tx = F.texts(findings)
            nk = norm_key(tx["pos"] + "；" + wq)
            focus = norm_key(tx["chief"] + tx["aux"])
            aux_k, hx_k, neg_k = norm_key(tx["aux"]), norm_key(tx["hx"]), norm_key(tx["neg"])
            fq, q2 = wq or fq, tq or wq or q
            case_reg = regions_of(tx["chief"]) or regions_of(wq[:60])
        lex, reasons = defaultdict(float), defaultdict(list)
        covered = []
        for name in self._name_list:
            pos = nk.find(name)
            if pos < 0:
                continue
            if any(a <= pos and pos + len(name) <= b for a, b in covered):
                continue
            covered.append((pos, pos + len(name)))
            base = 0.4 if len(name) == 2 else (0.7 if len(name) == 3 else 1.0)
            bonus = 0.3 if (focus and name in focus) else 0.0
            if st and len(name) >= 3:
                bonus = max(bonus, 0.3 if name in aux_k else (0.15 if name in hx_k else 0.0))
            for cid, w in self.names[name]:
                s = base * w + bonus
                if s > lex[cid]:
                    lex[cid] = s
                reasons[cid].append(f"词典命中「{name}」")
        
        cr = {}
        for qq, w in ((fq, 1.0), (q2, 0.6)):
            for r, (s, d) in enumerate(self.search(qq, "concept", 10)):
                v = w * (1.0 - r / 10.0)
                if v > cr.get(d["concept_id"], 0.0):
                    cr[d["concept_id"]] = v
                reasons[d["concept_id"]].append("概念表检索")
       
        wr = {}
        hits = self.search(fq, "west", 15, filt=lambda d: d["section"] in ("W_SX", "W_DX", "W_DEF", "W_BOOK"))
        for r, (s, d) in enumerate(hits):
            wr.setdefault(d["concept_id"], 1.0 - r / 15.0)
            reasons[d["concept_id"]].append("西医单元检索")
        tr = {}
        for r, (s, d) in enumerate(self.search(q2, "tcm", 10, filt=lambda d: d["section"] in ("T_NAME", "T_SYN"))):
            tr.setdefault(d["concept_id"], 1.0 - r / 10.0)
            reasons[d["concept_id"]].append("中医单元检索")
        cands = set(lex) | set(cr) | set(wr) | set(tr)
        negated = set()
        if st and neg_k:
            for name in self._name_list:
                if len(name) >= 3 and name in neg_k and name not in nk:
                    negated.update(cid for cid, _ in self.names[name])
        scored = []
        for cid in cands:
            if cid not in self.concepts:
                continue
            s = 0.45 * min(1.0, lex.get(cid, 0.0)) + 0.25 * cr.get(cid, 0.0) + 0.2 * wr.get(cid, 0.0) + 0.1 * tr.get(cid, 0.0)
            creg = self._creg.get(cid, set())
            if case_reg and creg:
                if case_reg & creg:
                    s += 0.15
                    reasons[cid].append("部位一致")
                else:
                    s *= 0.3
                    reasons[cid].append("部位冲突")
            if cid in negated:
                s *= 0.3
                reasons[cid].append("阴性结果排除")
            scored.append((cid, round(s, 4), sorted(set(reasons[cid]))))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    # ------------------------------------------------------------ 证据包
    def evidence(self, case_text: str, top_concepts: int = 2, budget: dict = None, plan: dict = None,
                 domains=("west", "tcm", "acupoint"), min_score: float = 0.3, relative_keep: float = 0.55,
                 max_points: int = 8, strip_md: bool = True, findings: dict = None) -> tuple:
        budget = dict(DEFAULT_BUDGET, **(budget or {}))
        plan = plan or DEFAULT_PLAN
        st = F.usable(findings)
        wq, tq = F.compose_queries(findings) if st else ("", "")
        qdom = {"west": wq or case_text, "tcm": tq or wq or case_text}
        routed = self.route(case_text, k=max(3, top_concepts), findings=findings if st else None)
        chosen = [r for r in routed if r[1] >= min_score]
        if chosen:
            chosen = [r for r in chosen if r[1] >= relative_keep * chosen[0][1]][:top_concepts]
        trace = {"query_mode": "structured" if st else "fulltext", "queries": {"west": wq, "tcm": tq} if st else None,
                 "concepts": [{"cid": c, "name": self.concepts[c]["name"], "score": s, "reasons": rs} for c, s, rs in chosen],
                 "routed_all": [(c, s) for c, s, _ in routed], "units": {"west": [], "tcm": [], "acupoint": []}}
        if not chosen:
            trace["reason"] = "no_concept_above_threshold"
            return "", trace
        blocks, used_gls, chosen_acu, chosen_bound = [], [], [], []
        for n, (cid, score, _) in enumerate(chosen, 1):
            c = self.concepts[cid]
            for dom in ("west", "tcm"):
                if dom not in domains or dom not in self.docs:
                    continue
                idxs = self.by_concept[dom].get(cid, [])
                if not idxs:
                    continue
                sims = self.sim(qdom[dom], dom, idxs)
                per = budget[dom] // len(chosen)
                lines, used = [], 0
                has_card = any(self.docs[dom][i]["source_type"] == "card" for i in idxs)
                for sec, cap in plan[dom]:
                    if cap <= 0:
                        continue
                    if sec == "W_BOOK" and has_card:
                        continue
                    if sec == "W_DEF" and any(self.docs[dom][i]["section"] == "W_DX" for i in idxs):
                        continue
                    cand = [i for i in idxs if self.docs[dom][i]["section"] == sec]
                    if sec == "T_ACU":
                        cand.sort(key=lambda i: -(sims[i] + _rec_bonus(self.docs[dom][i])))
                    else:
                        cand.sort(key=lambda i: -sims[i])
                    seen_sig, taken = set(), 0
                    for i in cand:
                        d = self.docs[dom][i]
                        if sec == "T_ACU":
                            sig = (d.get("therapy_type"), tuple(d.get("points") or [])[:6], d.get("syndrome"))
                            if sig in seen_sig:
                                continue
                            seen_sig.add(sig)
                        body = d["body"].replace("**", "") if strip_md else d["body"]
                        line = f"▪ {SEC_LABEL.get(sec, sec)}（{_src(d['source_title'])}{('·' + d['sub']) if d.get('sub') else ''}）：{body}"
                        if used + len(line) > per and taken > 0:
                            break
                        if used + len(line) > per:
                            line = line[:max(0, per - used)]
                        lines.append(line)
                        used += len(line)
                        taken += 1
                        trace["units"][dom].append(d["id"])
                        if sec == "T_ACU":
                            chosen_acu.append(d)
                            if d["source_id"] not in used_gls:
                                used_gls.append(d["source_id"])
                        elif sec == "T_BOUND" and d.get("points"):
                            chosen_bound.append(d)
                        if taken >= cap:
                            break
                    if used >= per:
                        break
                if dom == "tcm" and used < per:
                   
                    for gl in used_gls[:2]:
                        for i in self.by_concept["tcm"].get(gl, [])[:1]:
                            d = self.docs["tcm"][i]
                            body = d["body"].replace("**", "") if strip_md else d["body"]
                            line = f"▪ 指南通用注意（{_src(d['source_title'])}）：{body}"
                            if used + len(line) <= per + 200:
                                lines.append(line)
                                used += len(line)
                                trace["units"]["tcm"].append(d["id"])
                if lines:
                    if dom == "west":
                       
                        head = f"【西医证据·{n}】{c['name']}"
                    else:
                        head = f"【中医证据·{n}】{c['name']}" + (f"｜中医病名：{'、'.join(c['tcm_names'][:4])}" if c.get("tcm_names") else "")
                    blocks.append(head + "\n" + "\n".join(lines))
       
        if "tcm" in domains and not chosen_acu and "tcm" in self.docs:
            chief = " ".join(re.findall(r"【(?:主诉|现病史|查体)】([^\n【]{0,120})", case_text)) or case_text[:300]
            case_reg = regions_of(" ".join(re.findall(r"【主诉】([^\n【]{0,80})", case_text))) or regions_of(case_text[:200])
            if st:
                chief, case_reg = (tq or wq), (regions_of(F.texts(findings)["chief"]) or case_reg)
            lines, used, seen_sig = [], 0, set()
            ok = lambda d: d["section"] == "T_ACU" and d.get("point_based") and (not case_reg or (self._creg.get(d["concept_id"], set()) & case_reg))
            for s, d in self.search(chief, "tcm", 3, filt=ok):
                sig = (d.get("therapy_type"), tuple(d.get("points") or [])[:6])
                if sig in seen_sig:
                    continue
                seen_sig.add(sig)
                body = d["body"].replace("**", "") if strip_md else d["body"]
                line = f"▪ {d['concept_name']}·针灸与外治（{_src(d['source_title'])}{('·' + d['sub']) if d.get('sub') else ''}）：{body}"
                if used + len(line) > budget["tcm"] // 2:
                    break
                lines.append(line)
                used += len(line)
                chosen_acu.append(d)
                trace["units"]["tcm"].append(d["id"])
            if lines:
                blocks.append("【针灸方案参考（所选疾病无指南针灸条目，以下按症状检索，非本病专属，须结合病例取舍）】\n" + "\n".join(lines))
        # 穴位补齐
        if "acupoint" in domains and (chosen_acu or chosen_bound) and self.points:
            cnt = Counter()
            for rank, d in enumerate(chosen_acu):
                for j, p in enumerate(d.get("points") or []):
                    cnt[p] += (2.0 if j < 4 else 1.0) / (1 + 0.3 * rank)
            for d in chosen_bound:                      # 卡片「适用范围」里的常用穴
                for j, p in enumerate(d.get("points") or []):
                    cnt[p] += 1.5 if j < 4 else 0.8
            lines, used = [], 0
            for p, _ in cnt.most_common(max_points):
                d = self.points.get(p)
                if not d:
                    continue
                line = "- " + d.get("short", d["body"])
                if used + len(line) > budget["acupoint"]:
                    break
                lines.append(line)
                used += len(line)
                trace["units"]["acupoint"].append(d["point"])
            if lines:
                blocks.append("【穴位参考（标准穴位库：定位/深度/安全）】\n" + "\n".join(lines))
        text = strip_codes("\n\n".join(blocks))
        trace["chars"] = len(text)
        trace["sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return text, trace

    # ------------------------------------------------------------ 第二步：穴位安全卡 / 手法卡
    def point_cards(self, points: list, max_points: int = 12, max_chars: int = 1800) -> tuple:
        """给定穴名（规范名或别名，顺序即优先级），返回穴位库里对应卡片的文本块与所用穴名。库里没有的名字跳过。"""
        lines, used, taken, seen = [], 0, [], set()
        for p in points:
            name = re.sub(r"[（(].*?[)）]|穴$", "", str(p)).strip()
            d = self.points.get(name) or (self.points.get(name[:-1]) if name.endswith("穴") else None)
            if not d or d["point"] in seen:
                continue
            line = "- " + d.get("short", d["body"])
            if used + len(line) > max_chars or len(taken) >= max_points:
                break
            seen.add(d["point"])
            lines.append(line)
            used += len(line)
            taken.append(d["point"])
        return ("【穴位安全卡】（所列穴位在标准穴位库中的定位、刺法与深度上限、禁忌）\n" + "\n".join(lines)) if lines else "", taken

    def manip_names_in(self, text: str) -> list:
        """文本里出现的手法卡名（去重、按出现位置排序）。"""
        hits = {}
        for a in self._manip_names:
            i = (text or "").find(a)
            if i >= 0:
                hits.setdefault(self._manip_alias[a], i)
        return [k for k, _ in sorted(hits.items(), key=lambda kv: kv[1])]

    def manip_cards(self, names: list, max_cards: int = 4, max_chars: int = 1200) -> tuple:
        lines, used, taken = [], 0, []
        for n in names:
            c = self.manips.get(self._manip_alias.get(n, n))
            if not c or c["name"] in taken:
                continue
            line = f"- {c['name']}（{c['category']}）：{c['要领']}｜适应：{c['适应']}｜注意：{c['注意']}"
            if used + len(line) > max_chars or len(taken) >= max_cards:
                break
            lines.append(line)
            used += len(line)
            taken.append(c["name"])
        return ("【手法卡】（所用针刺手法的操作要领与注意）\n" + "\n".join(lines)) if lines else "", taken

    # ------------------------------------------------------------ 便捷
    def concept_units(self, cid: str, dom: str = None) -> list:
        out = []
        for cat in (self.docs if dom is None else [dom]):
            out += [self.docs[cat][i] for i in self.by_concept.get(cat, {}).get(cid, [])]
        return out
