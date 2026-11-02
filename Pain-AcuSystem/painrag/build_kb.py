# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import re
import json
from collections import defaultdict, Counter

from . import config as C
from .textutil import val, one_line, norm_key, split_text, pack_fields, safe_name
from . import sources as S
from .acupoints import AcuLexicon

SECTIONS = {
    "W_DEF": ("west", "定义与流行病学"), "W_SX": ("west", "疼痛特点与症状体征"), "W_DX": ("west", "诊断要点与辅助检查"),
    "W_PE": ("west", "体格检查试验"), "W_RED": ("west", "危险信号"), "W_DD": ("west", "鉴别诊断"),
    "W_STAGE": ("west", "病程分期"), "W_TX": ("west", "西医治疗与预后"), "W_BOOK": ("west", "书籍原文"),
    "T_NAME": ("tcm", "中医病名与诊断"), "T_PATH": ("tcm", "病因病机"), "T_SYN": ("tcm", "辨证分型"),
    "T_HERB": ("tcm", "中医内治"), "T_ACU": ("tcm", "针灸与外治"), "T_STAGE": ("tcm", "治疗分期"),
    "T_CAUTION": ("tcm", "禁忌与护理"), "T_DD": ("tcm", "中医鉴别"), "T_BOUND": ("tcm", "针灸适应证与安全边界"),
    "AP": ("acupoint", "穴位"), "CONCEPT": ("concept", "疾病概念"),
}
DOMAIN_LABEL = {"west": "西医", "tcm": "中医", "acupoint": "穴位", "concept": "概念"}
MD_ORDER_W = ["W_DEF", "W_SX", "W_DX", "W_PE", "W_RED", "W_DD", "W_STAGE", "W_TX", "W_BOOK"]
MD_ORDER_T = ["T_NAME", "T_PATH", "T_SYN", "T_STAGE", "T_ACU", "T_HERB", "T_BOUND", "T_CAUTION", "T_DD"]
POINT_THERAPY = re.compile(r"针|灸|刺|埋线|穴位|电针|拔罐|放血|刮痧")


def j(items, sep="；"):
    return sep.join(x for x in (one_line(i) for i in (items or [])) if x)


class KB:
    def __init__(self, lex: AcuLexicon):
        self.units, self._seen, self.lex = [], set(), lex
        self._n = Counter()
        self.concepts = {}          # cid -> 概念路由信息

    # ------------------------------------------------------------ 概念登记
    def concept(self, cid, name, aliases=(), icd="", icd_title="", tcm_names=(), west_names=(), gls=(), has_card=False,
                has_book=False, definition="", symptoms=""):
        c = self.concepts.setdefault(cid, dict(cid=cid, name=name, aliases=[], icd="", icd_title="", tcm_names=[],
                                               west_names=[], guidelines=[], has_card=False, has_book=False,
                                               definition="", symptoms=""))
        c["name"] = c["name"] or name
        for k, vals in (("aliases", aliases), ("tcm_names", tcm_names), ("west_names", west_names), ("guidelines", gls)):
            for v in vals:
                v = one_line(v)
                if v and v != c["name"] and v not in c[k]:
                    c[k].append(v)
        c["icd"], c["icd_title"] = c["icd"] or icd, c["icd_title"] or icd_title
        c["has_card"], c["has_book"] = c["has_card"] or has_card, c["has_book"] or has_book
        c["definition"], c["symptoms"] = c["definition"] or one_line(definition), c["symptoms"] or one_line(symptoms)
        return c

    # ------------------------------------------------------------ 加单元
    def add(self, cid, section, source_type, source_id, source_title, body, meta=None, split=True, overlap=0, sub=""):
        body = val(body)
        if not body:
            return 0
        dom, slab = SECTIONS[section]
        c = self.concepts.get(cid) or {"name": cid, "aliases": [], "tcm_names": [], "icd": ""}
        parts = split_text(body, C.MAX_CHARS, C.MIN_TAIL, overlap) if split else [body]
        n = 0
        for i, p in enumerate(parts, 1):
            key = (cid, section, source_id, norm_key(p)[:160])
            if key in self._seen:
                continue
            self._seen.add(key)
            names = c["tcm_names"] if dom == "tcm" else c["aliases"]
            head = (f"【{DOMAIN_LABEL[dom]}·{slab}】" if dom in ("west", "tcm") else f"【{slab}】") + c["name"]
            if names:
                head += f"（{'中医病名' if dom == 'tcm' else '别名'}：{'、'.join(names[:5])}）"
            if c.get("icd") and dom == "west":
                head += f"｜ICD-11 {c['icd']}"
            head += f"｜来源：{source_title}" + (f"·{sub}" if sub else "") + (f"（{i}/{len(parts)}）" if len(parts) > 1 else "")
            self._n[(cid, section, source_id)] += 1
            uid = f"{cid}|{section}|{source_id}|{self._n[(cid, section, source_id)]}"
            u = dict(id=uid, domain=dom, section=section, section_label=slab, concept_id=cid, concept_name=c["name"],
                     aliases=c["aliases"][:8], tcm_names=c["tcm_names"][:6], icd11=c.get("icd", ""),
                     source_type=source_type, source_id=source_id, source_title=source_title, sub=sub,
                     header=head, body=p, text=head + "\n" + p, chars=len(head) + 1 + len(p))
            if meta:
                u.update(meta)
            self.units.append(u)
            n += 1
        return n


# ==================================================================== 卡片 → 单元
def add_card(kb: KB, card: dict):
    cid = card["concept_id"]
    src = "书籍标注卡片" + ("[" + "/".join(sorted({re.match(r"(BK\d|中国疼痛病诊疗规范)", s.get("book", "")).group(1)
                                                   for s in card.get("sources", []) if re.match(r"(BK\d|中国疼痛病诊疗规范)", s.get("book", ""))})) + "]"
                        if card.get("sources") else "")
    # 定义与流行病学
    f = [("定义", card.get("definition")), ("流行病学与诱因", card.get("epidemiology"))]
    if card.get("staging"):
        st = card["staging"]
        f.append(("分型分期", "；".join(f"{s.get('stage','')}：{s.get('features','')}" if isinstance(s, dict) else str(s) for s in st)))
    for b in pack_fields(f, C.MAX_CHARS):
        kb.add(cid, "W_DEF", "card", "card", src, b, split=False)
    # 疼痛特点与症状体征
    pp = card.get("pain_profile") or {}
    lab = [("site", "部位"), ("quality", "性质"), ("radiation", "放射"), ("aggravating", "加重"), ("relieving", "缓解"), ("temporal", "时间特点")]
    f = [("疼痛特点", "；".join(f"{l}：{one_line(pp[k])}" for k, l in lab if pp.get(k))),
         ("主要症状", j(card.get("key_symptoms"))), ("主要体征", j(card.get("key_signs")))]
    if card.get("subtypes"):
        f.append(("亚型", "；".join(f"{s.get('name','')}（{s.get('site','')}）：{s.get('key','')}" for s in card["subtypes"] if isinstance(s, dict))))
    for b in pack_fields(f, C.MAX_CHARS):
        kb.add(cid, "W_SX", "card", "card", src, b, split=False)
    # 诊断要点与辅助检查
    w = card.get("workup") or {}
    f = [("最简诊断要点", j(card.get("minimal_criteria"))), ("首选检查", j(w.get("first_line"))),
         ("必要时检查", j(w.get("when_needed"))), ("实验室特点", w.get("lab_typical"))]
    for b in pack_fields(f, C.MAX_CHARS):
        kb.add(cid, "W_DX", "card", "card", src, b, split=False)
    # 体格检查试验（每块 ≤ 6 项）
    tests = [t for t in card.get("physical_tests") or [] if isinstance(t, dict) and t.get("test")]
    for i in range(0, len(tests), 6):
        lines = [f"- {t['test']}：{one_line(t.get('how',''))}；阳性：{one_line(t.get('positive',''))}" + (f"（{t['source']}）" if t.get("source") else "")
                 for t in tests[i:i + 6]]
        kb.add(cid, "W_PE", "card", "card", src, "\n".join(lines), split=True)
    # 危险信号
    rf = [f"- 若{one_line(r.get('finding',''))} → 警惕【{one_line(r.get('suspect',''))}】：{one_line(r.get('action',''))}"
          for r in card.get("red_flags") or [] if isinstance(r, dict)]
    if rf:
        kb.add(cid, "W_RED", "card", "card", src, "\n".join(rf))
    # 鉴别诊断（每块 ≤ 5 项）
    dd = [d for d in card.get("differentials") or [] if isinstance(d, dict) and d.get("disease")]
    for i in range(0, len(dd), 5):
        lines = [f"- {one_line(d['disease'])}" + (f"[{d['concept_id']}]" if d.get("concept_id") else "") + f"：{one_line(d.get('points',''))}" for d in dd[i:i + 5]]
        kb.add(cid, "W_DD", "card", "card", src, "\n".join(lines))
    # 西医治疗与预后
    f = [("西医治疗概要", card.get("western_treatment_summary")), ("常见误区", j(card.get("pitfalls"))), ("预后", card.get("prognosis"))]
    for b in pack_fields(f, C.MAX_CHARS):
        kb.add(cid, "W_TX", "card", "card", src, b, split=False)
    # 针灸适应证与安全边界 → 中医库
    ab = card.get("acupuncture_boundary") or {}
    f = [("适用范围", ab.get("suitable")), ("注意", j(ab.get("caution"))), ("不以针灸为首选/须先排除", j(ab.get("not_first_line")))]
    pts = kb.lex.extract(one_line(ab.get("suitable")))          # 卡片给出的常用穴，供穴位补齐
    for b in pack_fields(f, C.MAX_CHARS):
        kb.add(cid, "T_BOUND", "card", "card", src, b, meta=dict(points=pts), split=False)


# ==================================================================== 指南 → 单元
F3 = [("diagnostic_criteria", "诊断标准"), ("clinical_manifestation", "临床表现"), ("physical_exam", "体格检查"),
      ("auxiliary_exam", "辅助检查"), ("lab_test", "实验室检查"), ("diagnostic_points", "诊断要点"), ("subtype", "分型"),
      ("staging", "分期"), ("classification_type", "分级体系"), ("classification_detail", "分级内容"), ("severity", "严重程度"),
      ("risk_stratification", "危险分层"), ("outcome_measure", "疗效评价指标")]
F9 = [("basic_pathogenesis", "基本病机"), ("disease_location", "病位"), ("disease_nature", "病性"), ("etiology", "病因"),
      ("pathological_factor", "病理因素"), ("disease_tendency", "病势"), ("progression_pattern", "演变规律"), ("pathogenesis_detail", "病机详述")]


def ev_suffix(r, evd):
    rec, ev = one_line(r.get("recommendation")), one_line(r.get("evidence_level"))
    s = []
    if rec:
        s.append("推荐：" + rec)
    if ev:
        s.append("证据等级：" + ev)
    d = evd.get(val(r.get("evidence_def_id")))
    if d:
        s.append("（" + d + "）")
    return "；".join(s)


def add_guideline(kb: KB, cid, gl, glname, wid, wname, rel, tcm_names_out: list, only_tcm=False):

    sh = S.sheets(gl)
    if not sh:
        return
    evd = S.evidence_defs(gl)
    src = f"{gl} {glname}"
    did = wid if only_tcm else "-".join(wid.split("-")[:2])
    wid = "" if only_tcm else wid
    tcm_rows = [r for r in sh.get("中医病名主表", []) if r.get("tcm_disease_id") == did]
    n_tcm_in_gl = len([r for r in sh.get("中医病名主表", []) if r.get("tcm_disease_id")])
    syn = {r["tcm_syndrome_id"]: one_line(r.get("syndrome_name")) for r in sh.get("中医证型表", [])
           if r.get("tcm_syndrome_id") and (r.get("west_disease_id") == wid or (not val(r.get("west_disease_id")) and r.get("tcm_disease_id") == did))}
    syn_all = {r["tcm_syndrome_id"]: one_line(r.get("syndrome_name")) for r in sh.get("中医证型表", []) if r.get("tcm_syndrome_id")}
    stages = {r["stage_def_id"]: one_line(r.get("stage_name")) for r in sh.get("分期定义表", []) if r.get("stage_def_id")}
    scope_note = "" if wid else ("（本指南未按西医病名细分，为中医病名通用内容）" if n_tcm_in_gl else "")


    for r in tcm_rows:
        names = [one_line(r.get("disease_name_tcm"))] + S.split_names(r.get("disease_alias_tcm"))
        tcm_names_out.extend(n for n in names if n)
        f = [("中医病名", "、".join(n for n in names if n)), ("中医定义", r.get("disease_definition_tcm")), ("中医诊断标准", r.get("tcm_diagnostic_criteria"))]
        for b in pack_fields(f, C.MAX_CHARS):
            kb.add(cid, "T_NAME", "guideline", gl, src, b, meta=dict(row_id=r.get("tcm_disease_id")), split=False)

    for r in sh.get("西医病名映射表", []):
        if wid and r.get("west_disease_id") == wid:
            f = [("西医病名（指南原文）", one_line(r.get("disease_name_western")) + (f"，别名：{one_line(r.get('disease_alias_western'))}" if val(r.get("disease_alias_western")) else "")),
                 ("西医定义", r.get("disease_definition_western")), ("与中医病名的对应关系", r.get("relation_type"))]
            for b in pack_fields(f, C.MAX_CHARS):
                kb.add(cid, "W_DEF", "guideline", gl, src, b, meta=dict(row_id=wid), split=False)

    wdiag_ids = set()
    for r in sh.get("西医诊断与分级表", []):
        if not (wid and r.get("west_disease_id") == wid):
            continue
        wdiag_ids.add(r.get("west_diag_id"))
        sub = one_line(r.get("subtype"))
        f = [(lb, r.get(k)) for k, lb in F3 if k != "subtype"]
        if sub:
            f.insert(0, ("分型", sub))
        e = ev_suffix(r, evd)
        if e:
            f.append(("证据", e))
        for b in pack_fields(f, C.MAX_CHARS):
            kb.add(cid, "W_DX", "guideline", gl, src, b, meta=dict(row_id=r.get("west_diag_id"), subtype=sub), split=False, sub=sub)

    for r in sh.get("病机表", []):
        w9 = val(r.get("west_disease_id"))
        hit = (wid and w9 == wid) or (not w9 and r.get("disease_no") in (did, wid)) or (r.get("tcm_syndrome_id") in syn)
        if not hit:
            continue
        lvl = one_line(r.get("level"))
        sname = syn_all.get(r.get("tcm_syndrome_id"), "")
        f = [("病机层级", lvl + (f"·{sname}" if sname else ""))] + [(lb, r.get(k)) for k, lb in F9]
        for b in pack_fields(f, C.MAX_CHARS):
            kb.add(cid, "T_PATH", "guideline", gl, src + scope_note, b, meta=dict(row_id=r.get("pathogenesis_id"), syndrome=sname), split=False, sub=sname)
    
    for r in sh.get("中医证型表", []):
        w4 = val(r.get("west_disease_id"))
        if not ((wid and w4 == wid) or (not w4 and r.get("tcm_disease_id") == did)):
            continue
        sname = one_line(r.get("syndrome_name"))
        if not sname:
            continue
        f = [("证型", sname), ("辨证方法", r.get("syndrome_method")), ("主症", r.get("key_symptoms")), ("次症", r.get("secondary_symptoms")),
             ("舌象", r.get("tongue")), ("脉象", r.get("pulse")), ("辨证依据", r.get("diagnostic_criteria")), ("证型鉴别", r.get("differential"))]
        for b in pack_fields(f, C.MAX_CHARS):
            kb.add(cid, "T_SYN", "guideline", gl, src + scope_note, b,
                   meta=dict(row_id=r.get("tcm_syndrome_id"), syndrome=sname, tongue=one_line(r.get("tongue")), pulse=one_line(r.get("pulse"))), split=False, sub=sname)

    for r in sh.get("治疗表", []):
        wd, sid = val(r.get("west_diag_id")), val(r.get("tcm_syndrome_id"))
        hit = (wd in wdiag_ids) or (not wd and sid in syn) or (not wd and sid == "无证型区分" and n_tcm_in_gl == 1)
        if not hit:
            continue
        ttype = one_line(r.get("treatment_type"))
        sname = syn_all.get(sid, "" if sid == "无证型区分" else sid)
        f = [("类型", ttype + (f"｜证型：{sname}" if sname else "｜无证型区分")), ("治法", r.get("treatment_principle")), ("方剂", r.get("formula")),
             ("组成", r.get("composition")), ("加减", r.get("modification")), ("中成药", r.get("patent_medicine")),
             ("中药注射剂", r.get("injection_medicine")), ("西药", r.get("western_medicine")), ("证据", ev_suffix(r, evd))]
        sec = "W_TX" if "西药" in ttype else "T_HERB"
        for b in pack_fields(f, C.MAX_CHARS):
            kb.add(cid, sec, "guideline", gl, src, b, meta=dict(row_id=r.get("treat_id"), therapy_type=ttype, syndrome=sname,
                   evidence_level=one_line(r.get("evidence_level")), recommendation=one_line(r.get("recommendation"))), split=False, sub=sname)

    for r in sh.get("外治法表", []):
        w6 = val(r.get("west_disease_id"))
        if not ((wid and w6 == wid) or (not w6 and r.get("tcm_disease_id") == did)):
            continue
        ttype = one_line(r.get("therapy_type"))
        sid = val(r.get("tcm_syndrome_id"))
        sname = syn_all.get(sid, "" if sid in ("无证型区分", "") else sid)
        stage = one_line(r.get("disease_stage"))
        ctx = [ttype]
        if sname:
            ctx.append("证型：" + sname)
        elif sid == "无证型区分":
            ctx.append("无证型区分")
        if stage:
            ctx.append("分期：" + stage)
        if val(r.get("concurrent_symptom")) and r.get("concurrent_symptom") != "无证型区分":
            ctx.append("兼症：" + one_line(r.get("concurrent_symptom")))
        if val(r.get("symptom_sign")):
            ctx.append("适用症状体征：" + one_line(r.get("symptom_sign")))
        mp, ap, pa = one_line(r.get("main_points")), one_line(r.get("adjunct_points")), one_line(r.get("points_all"))
        f = [("疗法", "｜".join(ctx)), ("治则", r.get("treatment_principle")), ("主穴", mp), ("配穴", ap), ("取穴", "" if mp else pa),
             ("操作", r.get("operation")),
             ("频次/疗程", "；".join(x for x in (one_line(r.get("treatment_frequency")), one_line(r.get("treatment_course")), one_line(r.get("course_interval"))) if x)),
             ("注意事项", r.get("note")), ("适用", r.get("clinical_application")), ("证据", ev_suffix(r, evd))]
        pts = kb.lex.extract(" ".join((mp, ap, pa)))
        for b in pack_fields(f, C.MAX_CHARS):
            kb.add(cid, "T_ACU", "guideline", gl, src + scope_note, b,
                   meta=dict(row_id=r.get("acu_id"), therapy_type=ttype, syndrome=sname, stage=stage, points=pts,
                             point_based=bool(pts) and bool(POINT_THERAPY.search(ttype)),
                             evidence_level=one_line(r.get("evidence_level")), recommendation=one_line(r.get("recommendation"))),
                   split=False, sub=ttype + (f"·{sname}" if sname else "") + (f"·{stage}" if stage else ""))
    
    for r in sh.get("鉴别诊断表", []):
        dn = one_line(r.get("disease_name"))
        if not (r.get("disease_no") in (did, wid) or (dn and wname and (dn in wname or wname in dn))):
            continue
        n = one_line(r.get("diff_disease_name"))
        if not n:
            continue
        sec = "T_DD" if "中医" in one_line(r.get("diff_disease_type")) else "W_DD"
        kb.add(cid, sec, "guideline", gl, src, f"- {n}：{one_line(r.get('diff_points')) or '（指南仅列出需鉴别，未给出鉴别点）'}",
               meta=dict(row_id=r.get("diff_id"), diff_disease=n), sub=n)

    for r in sh.get("禁忌与护理表", []):
        if r.get("disease_no") not in (did, wid):
            continue
        f = [("适用范围", r.get("scope")), ("禁忌", r.get("contraindication")), ("操作注意", r.get("operation_caution")), ("护理与宣教", r.get("patient_care"))]
        for b in pack_fields(f, C.MAX_CHARS):
            kb.add(cid, "T_CAUTION", "guideline", gl, src, b, meta=dict(row_id=r.get("caution_id"), scope=one_line(r.get("scope"))), split=False)
    
    by_cat = defaultdict(list)
    for r in sh.get("分期定义表", []):
        if r.get("disease_no") not in (did, wid):
            continue
        rel_w = val(r.get("related_west_diag_id"))
        if rel_w and wid and not rel_w.startswith(wid) and re.match(r"GL\d+-D\d+-W\d+", rel_w):
            continue
        by_cat[one_line(r.get("stage_category")) or "分期"].append(r)
    for cat, rs in by_cat.items():
        rs.sort(key=lambda r: str(r.get("stage_order") or "9"))
        lines = [f"- {one_line(r.get('stage_name'))}：{one_line(r.get('stage_definition'))}" for r in rs if val(r.get("stage_name"))]
        note = one_line(rs[0].get("mapping_note"))
        body = f"分期体系：{cat}\n" + "\n".join(lines) + (f"\n对应关系说明：{note}" if note else "")
        kb.add(cid, "T_STAGE" if "治疗" in cat else "W_STAGE", "guideline", gl, src, body, meta=dict(stage_category=cat), sub=cat)


def add_guideline_general(kb: KB, gl, glname):
    sh = S.sheets(gl)
    rows = [r for r in sh.get("禁忌与护理表", []) if val(r.get("disease_no")) == "全指南通用" or "全指南" in one_line(r.get("scope"))]
    if not rows:
        return 0
    kb.concept(gl, f"{glname}（全指南通用注意事项）", gls=[f"{gl} {glname}"])
    n = 0
    for r in rows:
        f = [("适用范围", r.get("scope")), ("禁忌", r.get("contraindication")), ("操作注意", r.get("operation_caution")), ("护理与宣教", r.get("patient_care"))]
        for b in pack_fields(f, C.MAX_CHARS):
            n += kb.add(gl, "T_CAUTION", "guideline", gl, f"{gl} {glname}", b, meta=dict(row_id=r.get("caution_id"), scope="全指南通用", guideline_level=True), split=False)
    return n


# ==================================================================== 书籍原文 → 单元
_HEAD1 = re.compile(r"^【([^】]{1,14})】\s*$")


def is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 14:
        return False
    if _HEAD1.match(s):
        return True
    if re.search(r"[，,。；;：:！？、（）()\d]", s):
        return False
    return len(s) >= 2


def book_subsections(text: str) -> list:
    """按小标题切成 [(小标题, 正文)]；无小标题则整节一段。"""
    out, cur_h, buf = [], "", []
    for ln in text.split("\n"):
        if is_heading(ln):
            if buf and "".join(buf).strip():
                out.append((cur_h, "\n".join(buf).strip()))
            cur_h, buf = ln.strip().strip("【】"), []
        else:
            buf.append(ln)
    if buf and "".join(buf).strip():
        out.append((cur_h, "\n".join(buf).strip()))

    merged = []
    for h, t in out:
        if merged and len(t) < 80:
            merged[-1] = (merged[-1][0], merged[-1][1] + "\n" + (f"{h}：" if h else "") + t)
        elif not merged and len(t) < 80 and len(out) > 1:
            out[1] = (out[1][0], (f"{h}：" if h else "") + t + "\n" + out[1][1])
            continue
        else:
            merged.append((h, t))
    return merged


def add_book(kb: KB, cid, rec):
    src = f"{rec['book']}·{rec['heading']}"
    for h, t in book_subsections(rec["text"]):
        t = re.sub(r"\n{2,}", "\n", t)
        kb.add(cid, "W_BOOK", "book", rec["book"].split(" ")[0], src, t, meta=dict(book_file=rec.get("book_file", "")), overlap=C.BOOK_OVERLAP, sub=h)


# ==================================================================== 主流程
def build(write_md=True):
    os.makedirs(C.KB_DIR, exist_ok=True)
    pts, wl = S.load_acupoints()
    lex = AcuLexicon(pts, wl)
    kb = KB(lex)
    main, cid2gl, gl_titles = S.load_concepts()
    cards = S.load_cards()
    bsec = S.load_booksec()
    print(f"[源] 概念 {len(main)}｜有指南映射 {len(cid2gl)}｜卡片 {len(cards)}｜书籍匹配节 {sum(len(v) for v in bsec['matched'].values())}"
          f"｜书籍未匹配节 {len(bsec['unmatched'])}｜穴位 {len(pts)}")

    # ---- 概念登记（卡片名优先）----
    all_cids = sorted(set(cid2gl) | set(cards) | set(bsec["matched"]))
    for cid in all_cids:
        card, row = cards.get(cid), main.get(cid)
        name = (card or {}).get("canonical_name") or (row["name"] if row else cid)
        aliases = list((card or {}).get("aliases") or (row["aliases"] if row else []))
        icd = (card or {}).get("icd11", {}) if card else {}
        west_names = [w for _, _, _, w, _ in cid2gl.get(cid, [])]
        kb.concept(cid, name, aliases=aliases + [w for w in west_names if len(w) <= 20],
                   icd=(icd.get("code") if icd else "") or (row["icd"] if row else ""),
                   icd_title=(icd.get("title") if icd else "") or (row["icd_title"] if row else ""),
                   west_names=west_names, gls=[f"{g} {n}" for g, n, *_ in cid2gl.get(cid, [])],
                   has_card=bool(card), has_book=cid in bsec["matched"],
                   definition=(card or {}).get("definition", ""), symptoms=j((card or {}).get("key_symptoms")))

    # ---- 卡片 ----
    for cid, card in cards.items():
        add_card(kb, card)
    # ---- 指南（按概念）----
    for cid in all_cids:
        tcm_names = []
        for gl, glname, wid, wname, rel in cid2gl.get(cid, []):
            add_guideline(kb, cid, gl, glname, wid, wname, rel, tcm_names)
        if tcm_names:
            kb.concept(cid, kb.concepts[cid]["name"], tcm_names=tcm_names)
    # ---- 纯中医病名（表1 有、表2 未映射到概念）----
    mapped_wid = {wid for v in cid2gl.values() for _, _, wid, _, _ in v}
    tcm_only = []
    for gl in sorted(S.gl_files()):
        sh = S.sheets(gl)
        d2 = {r.get("tcm_disease_id") for r in sh.get("西医病名映射表", []) if r.get("west_disease_id") in mapped_wid}
        for r in sh.get("中医病名主表", []):
            did = r.get("tcm_disease_id")
            if did and did not in d2:
                name = one_line(r.get("disease_name_tcm"))
                kb.concept(did, name, aliases=S.split_names(r.get("disease_alias_tcm")), gls=[f"{gl} {gl_title_of(gl, gl_titles)}"])
                tn = []
                add_guideline(kb, did, gl, gl_title_of(gl, gl_titles), did, "", "", tn, only_tcm=True)
                kb.concept(did, name, tcm_names=tn)
                tcm_only.append((did, name))
    # ---- 指南通用注意事项 ----
    n_general = sum(add_guideline_general(kb, gl, gl_title_of(gl, gl_titles)) for gl in sorted(S.gl_files()))
    # ---- 书籍原文 ----
    for cid, recs in bsec["matched"].items():
        for rec in recs:
            add_book(kb, cid, rec)
    for i, rec in enumerate(bsec["unmatched"], 1):
        bid = f"B{i:04d}"
        kb.concept(bid, rec["title"], has_book=True)
        add_book(kb, bid, rec)
    # ---- 穴位 ----
    for p in pts:
        cid = "AP-" + (p.get("WHO code") or p["穴位名称"])
        kb.concept(cid, p["穴位名称"], aliases=p.get("aliases", []))
        kb.add(cid, "AP", "acupoint_lib", "acupoint_lib", "针灸穴位标准库_增强版", lex.doc_text(p),
               meta=dict(point=p["穴位名称"], code=p.get("WHO code", ""), meridian=p.get("归经", ""), category=p.get("穴位类别", ""),
                         risk=p.get("危险等级", ""), pregnancy=p.get("妊娠禁忌", ""), short=lex.doc_text(p, short=True)), split=False)
    # ---- 概念路由文档 ----
    for cid, c in kb.concepts.items():
        if cid.startswith("AP-") or re.match(r"^GL\d+$", cid):
            continue
        lines = ["规范名：" + c["name"]]
        if c["aliases"]:
            lines.append("别名：" + "、".join(c["aliases"][:12]))
        if c["tcm_names"]:
            lines.append("中医病名：" + "、".join(c["tcm_names"][:8]))
        if c["icd"]:
            lines.append(f"ICD-11：{c['icd']} {c['icd_title']}".strip())
        if c["definition"]:
            lines.append("定义：" + c["definition"][:200])
        if c["symptoms"]:
            lines.append("主要症状：" + c["symptoms"][:220])
        if c["guidelines"]:
            lines.append("相关指南：" + "；".join(c["guidelines"][:6]))
        lines.append("知识来源：" + "、".join(x for x, ok in (("书籍卡片", c["has_card"]), ("指南", bool(c["guidelines"])), ("书籍原文", c["has_book"])) if ok))
        kb.add(cid, "CONCEPT", "concept", "concept", "概念路由表", "\n".join(lines), split=False)

    # ---- 输出 ----
    write_outputs(kb, tcm_only, n_general, write_md)
    return kb


def gl_title_of(gl, gl_titles):
    return gl_titles.get(gl) or S.gl_title_from_file(gl)


def write_outputs(kb: KB, tcm_only, n_general, write_md):
    units = kb.units
    with open(os.path.join(C.KB_DIR, "units.jsonl"), "w", encoding="utf-8") as fh:
        for u in units:
            fh.write(json.dumps(u, ensure_ascii=False) + "\n")
    for cat in C.COLLECTIONS:
        with open(os.path.join(C.KB_DIR, f"{cat}.jsonl"), "w", encoding="utf-8") as fh:
            for u in units:
                if u["domain"] == cat:
                    fh.write(json.dumps(u, ensure_ascii=False) + "\n")
    json.dump(kb.concepts, open(os.path.join(C.KB_DIR, "concepts.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # ---- 报告 ----
    by_dom, by_sec, by_src = Counter(), Counter(), Counter()
    lens = defaultdict(list)
    for u in units:
        by_dom[u["domain"]] += 1
        by_sec[(u["domain"], u["section_label"])] += 1
        by_src[(u["domain"], u["source_type"])] += 1
        lens[u["domain"]].append(u["chars"])
    concepts = [c for cid, c in kb.concepts.items() if not cid.startswith("AP-") and not re.match(r"^GL\d+$", cid)]
    dom_of = defaultdict(set)
    for u in units:
        dom_of[u["concept_id"]].add(u["domain"])
    both = sum(1 for c in concepts if {"west", "tcm"} <= dom_of[c["cid"]])
    only_w = sum(1 for c in concepts if dom_of[c["cid"]] & {"west", "tcm"} == {"west"})
    only_t = sum(1 for c in concepts if dom_of[c["cid"]] & {"west", "tcm"} == {"tcm"})

    def q(L, p):
        L = sorted(L)
        return L[min(len(L) - 1, int(len(L) * p))] if L else 0
    R = ["# 分域 RAG 知识库 构建报告", "",
         f"- 知识单元总数：{len(units)}（西医 {by_dom['west']}｜中医 {by_dom['tcm']}｜穴位 {by_dom['acupoint']}｜概念 {by_dom['concept']}）",
         f"- 疾病概念：{len(concepts)}（西医+中医两域俱全 {both}｜仅西医 {only_w}｜仅中医 {only_t}）；纯中医病名（无西医映射）{len(tcm_only)} 个：{'、'.join(n for _, n in tcm_only)}",
         f"- 指南级通用注意事项单元：{n_general}",
         f"- 切块参数：MAX_CHARS={C.MAX_CHARS}，MIN_TAIL={C.MIN_TAIL}，书籍原文重叠 {C.BOOK_OVERLAP} 字（表头计入长度）", ""]
    R += ["## 各集合单元长度（字）", "", "| 集合 | 单元数 | 中位数 | P90 | 最大 | <150 字 |", "|---|---|---|---|---|---|"]
    for d in C.COLLECTIONS:
        L = lens[d]
        if L:
            R.append(f"| {d} | {len(L)} | {q(L, .5)} | {q(L, .9)} | {max(L)} | {sum(1 for x in L if x < 150)} |")
    R += ["", "## 各板块单元数", "", "| 域 | 板块 | 单元数 |", "|---|---|---|"]
    for (d, s), n in sorted(by_sec.items(), key=lambda x: (x[0][0], -x[1])):
        R.append(f"| {d} | {s} | {n} |")
    R += ["", "## 各来源单元数", "", "| 域 | 来源类型 | 单元数 |", "|---|---|---|"]
    for (d, s), n in sorted(by_src.items()):
        R.append(f"| {d} | {s} | {n} |")
    R += ["", "## 逐概念明细（ID｜名称｜西医单元｜中医单元｜来源）", ""]
    cnt = defaultdict(Counter)
    for u in units:
        cnt[u["concept_id"]][u["domain"]] += 1
    for c in sorted(concepts, key=lambda c: c["cid"]):
        srcs = "+".join(x for x, ok in (("卡片", c["has_card"]), ("指南", bool(c["guidelines"])), ("书籍", c["has_book"])) if ok)
        R.append(f"- {c['cid']}｜{c['name']}｜W {cnt[c['cid']]['west']}｜T {cnt[c['cid']]['tcm']}｜{srcs}")
    open(os.path.join(C.KB_DIR, "build_report.md"), "w", encoding="utf-8").write("\n".join(R))
    print("\n".join(R[:6]))

    # ---- 人读版 md ----
    if write_md:
        for dom, order in (("west", MD_ORDER_W), ("tcm", MD_ORDER_T)):
            d = os.path.join(C.MD_DIR, dom)
            os.makedirs(d, exist_ok=True)
            for f in os.listdir(d):
                if f.endswith(".md"):
                    os.remove(os.path.join(d, f))
            bycid = defaultdict(list)
            for u in units:
                if u["domain"] == dom:
                    bycid[u["concept_id"]].append(u)
            for cid, us in bycid.items():
                c = kb.concepts[cid]
                L = [f"# {c['name']}" + (f"（{'、'.join(c['aliases'][:6])}）" if c["aliases"] else ""),
                     f"概念ID {cid}" + (f" ｜ ICD-11 {c['icd']} {c['icd_title']}" if c["icd"] else "") + f" ｜ 域：{DOMAIN_LABEL[dom]}",
                     ("中医病名：" + "、".join(c["tcm_names"])) if c["tcm_names"] else "", ""]
                for sec in order:
                    ss = [u for u in us if u["section"] == sec]
                    if not ss:
                        continue
                    L.append(f"## {SECTIONS[sec][1]}")
                    for u in ss:
                        L.append(f"### 来源：{u['source_title']}" + (f"·{u['sub']}" if u.get("sub") else "") + f"  `{u['id']}`")
                        L.append(u["body"])
                        L.append("")
                open(os.path.join(d, safe_name(f"{cid}_{c['name']}.md")), "w", encoding="utf-8").write("\n".join(x for x in L if x is not None))
        print(f"[md] 已写入 {C.MD_DIR}\\west、\\tcm")


if __name__ == "__main__":
    build()
