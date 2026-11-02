#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tcm_study import needling as N
from tcm_study import schema as S
from tcm_study.config import load_cfg
from tcm_study.package import Package, read_jsonl
from tcm_study.doctor_view import standardize_doctor_view

TP_SCRUB = re.compile(r"[，,；;。]?\s*(?:舌[质体]?[淡红暗紫绛胖瘦边尖][^，。；;]{0,12}|舌苔[^，。；;]{0,10}|苔[白黄薄厚腻燥少剥润滑灰黑][^，。；;]{0,8}|脉[弦细滑沉数缓紧濡涩浮迟虚实结代洪微弱][^，。；;]{0,8})")


def bucket(case_id: str, seed: int, n: int) -> int:
    return int(hashlib.sha256(f"{seed}|{case_id}".encode()).hexdigest(), 16) % n


def standardize_followup_target_laterality(user_text: str, target_text: str, lex) -> tuple[str, dict]:
    
    t = str(target_text or "")
    m = re.search(r"(?s)(【取穴】)(.*?)(?=\n【|\Z)", t)
    if not m:
        return t, {"applied": False, "reason": "no_point_section"}
    body = m.group(2).strip()
    if not body:
        return t, {"applied": False, "reason": "empty_point_section"}
    new_body, rep = N.standardize_point_laterality_free_text(lex, body, user_text)
    if not rep.get("applied"):
        return t, rep
    new = t[:m.start(2)] + new_body + t[m.end(2):]
    return new, rep


# ------------------------------------------------------------------------------------------------
# 规则改写
# ------------------------------------------------------------------------------------------------
def _cut_sentence(s: str, n: int) -> str:
    s = (s or "").strip()
    if len(s) <= n:
        return s
    cut = max(s.rfind(p, 0, n) for p in "。；！？")
    return s[: cut + 1] if cut > n // 3 else s[:n]


def _fmt_basic(basic: str) -> str:
    sex = re.search(r"性别[:：]?(\S)", basic or "")
    age = re.search(r"年龄[:：]?([^；;，\s]+)", basic or "")
    return "".join(x for x in [sex.group(1) if sex else "", ("，" if sex and age else "") + age.group(1) if age else ""] if x) or "基本信息未记录"



NULL_WARNING_LINE = True
FALLBACK_NOTE = "疗程内症状无改善或加重，应重新评估诊断。"
WITH_LOCATION = True
DIFF_KB = None
DIFF_MAX_DD = 3
DIFF_MAX_RED = 2

_VITAL = re.compile(r"体温|脉搏|呼吸|血压|心率|\d+\s*(?:次/分|mmHg|℃)|^查体|^诊见")
_NEG = re.compile(r"^无|^未|^否认|正常|未见|未闻|阴性|未引出|未触及|无异常|无明显|神清|神志清|语利|对答|配合|发育|营养|形体适中|面色|痛苦貌|一般情况|生命体征")
_PERTINENT_NEG = re.compile(r"反跳痛|肌紧张|腹膜刺激|病理征|直腿抬高|叩击痛|Hoffmann|霍夫曼|巴宾斯基|Babinski|浮髌|抽屉|麦氏")
_TP = re.compile(r"舌|苔|脉[弦细滑数沉浮缓涩紧濡弱迟洪虚实]|脉象")


def positive_findings(exam: str, max_n: int = 4, max_len: int = 24) -> list[str]:
    
    out = []
    for fr in re.split(r"[，,；;。\n]|\s{1,}", exam or ""):
        fr = re.sub(r"^(?:查体|诊见|专科检查|体格检查)[:：]?", "", fr.strip(" ：:"))
        keep_neg = bool(_PERTINENT_NEG.search(fr))      # 有鉴别意义的阴性体征（无反跳痛、直腿抬高阴性、病理征未引出）保留
        if len(fr) < 2 or _VITAL.search(fr) or _TP.search(fr) or (_NEG.search(fr) and not keep_neg):
            continue
        if not keep_neg and re.search(r"无|未|不明显", fr) and not re.search(r"压痛|阳性|受限|肿|叩痛|减退|消失|亢进|异常", fr):
            continue
        fr = fr[:max_len]
        if fr not in out:
            out.append(fr)
        if len(out) >= max_n:
            break
    return out


def number_notes(notes: list[str]) -> str:
    
    em = [n for n in notes if n.startswith(S.EMERGENCY_TAG)]
    rest = [re.sub(r"^\d+[\.．]\s*", "", n) for n in notes if not n.startswith(S.EMERGENCY_TAG)]
    return "\n".join(em[:1] + [f"{i + 1}. {n}" for i, n in enumerate(rest)])


def rule_rewrite(user_text: str, old_answer: str, lex=None, pending_policy: str = "moderate") -> tuple[dict, dict]:

    inp = S.parse_input(user_text)
    old = S.parse_bracket_sections(old_answer)
    audit: dict = {}
    wm, mv1 = S.clean_dx(old.get("西医诊断", ""), "西医")
    tcm, mv2 = S.clean_dx(old.get("中医诊断", ""), "中医")
    if not tcm and mv1.get("中医诊断"):
        tcm, _ = S.clean_dx(mv1["中医诊断"], "中医")
    if not wm and mv2.get("西医诊断"):
        wm, _ = S.clean_dx(mv2["西医诊断"], "西医")
    pattern = (old.get("证型") or "").strip()
    if S.UNCERTAINTY_NOTE[:12] in pattern:
        pattern = ""
    pattern = pattern or mv1.get("证型") or mv2.get("证型") or ""
    audit["moved"] = {**mv1, **mv2}
    chief = inp.get("主诉") or ""
    if not chief or chief == "未记录":
        chief = _cut_sentence(re.sub(r"^[^。]{0,12}?(?:男|女)[，,]\s*\d+\s*岁[。，,]?", "", inp.get("现病史", "")), 40) or "主诉未记录"

    # 西医诊断：病历没写 → 症状性标签 + 待查；写了但按定义需要检查证据且病例无辅助检查 → 待查
    if not wm:
        wm = S.symptom_label_from_tcm(tcm, chief)
        exams, pend = S.pending_exams(wm, inp["_has_aux"], pending_policy)
        audit["wm_source"] = "symptom_label"
    else:
        exams, pend = S.pending_exams(wm, inp["_has_aux"], pending_policy)
        audit["wm_source"] = "doctor"
   
    pend = pend or S.is_symptom_label(wm)
  
    kb_cid, kb_proxy = (DIFF_KB.resolve(wm, user_text) if DIFF_KB is not None else (None, None))
    if pend and (not exams or exams == [S.GENERIC_EXAM]):
        exams = S.pending_exams(wm, False, pending_policy)[0]
        
    if (not exams or exams == [S.GENERIC_EXAM]) and kb_cid and (pend or not inp["_has_aux"]):
        exams = DIFF_KB.first_exams(kb_cid) or exams
        if exams and exams != [S.GENERIC_EXAM]:
            audit["exams_source"] = "kb"
    if exams == [S.GENERIC_EXAM]:
        exams = []
    if pend:
        if not exams:
            audit["exams_missing"] = True
        wm = S.mark_pending(wm)
        audit["wm_source"] += "+pending"

    # 证型
    if not pattern:
        pattern = S.UNCERTAINTY_NOTE if not inp["_has_tp"] else "病历未记录证型，建议四诊合参后明确"
        audit["pattern_source"] = "note"
    else:
        audit["pattern_source"] = "doctor"

    # 病机 / 治法
    path = (old.get("病机") or "").strip()
    principle = (old.get("治法") or "").strip()
    kb_p, kb_t = S.kb_pathogenesis_and_principle(pattern)
    if not path:
        path = kb_p or ("舌脉未记录，病机有待四诊合参后明确；据现有症状，以局部经络气血运行不畅、不通则痛为主。" if not inp["_has_tp"] else "经络气血运行不畅，不通则痛。")
        audit["path_source"] = "kb" if kb_p else "generic"
    if not principle:
        principle = kb_t or "疏通经络，调和气血，通络止痛"
        audit["principle_source"] = "kb" if kb_t else "generic"

    points = (old.get("取穴") or "").strip()
    
    if lex is not None and points:
        points, latrep = N.standardize_point_laterality(lex, points, user_text)
        audit["point_laterality"] = latrep
    doctor_op = (old.get("操作") or "").strip()
    audit["op_source"] = "doctor" if doctor_op else "generic"
  
    preg = {"remove": [], "notes": [], "pregnant": False}
    if lex is not None and points:
        preg = N.pregnancy_rules(lex, points, user_text)
        if preg["remove"]:
            points = N.remove_points(points, preg["remove"])
            audit["pregnancy_removed"] = preg["remove"]
   
    spec = N.needling_spec(lex, points, doctor_op, WITH_LOCATION) if (lex is not None and points) else []
    op0 = N.compose_operation(doctor_op, spec)
    op, op18 = N.standardize_operation_v18(lex, points, op0, user_text) if (lex is not None and points) else (op0, {"applied": False})
    audit["n_spec_points"] = len(spec)
    audit["operation"] = operation

  
    aux_s = inp.get("辅助检查", "").rstrip("。；，") if inp["_has_aux"] else ""
    tp_s = inp.get("舌脉", "") if inp["_has_tp"] else ""
    exam_s = re.sub(r"^查体[:：]\s*", "", inp.get("查体", "")).rstrip("。；，")
    hpi_s = (inp.get("现病史") or "").strip()
    chief_s = re.sub(r"^.{0,30}?主诉[:：]\s*", "", chief.strip())
    chief_s = re.split(r"[。；;]", chief_s)[0].strip("，,；;。 ")[:60] or chief.strip("。")[:60]
    think: list[str] = [f"患者主诉{chief_s}。"]
    if hpi_s and hpi_s != "未记录":
        hs = _cut_sentence(hpi_s, 110).strip("。；; ")
        if hs and hs not in chief_s:
            think.append(f"现病史显示：{hs}。")
    findings = []
    if exam_s and exam_s != "未记录":
        pos = positive_findings(exam_s)
        if pos:
            if DIFF_KB is not None:
                pos, interp = DIFF_KB.annotate_findings(wm, user_text, pos, max_n=2)
                if interp:
                    audit["finding_interpretations"] = interp
            findings.append("查体见" + "、".join(pos))
    if aux_s:
        findings.append("辅助检查示" + _cut_sentence(aux_s, 80).rstrip("。；，"))
    if tp_s:
        findings.append(re.sub(r"^舌脉[:：]?", "", tp_s).strip("。；，"))
    else:
        findings.append("舌脉未记录")
    if findings:
        think.append("；".join(findings) + "。")

    wm_plain = re.sub(r"（待查）|\(待查\)", "", wm).split("；")[0]
    sup = DIFF_KB.support(kb_cid, user_text) if (DIFF_KB is not None and kb_cid) else []
    audit["think_support"] = sup
    sym = S.is_symptom_label(wm)

    tcm_s = f"中医属“{tcm}”" if tcm else "中医病名待结合现有记录判断"
    wm_s = f"西医拟诊为“{wm_plain}”" + ("（证据尚需补充）" if pend else "")
    if tp_s and audit["pattern_source"] == "doctor":
        think.append(f"综合判断：{wm_s}；{tcm_s}，辨为{pattern}；病机为{path.rstrip('。')}。")
    else:
        path_brief = re.sub(r"^舌脉未记录，病机有待四诊合参后明确；?", "", path).strip("；;。 ")
        think.append(f"综合判断：{wm_s}；{tcm_s}；舌脉未记录，暂不强行辨证；现阶段病机考虑{(path_brief or '局部经络气血运行不畅').rstrip('。')}。")

    support_dx = (kb_proxy if sym and kb_proxy else wm_plain)
    if sup:
        support_text = f"病例中{'、'.join(sup)}"
    elif hpi_s and hpi_s != "未记录":
        hpi_support = _cut_sentence(hpi_s, 72).strip("。；; ")
        support_text = f"现病史中“{hpi_support}”" if hpi_support else "上述主诉及病程特征"
        if exam_s and exam_s != "未记录":
            support_text += "及已记录查体"
        if aux_s:
            support_text += "、辅助检查"
    elif aux_s:
        support_text = "已记录查体及辅助检查"
    elif exam_s and exam_s != "未记录":
        support_text = "已记录查体"
    else:
        support_text = "上述主诉及病程特征"

    gaps: list[str] = []
    if DIFF_KB is not None and kb_cid:
        miss = DIFF_KB.missing_keypoints(kb_cid, user_text, max_n=2)
        if miss:
            gaps.append("尚未记录" + "、".join(miss))
            audit["missing_keypoints"] = miss
  
    if exams and (pend or not inp["_has_aux"]):
        exam_gap = "尚缺少" + "、".join(exams[:2])
        if exam_gap not in gaps:
            gaps.append(exam_gap)
    if audit.get("finding_interpretations"):
        gaps.insert(0, "存在与典型表现不完全一致的已记录体征")

    evidence_sentence = f"支持诊断的依据为{support_text}，可支持{support_dx}这一拟诊方向"
    if gaps:
        evidence_sentence += "；但证据不足之处在于" + "；".join(dict.fromkeys(gaps))
    elif pend:
        evidence_sentence += "；但目前证据仍不足以作确定诊断"
    think.append(evidence_sentence.rstrip("。") + "。")

   
    dline, dinfo = ("", {})
    if DIFF_KB is not None:
        dline, dinfo = DIFF_KB.note(wm, user_text, max_dd=DIFF_MAX_DD, max_red=DIFF_MAX_RED)
        audit["differential"] = dinfo
        if dline:
            think.append(dline if dline.endswith("。") else dline + "。")

    # 注意事项
    notes: list[str] = []
    flags = S.red_flag_warnings(user_text)
    em_line, other = S.format_notes(flags)
    if em_line:
        notes.append(em_line)
    elif NULL_WARNING_LINE:
        notes.append(S.EMERGENCY_TAG + "无。")
    notes += other
    if exams:
        notes.append(("建议完善：" if pend else "必要时完善：") + "；".join(exams) + ("，以明确诊断。" if pend else "。"))
    if inp["_has_aux"] and S.aux_consistent(inp["辅助检查"], wm) is False:
        notes.append(f"辅助检查提示“{_cut_sentence(inp['辅助检查'].rstrip('。'), 40)}”，与目前诊断需相互印证/鉴别，必要时请相应专科评估。")
        audit["aux_inconsistent"] = True
   
    for n_ in preg["notes"]:
        if n_.startswith("【妊娠禁忌】"):
            notes.insert(1 if (notes and notes[0].startswith(S.EMERGENCY_TAG)) else 0, n_)
        else:
            notes.append(n_)
    audit["pregnancy"] = {k: preg[k] for k in ("pregnant", "a_points", "b_points") if k in preg}
    if lex is not None and points:
        try:
            items = lex.extract(points)
            names = [i["canonical"] for i in items if i["status"] in ("standard", "alias")]
            danger = [n for n in names if lex.rows.get(n, {}).get("禁深刺") == "是" or lex.rows.get(n, {}).get("危险等级") == "高"]
            if danger:
                notes.append("、".join(danger[:4]) + "不宜深刺，严格按解剖安全深度与方向进针。")
        except Exception:
            pass
 
    if not [n_ for n_ in notes if not S.NULL_WARNING_RE.match(n_)]:
        notes.append(FALLBACK_NOTE)
    audit["red_flags"] = [h["rule"] for h in flags]
    audit["pending"] = pend
  
    fixed = []
    for n_ in notes:
        n2, rm = S.sex_consistent(n_, user_text)
        if rm:
            audit.setdefault("sex_consistency_removed", []).extend(rm)
        if n2.strip(" ：:；;，,。"):
            fixed.append(n2.strip())
    notes = fixed

    reasoning0 = "".join(think)
    notes0 = number_notes(notes)
    reasoning1, notes1, view_audit = standardize_doctor_view(
        lex, reasoning0, notes0, user_text, points_text=points, western_dx=wm
    )
    audit["doctor_view"] = view_audit
    sec = {"诊疗思路": reasoning1, "西医诊断": wm, "中医诊断": tcm, "证型": pattern, "病机": path, "治法": principle,
           "取穴": points, "操作": op, "注意事项": notes1}

    return sec, audit


_SAFETY_NOTE = re.compile(r"不宜深刺|妊娠|排除妊娠|禁用穴")


def stage1_like(sec: dict) -> dict:
    
    s = dict(sec)
    s["操作"] = N.strip_spec(s.get("操作", "")) or N.GENERIC_OP
    lines = [l for l in s.get("注意事项", "").split("\n") if not _SAFETY_NOTE.search(l)]
    s["注意事项"] = number_notes(lines) if lines else FALLBACK_NOTE
    return s



TEACHER_SYS = "你是资深针灸科主任医师，负责把门诊病历改写成规范、审慎的临床辅助诊疗记录。只输出 JSON。"
TEACHER_USER = """下面是一份病例和医师的原始记录。医师给出的【西医诊断】【中医诊断】【证型】【取穴】是金标准，不得改动、不得质疑。
你的任务：补写或润色以下字段，返回 JSON（键：思路, 病机, 治法, 操作, 注意事项）：
- 思路：建议180–360字，写成连贯临床推理。按“主诉与病程→现病史及伴随症状→查体/辅助检查/舌脉→综合诊断与中医病位/病机→诊断依据→需鉴别疾病及关键区分点”组织。病例有明确发作/加重日期或时长才引用，未提供则不得补造。对有诊断意义、尤其与拟诊典型表现不一致的阴性体征，紧随原体征写括号解释，例如“局部无压痛（此体征与典型跖筋膜炎的固定压痛点不符，需结合重复查体及影像学进一步鉴别）”；括号解释必须有病例或检索证据支持。鉴别统一写“疾病（关键区分点：……）”。只有已有阴性检查或明确反证时才可写“排除/基本排除”，否则写“尚需排除/需鉴别”。
- 病机：≤120字，与证型一致；舌脉未记录时不得编造舌脉。若原记录已有病机，优先保留其核心含义。
- 治法：≤40字，与证型/取穴一致。若原记录已有治法，优先保留。
- 操作：固定写“1. 手法与疗程”与“2. 逐穴操作”；不得增加穴位。取穴侧别以规则层给出的【取穴】为准：非中线穴必须写（左侧）/（右侧）/（双侧），任督脉及明确中线穴不得写双侧；单侧肢体病变的局部穴及同一患肢远端经穴优先患侧，跨肢体/全身调节穴默认双侧，对侧取穴只有原记录明确时才写。标准穴位定位与深度最终会由确定性穴位安全层校正，阿是穴应结合病例疼痛/压痛部位与侧别描述。
- 注意事项：仅写医生执行针灸时需注意的安全与随访事项，不重复鉴别诊断。固定四项：1. 禁忌与特殊人群；2. 建议完善检查；3. 操作前后注意事项；4. 患者教育。若规则生成内容含【急危重症警示】，必须原样保留在四项之前；不得删除规则生成的妊娠/高风险穴位/检查等内容。
禁止：编造病例中没有的舌象、脉象、检查结果或具体发作日期；出现病例以外的新穴位；使用“仅供参考”“咨询医生”之类免责话术。

【病例】
{case}

【医师原始记录】
{old}

【规则生成的注意事项】
{notes}
"""



class Teacher:
    def __init__(self, url: str, model: str, api_key_env: str = "LLM_API_KEY", temperature: float = 0.2, max_tokens: int = 900,
                 extra_body: dict | None = None, cache_path: str | Path | None = None):
        self.url, self.model = url.rstrip("/") + "/chat/completions", model
        self.key, self.t, self.max_tokens = os.environ.get(api_key_env, "EMPTY"), temperature, max_tokens
        self.extra = dict(extra_body or {})
        self.cache: dict = {}
        self.n_err = 0
        
        self.cache_path = Path(cache_path) if cache_path else None
        self._disk: dict = {}
        if self.cache_path and self.cache_path.is_file():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    d = json.loads(line)
                    self._disk[d["k"]] = d["v"]
                except Exception:
                    pass
            print(f"[TEACHER] 读入磁盘缓存 {len(self._disk)} 条：{self.cache_path}", flush=True)

    def _hkey(self, case: str, old: str, notes: str) -> str:
        return hashlib.sha256(json.dumps([self.model, self.t, self.max_tokens, self.extra, TEACHER_SYS, TEACHER_USER, case, old, notes],
                                         ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def _store(self, j: tuple, v) -> None:
        self.cache[j] = v
        if v is not None and self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"k": self._hkey(*j), "v": v}, ensure_ascii=False) + "\n")

    def _ask(self, case: str, old: str, notes: str) -> dict | None:
        body = json.dumps({"model": self.model, "temperature": self.t, "max_tokens": self.max_tokens, **self.extra,
                           "messages": [{"role": "system", "content": TEACHER_SYS},
                                        {"role": "user", "content": TEACHER_USER.format(case=case, old=old, notes=notes)}]}).encode()
        for k in range(4):
            try:
                req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    txt = json.loads(r.read().decode())["choices"][0]["message"]["content"] or ""
                m = re.search(r"\{.*\}", txt, re.S)
                return json.loads(m.group(0)) if m else None
            except Exception as e:
                self.n_err += 1
                if self.n_err <= 3:
                    msg = e.read().decode(errors="ignore")[:200] if hasattr(e, "read") else str(e)[:200]
                    print(f"[TEACHER][WARN] {type(e).__name__}: {msg}", flush=True)
                time.sleep(3 * (k + 1))
        return None

    def ask(self, case: str, old: str, notes: str) -> dict | None:
        key = (case, old, notes)
        if key not in self.cache:
            hk = self._hkey(*key)
            if hk in self._disk:
                self.cache[key] = self._disk[hk]
            else:
                self._store(key, self._ask(case, old, notes))
        return self.cache[key]

    def prefetch(self, jobs: list[tuple[str, str, str]], workers: int = 8) -> None:
        """并发预取（结果只按输入缓存，与串行调用完全等价；顺序与随机数不受影响）。"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        todo = []
        for j in dict.fromkeys(jobs):
            if j in self.cache:
                continue
            hk = self._hkey(*j)
            if hk in self._disk:
                self.cache[j] = self._disk[hk]
            else:
                todo.append(j)
        print(f"[TEACHER] 缓存命中 {len(set(jobs)) - len(todo)} 条，需调用 API {len(todo)} 条", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futs = {ex.submit(self._ask, *j): j for j in todo}
            for n, f in enumerate(as_completed(futs), 1):
                self._store(futs[f], f.result())
                if n <= 3 or n % 10 == 0 or n == len(todo):
                    ok = sum(v is not None for v in self.cache.values())
                    print(f"[TEACHER] {n}/{len(todo)}  成功解析 {ok}  {time.time() - t0:.0f}s", flush=True)


def apply_teacher(teacher: Teacher, user_text: str, old_answer: str, sec: dict, lex=None) -> tuple[dict, str]:
    inp = S.parse_input(user_text)
    out = teacher.ask(user_text, old_answer, sec["注意事项"])
    if not out:
        return sec, "teacher_failed"
    new = dict(sec)
    for key, field, limit in (("思路", "诊疗思路", 260), ("病机", "病机", 160), ("治法", "治法", 60), ("操作", "操作", 400), ("注意事项", "注意事项", 700)):
        v = str(out.get(key) or "").strip()
        if not v or len(v) > limit:
            continue
        if not inp["_has_tp"] and (S.TONGUE_RE.search(v) or S.PULSE_RE.search(v)) and not S.UNCERTAINTY_NOTE[:6] in v and "未记录" not in v:
            return sec, "teacher_fabricated_tp"
        if re.search(r"仅供参考|咨询医生|专业医师|免责", v):
            v = re.sub(r"[^。；\n]*(?:仅供参考|咨询医生|专业医师|免责)[^。；\n]*[。；]?", "", v)
        if field == "注意事项" and sec["注意事项"].startswith(S.EMERGENCY_TAG) and not v.startswith(S.EMERGENCY_TAG):
            v = sec["注意事项"].split("\n")[0] + "\n" + v
        if field == "操作" and lex is not None:
            try:
                base = {i["canonical"] for i in lex.extract(sec["取穴"]) if i["status"] in ("standard", "alias")}
                extra = {i["canonical"] for i in lex.extract(v.replace("：", "、")) if i["status"] in ("standard", "alias")} - base
                if extra:
                    continue
            except Exception:
                pass
         
            v0 = N.compose_operation(N.strip_spec(v), N.needling_spec(lex, sec["取穴"], N.strip_spec(v), WITH_LOCATION))
            v, _ = N.standardize_operation_v18(lex, sec["取穴"], v0, user_text)
        if field == "注意事项":
            keep = [l for l in sec["注意事项"].split("\n") if _SAFETY_NOTE.search(l) and l not in v]
            if keep:
                v = v.rstrip() + "\n" + "\n".join(keep)
        new[field] = v

    rr, nn, _ = standardize_doctor_view(
        lex, new.get("诊疗思路", ""), new.get("注意事项", ""), user_text,
        points_text=new.get("取穴", ""), western_dx=new.get("西医诊断", "")
    )
    new["诊疗思路"], new["注意事项"] = rr, nn
    return new, "teacher_ok"


def scrub_tongue_pulse(user_text: str) -> str:
    t = re.sub(r"【舌脉】.*", "【舌脉】未记录", user_text)
    return TP_SCRUB.sub("", t)


def contrast_variant(user_text: str, sec: dict, audit: dict) -> tuple[str, dict] | None:
    inp = S.parse_input(user_text)
    if not inp["_has_tp"] or audit.get("pattern_source") != "doctor":
        return None
    u2 = scrub_tongue_pulse(user_text)
    s2 = dict(sec)
    s2["证型"] = S.UNCERTAINTY_NOTE
    s2["病机"] = TP_SCRUB.sub("", s2["病机"])
    s2["病机"] = re.sub(r"[^。；]*(?:舌|脉)[^。；]*[。；]?", "", s2["病机"]).strip() or "舌脉未记录，病机有待四诊合参后明确；据现有症状以经络气血运行不畅为主。"
    s2["诊疗思路"] = re.sub(r"舌脉：[^。]*。", "", TP_SCRUB.sub("", s2["诊疗思路"]))
    tcm = (s2.get("中医诊断") or "").strip()
    s2["诊疗思路"] = re.sub(r"[^。]*(?:舌|脉)[^。]*。", "", s2["诊疗思路"]).rstrip() + (f"中医属“{tcm}”；" if tcm else "") + "舌脉未记录，暂不辨证型。"
  
    tp = lambda s: bool(S.TONGUE_RE.search(s) or S.PULSE_RE.search(s))
    if any(tp(v) for v in s2.values()):
        return None
    return u2, s2



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--teacher", action="store_true", help="")
    ap.add_argument("--teacher-url", default="", help="")
    ap.add_argument("--teacher-model", default="", help="")
    ap.add_argument("--contrast", action="store_true", default=True)
    ap.add_argument("--no-contrast", dest="contrast", action="store_false")
    ap.add_argument("--rag-fraction", type=float, default=-1.0, help="")
    ap.add_argument("--rag-mode", choices=["replace", "add"], default="", help="")
    ap.add_argument("--rag-max-chars", type=int, default=0, help="")
    ap.add_argument("--allow-bm25-only", action="store_true", help="")
    ap.add_argument("--pending-policy", choices=["moderate", "strict"], default="moderate", help="")
    ap.add_argument("--limit", type=int, default=0, help="")
    ap.add_argument("--skip-verify", action="store_true", help="")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    global NULL_WARNING_LINE
    NULL_WARNING_LINE = bool(cfg["train"].get("null_warning_line", False))
    print(f"[SFT-DATA] 无急危征象时写'【急危重症警示】无。'：{NULL_WARNING_LINE}")
    global DIFF_KB, DIFF_MAX_DD, DIFF_MAX_RED
    if cfg["train"].get("differential_note", False):
        from tcm_study.differential import DxKB
        DIFF_KB = DxKB(cfg["outputs"]["rag_index"])
        DIFF_MAX_DD = int(cfg["train"].get("differential_max_dd", 3))
        DIFF_MAX_RED = int(cfg["train"].get("differential_max_red", 2))
        print(f"[SFT-DATA] 具体疾病级鉴别：开（W_DD最多{DIFF_MAX_DD}个；W_RED严重疾病最多{DIFF_MAX_RED}个）")
    global WITH_LOCATION
    WITH_LOCATION = bool(cfg["train"].get("point_location", False))
    print(f"[SFT-DATA] 临床化诊疗思路（体征括号解释+鉴别关键区分点）+病例/解剖感知取穴侧别+结构化操作；穴位定位：{WITH_LOCATION}")
    pkg = Package(cfg["paths"]["package_dir"])
    if a.skip_verify:
        print("[WARN] --skip-verify: package integrity NOT checked (debug only)")
        fingerprint = "UNVERIFIED"
    else:
        rep = pkg.verify()
        if not rep["ok"]:
            raise SystemExit(f"[ERROR] package verification failed: {rep['errors']}")
        fingerprint = pkg.fingerprint()
    seed = int(cfg["study"]["seed"])
    tc = cfg["train"]
    rows = read_jsonl(pkg.root / "train.jsonl")
    if a.limit:
        rows = rows[: a.limit]

    lex = None
    try:
        from tcm_study.acupoints import AcupointLexicon
        lex = AcupointLexicon(pkg._one(""))
    except Exception as e:
        print(f"[WARN] acupoint lexicon unavailable ({e}); skipping needling-safety notes")

    tcfg = cfg.get("teacher") or {}
    t_url = a.teacher_url or (tcfg.get("base_url") if (a.teacher or a.teacher_model) else "")
    t_model = a.teacher_model or (tcfg.get("model") if (a.teacher or a.teacher_url) else "")
    teacher = None
    if (a.teacher or a.teacher_url or a.teacher_model) and not (t_url and t_model):
        raise SystemExit("[ERROR] 要求使用教师模型，但 configs/study.yaml 的 teacher 块为空。本研究按伦理范围不使用教师模型："
                         "去掉 TEACHER=1 / --teacher，训练目标由规则生成。")
    if t_url and t_model:
        key_env = tcfg.get("api_key_env", "LLM_API_KEY")
        if not os.environ.get(key_env) and not t_url.startswith("") and "localhost" not in t_url:
            raise SystemExit(f"")
        teacher = Teacher(t_url, t_model, key_env, extra_body=tcfg.get("extra_body"),
                          cache_path=Path(cfg["paths"]["work_dir"]) / "sft" / "teacher_cache.jsonl")
        print(f"[TEACHER] {t_model} @ {t_url}  —— 注意：训练病例将发送到该服务；若为云端 API 请在 Methods 中说明")
    rag = None
    rag_fraction = float(tc.get("rag_fraction", 0.0)) if a.rag_fraction < 0 else a.rag_fraction
    rag_mode = a.rag_mode or str(tc.get("rag_mode", "replace"))
    from tcm_study.arms import CARDS_WRAPPER, EVIDENCE_WRAPPER
    if rag_fraction > 0:
        from tcm_study.rag import RagEvidence, RagIndex
        idx = RagIndex(cfg["outputs"]["rag_index"], cfg["rag"].get("device", "cpu"))
        if not idx.dense and not a.allow_bm25_only:
            raise SystemExit("[ERROR] 稠密索引未建（data/painrag_kb/*.npy 不存在）。RAFT 样本的证据必须与推理时完全一致：先 bash run_rag_bg.sh，"
                             "或仅本地预览时加 --allow-bm25-only")
        rcfg = dict(cfg["rag"])
        if a.rag_max_chars > 0:
            rcfg["max_chars"] = a.rag_max_chars
        rag = RagEvidence(idx, rcfg)
        print(f"[RAFT] rag_fraction={rag_fraction} mode={rag_mode} dense={idx.dense} budget={rag.budget}")

    tag = a.tag if a.fold is None else f"{a.tag}_fold{a.fold}"
    n_split = round(1 / float(tc["val_fraction"])) if a.fold is None else int(tc["cv_folds"])
    is_val = (lambda cid: bucket(cid, seed, n_split) == (0 if a.fold is None else a.fold))
    out_dir = Path(cfg["paths"]["work_dir"]) / "sft" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"n_in": len(rows), "initial": 0, "followup": 0, "contrast": 0, "rag": 0, "rag_cards": 0, "teacher_ok": 0, "teacher_failed": 0, "teacher_fabricated_tp": 0,
             "pending": 0, "emergency": 0, "spec_points": 0, "pregnancy_cases": 0, "pregnancy_removed": 0, "wm_source": {}, "pattern_source": {}}

    def stage2_user(user_rag: str, sec: dict) -> tuple[str, dict]:
        """RAFT stage-2 样本的 user：病例 + 证据 + 第一步输出 + 医师所用穴位的安全卡（+ 手法卡）。与推理 compose_stage2 完全同构。"""
        pts = []
        for it in lex.extract(sec.get("取穴", "")):
            if it["status"] in ("standard", "alias"):
                pts.append(it["canonical"])
            elif it["status"] == "whitelist":
                pts.append(it["token"])
        manips = rag.manip_names_in(sec.get("操作", ""))
        cards, clog = rag.cards(pts, manips)
        if rag.cards_cfg.get("safety_check", False):
            from tcm_study.safety_layer import guideline_safety_card, safety_check_card
            case_only = user_rag.split("\n\n【检索证据】", 1)[0]
            card = safety_check_card(lex, case_only, [p for p in pts if p in lex.rows])
            cards = (cards.strip() + "\n\n" + card) if cards.strip() else card
            if rag.cards_cfg.get("differential_card", False):
                dcard, dlog = rag.differential_card(sec.get("西医诊断", ""), case_only)
                if dcard:
                    cards = cards.strip() + "\n\n" + dcard
                    clog = {**clog, "differential": dlog}
            if rag.cards_cfg.get("guideline_safety", False):
                gcard = guideline_safety_card(ev_text, int(rag.cards_cfg.get("guideline_safety_max_chars", 1800)), include_red=False)
                if gcard:
                    cards = cards.strip() + "\n\n" + gcard
        s1 = S.render_sections(stage1_like(sec), extra=())
        return user_rag + CARDS_WRAPPER.format(stage1=s1, cards=cards or "【穴位安全卡】无（第一步所列穴位不在标准穴位库中）"), clog
    if teacher:
        jobs = []
        for r in rows:
            if r["meta"].get("task") == "initial":
                u, o = r["messages"][1]["content"], r["messages"][-1]["content"]
                jobs.append((u, o, rule_rewrite(u, o, lex, a.pending_policy)[0]["注意事项"]))
        print(f"[TEACHER] 并发预取 {len(jobs)} 条（concurrency={int(tcfg.get('concurrency', 8))}）", flush=True)
        teacher.prefetch(jobs, int(tcfg.get("concurrency", 8)))
    tr, va, review = [], [], []
    rnd = random.Random(seed)
    t0 = time.time()
    for k, r in enumerate(rows, 1):
        msgs, meta = r["messages"], dict(r["meta"])
        user, old = msgs[1]["content"], msgs[-1]["content"]
        followup_lat = {"applied": False, "reason": "not_followup"}
        if meta.get("task") != "initial" and lex is not None:
            old, followup_lat = standardize_followup_target_laterality(user, old, lex)
        sys_msg = {"role": "system", "content": S.SYSTEM_PROMPT}
        dest = va if is_val(meta["case_id"]) else tr
        
        force_safety_rag = rag is not None and bool(S.is_emergency_level(S.red_flag_warnings(user)))
        
        force_diff_rag = False
        if rag is not None and meta.get("task") == "initial" and bool(tc.get("force_differential_rag", False)):
            ref_sec = S.parse_bracket_sections(old, top_level_only=True)
            ref_wm = (ref_sec.get("西医诊断") or "").strip()
            if ref_wm:
                _, _dl = rag.differential_card(ref_wm, user)
                force_diff_rag = len(list((_dl or {}).get("dd") or []) + list((_dl or {}).get("red") or [])) >= 2
        use_rag = rag is not None and (force_safety_rag or force_diff_rag or bucket(meta["case_id"] + "|rag", seed, 1000) < rag_fraction * 1000)
        ev_text = ""
        if use_rag:
            ev_text, _ = rag.build(RagEvidence.query_from_messages([{"role": "user", "content": user}]))
            use_rag = bool(ev_text)
        user_rag = user + EVIDENCE_WRAPPER.format(evidence=ev_text) if use_rag else user
        if meta.get("task") != "initial":
            stats["followup"] += 1
            if use_rag and rag_mode == "replace":
                dest.append({"messages": [sys_msg, {"role": "user", "content": user_rag}, {"role": "assistant", "content": old}], "meta": {**meta, "variant": "orig+rag", "point_laterality": followup_lat}})
                stats["rag"] += 1
            else:
                dest.append({"messages": [sys_msg, {"role": "user", "content": user}, {"role": "assistant", "content": old}], "meta": {**meta, "variant": "orig", "point_laterality": followup_lat}})
                if use_rag:
                    dest.append({"messages": [sys_msg, {"role": "user", "content": user_rag}, {"role": "assistant", "content": old}], "meta": {**meta, "variant": "rag", "point_laterality": followup_lat}})
                    stats["rag"] += 1
            continue
        stats["initial"] += 1
        sec, audit = rule_rewrite(user, old, lex, a.pending_policy)
        status = "rules"
        if teacher:
            sec, status = apply_teacher(teacher, user, old, sec, lex)
            stats[status] += 1
        stats["pending"] += int(audit.get("pending", False))
       
        _diff_text = (sec.get("诊疗思路") or "") + "\n" + (sec.get("注意事项") or "")
        stats["differential"] = stats.get("differential", 0) + int(any(
            marker in _diff_text for marker in ("需鉴别：", "需排除：", "需优先排除：")
        ))
        stats["think_support"] = stats.get("think_support", 0) + int(bool(audit.get("think_support")))
        stats["fallback_note"] = stats.get("fallback_note", 0) + int(FALLBACK_NOTE in sec["注意事项"])
        stats["pending_no_exam"] = stats.get("pending_no_exam", 0) + int(bool(audit.get("exams_missing")))
        stats["emergency"] += int(S.has_real_warning(sec["注意事项"]))
        stats["spec_points"] += int(audit.get("n_spec_points", 0))
        stats["pregnancy_cases"] += int(bool(audit.get("pregnancy", {}).get("pregnant")))
        stats["pregnancy_removed"] += len(audit.get("pregnancy_removed", []))
        for kk in ("wm_source", "pattern_source"):
            stats[kk][audit.get(kk, "?")] = stats[kk].get(audit.get(kk, "?"), 0) + 1
        target = S.render_sections(sec, extra=())
        if use_rag and rag_mode == "replace":
            dest.append({"messages": [sys_msg, {"role": "user", "content": user_rag}, {"role": "assistant", "content": target}],
                         "meta": {**meta, "variant": "orig+rag", "audit": audit, "teacher": status}})
            stats["rag"] += 1
        else:
            dest.append({"messages": [sys_msg, {"role": "user", "content": user}, {"role": "assistant", "content": target}],
                         "meta": {**meta, "variant": "orig", "audit": audit, "teacher": status}})
        if use_rag:
            u3, clog = stage2_user(user_rag, sec)
            dest.append({"messages": [sys_msg, {"role": "user", "content": u3}, {"role": "assistant", "content": target}],
                         "meta": {**meta, "variant": "rag+cards", "force_safety_rag": bool(force_safety_rag), "force_differential_rag": bool(force_diff_rag),
                                  "cards": {k: clog[k] for k in ("points_carded", "manipulations_carded", "cards_sha256")}}})
            stats["rag_cards"] += 1
        review.append({"case_id": meta["case_id"], "variant": "orig+rag" if (use_rag and rag_mode == "replace") else "orig", "input": user, "old": old, "new": target, "teacher": status})
        if a.contrast:
            cv = contrast_variant(user, sec, audit)
            if cv:
                u2, s2 = cv
                dest.append({"messages": [sys_msg, {"role": "user", "content": u2}, {"role": "assistant", "content": S.render_sections(s2, extra=())}],
                             "meta": {**meta, "variant": "contrast_no_tp"}})
                stats["contrast"] += 1
        if use_rag and rag_mode == "add":
            dest.append({"messages": [sys_msg, {"role": "user", "content": user_rag}, {"role": "assistant", "content": target}],
                         "meta": {**meta, "variant": "rag"}})
            stats["rag"] += 1
        if k % 50 == 0:
            print(f"[ENRICH] {k}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)

    k_os = int((cfg.get("train") or {}).get("redflag_oversample", 1) or 1)
    if k_os > 1:
        rf = [r for r in tr if S.has_real_warning(r["messages"][-1]["content"])]
        tr += [dict(r, meta={**r["meta"], "oversample": i + 2}) for r in rf for i in range(k_os - 1)]
        stats["redflag_samples"] = len(rf)
        stats["redflag_oversample_added"] = len(rf) * (k_os - 1)
        print(f"[OVERSAMPLE] 带急危重症警示的训练样本 {len(rf)} 条 ×{k_os}（新增 {len(rf) * (k_os - 1)} 条）", flush=True)
    rnd.shuffle(tr)
    for name, data in (("train", tr), ("val", va)):
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats.update({"n_train": len(tr), "n_val": len(va), "tag": tag})
    (out_dir / "split_manifest.json").write_text(json.dumps({
        "tag": tag, "seed": seed, "n_train": len(tr), "n_val": len(va), "test_fingerprint": fingerprint,
        "train_case_ids_sha256": hashlib.sha256(",".join(sorted({r["meta"]["case_id"] for r in tr})).encode()).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(S.SYSTEM_PROMPT.encode()).hexdigest(),
        "enrich": {k: v for k, v in stats.items()}, "teacher": {"url": t_url, "model": t_model},
        "raft": {"rag_fraction": rag_fraction, "rag_mode": rag_mode, "dense": bool(rag and rag.index.dense), "budget": rag.budget if rag else None,
                 "domains": list(rag.domains) if rag else None, "cards": rag.cards_cfg if rag else None,
                 "encoder": rag.index.encoder_name if rag else None, "evidence_wrapper_sha256": hashlib.sha256(EVIDENCE_WRAPPER.encode()).hexdigest() if rag else None,
                 "cards_wrapper_sha256": hashlib.sha256(CARDS_WRAPPER.encode()).hexdigest() if rag else None},
        "leak_check": "no test case id in train/val (verified by Package.verify)"}, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import pandas as pd
        pd.DataFrame(review).to_excel(out_dir / "review_targets.xlsx", index=False)
    except Exception as e:
        print(f"[WARN] review xlsx not written: {e}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"[SFT-DATA] {tag}: train={len(tr)} val={len(va)} -> {out_dir}")


if __name__ == "__main__":
    main()
