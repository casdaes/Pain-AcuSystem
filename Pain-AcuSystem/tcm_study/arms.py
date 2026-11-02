from __future__ import annotations


import hashlib
import json
import os
import re
import time
import urllib.request
from collections import Counter
from pathlib import Path

from tcm_study import schema as S
from tcm_study.schema import SYSTEM_PROMPT_V3

EVIDENCE_WRAPPER = (
    "\n\n【检索证据】\n{evidence}\n\n"
    "请结合上述检索证据与病例事实作答；二者冲突时以病例事实为准，不得编造病例中没有的信息。"
    "特别注意：检索证据中的【鉴别诊断】用于判断本病例还可能是哪一些具体疾病；应优先选择与病例表现最匹配的具体病名，并说明关键区分点。"
    "检索证据中的【危险信号】是病种层面的风险知识，不等于本患者已经出现；"
    "只有病例原文已记录相应征象时才可写成当前【急危重症警示】，否则只作为需优先排除或随访观察依据。"
)
CARDS_WRAPPER = ("\n\n【第一步输出】\n{stage1}\n\n{cards}\n\n"
                 "请以第一步输出为基础完成最终方案。安全信息优先级固定为：病例事实/【病例安全核查】 > 穴位与手法硬安全约束 > 【指南安全提示卡】病种背景 > 模型自由补充。"
                 "【西医诊断】【中医诊断】【证型】【取穴】保持第一步不变；【诊疗思路】可以依据鉴别诊断卡补充或校正鉴别推理，但不得改变已经锁定的诊断、证型和取穴。对于病例中有诊断意义且与拟诊典型表现不一致的阴性体征，应在原体征后用括号说明其临床意义；括号解释必须能由病例事实或检索证据支持。"
                 "若提供【鉴别诊断卡】，在【诊疗思路】末段从其中选择与病例最相关的2–4个具体疾病名，写成‘需鉴别：疾病A（关键区分点：……）；疾病B（关键区分点：……）’，"
                 "不得只写疾病类别或‘其他疾病/相关病变/骨折等’；只有病例已有明确反证时才可写已排除，否则写‘尚需排除/需鉴别’。"
                 "【操作】必须固定写成两部分：‘1. 手法与疗程：…’和‘2. 逐穴操作：’，每个穴位单独一行，格式为‘- 穴位（左侧/右侧/双侧）：定位…。刺法与深度…。必要安全提示…。’；任督脉及其他中线穴不得标成双侧；单侧肢体病变的局部穴及同一患肢远端经穴优先取患侧，跨肢体/全身调节穴默认双侧，对侧取穴仅在病例或原医嘱明确时使用；最终侧别由确定性规则层按病例病侧与穴位解剖位置统一校正；"
                 "标准穴位的定位与深度以穴位安全卡为准，深度不得超过卡片上限，禁深刺或解剖危险区必须写明；阿是穴须结合病例的疼痛/压痛部位和侧别具体定位，不得写‘无固定位置’；不得另写‘逐穴针刺规范’造成重复；"
                 "【注意事项】不重复鉴别诊断，固定写四项：1.禁忌与特殊人群；2.建议完善检查；3.操作前后注意事项；4.患者教育。"
                 "【病例安全核查】列出的当前急危征象必须在四项之前以【急危重症警示】写明，严禁改写为‘无’；核查写明未记录急危征象时可写【急危重症警示】无。；"
                 "【指南安全提示卡】只作为病种级检查建议、禁忌与治疗边界，除非【病例安全核查】已经命中，否则不得把其中危险信号升级成患者当前急危警示；"
                 "穴位/手法安全卡中的深度、禁深刺、危险区、妊娠禁忌属于硬约束。核查写明妊娠相关禁忌不适用时不得写妊娠；列出具体妊娠禁用穴时只写这些穴，不得自行增列。"
                 "仍按九个栏目完整输出，不得新增栏目。")

_SYSTEM_PROMPTS = {"dataset": None, "v3": SYSTEM_PROMPT_V3}
_LOOP = re.compile(r"(.{4,60}?)(?:\1){4,}", re.S)          # 同一 4–60 字片段连续出现 ≥5 次
_EMPTY_LABELS = re.compile(r"[【〔]\S{2,8}[】〕]\s*(?:\n|$)")  # 只有栏目名没有内容
_WIN, _WIN_REPEATS = 12, 4


def compose_messages(messages: list[dict], evidence: str | None, system_prompt: str = "dataset") -> list[dict]:
    msgs = [dict(m) for m in messages if m["role"] in ("system", "user")]
    sp = _SYSTEM_PROMPTS.get(system_prompt, None)
    if sp is not None:
        if msgs and msgs[0]["role"] == "system":
            msgs[0] = {"role": "system", "content": sp}
        else:
            msgs.insert(0, {"role": "system", "content": sp})
    if evidence:
        msgs[-1] = {**msgs[-1], "content": msgs[-1]["content"] + EVIDENCE_WRAPPER.format(evidence=evidence)}
    return msgs


def compose_stage2(messages: list[dict], evidence: str | None, stage1_text: str, cards: str, system_prompt: str = "dataset") -> list[dict]:
    """Step-2 prompt = step-1 prompt + step-1 output + cards (single turn; identical for every RAG arm and for RAFT samples)."""
    msgs = compose_messages(messages, evidence, system_prompt)
    msgs[-1] = {**msgs[-1], "content": msgs[-1]["content"] + CARDS_WRAPPER.format(stage1=stage1_text.strip(), cards=cards.strip() or "【穴位安全卡】无（第一步所列穴位不在标准穴位库中）")}
    return msgs


_DIFF_LINE = re.compile(r"需鉴别[：:]|需(?:优先)?排除[：:]")


def _diff_lines(notes: str) -> list[str]:
    return [ln.strip() for ln in (notes or "").splitlines() if _DIFF_LINE.search(ln)]


def _dx_name(x: str) -> str:
    return re.split(r"[（(]", x or "", 1)[0].strip(" 。；;,，")


def _card_candidate_names(dlog: dict | None) -> list[str]:
    out = []
    for x in list((dlog or {}).get("dd") or []) + list((dlog or {}).get("red") or []):
        n = _dx_name(str(x))
        if n and n not in out:
            out.append(n)
    return out


def _same_dx_name(a: str, b: str) -> bool:
    a, b = _dx_name(a), _dx_name(b)
    if not a or not b:
        return False
    if a == b:
        return True
    return min(len(a), len(b)) >= 4 and (a in b or b in a)


def _pred_diff_names_from_notes(notes: str) -> list[str]:
    """轻量解析明确的需鉴别/需排除病名；仅用于 stage2 选择与审计，不替代 pilot 指标。"""
    out = []
    for line in (notes or "").splitlines():
        if not _DIFF_LINE.search(line):
            continue
        for ch in re.split(r"(?=需鉴别[：:]|需(?:优先)?排除[：:])", line):
            m = re.match(r"(?:需鉴别|需(?:优先)?排除)[：:](.*)", ch.strip())
            if not m:
                continue
            for item in re.split(r"[；;]", m.group(1)):
                n = re.split(r"[，,：:]", _dx_name(item), 1)[0].strip()
                if 2 <= len(n) <= 24 and n not in out:
                    out.append(n)
    return out


def _card_hit_stats(text: str, dlog: dict | None) -> tuple[int, int, float]:
    """(卡片候选命中数, 输出鉴别病名数, 卡片内precision代理)。"""
    sec = S.parse_bracket_sections(text, top_level_only=True)
    pred = _pred_diff_names_from_notes(((sec.get("诊疗思路") or "") + "\n" + (sec.get("注意事项") or "")))
    cand = _card_candidate_names(dlog)
    hits = sum(1 for c in cand if any(_same_dx_name(c, p) for p in pred))
    proxy_p = hits / len(pred) if pred else 0.0
    return hits, len(pred), proxy_p


def _restore_stage1_card_differentials(stage1_text: str, final_text: str, dlog: dict | None, min_specific: int = 2) -> tuple[str, dict]:
   
    before_hits, _, _ = _card_hit_stats(final_text, dlog)
    need = min(max(1, int(min_specific)), len(_card_candidate_names(dlog)))
    if need <= 0 or before_hits >= need:
        return final_text, {"applied": False, "reason": "enough_specific", "hits_before": before_hits, "hits_after": before_hits}
    s1 = S.parse_bracket_sections(stage1_text, top_level_only=True)
    sf = S.parse_bracket_sections(final_text, top_level_only=True)
    n1 = ((s1.get("诊疗思路") or "") + "\n" + (s1.get("注意事项") or "")).strip()
    nf = ((sf.get("诊疗思路") or "") + "\n" + (sf.get("注意事项") or "")).strip()
    candidates = _card_candidate_names(dlog)
    existing = _pred_diff_names_from_notes(nf)
    recovered = []
    # 从 stage1 的鉴别分句中逐项恢复，避免整行带回与卡片无关的病名。
    for line in _diff_lines(n1):
        for ch in re.split(r"(?=需鉴别[：:]|需(?:优先)?排除[：:])", line):
            m = re.match(r"(需鉴别|需(?:优先)?排除)[：:](.*)", ch.strip())
            if not m:
                continue
            label, body = m.group(1), m.group(2)
            for item in re.split(r"[；;]", body):
                item = item.strip(" 。；;")
                if not item:
                    continue
                nm = re.split(r"[，,：:]", _dx_name(item), 1)[0].strip()
                if not any(_same_dx_name(nm, c) for c in candidates):
                    continue
                if any(_same_dx_name(nm, e) for e in existing + [x[1] for x in recovered]):
                    continue
                recovered.append((label, nm, item))
                if before_hits + len(recovered) >= need:
                    break
            if before_hits + len(recovered) >= need:
                break
        if before_hits + len(recovered) >= need:
            break
    if not recovered:
        return final_text, {"applied": False, "reason": "no_stage1_supported_candidate", "hits_before": before_hits, "hits_after": before_hits}
    groups = {"需鉴别": [], "需优先排除": []}
    for label, _, item in recovered:
        groups["需优先排除" if "排除" in label else "需鉴别"].append(item)
    add_lines = []
    for label in ("需鉴别", "需优先排除"):
        if groups[label]:
            add_lines.append(label + "：" + "；".join(groups[label]) + "。")
    base_reason = (sf.get("诊疗思路") or "").strip()
    sf["诊疗思路"] = (base_reason + (" " if base_reason else "") + "".join(add_lines)).strip()
    out = S.render_sections({k: v for k, v in sf.items() if not k.startswith("_")}, extra=())
    after_hits, _, _ = _card_hit_stats(out, dlog)
    return out, {"applied": True, "reason": "restore_stage1_model_candidates", "hits_before": before_hits, "hits_after": after_hits, "recovered": [x[1] for x in recovered]}


def _merge_stage2_notes(stage1_notes: str, stage2_notes: str) -> tuple[str, bool]:
    
    a, b = (stage1_notes or "").strip(), (stage2_notes or "").strip()
    if not b:
        return a, False
    if _diff_lines(b) or not _diff_lines(a):
        return b, False
    lines = b.splitlines()
    insert_at = 1 if lines and lines[0].startswith(S.EMERGENCY_TAG) else 0
    for ln in reversed(_diff_lines(a)):
        lines.insert(insert_at, ln)
    return "\n".join(lines).strip(), True


def ensure_differential_from_card(text: str, dlog: dict | None, min_specific: int = 2) -> tuple[str, dict]:
    
    dlog = dict(dlog or {})
    dd = [str(x).strip() for x in (dlog.get("dd") or []) if str(x).strip()]
    red = [str(x).strip() for x in (dlog.get("red") or []) if str(x).strip()]
    expected = dd + red
    if not expected:
        return text, {"applied": False, "reason": "no_candidates", "hits_before": 0}
    def name(x: str) -> str:
        return re.split(r"[（(]", x, 1)[0].strip()
    sec = S.parse_bracket_sections(text, top_level_only=True)
    notes = (sec.get("注意事项") or "").strip()
    reasoning = (sec.get("诊疗思路") or "").strip()
    diff_text = reasoning + "\n" + notes
    hits = [name(x) for x in expected if name(x) and name(x) in diff_text]
    need = min(max(1, int(min_specific)), len(expected))
    if len(set(hits)) >= need:
        return text, {"applied": False, "reason": "enough_specific", "hits_before": len(set(hits))}
    inserts = []
    if dd:
        inserts.append("需鉴别：" + "；".join(dd[:3]) + "。")
    if red:
        inserts.append("需优先排除：" + red[0] + "。")
    keep_reason = [ln for ln in reasoning.splitlines() if not _DIFF_LINE.search(ln)]
    sec["诊疗思路"] = "\n".join(x for x in keep_reason if x.strip()).strip()
    if sec["诊疗思路"] and inserts:
        sec["诊疗思路"] += " "
    sec["诊疗思路"] += "".join(inserts)
    return S.render_sections(sec, extra=()), {
        "applied": True, "reason": "insufficient_specific", "hits_before": len(set(hits)),
        "dd_added": dd[:3], "red_added": red[:1],
    }


def _merge_stage2_reasoning(stage1_reasoning: str, stage2_reasoning: str) -> tuple[str, bool]:
    """v18.1: stage2 may use the differential card to refine reasoning; if it drops stage1 explicit differential lines, keep them."""
    a, b = (stage1_reasoning or "").strip(), (stage2_reasoning or "").strip()
    if not b:
        return a, False
    da, db = _diff_lines(a), _diff_lines(b)
    if db or not da:
        return b, False
    return (b.rstrip() + " " + "".join(x if x.endswith("。") else x + "。" for x in da)).strip(), True


def splice_stage2(stage1_text: str, stage2_text: str, stage2_truncated: bool = False) -> tuple[str, dict]:
    
    s1, s2 = S.parse_bracket_sections(stage1_text, top_level_only=True), S.parse_bracket_sections(stage2_text, top_level_only=True)
    sec = {k: (s1.get(k) or "").strip() for k in S.SECTIONS_V3}
    taken, kept = [], []
    diff_restored = False
    reasoning_diff_restored = False
    for k in ("诊疗思路", "操作", "注意事项"):
        v = (s2.get(k) or "").strip()
        if not v or v == "无":
            continue
        stub = len(v) < 15 and len(sec[k]) >= 15
        trunc = stage2_truncated and k in ("诊疗思路", "注意事项") and len(v) < len(sec[k])
        if stub or trunc:
            kept.append({"field": k, "reason": "stub" if stub else "stage2_truncated", "stage2_len": len(v), "stage1_len": len(sec[k])})
            continue
        if k == "诊疗思路":
            v, reasoning_diff_restored = _merge_stage2_reasoning(sec[k], v)
        elif k == "注意事项":
            v, diff_restored = _merge_stage2_notes(sec[k], v)
        sec[k] = v
        taken.append(k)
    info = {"stage2_fields_used": taken, "stage1_fields_kept": kept, "stage1_differential_restored": bool(diff_restored or reasoning_diff_restored),
            "stage2_points_changed": _norm_pts(s1.get("取穴")) != _norm_pts(s2.get("取穴")) if s2.get("取穴") else None}
    return S.render_sections(sec, extra=()), info


def _norm_pts(s: str | None) -> str:
    return re.sub(r"[（(].*?[)）]|[\s、，,；;]", "", s or "")


def prompt_template_text(system_prompt: str = "dataset") -> str:

    sp = _SYSTEM_PROMPTS.get(system_prompt, None)
    return EVIDENCE_WRAPPER + "\n" + CARDS_WRAPPER + ("\n\n[SYSTEM:" + system_prompt + "]\n" + sp if sp else "")


def sha(msgs: list[dict]) -> str:
    return hashlib.sha256(json.dumps(msgs, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def is_degenerate(text: str) -> str | None:

    t = text or ""
    if len(t.strip()) < 12:
        return "too_short"
    t_norm = re.sub(r"[\d\s．.、)）]+", "", t)
    for m in _LOOP.finditer(t_norm):
        if len(set(m.group(1))) >= 3:
            return "repetition_loop"
   
    if len(t_norm) >= 300:
        
        t_win = re.sub(r"[\d\s．.、)）]+", "", re.sub(r"【(?:操作|注意事项|穴位安全卡|手法卡)】[^【]*", "", t))
        win = Counter(t_win[i:i + _WIN] for i in range(0, max(1, len(t_win) - _WIN)))
        frag, n = win.most_common(1)[0] if win else ("", 0)
        if n >= _WIN_REPEATS and len(set(frag)) >= 5 and not re.fullmatch(r"[（(]?(?:双侧|左侧|右侧|患侧)[)）]?[、，]?", frag[:6]):
            return "repetition_loop"
    labels = _EMPTY_LABELS.findall(t)
    if len(labels) >= 3 and len(re.sub(r"[【〔]\S{2,8}[】〕]", "", t).strip()) < 30:
        return "empty_labels"
    return None


def _retry_reason(text: str, n_new: int | None, max_new: int) -> str | None:
    r = is_degenerate(text)
    if r:
        return r
    if n_new is not None and max_new and n_new >= max_new and not re.search(r"【注意事项】[^【]{20,}", text or ""):
        return "hit_max_tokens"
    return None


DEFAULT_THINKING = {"enabled": False, "budget_tokens": 4096, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 20260920,
                    "early_stop_text": "\n\n思考时间有限，现在根据以上思考直接给出最终方案。\n</think>\n\n"}


def thinking_cfg(cfg_or_block: dict | None) -> dict:
    
    b = cfg_or_block or {}
    if "generation" in b:
        b = (b["generation"] or {}).get("thinking") or {}
    return {**DEFAULT_THINKING, **b}


def _case_seed(base: int, prompt: str, attempt: int) -> int:
   
    return (int(base) + int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16) + 7919 * attempt) % (2 ** 31 - 1)


class HFGenerator:
   

    def __init__(self, model_path: str, adapter: str | None, max_new_tokens: int, repetition_penalty: float, four_bit: bool = True,
                 loop_retry: bool = True, retry_temperature: float = 0.7, no_repeat_ngram_size: int = 0,
                 chat_template_kwargs: dict | None = None, thinking: dict | None = None, merge_lora: bool = False):
        import torch
        from transformers import AutoTokenizer, BitsAndBytesConfig
        from tcm_pipeline.common import compute_dtype, load_local_causal_lm
        self.torch = torch
        cuda = torch.cuda.is_available()
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype(True),
                                   bnb_4bit_use_double_quant=True) if (cuda and four_bit) else None
        self.tok = AutoTokenizer.from_pretrained(adapter or model_path, trust_remote_code=True, local_files_only=True)
        if adapter:
            ac = Path(adapter, "adapter_config.json")
            old_base = json.loads(ac.read_text(encoding="utf-8")).get("base_model_name_or_path") if ac.is_file() else None
            if old_base and Path(old_base).name != Path(model_path).name:
                raise SystemExit(f"[STOP] LoRA {adapter} 是在 {old_base} 上训练的，与当前基座 {model_path} 不匹配。请先用新基座重训（run_train_bg.sh）。")
        model = load_local_causal_lm(model_path, dtype=compute_dtype(True), trust_remote_code=True, local_files_only=True,
                                     quantization_config=quant, device_map={"": 0} if cuda else None, low_cpu_mem_usage=True)
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
            if merge_lora and quant is None:
                
                model = model.merge_and_unload()
        self.precision = "nf4" if quant is not None else "bf16"
        self._setup(model, max_new_tokens, repetition_penalty, loop_retry, retry_temperature, no_repeat_ngram_size, chat_template_kwargs, thinking)

    def _setup(self, model, max_new_tokens, repetition_penalty, loop_retry, retry_temperature, no_repeat_ngram_size, chat_template_kwargs, thinking):
        """与加载无关的初始化（单测里用假模型直接调用）。"""
        self.model = model.eval()
        self.chat_kwargs = dict(chat_template_kwargs or {})
        self.think = thinking_cfg(thinking)
        if self.chat_kwargs.get("enable_thinking") is False:
            self.think["enabled"] = False
        gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        eos = gen_eos if gen_eos is not None else self.tok.eos_token_id
        self.eos = [eos] if isinstance(eos, int) else list(eos)
        self.think_end = None
        if self.think["enabled"]:
            tid = self.tok.convert_tokens_to_ids("</think>")
            if not isinstance(tid, int) or tid < 0 or tid == self.tok.unk_token_id:
                raise SystemExit("[ERROR] 分词器里没有 </think>：该基座不支持思考模式，请把 generation.thinking.enabled 设为 false")
            self.think_end = tid
            self.early_ids = self.tok(self.think["early_stop_text"], add_special_tokens=False)["input_ids"]
        self.max_new_tokens, self.rep = int(max_new_tokens), float(repetition_penalty)
        self.loop_retry, self.retry_t, self.nrng = bool(loop_retry), float(retry_temperature), int(no_repeat_ngram_size)
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos[0]

  
    def _gen_plain(self, ids, sample: bool, cap: int | None = None):
        kw = dict(max_new_tokens=int(cap or self.max_new_tokens), repetition_penalty=self.rep, eos_token_id=self.eos, pad_token_id=self.pad)
        if self.nrng > 0:
            kw["no_repeat_ngram_size"] = self.nrng
        kw.update(dict(do_sample=True, temperature=self.retry_t, top_p=0.9) if sample else dict(do_sample=False))
        with self.torch.no_grad():
            out = self.model.generate(ids, **kw)
        text = self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.S)   # 万一模型仍输出思考块，只保留正式回答
        return {"text": text, "n_new": int(out.shape[1] - ids.shape[1]), "reasoning": None, "reasoning_tokens": 0, "think_budget_hit": False}

    # ---- 思考：两阶段 ---------------------------------------------------------------------------------------------
    def _sample_kw(self, max_new: int, eos: list[int]) -> dict:
        th = self.think
        kw = dict(do_sample=True, temperature=float(th["temperature"]), top_p=float(th["top_p"]), top_k=int(th["top_k"]),
                  max_new_tokens=int(max_new), repetition_penalty=self.rep, eos_token_id=eos, pad_token_id=self.pad)
        if self.nrng > 0:
            kw["no_repeat_ngram_size"] = self.nrng
        return kw

    def _gen_think(self, ids, seed: int, cap: int | None = None):
        t = self.torch
        t.manual_seed(seed)
        if t.cuda.is_available():
            t.cuda.manual_seed_all(seed)
        L = ids.shape[1]
        with t.no_grad():
            out1 = self.model.generate(ids, **self._sample_kw(self.think["budget_tokens"], [self.think_end] + self.eos))
        new1 = out1[0, L:].tolist()
        budget_hit, closed = False, bool(new1) and new1[-1] == self.think_end
        if closed:
            reasoning_ids, seq = new1[:-1], out1
        elif new1 and new1[-1] in self.eos:
           
            text = self.tok.decode(new1, skip_special_tokens=True)
            return {"text": text.strip(), "n_new": len(new1), "reasoning": "", "reasoning_tokens": 0, "think_budget_hit": False, "think_unclosed": True}
        else:
            budget_hit, reasoning_ids = True, new1
            seq = t.cat([out1, t.tensor([self.early_ids], device=out1.device, dtype=out1.dtype)], dim=1)
        with t.no_grad():
            out2 = self.model.generate(seq, **self._sample_kw(int(cap or self.max_new_tokens), self.eos))
        ans_ids = out2[0, seq.shape[1]:]
        text = self.tok.decode(ans_ids, skip_special_tokens=True)
        text = re.sub(r"^\s*(?:<think>.*?)?</think>\s*", "", text, flags=re.S).lstrip()
        return {"text": text, "n_new": int(ans_ids.shape[0]), "reasoning": self.tok.decode(reasoning_ids, skip_special_tokens=True).strip(),
                "reasoning_tokens": len(reasoning_ids), "think_budget_hit": budget_hit}

    def generate(self, messages: list[dict], max_new_tokens: int | None = None) -> dict:
        """max_new_tokens：本次调用的答案上限（RAG 臂第二步用 generation.max_new_tokens_stage2）；默认用构造时的值。"""
        t = self.torch
        cap = int(max_new_tokens or self.max_new_tokens)
        
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **self.chat_kwargs)
        ids = self.tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.model.device)
        if t.cuda.is_available():
            t.cuda.reset_peak_memory_stats()
        t0 = time.time()
        if self.think["enabled"]:
            r = self._gen_think(ids, _case_seed(self.think["seed"], prompt, 0), cap)
        else:
            r = self._gen_plain(ids, sample=False, cap=cap)
        reason, retried = _retry_reason(r["text"], r["n_new"], cap), False
        if reason and self.loop_retry:
           
            if self.think["enabled"]:
                r2 = self._gen_think(ids, _case_seed(self.think["seed"], prompt, 1), cap)
            else:
                t.manual_seed(20260920)
                r2 = self._gen_plain(ids, sample=True, cap=cap)
            if not _retry_reason(r2["text"], r2["n_new"], cap):
                r, retried = r2, True
        return {"text": r["text"], "reasoning": r["reasoning"], "reasoning_tokens": r["reasoning_tokens"], "think_budget_hit": r["think_budget_hit"],
                "think_unclosed": bool(r.get("think_unclosed")),
                "latency_s": round(time.time() - t0, 2), "prompt_tokens": int(ids.shape[1]), "new_tokens": r["n_new"],
                "max_new_tokens": cap, "hit_max_tokens": bool(r["n_new"] is not None and r["n_new"] >= cap),
                "peak_vram_gb": round(t.cuda.max_memory_allocated() / 1024 ** 3, 2) if t.cuda.is_available() else None,
                "degenerate_first_pass": reason, "retried": retried, "still_degenerate": bool(_retry_reason(r["text"], r["n_new"], cap))}

    def close(self) -> None:
        import gc
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class OpenAICompat:
    
    def __init__(self, base_url: str, model: str, api_key_env: str, max_new_tokens: int, loop_retry: bool = True, retry_temperature: float = 0.7,
                 extra_body: dict | None = None, thinking: dict | None = None, stream: bool = False):
        if not model:
            raise SystemExit("[ERROR] arms.A5.model is empty in configs/study.yaml")
        from tcm_study.net import force_ipv4
        force_ipv4()
        self.url, self.model = base_url.rstrip("/") + "/chat/completions", model
        self.key, self.max_new_tokens = os.environ.get(api_key_env, "EMPTY"), int(max_new_tokens)
        self.loop_retry, self.retry_t = bool(loop_retry), float(retry_temperature)
        self.extra = dict(extra_body or {})
        self.think = thinking_cfg(thinking)
        self.stream = bool(stream)

    def _body(self, messages, attempt: int, cap: int | None = None) -> dict:
        cap = int(cap or self.max_new_tokens)
        b = {"model": self.model, "messages": messages}
        if self.think["enabled"]:
            th = self.think
            prompt = json.dumps(messages, ensure_ascii=False, sort_keys=True)
            b.update(temperature=float(th["temperature"]), top_p=float(th["top_p"]), top_k=int(th["top_k"]),
                     seed=_case_seed(th["seed"], prompt, attempt), max_tokens=int(th["budget_tokens"]) + cap)
        elif attempt == 0:
            b.update(temperature=0, max_tokens=cap)
        else:
            b.update(temperature=self.retry_t, top_p=0.9, seed=20260920, max_tokens=cap)
        b.update(self.extra)
        if self.stream:
            b.update(stream=True, stream_options={"include_usage": True})
        return b

    @staticmethod
    def _read_stream(r) -> dict:
        content, reasoning, usage, finish = [], [], {}, None
        for raw in r:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            d = json.loads(data)
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                if delta.get("content"):
                    content.append(delta["content"])
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
        return {"content": "".join(content), "reasoning": "".join(reasoning), "usage": usage, "finish_reason": finish}

    def _call(self, messages, attempt: int, cap: int | None = None) -> dict:
        body = json.dumps(self._body(messages, attempt, cap)).encode()
        last = None
        for k in range(4):
            try:
                req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
                with urllib.request.urlopen(req, timeout=900) as r:
                    if self.stream:
                        return self._read_stream(r)
                    d = json.loads(r.read().decode())
                ch = d["choices"][0]
                return {"content": ch["message"].get("content") or "", "reasoning": ch["message"].get("reasoning_content") or "",
                        "usage": d.get("usage") or {}, "finish_reason": ch.get("finish_reason")}
            except Exception as e:
                last = e
                if hasattr(e, "read"):
                    try:
                        last = f"{e} {e.read().decode(errors='ignore')[:300]}"
                    except Exception:
                        pass
                time.sleep(3 * (k + 1))
        raise RuntimeError(f"A5 request failed: {last}")

    def _parse(self, d: dict, cap: int | None = None) -> dict:
        cap = int(cap or self.max_new_tokens)
        content, reasoning = d["content"], d["reasoning"]
        m = re.match(r"^\s*(?:<think>)?(.*?)</think>\s*", content, re.S)
        if m and "【西医诊断】" not in m.group(1):
            reasoning, content = (reasoning + m.group(1)).strip(), content[m.end():]
        u = d.get("usage") or {}
        r_tok = (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or u.get("reasoning_tokens")
        comp = u.get("completion_tokens")
        n_ans = (comp - r_tok) if (comp is not None and r_tok is not None) else (None if self.think["enabled"] else comp)
        if d.get("finish_reason") == "length":
            n_ans = cap
        return {"text": content.strip(), "reasoning": reasoning.strip() or None, "reasoning_tokens": r_tok, "n_new": n_ans,
                "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": comp, "finish_reason": d.get("finish_reason")}

    def generate(self, messages: list[dict], max_new_tokens: int | None = None) -> dict:
        t0 = time.time()
        cap = int(max_new_tokens or self.max_new_tokens)
        r = self._parse(self._call(messages, 0, cap), cap)
        reason, retried = _retry_reason(r["text"], r["n_new"], cap), False
        if reason and self.loop_retry:
            r2 = self._parse(self._call(messages, 1, cap), cap)
            if not _retry_reason(r2["text"], r2["n_new"], cap):
                r, retried = r2, True
        return {"text": r["text"], "reasoning": r["reasoning"], "reasoning_tokens": r["reasoning_tokens"],
                "think_budget_hit": None, "think_unclosed": False, "finish_reason": r["finish_reason"],
                "latency_s": round(time.time() - t0, 2), "prompt_tokens": r["prompt_tokens"], "new_tokens": r["n_new"],
                "completion_tokens": r["completion_tokens"], "max_new_tokens": cap,
                "hit_max_tokens": bool(r.get("finish_reason") == "length" or (r["n_new"] is not None and r["n_new"] >= cap)),
                "peak_vram_gb": None, "degenerate_first_pass": reason, "retried": retried,
                "still_degenerate": bool(_retry_reason(r["text"], r["n_new"], cap))}

    def close(self) -> None:
        pass


def generate_two_stage(gen, messages: list[dict], evidence: str, rag, lex, system_prompt: str = "dataset",
                       stage2_max_new_tokens: int | None = None,
                       apply_differential_guard: bool | None = None) -> dict:
  
    m1 = compose_messages(messages, evidence, system_prompt)
    g1 = gen.generate(m1)
    sec1 = S.parse_bracket_sections(g1["text"], top_level_only=True)
    points = []
    for it in lex.extract(sec1.get("取穴", "")):
        if it["status"] in ("standard", "alias"):
            points.append(it["canonical"])
        elif it["status"] == "whitelist":
            points.append(it["token"])
    manips = rag.manip_names_in((sec1.get("操作") or "") + "\n" + (sec1.get("取穴") or ""))
    cards, clog = rag.cards(points, manips)
    dlog = {}
    if getattr(rag, "cards_cfg", {}).get("safety_check", False):
        
        from tcm_study.safety_layer import guideline_safety_card, safety_check_card
        inp = "\n".join(m["content"] for m in messages if m["role"] == "user").split("\n\n【检索证据】", 1)[0]   # 只用病例本身
        card = safety_check_card(lex, inp, [p for p in points if p in lex.rows])
        cards = (cards.strip() + "\n\n" + card) if cards.strip() else card
        if getattr(rag, "cards_cfg", {}).get("differential_card", False):
            dcard, dlog = rag.differential_card(sec1.get("西医诊断", ""), inp)
            if dcard:
                cards = cards.strip() + "\n\n" + dcard
                clog = {**clog, "differential": dlog}
        if getattr(rag, "cards_cfg", {}).get("guideline_safety", False):
            gcard = guideline_safety_card(evidence, int(rag.cards_cfg.get("guideline_safety_max_chars", 1800)), include_red=False)
            if gcard:
                cards = cards.strip() + "\n\n" + gcard
        clog = {**clog, "safety_check": True, "cards_sha256": __import__("hashlib").sha256(cards.encode("utf-8")).hexdigest()}
    m2 = compose_stage2(messages, evidence, g1["text"], cards, system_prompt)
    g2 = gen.generate(m2, max_new_tokens=stage2_max_new_tokens) if stage2_max_new_tokens else gen.generate(m2)
    final, info = splice_stage2(g1["text"], g2["text"], stage2_truncated=bool(g2.get("hit_max_tokens")))

    
    min_hits = int(getattr(rag, "cards_cfg", {}).get("differential_min_model_hits", 2) or 2)
    final, restore_info = _restore_stage1_card_differentials(g1["text"], final, dlog, min_specific=min_hits)
    info = {**info, "stage1_card_differential_restore": restore_info}

    retry_info = {"attempted": False, "used": False}
    if bool(getattr(rag, "cards_cfg", {}).get("differential_retry_once", False)) and len(_card_candidate_names(dlog)) >= min_hits:
        h0, n0, pp0 = _card_hit_stats(final, dlog)
        if h0 < min_hits:
            retry_info.update({"attempted": True, "hits_before": h0, "pred_before": n0, "card_precision_before": round(pp0, 4)})
            m2r = [dict(x) for x in m2]
           
            top_k = int(getattr(rag, "cards_cfg", {}).get("differential_retry_top_k", 3) or 3)
            top_names = _card_candidate_names(dlog)[:max(min_hits, top_k)]
            max_pred = int(getattr(rag, "cards_cfg", {}).get("differential_retry_max_predictions", 3) or 3)
            cand_text = "、".join(top_names)
            reminder = (
                "\n\n【鉴别覆盖复核】上一版第二步对【鉴别诊断卡】覆盖不足。请重新完整输出九栏；"
                f"本次只允许围绕以下高相关候选作鉴别：{cand_text}。"
                f"【诊疗思路】末段优先保留其中与病例最匹配的{min_hits}个具体疾病名，并各写一个关键区分点或所需检查；"
                f"具体鉴别病名总数不得超过{max_pred}个。"
                "若现有资料尚不能确认，只能写‘需鉴别/需优先排除’，不要把候选写成已确诊；"
                "不得加入上述候选之外的疾病。其余诊断、证型、取穴保持第一步不变。"
            )
            m2r[-1] = {**m2r[-1], "content": m2r[-1]["content"] + reminder}
            g2r = gen.generate(m2r, max_new_tokens=stage2_max_new_tokens) if stage2_max_new_tokens else gen.generate(m2r)
            fr, ir = splice_stage2(g1["text"], g2r["text"], stage2_truncated=bool(g2r.get("hit_max_tokens")))
            fr, rr = _restore_stage1_card_differentials(g1["text"], fr, dlog, min_specific=min_hits)
            h1, n1, pp1 = _card_hit_stats(fr, dlog)
          
            min_card_p = float(getattr(rag, "cards_cfg", {}).get("differential_retry_min_card_precision", 0.80) or 0.80)
            precision_floor = max(min_card_p, pp0)
            use_retry = (
                h1 > h0
                and n1 > 0
                and pp1 >= precision_floor
                and n1 <= max_pred
                and not g2r.get("still_degenerate")
            )
            retry_info.update({
                "hits_after": h1, "pred_after": n1,
                "card_precision_after": round(pp1, 4),
                "precision_floor": round(precision_floor, 4),
                "max_predictions": max_pred,
                "used": bool(use_retry),
            })
            if use_retry:
                g2, final = g2r, fr
                info = {**ir, "stage1_card_differential_restore": rr}
                m2 = m2r
    info = {**info, "differential_model_retry": retry_info}

    model_final = final
    use_guard = getattr(rag, "cards_cfg", {}).get("differential_guard", False) if apply_differential_guard is None else bool(apply_differential_guard)
    if use_guard:
        final, dg = ensure_differential_from_card(final, dlog, min_specific=min_hits)
        info = {**info, "differential_guard": dg}
    else:
        info = {**info, "differential_guard": {"applied": False, "reason": "disabled_for_research_comparison"}}
    return {"text": final, "text_pre_guard": model_final, "differential_log": dlog, "stage1_output": g1["text"], "stage2_output": g2["text"], "cards": cards, "cards_log": clog, "splice": info,
            "stage2_max_new_tokens": g2.get("max_new_tokens"), "stage2_hit_max_tokens": bool(g2.get("hit_max_tokens")),
            "prompt_sha256": sha(m1), "stage2_prompt_sha256": sha(m2),
            "latency_s": round(g1["latency_s"] + g2["latency_s"], 2), "latency_stage1_s": g1["latency_s"], "latency_stage2_s": g2["latency_s"],
            "prompt_tokens": g1.get("prompt_tokens"), "stage2_prompt_tokens": g2.get("prompt_tokens"),
            "new_tokens": (g1.get("new_tokens") or 0) + (g2.get("new_tokens") or 0),
            "peak_vram_gb": max(x for x in (g1.get("peak_vram_gb"), g2.get("peak_vram_gb")) if x is not None) if (g1.get("peak_vram_gb") or g2.get("peak_vram_gb")) else None,
            "degenerate_first_pass": g1.get("degenerate_first_pass"), "retried": bool(g1.get("retried")),
            "stage2_degenerate_first_pass": g2.get("degenerate_first_pass"), "stage2_retried": bool(g2.get("retried")),
            "still_degenerate": bool(g1.get("still_degenerate") or g2.get("still_degenerate")),
          
            "stage1_reasoning": g1.get("reasoning"), "stage2_reasoning": g2.get("reasoning"),
            "reasoning_tokens": g1.get("reasoning_tokens"), "stage2_reasoning_tokens": g2.get("reasoning_tokens"),
            "think_budget_hit": g1.get("think_budget_hit"), "stage2_think_budget_hit": g2.get("think_budget_hit"),
            "think_unclosed": bool(g1.get("think_unclosed") or g2.get("think_unclosed"))}
