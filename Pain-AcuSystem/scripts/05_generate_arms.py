#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tcm_study.acupoints import AcupointLexicon
from tcm_study import schema as S
from tcm_study.needling import correct_pregnancy_lists
from tcm_study.safety_layer import apply as rule_layer
from tcm_study.arms import HFGenerator, OpenAICompat, compose_messages, generate_two_stage, sha, thinking_cfg
from tcm_study.config import load_cfg
from tcm_study.package import CASE_ID_RE, Package
from tcm_study.rag import RagEvidence, RagIndex
from tcm_study.state import StudyState

RAG_ARMS = {"A2", "A4", "A5"}
KEEP = ("max_new_tokens", "hit_max_tokens", "stage2_max_new_tokens", "stage2_hit_max_tokens", "reasoning_tokens", "stage2_reasoning_tokens", "think_budget_hit", "stage2_think_budget_hit", "think_unclosed", "finish_reason",
        "latency_s", "latency_stage1_s", "latency_stage2_s", "prompt_tokens", "stage2_prompt_tokens", "new_tokens", "peak_vram_gb",
        "degenerate_first_pass", "retried", "stage2_degenerate_first_pass", "stage2_retried", "still_degenerate")


def load_items(cfg, pkg, which):
    if which == "initial":
        return [{"case_id": r["case_id"], "messages": r["messages"], "reference": r["reference"]} for r in pkg.test_initial()]
    if which == "followup":
        return [{"case_id": r["case_id"], "messages": r["messages"], "reference": r["reference"]} for r in pkg.test_followup()]
    raw = pd.read_excel(pkg.files["pilot"], sheet_name="", header=3, dtype=str, keep_default_na=False)
    ids = [x for x in raw["case_id"] if CASE_ID_RE.match(x)]
    train = {r["meta"]["case_id"]: r for r in pkg.train() if r["meta"].get("task") == "initial"}
    test_ids = set(pkg.testlist()["case_id"])
    assert not (set(ids) & test_ids), "pilot cases must not overlap the test set"
    return [{"case_id": i, "messages": train[i]["messages"][:-1], "reference": train[i]["messages"][-1]["content"]} for i in ids if i in train]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/study.yaml")
    ap.add_argument("--set", choices=["initial", "followup", "pilot"], default="initial")
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reason", default="")
    ap.add_argument("--evidence-only", action="store_true",
                    help="只建检索证据 evidence.jsonl 后退出（多卡并行生成前先跑一次，避免几个进程同时写同一个证据文件）")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    pkg = Package(cfg["paths"]["package_dir"])
    work = Path(cfg["paths"]["work_dir"])
    state = StudyState(work)
    fu_arms = list(cfg["rating"].get("followup_arms") or ["A4", "A5"])
    default_arms = {"initial": ["A1", "A2", "A3", "A4", "A5", "A6"], "followup": fu_arms + ["A6"], "pilot": ["A5"]}[a.set]
    arms = a.arms or default_arms
    if a.set == "initial":
        state.require("frozen"); state.forbid("generated", force=a.force, reason=a.reason)
    elif a.set == "followup":
        state.require("locked")
    if a.set != "pilot":
        import subprocess
        r = subprocess.run([sys.executable, "scripts/04_freeze_study.py", "--config", a.config, "--verify"], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(r.stdout + r.stderr)
    items = load_items(cfg, pkg, a.set)
    out = work / "outputs" / a.set
    out.mkdir(parents=True, exist_ok=True)
    gc = cfg["generation"]
    two_stage = bool(gc.get("two_stage_rag", True))
    print(f"[GEN] set={a.set} n={len(items)} arms={arms} two_stage_rag={two_stage}")

    def done(arm):
        p = out / f"raw_{arm}.jsonl"
        return {json.loads(l)["case_id"] for l in p.open(encoding="utf-8")} if p.exists() else set()

    evid, rag, lex = {}, None, None
    ep = out / "evidence.jsonl"
    if ep.exists():
        evid = {d["case_id"]: (d["evidence"], d["log"]) for d in map(json.loads, ep.open(encoding="utf-8"))}
    if set(arms) & RAG_ARMS:
        rag = RagEvidence(RagIndex(cfg["outputs"]["rag_index"], cfg["rag"]["device"]), cfg["rag"])
        lex = AcupointLexicon(pkg.files["acupoint_lib"])
        if any(i["case_id"] not in evid for i in items):
            with ep.open("a", encoding="utf-8") as f:
                for it in items:
                    if it["case_id"] not in evid:
                        text, log = rag.build(RagEvidence.query_from_messages(it["messages"]))
                        evid[it["case_id"]] = (text, log)
                        f.write(json.dumps({"case_id": it["case_id"], "evidence": text, "log": log}, ensure_ascii=False) + "\n")

    if a.evidence_only:
        print(f"[GEN] 检索证据就绪：{sum(1 for i in items if i['case_id'] in evid)}/{len(items)} -> {ep}")
        return
    sys_prompt = gc.get("system_prompt", "dataset")
    gen_kw = dict(loop_retry=bool(gc.get("loop_retry", True)), retry_temperature=float(gc.get("retry_temperature", 0.7)))
    think = thinking_cfg(cfg)
    post = dict(gc.get("postprocess") or {})
    plex = AcupointLexicon(pkg.files["acupoint_lib"]) if (post.get("pregnancy_list") or post.get("rule_layer")) else None
    c5 = cfg["arms"]["A5"]
    think_tag = "think" if think["enabled"] else "nothink"
    
    precision = str(gc.get("precision", "nf4")).lower()
    if precision not in ("nf4", "bf16"):
        raise SystemExit(f"[STOP] generation.precision 只能是 nf4 或 bf16，当前 {precision}")
    hf_kw = dict(four_bit=(precision == "nf4"), merge_lora=bool(gc.get("merge_lora", True)))
    prec_tag = "" if precision == "nf4" else f"|{precision}"
    mask = list(cfg["train"].get("loss_mask_sections") or [])
    upw = float(cfg["train"].get("loss_upweight", 1.0) or 1.0)
    lora_tag = "+lora" + (f"(mask={'/'.join(mask)})" if mask else "") + (f"(w{upw:g})" if upw != 1.0 else "")
    model_id = {arm: (f"api:{c5['model']}" if c5["backend"] == "openai_compatible" else f"hf:{Path(c5['hf_path']).name}") + f"|{think_tag}"
                if arm == "A5" else f"hf:{Path(cfg['models']['base_llm']).name}" + (lora_tag if arm in ("A3", "A4") else "") + f"|{think_tag}{prec_tag}"
                for arm in ("A1", "A2", "A3", "A4", "A5")}
    if {"A3", "A4"} & set(arms):
        
        man_p = Path(cfg["outputs"]["lora_adapter"]) / "TRAIN_MANIFEST.json"
        man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.exists() else {}
        want_kw = cfg["train"].get("chat_template_kwargs", cfg["models"].get("chat_template_kwargs")) or {}
        got_mask, got_kw = man.get("loss_mask_sections"), man.get("train_chat_template_kwargs")
        want_keep = list(cfg["train"].get("loss_mask_keep_prefixes") or [])
        got_keep = man.get("loss_mask_keep_prefixes", [] if not want_keep else None)
        want_up = (sorted(cfg["train"].get("loss_upweight_prefixes") or []), float(cfg["train"].get("loss_upweight", 1.0) or 1.0))
        got_up = (sorted(man.get("loss_upweight_prefixes") or []), float(man.get("loss_upweight", 1.0) or 1.0))
        if got_up[1] != 1.0 and man.get("loss_upweight_mode") != "overlap+decision":
            got_up = (got_up[0], f"{got_up[1]:g}（旧加权方式 {man.get('loss_upweight_mode')}：决定点未双向加权）")
        if want_up[1] == 1.0 and got_up[1] == 1.0:
            want_up = got_up = ([], 1.0)
        want_null, got_null = bool(cfg["train"].get("null_warning_line", False)), bool(man.get("null_warning_line", False))
        if want_null != got_null:
            got_up = (got_up[0], f"{got_up[1]}（训练目标 null_warning_line={got_null}，配置 {want_null}）")
        if got_mask is None or sorted(got_mask) != sorted(mask) or (got_kw is not None and got_kw != want_kw) or sorted(got_keep or []) != sorted(want_keep) or got_keep is None or got_up != want_up:
            raise SystemExit(f"[GEN][BLOCKED] {cfg['outputs']['lora_adapter']} 的训练设置与配置不符："
                             f"loss_mask_sections 训练时={got_mask}（配置 {mask}），保留行={got_keep}（配置 {want_keep}），训练模板={got_kw}（配置 {want_kw}），加权={got_up}（配置 {want_up}）。"
                            )
    for arm in arms:
        p = out / f"raw_{arm}.jsonl"
        if arm in model_id and p.exists():
            olds = {json.loads(l).get("model_id", "<未记录>") for l in p.open(encoding="utf-8") if l.strip()}
            if olds - {model_id[arm]}:
                raise SystemExit(f"[GEN][BLOCKED] {p} 里已有来自 {sorted(olds)} 的输出，当前 {arm} = {model_id[arm]}。"
                                 "把旧文件移走后再生成，避免不同模型/解码方式的输出混在一起。")
    print(f"[GEN] thinking={think['enabled']} answer_cap={gc['max_new_tokens']} stage2_cap={gc.get('max_new_tokens_stage2') or gc['max_new_tokens']} "
          f"postprocess={post}  A5={model_id['A5']}")

    def run(arm, gen):
        seen = done(arm)
        with (out / f"raw_{arm}.jsonl").open("a", encoding="utf-8") as f:
            for k, it in enumerate(items, 1):
                if it["case_id"] in seen:
                    continue
                rec = {"case_id": it["case_id"], "arm": arm, "model_id": model_id.get(arm), "rag": arm in RAG_ARMS, "system_prompt": sys_prompt}
                if arm in RAG_ARMS and two_stage:
                   
                    g = generate_two_stage(gen, it["messages"], evid[it["case_id"]][0], rag, lex, sys_prompt,
                                           stage2_max_new_tokens=int(gc.get("max_new_tokens_stage2") or 0) or None,
                                           apply_differential_guard=False)
                    rec.update({"two_stage": True, "prompt_sha256": g["prompt_sha256"], "stage2_prompt_sha256": g["stage2_prompt_sha256"],
                                "evidence_sha256": evid[it["case_id"]][1]["evidence_sha256"], "cards_sha256": g["cards_log"]["cards_sha256"],
                                "cards_log": g["cards_log"], "splice": g["splice"], "raw_output": g["text"],
                                "raw_output_pre_guard": g.get("text_pre_guard", g["text"]), "differential_log": g.get("differential_log"),
                                "stage1_output": g["stage1_output"], "stage2_output": g["stage2_output"], "cards": g["cards"]})
                else:
                    msgs = compose_messages(it["messages"], evid[it["case_id"]][0] if arm in RAG_ARMS else None, sys_prompt)
                    g = gen.generate(msgs)
                    rec.update({"two_stage": False, "prompt_sha256": sha(msgs), "evidence_sha256": evid[it["case_id"]][1]["evidence_sha256"] if arm in RAG_ARMS else None,
                                "raw_output": g["text"]})
                rec.update({k2: g.get(k2) for k2 in KEEP})
                
                model_text, fixed, pp = rec["raw_output"], rec["raw_output"], {}
               
                fixed, had_null = S.strip_null_warning(fixed)
                if had_null:
                    pp["null_warning"] = ["【急危重症警示】无"]
                if post.get("sex_consistency"):
                    inp_text = "\n".join(m_["content"] for m_ in it["messages"] if m_["role"] == "user")
                    fixed, removed = S.sex_consistent(fixed, inp_text)
                    pp["sex_consistency"] = removed
                if post.get("pregnancy_list"):
                   
                    fixed, removed = correct_pregnancy_lists(plex, fixed)
                    pp["pregnancy_list"] = removed
                if post.get("rule_layer"):
                    inp_text = "\n".join(m_["content"] for m_ in it["messages"] if m_["role"] == "user")
                    fixed, rl = rule_layer(plex, fixed, inp_text, strict_redflags=True,
                                           operation_v18=bool(post.get("operation", True)),
                                           doctor_view_v181=bool(post.get("doctor_view", True)),
                                           point_laterality_v19=bool(post.get("point_laterality", True)))
                    if rl.get("added") or rl.get("removed"):
                        pp["rule_layer"] = rl
                if pp:
                    rec["postprocess"] = pp
                if fixed != model_text:
                    rec["raw_output_model"], rec["raw_output"] = model_text, fixed
               
                if rec.get("two_stage"):
                    rec.update({"stage1_reasoning": g.get("stage1_reasoning"), "stage2_reasoning": g.get("stage2_reasoning")})
                else:
                    rec["reasoning"] = g.get("reasoning")
                rec["generated_at"] = time.strftime("%F %T")
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                flag = f"  (retried: {g.get('degenerate_first_pass')})" if g.get("retried") else ""
                flag += "  [STILL DEGENERATE]" if g.get("still_degenerate") else ""
                print(f"[GEN] {arm} {k}/{len(items)} {it['case_id']}{flag}", flush=True)

    for group, adapter in ((["A1", "A2"], None), (["A3", "A4"], cfg["outputs"]["lora_adapter"])):
        todo = [x for x in group if x in arms]
        if todo:
            g = HFGenerator(cfg["models"]["base_llm"], adapter, gc["max_new_tokens"], gc["repetition_penalty"],
                            no_repeat_ngram_size=int(gc.get("no_repeat_ngram_size", 0)),
                            chat_template_kwargs=cfg["models"].get("chat_template_kwargs"), thinking=think, **hf_kw, **gen_kw)
            print(f"[GEN] {'/'.join(todo)} 推理精度 {getattr(g, 'precision', '?')}" + ("，LoRA 已合并" if adapter and hf_kw["merge_lora"] and getattr(g, 'precision', '') == "bf16" else ""), flush=True)
            for arm in todo:
                run(arm, g)
            g.close()
    if "A5" in arms:
        g = HFGenerator(c5["hf_path"], None, gc["max_new_tokens"], gc["repetition_penalty"], no_repeat_ngram_size=int(gc.get("no_repeat_ngram_size", 0)),
                        chat_template_kwargs=c5.get("chat_template_kwargs"), thinking=think, **gen_kw) if c5["backend"] == "hf" else \
            OpenAICompat(c5["base_url"], c5["model"], c5["api_key_env"], gc["max_new_tokens"], extra_body=c5.get("extra_body"),
                         thinking=think, stream=bool(c5.get("stream", False)), **gen_kw)
        run("A5", g)
        g.close()
    if "A6" in arms and a.set != "pilot":
        with (out / "raw_A6.jsonl").open("w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps({"case_id": it["case_id"], "arm": "A6", "rag": False, "raw_output": it["reference"], "source": "reference"}, ensure_ascii=False) + "\n")
    
    bad = []
    for arm in arms:
        p = out / f"raw_{arm}.jsonl"
        if p.exists():
            bad += [(arm, json.loads(l)["case_id"]) for l in p.open(encoding="utf-8") if json.loads(l).get("still_degenerate")]
    if bad:
        print(f"[GEN][WARN] {len(bad)} outputs are still degenerate after the retry: {bad[:10]}{' …' if len(bad) > 10 else ''}")
    if a.set == "initial" and all(len(done(x)) == len(items) for x in ["A1", "A2", "A3", "A4", "A5"]):
        state.advance("generated", {"n_cases": len(items), "two_stage_rag": two_stage, "still_degenerate": bad})
        print("[GEN][DONE] stage 'generated' recorded (one-shot)")
    if a.set == "followup" and all(len(done(x)) == len(items) for x in fu_arms if x != "A6"):
        state.advance("followup_generated", {"n": len(items)})


if __name__ == "__main__":
    main()
