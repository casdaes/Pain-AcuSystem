#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PREFIX = "请根据以下病例给出针灸诊疗方案。\n\n"


def load_cases(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        out = []
        for k, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            d = json.loads(line)
            if "messages" in d:
                text = "\n".join(m["content"] for m in d["messages"] if m["role"] == "user")
            else:
                text = d["text"]
            out.append({"case_id": d.get("case_id") or f"case{k}", "text": text})
        return out
    return [{"case_id": path.stem, "text": path.read_text(encoding="utf-8")}]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "study.yaml"))
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-rule-layer", action="store_true", help="不加安全规则层")
    a = ap.parse_args()

    from tcm_study import schema as S
    from tcm_study.acupoints import AcupointLexicon
    from tcm_study.arms import HFGenerator, generate_two_stage, thinking_cfg
    from tcm_study.config import load_cfg
    from tcm_study.needling import correct_pregnancy_lists
    from tcm_study.package import Package
    from tcm_study.rag import RagEvidence, RagIndex
    from tcm_study.safety_layer import apply as rule_layer

    cfg = load_cfg(a.config)
    gc = cfg["generation"]
    post = dict(gc.get("postprocess") or {})
    pkg = Package(cfg["paths"]["package_dir"])
    lex = AcupointLexicon(pkg.files["acupoint_lib"])

    # 与 05 相同的 LoRA 一致性核对
    adapter = cfg["outputs"]["lora_adapter"]
    man_p = Path(adapter) / "TRAIN_MANIFEST.json"
    if not man_p.exists():
        raise SystemExit(f"[STOP] 找不到 {man_p}：A4 需要训练好的 LoRA（bash run_train_bg.sh 或 run_all_v12.sh STEPS=data,train）")
    man = json.loads(man_p.read_text(encoding="utf-8"))
    want = (list(cfg["train"].get("loss_mask_sections") or []), list(cfg["train"].get("loss_mask_keep_prefixes") or []))
    got = (man.get("loss_mask_sections") or [], man.get("loss_mask_keep_prefixes") or [])
    want_w = (sorted(cfg["train"].get("loss_upweight_prefixes") or []), float(cfg["train"].get("loss_upweight", 1.0) or 1.0))
    got_w = (sorted(man.get("loss_upweight_prefixes") or []), float(man.get("loss_upweight", 1.0) or 1.0))
    if got_w[1] != 1.0 and man.get("loss_upweight_mode") != "overlap+decision":
        got_w = (got_w[0], f"{got_w[1]:g}（旧加权方式 {man.get('loss_upweight_mode')}）")
    if want_w[1] == 1.0 and got_w[1] == 1.0:
        want_w = got_w = ([], 1.0)
    if bool(cfg["train"].get("null_warning_line", False)) != bool(man.get("null_warning_line", False)):
        got_w = (got_w[0], "null_warning_line 不符")
    if sorted(want[0]) != sorted(got[0]) or sorted(want[1]) != sorted(got[1]) or want_w != got_w:
        raise SystemExit(f"[STOP] LoRA 训练设置 {got} 与配置 {want} 不符，请用当前配置重训")

    rag = RagEvidence(RagIndex(cfg["outputs"]["rag_index"], cfg["rag"]["device"]), cfg["rag"])
    if not rag.index.dense:
        print("[WARN] 知识库没有稠密索引，检索只用 BM25，与研究中的 A4 不一致（请运行 scripts/03_build_rag.py 重建稠密索引）")
    gen = HFGenerator(cfg["models"]["base_llm"], adapter, gc["max_new_tokens"], gc["repetition_penalty"],
                      no_repeat_ngram_size=int(gc.get("no_repeat_ngram_size", 0)), chat_template_kwargs=cfg["models"].get("chat_template_kwargs"),
                      thinking=thinking_cfg(cfg), four_bit=str(gc.get("precision", "nf4")).lower() == "nf4",
                      merge_lora=bool(gc.get("merge_lora", True)), loop_retry=bool(gc.get("loop_retry", True)), retry_temperature=float(gc.get("retry_temperature", 0.7)))
    sys_prompt = gc.get("system_prompt", "v3")
    out_f = None
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        out_f = open(a.out, "w", encoding="utf-8")
    try:
        for case in load_cases(Path(a.input)):
            text = case["text"].strip()
            if not text.startswith("请根据以下病例"):
                text = PREFIX + text
            messages = [{"role": "user", "content": text}]
            evidence, ev_log = rag.build(RagEvidence.query_from_messages(messages))
            g = generate_two_stage(gen, messages, evidence, rag, lex, sys_prompt,
                                   stage2_max_new_tokens=int(gc.get("max_new_tokens_stage2") or 0) or None)
            model_out, final, pp = g["text"], g["text"], {}
            final, had_null = S.strip_null_warning(final)              # v14.3：删除格式占位行"【急危重症警示】无。"
            if had_null:
                pp["null_warning"] = ["【急危重症警示】无"]
            if post.get("sex_consistency"):
                final, rm = S.sex_consistent(final, text)
                pp["sex_consistency"] = rm
            if post.get("pregnancy_list"):
                final, rm = correct_pregnancy_lists(lex, final)
                pp["pregnancy_list"] = rm
            rl = None
            if not a.no_rule_layer:
                final, rl = rule_layer(lex, final, text, strict_redflags=True)
            print("=" * 30, case["case_id"], "=" * 30)
            print(final)
            changes = [f"{k}: {v}" for k, v in pp.items() if v] + ([f"规则层追加 {rl['added']}，删除 {rl['removed']}"] if rl and (rl["added"] or rl["removed"]) else [])
            if changes:
                print("-- 后处理：" + "；".join(changes))
            if out_f:
                out_f.write(json.dumps({"case_id": case["case_id"], "final": final, "model_output": model_out, "postprocess": pp, "rule_layer": rl,
                                        "stage1_output": g["stage1_output"], "stage2_output": g["stage2_output"], "cards": g["cards"],
                                        "evidence": evidence, "evidence_log": ev_log, "latency_s": g.get("latency_s")}, ensure_ascii=False) + "\n")
                out_f.flush()
    finally:
        if out_f:
            out_f.close()
        gen.close()
    print("\n[注意] 输出供执业医师审核，不是最终医嘱。")


if __name__ == "__main__":
    main()
