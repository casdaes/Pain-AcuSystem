
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import importlib.util
import json
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/study.yaml")
    ap.add_argument("--bf16", action="store_true", help="不量化，按 bf16 加载")
    ap.add_argument("--think-budget", type=int, default=1024, help="冒烟测试的思考预算（正式生成用 generation.thinking.budget_tokens）")
    ap.add_argument("--check-loss", action="store_true", help="只检查训练用的省显存损失与标准损失是否一致（不生成）")
    a = ap.parse_args()

    import torch
    import transformers
    from transformers import AutoTokenizer, BitsAndBytesConfig

    from tcm_pipeline.common import compute_dtype, decoder_layer_classes, load_local_causal_lm, lora_target_modules
    from tcm_study.config import load_cfg
    from tcm_study.schema import SYSTEM_PROMPT_V3

    cfg = load_cfg(a.config)
    base = cfg["models"]["base_llm"]
    chat_kw = cfg["models"].get("chat_template_kwargs") or {}
    train_kw = cfg["train"].get("chat_template_kwargs", chat_kw) or {}
    from tcm_study.arms import HFGenerator, thinking_cfg
    think = thinking_cfg(cfg)
    print(f"[INFO] transformers {transformers.__version__}  torch {torch.__version__}  base={base}  chat_kwargs={chat_kw}")
    if not Path(base).is_dir():
        raise SystemExit(f"[FAIL] 找不到基座目录 {base}")


    for mod, pkg in (("fla", "flash-linear-attention"), ("causal_conv1d", "causal-conv1d")):
        if importlib.util.find_spec(mod) is None:
            print(f"[WARN] 未安装 {pkg}：Qwen3.5 的线性注意力会退回纯 PyTorch 实现，训练/生成会明显变慢。可选：pip install {pkg}")
        else:
            print(f"[OK] {pkg} 已安装")


    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True, local_files_only=True)
    msgs = [{"role": "system", "content": "你是针灸科医生。"}, {"role": "user", "content": "患者腰痛三天，请给出针灸方案。"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **chat_kw)
    tprompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **train_kw)
    open_think = lambda p: p.rstrip().endswith("<think>")
    print(f"[INFO] 生成提示结尾：...{prompt[-40:]!r}   训练提示结尾：...{tprompt[-40:]!r}")
    if open_think(prompt) == bool(think["enabled"]):
        print(f"[OK] 生成：思考模式{'开' if think['enabled'] else '关'}（与 generation.thinking.enabled 一致）")
    else:
        print("[FAIL] 生成提示的思考开关与 generation.thinking.enabled 不一致；检查 models.chat_template_kwargs")
    print(f"[{'OK' if not open_think(tprompt) else 'WARN'}] 训练：{'非思考模板（loss 只算答案）' if not open_think(tprompt) else '思考模板——训练数据没有思考过程，会教模型不思考'}")
    if think["enabled"]:
        tid = tok.convert_tokens_to_ids("</think>")
        print(f"[{'OK' if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id else 'FAIL'}] </think> 是单个 token（id={tid}），思考预算截断依赖它")

    
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        need = 10 if a.bf16 else 5
        print(f"[INFO] 当前卡空闲显存 {free / 2**30:.1f}/{total / 2**30:.1f} GiB")
        if free / 2**30 < need:
            raise SystemExit(f"[FAIL] 这张卡只剩 {free / 2**30:.1f} GiB（需要约 {need} GiB）：可能正被训练或别人占用。用 nvidia-smi 找一张空卡再跑。")
    quant = None if a.bf16 or not torch.cuda.is_available() else BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype(True), bnb_4bit_use_double_quant=True)
    t0 = time.time()
    model = load_local_causal_lm(base, dtype=compute_dtype(True), trust_remote_code=True, local_files_only=True,
                                 quantization_config=quant, device_map={"": 0} if torch.cuda.is_available() else None,
                                 low_cpu_mem_usage=True).eval()
    print(f"[OK] 加载 {type(model).__name__}（{'bf16' if quant is None else '4-bit'}）用时 {time.time() - t0:.0f}s；"
          f"FSDP 解码层类 = {decoder_layer_classes(model)}")

    tm = lora_target_modules(model, list(cfg["train"]["target_modules"]))
    import re
    names = [n for n, _ in model.named_modules()]
    hit = [n for n in names if (re.fullmatch(tm, n) if isinstance(tm, str) else any(n.endswith("." + x) for x in tm))]
    vis = [n for n in hit if "visual" in n or "mtp" in n]
    kinds = sorted({n.rsplit(".", 1)[-1] for n in hit})
    print(f"[{'OK' if hit and not vis else 'FAIL'}] LoRA 将挂到 {len(hit)} 个线性层，类型 {kinds}；其中视觉/MTP 层 {len(vis)} 个（应为 0）")

    if a.check_loss:
        from tcm_pipeline.common import selective_causal_lm_loss
        prompt_ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        ans_ids = tok("【诊疗思路】腰痛三天，多属经络气血阻滞。【取穴】肾俞、大肠俞、委中、腰阳关。<|im_end|>", return_tensors="pt", add_special_tokens=False)["input_ids"]
        ids = torch.cat([prompt_ids, ans_ids], 1).to(model.device)
        labels = ids.clone(); labels[:, :prompt_ids.shape[1]] = -100
        inp = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        with torch.no_grad():
            full = float(model(**inp, labels=labels).loss)
            sel = float(selective_causal_lm_loss(model, inp, labels)[0])
        ok = abs(full - sel) < 1e-2 * max(1.0, abs(full))
        print(f"[{'OK' if ok else 'FAIL'}] 损失一致性：标准 {full:.4f}  省显存版 {sel:.4f}")
        return

    case = None
    p = Path(cfg["paths"]["package_dir"]) / "test_initial_inputs.jsonl"
    if p.is_file():
        case = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
    user = next((m["content"] for m in (case or {}).get("messages", []) if m["role"] == "user"), msgs[-1]["content"])
    m2 = [{"role": "system", "content": SYSTEM_PROMPT_V3}, {"role": "user", "content": user}]
    
    gc = cfg["generation"]
    g = HFGenerator.__new__(HFGenerator)
    g.torch, g.tok = torch, tok
    g._setup(model, gc["max_new_tokens"], gc["repetition_penalty"], False, float(gc.get("retry_temperature", 0.7)),
             int(gc.get("no_repeat_ngram_size", 0)), chat_kw, {**think, "budget_tokens": min(int(think["budget_tokens"]), a.think_budget)})
    t0 = time.time()
    r = g.generate(m2)
    dt = time.time() - t0
    ans = r["text"]
    n_all = (r["new_tokens"] or 0) + (r["reasoning_tokens"] or 0)
    print(f"[INFO] 思考 {r['reasoning_tokens']} token（预算{'用完、已强制收尾' if r['think_budget_hit'] else '内结束'}）+ 答案 {r['new_tokens']} token，"
          f"用时 {dt:.1f}s（{n_all / max(dt, 1e-6):.1f} token/s）")
    if r.get("reasoning"):
        print("-" * 25 + " 思考（前 600 字）" + "-" * 25 + "\n" + r["reasoning"][:600])
    print("-" * 25 + " 答案 " + "-" * 25 + "\n" + ans[:1500] + "\n" + "-" * 60)
    n_sec = len(re.findall(r"【[^】]{2,6}】", ans))
    print(f"[{'OK' if '<think>' not in ans and '</think>' not in ans else 'FAIL'}] 答案中没有思考标签；识别到 {n_sec} 个【栏目】"
          + ("" if n_sec >= 5 else "  [WARN] 栏目偏少，检查答案格式"))


if __name__ == "__main__":
    main()
