from __future__ import annotations
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import gc
import os
import shutil
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from tcm_pipeline.common import (
    bf16_available,
    compute_dtype,
    cuda_memory_gib,
    gradient_accumulation,
    decoder_layer_classes,
    load_local_causal_lm,
    lora_target_modules,
    set_all_seeds,
    set_process_cuda_device,
    trainer_tokenizer_kw,
    training_args_kwargs,
    world_size,
)

from tcm_study.config import load_cfg


class SelectiveLogitsTrainer(Trainer):
   

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        from tcm_pipeline.common import selective_causal_lm_loss
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        weights = inputs.pop("loss_weight", None)
        loss, out = selective_causal_lm_loss(model, inputs, labels, weights=weights)
        return (loss, out) if return_outputs else loss



class WeightedCollator:
  

    def __init__(self, base):
        self.base = base

    def __call__(self, features):
        ws = [f.pop("loss_weight") for f in features] if features and "loss_weight" in features[0] else None
        batch = self.base(features)
        if ws is not None:
            L = batch["input_ids"].shape[1]
            batch["loss_weight"] = torch.tensor([list(w[:L]) + [0.0] * (L - len(w)) for w in ws], dtype=torch.float32)
        return batch


class AssistantOnlyProcessor:
    
    _EVID = "【检索证据】"

    _STAGE2 = "\n\n【第一步输出】"

    def __init__(self, tokenizer, max_length: int, chat_kwargs: dict | None = None, mask_sections: list[str] | None = None,
                 mask_keep_prefixes: list[str] | None = None, upweight_prefixes: list[str] | None = None, upweight: float = 1.0):
        self.tok = tokenizer
        self.chat_kwargs = dict(chat_kwargs or {})
        self.max_length = int(max_length)
        self.n_trimmed = 0
        self.n_dropped = 0
        
        self.mask_sections = [x for x in (mask_sections or []) if x]

        self.mask_keep = [x for x in (mask_keep_prefixes or []) if x]
       
        self.upweight_prefixes = [x for x in (upweight_prefixes or []) if x]
        self.upweight = float(upweight or 1.0)
        self.n_upweighted_tokens = 0
        self.n_kept_tokens = 0
        self.n_masked_tokens = 0
        self.n_answer_tokens = 0
        self._offsets_ok = True

    @staticmethod
    def _common_prefix_len(a: list[int], b: list[int]) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _mask_spans(self, full_text: str, answer_start: int) -> list[tuple[int, int]]:
        
        if not self.mask_sections:
            return []
        from tcm_study.schema import TOP_LEVEL, _TOP_MARK, _ALIASES
        ans = full_text[answer_start:]
        end_tok = ans.find("<|im_end|>")
        ans_end = answer_start + (end_tok if end_tok >= 0 else len(ans))
        marks = [m for m in _TOP_MARK.finditer(full_text, answer_start, ans_end) if m.group(1).strip() in TOP_LEVEL]
        spans = []
        for i, m in enumerate(marks):
            name = _ALIASES.get(m.group(1).strip(), m.group(1).strip())
            if name in self.mask_sections:
                stop = marks[i + 1].start() if i + 1 < len(marks) else ans_end
                while stop > m.end() and full_text[stop - 1] in " \t\r\n":   # 栏目间的换行仍计 loss
                    stop -= 1
                spans += self._carve_keep(full_text, m.end(), stop)
        return spans

    def _upweight_spans(self, full_text: str, answer_start: int) -> list[tuple[int, int]]:
       
        if not self.upweight_prefixes or self.upweight == 1.0:
            return []
        import re as _re
        from tcm_study.schema import TOP_LEVEL, _TOP_MARK, _ALIASES
        ans = full_text[answer_start:]
        end_tok = ans.find("<|im_end|>")
        ans_end = answer_start + (end_tok if end_tok >= 0 else len(ans))
        marks = [m for m in _TOP_MARK.finditer(full_text, answer_start, ans_end) if m.group(1).strip() in TOP_LEVEL]
        pat = _re.compile(r"[ \t]*(?:\d+[\.．、)）]\s*|[①②③④⑤⑥⑦⑧⑨]\s*)?(?:" + "|".join(_re.escape(k) for k in self.upweight_prefixes) + ")")
        spans = []
        for i, m in enumerate(marks):
            if _ALIASES.get(m.group(1).strip(), m.group(1).strip()) != "注意事项":
                continue
            stop = marks[i + 1].start() if i + 1 < len(marks) else ans_end
           
            spans.append((m.end() - 1, m.end()))
            pos = m.end()
            while pos < stop:
                nl = full_text.find("\n", pos, stop)
                end = stop if nl < 0 else nl
                if pat.match(full_text, pos, end):
                    spans.append((pos, end))
                pos = end + 1
        return spans

    def _carve_keep(self, full_text: str, a: int, b: int) -> list[tuple[int, int]]:
        """从 [a,b) 里挖掉"以保留标签开头的行"，返回剩余的 mask 区间。"""
        if not self.mask_keep:
            return [(a, b)]
        out, cur, pos = [], a, a
        import re as _re
        pat = _re.compile(r"[ \t]*(?:\d+[\.．、)）]\s*|[①②③④⑤⑥⑦⑧⑨]\s*)?(?:" + "|".join(_re.escape(k) for k in self.mask_keep) + ")")
        while pos < b:
            nl = full_text.find("\n", pos, b)
            end = b if nl < 0 else nl
            if pat.match(full_text, pos, end):
                if pos > cur:
                    out.append((cur, pos))
                cur = end                                  # 这一行（不含换行）计 loss
            pos = end + 1
        if cur < b:
            out.append((cur, b))
        return [(x, y) for x, y in out if y > x]

    def _encode(self, messages):
        full_text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, **self.chat_kwargs)
        prompt_text = self.tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, **self.chat_kwargs)
        enc = None
        need_offs = bool(self.mask_sections) or (bool(self.upweight_prefixes) and self.upweight != 1.0)
        if need_offs and self._offsets_ok:
            try:
                enc = self.tok(full_text, truncation=False, add_special_tokens=False, return_offsets_mapping=True)
            except Exception as e:                      # 慢速分词器不支持 offsets：退回不 mask，并明确告警
                self._offsets_ok = False
                print(f"[WARN] 分词器不支持 return_offsets_mapping（{type(e).__name__}），loss_mask_sections / 加权未生效", flush=True)
        if enc is None:
            enc = self.tok(full_text, truncation=False, add_special_tokens=False)
        full_ids = enc["input_ids"]
        prompt_ids = self.tok(prompt_text, truncation=False, add_special_tokens=False)["input_ids"]
        cut = self._common_prefix_len(full_ids, prompt_ids)
        if cut < min(len(prompt_ids), 32):
            cut = min(len(prompt_ids), len(full_ids))
        masked = [False] * len(full_ids)
        offs = enc.get("offset_mapping") if hasattr(enc, "get") else None
        if offs is not None and self.mask_sections:
            ans_char = offs[cut][0] if cut < len(offs) else len(full_text)
            spans = self._mask_spans(full_text, ans_char)
            for i in range(cut, len(full_ids)):
                s0, e0 = offs[i]
                if e0 > s0 and any(a <= s0 and e0 <= b for a, b in spans):
                    masked[i] = True
        weight = [1.0] * len(full_ids)
        if offs is not None and self.upweight_prefixes and self.upweight != 1.0:
            ans_char = offs[cut][0] if cut < len(offs) else len(full_text)
            wsp = self._upweight_spans(full_text, ans_char)
            for i in range(cut, len(full_ids)):
                s0, e0 = offs[i]
                
                if e0 > s0 and not masked[i] and any(s0 < b and e0 > a for a, b in wsp):
                    weight[i] = self.upweight
        self._last_weight = weight
        return full_ids, cut, masked

    def _shrink_user(self, messages, excess_tokens: int):
        """Return a copy with the last user message shortened by ~excess_tokens (chars ≈ tokens for Chinese)."""
        msgs = [dict(m) for m in messages]
        ui = max(i for i, m in enumerate(msgs) if m["role"] == "user")
        u = msgs[ui]["content"]
        n = int(excess_tokens * 1.3) + 8
        if self._EVID in u:
            head, ev = u.split(self._EVID, 1)
            tail = ""
            if self._STAGE2 in ev:
                ev, tail = ev.split(self._STAGE2, 1)
                tail = self._STAGE2 + tail
            ev = ev[: max(0, len(ev) - n)]
            msgs[ui]["content"] = head + self._EVID + ev + ("\n（检索证据因长度截断）" if ev else "") + tail
        else:
            msgs[ui]["content"] = u[: max(64, len(u) - n)] + "…"
        return msgs

    def __call__(self, batch):
        # _n_answer/_n_masked：每条样本的计数写成列，map 之后在主进程求和（datasets 会复制处理函数，对象上的计数器在主进程里恒为 0）
        out = {"input_ids": [], "attention_mask": [], "labels": [], "_n_answer": [], "_n_masked": [], "_n_upweighted": []}
        if self.upweight_prefixes and self.upweight != 1.0:
            out["loss_weight"] = []
        for messages in batch["messages"]:
            if not messages or messages[-1].get("role") != "assistant":
                continue
            full_ids, cut, masked = self._encode(messages)
            answer_ids = full_ids[cut:]
            if not answer_ids or len(answer_ids) > self.max_length - 64:
                self.n_dropped += 1
                continue
            tries = 0
            while len(full_ids) > self.max_length and tries < 6:
                messages = self._shrink_user(messages, len(full_ids) - self.max_length)
                full_ids, cut, masked = self._encode(messages)
                tries += 1
                self.n_trimmed += tries == 1
            if len(full_ids) > self.max_length:
                self.n_dropped += 1
                continue
            answer_ids = full_ids[cut:]
            prompt_ids = full_ids[:cut]
            out["input_ids"].append(prompt_ids + answer_ids)
            out["attention_mask"].append([1] * len(full_ids))
            labels = [-100] * len(prompt_ids) + [(-100 if masked[cut + j] else t) for j, t in enumerate(answer_ids)]
            self.n_kept_tokens += sum(1 for j in range(len(answer_ids)) if not masked[cut + j])
            self.n_answer_tokens += len(answer_ids)
            self.n_masked_tokens += sum(masked[cut:])
            out["labels"].append(labels)
            out["_n_answer"].append(len(answer_ids))
            out["_n_masked"].append(int(sum(masked[cut:])))
            w = self._last_weight
            out["_n_upweighted"].append(int(sum(1 for x in w[cut:] if x != 1.0)))
            if "loss_weight" in out:
                out["loss_weight"].append(w)
        return out


def choose_single_gpu_profile(c: dict, tc: dict, rank: int) -> tuple[dict, list[str], str]:
    """Choose a conservative SFT profile from *actual* free VRAM before load."""
    c = dict(c)
    targets = list(tc["target_modules"])
    profile = os.environ.get("TCM_SFT_PROFILE", "auto").strip().lower()
    free_gib, total_gib = cuda_memory_gib(rank)

    # User overrides always win.
    max_override = os.environ.get("TCM_SFT_MAX_LENGTH_OVERRIDE")
    rank_override = os.environ.get("TCM_SFT_LORA_R_OVERRIDE")
    target_override = os.environ.get("TCM_SFT_TARGET_MODULES_OVERRIDE")

    if profile == "auto":
        # These thresholds intentionally leave headroom for other users/services.
        if free_gib >= 20.0:
            profile = "normal"
        elif free_gib >= 16.0:
            profile = "balanced"
        else:
            profile = "emergency"

   
    if profile == "normal":
        c["max_length"] = min(int(c["max_length"]), 10240)
    elif profile == "balanced":
        c["max_length"] = min(int(c["max_length"]), 2560)
        c["lora_r"] = min(int(c["lora_r"]), 16)
        c["lora_alpha"] = min(int(c["lora_alpha"]), 2 * int(c["lora_r"]))
        targets = [x for x in targets if x in {"q_proj", "k_proj", "v_proj", "o_proj"}]
    elif profile == "emergency":
        c["max_length"] = min(int(c["max_length"]), 1024)
        c["lora_r"] = min(int(c["lora_r"]), 8)
        c["lora_alpha"] = min(int(c["lora_alpha"]), 2 * int(c["lora_r"]))
        targets = [x for x in targets if x in {"q_proj", "v_proj"}]
    else:
        raise ValueError("TCM_SFT_PROFILE 只能是 auto/normal/balanced/emergency")

    if max_override:
        # 显式 override 真正覆盖 profile 值；便于在显存允许时做 8192/9216/10240 预跑。
        c["max_length"] = int(max_override)
    if rank_override:
        c["lora_r"] = min(int(c["lora_r"]), int(rank_override))
        c["lora_alpha"] = min(int(c["lora_alpha"]), 2 * int(c["lora_r"]))
    if target_override:
        wanted = [x.strip() for x in target_override.split(",") if x.strip()]
        targets = wanted

    if rank == 0:
        print(
            f"[SFT PROFILE] profile={profile} free_before_load={free_gib:.2f}/{total_gib:.2f} GiB "
            f"max_length={c['max_length']} lora_r={c['lora_r']} targets={targets}"
        )
    return c, targets, profile


def latest_checkpoint(out_dir: str) -> Path | None:
    p = Path(out_dir)
    if not p.is_dir():
        return None
    ckpts = []
    for x in p.glob("checkpoint-*"):
        if x.is_dir():
            try:
                step = int(x.name.rsplit("-", 1)[1])
            except Exception:
                continue
            ckpts.append((step, x))
    return max(ckpts, default=(None, None))[1]


def train_once(config_path: str) -> str:
    cfg = load_cfg(config_path)
    tc = cfg["train"]
    c = dict(tc["sft"])
    set_all_seeds(int(tc["seed"]))
    rank = set_process_cuda_device()
    target_modules = list(tc["target_modules"])
    active_profile = "multi"

    lowmem_single = os.environ.get("TCM_LOW_MEMORY_SINGLE", "0") == "1" and world_size() == 1
    if lowmem_single:
        c, target_modules, active_profile = choose_single_gpu_profile(c, tc, rank)

    tag = os.environ.get("SFT_DATA_TAG", "final")
    base_path = cfg["models"]["base_llm"]
   
    suffix = os.environ.get("SFT_ADAPTER_SUFFIX", "" if tag == "final" else "_" + tag)
    out_dir = cfg["outputs"]["lora_adapter"] + suffix
    if not Path(base_path).is_dir():
        raise FileNotFoundError(f"base model not found: {base_path}")

    tok = AutoTokenizer.from_pretrained(
        base_path, trust_remote_code=True, use_fast=True, local_files_only=True
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    use_cuda = torch.cuda.is_available()
    use_4bit = bool(tc.get("qlora", True) and use_cuda)
    multi_gpu_mode = os.environ.get("TCM_MULTI_GPU_MODE", "ddp").strip().lower()
    fsdp_mode = multi_gpu_mode == "fsdp" and world_size() > 1
    quant = None
    if use_4bit:
        bnb_kwargs = dict(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype(bool(tc.get("bf16", True))),
            bnb_4bit_use_double_quant=True,
        )
        if fsdp_mode:
            bnb_kwargs["bnb_4bit_quant_storage"] = compute_dtype(bool(tc.get("bf16", True)))
        quant = BitsAndBytesConfig(**bnb_kwargs)

    device_map = None if fsdp_mode else ({"": rank} if use_4bit and use_cuda else None)
    model = load_local_causal_lm(
        base_path,
        dtype=compute_dtype(bool(tc.get("bf16", True))),
        trust_remote_code=True,
        local_files_only=True,
        quantization_config=quant,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    layer_cls = decoder_layer_classes(model)        # 在套 LoRA 之前读解码层类名（FSDP 用）
    model.config.use_cache = False
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = False
    if use_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=bool(tc.get("gradient_checkpointing", True))
        )
    if bool(tc.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    target_modules = lora_target_modules(model, target_modules)   # Qwen3.5：只给语言模型加 LoRA（排除视觉塔与 MTP 头）
    if rank == 0:
        print(f"[LoRA] target_modules={target_modules}")
    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(c["lora_r"]),
        lora_alpha=int(c["lora_alpha"]),
        lora_dropout=float(c["lora_dropout"]),
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    if rank == 0 and use_cuda:
        free_after, total = cuda_memory_gib(rank)
        print(f"[VRAM] after model+LoRA load: free={free_after:.2f}/{total:.2f} GiB")

    data_dir = Path(cfg["paths"]["work_dir"]) / "sft" / tag
    ds = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "val.jsonl"),
        },
    )
    before_train, before_val = len(ds["train"]), len(ds["validation"])
   
    train_chat_kw = tc.get("chat_template_kwargs", cfg["models"].get("chat_template_kwargs"))
    mask_sections = list(tc.get("loss_mask_sections") or [])
    mask_keep = list(tc.get("loss_mask_keep_prefixes") or [])
    up_pref = list(tc.get("loss_upweight_prefixes") or [])
    up_w = float(tc.get("loss_upweight", 1.0) or 1.0)
    processor = AssistantOnlyProcessor(tok, int(c["max_length"]), train_chat_kw, mask_sections, mask_keep, up_pref, up_w)
    cols = ds["train"].column_names
    ds = ds.map(
        processor,
        batched=True,
        remove_columns=cols,
        num_proc=1,
        load_from_cache_file=False,
        desc="Apply chat template + assistant-only loss",
    )
    n_answer = sum(ds["train"]["_n_answer"]) + sum(ds["validation"]["_n_answer"])
    n_masked = sum(ds["train"]["_n_masked"]) + sum(ds["validation"]["_n_masked"])
    n_up = sum(ds["train"]["_n_upweighted"]) + sum(ds["validation"]["_n_upweighted"])
    n_up_rows = sum(1 for x in ds["train"]["_n_upweighted"] if x)
    ds = ds.remove_columns(["_n_answer", "_n_masked", "_n_upweighted"])
    if rank == 0:
        dropped_train = before_train - len(ds["train"])
        dropped_val = before_val - len(ds["validation"])
        print(
            f"[DATA] train={len(ds['train'])}/{before_train} val={len(ds['validation'])}/{before_val} "
            f"dropped={dropped_train}/{dropped_val} (max_length={c['max_length']}; user text trimmed when too long, system prompt always kept)"
        )
        if dropped_train > 0.02 * before_train:
            print(f"[WARN] {dropped_train} training samples dropped for length — raise train.sft.max_length (v3 targets are longer than v1)")
        if mask_sections:
            frac = n_masked / max(1, n_answer)
            print(f"[MASK] 不计 loss 的栏目 {mask_sections}（保留行 {mask_keep}）：{n_masked}/{n_answer} 个答案 token（{frac:.1%}，训练集+验证集）")
            if n_masked == 0:
                raise SystemExit("[STOP] 配置了 train.loss_mask_sections，但一个 token 都没有被 mask（分词器不支持 offsets 或栏目名不匹配），停止训练")
    if up_pref and up_w != 1.0:
        if rank == 0:
            print(f"[WEIGHT] 注意事项中以 {up_pref} 开头的行 ×{up_w}：{n_up} 个 token，训练集 {n_up_rows} 条样本含加权行", flush=True)
        if n_up == 0:
            raise SystemExit("[STOP] 配置了 train.loss_upweight，但没有任何 token 被加权（行首标签不匹配或分词器不支持 offsets），停止训练")
    if len(ds["train"]) == 0:
        raise RuntimeError("低显存截断后训练集为空；请提高 max_length 或缩短 SFT target。")

    accum = gradient_accumulation(
        int(c["target_global_batch_size"]), int(c["per_device_train_batch_size"])
    )
    if rank == 0:
        effective = int(c["per_device_train_batch_size"]) * accum * world_size()
        print(f"[BATCH] world_size={world_size()} grad_accum={accum} effective_global_batch={effective}")

    distributed_extra = {}
    if fsdp_mode:
        distributed_extra = {
            "fsdp": "full_shard auto_wrap",
            "fsdp_config": {
                "transformer_layer_cls_to_wrap": layer_cls,
                "use_orig_params": False,
                "limit_all_gathers": True,
                "forward_prefetch": False,
                "sync_module_states": True,
            },
        }
        if rank == 0:
            print("[DIST] FSDP-QLoRA full_shard 已启用。")
    elif world_size() > 1:
        if rank == 0:
            print("[DIST] DDP-QLoRA 已启用。")
    elif rank == 0:
        print(f"[DIST] 单 GPU QLoRA 已启用；低显存档位={active_profile}。")

    import math as _m
    _steps_per_epoch = _m.ceil(_m.ceil(len(ds["train"]) / (int(c["per_device_train_batch_size"]) * world_size())) / accum)
    train_kwargs = training_args_kwargs(
        _total_steps=int(_m.ceil(_steps_per_epoch * float(c["epochs"]))),
        output_dir=out_dir,
        num_train_epochs=float(c["epochs"]),
        learning_rate=float(c["learning_rate"]),
        per_device_train_batch_size=int(c["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(c["per_device_eval_batch_size"]),
        gradient_accumulation_steps=accum,
        warmup_ratio=float(c["warmup_ratio"]),
        weight_decay=float(c["weight_decay"]),
        logging_steps=int(c["logging_steps"]),
        eval_strategy="steps",
        eval_steps=int(c["eval_steps"]),
        save_strategy="steps",
        save_steps=int(c["save_steps"]),
        save_total_limit=int(c["save_total_limit"]),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bool(tc.get("bf16", True) and bf16_available()),
        fp16=bool(tc.get("fp16", False) and use_cuda and not bf16_available()),
        tf32=bool(tc.get("tf32", True) and use_cuda),
        gradient_checkpointing=bool(tc.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="cosine",
        optim="adamw_torch" if fsdp_mode else ("paged_adamw_8bit" if use_4bit else "adamw_torch"),
        max_grad_norm=1.0,
        group_by_length=True,
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        prediction_loss_only=True,
        seed=int(tc["seed"]),
        data_seed=int(tc["seed"]),
    )
    # Current Transformers exposes torch_empty_cache_steps. Add it only when
    # this installed version supports the argument.
    import inspect
    if "torch_empty_cache_steps" in inspect.signature(TrainingArguments.__init__).parameters:
        train_kwargs["torch_empty_cache_steps"] = 25 if lowmem_single else None

    args = TrainingArguments(**train_kwargs, **distributed_extra)
    collator = DataCollatorForSeq2Seq(
        tokenizer=tok,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    if up_pref and up_w != 1.0:
        collator = WeightedCollator(collator)
        _b = collator([dict(ds["train"][i]) for i in range(min(2, len(ds["train"])))])   # 启动前自检：权重与 labels 同形、补齐位置为 0
        if tuple(_b["loss_weight"].shape) != tuple(_b["labels"].shape):
            raise SystemExit(f"[STOP] loss_weight 形状 {tuple(_b['loss_weight'].shape)} 与 labels {tuple(_b['labels'].shape)} 不一致")
        if rank == 0:
            print(f"[WEIGHT] 批次自检通过：loss_weight {tuple(_b['loss_weight'].shape)}，最大权重 {float(_b['loss_weight'].max()):g}", flush=True)
    TrainerCls = SelectiveLogitsTrainer if (bool(c.get("selective_logits", True)) or (up_pref and up_w != 1.0)) else Trainer
    if rank == 0:
        print(f"[LOSS] {'只对回答部分计算 logits（省显存，损失与全量计算相同）' if TrainerCls is SelectiveLogitsTrainer else '全序列 logits'}")
    trainer = TrainerCls(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        **trainer_tokenizer_kw(tok),
    )
    if fsdp_mode and getattr(trainer.accelerator.state, "fsdp_plugin", None):
        from peft.utils.other import fsdp_auto_wrap_policy

        trainer.accelerator.state.fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(trainer.model)

    resume = latest_checkpoint(out_dir)
    if resume is not None:
        import json as _json
        ac = Path(resume, "adapter_config.json")
        old_base = _json.loads(ac.read_text(encoding="utf-8")).get("base_model_name_or_path") if ac.is_file() else None
        if old_base and Path(old_base).name != Path(base_path).name:
            raise SystemExit(f"[STOP] {out_dir} 里的 checkpoint 是用 {old_base} 训练的，与当前基座 {base_path} 不同，不能续训。\n"
                             f"       先把旧目录改名备份：mv {out_dir} {out_dir}_old_{Path(old_base).name}")
    if resume is not None:
        if rank == 0:
            print(f"[RESUME] 发现兼容 checkpoint：{resume}")
        trainer.train(resume_from_checkpoint=str(resume))
    else:
        trainer.train()

    if fsdp_mode and getattr(trainer.accelerator.state, "fsdp_plugin", None):
        try:
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        except Exception as e:
            if rank == 0:
                print(f"[WARN] 无法显式切换 FULL_STATE_DICT：{e}")
    trainer.save_model(out_dir)
    if rank == 0:
        tok.save_pretrained(out_dir)
        import json as _json
        from tcm_study.package import Package as _P
        man = _json.loads((data_dir / "split_manifest.json").read_text(encoding="utf-8"))
        Path(out_dir, "TRAIN_MANIFEST.json").write_text(_json.dumps({**man, "base_model": base_path, "train_chat_template_kwargs": train_chat_kw, "loss_mask_sections": mask_sections, "loss_mask_keep_prefixes": mask_keep, "loss_upweight_prefixes": up_pref, "loss_upweight": up_w, "loss_upweight_mode": "overlap+decision" if up_w != 1.0 else None, "null_warning_line": bool(tc.get("null_warning_line", False)), "lora": {k: c[k] for k in ("lora_r", "lora_alpha", "lora_dropout", "learning_rate", "epochs", "max_length")}}, indent=2), encoding="utf-8")
        print(f"[MANIFEST] wrote {out_dir}/TRAIN_MANIFEST.json")
        print(f"[OK] SFT LoRA: {out_dir}")
    return active_profile


def main(config_path: str):
    try:
        train_once(config_path)
        return
    except torch.OutOfMemoryError:
        
        if os.environ.get("TCM_LOW_MEMORY_SINGLE", "0") != "1":
            raise
        cfg = load_cfg(config_path)
        out_dir = cfg["outputs"]["lora_adapter"]
        if latest_checkpoint(out_dir) is not None:
            print("[OOM] 已存在 SFT checkpoint；为避免破坏 checkpoint，不自动改 LoRA 结构重试。")
            raise
        current = os.environ.get("TCM_SFT_PROFILE", "auto").strip().lower()
        if current == "emergency":
            raise

        print("[OOM-RETRY] 首个单卡档位 OOM；将重启 Python 进程并降到 emergency：")
        print("            max_length<=1024, LoRA r<=8, target_modules=q_proj,v_proj")
        
        if Path(out_dir).exists():
            shutil.rmtree(out_dir)
        os.environ["TCM_SFT_PROFILE"] = "emergency"
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), "--config", config_path])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/study.yaml")
    main(ap.parse_args().config)
