from __future__ import annotations

import inspect
import math
import json
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml

EMPTY = {"", "无", "不详", "暂无", "否", "none", "nan", "null", "同前", "未提供"}


def configure_offline() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def clean(x: Any) -> str:
    if x is None:
        return ""
    s = unicodedata.normalize("NFKC", str(x)).replace("\u3000", " ").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return "" if s.lower() in EMPTY else s


def norm_key(x: Any) -> str:
    s = clean(x).lower()
    s = s.replace("⻛", "风").replace("⻩", "黄").replace("⽓", "气").replace("⾎", "血")
    s = re.sub(r"[\s，,。；;：:（）()、/\\\-]+", "", s)
    return s


def join_fields(items: Iterable[tuple[str, Any]]) -> str:
    out = []
    for k, v in items:
        v = clean(v)
        if v:
            out.append(f"{k}：{v}")
    return "\n".join(out)


def load_config(path: str) -> dict:
    configure_offline()
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    root = Path(cfg.get("project_dir", p.parent.parent)).expanduser()
    if not root.is_absolute():
        root = (p.parent / root).resolve()
    cfg["project_dir"] = str(root)

    def resolve(v: Any) -> Any:
        if not isinstance(v, str) or not v.strip():
            return v
        q = Path(v).expanduser()
        return str(q if q.is_absolute() else (root / q).resolve())

    for section in ("files", "outputs"):
        for k, v in list(cfg.get(section, {}).items()):
            # Metadata such as an Excel sheet name is not a filesystem path.
            if section == "files" and str(k).endswith("_sheet"):
                continue
            cfg[section][k] = resolve(v)

   
    for _, scfg in list(cfg.get("specialists", {}).items()):
        if not isinstance(scfg, dict):
            continue
        for k in ("adapter", "train_file", "val_file"):
            if k in scfg:
                scfg[k] = resolve(scfg[k])
    dcfg = cfg.get("decontamination", {})
    if isinstance(dcfg, dict) and dcfg.get("test_dir"):
        dcfg["test_dir"] = resolve(dcfg["test_dir"])

    ecfg = cfg.get("evaluation", {})
    if isinstance(ecfg, dict) and ecfg.get("test_dir"):
        ecfg["test_dir"] = resolve(ecfg["test_dir"])

    for k, v in list(cfg.get("models", {}).items()):
        cfg["models"][k] = resolve(v)
    return cfg


def dump_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stable_split(rows: list[dict], val_ratio: float, test_ratio: float, seed: int):
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    n = len(rows)
    n_test = max(1, round(n * test_ratio)) if n >= 10 and test_ratio > 0 else 0
    n_val = max(1, round(n * val_ratio)) if n >= 10 and val_ratio > 0 else 0
    return rows[n_test + n_val :], rows[n_test : n_test + n_val], rows[:n_test]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bf16_available() -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def compute_dtype(prefer_bf16: bool = True):
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if prefer_bf16 and bf16_available() else torch.float16


def world_size() -> int:
    return max(1, int(os.environ.get("WORLD_SIZE", "1")))


def local_rank() -> int:
    return max(0, int(os.environ.get("LOCAL_RANK", "0")))


def set_process_cuda_device() -> int:
    rank = local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    return rank


def gradient_accumulation(target_global_batch: int, per_device_batch: int) -> int:
    denom = max(1, per_device_batch * world_size())
    return max(1, int(round(target_global_batch / denom)))


def resolve_embedding_model(cfg):
    """
    Resolve local embedding model.

    Priority:
    1. Environment variable TCM_BGE_MODEL
    2. configs/config.yaml
    3. fixed local fallback

    Important:
    Never automatically append '-modelscope'.
    """
    import os
    from pathlib import Path

    candidates = []

    # ---------------------------------------------------------
    # 1. Environment variable has highest priority
    # ---------------------------------------------------------
    env_path = os.environ.get("TCM_BGE_MODEL", "").strip()
    if env_path:
        candidates.append(env_path)

    # ---------------------------------------------------------
    # 2. Read possible config locations
    # ---------------------------------------------------------
    if isinstance(cfg, dict):
        for key in (
            "embedding_model_local_path",
            "embedding_model",
            "bge_model",
            "bge_model_path",
        ):
            value = cfg.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())

        rag_cfg = cfg.get("rag", {})
        if isinstance(rag_cfg, dict):
            for key in (
                "embedding_model_local_path",
                "embedding_model",
                "model_path",
                "bge_model",
                "bge_model_path",
            ):
                value = rag_cfg.get(key)
                if isinstance(value, str) and value.strip():
                    candidates.append(value.strip())

        model_cfg = cfg.get("models", {})
        if isinstance(model_cfg, dict):
            for key in (
                "embedding_model_local_path",
                "embedding_model",
                "bge_model",
                "bge_model_path",
            ):
                value = model_cfg.get(key)
                if isinstance(value, str) and value.strip():
                    candidates.append(value.strip())

    # ---------------------------------------------------------
    # Deduplicate
    # ---------------------------------------------------------
    normalized = []
    seen = set()

    for value in candidates:
        if not value:
            continue

        p = str(Path(value).expanduser())

        # Repair accidental repeated suffixes
        while "-modelscope-modelscope" in p:
            p = p.replace(
                "-modelscope-modelscope",
                "-modelscope"
            )

        if p not in seen:
            seen.add(p)
            normalized.append(p)

    # ---------------------------------------------------------
    # Check model directory
    # ---------------------------------------------------------
    checked = []

    for candidate in normalized:
        p = Path(candidate)
        checked.append(str(p))

        if not p.is_dir():
            continue

        # HuggingFace / SentenceTransformer model indicators
        weight_files = [
            p / "model.safetensors",
            p / "pytorch_model.bin",
        ]

        has_sharded_safetensors = any(
            p.glob("model-*.safetensors")
        )

        has_sharded_bin = any(
            p.glob("pytorch_model-*.bin")
        )

        has_config = (p / "config.json").exists()

        if (
            any(x.exists() for x in weight_files)
            or has_sharded_safetensors
            or has_sharded_bin
        ):
            print(f"[EMBEDDING] selected: {p}")
            return str(p)

        # SentenceTransformer may put transformer weights in subfolder
        if (p / "modules.json").exists():
            return str(p)

        # Some ModelScope snapshots are still valid HF directories.
        if has_config:
            possible_weights = (
                list(p.glob("*.safetensors"))
                + list(p.glob("*.bin"))
            )
            if possible_weights:
                print(f"[EMBEDDING] selected: {p}")
                return str(p)

    raise FileNotFoundError(
        "未找到本地 embedding 模型。\n检查：\n- "
        + "\n- ".join(checked)
    )


def training_args_kwargs(**kwargs) -> dict:
    """兼容 transformers 不同 4.x 小版本对 eval_strategy 的命名差异。"""
    from transformers import TrainingArguments

    params = inspect.signature(TrainingArguments.__init__).parameters
    out = dict(kwargs)
    if "eval_strategy" not in params and "eval_strategy" in out:
        out["evaluation_strategy"] = out.pop("eval_strategy")
    elif "evaluation_strategy" not in params and "evaluation_strategy" in out:
        out["eval_strategy"] = out.pop("evaluation_strategy")
   
    total = out.pop("_total_steps", None)
    if "warmup_ratio" in out and "warmup_ratio" not in params:
        ratio = float(out.pop("warmup_ratio"))
        if total:
            out["warmup_steps"] = int(math.ceil(ratio * total))
        else:
            out["warmup_steps"] = ratio
        print(f"[COMPAT] transformers {__import__('transformers').__version__}: warmup_ratio={ratio} -> warmup_steps={out['warmup_steps']}"
              f"（总步数 {total}）", flush=True)
    unknown = [k for k in out if k not in params]
    if unknown:
        print(f"[COMPAT][WARN] 当前 transformers 不支持这些 TrainingArguments，已忽略：{unknown}", flush=True)
        for k in unknown:
            out.pop(k)
    return out


def trainer_tokenizer_kw(tokenizer) -> dict:
    from transformers import Trainer

    params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in params:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}




def _is_multimodal_checkpoint(model_path: str) -> bool:
    """Qwen3.5 等原生多模态模型的 config 带 vision_config / text_config，权重前缀是 model.language_model.*。"""
    import json as _json
    p = Path(model_path) / "config.json"
    if not p.is_file():
        return False
    c = _json.loads(p.read_text(encoding="utf-8"))
    return "vision_config" in c or any("ConditionalGeneration" in a for a in c.get("architectures", []))


def load_local_causal_lm(model_path: str, *, dtype, **kwargs):
    """Load a local causal LM with Transformers 4.x/5.x dtype compatibility.

    Newer Transformers deprecates ``torch_dtype`` in favor of ``dtype``.
    Older 4.x releases may not accept ``dtype`` yet, so fall back only when
    the failure is specifically about that keyword.

    Qwen3.5（Qwen3_5ForConditionalGeneration）是原生多模态模型：必须用图文类加载整个 checkpoint，
    只喂文本即可。若用纯文本类加载，权重前缀对不上，会静默得到随机初始化的语言层——所以这里按 config
    选类，并检查加载报告，缺权重时直接报错。
    """
    from transformers import AutoModelForCausalLM

    classes = [AutoModelForCausalLM]
    if _is_multimodal_checkpoint(model_path):
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError as exc:
            raise ImportError("该基座是多模态 checkpoint（如 Qwen3.5），需要支持它的新版 transformers："
                              "pip install -U transformers，并确认 `from transformers import Qwen3_5ForConditionalGeneration` 可用") from exc
        classes = [AutoModelForImageTextToText]

    def _load(cls, dkey):
        return cls.from_pretrained(model_path, **{dkey: dtype}, output_loading_info=True, **kwargs)

    cls = classes[0]
    try:
        model, info = _load(cls, "dtype")
    except (TypeError, ValueError) as exc:
        msg = str(exc)
        if "dtype" not in msg or "unexpected" not in msg.lower():
            raise
        model, info = _load(cls, "torch_dtype")
    missing = [k for k in (info or {}).get("missing_keys", []) if "lm_head" not in k and "mtp" not in k]
    if missing:
        raise RuntimeError(f"基座权重没有完整加载（缺 {len(missing)} 个，如 {missing[:3]}）。"
                           "多半是 transformers 版本不支持该模型或用错了模型类，请升级 transformers 后重试。")
    return model


def lora_target_modules(model, names: list[str]):
    
    all_names = [n for n, _ in model.named_modules()]
    present = [x for x in names if any(n.endswith("." + x) for n in all_names)]
    absent = [x for x in names if x not in present]
    if absent:
        print(f"[LoRA] 基座中没有这些层，已跳过：{absent}")
    if not present:
        raise ValueError(f"target_modules {names} 在基座中一个都没找到")
    if any((".visual." in n or n.startswith("visual") or ".mtp." in n or n.startswith("mtp")) for n in all_names):
        import re as _re
        alt = "|".join(_re.escape(x) for x in present)
        return rf"^(?!.*(?:visual|mtp)).*\.(?:{alt})$"
    return present


def decoder_layer_classes(model) -> list[str]:
    """FSDP 要包裹的解码层类名（从模型的 _no_split_modules 读，去掉视觉块）。"""
    names = list(getattr(model, "_no_split_modules", None) or [])
    names = [n for n in names if "Vision" not in n and "Visual" not in n]
    return names or ["Qwen2DecoderLayer"]


def cuda_memory_gib(device: int = 0) -> tuple[float, float]:
    """Return (free_GiB, total_GiB) for the selected visible CUDA device."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    free_b, total_b = torch.cuda.mem_get_info(device)
    gib = 1024 ** 3
    return free_b / gib, total_b / gib

def parse_json_object(text: str) -> dict:
    """Robustly extract one JSON object from LLM output."""
    if text is None:
        raise ValueError("模型输出为空。")
    raw = str(text).strip()
    if not raw:
        raise ValueError("模型输出为空。")

    def _loads(candidate: str):
        obj = json.loads(candidate)
        if not isinstance(obj, dict):
            raise ValueError("模型输出 JSON 不是对象。")
        return obj

    s = re.sub(r"^\s*```(?:json|JSON)?\s*", "", raw)
    s = re.sub(r"\s*```\s*$", "", s)
    try:
        return _loads(s)
    except Exception:
        pass

    starts = [i for i, ch in enumerate(s) if ch == "{"]
    for start in starts:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(s)):
            ch = s[i]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start:i + 1]
                    try:
                        return _loads(candidate)
                    except Exception:
                        break

    cleaned = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            return _loads(cleaned[start:end + 1])
        except Exception:
            pass

    if start >= 0:
        candidate = cleaned[start:].strip()
        candidate = re.sub(r"\s*```.*$", "", candidate, flags=re.S)
        quote_count = 0
        escape = False
        for ch in candidate:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                quote_count += 1
        if quote_count % 2 == 1:
            candidate += '"'
        candidate = re.sub(r"[,\s]+$", "", candidate)

        stack = []
        in_string = False
        escape = False
        for ch in candidate:
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch in "[{":
                stack.append(ch)
            elif ch == "]" and stack and stack[-1] == "[":
                stack.pop()
            elif ch == "}" and stack and stack[-1] == "{":
                stack.pop()
        candidate += "".join("]" if ch == "[" else "}" for ch in reversed(stack))
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return _loads(candidate)
        except Exception:
            pass

    preview = raw[:2000].replace("\n", "\\n")
    raise ValueError("模型输出中未找到可解析 JSON 对象。 RAW_OUTPUT_PREVIEW=" + preview)



def selective_causal_lm_loss(model, inputs: dict, labels, weights=None):

    import torch.nn.functional as F
    shift = labels[:, 1:]
    keep = (shift != -100).any(dim=0).nonzero(as_tuple=True)[0]
    if keep.numel() == 0:
        keep = torch.tensor([labels.shape[1] - 2], device=labels.device)
    out = model(**inputs, logits_to_keep=keep, use_cache=False)
    logits = out.logits.float()                                      # [B, K, V]
    tgt = shift[:, keep]
    if (tgt != -100).any() and weights is not None:
      
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100, reduction="none")
        w = weights[:, 1:][:, keep].to(ce.device).reshape(-1) * (tgt.reshape(-1) != -100).float()
        loss = (ce * w).sum() / w.sum().clamp_min(1e-6)
    elif (tgt != -100).any():
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100)
    else:
        loss = logits.sum() * 0.0
    return loss, out
