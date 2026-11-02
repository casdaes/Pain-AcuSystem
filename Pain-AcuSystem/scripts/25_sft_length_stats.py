#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def ids_from_messages(tok, messages, chat_kw):

    full_text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, **chat_kw
    )
    prompt_text = tok.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, **chat_kw
    )
    full_ids = tok(full_text, truncation=False, add_special_tokens=False)["input_ids"]
    prompt_ids = tok(prompt_text, truncation=False, add_special_tokens=False)["input_ids"]

    n = min(len(full_ids), len(prompt_ids))
    cut = 0
    while cut < n and full_ids[cut] == prompt_ids[cut]:
        cut += 1
    if cut < min(len(prompt_ids), 32):
        cut = min(len(prompt_ids), len(full_ids))
    return len(full_ids), len(prompt_ids), max(0, len(full_ids) - cut)


def q(a, p):
    return int(round(float(np.percentile(a, p)))) if len(a) else 0


def report(name, rows, limits):
    a = np.array([r[0] for r in rows], dtype=int)
    ans = np.array([r[2] for r in rows], dtype=int)
    print(f"\n[{name}] n={len(a)}")
    if not len(a):
        return
    print(f"full: mean={int(round(a.mean()))} P50={q(a,50)} P75={q(a,75)} P90={q(a,90)} P95={q(a,95)} P97={q(a,97)} P99={q(a,99)} max={int(a.max())}")
    print(f"answer: mean={int(round(ans.mean()))} P95={q(ans,95)} P99={q(ans,99)} max={int(ans.max())}")
    for lim in limits:
        n = int((a > lim).sum())
        ans_too_long = int((ans > lim - 64).sum())
        print(f">{lim}: full={n}/{len(a)} ({n/len(a)*100:.1f}%), answer-alone-too-long={ans_too_long}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="work/sft/v5")
    ap.add_argument("--model", default="")
    ap.add_argument("--limits", nargs="*", type=int, default=[7168, 8192, 9216, 10240, 12288])
    a = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "configs/study.yaml").read_text(encoding="utf-8"))
    model = a.model or cfg["models"]["base_llm"]
    chat_kw = dict(cfg["models"].get("chat_template_kwargs") or {})
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    all_rows = []
    by_variant = defaultdict(list)
    for split in ("train", "val"):
        fn = ROOT / a.dir / f"{split}.jsonl"
        if not fn.exists():
            continue
        for line in fn.open(encoding="utf-8"):
            x = json.loads(line)
            lens = ids_from_messages(tok, x["messages"], chat_kw)
            all_rows.append(lens)
            by_variant[str((x.get("meta") or {}).get("variant", "?"))].append(lens)

    report("ALL", all_rows, a.limits)
    for k in sorted(by_variant):
        report("variant=" + k, by_variant[k], a.limits)


if __name__ == "__main__":
    main()
