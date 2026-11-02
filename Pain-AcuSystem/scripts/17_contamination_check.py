#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tcm_study.acupoints import AcupointLexicon
from tcm_study.arms import HFGenerator, OpenAICompat
from tcm_study.config import load_cfg
from tcm_study.normalize import normalize
from tcm_study.package import Package

SYS_CONT = "你是医学病历文本复原助手。下面给出一份针灸科病历的前半部分，请从中断处开始，逐字续写这份病历原文的后续内容（不要改写、不要总结、不要加解释）。"
SYS_RX = "你是针灸科医师。请只输出该病例医师实际采用的针灸取穴（穴名，用、分隔），不要输出其他内容。"


def lcs_len(a: str, b: str) -> int:
    """Longest common substring length (dynamic programming on chars; inputs are short)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def ngram_overlap(a: str, b: str, n: int = 4) -> float:
    ga = {a[i:i + n] for i in range(len(a) - n + 1)}
    gb = {b[i:i + n] for i in range(len(b) - n + 1)}
    return len(ga & gb) / len(gb) if gb else 0.0


def clean(t: str) -> str:
    return re.sub(r"\s+", "", t or "")


def split_prefix(text: str, frac: float) -> tuple[str, str]:
    cut = int(len(text) * frac)
    m = max(text.rfind(p, 0, cut) for p in "。；，\n")
    if m > cut // 2:
        cut = m + 1
    return text[:cut], text[cut:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/study.yaml")
    ap.add_argument("--backend", choices=["a5", "base", "lora"], required=True)
    ap.add_argument("--confirm", action="store_true", help="")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    cc = cfg.get("contamination", {})
    n_ref, frac, min_verb = int(cc.get("n_reference_cases", 55)), float(cc.get("prefix_fraction", 0.55)), int(cc.get("min_verbatim_chars", 25))
    pkg = Package(cfg["paths"]["package_dir"])
    teaching = set(cfg["stats"]["provenance_teaching_sources"])
    ann = pkg.annotation().set_index("case_id")
    lex = AcupointLexicon(pkg.files["acupoint_lib"])
    seed = int(cfg["study"]["seed"])

    def group_of(cid: str) -> str:
        src = ann.loc[cid, "来源"] if cid in ann.index else ""
        return "teaching" if src in teaching else "hospital"

    items = []
    for r in pkg.test_initial():
        items.append({"set": "test", "group": group_of(r["case_id"]), "case_id": r["case_id"],
                      "text": "\n".join(m["content"] for m in r["messages"] if m["role"] == "user"), "reference": r["reference"]})
    test_ids = {i["case_id"] for i in items}
    train = [r for r in pkg.train() if r["meta"].get("task") == "initial" and r["meta"]["case_id"] not in test_ids]
    rnd = random.Random(seed)
    
    train_groups = ("teaching",) if a.backend == "a5" else ("hospital", "teaching")
    for grp in ("hospital", "teaching"):
        pool = [r for r in train if group_of(r["meta"]["case_id"]) == grp]
        rnd.shuffle(pool)
        if grp not in train_groups:
            continue
        for r in pool[:n_ref]:
            items.append({"set": "train", "group": grp, "case_id": r["meta"]["case_id"],
                          "text": "\n".join(m["content"] for m in r["messages"] if m["role"] == "user"), "reference": r["messages"][-1]["content"]})
    if a.limit:
        items = items[: a.limit]
    n_tr = {g: sum(i["set"] == "train" and i["group"] == g for i in items) for g in ("hospital", "teaching")}
    print(f"[CONTAM] backend={a.backend} items={len(items)} (test {sum(i['set']=='test' for i in items)}, "
          f"train hospital/teaching {n_tr['hospital']}/{n_tr['teaching']}"
          f"{'; hospital training cases are NOT sent to the API, use set=test/group=hospital as the unpublished contrast' if a.backend == 'a5' else ''})")

    gc = cfg["generation"]
    if a.backend == "a5":
        if not a.confirm:
            raise SystemExit("[BLOCKED] --backend a5 sends the 55 test cases and 55 published teaching cases to the A5 API "
                             "(no hospital training records). Add --confirm if the ethics approval covers this.")
        c5 = cfg["arms"]["A5"]
        
        xb = {k: v for k, v in (c5.get("extra_body") or {}).items() if k not in ("thinking_budget", "chat_template_kwargs")}
        if "enable_thinking" in xb or "dashscope" in c5["base_url"]:
            xb["enable_thinking"] = False
        if "chat_template_kwargs" in (c5.get("extra_body") or {}):
            xb["chat_template_kwargs"] = {**c5["extra_body"]["chat_template_kwargs"], "enable_thinking": False}
        gen = OpenAICompat(c5["base_url"], c5["model"], c5["api_key_env"], 400, loop_retry=False, extra_body=xb,
                           thinking={"enabled": False}, stream=bool(c5.get("stream", False)))
    else:
        gen = HFGenerator(cfg["models"]["base_llm"], cfg["outputs"]["lora_adapter"] if a.backend == "lora" else None, 400, 1.0, loop_retry=False,
                          chat_template_kwargs={**(cfg["models"].get("chat_template_kwargs") or {}), "enable_thinking": False},
                          thinking={"enabled": False})

    rows, t0 = [], time.time()
    for k, it in enumerate(items, 1):
        body = re.sub(r"^请根据以下病例给出针灸诊疗方案。\s*", "", it["text"]).strip()
        pre, rest = split_prefix(body, frac)
        cont = gen.generate([{"role": "system", "content": SYS_CONT}, {"role": "user", "content": pre}])["text"]
        c_rest, c_cont = clean(rest)[:300], clean(cont)[:300]
        rx = gen.generate([{"role": "system", "content": SYS_RX}, {"role": "user", "content": body}])["text"]
        ref_pts = set(lex.expand(lex.extract(normalize(it["reference"])[0].get("取穴", ""))))
        out_pts = set(lex.expand(lex.extract(rx)))
        jac = len(ref_pts & out_pts) / len(ref_pts | out_pts) if (ref_pts | out_pts) else float("nan")
        rows.append({"set": it["set"], "group": it["group"], "case_id": it["case_id"], "lcs_chars": lcs_len(c_rest, c_cont),
                     "ngram4_overlap": round(ngram_overlap(c_rest, c_cont), 4), "rx_jaccard": round(jac, 4) if jac == jac else None,
                     "continuation": cont[:300], "rx_output": rx[:200]})
        if k % 10 == 0:
            print(f"[CONTAM] {k}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    gen.close()
    out = Path(cfg["paths"]["work_dir"]) / "contamination"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{a.backend}_rows.csv", index=False, encoding="utf-8-sig")
    df["suspect"] = df["lcs_chars"] >= min_verb
    g = df.groupby(["set", "group"]).agg(n=("case_id", "count"), lcs_median=("lcs_chars", "median"), lcs_max=("lcs_chars", "max"),
                                        suspect_n=("suspect", "sum"), ngram4_mean=("ngram4_overlap", "mean"), rx_jaccard_mean=("rx_jaccard", "mean")).reset_index()
    L = [f"# 污染/记忆探针结果（backend={a.backend}，{time.strftime('%F %T')}）", "",
         f"判读：若 test/teaching 与 train/teaching 的最长公共子串、4-gram 重合明显高于 hospital 组，提示已发表医案可能进入预训练；疑似阈值 {min_verb} 字。",
         *(["注：a5 后端不向 API 传输训练集医院病例；未发表病例的对照为测试集内的 35 例医院病例（set=test, group=hospital）。"] if a.backend == "a5" else []),
         "", g.to_markdown(index=False) if hasattr(g, "to_markdown") else g.to_string(index=False), "",
         "## 疑似逐字记忆的病例", ""]
    for r in df[df["suspect"]].sort_values("lcs_chars", ascending=False).itertuples():
        L.append(f"- {r.set}/{r.group} {r.case_id}: LCS={r.lcs_chars} 字, 4-gram={r.ngram4_overlap}")
    (out / f"{a.backend}_summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:12]))
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
