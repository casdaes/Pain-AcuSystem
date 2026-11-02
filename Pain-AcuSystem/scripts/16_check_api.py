#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tcm_study.config import load_cfg

PING = [{"role": "system", "content": "你是测试助手。"}, {"role": "user", "content": "请只回复四个字：连接正常"}]


def call(base_url: str, model: str, key: str, body_extra: dict) -> tuple[bool, str, float]:
    body = json.dumps({"model": model, "messages": PING, "max_tokens": 20, **body_extra}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read().decode())
        return True, d["choices"][0]["message"]["content"].strip(), time.time() - t0
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.read().decode(errors='ignore')[:300]}", time.time() - t0
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", time.time() - t0


def check(label: str, base_url: str, model: str, key_env: str, body_extra: dict | None = None, greedy_only: bool = False) -> bool:
    print(f"\n== {label}: {model} @ {base_url}")
    if not base_url or not model:
        print("   (未配置，跳过)")
        return False
    key = os.environ.get(key_env, "")
    local = base_url.startswith("http://127.0.0.1") or "localhost" in base_url
    if not key and not local:
        print(f"   [FAIL] 环境变量 {key_env} 未设置")
        return False
    if not local:
        print("   [NOTE] 这是云端端点：正式生成/教师改写时病例文本会发送到该服务，请核对脱敏与 Methods 描述")
    ok = True
    for name, extra in (("greedy(temperature=0)", {"temperature": 0}),
                        ("retry(sampling+seed)", {"temperature": 0.7, "top_p": 0.9, "seed": 20260920}))[:1 if greedy_only else 2]:
        good, msg, dt = call(base_url, model, key or "EMPTY", {**(body_extra or {}), **extra})
        print(f"   [{'OK' if good else 'FAIL'}] {name:24s} {dt:5.1f}s  {msg[:120]}")
        ok = ok and good
    return ok


def check_a5(cfg: dict) -> bool:
    """按正式生成的同一条代码路径（tcm_study.arms.OpenAICompat：思考模式、流式、采样参数、种子）发一句测试话。"""
    from tcm_study.arms import OpenAICompat, thinking_cfg
    c5 = cfg["arms"]["A5"]
    th = {**thinking_cfg(cfg), "budget_tokens": 256}
    xb = dict(c5.get("extra_body") or {})
    if "thinking_budget" in xb:
        xb["thinking_budget"] = 256
    print(f"\n== A5（正式代码路径）: {c5['model']} @ {c5['base_url']}  thinking={th['enabled']} stream={bool(c5.get('stream'))} extra_body={xb}")
    if not os.environ.get(c5.get("api_key_env", "LLM_API_KEY")):
        print(f"   [FAIL] 环境变量 {c5.get('api_key_env', 'LLM_API_KEY')} 未设置")
        return False
    g = OpenAICompat(c5["base_url"], c5["model"], c5.get("api_key_env", "LLM_API_KEY"), 60, loop_retry=False, extra_body=xb,
                     thinking=th, stream=bool(c5.get("stream", False)))
    ok = True
    for attempt, name in ((0, "首次请求"), (1, "重试请求（换种子）")):
        t0 = time.time()
        try:
            r = g._parse(g._call(PING, attempt))
        except Exception as e:
            print(f"   [FAIL] {name}: {str(e)[:300]}")
            ok = False
            continue
        think_ok = (bool(r["reasoning"]) == bool(th["enabled"]))
        print(f"   [{'OK' if r['text'] and think_ok else 'WARN'}] {name} {time.time() - t0:5.1f}s  正文={r['text'][:40]!r}  "
              f"思考={len(r['reasoning'] or '')} 字  usage: completion={r['completion_tokens']} reasoning={r['reasoning_tokens']}  finish={r['finish_reason']}")
        if not think_ok:
            print("   [WARN] " + ("配置要求思考，但没有返回思考内容：检查 extra_body（百炼 enable_thinking / vLLM chat_template_kwargs）"
                                if th["enabled"] else "配置不要求思考，却返回了思考内容：检查 extra_body"))
        ok = ok and bool(r["text"])
    return ok


def main() -> None:
    from tcm_study.net import force_ipv4
    force_ipv4()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/study.yaml")
    ap.add_argument("--which", choices=["all", "a5", "teacher"], default="all")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    res = []
    if a.which in ("all", "a5"):
        c5 = cfg["arms"]["A5"]
        if c5["backend"] == "openai_compatible":
            res.append(check_a5(cfg))
        else:
            print(f"\n== A5: backend=hf ({c5.get('hf_path')})，本脚本不检查本地权重")
    if a.which in ("all", "teacher"):
        t = cfg.get("teacher") or {}
        if t.get("base_url") or t.get("model"):
            res.append(check("teacher", t.get("base_url", ""), t.get("model", ""), t.get("api_key_env", "LLM_API_KEY"), t.get("extra_body"), greedy_only=True))
        else:
            print("\n== teacher: 未启用（study.yaml 的 teacher 块留空；本研究不使用教师模型改写训练目标）")
    if not res:
        print("\n[RESULT] 没有需要检查的云端端点")
    else:
        print("\n[RESULT]", "全部通过" if all(res) else "有失败项，见上")


if __name__ == "__main__":
    main()
