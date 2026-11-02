# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import json
import os
import re
import sys
import io

from . import config as C

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def read_text(arg: str) -> str:
    if arg.startswith("@"):
        p = arg[1:]
        raw = open(p, encoding="utf-8").read()
        if p.endswith(".json") or p.endswith(".jsonl"):
            d = json.loads(raw.splitlines()[0] if p.endswith(".jsonl") else raw)
            return case_from_record(d)
        return raw
    return arg


def case_from_record(d: dict) -> str:
    
    msgs = d.get("messages") or []
    u = "\n".join(m["content"] for m in msgs if m.get("role") == "user")
    return re.sub(r"^请根据以下病例给出针灸诊疗方案。\s*", "", u, flags=re.M).strip() or d.get("input", "")


def cmd_build(a):
    from .build_kb import build
    build(write_md=not a.no_md)


def cmd_index(a):
    from .index import build_bm25, build_dense, write_manifest
    bm = build_bm25(C.KB_DIR)
    de = {}
    if a.dense:
        de = build_dense(C.KB_DIR, a.model, a.device, a.batch)
    write_manifest(C.KB_DIR, bm, de, a.model or C.EMBED_MODEL)


def cmd_query(a):
    from .retriever import PainRAG
    rag = PainRAG(use_dense="off" if a.no_dense else "auto")
    q = read_text(a.text)
    filt = (lambda d: d["section"] == a.section) if a.section else None
    for s, d in rag.search(q, a.domain, a.k, filt):
        print(f"\n=== {s:.3f}  {d['id']}")
        print(d["text"][: a.chars])


def cmd_route(a):
    from .retriever import PainRAG
    rag = PainRAG(use_dense="off" if a.no_dense else "auto")
    for cid, s, rs in rag.route(read_text(a.text), a.k):
        c = rag.concepts[cid]
        print(f"{s:.3f}  {cid}  {c['name']}  {'/'.join(rs)}")


def cmd_evidence(a):
    from .retriever import PainRAG
    rag = PainRAG(use_dense="off" if a.no_dense else "auto")
    budget = {k: v for k, v in (("west", a.budget_west), ("tcm", a.budget_tcm), ("acupoint", a.budget_acu)) if v}
    domains = tuple(a.domains.split(","))
    text, trace = rag.evidence(read_text(a.text), top_concepts=a.top, budget=budget, domains=domains)
    print(text)
    print("\n--- trace ---")
    print(json.dumps({k: v for k, v in trace.items() if k != "routed_all"}, ensure_ascii=False, indent=1))
    if a.json:
        json.dump({"evidence": text, "trace": trace}, open(a.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def cmd_eval(a):
    
    from .retriever import PainRAG
    from .textutil import norm_key
    rag = PainRAG(use_dense="off" if a.no_dense else "auto")
    recs = [json.loads(l) for l in open(a.cases, encoding="utf-8") if l.strip()]
    recs = [r for r in recs if r.get("meta", {}).get("task", "initial") == "initial"]
    if a.n:
        recs = recs[: a.n]
    fmap = {}
    if a.findings:
        fmap = {d["case_id"]: d for d in (json.loads(l) for l in open(a.findings, encoding="utf-8") if l.strip())}
        print(f"[结构化路由] 要素文件 {a.findings}：{len(fmap)} 例")

    def names_of(cid):
        c = rag.concepts[cid]
        return [norm_key(x) for x in [c["name"]] + c.get("aliases", []) + c.get("west_names", []) if len(norm_key(x)) >= 2]

    def match(dx, cid):
        dxk = norm_key(dx)
        return any(n == dxk or n in dxk or (len(n) >= 3 and dxk in n) for n in names_of(cid))

    all_names = {n: cid for cid in rag.concepts if not cid.startswith("AP-") for n in names_of(cid)}
    tot = cov = hit1 = hit2 = none = 0
    miss = []
    for r in recs:
        ans = "\n".join(m["content"] for m in r.get("messages", []) if m.get("role") == "assistant")
        m = re.search(r"【西医诊断】\s*([^\n]{2,60})", ans)
        if not m:
            continue
        dx = re.split(r"[；;（(，,]", m.group(1).strip())[0]
        tot += 1
        dxk = norm_key(dx)
        covered = any(n == dxk or n in dxk or (len(n) >= 3 and dxk in n) for n in all_names)
        cov += covered
        routed = rag.route(case_from_record(r), 2, fmap.get(r.get("meta", {}).get("case_id")))
        if not routed or routed[0][1] < 0.3:
            none += 1
        h1 = bool(routed) and match(dx, routed[0][0])
        h2 = h1 or (len(routed) > 1 and match(dx, routed[1][0]))
        hit1 += h1
        hit2 += h2
        if covered and not h2 and len(miss) < 40:
            miss.append((r.get("meta", {}).get("case_id"), dx, [(c, s, rag.concepts[c]["name"]) for c, s, _ in routed]))
    print(f"病例 {tot}｜诊断在库内有概念可对应 {cov}（{cov/tot:.1%}）｜Top1 命中 {hit1}（{hit1/tot:.1%}；占可对应 {hit1/max(cov,1):.1%}）"
          f"｜Top2 命中 {hit2}（{hit2/tot:.1%}；占可对应 {hit2/max(cov,1):.1%}）｜无概念达阈值 {none}")
    print("\n未命中样例（可对应但路由错）：")
    for x in miss:
        print(" ", x)


def cmd_stats(a):
    p = os.path.join(C.KB_DIR, "build_report.md")
    print(open(p, encoding="utf-8").read()[:3000] if os.path.exists(p) else "尚未 build")
    m = os.path.join(C.KB_DIR, "manifest.json")
    if os.path.exists(m):
        print(open(m, encoding="utf-8").read())


def main():
    ap = argparse.ArgumentParser(prog="painrag")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("build"); p.add_argument("--no-md", action="store_true"); p.set_defaults(f=cmd_build)
    p = sp.add_parser("index"); p.add_argument("--dense", action="store_true"); p.add_argument("--model", default=None)
    p.add_argument("--device", default=None); p.add_argument("--batch", type=int, default=32); p.set_defaults(f=cmd_index)
    p = sp.add_parser("query"); p.add_argument("text"); p.add_argument("--domain", default="west", choices=list(C.COLLECTIONS))
    p.add_argument("--k", type=int, default=5); p.add_argument("--section", default=None); p.add_argument("--chars", type=int, default=500)
    p.add_argument("--no-dense", action="store_true"); p.set_defaults(f=cmd_query)
    p = sp.add_parser("route"); p.add_argument("text"); p.add_argument("--k", type=int, default=5); p.add_argument("--no-dense", action="store_true"); p.set_defaults(f=cmd_route)
    p = sp.add_parser("evidence"); p.add_argument("text"); p.add_argument("--top", type=int, default=2)
    p.add_argument("--budget-west", type=int, default=0); p.add_argument("--budget-tcm", type=int, default=0); p.add_argument("--budget-acu", type=int, default=0)
    p.add_argument("--domains", default="west,tcm,acupoint"); p.add_argument("--json", default=None); p.add_argument("--no-dense", action="store_true"); p.set_defaults(f=cmd_evidence)
    p = sp.add_parser("eval-routing"); p.add_argument("--cases", required=True); p.add_argument("--n", type=int, default=0); p.add_argument("--no-dense", action="store_true"); p.add_argument("--findings", default=None); p.set_defaults(f=cmd_eval)
    p = sp.add_parser("stats"); p.set_defaults(f=cmd_stats)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
