#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tcm_study.config import load_cfg
from tcm_study.package import Package
from tcm_study.rag import RagEvidence, RagIndex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/study.yaml")
    ap.add_argument("--no-dense", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    kb = cfg["paths"]["painrag_kb"]
    from painrag.index import build_bm25, build_dense, write_manifest
    bm = build_bm25(kb)
    de = {}
    if not a.no_dense:
        de = build_dense(kb, cfg["models"]["embedding"], cfg["rag"].get("device", "cpu"), int(cfg["rag"].get("batch_size", 32)))
    write_manifest(kb, bm, de, cfg["models"]["embedding"])
    print(json.dumps({"bm25": bm, "dense": de}, ensure_ascii=False))
   
    pkg = Package(cfg["paths"]["package_dir"])
    row = next(r for r in pkg.train() if r["meta"].get("task") == "initial")
    ev = RagEvidence(RagIndex(kb, cfg["rag"].get("device", "cpu")), cfg["rag"])
    text, log = ev.build(RagEvidence.query_from_messages(row["messages"][:-1]))
    print(f"[SMOKE] {row['meta']['case_id']}: dense={log['dense']} chars={log['evidence_chars']} concepts={[c['name'] for c in (log['concepts'] or [])]}")
    print(text[:600])


if __name__ == "__main__":
    main()
