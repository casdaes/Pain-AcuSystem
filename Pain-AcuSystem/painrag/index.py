# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import json
import time

from . import config as C
from .bm25 import BM25


def load_docs(kb_dir: str, cat: str) -> list:
    p = os.path.join(kb_dir, f"{cat}.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def build_bm25(kb_dir: str = None) -> dict:
    kb_dir = kb_dir or C.KB_DIR
    man = {}
    for cat in C.COLLECTIONS:
        docs = load_docs(kb_dir, cat)
        if not docs:
            continue
        t0 = time.time()
        BM25().fit([d["text"] for d in docs]).save(os.path.join(kb_dir, f"{cat}.bm25"))
        man[cat] = len(docs)
        print(f"[BM25] {cat:9s} docs={len(docs):6d}  {time.time() - t0:.1f}s", flush=True)
    return man


def build_dense(kb_dir: str = None, model: str = None, device: str = None, batch_size: int = 32) -> dict:
    kb_dir, model, device = kb_dir or C.KB_DIR, model or C.EMBED_MODEL, device or C.EMBED_DEVICE
    import numpy as np
    from sentence_transformers import SentenceTransformer
    try:
        import faiss
    except Exception:
        faiss = None
    enc = SentenceTransformer(model, device=device)
    man = {}
    for cat in C.COLLECTIONS:
        docs = load_docs(kb_dir, cat)
        if not docs:
            continue
        t0 = time.time()
        vec = enc.encode([d["text"] for d in docs], batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True).astype("float32")
        np.save(os.path.join(kb_dir, f"{cat}.npy"), vec)
        if faiss is not None:
            idx = faiss.IndexFlatIP(vec.shape[1])
            idx.add(vec)
            faiss.write_index(idx, os.path.join(kb_dir, f"{cat}.faiss"))
        man[cat] = len(docs)
        print(f"[DENSE] {cat:9s} docs={len(docs):6d} dim={vec.shape[1]}  {time.time() - t0:.0f}s", flush=True)
    open(os.path.join(kb_dir, "embedding_model.txt"), "w", encoding="utf-8").write(model)
    return man


def write_manifest(kb_dir: str, bm25: dict, dense: dict, model: str):
    kb_dir = kb_dir or C.KB_DIR
    p = os.path.join(kb_dir, "manifest.json")
    old = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    old.update({"built_at": time.strftime("%Y-%m-%d %H:%M:%S"), "bm25": bm25 or old.get("bm25"),
                "dense": dense or old.get("dense"), "encoder": model if dense else old.get("encoder"),
                "categories": {c: (bm25 or old.get("bm25") or {}).get(c) for c in C.COLLECTIONS}})
    json.dump(old, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
