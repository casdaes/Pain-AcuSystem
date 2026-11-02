#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

CATS = ("west", "tcm", "acupoint", "concept")


def _npy_rows(path: Path) -> int | None:
    with path.open("rb") as f:
        if f.read(6) != b"\x93NUMPY":
            return None
        major = f.read(2)[0]
        header_len = int.from_bytes(f.read(2 if major == 1 else 4), "little")
        header = ast.literal_eval(f.read(header_len).decode("latin1"))
        start = f.tell()
    shape = header["shape"]
    itemsize = int(header["descr"][-1]) if header["descr"][-1].isdigit() else 4
    n_values = 1
    for dim in shape:
        n_values *= dim
    if path.stat().st_size != start + n_values * itemsize:
        return -1
    return shape[0]


def check(kb: Path, require_dense: bool = True) -> dict:
    report = {"kb": str(kb), "ok": True, "problems": [], "cats": {}}
    manifest = json.loads((kb / "manifest.json").read_text(encoding="utf-8")) if (kb / "manifest.json").exists() else {}
    expected = manifest.get("bm25") or {}
    for cat in CATS:
        info = {}
        source = kb / f"{cat}.jsonl"
        if not source.exists():
            report["problems"].append(f"{cat}.jsonl is missing")
            continue
        n_docs = n_bad = 0
        with source.open("rb") as f:
            for line in f:
                if not line.strip():
                    continue
                n_docs += 1
                try:
                    json.loads(line)
                except Exception:
                    n_bad += 1
        info.update(docs=n_docs, bad_lines=n_bad, bytes=source.stat().st_size)
        if n_bad:
            report["problems"].append(f"{cat}.jsonl contains {n_bad} invalid JSON lines")
        if expected.get(cat) is not None and expected[cat] != n_docs:
            report["problems"].append(f"{cat}.jsonl has {n_docs} rows; manifest expects {expected[cat]}")
        if not (kb / f"{cat}.bm25").exists():
            report["problems"].append(f"{cat}.bm25 is missing")
        dense = kb / f"{cat}.npy"
        if dense.exists():
            rows = _npy_rows(dense)
            info["dense_rows"] = rows
            if rows == -1:
                report["problems"].append(f"{cat}.npy appears truncated")
            elif rows != n_docs:
                report["problems"].append(f"{cat}.npy has {rows} rows; {cat}.jsonl has {n_docs}")
        elif require_dense:
            report["problems"].append(f"{cat}.npy is missing")
        report["cats"][cat] = info
    dense_cats = [cat for cat in CATS if (kb / f"{cat}.npy").exists()]
    if dense_cats and len(dense_cats) != len(CATS):
        report["problems"].append(f"Dense vectors are present for only a subset of collections: {dense_cats}")
    report["dense"] = len(dense_cats) == len(CATS)
    report["ok"] = not report["problems"]
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kb", nargs="?", default="data/painrag_kb")
    ap.add_argument("--allow-bm25-only", action="store_true")
    args = ap.parse_args()
    result = check(Path(args.kb), require_dense=not args.allow_bm25_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
