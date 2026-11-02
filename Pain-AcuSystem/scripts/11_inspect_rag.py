#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tcm_study.config import load_cfg
from tcm_study.package import Package
from tcm_study.rag import RagEvidence, RagIndex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/study.yaml")
    ap.add_argument("--case-id", required=True)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    pkg = Package(cfg["paths"]["package_dir"])
    if a.case_id in set(pkg.testlist()["case_id"]):
        raise SystemExit("[BLOCKED] refusing to inspect retrieval on a TEST case (would allow tuning on the test set)")
    row = next(r for r in pkg.train() if r["meta"]["case_id"] == a.case_id and r["meta"].get("task") == "initial")
    ev = RagEvidence(RagIndex(cfg["outputs"]["rag_index"], cfg["rag"]["device"]), cfg["rag"])
    text, log = ev.build(RagEvidence.query_from_messages(row["messages"][:-1]))
    print(text)
    print(log)


if __name__ == "__main__":
    main()
