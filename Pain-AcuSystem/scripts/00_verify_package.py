#!/usr/bin/env python3
from __future__ import annotations

"""Audit the data package: counts, distribution, leakage, blind map, rater assignment.  No model needed."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tcm_study.package import Package


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="")
    ap.add_argument("--expect-fingerprint-prefix", default="605c4388", help="")
    a = ap.parse_args()
    rep = Package(a.package).verify()
    print("== 测试集分布（病种 × 任务模式）==")
    d = rep["distribution"].copy()
    d.loc["合计"] = d.sum()
    print(d.to_string())
    for n in rep["notes"]:
        print("  -", n)
    fp = rep["fingerprint_sha256"]
    print(f"test-set fingerprint SHA-256 = {fp}")
    if a.expect_fingerprint_prefix and not fp.startswith(a.expect_fingerprint_prefix):
        rep["errors"].append(f"fingerprint {fp[:8]} != expected prefix {a.expect_fingerprint_prefix}")
    for w in rep["warnings"]:
        print("[WARN]", w)
    for e in rep["errors"]:
        print("[ERROR]", e)
    print("[PASS]" if not rep["errors"] else "[FAIL]")
    Path("work").mkdir(exist_ok=True)
    Path("").write_text(json.dumps({k: (v.to_dict() if hasattr(v, "to_dict") else v) for k, v in rep.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
    sys.exit(0 if not rep["errors"] else 1)


if __name__ == "__main__":
    main()
