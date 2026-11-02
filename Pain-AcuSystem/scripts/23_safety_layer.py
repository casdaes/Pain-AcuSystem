#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tcm_study.acupoints import AcupointLexicon     # noqa: E402
from tcm_study.config import load_cfg              # noqa: E402
from tcm_study.package import Package              # noqa: E402
from tcm_study.safety_layer import apply           # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "study.yaml"))
    ap.add_argument("--set", choices=["initial", "followup", "pilot"], default="initial")
    ap.add_argument("--arms", nargs="+", default=["A1", "A2", "A3", "A4", "A5"])
    ap.add_argument("--work", default="", help="覆盖 paths.work_dir（一般不用）")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    pkg = Package(cfg["paths"]["package_dir"])
    lex = AcupointLexicon(pkg.files["acupoint_lib"])
    work = Path(a.work or cfg["paths"]["work_dir"])
    src, dst = work / "outputs" / a.set, work / "outputs" / f"{a.set}_safety"
    dst.mkdir(parents=True, exist_ok=True)
    spec = __import__("importlib.util").util.spec_from_file_location("gen05", ROOT / "scripts" / "05_generate_arms.py")
    m = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(m)
    inputs = {it["case_id"]: "\n".join(x["content"] for x in it["messages"] if x["role"] == "user") for it in m.load_items(cfg, pkg, a.set)}
    for arm in a.arms:
        p = src / f"raw_{arm}.jsonl"
        if not p.exists():
            print(f"[SKIP] {p} 不存在")
            continue
        rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
        cnt, n_touched, removed = Counter(), 0, 0
        with (dst / f"raw_{arm}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                new, rep = apply(lex, r["raw_output"], inputs.get(r["case_id"], ""))
                touched = new != r["raw_output"]
                n_touched += touched
                removed += len(rep["removed"])
                cnt.update(x.split(":")[0] for x in rep["added"])
                f.write(json.dumps({**r, "raw_output_model": r["raw_output"], "raw_output": new, "safety_layer": rep}, ensure_ascii=False) + "\n")
        print(f"{arm}: {len(rows)} 例，规则层改动 {n_touched} 例 | 追加 {dict(cnt)} | 删除与性别/年龄不符的妊娠句 {removed} 处")
    print(f"[OK] {dst}  （模型 + 规则层 = 新系统，只作单独/次要分析）")


if __name__ == "__main__":
    main()
