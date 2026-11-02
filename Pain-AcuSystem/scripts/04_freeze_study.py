#!/usr/bin/env python3
from __future__ import annotations


import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tcm_study import analysis
from tcm_study.arms import prompt_template_text
from tcm_study.config import load_cfg
from tcm_study.package import Package
from tcm_study.state import StudyState, hash_tree, sha256_file, sha256_json, sha256_text


def compute(cfg: dict, pkg: Package, mode: str) -> dict:
    hashes = {}
    for k, p in (("base_llm", cfg["models"]["base_llm"]), ("lora_adapter", cfg["outputs"]["lora_adapter"]), ("rag_index", cfg["outputs"]["rag_index"])):
        pp = Path(p)
        if not pp.exists():
            hashes[k] = {"digest": "MISSING", "path": p}
            continue
        pats = ("*.safetensors", "*.json", "*.model", "*.txt", "*.tiktoken") if k != "rag_index" else ("*",)
        h = hash_tree(pp, mode, pats)
        hashes[k] = {"digest": h["digest"], "n_files": len(h["files"]), "path": p}
    hb = pd.read_excel(pkg.files["handbook"], sheet_name=None, header=None, dtype=str, keep_default_na=False)
    ti = pkg.test_initial()
    return {
        "test_set_fingerprint": pkg.fingerprint(), "train_jsonl_sha256": sha256_file(pkg.files["train"]),
        "weights": hashes,
        "prompts_sha256": sha256_json({"system": sorted({m["content"] for r in ti for m in r["messages"] if m["role"] == "system"}),
                                       "evidence_wrapper": prompt_template_text(cfg["generation"].get("system_prompt", "dataset"))}),
        "retrieval_config_sha256": sha256_json({"rag": cfg["rag"], "generation": cfg["generation"]}),
        "sap_sha256": sha256_text(hb["09_统计分析计划"].to_csv()), "never_event_list_sha256": sha256_text(hb["02_关键安全错误清单"].to_csv()),
        "assignment_sha256": sha256_text(hb["07_评分者分配"].to_csv()),
        "analysis_plan": analysis.plan(cfg), "rated_arms": list(cfg["rating"].get("rated_arms") or ["A1", "A2", "A3", "A4", "A5"]), "seed": cfg["study"]["seed"],
        "a5": {k: v for k, v in cfg["arms"]["A5"].items() if k != "api_key_env"}, "hash_mode": mode,
    }


def diff(a: dict, b: dict) -> list[str]:
    bad = [k for k in a if k not in ("weights", "hash_mode") and a[k] != b.get(k)]
    bad += [f"weights.{k}" for k, v in a["weights"].items() if v["digest"] != b["weights"][k]["digest"]]
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/study.yaml")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--hash-mode", default="full", choices=["full", "fast"])
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    pkg = Package(cfg["paths"]["package_dir"])
    work = Path(cfg["paths"]["work_dir"])
    state = StudyState(work)
    fpath = work / "STUDY_FREEZE.json"
    if a.verify:
        old = json.loads(fpath.read_text(encoding="utf-8"))
        bad = diff(old, compute(cfg, pkg, old["hash_mode"]))
        if bad:
            raise SystemExit(f"[FREEZE][MISMATCH] {bad}")
        print("[FREEZE][OK] identical to the frozen configuration")
        return
    state.forbid("generated", "frozen", force=False)
    rep = pkg.verify()
    if not rep["ok"]:
        raise SystemExit(f"[BLOCKED] package verification failed: {rep['errors']}")
    state.advance("verified", {"fingerprint": rep["fingerprint_sha256"]}) if not state.done("verified") else None
    problems = []
    pl = analysis.plan(cfg)
    a5 = cfg["arms"]["A5"]
    if a5["backend"] == "openai_compatible" and not a5.get("model") or a5["backend"] == "hf" and not a5.get("hf_path"):
        problems.append("arms.A5 model is not set")
    if not Path(cfg["outputs"]["lora_adapter"]).is_dir():
        problems.append("LoRA adapter missing: run scripts/02_train_sft.py")
    if not Path(cfg["outputs"]["rag_index"]).is_dir():
        problems.append("RAG index missing: run scripts/03_build_rag.py")
    else:
        import importlib.util as _iu
        _spec = _iu.spec_from_file_location("check_rag_index", Path(__file__).resolve().parent / "20_check_rag_index.py")
        _m = _iu.module_from_spec(_spec); _spec.loader.exec_module(_m)
        _kb = _m.check(Path(cfg["outputs"]["rag_index"]))
        problems += [f"knowledge base: {x} (rebuild with scripts/03_build_rag.py)" for x in _kb["problems"]]
    if problems:
        print("[FREEZE][BLOCKED]")
        [print("  -", p) for p in problems]
        raise SystemExit(1)
    frozen = compute(cfg, pkg, a.hash_mode)
    fpath.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    state.advance("frozen", {"primary_endpoint": pl["primary_endpoint"], "n_contrasts": len(pl["contrasts"])})
    print(f"[FREEZE][OK] {fpath}  primary={pl['primary_endpoint']} ({pl['primary_population']}), {len(pl['contrasts'])} contrasts among {pl['compare_arms']}  fingerprint={frozen['test_set_fingerprint'][:12]}…")


if __name__ == "__main__":
    main()
