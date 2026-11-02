#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tcm_study import ratings
from tcm_study.config import load_cfg
from tcm_study.package import Package
from tcm_study.state import StudyState, sha256_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/study.yaml")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--collect", action="store_true")
    g.add_argument("--lock", action="store_true")
    g.add_argument("--stage2-sheets", action="store_true")
    g.add_argument("--stage2-collect", action="store_true")
    g.add_argument("--unblind", action="store_true")
    ap.add_argument("--skip-stage2", action="store_true", help="unblind without stage-2 concordance ratings (they will be missing in the analysis)")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    work = Path(cfg["paths"]["work_dir"])
    state = StudyState(work)
    pkg = Package(cfg["paths"]["package_dir"])
    rd = work / "rating" / "returned"
    if a.collect:
        state.require("sheets_made"); state.forbid("locked", force=False)
        info = ratings.collect(rd, work / "rating", cfg["rating"]["raters"], pkg.assignment())
        state.advance("scored", info)
        print(json.dumps(info, ensure_ascii=False))
        if info.get("arbitrators"):
            print(f"[ARB] arbitrators {info['arbitrators']}")
        elif info.get("n_majority_disagreements"):
            print(f"[MAJORITY] {info['n_majority_disagreements']}")
    elif a.lock:
        state.require("scored"); state.forbid("locked", force=False)
        info = ratings.lock(work / "rating")
        state.advance("locked", info)
        print(json.dumps(info, ensure_ascii=False))
       
    elif a.stage2_sheets:
        state.require("locked")
        info = ratings.make_stage2_sheets(pkg, work / "rating", int(cfg["study"]["seed"]))
        state.advance("stage2_made", info)
        print(json.dumps(info, ensure_ascii=False))
    
    elif a.stage2_collect:
        state.require("stage2_made")
        info = ratings.collect_stage2(rd, work / "rating", cfg["rating"]["raters"])
        state.advance("stage2_scored", info)
        print(json.dumps(info, ensure_ascii=False))
    else:
        state.require("locked"); state.forbid("unblinded", force=False)
        if cfg["rating"].get("stage2_concordance", True) and not state.done("stage2_scored") and not a.skip_stage2:
            raise SystemExit("[BLOCKED] stage-2 concordance ratings not collected (--stage2-sheets / --stage2-collect); use --skip-stage2 to unblind without them")
        lk = state.data["stages"]["locked"]["meta"]
        if sha256_file(work / "rating/outputs_final_blinded.csv") != lk["final_sha256"] or sha256_file(work / "rating/ratings_long_blinded.csv") != lk["long_sha256"]:
            raise SystemExit("[ERROR] locked rating files changed after locking")
        lengths = {tuple(k.split("|")): v for k, v in json.loads((work / "SEALED/normalized_lengths.json").read_text(encoding="utf-8")).items()}
        info = ratings.unblind(pkg, cfg, work / "rating", work / "analysis", lengths)
        state.advance("unblinded", info)
        print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
