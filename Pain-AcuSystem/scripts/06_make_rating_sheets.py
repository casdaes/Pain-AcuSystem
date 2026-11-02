#!/usr/bin/env python3
from __future__ import annotations


import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tcm_study.acupoints import AcupointLexicon, main_points, output_autometrics, point_metrics
from tcm_study.config import load_cfg
from tcm_study.normalize import normalize
from tcm_study.package import Package
from tcm_study.ratings import make_sheets
from tcm_study.state import StudyState


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/study.yaml")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    pkg = Package(cfg["paths"]["package_dir"])
    work = Path(cfg["paths"]["work_dir"])
    state = StudyState(work)
    state.require("generated"); state.forbid("sheets_made", force=False)
    lex = AcupointLexicon(pkg.files["acupoint_lib"])
    ti = {r["case_id"]: r for r in pkg.test_initial()}
    texts, rows = {}, []
    frac = float(cfg["stats"]["main_point_fraction"])
    ref_sec = {cid: normalize(r["reference"])[0] for cid, r in ti.items()}
    ref_pts = {cid: lex.expand(lex.extract(s.get("取穴", ""))) for cid, s in ref_sec.items()}
    rated = list(cfg["rating"].get("rated_arms") or ["A1", "A2", "A3", "A4", "A5"])
    for arm in rated:
        for l in (work / "outputs/initial" / f"raw_{arm}.jsonl").open(encoding="utf-8"):
            d = json.loads(l)
            cid = d["case_id"]
            sec, text = normalize(d["raw_output"])
            texts[(cid, arm)] = text
            in_text = "\n".join(m["content"] for m in ti[cid]["messages"] if m["role"] == "user")
            am = output_autometrics(lex, sec, d["raw_output"], in_text, ti[cid]["task"])
            row = {"case_id": cid, "arm": arm, "task": ti[cid]["task"], **{k: v for k, v in am.items() if k not in ("points", "safety_flags", "unknown_tokens", "code_mismatch")},
                   "unknown_tokens": ";".join(am["unknown_tokens"]), "safety_flag_candidates": ";".join(f["candidate"] + ":" + f["point"] for f in am["safety_flags"]),
                   "latency_s": d.get("latency_s"), "peak_vram_gb": d.get("peak_vram_gb")}
            if arm != "A6":
                pts = am["points"]
                row.update(point_metrics(pts, ref_pts[cid], main_points(pts, frac), main_points(ref_pts[cid], frac)))
            rows.append(row)
    sealed = work / "SEALED"
    sealed.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(sealed / "advisory_autometrics.csv", index=False, encoding="utf-8-sig")
    (sealed / "normalized_lengths.json").write_text(json.dumps({f"{c}|{a}": len(t) for (c, a), t in texts.items()}), encoding="utf-8")
    info = make_sheets(pkg, texts, work / "rating", int(cfg["study"]["seed"]), rated)
    (work / "rating" / "blinding_check.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    state.advance("sheets_made", {"n_rows": info["n_rows"]})
    print(json.dumps(info["chars_by_arm"], ensure_ascii=False))
    if info["possible_arm_identifiers_in_text"]:
        print("[WARN] outputs contain possible arm identifiers (see work/rating/blinding_check.json):", info["possible_arm_identifiers_in_text"][:5])


if __name__ == "__main__":
    main()
