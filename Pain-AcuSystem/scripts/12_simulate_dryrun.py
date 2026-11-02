#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tcm_study import analysis, ratings
from tcm_study.acupoints import AcupointLexicon, main_points, output_autometrics, point_metrics
from tcm_study.config import load_cfg
from tcm_study.normalize import normalize
from tcm_study.package import Package

P_SE = {"A1": .16, "A2": .11, "A3": .09, "A4": .06, "A5": .05, "A6": .04}
QUAL = {"A1": 5.6, "A2": 6.4, "A3": 6.6, "A4": 7.4, "A5": 7.6, "A6": 7.0}


def fake_output(ref: str, arm: str, rng) -> str:
    sec, _ = normalize(ref)
    pts = sec.get("取穴", "")
    if arm in ("A1", "A2") and rng.random() < .5:
        pts = pts + "、不存在穴"
    body = f"【西医诊断】{sec.get('西医诊断', '')}\n【中医诊断】{sec.get('中医诊断', '')}\n【证型】{sec.get('证型', '')}\n【治法】{sec.get('治法', '')}\n【取穴】{pts}\n【操作】{sec.get('操作', '')[:120]}"
    return ("您好，" if arm == "A5" else "") + body + ("\n以上内容仅供参考。" if arm in ("A1", "A5") else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/study.yaml")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    pkg = Package(cfg["paths"]["package_dir"])
    rng = np.random.default_rng(11)
    tmp = Path(tempfile.mkdtemp(prefix="study_sim_"))
    lex = AcupointLexicon(pkg.files["acupoint_lib"])
    ti = {r["case_id"]: r for r in pkg.test_initial()}
    bmap = pkg.blind_map()
    rated = list(cfg["rating"].get("rated_arms") or ["A1", "A2", "A3", "A4", "A5"])
    texts, lengths, rows = {}, {}, []
    for cid, r in ti.items():
        for arm in rated:
            raw = r["reference"] if arm == "A6" else fake_output(r["reference"], arm, rng)
            sec, text = normalize(raw)
            texts[(cid, arm)] = text
            am = output_autometrics(lex, sec, raw, "\n".join(m["content"] for m in r["messages"] if m["role"] == "user"), r["task"])
            row = {"case_id": cid, "arm": arm, "unknown_rate": am["unknown_rate"]}
            if arm != "A6":
                ref_pts = lex.expand(lex.extract(normalize(r["reference"])[0].get("取穴", "")))
                row.update(point_metrics(am["points"], ref_pts, main_points(am["points"]), main_points(ref_pts)))
            rows.append(row)
    info = ratings.make_sheets(pkg, texts, tmp / "rating", int(cfg["study"]["seed"]), rated)
    print("[SIM] sheets:", len(info["files"]), "files; mean chars by arm:", {k: round(v["mean"]) for k, v in info["chars_by_arm"].items()})
    truth = {}
    code_arm = {r["盲评编码"]: (r["case_id"], r["arm"]) for r in bmap.to_dict("records") if r["arm"] in rated}
    for code, (cid, arm) in code_arm.items():
        truth[code] = sorted({f"SE-{k}" for k in rng.choice([3, 4, 5, 8, 9, 13, 16], size=1)}) if rng.random() < P_SE[arm] else []
    (tmp / "rating" / "returned").mkdir()
    rs = random.Random(3)
    for f in info["files"]:
        df = pd.read_excel(f, sheet_name="评分录入", dtype=str, keep_default_na=False)
        for i, r in df.iterrows():
            cid, arm = code_arm[r["盲评编码"]]
            t = truth[r["盲评编码"]]
            seen = [c for c in t if rng.random() < .85] + ([f"SE-{rng.integers(1, 17)}"] if rng.random() < .02 else [])
            seen = sorted(set(seen))
            q = QUAL[arm] + rng.normal(0, 1.0)
            safety = int(np.clip(round(q), 1, 10)) if not seen else int(np.clip(round(rng.normal(1.5, .5)), 1, 2))
            df.loc[i, "安全性(1-10)"] = str(safety)
            df.loc[i, "准确性(1-10)"] = str(int(np.clip(round(q + rng.normal(0, .8)), 1, 10)))
            df.loc[i, "可用性(1-10)"] = str(int(np.clip(round(q - (1 if arm == "A6" else 0) + rng.normal(0, .8)), 1, 10)))
            df.loc[i, ratings.se_col(df)] = ";".join(seen) if seen else "无"
            for c, vals in ratings.STAGE1_CHOICES.items():
                df.loc[i, c] = rs.choice(vals[:2] if c != "信息缺失识别" else vals[:3])
            df.loc[i, "评分备注"] = ""
        with pd.ExcelWriter(tmp / "rating/returned" / Path(f).name) as w:
            df.to_excel(w, sheet_name="评分录入", index=False)
    cinfo = ratings.collect(tmp / "rating/returned", tmp / "rating", cfg["rating"]["raters"], pkg.assignment())
    ap_x = tmp / "rating/adjudication_third_rater.xlsx"
    if ap_x.exists():
        d = pd.read_excel(ap_x, dtype=str, keep_default_na=False)
        for i, r in d.iterrows():
            d.loc[i, "仲裁_命中的SE编号(无则填无)"] = ";".join(truth[r["盲评编码"]]) or "无"
        d.to_excel(ap_x, index=False)
    linfo = ratings.lock(tmp / "rating")
    
    s2info = ratings.make_stage2_sheets(pkg, tmp / "rating", int(cfg["study"]["seed"]))
    for f in s2info["files"]:
        df = pd.read_excel(f, sheet_name="一致性判定", dtype=str, keep_default_na=False)
        for i, r in df.iterrows():
            for c, vals in ratings.STAGE2_CHOICES.items():
                df.loc[i, c] = rs.choice(vals[:3])
        with pd.ExcelWriter(tmp / "rating/returned" / Path(f).name) as w:
            df.to_excel(w, sheet_name="一致性判定", index=False)
    s2c = ratings.collect_stage2(tmp / "rating/returned", tmp / "rating", cfg["rating"]["raters"])
    uinfo = ratings.unblind(pkg, cfg, tmp / "rating", tmp / "analysis", {(c, a2): len(t) for (c, a2), t in texts.items()})
    print("[SIM]", json.dumps({"collect": cinfo, "lock": {k: v for k, v in linfo.items() if "sha" not in k}, "stage2": {k: v for k, v in s2c.items() if "sha" not in k}, "unblind": uinfo}, ensure_ascii=False))
    res = analysis.run_all(tmp / "analysis", cfg, analysis.plan(cfg), Path("R/mixed_models.R"), pd.DataFrame(rows))
    spec = importlib.util.spec_from_file_location("s08", Path(__file__).parent / "08_analyze_study.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    print(m.md(res))
    print("[SIM] reliability:", json.dumps(res["reliability"], ensure_ascii=False)[:400])
    print(f"[SIM] OK (temp dir {tmp}); synthetic data - plumbing rehearsal only")


if __name__ == "__main__":
    main()
