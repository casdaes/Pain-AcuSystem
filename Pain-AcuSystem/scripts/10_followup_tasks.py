#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tcm_study.config import load_cfg
from tcm_study.normalize import normalize
from tcm_study.package import Package, arm_id
from tcm_study.ratings import se_col, se_set
from tcm_study.state import StudyState

DIR_VALUES = ["合理", "应调未调", "不应调却调", "调整方向错误"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/study.yaml")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--make-sheets", action="store_true")
    g.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    pkg = Package(cfg["paths"]["package_dir"])
    work = Path(cfg["paths"]["work_dir"])
    state = StudyState(work)
    state.require("locked")
    xl = pkg.files["followup_xlsx"]
    form = pd.read_excel(xl, sheet_name="", dtype=str, keep_default_na=False)
    bmap = pd.read_excel(xl, sheet_name="", header=2, dtype=str, keep_default_na=False)
    bmap = bmap[bmap["盲评编码"] != ""].copy()
    bmap["arm"] = bmap["实际臂"].map(arm_id)
    code_to = {r["盲评编码"]: (r["case_id"], r["arm"]) for r in bmap.to_dict("records")}
    fu_arms = list(cfg["rating"].get("followup_arms") or ["A4", "A5"])
    form = form[form["盲评编码"].map(lambda c: code_to[c][1] in fu_arms)].copy()
    fdir = work / "followup"
    if a.make_sheets:
        texts = {}
        for arm in fu_arms:
            for l in (work / "outputs/followup" / f"raw_{arm}.jsonl").open(encoding="utf-8"):
                d = json.loads(l)
                texts[(d["case_id"], arm)] = normalize(d["raw_output"])[1] if arm != "A6" else d["raw_output"]
        inp = {r["case_id"]: "\n".join(m["content"] for m in r["messages"] if m["role"] == "user") for r in pkg.test_followup()}
        form.insert(7, "待评方案", [texts[code_to[c]] for c in form["盲评编码"]])
        form.insert(7, "复诊题输入", [inp[c] for c in form["case_id"]])
        (fdir / "returned").mkdir(parents=True, exist_ok=True)
        for r in sorted(form["评分者"].unique()):
            form[form["评分者"] == r].to_excel(fdir / f"rater_{r}.xlsx", sheet_name="复诊评分", index=False)
        print(f"[OK] {form['评分者'].nunique()} follow-up sheets in {fdir}")
        return
    frames = [pd.read_excel(p, sheet_name="复诊评分", dtype=str, keep_default_na=False) for p in sorted((fdir / "returned").glob("rater_*.xlsx"))]
    d = pd.concat(frames, ignore_index=True)
    d["arm"] = d["盲评编码"].map(lambda c: code_to[c][1])
    sec = se_col(d)
    d["direction"] = d["① 调整方向"]
    d["score"] = pd.to_numeric(d["③ 合理性(1-10)"])
    d["agree"] = d["② 与真实医生一致"].map({"是": 2, "部分": 1, "否": 0})
    d["se_any"] = d[sec].map(lambda v: int(bool(se_set(v))))
    out = {}
    for arm, s in d.groupby("arm"):
        out[arm] = {"n_ratings": len(s), "direction": s["direction"].value_counts().to_dict(), "share_reasonable": round(float((s["direction"] == "合理").mean()), 3),
                    "mean_score": round(float(s["score"].mean()), 2), "mean_agreement_0_2": round(float(s["agree"].mean()), 2), "se_rate": round(float(s["se_any"].mean()), 3)}
    (fdir / "FOLLOWUP_SUMMARY.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("[NOTE] exploratory and descriptive only (n=21 tasks)")


if __name__ == "__main__":
    main()
