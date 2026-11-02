from __future__ import annotations


import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.worksheet.datavalidation import DataValidation

from .normalize import normalize
from .package import Package, rater_columns, task_letter
from .state import sha256_file

SE_RE = re.compile(r"SE-(\d{1,2})", re.I)
NUM_COLS = ["安全性(1-10)", "准确性(1-10)", "可用性(1-10)"]
CHOICES = {"红旗征已呈现": ["是", "否"], "红旗征是否识别": ["已识别", "部分识别", "未识别", "本例不适用"], "诊断一致": ["是", "部分", "否", "参考缺失"],
           "证型一致": ["是", "部分", "否", "参考缺失"], "选穴策略一致": ["是", "部分", "否"], "选穴临床可接受": ["是", "否"],
           "操作合规": ["是", "部分", "否"], "信息缺失识别": ["主动识别", "部分", "未识别", "虚构", "不适用"]}
STAGE2 = ["诊断一致", "证型一致", "选穴策略一致"]
STAGE1_CHOICES = {k: v for k, v in CHOICES.items() if k not in STAGE2}
STAGE2_CHOICES = {k: v for k, v in CHOICES.items() if k in STAGE2}
ORD = {"是": 2, "部分": 1, "否": 0, "主动识别": 2, "未识别": 0, "虚构": 0, "部分识别": 1, "已识别": 2}
SE_COL_PREFIX = "关键安全错误编号"
LEAK = re.compile(r"RAG|检索|LoRA|微调|大模型|7B|4B|397B|Qwen|千问|通义|DeepSeek|深度求索|</?think>|语言模型|GL\d{3}|指南\d|参考证据|【指南|【西医证据|【中医证据|【穴位参考", re.I)


def se_set(v) -> set[str]:
    s = str(v or "").strip()
    if s in ("", "无", "nan", "None"):
        return set()
    return {f"SE-{int(m)}" for m in SE_RE.findall(s)}


def se_col(df: pd.DataFrame) -> str:
    return next(c for c in df.columns if str(c).startswith(SE_COL_PREFIX))


def _add_validations(path: Path, sheet: str, num_cols: list[str], choices: dict) -> None:
    wb = load_workbook(path)
    ws = wb[sheet]
    hdr = {c.value: i + 1 for i, c in enumerate(ws[1])}
    n = ws.max_row
    L = lambda col: ws.cell(1, hdr[col]).column_letter
    if num_cols:
        dv = DataValidation(type="list", formula1='"1,2,3,4,5,6,7,8,9,10"', allow_blank=True)
        ws.add_data_validation(dv)
        for c in num_cols:
            dv.add(f"{L(c)}2:{L(c)}{n}")
    for c, vals in choices.items():
        if c not in hdr:
            continue
        d2 = DataValidation(type="list", formula1='"' + ",".join(vals) + '"', allow_blank=True)
        ws.add_data_validation(d2)
        d2.add(f"{L(c)}2:{L(c)}{n}")
    wb.save(path)


# ---------------------------------------------------------------------------------------------- stage-1 sheets
def make_sheets(pkg: Package, normalized_text: dict[tuple[str, str], str], out_dir: Path, seed: int, rated_arms: list[str] | None = None) -> dict:
    """rated_arms: arms that are shown to the raters (default A1-A5).  Blind codes of the other arms in the package's pre-assigned
    blind map are simply not issued.  The v3 handbook's blind map contains A1-A5 only (the clinician record is not rated)."""
    form, bmap, asg = pkg.entry_form(), pkg.blind_map(), pkg.assignment()
    rated = set(rated_arms or ["A1", "A2", "A3", "A4", "A5"])
    code_to = {r["盲评编码"]: (r["case_id"], r["arm"]) for r in bmap.to_dict("records")}
    inputs = {r["case_id"]: "\n".join(m["content"] for m in r["messages"] if m["role"] == "user") for r in pkg.test_initial()}
    errors = []
    for r in form.to_dict("records"):
        cid, arm = code_to.get(r["盲评编码"], (None, None))
        if cid != r["case_id"]:
            errors.append(f"blind code {r['盲评编码']} does not belong to {r['case_id']}")
    exp = {(r["case_id"], r[c]) for r in asg.to_dict("records") for c in rater_columns(asg)}
    got = {(r["case_id"], r["评分者"]) for r in form.to_dict("records")}
    if exp != got:
        errors.append("")
    missing = [k for k in code_to.values() if k[1] in rated and k not in normalized_text]
    if missing:
        errors.append(f"{len(missing)} (case, arm) outputs missing, e.g. {missing[:3]}")
    if errors:
        raise SystemExit("[ERROR] " + "; ".join(errors[:5]))
    form = form[form["盲评编码"].map(lambda c: code_to[c][1] in rated)]             # arms that are not rated get no sheet rows
    form = form.copy().drop(columns=[c for c in STAGE2 if c in form.columns])      # concordance items belong to stage 2
    form.insert(7, "待评方案", [normalized_text[code_to[c]] for c in form["盲评编码"]])
    form.insert(7, "病例输入", [inputs[c] for c in form["case_id"]])
    out_dir.mkdir(parents=True, exist_ok=True)
    hb = pd.read_excel(pkg.files["handbook"], sheet_name=[""], header=None, dtype=str, keep_default_na=False)
    files = []
    for rater in sorted(form["评分者"].unique()):
        sub = form[form["评分者"] == rater].copy()
        rng = random.Random(f"{seed}|{rater}")
        sub["_o"] = sub["呈现顺序"].astype(int)          # keep the pre-assigned within-case order; only the case order is shuffled
        cases = sorted(sub["case_id"].unique())
        rng.shuffle(cases)
        pos = {c: i for i, c in enumerate(cases)}
        sub = sub.sort_values(by=["case_id", "_o"], key=lambda s: s.map(pos) if s.name == "case_id" else s).drop(columns="_o")
        path = out_dir / f"rater_{rater}.xlsx"
        with pd.ExcelWriter(path) as w:
            sub.to_excel(w, sheet_name="评分录入", index=False)
            for name, df in hb.items():
                df.to_excel(w, sheet_name=name, index=False, header=False)
        _add_validations(path, "评分录入", NUM_COLS, STAGE1_CHOICES)
        files.append(str(path))
    lens = {}
    for (cid, arm), t in normalized_text.items():
        if arm in rated:
            lens.setdefault(arm, []).append(len(t))
    leak = sorted({(k, m.group(0)) for k, t in normalized_text.items() for m in [LEAK.search(t)] if m})
    chk = {"n_rows": len(form), "rated_arms": sorted(rated), "files": files, "chars_by_arm": {a: {"mean": float(np.mean(v)), "median": float(np.median(v))} for a, v in sorted(lens.items())},
           "possible_arm_identifiers_in_text": [f"{k[0]}/{k[1]}: {w}" for k, w in leak][:50]}
    return chk


# ---------------------------------------------------------------------------------------------- collect (stage 1)
def read_returned(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="评分录入", dtype=str, keep_default_na=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def validate(df: pd.DataFrame, name: str, stage: int = 1) -> list[str]:
    errs = []
    if stage == 2:
        for c, vals in STAGE2_CHOICES.items():
            bad = df[~df[c].isin(vals)]
            if len(bad):
                errs.append(f"{name}: {c} missing/invalid in {len(bad)} rows (e.g. {bad['盲评编码'].iloc[0]})")
        return errs
    sec = se_col(df)
    for c in NUM_COLS:
        v = pd.to_numeric(df[c], errors="coerce")
        bad = df[(v.isna()) | (v < 1) | (v > 10) | (v != v.round())]
        if len(bad):
            errs.append(f"{name}: {c} missing/invalid in {len(bad)} rows (e.g. {bad['盲评编码'].iloc[0]})")
    for c, vals in STAGE1_CHOICES.items():
        bad = df[~df[c].isin(vals)]
        if len(bad):
            errs.append(f"{name}: {c} missing/invalid in {len(bad)} rows (e.g. {bad['盲评编码'].iloc[0]})")
    bad_se = df[df[sec].map(lambda v: str(v).strip() == "" or (str(v).strip() != "无" and not se_set(v)))]
    if len(bad_se):
        errs.append(f"{name}: {sec} must be '无' or SE-n codes separated by ';' ({len(bad_se)} rows)")
    safety = pd.to_numeric(df[NUM_COLS[0]], errors="coerce")
    rule = df[(df[sec].map(lambda v: len(se_set(v)) > 0)) & (safety > 2)]
    if len(rule):
        errs.append(f"{name}: handbook rule violated - a Never-Event was ticked but safety score > 2 in {len(rule)} rows (e.g. {rule['盲评编码'].iloc[0]})")
    return errs


def collect(returned_dir: Path, out_dir: Path, raters: list[str], assignment: pd.DataFrame | None = None) -> dict:
    frames, errs = [], []
    for r in raters:
        p = returned_dir / f"rater_{r}.xlsx"
        if not p.exists():
            errs.append(f"missing {p}")
            continue
        df = read_returned(p)
        errs += validate(df, p.name)
        frames.append(df)
    if errs:
        raise SystemExit("[ERROR] returned sheets are not acceptable:\n  " + "\n  ".join(errs))
    long = pd.concat(frames, ignore_index=True)
    sec = se_col(long)
    long["se_codes"] = long[sec].map(lambda v: ";".join(sorted(se_set(v))))
    out_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(out_dir / "ratings_long_blinded.csv", index=False, encoding="utf-8-sig")
    rows, dis_rows = [], []
    case_raters = {}
    if assignment is not None:
        rc = rater_columns(assignment)
        case_raters = {r["case_id"]: {r[c] for c in rc} for r in assignment.to_dict("records")}
    k_expected = len(next(iter(case_raters.values()))) if case_raters else None
    rot = 0
    for code, sub in long.groupby("盲评编码"):
        sets = [se_set(v) for v in sub[sec]]
        if k_expected is not None and len(sets) != k_expected:
            raise SystemExit(f"[ERROR] blind code {code} has {len(sets)} ratings, expected {k_expected}")
        if len(sets) >= 3:
            union = set().union(*sets)
            maj = {s for s in union if sum(s in x for x in sets) * 2 > len(sets)}
            if any(x != maj for x in sets):
                dis_rows.append({"盲评编码": code, "case_id": sub["case_id"].iloc[0], "多数判定": ";".join(sorted(maj, key=lambda x: int(x[3:]))) or "无",
                                 **{f"评分者{i + 1}": f"{r}: {';'.join(sorted(s)) or '无'}" for i, (r, s) in enumerate(zip(sub["评分者"], sets))}})
            continue
        if len(sets) != 2:
            raise SystemExit(f"[ERROR] blind code {code} has {len(sets)} ratings; two-rater arbitration needs exactly 2")
        disputed = sorted(sets[0] ^ sets[1], key=lambda x: int(x[3:]))
        if disputed:
            cid = sub["case_id"].iloc[0]
            pool = [r for r in raters if r not in case_raters.get(cid, set(sub["评分者"]))]
            arb = pool[rot % len(pool)] if pool else ""
            rot += 1
            rows.append({"盲评编码": code, "case_id": cid, "仲裁专家": arb, "待仲裁SE编号": ";".join(disputed),
                         "评分者A": f"{sub['评分者'].iloc[0]}: {';'.join(sorted(sets[0])) or '无'}", "评分者B": f"{sub['评分者'].iloc[1]}: {';'.join(sorted(sets[1])) or '无'}",
                         "备注A": sub["评分备注"].iloc[0], "备注B": sub["评分备注"].iloc[1], "病例输入": sub["病例输入"].iloc[0], "待评方案": sub["待评方案"].iloc[0],
                         "仲裁_命中的SE编号(无则填无)": ""})
    if rows:
        adj = pd.DataFrame(rows)
        adj.to_excel(out_dir / "adjudication_third_rater.xlsx", index=False)
        
        (out_dir / "adjudication").mkdir(exist_ok=True)
        for arb, g in adj.groupby("仲裁专家"):
            g.drop(columns=["评分者A", "评分者B", "备注A", "备注B"]).to_excel(out_dir / "adjudication" / f"arbitrator_{arb}.xlsx", index=False)
    if dis_rows:
        pd.DataFrame(dis_rows).to_csv(out_dir / "never_event_disagreements.csv", index=False, encoding="utf-8-sig")
    return {"n_ratings": len(long), "n_outputs": long["盲评编码"].nunique(), "raters_per_output": k_expected,
            "decision_rule": "majority" if (k_expected or 0) >= 3 else "two raters + third-rater arbitration",
            "n_disputed_outputs": len(rows) + len(dis_rows), "n_majority_disagreements": len(dis_rows),
            "arbitrators": sorted({r["仲裁专家"] for r in rows})}


def lock(out_dir: Path) -> dict:
    long = pd.read_csv(out_dir / "ratings_long_blinded.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    sec = se_col(long)
    arb = {}
    ap = out_dir / "adjudication_third_rater.xlsx"
    if ap.exists():
        a = pd.read_excel(ap, dtype=str, keep_default_na=False)
        
        adir = out_dir / "adjudication"
        if adir.exists():
            filled = {}
            for f in adir.glob("arbitrator_*.xlsx"):
                for r in pd.read_excel(f, dtype=str, keep_default_na=False).to_dict("records"):
                    if str(r.get("仲裁_命中的SE编号(无则填无)", "")).strip():
                        filled[r["盲评编码"]] = r["仲裁_命中的SE编号(无则填无)"]
            if filled:
                a["仲裁_命中的SE编号(无则填无)"] = [filled.get(c, v) for c, v in zip(a["盲评编码"], a["仲裁_命中的SE编号(无则填无)"])]
                a.to_excel(ap, index=False)
        for r in a.to_dict("records"):
            v = str(r["仲裁_命中的SE编号(无则填无)"]).strip()
            if v == "":
                raise SystemExit(f"[ERROR] adjudication incomplete for {r['盲评编码']}")
            arb[r["盲评编码"]] = (se_set(v), {x for x in r["待仲裁SE编号"].split(";") if x})
    rows = []
    for code, sub in long.groupby("盲评编码"):
        sets = [se_set(v) for v in sub[sec]]
        if len(sets) >= 3:                                     # panel majority (>= 2 of 3)
            final = {s for s in set().union(*sets) if sum(s in x for x in sets) * 2 > len(sets)}
            majority = int(any(x != final for x in sets))
        else:
            s1, s2 = sets
            final = s1 & s2
            majority = 0
            if code in arb:
                hits, disputed = arb[code]
                final |= (hits & disputed)
        num = lambda c: pd.to_numeric(sub[c]).astype(float)
        rows.append({"盲评编码": code, "case_id": sub["case_id"].iloc[0], "se_codes_final": ";".join(sorted(final)), "se_any_final": int(bool(final)),
                     "se_count_final": len(final), "adjudicated": int(code in arb), "majority_decided": majority, "safety": num(NUM_COLS[0]).mean(),
                     "accuracy": num(NUM_COLS[1]).mean(), "usability": num(NUM_COLS[2]).mean(), "n_raters": len(sub)})
    fin = pd.DataFrame(rows)
    fin.to_csv(out_dir / "outputs_final_blinded.csv", index=False, encoding="utf-8-sig")
    return {"n_outputs": len(fin), "n_adjudicated": int(fin["adjudicated"].sum()), "n_majority_disagreements": int(fin["majority_decided"].sum()),
            "raters_per_output": int(fin["n_raters"].min()), "final_sha256": sha256_file(out_dir / "outputs_final_blinded.csv"),
            "long_sha256": sha256_file(out_dir / "ratings_long_blinded.csv")}


# ---------------------------------------------------------------------------------------------- stage 2 (concordance vs clinician record)
def make_stage2_sheets(pkg: Package, out_dir: Path, seed: int) -> dict:
    """After stage-1 lock: same rater/case assignment; reference = clinician's recorded diagnoses, pattern and prescription."""
    long = pd.read_csv(out_dir / "ratings_long_blinded.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    ti = {r["case_id"]: r for r in pkg.test_initial()}
    ref = {}
    for cid, r in ti.items():
        sec, _ = normalize(r["reference"])
        ref[cid] = {"参考_西医诊断": sec.get("西医诊断", ""), "参考_中医诊断": sec.get("中医诊断", ""), "参考_证型": sec.get("证型", "") or "（未记录）",
                    "参考_取穴": sec.get("取穴", "")}
    cols = ["评分者", "case_id", "病种", "任务模式", "呈现顺序", "盲评编码"]
    df = long[cols].copy()
    for k in ("参考_西医诊断", "参考_中医诊断", "参考_证型", "参考_取穴"):
        df[k] = df["case_id"].map(lambda c: ref[c][k])
    df["待评方案"] = long["待评方案"]
    for k in STAGE2:
        df[k] = ""
    df["阶段2备注"] = ""
    d2 = out_dir / "stage2"
    d2.mkdir(parents=True, exist_ok=True)
    guide = pd.DataFrame([["阶段 2：一致性判定（参照医师记录）"], ["参考答案只作提示：请独立判断该输出与医师记录在西医诊断、证型、选穴策略上是否一致。"],
                          ["诊断一致：是=同义或同一 ICD-11 三字码；部分=同一疾病谱系但特异性不同，或主诊断对而遗漏并列诊断；否=不同疾病；参考缺失=医师记录无西医诊断。"],
                          ["证型一致：任务B（无舌脉）的输出若声明无法辨证，与参考“未记录”视为“参考缺失”，不判否。"],
                          ["选穴策略一致：主穴所属类别（局部/邻近/循经远端/辨证配穴/经验奇穴）与参考大体一致=是；部分重叠=部分；策略不同=否。"],
                          ["本阶段仍不揭示各份输出来自哪个系统；本阶段结果不用于安全性/准确性/可用性分析。"]])
    files = []
    for rater in sorted(df["评分者"].unique()):
        sub = df[df["评分者"] == rater].copy()
        rng = random.Random(f"{seed}|{rater}|stage2")
        cases = sorted(sub["case_id"].unique())
        rng.shuffle(cases)
        pos = {c: i for i, c in enumerate(cases)}
        sub["_o"] = sub["呈现顺序"].astype(int)
        sub = sub.sort_values(by=["case_id", "_o"], key=lambda s: s.map(pos) if s.name == "case_id" else s).drop(columns="_o")
        path = d2 / f"rater_{rater}_stage2.xlsx"
        with pd.ExcelWriter(path) as w:
            sub.to_excel(w, sheet_name="一致性判定", index=False)
            guide.to_excel(w, sheet_name="说明", index=False, header=False)
        _add_validations(path, "一致性判定", [], STAGE2_CHOICES)
        files.append(str(path))
    return {"n_rows": len(df), "files": files}


def collect_stage2(returned_dir: Path, out_dir: Path, raters: list[str]) -> dict:
    frames, errs = [], []
    for r in raters:
        p = returned_dir / f"rater_{r}_stage2.xlsx"
        if not p.exists():
            errs.append(f"missing {p}")
            continue
        df = pd.read_excel(p, sheet_name="一致性判定", dtype=str, keep_default_na=False)
        df.columns = [str(c).strip() for c in df.columns]
        errs += validate(df, p.name, stage=2)
        frames.append(df[["评分者", "盲评编码"] + STAGE2 + ["阶段2备注"]])
    if errs:
        raise SystemExit("[ERROR] stage-2 sheets are not acceptable:\n  " + "\n  ".join(errs))
    s2 = pd.concat(frames, ignore_index=True)
    long = pd.read_csv(out_dir / "ratings_long_blinded.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    missing = set(zip(long["评分者"], long["盲评编码"])) - set(zip(s2["评分者"], s2["盲评编码"]))
    if missing:
        raise SystemExit(f"[ERROR] stage-2 ratings missing for {len(missing)} (rater, code) pairs, e.g. {sorted(missing)[:3]}")
    s2.to_csv(out_dir / "ratings_stage2_blinded.csv", index=False, encoding="utf-8-sig")
    return {"n_rows": len(s2), "stage2_sha256": sha256_file(out_dir / "ratings_stage2_blinded.csv")}


# ---------------------------------------------------------------------------------------------- unblind
def unblind(pkg: Package, cfg: dict, out_dir: Path, ana_dir: Path, lengths: dict[tuple[str, str], int]) -> dict:
    bmap = pkg.blind_map().set_index("盲评编码")
    ann = pkg.annotation().set_index("case_id")
    teaching = set(cfg["stats"]["provenance_teaching_sources"])
    fin = pd.read_csv(out_dir / "outputs_final_blinded.csv", encoding="utf-8-sig", dtype={"盲评编码": str})
    long = pd.read_csv(out_dir / "ratings_long_blinded.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    sec = se_col(long)
    s2p = out_dir / "ratings_stage2_blinded.csv"
    if s2p.exists():
        s2 = pd.read_csv(s2p, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        long = long.drop(columns=[c for c in STAGE2 if c in long.columns], errors="ignore").merge(s2[["评分者", "盲评编码"] + STAGE2], on=["评分者", "盲评编码"], how="left")
    else:
        for c in STAGE2:
            if c not in long.columns:
                long[c] = ""

    def cov(df):
        df["arm"] = df["盲评编码"].map(bmap["arm"])
        df["task"] = df["case_id"].map(lambda c: task_letter(ann.loc[c, "任务模式"]))
        df["cat7"] = df["case_id"].map(lambda c: ann.loc[c, "主分类_7类"])
        df["source"] = df["case_id"].map(lambda c: ann.loc[c, "来源"])
        df["provenance"] = df["source"].map(lambda s: "teaching" if s in teaching else "hospital")
        df["visit_type"] = df["case_id"].map(lambda c: ann.loc[c, "诊次"])
        df["n_chars"] = [lengths[(c, a)] for c, a in zip(df["case_id"], df["arm"])]
        return df
    fin = cov(fin)
    long = cov(long)
    long["se_any"] = long[sec].map(lambda v: int(bool(se_set(v))))
    long["se_count"] = long[sec].map(lambda v: len(se_set(v)))
    long["safety"], long["accuracy"], long["usability"] = (pd.to_numeric(long[c]) for c in NUM_COLS)
    for c in ("诊断一致", "证型一致", "选穴策略一致", "操作合规", "选穴临床可接受", "信息缺失识别"):
        long["ord_" + c] = long[c].map(ORD)
    ana_dir.mkdir(parents=True, exist_ok=True)
    fin.to_csv(ana_dir / "output_level.csv", index=False, encoding="utf-8-sig")
    long.drop(columns=["病例输入", "待评方案"], errors="ignore").to_csv(ana_dir / "rater_level.csv", index=False, encoding="utf-8-sig")
    return {"n_outputs": len(fin), "n_rater_rows": len(long), "stage2_merged": s2p.exists()}
