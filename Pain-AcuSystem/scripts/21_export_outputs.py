#!/usr/bin/env python3
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tcm_study.config import load_cfg          # noqa: E402
from tcm_study.package import Package          # noqa: E402

XL_MAX = 32000                                  # Excel 单元格上限 32767 字符


def _clean(v):
    if not isinstance(v, str):
        return v
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
    v = ILLEGAL_CHARACTERS_RE.sub("", v)
    return v if len(v) <= XL_MAX else v[:XL_MAX] + f"\n……（超过 Excel 单元格上限，已截断；全文见 raw_*.jsonl，共 {len(v)} 字）"


def _items(cfg, pkg, which: str) -> list[dict]:
    spec = importlib.util.spec_from_file_location("gen05", ROOT / "scripts" / "05_generate_arms.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.load_items(cfg, pkg, which)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "study.yaml"))
    ap.add_argument("--set", choices=["initial", "pilot", "followup"], default="initial")
    ap.add_argument("--arms", nargs="+", default=["A1", "A2", "A3", "A4"])
    ap.add_argument("--reference", action="store_true", help="加一列医师记录（A6）")
    ap.add_argument("--reasoning", action="store_true", help="加'思考'表")
    ap.add_argument("--work", default="", help="覆盖 paths.work_dir（一般不用）")
    ap.add_argument("--out", default="")
    ap.add_argument("--subdir", default="", help="读 work/outputs/<subdir>（如 initial_safety）；默认与 --set 相同")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    pkg = Package(cfg["paths"]["package_dir"])
    work = Path(a.work or cfg["paths"]["work_dir"])
    src = work / "outputs" / (a.subdir or a.set)
    items = _items(cfg, pkg, a.set)
    meta = {}
    if a.set == "initial":
        meta = {r["case_id"]: r for r in pkg.test_initial()}

    recs = {}
    for arm in a.arms:
        p = src / f"raw_{arm}.jsonl"
        recs[arm] = {d["case_id"]: d for d in map(json.loads, p.open(encoding="utf-8")) if d} if p.exists() else {}
        if not p.exists():
            print(f"[WARN] {p} 不存在：{arm} 列留空")

    comp, detail, think = [], [], []
    for it in items:
        cid = it["case_id"]
        row = {"case_id": cid, "任务": meta.get(cid, {}).get("task", ""), "分类": meta.get(cid, {}).get("cat7", ""),
               "病例输入(全文)": "\n".join(m["content"] for m in it["messages"] if m["role"] == "user")}
        for arm in a.arms:
            d = recs[arm].get(cid)
            row[f"{arm} 输出"] = d["raw_output"] if d else ""
            if not d:
                continue
            ts = bool(d.get("two_stage"))
            detail.append({
                "case_id": cid, "臂": arm, "模型": d.get("model_id"), "两步RAG": ts, "成品输出": d["raw_output"],
                "第一步原始输出": d.get("stage1_output", "") if ts else "", "第二步原始输出": d.get("stage2_output", "") if ts else "",
                "思考token(第一步/单步)": d.get("reasoning_tokens"), "思考token(第二步)": d.get("stage2_reasoning_tokens") if ts else None,
                "答案token(合计)": d.get("new_tokens"),
                "思考超预算": bool(d.get("think_budget_hit")) or bool(d.get("stage2_think_budget_hit")),
                "思考未闭合": bool(d.get("think_unclosed")),
                "重试": bool(d.get("retried")) or bool(d.get("stage2_retried")),
                "首轮退化原因": d.get("degenerate_first_pass") or d.get("stage2_degenerate_first_pass") or "",
                "仍退化": bool(d.get("still_degenerate")), "耗时s": d.get("latency_s"), "生成时间": d.get("generated_at")})
            if a.reasoning:
                think.append({"case_id": cid, "臂": arm,
                              "思考(第一步/单步)": d.get("stage1_reasoning") if ts else d.get("reasoning"),
                              "思考(第二步)": d.get("stage2_reasoning") if ts else ""})
        if a.reference:
            row["医师记录(A6，参照)"] = it.get("reference", "")
        comp.append(row)

    summ = []
    for arm in a.arms:
        ds = [d for d in detail if d["臂"] == arm]
        n = len(ds)
        avg = lambda k: round(sum((d[k] or 0) for d in ds) / n) if n else None
        summ.append({"臂": arm, "完成": f"{n}/{len(items)}", "模型": ";".join(sorted({str(d["模型"]) for d in ds})),
                     "平均思考token(第一步/单步)": avg("思考token(第一步/单步)"), "平均答案token": avg("答案token(合计)"),
                     "思考超预算": sum(d["思考超预算"] for d in ds), "思考未闭合": sum(d["思考未闭合"] for d in ds),
                     "重试": sum(d["重试"] for d in ds), "仍退化": sum(d["仍退化"] for d in ds), "平均耗时s": avg("耗时s")})

    out = Path(a.out) if a.out else work / "exports" / f"{'-'.join([a.arms[0], a.arms[-1]]) if len(a.arms) > 1 else a.arms[0]}_{a.set}.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheets = {"对比": pd.DataFrame(comp), "明细": pd.DataFrame(detail)}
    if a.reasoning:
        sheets["思考"] = pd.DataFrame(think)
    sheets["汇总"] = pd.DataFrame(summ)
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df = df.map(_clean) if hasattr(df, "map") else df.applymap(_clean)
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            from openpyxl.styles import Alignment
            for col in ws.columns:
                head = str(col[0].value or "")
                wide = any(k in head for k in ("输出", "输入", "思考(", "医师记录"))
                ws.column_dimensions[col[0].column_letter].width = 60 if wide else max(10, min(28, len(head) * 2 + 2))
                if wide:
                    for c in col[1:]:
                        c.alignment = Alignment(wrap_text=True, vertical="top")
            ws.freeze_panes = "B2"
    print(pd.DataFrame(summ).to_string(index=False))
    print(f"[OK] {out}   （带臂标签：只供研究者核对，不要发给评分专家）")


if __name__ == "__main__":
    main()
