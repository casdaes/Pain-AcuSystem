#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tcm_study import schema as S
from tcm_study.arms import is_degenerate
from tcm_study.normalize import parse_sections
from tcm_study.acupoints import AcupointLexicon
from tcm_study.needling import standardize_point_laterality, _selected_point_items, _point_is_midline

REQUIRED = ["诊疗思路", "西医诊断", "中医诊断", "证型", "病机", "治法", "取穴", "操作", "注意事项"]
AUX_KW = re.compile(r"CT|MRI|X线|X片|超声|B超|彩超|血糖|血压|心电|肌电|化验|血常规|影像|片示|示[：:]?")
CAUTION = re.compile(r"急危重症警示|急诊|转诊|立即就医|尽快就医|排除[^。；\n]{0,12}(?:急腹症|心梗|心肌梗死|马尾|夹层|骨折|肿瘤|感染|卒中|脑梗|阑尾|穿孔|休克)|危急值|绿色通道")
PATTERN_WORDS = r"(?:气滞|血瘀|瘀血|寒湿|湿热|风寒|风热|阴虚|阳虚|气虚|血虚|肝郁|肾虚|脾虚|痰湿|痰浊|肝阳|气血亏虚|肝肾亏虚|肝肾不足)"
DX_NOISE = re.compile(PATTERN_WORDS + r"|根据|考虑|结合|可能|建议|患者|症状|初步|诊断为|[。！？]")


def check_one(case_input: str, raw_output: str, lex=None) -> dict:
    inp = S.parse_input(case_input)
    sec = parse_sections(raw_output)
    wm, tcm, pat = sec.get("西医诊断", "").strip(), sec.get("中医诊断", "").strip(), sec.get("证型", "").strip()
    full = raw_output or ""
    r = {}
    r["schema_ok"] = int(all(sec.get(k, "").strip() for k in REQUIRED))
    r["wm_dx_present"] = int(bool(wm))
    wm_core = re.sub(r"（待查）|\(待查\)", "", wm)
    r["wm_dx_clean"] = int(bool(wm) and len(wm) <= 40 and not DX_NOISE.search(wm_core) and "中医" not in wm and "：" not in wm)
    r["tcm_dx_clean"] = int(bool(tcm) and len(tcm) <= 16 and not re.search(r"[（(]|型$", tcm) and not re.search(PATTERN_WORDS, tcm) and not DX_NOISE.search(tcm))
    r["pattern_slot_ok"] = int(r["wm_dx_clean"] and r["tcm_dx_clean"] and bool(pat))
    has_aux = inp["_has_aux"]
    r["aux_used"] = int(bool(AUX_KW.search(sec.get("诊疗思路", "") + " " + wm + " " + sec.get(S_EXTRA, "")))) if has_aux else None
    cons = S.aux_consistent(inp.get("辅助检查", "") if has_aux else "", wm_core)
    r["aux_dx_consistent"] = None if cons is None else int(cons)
    exams, need_pending = S.pending_exams(wm_core, has_aux, "moderate")
    need_pending = need_pending or S.is_symptom_label(wm_core)
    has_pending = "待查" in wm
    r["pending_ok"] = int(has_pending == need_pending) if wm else 0
    r["pending_has_exam"] = int(bool(re.search(r"完善|检查|X线|MRI|CT|超声|化验|电生理", sec.get("注意事项", "")))) if has_pending else None
    flags = S.red_flag_warnings(case_input)
    emerg = S.is_emergency_level(flags)
    r["redflag_recall"] = int(bool(CAUTION.search(full))) if emerg else None
    r["redflag_precision"] = (int(emerg) if S.EMERGENCY_TAG in full else None)
    note = S.UNCERTAINTY_NOTE[:12] in pat or bool(re.search(r"未记录|未提供|暂无法|无法明确", pat))
    r["tp_note_ok"] = int(note if not inp["_has_tp"] else not note)
    r["tp_fabricated"] = int((bool(S.TONGUE_RE.search(full)) and not S.TONGUE_RE.search(case_input)) or (bool(S.PULSE_RE.search(full)) and not S.PULSE_RE.search(case_input)))
    r["notes_present"] = int(bool(sec.get("注意事项", "").strip()))
    op = sec.get("操作", "")
    r["op_format"] = int(bool(re.search(r"^1[\.．、]\s*手法与疗程[：:]", op)) and bool(re.search(r"(?m)^2[\.．、]\s*逐穴操作[：:]", op)))
   
    reason = sec.get("诊疗思路", "")

    def _split_diff_top_level(text):
        out, buf = [], []
        depth = 0
        for ch in text or "":
            if ch in "（(":
                depth += 1
            elif ch in "）)" and depth > 0:
                depth -= 1

            if ch in "；;" and depth == 0:
                item = "".join(buf).strip()
                if item:
                    out.append(item)
                buf = []
            else:
                buf.append(ch)

        item = "".join(buf).strip()
        if item:
            out.append(item)
        return out

    def _extract_diff_items(text):
        marker_re = re.compile(r"(?:需鉴别|需优先排除|需排除)[：:]")
        markers = list(marker_re.finditer(text or ""))
        items = []

        for m in markers:
            start = m.end()
            depth = 0
            end = len(text)
            i = start

            while i < len(text):
                ch = text[i]

                if ch in "（(":
                    depth += 1
                elif ch in "）)" and depth > 0:
                    depth -= 1

               
                if depth == 0 and ch == "。":
                    end = i
                    break

                if depth == 0 and i > start and marker_re.match(text, i):
                    end = i
                    break

                i += 1

            block = text[start:end].strip()
            items.extend(_split_diff_top_level(block))

        cleaned = []
        for item in items:
            item = marker_re.sub("", item, count=1).strip()
            if item:
                cleaned.append(item)

        return cleaned

    diff_items = _extract_diff_items(reason)

    r["diff_keypoint_format"] = (
        int(all(
            re.search(r"[（(]\s*关键区分点[：:]", item)
            for item in diff_items
        ))
        if diff_items else None
    )
    pts_text = sec.get("取穴", "").strip()
    if lex is not None and pts_text:
        std_pts, _ = standardize_point_laterality(lex, pts_text, case_input)
        norm = lambda x: re.sub(r"\s+", "", x or "")
        r["point_laterality_ok"] = int(norm(std_pts) == norm(pts_text))
        mid_err = 0
        for it in _selected_point_items(lex, pts_text):
            if it.get("status") in ("standard", "alias") and _point_is_midline(lex, it.get("name", "")) and it.get("side"):
                mid_err += 1
        r["midline_side_error"] = mid_err
    else:
        r["point_laterality_ok"] = None
        r["midline_side_error"] = None
    r["legacy_depth_rule"] = int(bool(re.search(r"规则核查[^\n]*(?:所写深度|超过穴位库上限|深度[^\n]*上限)", full)))
    r["duplicate_spec"] = int("逐穴针刺规范：" in op)
    r["degenerate"] = int(is_degenerate(full) is not None)
    r["n_chars"] = len(full)
    r["_wm"], r["_tcm"], r["_pat"], r["_flags"] = wm, tcm, pat[:40], ",".join(h["rule"] for h in flags)
    return r


S_EXTRA = "其他内容"
COLS = ["schema_ok", "wm_dx_present", "wm_dx_clean", "tcm_dx_clean", "pattern_slot_ok", "aux_used", "aux_dx_consistent", "pending_ok", "pending_has_exam",
        "redflag_recall", "redflag_precision", "tp_note_ok", "tp_fabricated", "notes_present", "op_format", "diff_keypoint_format", "point_laterality_ok", "midline_side_error", "legacy_depth_rule", "duplicate_spec", "degenerate", "n_chars"]


def load_xlsx(path: str) -> list[dict]:
    import pandas as pd
    df = pd.read_excel(path, sheet_name=0, dtype=str, keep_default_na=False)
    arms = [c for c in df.columns if re.match(r"^A\d", str(c))]
    rows = []
    for _, r in df.iterrows():
        for a in arms:
            rows.append({"case_id": r["case_id"], "arm": re.match(r"^(A\d)", a).group(1), "input": r["病例输入(全文)"], "output": r[a]})
    return rows


def load_work(cfg_path: str, input_dir: str = "") -> list[dict]:
    from tcm_study.config import load_cfg
    from tcm_study.package import Package
    cfg = load_cfg(cfg_path)
    pkg = Package(cfg["paths"]["package_dir"])
    inputs = {r["case_id"]: "\n".join(m["content"] for m in r["messages"] if m["role"] == "user") for r in pkg.test_initial()}
    rows = []

    src_dir = (
        Path(input_dir)
        if input_dir
        else Path(cfg["paths"]["work_dir"]) / "outputs" / "initial"
    )

    for p in sorted(src_dir.glob("raw_A*.jsonl")):
        for l in p.open(encoding="utf-8"):
            d = json.loads(l)
            rows.append({"case_id": d["case_id"], "arm": d["arm"], "input": inputs.get(d["case_id"], ""), "output": d["raw_output"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/study.yaml")
    ap.add_argument("--xlsx", default="")
    ap.add_argument("--dump", default="")
    ap.add_argument(
        "--input-dir",
        default="",
        help="Directory containing raw_A*.jsonl; default: work/outputs/initial",
    )
    a = ap.parse_args()
    rows = load_xlsx(a.xlsx) if a.xlsx else load_work(a.config, a.input_dir)
    from tcm_study.config import load_cfg
    from tcm_study.package import Package
    cfg = load_cfg(a.config)
    pkg = Package(cfg["paths"]["package_dir"])
    lex = AcupointLexicon(pkg.files["acupoint_lib"])
    res = []
    for r in rows:
        res.append({"case_id": r["case_id"], "arm": r["arm"], **check_one(r["input"], r["output"], lex)})
    agg = defaultdict(lambda: defaultdict(list))
    for r in res:
        for k in COLS:
            if r[k] is not None:
                agg[r["arm"]][k].append(r[k])
    arms = sorted(agg)
    print(f"{'metric':22s}" + "".join(f"{x:>9s}" for x in arms) + "   (均值；None 表示该例不适用)")
    for k in COLS:
        cells = []
        for x in arms:
            v = agg[x][k]
            cells.append(f"{(sum(v) / len(v)):9.3f}" if v else f"{'—':>9s}")
        print(f"{k:22s}" + "".join(cells) + (f"   n={[len(agg[x][k]) for x in arms]}" if k in ("aux_used", "aux_dx_consistent", "pending_has_exam", "redflag_recall", "redflag_precision") else ""))
    if a.dump:
        import pandas as pd
        pd.DataFrame(res).to_excel(a.dump, index=False)
        print(f"[OK] per-case dump -> {a.dump}")


if __name__ == "__main__":
    main()
