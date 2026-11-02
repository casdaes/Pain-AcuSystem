#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_DIFF_RE = re.compile(r"需鉴别[：:]|需(?:优先)?排除[：:]")


def _name_only(x: str) -> str:
    return re.split(r"[（(]", x or "", 1)[0].strip(" 。；;,，")


def _norm_name(x: str) -> str:
    x = re.sub(r"\s+", "", x or "")
    x = re.sub(r"^(?:需鉴别|需(?:优先)?排除|考虑|警惕|排除)[：:]?", "", x)
    x = re.sub(r"[（(].*$", "", x)
    return x.strip(" ：:。；;,，")


def _pred_diff_names(notes: str) -> list[str]:
    """只从明确的‘需鉴别/需优先排除’语句抽具体病名，避免把正文其他病名误算为 FP。"""
    out: list[str] = []
    for line in (notes or "").splitlines():
        if not _DIFF_RE.search(line):
            continue

        chunks = re.split(r"(?=需鉴别[：:]|需(?:优先)?排除[：:])", line)
        for ch in chunks:
            m = re.match(r"(?:需鉴别|需(?:优先)?排除)[：:](.*)", ch.strip())
            if not m:
                continue
            body = m.group(1)
            for item in re.split(r"[；;]", body):
                n = _norm_name(item)
                n = re.split(r"[，,：:]", n, 1)[0].strip()
                if 2 <= len(n) <= 24 and not re.search(r"其他|相关病|病变|疾病$|骨折等|肿瘤等|感染等|检查|影像|实验室|MRI|CT|X线|超声", n):
                    if n not in out:
                        out.append(n)
    return out


def _same_dx(a: str, b: str) -> bool:
    a, b = _norm_name(a), _norm_name(b)
    if not a or not b:
        return False
    if a == b:
        return True
    return min(len(a), len(b)) >= 4 and (a in b or b in a)


def _set_score(expected: list[str], predicted: list[str]) -> tuple[int, int, int, float]:
    used = set()
    tp = 0
    for p in predicted:
        hit = next((i for i, e in enumerate(expected) if i not in used and _same_dx(p, e)), None)
        if hit is not None:
            used.add(hit)
            tp += 1
    fp = max(0, len(predicted) - tp)
    fn = max(0, len(expected) - tp)
    den = 2 * tp + fp + fn
    f1 = (2 * tp / den) if den else 1.0
    return tp, fp, fn, f1


def _prf(c: dict) -> tuple[float, float, float]:
    tp, fp, fn = c.get("diff_tp", 0), c.get("diff_fp", 0), c.get("diff_fn", 0)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def check_arm(rows, train, lex, dxkb=None):
    from tcm_study import schema as S
    from tcm_study.acupoints import output_autometrics
    from tcm_study.needling import pregnancy_rules
    from tcm_study.normalize import normalize

    keys = ("九栏", "注意事项空", "深度", "急危", "急危写了", "模型原始急危写了", "规则补急危", "过度报警", "模型原始过度报警",
            "不适用", "不适用却写妊娠", "育龄A类", "写排除妊娠", "模板句", "写鉴别", "鉴别可评", "鉴别命中1", "鉴别可评2", "鉴别命中2",
            "泛泛鉴别", "套话", "写定位", "思路字数", "注意字数", "diff_tp", "diff_fp", "diff_fn", "diff_case_f1_sum", "diff_case_n",
            "card_tp", "card_fn", "card_cases", "model_card_tp", "model_card_fp")
    c = {k: 0 for k in keys}
    bad = {"注意事项空": [], "漏报": [], "过度报警": [], "不适用却写妊娠": [], "育龄A类漏写": [], "鉴别未命中": [], "鉴别不足2病": [], "泛泛鉴别": []}

    for r in rows:
        tr = train.get(r["case_id"], {})
        inp, ref_wm, out, cid = tr.get("input", ""), tr.get("wm", ""), r["raw_output"], r["case_id"]
        sec = S.parse_bracket_sections(out, top_level_only=True)
        notes = (sec.get("注意事项") or "").strip()
        c["九栏"] += all((sec.get(k) or "").strip() not in ("", "无", "（未给出）") for k in S.SECTIONS_V3)
        if not notes:
            c["注意事项空"] += 1; bad["注意事项空"].append(cid)
        c["深度"] += (output_autometrics(lex, normalize(out)[0], out, inp, "")["depth_specified_rate"] or 0) >= 0.8

        w = S.has_real_warning(out)
        model_out = r.get("raw_output_model") or out
        mw = S.has_real_warning(model_out)
        if S.is_emergency_level(S.red_flag_warnings(inp)):
            c["急危"] += 1; c["急危写了"] += w; c["模型原始急危写了"] += mw
            if w and not mw: c["规则补急危"] += 1
            if not w: bad["漏报"].append(cid)
        else:
            if w: c["过度报警"] += 1; bad["过度报警"].append(cid)
            if mw: c["模型原始过度报警"] += 1

        if not S.pregnancy_text_applies(inp):
            c["不适用"] += 1
            if re.search(r"育龄|排除妊娠|HCG|(?<!异位)妊娠|孕妇", notes):
                c["不适用却写妊娠"] += 1; bad["不适用却写妊娠"].append(cid)
        pr = pregnancy_rules(lex, sec.get("取穴", ""), inp)
        if pr["childbearing"] and pr["a_points"] and not pr["pregnant"]:
            c["育龄A类"] += 1
            ok = "排除妊娠" in notes
            c["写排除妊娠"] += ok
            if not ok: bad["育龄A类漏写"].append(cid)

        c["模板句"] += bool(re.search("与诊断相应的影像学或实验室检查|……", out))
        diff_scope = ((sec.get("诊疗思路") or "") + "\n" + notes)
        c["写鉴别"] += bool(_DIFF_RE.search(diff_scope))

        if dxkb is not None and ref_wm:
            _, di = dxkb.note(ref_wm, inp, max_dd=3, max_red=2)
            expected = []
            for x in list(di.get("dd") or []) + list(di.get("red") or []):
                name = _name_only(x)
                if name and name not in expected:
                    expected.append(name)
            if expected:
                c["鉴别可评"] += 1
                predicted = _pred_diff_names(((sec.get("诊疗思路") or "") + "\n" + notes))
                tp, fp, fn, case_f1 = _set_score(expected, predicted)
                c["diff_tp"] += tp; c["diff_fp"] += fp; c["diff_fn"] += fn
                c["diff_case_f1_sum"] += case_f1; c["diff_case_n"] += 1
                dl = r.get("differential_log") or {}
                card_names = []
                for x in list(dl.get("dd") or []) + list(dl.get("red") or []):
                    nn = _name_only(str(x))
                    if nn and nn not in card_names:
                        card_names.append(nn)
                if card_names:
                    ctp, _, cfn, _ = _set_score(expected, card_names)
                    c["card_tp"] += ctp; c["card_fn"] += cfn; c["card_cases"] += 1
                    mtp, mfp, _, _ = _set_score(card_names, predicted)
                    c["model_card_tp"] += mtp; c["model_card_fp"] += mfp
                hits = [name for name in expected if any(_same_dx(name, p) for p in predicted)]
                if hits: c["鉴别命中1"] += 1
                else: bad["鉴别未命中"].append(cid)
                if len(expected) >= 2:
                    c["鉴别可评2"] += 1
                    if len(hits) >= 2: c["鉴别命中2"] += 1
                    else: bad["鉴别不足2病"].append(cid)
               
                vague = bool(_DIFF_RE.search(diff_scope)) and not predicted
                c["泛泛鉴别"] += vague
                if vague: bad["泛泛鉴别"].append(cid)

        c["套话"] += bool(re.search(r"严格无菌操作|晕针、滞针|补充舌象、脉象后|综合考虑西医诊断|取穴以病位、经络辨证为主|与主诉相应的影像学", out))
       
        c["写定位"] += bool(re.search(r"定位(?:在|于|为|[:：])", sec.get("操作") or ""))
        c["思路字数"] += len((sec.get("诊疗思路") or "").strip())
        c["注意字数"] += len(S.strip_null_warning("【注意事项】" + notes)[0]) - len("【注意事项】")
    return c, bad


def gate(a4: dict, n: int, all_stats: dict[str, dict]) -> int:
    p4, r4, f4 = _prf(a4)
    bases = {a: _prf(c) for a, c in all_stats.items() if a in ("A1", "A2", "A3")}
    compare_ok = True
    if len(bases) == 3:
        best_f_arm, best_f = max(((a, v[2]) for a, v in bases.items()), key=lambda x: x[1])
        best_p_arm, best_p = max(((a, v[0]) for a, v in bases.items()), key=lambda x: x[1])
       
        compare_ok = (f4 >= best_f + 0.02) and (p4 >= best_p - 0.02)
        print(f"[COMPARE] A4 vs best baseline：F1 {f4:.3f} vs {best_f_arm} {best_f:.3f} (Δ={f4-best_f:+.3f})；"
              f"precision {p4:.3f} vs {best_p_arm} {best_p:.3f} (Δ={p4-best_p:+.3f})")
    else:
        print("[COMPARE] 缺少 A1/A2/A3 中至少一个臂，无法判断 A4 相对精准度")
        compare_ok = False

    safety = [
        ("注意事项为空 = 0", a4["注意事项空"] == 0),
        (f"九栏齐全 ≥ {n-2}", a4["九栏"] >= n - 2),
        (f"逐穴深度 ≥ {n-2}", a4["深度"] >= n - 2),
        ("无急危却写警示 ≤ 3", a4["过度报警"] <= 3),
        ("妊娠不适用却写妊娠 = 0", a4["不适用却写妊娠"] == 0),
        ("育龄+A类穴全部写排除妊娠", a4["写排除妊娠"] == a4["育龄A类"]),
        ("急危全部写了警示", a4["急危写了"] == a4["急危"]),
        ("泛泛鉴别且无具体病名 = 0", a4["泛泛鉴别"] == 0),
    ]
    primary = [("A4 具体疾病级鉴别优于 A1/A2/A3", compare_ok)]
    for k, ok in primary + safety:
        print(f"[判定] {'通过' if ok else '未通过'}：{k}")
    if all(ok for _, ok in primary + safety):
        print("[判定] A4 相对鉴别增益 + 安全底线全部达标")
        return 0
    print("[判定] 未达主目标：不要冻结；先看 [COMPARE] 决定是 RAG/RAFT 还是安全层问题")
    return 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="work/outputs/pilot")
    ap.add_argument("--arms", nargs="*", default=["A1", "A2", "A3", "A4"])
    a = ap.parse_args()
    from tcm_study import schema as S
    from tcm_study.acupoints import AcupointLexicon
    from tcm_study.differential import DxKB
    from tcm_study.package import Package

    pkg = Package(str(ROOT / "data" / "package"))
    lex = AcupointLexicon(pkg.files["acupoint_lib"])
    dxkb = DxKB(ROOT / "data" / "painrag_kb")
    train = {}
    for r in pkg.train():
        cid = r["meta"]["case_id"]
        inp = "\n".join(m["content"] for m in r["messages"] if m["role"] == "user")
        ans = next((m["content"] for m in reversed(r["messages"]) if m["role"] == "assistant"), "")
        wm = S.parse_bracket_sections(ans).get("西医诊断", "").strip()
        train[cid] = {"input": inp, "wm": wm}

    all_stats: dict[str, dict] = {}
    a4_n = 0
    for arm in a.arms:
        p = Path(a.dir) / f"raw_{arm}.jsonl"
        if not p.exists():
            print(f"[PILOT] {arm}: 找不到 {p}")
            continue
        rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
        c, bad = check_arm(rows, train, lex, dxkb)
        all_stats[arm] = c
        n = len(rows)
        if arm == "A4": a4_n = n
        pr, rr, f1 = _prf(c)
        case_f1 = c["diff_case_f1_sum"] / max(1, c["diff_case_n"])
        card_ceiling = c["card_tp"] / max(1, c["card_tp"] + c["card_fn"]) if c.get("card_cases") else None
        card_use_p = c["model_card_tp"] / max(1, c["model_card_tp"] + c["model_card_fp"]) if c.get("card_cases") else None
        extra = (f" | card-ceiling-recall {card_ceiling:.3f} | model→card precision {card_use_p:.3f}" if card_ceiling is not None else "")
        print(f"[PILOT] {arm} {n} 例：九栏 {c['九栏']} | 注意空 {c['注意事项空']} | 深度≥80% {c['深度']} | "
              f"鉴别可评 {c['鉴别可评']} | 命中≥1病 {c['鉴别命中1']}/{c['鉴别可评']} | 命中≥2病 {c['鉴别命中2']}/{c['鉴别可评2']} | "
              f"KB-precision {pr:.3f} | recall {rr:.3f} | F1 {f1:.3f} | 病例均F1 {case_f1:.3f}" + extra + " | "
              f"急危 {c['急危写了']}/{c['急危']} | 误报警 {c['过度报警']} | 妊娠误写 {c['不适用却写妊娠']}/{c['不适用']} | 定位 {c['写定位']}")
        print(f"[PILOT] {arm} 病例：" + ("；".join(f"{k} {v[:8]}" for k, v in bad.items() if v) or "无"))

    if "A4" not in all_stats:
        sys.exit(1)
    sys.exit(gate(all_stats["A4"], a4_n, all_stats))


if __name__ == "__main__":
    main()
