# -*- coding: utf-8 -*-

import sys, json, re, os, collections
sys.stdout.reconfigure(encoding="utf-8")
B = r""
sys.path.insert(0, B); os.chdir(B)
import pandas as pd
from painrag.retriever import PainRAG, regions_of
from painrag.textutil import norm_key
from painrag.cli import case_from_record

rag = PainRAG(use_dense="off", verbose=False)
recs = [json.loads(l) for l in open("", encoding="utf-8") if l.strip()]
recs = {r["meta"]["case_id"]: r for r in recs if r["meta"]["task"] == "initial"}
fmap = {d["case_id"]: d for d in (json.loads(l) for l in open("", encoding="utf-8"))}
recs = {k: v for k, v in recs.items() if k in fmap}          #
ann = pd.read_excel("", sheet_name="", dtype=str, keep_default_na=False).set_index("case_id")

MODE = sys.argv[1] if len(sys.argv) > 1 else "explore"
icd_re = re.compile(r"[0-9A-Z]{2}[0-9A-Z]{2}(?:\.[0-9A-Z]{1,2})?")
def icds(s):
    return [x for x in icd_re.findall(str(s or "").upper()) if not x.isdigit()]

if MODE == "explore":
    a = ann.loc[list(recs)]
    print("训练初诊", len(a))
    print("来源:", a["来源"].value_counts().to_dict())
    print("ICD-11 非空:", (a["ICD-11"].str.strip() != "").sum(), "| ICD来源:", a["ICD来源"].value_counts().to_dict())
    print("ICD 样例:", a["ICD-11"][a["ICD-11"] != ""].head(25).tolist())
    print("主分类_部位:", a["主分类_部位"].value_counts().to_dict())
    print("主分类_7类:", a["主分类_7类"].value_counts().to_dict())
    print("西医诊断_标准化 样例:", a["西医诊断_标准化"][a["西医诊断_标准化"] != ""].head(12).tolist())
    c = [v for k, v in rag.concepts.items() if not k.startswith("AP-") and not re.match(r"^GL\d+$", k)]
    print("概念", len(c), "有icd", sum(bool(x.get("icd")) for x in c), "样例", [(x["name"], x.get("icd")) for x in c[:12]])
    print("概念 icd 首字符分布:", collections.Counter((x.get("icd") or "?")[:1] for x in c).most_common())
    print("病例 ICD 首字符分布:", collections.Counter(z[:1] for s in a["ICD-11"] for z in icds(s)[:1]).most_common())
    print("按来源 ICD 非空:", a.groupby("来源")["ICD-11"].apply(lambda s: (s.str.strip() != "").sum()).to_dict())
    print("有阳性辅助检查的病例:", sum(bool(fmap[k]["阳性辅助检查"]) for k in recs), "| 其中有ICD:", sum(bool(fmap[k]["阳性辅助检查"]) and a.loc[k, "ICD-11"].strip() != "" for k in recs))
    sys.exit()

# ============================ 评估 ============================
REG6 = {"头": {"头面痛"}, "面口": {"头面痛"}, "颈": {"颈肩上肢痛"}, "肩": {"颈肩上肢痛"}, "肘臂": {"颈肩上肢痛"}, "腕手": {"颈肩上肢痛"},
        "背": {"腰背痛"}, "腰": {"腰背痛"}, "骶臀髋": {"腰背痛", "下肢关节痛"}, "膝": {"下肢关节痛"}, "踝足": {"下肢关节痛"},
        "腹胃": {"内脏/盆腔痛"}, "盆会阴": {"内脏/盆腔痛"}, "胸": {"内脏/盆腔痛", "腰背痛"}, "全身": {"广泛/多部位痛"}}
CAT = {"F": "肌骨", "N": "肌骨", "8": "神经", "D": "消化", "G": "泌尿生殖", "1": "感染", "2": "肿瘤", "6": "精神", "B": "循环", "C": "呼吸", "E": "皮肤", "5": "内分泌", "9": "眼", "A": "耳", "4": "免疫", "7": "睡眠", "J": "妊娠", "L": "发育", "S": "传统医学"}
def cat_of(code):
    return None if not code or code[0] == "M" else CAT.get(code[0], code[0])
def creg6(cid):
    out = set()
    for r in rag._creg.get(cid, set()):
        out |= REG6.get(r, set())
    return out
def names_of(cid):
    c = rag.concepts[cid]
    return [norm_key(x) for x in [c["name"]] + c.get("aliases", []) + c.get("west_names", []) if len(norm_key(x)) >= 2]
def m_name(dx, cid):
    dxk = norm_key(dx)
    return any(n == dxk or n in dxk or (len(n) >= 3 and dxk in n) for n in names_of(cid))

cases = []
for cid, r in recs.items():
    a = ann.loc[cid]
    code = (icds(a["ICD-11"]) or [None])[0]
    m = re.search(r"【西医诊断】\s*([^\n]{2,60})", r["messages"][-1]["content"])
    if not code or not m:
        continue
    cases.append({"cid": cid, "dx": re.split(r"[；;（(，,]", m.group(1).strip())[0], "icd": code, "reg": a["主分类_部位"], "src": a["来源"],
                  "icd_src": a["ICD来源"], "aux": bool(fmap[cid]["阳性辅助检查"]), "text": case_from_record(r)})
print("有西医诊断且有 ICD 的训练初诊病例:", len(cases), "| ICD 为推断待核:", sum(c["icd_src"].startswith("推断") for c in cases))

def judge(c, concept_id):
    k = rag.concepts[concept_id]
    kc = (icds(k.get("icd")) or [None])[0]
    reg_ok = c["reg"] in creg6(concept_id) if c["reg"] not in ("非疼痛", "广泛/多部位痛") else None
    ca, cb = cat_of(c["icd"]), cat_of(kc)
    cat_ok = (ca is None or cb is None or ca == cb)
    return {"名称": m_name(c["dx"], concept_id), "ICD四位": bool(kc) and kc[:4] == c["icd"][:4], "ICD前三位": bool(kc) and kc[:3] == c["icd"][:3],
            "ICD章": bool(kc) and kc[0] == c["icd"][0], "部位": reg_ok, "部位+类别": (reg_ok and cat_ok) if reg_ok is not None else None}
METRICS = ["名称", "ICD四位", "ICD前三位", "ICD章", "部位", "部位+类别"]
routed = {"全文": {}, "结构化": {}}
for c in cases:
    routed["全文"][c["cid"]] = [x[0] for x in rag.route(c["text"], 2)]
    routed["结构化"][c["cid"]] = [x[0] for x in rag.route(c["text"], 2, fmap[c["cid"]])]

SUBSETS = [("全部", lambda c: True), ("医院门诊（H1/H2/H3）", lambda c: c["src"].startswith("医院")), ("文献个案+名家医案", lambda c: c["src"] in ("中医成果库", "名中医书籍")),
           ("有影像/化验阳性结果", lambda c: c["aux"]), ("可靠子集（个案医案 或 有影像化验）", lambda c: c["src"] in ("中医成果库", "名中医书籍") or c["aux"]),
           ("可靠子集且 ICD 非推断", lambda c: (c["src"] in ("中医成果库", "名中医书籍") or c["aux"]) and not c["icd_src"].startswith("推断"))]
rows = []
for sname, f in SUBSETS:
    sub = [c for c in cases if f(c)]
    for mode in ("全文", "结构化"):
        for top in (1, 2):
            row = {"子集": sname, "n": len(sub), "路由": mode, "Top": top}
            for mt in METRICS:
                hit = den = 0
                for c in sub:
                    js = [judge(c, x)[mt] for x in routed[mode][c["cid"]][:top]]
                    if any(j is None for j in js) or not js:
                        if not js:
                            den += 1
                        continue
                    den += 1; hit += any(js)
                row[mt] = f"{hit}/{den} ({hit / den:.0%})" if den else "-"
            rows.append(row)
df = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20); pd.set_option("display.unicode.east_asian_width", True)
print(df.to_string(index=False))
df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "route_eval2.csv"), index=False, encoding="utf-8-sig")

kb4 = {(icds(v.get("icd")) or [""])[0][:4] for v in rag.concepts.values()}
for sname, f in SUBSETS[:5]:
    sub = [c for c in cases if f(c)]
    print(f"库内有同四位码概念：{sname} {sum(c['icd'][:4] in kb4 for c in sub)}/{len(sub)}")
