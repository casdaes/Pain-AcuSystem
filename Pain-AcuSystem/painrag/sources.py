# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import re
import json
import glob
from collections import defaultdict
import openpyxl

from . import config as C
from .textutil import val

SPLIT_NAMES = re.compile(r"[；;、，,/|\n]+")


def split_names(s) -> list:
    return [x.strip() for x in SPLIT_NAMES.split(val(s)) if x.strip()]


# ------------------------------------------------------------------ 概念主表 + 映射
def load_concepts():

    wb = openpyxl.load_workbook(C.MAP_XLSX, read_only=True)
    main = {}
    for r in list(wb[""].iter_rows(values_only=True))[1:]:
        if not r[0]:
            continue
        main[r[0]] = dict(cid=r[0], name=val(r[1]), aliases=split_names(r[2]), icd=val(r[6]), icd_title=val(r[7]),
                          post=val(r[14]) if len(r) > 14 else "", books=val(r[5]))
    cid2gl, gl_titles = defaultdict(list), {}
    for r in list(wb[""].iter_rows(values_only=True))[1:]:
        if not r[0]:
            continue
        gl_titles[r[0]] = val(r[1])
        if r[6]:
            cid2gl[r[6]].append((r[0], val(r[1]), val(r[2]), val(r[3]), val(r[5])))
    return main, cid2gl, gl_titles


# ------------------------------------------------------------------ 指南工作簿
GL_FILES = None
_cache = {}


def gl_files():
    global GL_FILES
    if GL_FILES is None:
        GL_FILES = {os.path.basename(f).split("_")[0]: f for f in glob.glob(os.path.join(C.GL_DIR, ""))}
    return GL_FILES


def gl_title_from_file(gl: str) -> str:
    f = gl_files().get(gl)
    if not f:
        return gl
    return os.path.splitext(os.path.basename(f))[0].split("_", 1)[-1]


def sheets(gl: str) -> dict:
 
    if gl in _cache:
        return _cache[gl]
    f = gl_files().get(gl)
    if not f:
        _cache[gl] = {}
        return {}
    w = openpyxl.load_workbook(f, read_only=True, data_only=True)
    d = {}
    for s in w.sheetnames:
        rs = list(w[s].iter_rows(values_only=True))
        if len(rs) >= 2:
            hdr = [str(h).strip() if h else "" for h in rs[0]]
            d[s.split("_", 1)[-1]] = [dict(zip(hdr, r)) for r in rs[2:] if any(x not in (None, "") for x in r)]
    w.close()
    _cache[gl] = d
    return d


def evidence_defs(gl: str) -> dict:
   
    out = {}
    for r in sheets(gl).get("证据分级说明表", []):
        if r.get("evidence_def_id"):
            out[r["evidence_def_id"]] = f"{val(r.get('level_code'))}：{val(r.get('level_definition'))}".strip("：")
    return out


# ------------------------------------------------------------------ 卡片
def load_cards() -> dict:
    cards = {}
    for f in glob.glob(os.path.join(C.CARDS_DIR, "")):
        try:
            c = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] 卡片无法解析 {os.path.basename(f)}: {e}")
            continue
        if c.get("concept_id"):
            cards[c["concept_id"]] = c
    return cards


# ------------------------------------------------------------------ 书籍原文
def load_booksec() -> dict:
    if not os.path.exists(C.BOOKSEC):
        print(f"[WARN] 未找到 book_sections.json：{C.BOOKSEC}")
        return {"matched": {}, "unmatched": []}
    return json.load(open(C.BOOKSEC, encoding="utf-8"))


# ------------------------------------------------------------------ 穴位库
ACU_COLS = ["穴位类别", "WHO code", "穴位名称", "别名", "定位", "归经", "深度（针灸学教科书）", "深度上限_寸", "主要针法",
            "禁深刺", "限提插捻转", "出针按压", "妊娠禁忌", "妊娠说明", "解剖危险区", "危险等级", "对应SE条目", "备注"]


def load_acupoints() -> list:
    
    w = openpyxl.load_workbook(C.ACU_XLSX, read_only=True, data_only=True)
    rs = list(w["穴位标准库"].iter_rows(values_only=True))
    hdr = [str(h).strip() if h else "" for h in rs[0]]
    pts = []
    for r in rs[1:]:
        d = dict(zip(hdr, r))
        if not val(d.get("穴位名称")):
            continue
        d = {k: val(v) for k, v in d.items() if k}
        d["aliases"] = [x for x in re.split(r"[|｜、;；/]", d.get("别名", "")) if x.strip()]
        pts.append(d)
    wl = []
    if "非单穴白名单" in w.sheetnames:
        rows = list(w["非单穴白名单"].iter_rows(values_only=True))
        for r in rows:
            if r and r[0] and r[1] and r[2] and str(r[0]).strip() not in ("类别",) and not str(r[0]).startswith(("非单穴", "这些")):
                wl.append((val(r[0]), val(r[1]), val(r[2]), val(r[3])))
    w.close()
    return pts, wl
