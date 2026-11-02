# -*- coding: utf-8 -*-
from __future__ import annotations
import re

LIST_KEYS = ("阳性症状", "阴性症状", "诱因", "加重因素", "缓解因素", "时间规律", "阳性体征", "阴性体征", "阳性辅助检查", "阴性辅助检查", "既往史与合并症")
EMPTY = {"", "未记录", "未描述", "不详", "无", "舌脉不详", "未提供"}
_NOTE = re.compile(r"[（(][^（）()]*[）)]")          # 标注者写在括号里的说明不进查询


def _clean(v) -> str:
    s = _NOTE.sub("", str(v or "")).strip(" ，。；;、")
    return "" if s in EMPTY else s


def _items(f: dict, key: str, limit: int = 12) -> list:
    v = f.get(key) or []
    if isinstance(v, str):
        v = [v]
    out = []
    for x in v:
        s = _clean(x)
        if s and s not in out:
            out.append(s)
    return out[:limit]


def chief(f: dict) -> dict:
    c = f.get("主诉") or {}
    part = [x for x in (_clean(p) for p in (c.get("部位") or [])) if x]
    qual = [x for x in (_clean(p) for p in (c.get("性质") or [])) if x and "未描述" not in x]
    return {"部位": part, "性质": qual, "侧别": _clean(c.get("侧别")), "病程": _clean(c.get("病程"))}


def compose_queries(f: dict) -> tuple:
    """返回 (西医查询, 中医查询)。字段全空时返回 ('', '')。"""
    c = chief(f)
    who = _clean(f.get("年龄性别"))
    side = c["侧别"] if c["侧别"] not in ("双侧", "") and not any(c["侧别"] in p for p in c["部位"]) else ""
    cc = (side + "、".join(c["部位"])) + ("".join(("，" + q) if i == 0 else ("、" + q) for i, q in enumerate(c["性质"])) if c["性质"] else "") + (("，" + c["病程"]) if c["病程"] else "")
    seg = []
    if cc:
        seg.append("主诉：" + cc)
    for label, key in (("诱因", "诱因"), ("加重", "加重因素"), ("缓解", "缓解因素"), ("症状", "阳性症状"), ("体征", "阳性体征"), ("辅助检查", "阳性辅助检查"), ("既往", "既往史与合并症")):
        it = _items(f, key)
        if it:
            seg.append(label + "：" + ("；" if key in ("阳性体征", "阳性辅助检查") else "，").join(it))
    west = (who + "。" if who else "") + "。".join(seg) + ("。" if seg else "")
    tp = f.get("舌脉") or {}
    tongue, pulse = _clean(tp.get("舌")), _clean(tp.get("脉"))
    body = [cc] if cc else []
    body += _items(f, "阳性症状")
    body += ["遇" + x.lstrip("遇") + "加重" if not x.endswith(("加重", "甚", "重")) and len(x) <= 6 else x for x in _items(f, "加重因素", 4)]
    body += ["得" + x.lstrip("得") + "则减" if not x.endswith(("减", "缓", "舒", "缓解")) and len(x) <= 6 else x for x in _items(f, "缓解因素", 4)]
    tcm = (who + "。" if who else "") + "，".join(body) + ("。" if body else "") + "；".join(x for x in (tongue, pulse) if x) + ("。" if (tongue or pulse) else "")
    if not seg and not body:
        return "", ""
    return west, tcm


def texts(f: dict) -> dict:
    """路由用的几段文本：pos=全部阳性要素；neg=全部阴性要素；aux=阳性辅助检查；hx=既往史；chief=主诉部位。"""
    c = chief(f)
    pos = "；".join(["、".join(c["部位"] + c["性质"])] + _items(f, "阳性症状") + _items(f, "阳性体征") + _items(f, "阳性辅助检查") + _items(f, "既往史与合并症") + _items(f, "诱因"))
    neg = "；".join(_items(f, "阴性症状") + _items(f, "阴性体征") + _items(f, "阴性辅助检查"))
    return {"pos": pos, "neg": neg, "aux": "；".join(_items(f, "阳性辅助检查")), "hx": "；".join(_items(f, "既往史与合并症")), "chief": "、".join(c["部位"])}


def usable(f) -> bool:
    """结构化结果是否足以用于检索（至少有部位或症状）。"""
    if not isinstance(f, dict):
        return False
    c = chief(f)
    return bool(c["部位"] or _items(f, "阳性症状") or _items(f, "阳性体征"))
