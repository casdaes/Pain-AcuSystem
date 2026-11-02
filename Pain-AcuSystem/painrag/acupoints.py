# -*- coding: utf-8 -*-

from __future__ import annotations
import re
from .textutil import val


class AcuLexicon:
    def __init__(self, points: list, whitelist: list = None):
        self.points = points                               # [dict]
        self.by_name = {}                                  # 规范名/别名 -> dict
        for p in points:
            self.by_name[p["穴位名称"]] = p
            for a in p.get("aliases", []):
                self.by_name.setdefault(a, p)
            if p.get("WHO code"):
                self.by_name.setdefault(p["WHO code"].upper(), p)
        names = sorted(self.by_name.keys(), key=len, reverse=True)
        self._re = re.compile("|".join(re.escape(n) for n in names if n))
        self.whitelist = [(cat, std, re.compile(rx), note) for cat, std, rx, note in (whitelist or []) if rx]

    def canonical(self, name: str) -> str:
        p = self.by_name.get(name) or self.by_name.get(name.upper())
        return p["穴位名称"] if p else ""

    def get(self, name: str) -> dict:
        return self.by_name.get(name) or self.by_name.get(name.upper()) or {}

    def extract(self, text: str) -> list:
        """返回文本中出现的规范穴名（去重、保持先后）；夹脊/阿是穴等非单穴表述按白名单规范表述返回。"""
        text = val(text)
        if not text:
            return []
        out, seen = [], set()
        for m in self._re.finditer(text):
            c = self.canonical(m.group(0))
            if c and c not in seen:
                seen.add(c)
                out.append(c)
        for cat, std, rx, note in self.whitelist:
            if std not in seen and rx.search(text):
                seen.add(std)
                out.append(std)
        return out

    def doc_text(self, p: dict, short: bool = False) -> str:
        """穴位库单穴文档正文；short=True 为证据包用的一行摘要。"""
        code = p.get("WHO code", "")
        name = p["穴位名称"] + (f"（{code}）" if code else "")
        safety = []
        if p.get("禁深刺") == "是":
            safety.append("禁深刺")
        if p.get("限提插捻转") == "是":
            safety.append("限提插捻转")
        if p.get("出针按压") == "是":
            safety.append("出针后按压")
        if p.get("妊娠禁忌") and p["妊娠禁忌"] not in ("无记载", "否"):
            safety.append("妊娠：" + p["妊娠禁忌"] + (f"（{p['妊娠说明']}）" if p.get("妊娠说明") else ""))
        if p.get("解剖危险区") and p["解剖危险区"] != "无记载":
            safety.append(f"解剖危险区：{p['解剖危险区']}" + (f"（危险等级 {p['危险等级']}）" if p.get("危险等级") else ""))
        depth = p.get("深度（针灸学教科书）", "")
        if p.get("深度上限_寸"):
            depth += f"（深度上限 {p['深度上限_寸']} 寸）"
        if short:
            parts = [name, p.get("归经", ""), "定位：" + p.get("定位", ""), "刺法：" + depth]
            if safety:
                parts.append("安全：" + "；".join(safety))
            return "｜".join(x for x in parts if x and not x.endswith("："))
        lines = [f"穴位：{name}｜类别：{p.get('穴位类别','')}｜归经：{p.get('归经','')}"]
        if p.get("aliases"):
            lines.append("别名：" + "、".join(p["aliases"]))
        if p.get("定位"):
            lines.append("定位：" + p["定位"])
        if depth:
            lines.append("针刺深度与方向：" + depth)
        if p.get("主要针法"):
            lines.append("主要针法：" + p["主要针法"])
        if safety:
            lines.append("安全提示：" + "；".join(safety))
        if p.get("备注"):
            lines.append("备注：" + p["备注"])
        return "\n".join(lines)
