# -*- coding: utf-8 -*-
from __future__ import annotations
import hashlib
import os
import re

from .retriever import PainRAG

INSTR_RE = re.compile(r"^请根据以下病例给出针灸诊疗方案。\s*", re.M)
EVIDENCE_WRAPPER = ("\n\n【检索证据】\n{evidence}\n\n请结合上述检索证据与病例事实作答；二者冲突时以病例事实为准，"
                    "不得编造病例中没有的信息；穴位定位/深度以【穴位参考】为准。")


class Evidence:
    def __init__(self, kb_dir: str = None, max_chars: int = 6000, budget: dict = None, top_concepts: int = 2,
                 domains=("west", "tcm", "acupoint"), device: str = None):
        self.rag = PainRAG(kb_dir=kb_dir, use_dense="auto", device=device or os.getenv("TCM_RAG_DEVICE", "cpu"))
        self.max_chars, self.budget, self.top, self.domains = max_chars, budget, top_concepts, domains
        self._cache = {}

    @staticmethod
    def query_from_messages(messages: list) -> str:
        return INSTR_RE.sub("", "\n".join(m["content"] for m in messages if m["role"] == "user")).strip()

    def build(self, query: str) -> tuple:
        key = hashlib.sha256(query.encode("utf-8")).hexdigest()
        if key in self._cache:
            return self._cache[key]
        text, trace = self.rag.evidence(query, top_concepts=self.top, budget=self.budget, domains=self.domains)
        text = text[: self.max_chars]
        log = {"concepts": trace.get("concepts"), "units": trace.get("units"), "evidence_chars": len(text),
               "evidence_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
        self._cache[key] = (text, log)
        return text, log

    def compose(self, messages: list) -> list:
        text, _ = self.build(self.query_from_messages(messages))
        msgs = [dict(m) for m in messages if m["role"] in ("system", "user")]
        if text:
            msgs[-1] = {**msgs[-1], "content": msgs[-1]["content"] + EVIDENCE_WRAPPER.format(evidence=text)}
        return msgs
