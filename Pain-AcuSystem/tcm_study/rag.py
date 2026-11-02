from __future__ import annotations


import hashlib
import os
import re
from collections import OrderedDict
from pathlib import Path

INSTR_RE = re.compile(r"^请根据以下病例给出针灸诊疗方案。\s*", re.M)
DEFAULT_BUDGET = {"west": 1400, "tcm": 2000, "acupoint": 600}
DEFAULT_CARDS = {"max_points": 12, "max_chars": 1800, "manipulation_cards": True, "max_manip_cards": 4, "manip_max_chars": 1200,
                 "safety_check": True, "differential_card": True, "differential_card_max_chars": 2200,
                 "differential_max_dd": 3, "differential_max_red": 2,
                 "guideline_safety": True, "guideline_safety_max_chars": 1800}


class RagIndex:

    def __init__(self, index_dir: str | Path, device: str | None = None, cache: int = 2048, use_dense: str = "auto"):
        from painrag.retriever import PainRAG
        self.dir = Path(index_dir)
        if not (self.dir / "concepts.json").exists():
            raise FileNotFoundError(f"painrag knowledge base not found in {self.dir} (expected concepts.json, west.jsonl, tcm.jsonl, ...)")
        self.rag = PainRAG(kb_dir=str(self.dir), use_dense=use_dense, device=device or os.getenv("TCM_RAG_DEVICE", "cpu"))
        self.docs = self.rag.docs
        self.dense = bool(self.rag.vec)
        self.encoder_name = (self.dir / "embedding_model.txt").read_text(encoding="utf-8").strip() if self.dense and (self.dir / "embedding_model.txt").exists() else None
        if not self.dense:
            print("[RAG][WARN] dense vectors (<cat>.npy) not found - BM25 only. Run scripts/03_build_rag.py before training/generation.", flush=True)

    def search(self, cat: str, query: str, k: int) -> list[tuple[float, dict]]:
        return self.rag.search(query, cat, k)


class RagEvidence:
    def __init__(self, index: RagIndex, cfg: dict):
        self.index, self.cfg = index, cfg
        self.budget = dict(DEFAULT_BUDGET, **(cfg.get("budget") or {}))
        self.top = int(cfg.get("top_concepts", 2))
        self.domains = tuple(cfg.get("domains") or ("west", "tcm"))
        self.min_score = float(cfg.get("min_score", 0.3))
        self.max_chars = int(cfg.get("max_chars", 6000))
        self.cards_cfg = dict(DEFAULT_CARDS, **(cfg.get("cards") or {}))
        self._dxkb = None
        if self.cards_cfg.get("differential_card", True):
            from tcm_study.differential import DxKB
            self._dxkb = DxKB(self.index.dir)
        self._cache: "OrderedDict[str, tuple[str, dict]]" = OrderedDict()

    @staticmethod
    def query_from_messages(messages: list[dict]) -> str:
        return INSTR_RE.sub("", "\n".join(m["content"] for m in messages if m["role"] == "user")).strip()

    def build(self, query: str) -> tuple[str, dict]:
        """Step 1: disease-level evidence for the case text (deterministic for a given query; cached)."""
        key = "ev|" + hashlib.sha256(query.encode("utf-8")).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        text, trace = self.index.rag.evidence(query, top_concepts=self.top, budget=self.budget, domains=self.domains, min_score=self.min_score)
        text = text[: self.max_chars]
        log = {"concepts": trace.get("concepts"), "units": trace.get("units"), "selection_reason": trace.get("reason", "ok"),
               "top_score": (trace.get("concepts") or [{}])[0].get("score") if trace.get("concepts") else None,
               "dense": self.index.dense, "budget": self.budget, "domains": list(self.domains),
               "evidence_chars": len(text), "evidence_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
        self._cache[key] = (text, log)
        if len(self._cache) > 4096:
            self._cache.popitem(last=False)
        return text, log

    def cards(self, points: list[str], manipulations: list[str] | None = None) -> tuple[str, dict]:
        """Step 2: acupoint safety cards for the given points (+ manipulation cards). Deterministic; cached."""
        c = self.cards_cfg
        manipulations = list(manipulations or []) if c.get("manipulation_cards", True) else []
        key = "cards|" + hashlib.sha256(("|".join(points) + "#" + "|".join(manipulations)).encode("utf-8")).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        ptext, taken = self.index.rag.point_cards(points, int(c["max_points"]), int(c["max_chars"]))
        mtext, mtaken = self.index.rag.manip_cards(manipulations, int(c["max_manip_cards"]), int(c["manip_max_chars"])) if manipulations else ("", [])
        text = "\n\n".join(t for t in (ptext, mtext) if t)
        log = {"points_requested": list(points), "points_carded": taken, "manipulations_requested": manipulations, "manipulations_carded": mtaken,
               "cards_chars": len(text), "cards_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}
        self._cache[key] = (text, log)
        return text, log

    def manip_names_in(self, text: str) -> list[str]:
        return self.index.rag.manip_names_in(text)

    def differential_card(self, western_dx: str, case_text: str) -> tuple[str, dict]:
    
        if self._dxkb is None or not (western_dx or "").strip():
            return "", {"concept": None}
        c = self.cards_cfg
        return self._dxkb.card(
            western_dx,
            case_text,
            max_dd=int(c.get("differential_max_dd", 3)),
            max_red=int(c.get("differential_max_red", 2)),
            max_chars=int(c.get("differential_card_max_chars", 2200)),
        )
