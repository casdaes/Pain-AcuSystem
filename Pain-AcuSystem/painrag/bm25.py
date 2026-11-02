# -*- coding: utf-8 -*-

from __future__ import annotations
import math
import pickle
import heapq
from array import array
from collections import Counter, defaultdict

from .textutil import tokenize


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.post = {}
        self.dl = array("I")
        self.avgdl = 1.0
        self.idf = {}
        self.N = 0

    def fit(self, texts: list):
        tmp = defaultdict(lambda: ([], []))
        self.dl = array("I")
        for i, t in enumerate(texts):
            toks = tokenize(t)
            self.dl.append(len(toks))
            for tok, tf in Counter(toks).items():
                a, b = tmp[tok]
                a.append(i)
                b.append(min(tf, 65535))
        self.N = len(texts)
        self.avgdl = (sum(self.dl) / self.N) if self.N else 1.0
        self.post = {tok: (array("I", a), array("H", b)) for tok, (a, b) in tmp.items()}
        self.idf = {tok: math.log(1 + (self.N - len(a) + 0.5) / (len(a) + 0.5)) for tok, (a, _) in self.post.items()}
        return self

    def scores(self, query: str, only: set = None) -> dict:

        q = Counter(tokenize(query))
        sc = defaultdict(float)
        k1, b, avg = self.k1, self.b, self.avgdl
        for tok, qtf in q.items():
            p = self.post.get(tok)
            if not p:
                continue
            idf = self.idf[tok]
            idxs, tfs = p
            for i, tf in zip(idxs, tfs):
                if only is not None and i not in only:
                    continue
                dl = self.dl[i]
                sc[i] += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg))
        return sc

    def search(self, query: str, k: int = 10, only: set = None) -> list:
        sc = self.scores(query, only)
        if not sc:
            return []
        top = heapq.nlargest(k, sc.items(), key=lambda x: x[1])
        return [(float(s), int(i)) for i, s in top]

    def save(self, path: str):
        with open(path, "wb") as fh:
            pickle.dump(dict(k1=self.k1, b=self.b, post=self.post, dl=self.dl, avgdl=self.avgdl, idf=self.idf, N=self.N), fh, protocol=4)

    @classmethod
    def load(cls, path: str) -> "BM25":
        d = pickle.load(open(path, "rb"))
        o = cls(d["k1"], d["b"])
        o.post, o.dl, o.avgdl, o.idf, o.N = d["post"], d["dl"], d["avgdl"], d["idf"], d["N"]
        return o
