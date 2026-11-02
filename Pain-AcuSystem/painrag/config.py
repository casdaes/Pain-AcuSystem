# -*- coding: utf-8 -*-

import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                   # 项目根


def _p(env, default):
    return os.environ.get(env, default)


# ---- 源数据 ----
_SRC = _p("PAINRAG_SRC", r"")
CARDS_DIR   = _p("PAINRAG_CARDS",   os.path.join(_SRC, "书籍疾病卡片"))
GL_DIR      = _p("PAINRAG_GL",      os.path.join(_SRC, "所有指南信息抽提整合"))
MAP_XLSX    = _p("PAINRAG_MAP",     os.path.join(_SRC, "指南.xlsx"))
BOOKSEC     = _p("PAINRAG_BOOKSEC", os.path.join(_SRC, "RAG知识库", "book_sections.json"))
ACU_XLSX    = _p("PAINRAG_ACU",     os.path.join(ROOT, "data", "package", "针灸穴位标准库.xlsx"))

# ---- 知识库与索引 ----
KB_DIR      = _p("PAINRAG_KB",      os.path.join(ROOT, "data", "painrag_kb"))
MD_DIR      = _p("PAINRAG_MD",      os.path.join(ROOT, "data", "painrag_md"))
COLLECTIONS = ("west", "tcm", "acupoint", "concept")

# ---- 切块参数（字符数）----
MAX_CHARS   = int(_p("PAINRAG_MAX_CHARS", "650"))
MIN_TAIL    = 120
BOOK_OVERLAP = 60

EMBED_MODEL = _p("PAINRAG_EMBED", "BAAI/bge-m3")
EMBED_DEVICE = _p("PAINRAG_DEVICE", "cpu")
