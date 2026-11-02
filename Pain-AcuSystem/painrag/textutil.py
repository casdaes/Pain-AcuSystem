# -*- coding: utf-8 -*-
from __future__ import annotations
import re
import unicodedata
import hashlib

EMPTY = {None, "", "-", "无", "未提取", "未知", "None", "nan", "－", "—"}
_WS = re.compile(r"[ \t　]+")
_SENT_END = re.compile(r"(?<=[。！？；;!?])|(?<=\n)")


def val(x) -> str:
    """单元格/字段取值：空值统一为 ''，压缩空白。"""
    if x is None:
        return ""
    s = str(x)
    if s.strip() in EMPTY:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _WS.sub(" ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def one_line(x) -> str:
    return re.sub(r"\s*\n\s*", " ", val(x))


def norm_key(s: str) -> str:
    """匹配用归一化：NFKC、小写、去空白与常见标点。"""
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"[\s，,。．.、；;：:（）()\[\]【】《》\"'“”‘’/／\-—–~～·]+", "", s)


def sentences(text: str) -> list:
    out = []
    for piece in _SENT_END.split(text):
        if piece and piece.strip():
            out.append(piece)
    return out


def split_text(text: str, max_chars: int, min_tail: int = 120, overlap: int = 0) -> list:
    """按句累积切块：每块 ≤ max_chars；末块不足 min_tail 则并入前块；overlap 为相邻块重叠的字符数（取整句）。"""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    sents = sentences(text)
    # 单句超长时硬切
    fixed = []
    for s in sents:
        while len(s) > max_chars:
            fixed.append(s[:max_chars])
            s = s[max_chars:]
        fixed.append(s)
    chunks, cur = [], ""
    for s in fixed:
        if cur and len(cur) + len(s) > max_chars:
            chunks.append(cur)
            tail = ""
            if overlap:
                # 取上一块末尾若干整句作为重叠
                for t in reversed(sentences(cur)):
                    if len(tail) + len(t) > overlap:
                        break
                    tail = t + tail
            cur = tail + s
        else:
            cur += s
    if cur:
        if chunks and len(cur.strip()) < min_tail and len(chunks[-1]) + len(cur) <= max_chars + min_tail:
            chunks[-1] += cur
        else:
            chunks.append(cur)
    return [c.strip() for c in chunks if c.strip()]


def pack_fields(fields, max_chars: int, sep: str = "\n") -> list:
    """把 [(标签, 内容), ...] 打包成若干块：尽量整字段放一块；单字段超长则按句切。返回 [块文本, ...]。"""
    items = []
    for lab, txt in fields:
        txt = val(txt)
        if not txt:
            continue
        head = f"**{lab}**：" if lab else ""
        if len(head) + len(txt) <= max_chars:
            items.append(head + txt)
        else:
            parts = split_text(txt, max_chars - len(head) - 6, min_tail=80)
            for i, p in enumerate(parts, 1):
                items.append(f"{head}（{i}/{len(parts)}）{p}")
    chunks, cur = [], ""
    for it in items:
        if cur and len(cur) + len(sep) + len(it) > max_chars:
            chunks.append(cur)
            cur = it
        else:
            cur = (cur + sep + it) if cur else it
    if cur:
        chunks.append(cur)
    return chunks


_CJK = re.compile(r"[一-鿿㐀-䶿]+")
_ALNUM = re.compile(r"[a-z0-9][a-z0-9.\-]*")


def tokenize(text: str) -> list:
    """BM25 分词：中文取字二元组（单字串取单字），英文/数字取整词。"""
    t = unicodedata.normalize("NFKC", text or "").lower()
    toks = []
    for m in _CJK.finditer(t):
        run = m.group(0)
        if len(run) == 1:
            toks.append(run)
        else:
            toks.extend(run[i:i + 2] for i in range(len(run) - 1))
    toks.extend(_ALNUM.findall(t))
    return toks


def sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def safe_name(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|\n]', "_", s)
