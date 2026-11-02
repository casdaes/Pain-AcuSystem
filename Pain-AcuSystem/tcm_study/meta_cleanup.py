import re


_META_MARKERS = re.compile(
    r"若严格遵循"
    r"|系统强制"
    r"|未记录写未记录"
    r"|此处应修正"
    r"|根据指令"
    r"|本案已记录"
    r"|直接写证型"
)

_SECTION_RE = re.compile(
    r"(【证型】)(.*?)(?=\n?【[^】]+】|\Z)",
    re.S,
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).strip("。；;：:")


def _clean_syndrome_body(body: str):
    original = str(body or "")
    out = original
    removed = []

    # 1) 删除明确包含系统/规则元话语的“（注：……）”
    note_pat = re.compile(r"（注：.*?）", re.S)

    def drop_note(m):
        txt = m.group(0)
        if _META_MARKERS.search(txt):
            removed.append(txt)
            return ""
        return txt

    out = note_pat.sub(drop_note, out)

    # 2) 删除明确的“修正/根据指令”元话语行
    lines = []
    for line in out.splitlines():
        stripped = line.strip()
        probe = stripped.strip("*").strip()

        if (
            re.match(r"^(?:修正|更正)\s*[:：]", probe)
            and _META_MARKERS.search(probe)
        ):
            removed.append(stripped)
            continue

        if _META_MARKERS.search(probe):
            removed.append(stripped)
            continue

        lines.append(line)

    out = "\n".join(lines)

    # 3) 删除与首个证型完全相同的重复“证型：XXX”
    meaningful = [x.strip() for x in out.splitlines() if x.strip()]

    if meaningful:
        first = meaningful[0]
        first_norm = _norm(first)

        kept = [first]

        for line in meaningful[1:]:
            m = re.fullmatch(r"证型\s*[:：]\s*(.+)", line)

            if m and _norm(m.group(1)) == first_norm:
                removed.append(line)
                continue

            if _norm(line) == first_norm:
                removed.append(line)
                continue

            kept.append(line)

        out = "\n".join(kept)

    out = re.sub(r"\n{3,}", "\n\n", out).strip()

    return out, removed


def clean_meta_commentary(text: str):
 
    text = str(text or "")
    removed_all = []

    def repl(m):
        header = m.group(1)
        body = m.group(2)

        cleaned, removed = _clean_syndrome_body(body)

        # No actual meta-commentary removal -> preserve original bytes exactly.
        # This prevents harmless whitespace/newline normalization from making
        # every record appear changed.
        if not removed:
            return m.group(0)

        removed_all.extend(removed)

        # Preserve the original trailing whitespace before the next section.
        trailing = body[len(body.rstrip()):]

        return header + cleaned + trailing

    out = _SECTION_RE.sub(repl, text)

    audit = {
        "applied": bool(removed_all),
        "removed": removed_all,
    }

    return out, audit
