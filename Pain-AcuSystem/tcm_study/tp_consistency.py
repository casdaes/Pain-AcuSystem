import re

from tcm_study import schema as S


_FALSE_COMPLETE_TP = [
    # 舌象、脉象未记录 / 舌脉未记录 / 未提供 / 缺失 / 不详
    re.compile(
        r"(?:舌象[、，,及和与]?脉象|舌脉)"
        r"(?:资料|信息|情况)?"
        r"(?:未记录|未提供|缺失|不详|资料不足)"
        r"[，,]?"
    ),

    # 尚未记录舌象、脉象
    re.compile(
        r"(?:尚未|未)"
        r"(?:记录|提供)"
        r"(?:详细)?"
        r"(?:舌象[、，,及和与]?脉象|舌脉)"
        r"[，,；;。]?"
    ),

    # 建议/需/应补充舌脉……
    re.compile(
        r"(?:建议|需|应|可)"
        r"(?:进一步)?"
        r"补充"
        r"(?:舌脉(?:资料|信息)?|舌象[、，,及和与]?脉象)"
        r"(?:后)?"
        r"(?:再|以便)?"
        r"(?:四诊合参后?)?"
        r"(?:明确辨证(?:分型)?|再定证型|辨证)?"
        r"[，,；;。]?"
    ),
]


def clean_false_tp_missing(text: str, input_text: str):

    text = str(text or "")
    source = str(input_text or "")

    has_tongue = bool(S.TONGUE_RE.search(source))
    has_pulse = bool(S.PULSE_RE.search(source))

    report = {
        "applied": False,
        "has_tongue": has_tongue,
        "has_pulse": has_pulse,
        "removed": [],
        "replaced": [],
    }

    if not (has_tongue or has_pulse):
        return text, report

    out = text

    combined_missing = re.compile(
        r"(?:舌象[、，,及和与]?脉象|舌脉|舌、脉)"
        r"(?:资料|信息|情况)?"
        r"(?:均)?"
        r"(?:未记录|未提供|缺失|不详|资料不足)"
    )

    combined_supplement = re.compile(
        r"((?:建议|需|应|可)(?:进一步)?补充)"
        r"(?:舌脉(?:资料|信息)?|舌象[、，,及和与]?脉象)"
    )

    # ---------------------------------------------------------
    # Complete tongue + pulse: both blanket claims are false.
    # ---------------------------------------------------------
    if has_tongue and has_pulse:
        for pat in _FALSE_COMPLETE_TP:
            hits = [m.group(0) for m in pat.finditer(out)]
            if hits:
                report["removed"].extend(hits)
                out = pat.sub("", out)

    # ---------------------------------------------------------
    # Tongue exists, pulse missing.
    # ---------------------------------------------------------
    elif has_tongue and not has_pulse:
        for m in list(combined_missing.finditer(out)):
            old = m.group(0)
            report["replaced"].append({
                "old": old,
                "new": "脉象未记录",
            })
        out = combined_missing.sub("脉象未记录", out)

        def pulse_repl(m):
            old = m.group(0)
            new = m.group(1) + "脉象"
            report["replaced"].append({
                "old": old,
                "new": new,
            })
            return new

        out = combined_supplement.sub(pulse_repl, out)

    # ---------------------------------------------------------
    # Pulse exists, tongue missing.
    # ---------------------------------------------------------
    elif has_pulse and not has_tongue:
        for m in list(combined_missing.finditer(out)):
            old = m.group(0)
            report["replaced"].append({
                "old": old,
                "new": "舌象未记录",
            })
        out = combined_missing.sub("舌象未记录", out)

        def tongue_repl(m):
            old = m.group(0)
            new = m.group(1) + "舌象"
            report["replaced"].append({
                "old": old,
                "new": new,
            })
            return new

        out = combined_supplement.sub(tongue_repl, out)

    if not report["removed"] and not report["replaced"]:
        return text, report

    out = re.sub(r"\s+([，,；;。])", r"\1", out)
    out = re.sub(r"[，,]\s*[；;]", "；", out)
    out = re.sub(r"[；;]\s*[；;]", "；", out)
    out = re.sub(r"[；;]\s*。", "。", out)
    out = re.sub(r"。\s*。", "。", out)
    out = re.sub(r"，\s*，", "，", out)
    out = re.sub(r"\n[ \t]+", "\n", out)

    report["applied"] = True
    return out, report

