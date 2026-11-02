from __future__ import annotations

import re


_KP_TAG = "（关键区分点："


def _balanced_close(text: str, open_pos: int) -> int:
    if (
        open_pos < 0
        or open_pos >= len(text)
        or text[open_pos] not in "（("
    ):
        return -1

    depth = 0

    for i in range(open_pos, len(text)):
        ch = text[i]

        if ch in "（(":
            depth += 1

        elif ch in "）)":
            depth -= 1

            if depth == 0:
                return i

    return -1


def _norm_keypoint_body(text: str) -> str:
    return re.sub(
        r"[\s，,。；;：:、（）()]",
        "",
        str(text or ""),
    )


def merge_adjacent_keypoint_wrappers(
    text: str,
) -> tuple[str, list[dict]]:
    """合并连续的关键区分点包装。

    疾病（关键区分点：A）（关键区分点：A）
      -> 疾病（关键区分点：A）

    疾病（关键区分点：A）（关键区分点：B）
      -> 疾病（关键区分点：A；B）
    """
    out = str(text or "")
    changes = []

    while True:
        changed = False
        pos = 0

        while True:
            open1 = out.find(_KP_TAG, pos)

            if open1 == -1:
                break

            close1 = _balanced_close(out, open1)

            if close1 == -1:
                break

            j = close1 + 1

            while j < len(out) and out[j].isspace():
                j += 1

            if not out.startswith(_KP_TAG, j):
                pos = close1 + 1
                continue

            open2 = j
            close2 = _balanced_close(out, open2)

            if close2 == -1:
                break

            body1 = out[
                open1 + len(_KP_TAG):
                close1
            ].strip()

            body2 = out[
                open2 + len(_KP_TAG):
                close2
            ].strip()

            if not body1 or not body2:
                pos = close2 + 1
                continue

            if (
                _norm_keypoint_body(body1)
                == _norm_keypoint_body(body2)
            ):
                merged = body1

            else:
                merged = (
                    body1.rstrip("；; ")
                    + "；"
                    + body2.lstrip("；; ")
                )

            old = out[open1:close2 + 1]
            new = f"（关键区分点：{merged}）"

            out = (
                out[:open1]
                + new
                + out[close2 + 1:]
            )

            changes.append({
                "old": old,
                "new": new,
            })

            changed = True
            break

        if not changed:
            break

    return out, changes


def _split_top_level_dunhao(text: str) -> list[str]:
    """只在括号层级0按顿号拆分。"""
    out = []
    buf = []
    depth = 0

    for ch in str(text or ""):

        if ch in "（(":
            depth += 1

        elif ch in "）)" and depth > 0:
            depth -= 1

        if ch == "、" and depth == 0:

            item = "".join(buf).strip()

            if item:
                out.append(item)

            buf = []

        else:
            buf.append(ch)

    item = "".join(buf).strip()

    if item:
        out.append(item)

    return out


def _clean_exam_fragment(
    text: str,
    *,
    remove_glucose: bool,
    remove_arterial_us: bool,
) -> tuple[str, dict]:
    """只清理检查列表中的已完成同义项目。"""

    items = _split_top_level_dunhao(text)

    report = {
        "blood_glucose_removed": 0,
        "lower_limb_arterial_ultrasound_removed": 0,
    }

    kept = []

    for item in items:
        x = item.strip()

        # ----------------------------------------------------
        # 血糖：只删除整个列表项就是“血糖/空腹血糖/餐后血糖”
        # ----------------------------------------------------
        if (
            remove_glucose
            and re.fullmatch(
                r"(?:空腹|餐后)?血糖",
                x,
                re.I,
            )
        ):
            report["blood_glucose_removed"] += 1
            continue

        # ----------------------------------------------------
        # 已有动脉超声：
        #
        # 下肢动脉彩超/CTA/MRA
        # ->
        # CTA/MRA
        #
        # 绝不删除 CTA/MRA。
        # ----------------------------------------------------
        if remove_arterial_us:

            x2, n = re.subn(
                r"(?:下肢)?(?:动脉|血管)"
                r"(?:彩超|超声)\s*/\s*",
                "",
                x,
                flags=re.I,
            )

            if n:
                report[
                    "lower_limb_arterial_ultrasound_removed"
                ] += n

                x = x2.strip()

            elif re.fullmatch(
                r"(?:下肢)?(?:动脉|血管)"
                r"(?:彩超|超声)",
                x,
                re.I,
            ):
                report[
                    "lower_limb_arterial_ultrasound_removed"
                ] += 1

                continue

        kept.append(x)

    return "、".join(kept), report


def cleanup_recorded_exam_redundancy(
    reason: str,
    notes: str,
    input_text: str,
) -> tuple[str, str, dict]:

    inp = str(input_text or "")
    reason_out = str(reason or "")
    notes_out = str(notes or "")

    has_glucose_result = bool(
        re.search(
            r"(?:空腹|餐后)?血糖\s*[：:]\s*"
            r"\d+(?:\.\d+)?",
            inp,
            re.I,
        )
    )

    has_lower_limb_arterial_us = bool(
        re.search(
            r"(?:下肢)?(?:动脉|血管).{0,8}"
            r"(?:超声|彩超)",
            inp,
            re.I | re.S,
        )
        or (
            re.search(
                r"血管超声\s*[：:]",
                inp,
                re.I,
            )
            and re.search(
                r"下肢动脉|动脉闭塞",
                inp,
                re.I,
            )
        )
    )

    report = {
        "blood_glucose_removed": 0,
        "lower_limb_arterial_ultrasound_removed": 0,
    }

    def clean_fragment(m):
        old = m.group(0)

        new, rep = _clean_exam_fragment(
            old,
            remove_glucose=has_glucose_result,
            remove_arterial_us=has_lower_limb_arterial_us,
        )

        report["blood_glucose_removed"] += (
            rep["blood_glucose_removed"]
        )

        report[
            "lower_limb_arterial_ultrasound_removed"
        ] += rep[
            "lower_limb_arterial_ultrasound_removed"
        ]

        return new

  
    reason_out = re.sub(
        r"[^。！？\n]*(?:尚缺少|缺少)"
        r"[^。！？\n]*[。！？]?",
        clean_fragment,
        reason_out,
    )

   
    notes_out = re.sub(
        r"2[\.．、]\s*建议完善检查[：:]"
        r".*?"
        r"(?=3[\.．、]\s*操作前后注意事项|\Z)",
        clean_fragment,
        notes_out,
        flags=re.S,
    )

    return reason_out, notes_out, report


def final_content_cleanup(
    reason: str,
    notes: str,
    input_text: str,
) -> tuple[str, str, dict]:

    reason0 = str(reason or "")
    notes0 = str(notes or "")

    reason1, kp_changes = (
        merge_adjacent_keypoint_wrappers(reason0)
    )

    reason2, notes2, exam_report = (
        cleanup_recorded_exam_redundancy(
            reason1,
            notes0,
            input_text,
        )
    )

    report = {
        "applied": bool(
            kp_changes
            or reason2 != reason0
            or notes2 != notes0
        ),
        "keypoint_wrapper_merges": kp_changes,
        "exam_redundancy": exam_report,
    }

    return reason2, notes2, report
