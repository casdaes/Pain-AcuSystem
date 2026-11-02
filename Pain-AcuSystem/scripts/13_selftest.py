#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tcm_study import needling as N
from tcm_study import stats as ss
from tcm_study.acupoints import AcupointLexicon, output_autometrics, point_metrics
from tcm_study.arms import compose_messages, compose_stage2, ensure_differential_from_card, is_degenerate, splice_stage2
from tcm_study.config import load_cfg
from tcm_study.normalize import normalize
from tcm_study.package import Package
from tcm_study.rag import RagEvidence, RagIndex


def main() -> None:
    cfg = load_cfg("configs/study.yaml")
    pkg = Package(cfg["paths"]["package_dir"])
    rep = pkg.verify()
    assert rep["ok"], rep["errors"]
    assert rep["fingerprint_sha256"].startswith("605c4388")
    lex = AcupointLexicon(pkg.files["acupoint_lib"])
    ti = pkg.test_initial()
    tot = unk = 0
    for r in ti:
        for i in lex.extract(normalize(r["reference"])[0].get("取穴", "")):
            if i["status"] in ("standard", "alias", "unknown"):
                tot += 1
                unk += i["status"] == "unknown"
    assert unk / tot < 0.02, (unk, tot)
    print(f"[OK] acupoint lexicon resolves {tot - unk}/{tot} reference points ({(tot - unk) / tot:.1%})")
    sec, text = normalize("您好！\n**【取穴】** 肾俞（双侧）、腰3-5夹脊穴、局部阿是穴\n【操作】手法：直刺1寸\n以上内容仅供参考。")
    assert "仅供参考" not in text and "您好" not in text and sec["取穴"].startswith("肾俞")
    am = output_autometrics(lex, sec, "舌红苔黄，脉弦。" + text, "腰痛3天。", "B")
    assert am["tongue_fabricated"] and am["pulse_fabricated"] and am["missing_info"] == "fabricated"
    assert point_metrics(["肾俞"], ["肾俞", "委中"])["f1"] > 0.6
    assert "【方义】" not in text and text.count("【") == 9, text
    print("[OK] normalisation, fabrication detector, point metrics")
   
    spec = N.needling_spec(lex, "肩井（双侧）、肾俞（双侧）、委中、阿是穴")
    assert spec[0].startswith("肩井：") and "禁深刺" in spec[0] and "气胸" in spec[0], spec
    assert any(s.startswith("肾俞：") and "寸" in s for s in spec), spec
    op = N.compose_operation("手法：青龙摆尾针法；配合微波治疗。", spec)
    assert op.startswith("手法：青龙摆尾") and N.SPEC_LABEL in op and N.strip_spec(op) == "手法：青龙摆尾针法；配合微波治疗。"
    assert N.needling_spec(lex, "肾俞、委中", "肾俞直刺1寸，委中") == [s for s in N.needling_spec(lex, "肾俞、委中") if s.startswith("委中")]
    pr = N.pregnancy_rules(lex, "合谷、三阴交、肾俞、足三里", "【基本信息】性别：女；年龄：28岁\n【现病史】停经8周，腰痛3天")
    assert pr["pregnant"] and pr["a_points"] == ["合谷", "三阴交"] and pr["remove"] == [] and pr["notes"][0].startswith("【妊娠禁忌】") and any("肾俞" in n for n in pr["notes"]), pr
    assert N.remove_points("合谷（双侧）、三阴交、肾俞（双侧）、足三里", ["合谷", "三阴交"]) == "肾俞（双侧）、足三里"
    pr2 = N.pregnancy_rules(lex, "合谷、足三里", "【基本信息】性别：女；年龄：30岁\n【现病史】头痛3天")
    assert not pr2["pregnant"] and pr2["childbearing"] and not pr2["remove"] and "排除妊娠" in pr2["notes"][0]
    assert N.pregnancy_rules(lex, "合谷", "【基本信息】性别：男；年龄：30岁")["notes"] == []
    assert not N.pregnancy_rules(lex, "合谷", "【基本信息】性别：女；年龄：32岁\n【现病史】停经1年，畏寒")["pregnant"], "amenorrhoea is not pregnancy"
    assert not N.pregnancy_rules(lex, "合谷", "【基本信息】性别：女；年龄：32岁\n【现病史】既往无妊娠史")["pregnant"]
    assert N.pregnancy_rules(lex, "至阴", "【基本信息】性别：女；年龄：26岁\n【现病史】妊娠30周，妇科检查为臀位")["pregnant"]
    dv = N.depth_violations(lex, "【取穴】肩井、肾俞、委中\n【操作】肩井：直刺0.8寸；肾俞：直刺0.5~1寸；委中直刺1~1.5寸。")
    assert dv["n_with_depth"] == 3 and [v["point"] for v in dv["violations"]] == ["肩井"], dv
    dv_loc = N.depth_violations(lex, "【取穴】三阴交、悬钟\n【操作】三阴交：内踝尖上3寸，直刺1~1.5寸；悬钟：外踝尖上3寸，直刺0.5~0.8寸。")
    assert dv_loc["n_with_depth"] == 2 and not dv_loc["violations"], dv_loc
    op18, r18 = N.standardize_operation(lex, "肩髃（右）、阿是穴（右）、内庭（右）",
        "肩髃（右）：直刺0.8~1.5寸；阿是穴（右）：围刺；内庭（右）：直刺0.5~1.0寸。",
        "【基本信息】性别：男；年龄：50岁\n【主诉】右肩疼痛3月\n【查体】右肩前外侧压痛、外展受限")
    assert op18.startswith("1. 手法与疗程：") and "\n2. 逐穴操作：" in op18 and "阿是穴（右侧）：定位在右肩部" in op18, op18
    assert "内庭（右侧）" in op18 and "0.5~0.8寸" in op18 and any(x["point"] == "内庭" for x in r18["depth_corrections"]), r18
    assert not N.depth_violations(lex, "【取穴】肩髃、内庭\n【操作】" + op18)["violations"], op18
    am2 = output_autometrics(lex, normalize("【取穴】肩井、肾俞\n【操作】肩井：直刺0.8寸；肾俞：直刺1寸。")[0], "【取穴】肩井、肾俞\n【操作】肩井：直刺0.8寸；肾俞：直刺1寸。", "腰痛", "B")
    assert am2["n_depth_over_limit"] == 1 and am2["depth_specified_rate"] == 1.0, am2
    print("[OK] needling spec, pregnancy rules, depth-over-limit metric")
    # v5：循环检测补非连续重复；两步拼接
    loop = "【操作】" + "".join(f"中药贴敷环跳（双侧）、委中（双侧）、悬空灸腰阳关、点穴疗法{i}、" for i in range(8))
    assert is_degenerate(loop) == "repetition_loop", "non-contiguous loop must be caught"
    ok_text = "【诊疗思路】患者女，55岁，腰痛2天。" * 1 + "【西医诊断】腰痛（待查）\n【中医诊断】腰痛\n【证型】无\n【病机】经络不通\n【治法】通络\n【取穴】肾俞（双侧）、大肠俞（双侧）、委中（双侧）、环跳（双侧）\n【操作】手法：平补平泻\n【注意事项】1. 完善腰椎X线。\n2. 严格无菌操作。"
    assert is_degenerate(ok_text) is None, "laterality repeats must not be flagged"
    s1 = "【诊疗思路】a\n【西医诊断】腰痛\n【中医诊断】腰痛\n【证型】无\n【病机】x\n【治法】y\n【取穴】肾俞、委中\n【操作】平补平泻\n【注意事项】1. 完善检查。"
    s2 = "【诊疗思路】b\n【西医诊断】腰椎间盘突出\n【中医诊断】腰痛\n【证型】无\n【病机】x\n【治法】y\n【取穴】肾俞、委中、环跳\n【操作】平补平泻；肾俞：直刺0.5~1寸；委中：直刺1~1.5寸\n【注意事项】1. 完善检查。\n2. 委中不宜深刺。\n【方义】zzz"
    final, info = splice_stage2(s1, s2)
    assert "【西医诊断】腰痛\n" in final and "【取穴】肾俞、委中\n" in final and "肾俞：直刺0.5~1寸" in final and "委中不宜深刺" in final and "方义" not in final, final
    assert info["stage2_fields_used"] == ["诊疗思路", "操作", "注意事项"] and info["stage2_points_changed"] is True
    s1d = s1.replace("【注意事项】1. 完善检查。", "【注意事项】需鉴别：腰椎间盘突出（放射痛）；腰椎管狭窄（间歇性跛行）。\n1. 完善检查。")
    s2d = s2.replace("【注意事项】1. 完善检查。\n2. 委中不宜深刺。", "【注意事项】1. 必要时完善腰椎MRI。\n2. 委中不宜深刺。")
    merged, mi = splice_stage2(s1d, s2d)
    assert "腰椎间盘突出" in merged and mi["stage1_differential_restored"], (merged, mi)
    guarded, gi = ensure_differential_from_card(s2, {"dd": ["腰椎间盘突出（放射痛）", "腰椎管狭窄（跛行）"], "red": []}, 2)
    assert gi["applied"] and "腰椎间盘突出" in guarded and "腰椎管狭窄" in guarded, (guarded, gi)
    print("[OK] loop detector, stage-2 merge, differential guard")
  
    from tcm_study import schema as S
    from tcm_study.safety_layer import apply as rule_layer, safety_check_card
    male, old_f, young_f = "【基本信息】性别：男；年龄：45岁", "【基本信息】性别：女；年龄：70岁", "【基本信息】性别：女；年龄：30岁"
    t = "【操作】三阴交：直刺1~1.5寸，孕妇禁针；\n【注意事项】1. 需除外异位妊娠。\n2. 昆仑属妊娠禁用穴；太冲可诱发宫缩。\n3. 育龄女性，合谷为妊娠禁用穴，针前需排除妊娠，查尿HCG。\n4. 严格无菌操作。"
    o, rm = S.sex_consistent(t, old_f)
    assert "孕妇禁针" in o and "异位妊娠" in o and "昆仑" not in o and "育龄" not in o and "HCG" not in o and "2. 严格无菌操作" in o, o
    assert S.sex_consistent(t, young_f)[0] == t
    o, rm = N.correct_pregnancy_lists(lex, "1. 内关、中脘等均为妊娠禁忌穴。\n2. 合谷、天枢为妊娠禁用穴。")
    assert "内关" not in o and "中脘" not in o and "天枢" not in o and "合谷为妊娠禁用穴" in o, o
    card = safety_check_card(lex, male + "\n【主诉】腹痛", ["合谷"])
    assert "不适用" in card and "【急危重症警示】只写" in card
    assert S.strip_null_warning("【注意事项】【急危重症警示】无。\n1. 严格无菌。")[0] == "【注意事项】1. 严格无菌。"
    assert S.has_real_warning("【注意事项】【急危重症警示】血压低。") and not S.has_real_warning("【注意事项】【急危重症警示】无\n1. a")
    assert "急危重症警示" not in normalize("【取穴】肾俞\n【注意事项】【急危重症警示】无。\n1. 严格无菌。")[1]
    card = safety_check_card(lex, young_f + "\n【主诉】腰痛", ["合谷", "关元"])
    assert "合谷" in card and "排除妊娠" in card and "关元" not in card.split("妊娠：")[1], card
    card = safety_check_card(lex, "【基本信息】性别：女；年龄：57岁\n【主诉】右下腹痛3天\n【查体】麦氏点压痛，反跳痛", [])
    assert "已记录的急危征象" in card and "查尿HCG" not in card and "原文：；" not in card, card
    out, rep = rule_layer(lex, "【取穴】肾俞\n【操作】肾俞：直刺0.5寸\n【注意事项】1. 严格无菌操作。", "【基本信息】性别：女；年龄：57岁\n【主诉】右下腹痛3天\n【查体】麦氏点压痛，反跳痛", strict_redflags=True)
    assert out.split("【注意事项】")[1].lstrip().startswith(S.EMERGENCY_TAG), out
    old_depth = "【取穴】内庭\n【操作】内庭：直刺0.5~1.0寸\n【注意事项】（规则核查）内庭所写深度1.0超过穴位库上限0.8，应按上限执行。"
    fixed18, frep18 = rule_layer(lex, old_depth, "【基本信息】性别：男；年龄：40岁\n【主诉】足部疼痛", strict_redflags=True)
    assert "规则核查" not in fixed18 and "0.5~0.8寸" in fixed18 and frep18.get("operation"), fixed18
    assert "1. 禁忌与特殊人群：" in fixed18 and "2. 建议完善检查：" in fixed18 and "3. 操作前后注意事项：" in fixed18 and "4. 患者教育：" in fixed18, fixed18

    from tcm_study.doctor_view import standardize_doctor_view
    tp_reason = (
        "患者主诉左踝扭伤1日。查体见舌质暗，苔薄黄，脉涩；"
        "舌脉未记录，暂不强行辨证；现阶段病机为局部经络气血不畅。"
        "需鉴别：踝关节炎（慢性僵硬）；痛风（红肿热痛）；下胫腓联合损伤（外旋试验）；"
        "腓骨肌腱滑脱（肌腱滑移）；肿瘤（持续痛；MRI）。"
        "需优先排除：隐匿性骨折（持续痛；MRI）；Ⅲ度扭伤（明显不稳；MRI）。"
    )
    dup_notes = (
        "4. 患者教育：针后短暂休息并保持针孔清洁；根据病情注意足部保暖，避免诱发或加重症状的过度活动或负荷；"
        "按疗程复诊，若症状明显加重及时就医。针后短暂休息并保持针孔清洁；"
        "根据病情注意保暖，避免诱发或加重症状的过度活动或负荷；按疗程复诊，若症状明显加重及时就医。"
    )
    tp_input = "【基本信息】性别：女；年龄：23岁\n【主诉】左踝扭伤1日\n【舌脉】舌质暗，苔薄黄，脉涩"
    rr, nn, aa = standardize_doctor_view(lex, tp_reason, dup_notes, tp_input, "丘墟、昆仑", "踝关节扭伤")
    assert "舌脉未记录" not in rr and aa["tongue_pulse_false_missing_removed"], (rr, aa)
    assert aa["differential_cap"]["after_items"] == 4 and aa["differential_cap"]["before_items"] == 7, aa
    assert rr.count("需鉴别：") == 1 and rr.count("需优先排除：") == 1 and "腓骨肌腱滑脱" not in rr and "Ⅲ度扭伤" not in rr, rr
    assert nn.count("针后短暂休息并保持针孔清洁") == 1 and nn.count("按疗程复诊") == 1 and "足部保暖" in nn, nn
    no_tp_reason = "患者腰痛3天；舌脉未记录，暂不强行辨证；需鉴别：腰肌劳损。"
    rr2, _, aa2 = standardize_doctor_view(lex, no_tp_reason, "1. 禁忌与特殊人群：注意安全。", "【基本信息】性别：男；年龄：40岁\n【主诉】腰痛3天\n【舌脉】未记录")
    assert "舌脉未记录" in rr2 and not aa2["tongue_pulse_false_missing_removed"], rr2

    false_out, _ = rule_layer(lex, "【取穴】肾俞\n【操作】肾俞：直刺0.5寸\n【注意事项】【急危重症警示】疑似严重病变，立即急诊。\n1. 复查。", "【基本信息】性别：男；年龄：30岁\n【主诉】轻度腰痛，可正常活动", strict_redflags=True)
    assert not S.has_real_warning(false_out), false_out
    print("[OK] sex consistency (incl. pregnancy clauses in notes), pregnancy list check, case safety card, rule layer")
    from tcm_study.differential import DxKB
    dkb = DxKB(cfg["outputs"]["rag_index"])
    dl, di = dkb.note("肩关节周围炎", "【基本信息】性别：男；年龄：55岁\n【主诉】右肩痛3月，夜间加重，上举受限")
    assert dl.startswith("需鉴别：") and di["concept"] and len(di.get("dd") or []) >= 2, (dl, di)
    assert all("（" in x or len(x) >= 2 for x in di.get("dd") or []), di
    assert dkb.note("（待查）", "x")[0] == ""
    assert "妊娠" not in dkb.note("腹痛", "【基本信息】性别：男；年龄：45岁")[0]
    print(f"[OK] concrete differential diseases from KB: {dl[:80]}…")
   
    dl2, di2 = dkb.note("膝关节痛（待查）", "【基本信息】性别：女；年龄：62岁\n【主诉】双膝疼痛2年")
    assert di2.get("proxy") and dl2.startswith("需鉴别：膝骨关节炎"), (dl2, di2)
    assert dkb.note("踝扭伤", "扭伤右踝")[1].get("proxy") is False
    assert "关键区分点：" in dl2, dl2
    ann, annlog = dkb.annotate_findings("足跟痛（待查）", "【主诉】左足跟疼痛二月余\n【查体】局部无压痛", ["局部无压痛"])
    assert annlog and "跖筋膜炎" in ann[0] and "不一致" in ann[0] and "压痛" in ann[0], (ann, annlog)
   
    pts19, lat19 = N.standardize_point_laterality(lex, "肩髃、肩髎、合谷、关元、腰阳关", "【主诉】左肩疼痛3个月")
    assert pts19 == "肩髃（左侧）、肩髎（左侧）、合谷（左侧）、关元、腰阳关", (pts19, lat19)
    pts19b, _ = N.standardize_point_laterality(lex, "阿是穴、太溪、昆仑、水泉、中极", "【主诉】右足跟疼痛二月余")
    assert pts19b == "阿是穴（右侧）、太溪（右侧）、昆仑（右侧）、水泉（右侧）、中极", pts19b
    free19, frep19 = N.standardize_point_laterality_free_text(
        lex, "前方加足三里（双侧）、关元（双侧）、腰眼（双侧）、承浆（双侧）", "【主诉】腰痛"
    )
    assert "关元（" not in free19 and "承浆（" not in free19 and "腰眼（双侧）" in free19 and frep19["applied"], free19
    sp = N.needling_spec(lex, "足三里、委中", with_location=True)
    assert sp[0].startswith("足三里（定位：") and "ST35" not in sp[0] and "腘横纹" in sp[1], sp
    dv2 = N.depth_violations(lex, "【取穴】足三里、犊鼻\n【操作】" + "；".join(N.needling_spec(lex, "足三里、犊鼻", with_location=True)))
    assert not dv2["violations"] and dv2["n_with_depth"] == 2, dv2
    import importlib.util as _u0
    _s0 = _u0.spec_from_file_location("e01b", "scripts/01b_enrich_sft_targets.py"); e01b = _u0.module_from_spec(_s0); _s0.loader.exec_module(e01b)
    assert e01b.positive_findings("体温：36.5℃，血压：120/80mmHg；神清语利，腰部压痛，左侧直腿抬高试验阳性，皮肤无破损，舌淡苔白") == ["腰部压痛", "左侧直腿抬高试验阳性"]
    e01b.DIFF_KB, e01b.NULL_WARNING_LINE, e01b.WITH_LOCATION = dkb, True, True
    u = "【基本信息】性别：男；年龄：45岁\n【主诉】腰痛3天\n【现病史】搬重物后腰痛\n【查体】腰部压痛\n【辅助检查】未提供\n【舌脉】未记录"
    sec16, _ = e01b.rule_rewrite(u, "【西医诊断】急性腰扭伤\n【中医诊断】腰痛\n【取穴】肾俞、委中", lex)
    assert not re.search(r"严格无菌|补充舌象|综合考虑|取穴以病位", sec16["诊疗思路"] + sec16["注意事项"]), sec16
    assert "腰部压痛" in sec16["诊疗思路"] and "支持诊断的依据为" in sec16["诊疗思路"] and "证据不足之处" in sec16["诊疗思路"] and "需鉴别：" in sec16["诊疗思路"] and "关键区分点：" in sec16["诊疗思路"] and "定位" in sec16["操作"] and sec16["注意事项"].startswith("1. 禁忌与特殊人群：") and "2. 建议完善检查：" in sec16["注意事项"], sec16
    try:
        import importlib.util as _u
        _spec = _u.spec_from_file_location("tr", "scripts/02_train_sft.py"); tr = _u.module_from_spec(_spec); _spec.loader.exec_module(tr)
    except Exception as e:
        print(f"[SKIP] loss weighting test（{type(e).__name__}: 训练依赖未安装）")
    else:
        class _Tok:
            SPEC = ["<|im_start|>", "<|im_end|>", "】【"]
            def apply_chat_template(self, m, tokenize=False, add_generation_prompt=False, **kw):
                s_ = "".join(f"<|im_start|>{x['role']}\n{x['content']}<|im_end|>\n" for x in m)
                return s_ + ("<|im_start|>assistant\n" if add_generation_prompt else "")
            def __call__(self, text, truncation=False, add_special_tokens=False, return_offsets_mapping=False):
                ids, offs, i = [], [], 0
                while i < len(text):
                    for k, sp in enumerate(self.SPEC):
                        if text.startswith(sp, i):
                            ids.append(10**6 + k); offs.append((i, i + len(sp))); i += len(sp); break
                    else:
                        ids.append(ord(text[i])); offs.append((i, i + 1)); i += 1
                return {"input_ids": ids, "offset_mapping": offs} if return_offsets_mapping else {"input_ids": ids}
        ans = "【诊疗思路】腰痛\n【注意事项】【急危重症警示】血压低。\n1. 严格无菌操作。"
        msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "病例"}, {"role": "assistant", "content": ans}]
        pr = tr.AssistantOnlyProcessor(_Tok(), 4096, {}, [], [], ["【急危重症警示】"], 5.0)
        ob = pr({"messages": [msgs]})
        w = dict(zip(ob["input_ids"][0], ob["loss_weight"][0]))
        assert w[10**6 + 2] == 5.0 and w[ord("严")] == 1.0 and w[ord("危")] == 5.0, "decision token 】【 must be up-weighted"
        ans2 = "【诊疗思路】腰痛\n【注意事项】1. 严格无菌操作。"            # 不写警示的样本：标题收尾的"】"同样加权（决定点双向）
        ob2 = pr({"messages": [[msgs[0], msgs[1], {"role": "assistant", "content": ans2}]]})
        ws2 = [x for t, x in zip(ob2["input_ids"][0], ob2["loss_weight"][0]) if t == ord("】")]
        assert ws2 and ws2[-1] == 5.0 and sum(1 for x in ob2["loss_weight"][0] if x != 1.0) == 1, ob2["loss_weight"][0]
        print("[OK] loss weighting: decision token after 【注意事项 weighted in both directions (】【 / 】)")

    idx = RagIndex(cfg["paths"]["painrag_kb"], "cpu")
    assert set(idx.docs) >= {"west", "tcm", "acupoint", "concept"}, set(idx.docs)
    assert len(idx.docs["acupoint"]) == 419 and len(idx.docs["west"]) > 8000 and len(idx.docs["tcm"]) > 10000
    assert not any("case_id" in d or "【主诉】" in d.get("text", "") for cat in ("west", "tcm") for d in idx.docs[cat][:2000]), "knowledge base must not contain clinical cases"
    ev = RagEvidence(idx, cfg["rag"])
    row = next(r for r in pkg.train() if r["meta"]["task"] == "initial")
    q = RagEvidence.query_from_messages(row["messages"][:-1])
    text, log = ev.build(q)
    assert log["evidence_sha256"] and log["evidence_chars"] <= int(cfg["rag"]["max_chars"])
    assert ("【中医证据" in text or "【西医证据" in text or "【针灸方案参考" in text), text[:200]
    text2, log2 = ev.build(q)
    assert log2["evidence_sha256"] == log["evidence_sha256"], "evidence must be deterministic"
    m = compose_messages(row["messages"][:-1], text)
    assert "不得编造病例中没有的信息" in m[-1]["content"] and "病种层面的风险知识" in m[-1]["content"]
    assert compose_messages(row["messages"][:-1], None)[-1]["content"] == row["messages"][-2 if row["messages"][-1]["role"] == "assistant" else -1]["content"]
    import re as _re
    assert not _re.search(r"[（(]D\d{4}|ICD-11|匹配度|\[BK\d|\[GL\d", text), "evidence must not carry KB identifiers (unblinding)"
    assert "【穴位参考" not in text, "step 1 must not contain the acupoint block (domains = west/tcm)"
    cards, clog = ev.cards(["肩井", "肾俞（双侧）", "不存在穴"], ["青龙摆尾针法"])
    assert "【穴位安全卡】" in cards and clog["points_carded"] == ["肩井", "肾俞"] and "【手法卡】" in cards and clog["manipulations_carded"] == ["青龙摆尾"], clog
    assert ev.cards(["肩井", "肾俞（双侧）", "不存在穴"], ["青龙摆尾针法"])[1]["cards_sha256"] == clog["cards_sha256"]
    assert ev.manip_names_in("手法：烧山火；配合温针灸") == ["烧山火", "温针灸"]
    from tcm_study.safety_layer import guideline_safety_card
    dcard, dlog = ev.differential_card("慢性胆囊炎", "【主诉】右上腹痛伴恶心呕吐\n【辅助检查】胆囊超声示胆囊壁增厚")
    assert "【鉴别诊断卡】" in dcard and "急性胆囊炎" in dcard and "胸壁结核" not in dcard and dlog.get("concept_name") == "慢性胆囊炎", (dcard, dlog)
    gcard = guideline_safety_card(text)
    
    if "禁忌与护理" in text or "针灸适应证与安全边界" in text or "指南通用注意" in text:
        assert "【指南安全提示卡】" in gcard and "不代表患者已经出现" in gcard
    extra_cards = cards + (("\n\n" + dcard) if dcard else "") + (("\n\n" + gcard) if gcard else "")
    m2 = compose_stage2(row["messages"][:-1], text, "【取穴】肩井", extra_cards)
    assert "【第一步输出】" in m2[-1]["content"] and "【穴位安全卡】" in m2[-1]["content"] and m2[-1]["content"].endswith("不得新增栏目。")
    print(f"[OK] RAG evidence ({'dense+bm25' if idx.dense else 'bm25 only'}): {log['evidence_chars']} chars, concepts: {[c['name'] for c in (log['concepts'] or [])]}; cards: {clog['cards_chars']} chars")
    assert abs(ss.chi2_sf(3.841459, 1) - 0.05) < 1e-4 and abs(ss.chi2_sf(9.487729, 4) - 0.05) < 1e-4
    fr = ss.friedman(np.array([[1, 2, 3], [1, 3, 2], [1, 2, 3], [2, 1, 3], [1, 2, 3], [1, 2, 3]]))          # textbook value: chi2 = 8.333, df = 2
    assert abs(fr["statistic"] - 8.3333) < 1e-3 and abs(fr["p"] - 0.01550) < 1e-4, fr
    w = ss.wilcoxon_signed_rank(np.array([1.5, 2.5, -0.5, 3.0, 1.0, 2.0, -1.2, 0.8, 2.2, 1.7]))
    assert w["n_nonzero"] == 10 and abs(w["w_plus"] - 50.0) < 1e-9 and 0.01 < w["p"] < 0.03, w
    cq = ss.cochran_q(np.array([[1, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 0], [0, 0, 0], [1, 1, 0], [1, 0, 0]]))
    assert abs(cq["statistic"] - 9.0) < 1e-9 and cq["df"] == 2, cq
    pc = ss.paired_score_contrast(np.array([7, 8, 6, 9, 7.5, 8]), np.array([6, 7, 6, 7, 7, 6.5]), boot=2000)
    assert pc["mean_diff"] > 0 and pc["ci95"][0] <= pc["mean_diff"] <= pc["ci95"][1]
    assert abs(ss.holm({"a": 0.01, "b": 0.04, "c": 0.03})["b"] - 0.06) < 1e-12 and ss.mcnemar_exact(5, 0) == 0.0625
    assert abs(ss.mdd_paired(55, 1.5, 0.05, 0.8) - 0.5667) < 1e-3
    assert ss.icc2k(np.array([[5, 5], [7, 7], [9, 9], [3, 3]])) > 0.99
    print("[OK] statistics")
    print("[OK] selftest passed")


if __name__ == "__main__":
    main()
