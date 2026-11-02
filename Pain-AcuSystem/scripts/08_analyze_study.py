#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tcm_study import analysis
from tcm_study.analysis import ARM_LABEL
from tcm_study.config import load_cfg
from tcm_study.state import StudyState

EP_CN = {"overall": "综合分（三维度均分）", "safety": "安全性", "accuracy": "准确性", "usability": "可用性"}


def _p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _score_block(fam: dict, title: str) -> list[str]:
    o = fam["omnibus"]
    L = ["", f"### {title}（n={fam['n_cases']} 例）", "",
         "各臂均值（标准差）：" + "；".join(f"{a} {fam['means'][a]:.2f}（{fam['sds'][a]:.2f}）" for a in fam["means"]),
         f"Friedman 检验：χ²({o['df']}) = {o['statistic']:.2f}，p = {_p(o['p'])}，Kendall W = {o['kendall_w']:.3f}", "",
         "| 比较（前 − 后） | 均差 | 95%CI | Wilcoxon p | Holm 校正 p | dz | 秩二列 r | 结论 |", "|---|---|---|---|---|---|---|---|"]
    for k, r in fam["pairwise"].items():
        x, y = k.split("_minus_")
        L.append(f"| {x} − {y} | {r['mean_diff']:+.2f} | {r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f} | {_p(r['p'])} | {_p(r['p_holm'])} | {r['cohen_dz']:+.2f} | {r['rank_biserial']:+.2f} | {'差异有统计学意义' if r['significant'] else '未见差异'} |")
    return L


def _binary_block(fam: dict, title: str) -> list[str]:
    o = fam["omnibus"]
    L = ["", f"### {title}（n={fam['n_cases']} 例）", "",
         "各臂发生率：" + "；".join(f"{a} {fam['events'][a]}/{fam['n_cases']}（{fam['rates'][a]*100:.1f}%，Wilson95% {fam['wilson95'][a][0]*100:.1f}–{fam['wilson95'][a][1]*100:.1f}）" for a in fam["events"]),
         f"Cochran Q 检验：Q({o['df']}) = {o['statistic']:.2f}，p = {_p(o['p'])}", "",
         "| 比较（前 − 后） | 率差（百分点） | Tango 95%CI | 仅前/仅后 | McNemar 精确 p | Holm 校正 p | 结论 |", "|---|---|---|---|---|---|---|"]
    for k, r in fam["pairwise"].items():
        x, y = k.split("_minus_")
        L.append(f"| {x} − {y} | {r['diff']*100:+.1f} | {r['ci95'][0]*100:+.1f}, {r['ci95'][1]*100:+.1f} | {r['n10']}/{r['n01']} | {_p(r['p'])} | {_p(r['p_holm'])} | {'差异有统计学意义' if r['significant'] else '未见差异'} |")
    return L


def md(res: dict) -> str:
    pl = res["plan"]
    pop_cn = {"all": "全部测试病例", "A": "任务A", "B": "任务B"}[pl["primary_population"]]
    L = ["# 结果（按预先冻结的分析计划：配对双侧差异检验）", "",
         f"评分输出 {res['n_outputs']} 份；病例 {res['n_cases']} 例；参加评分的臂 {res['arms_rated']}；参加假设检验的臂 {pl['compare_arms']}",
         "臂：" + "；".join(f"{a}={ARM_LABEL[a]}" for a in res["arms_rated"]),
         f"主要终点：{EP_CN[pl['primary_endpoint']]}，人群：{pop_cn}；两两比较 {len(pl['contrasts'])} 对，Holm 校正，双侧 α={pl['alpha_two_sided']}", "",
         "## 各臂描述（按任务）", "", "| 臂|任务 | n | 综合分 | 安全性 | 准确性 | 可用性 | 关键安全错误 | 平均字数 |", "|---|---|---|---|---|---|---|---|"]
    for k, v in res["arm_summary"].items():
        L.append(f"| {k} | {v['n']} | {v['mean_overall']}（{v['sd_overall']}） | {v['mean_safety']} | {v['mean_accuracy']} | {v['mean_usability']} | {v['se_events']}（{v['se_rate']*100:.1f}%） | {v['mean_chars']} |")
    L += ["", "## 主要分析"] + _score_block(res["primary"], f"{EP_CN[pl['primary_endpoint']]}，{pop_cn}")
    s = res["primary_sensitivity_exclude_cases"]
    if "pairwise" in s:
        L += ["", f"敏感性（剔除标记病例后 n={s['n_cases']}）：Friedman p = {_p(s['omnibus']['p'])}；" +
              "；".join(f"{k.replace('_minus_', '−')} {r['mean_diff']:+.2f}（Holm p {_p(r['p_holm'])}）" for k, r in s["pairwise"].items())]
    am = res["primary_sensitivity_adjusted_model"]
    L += ["", f"敏感性（调整任务、病种、输出长度的稳健回归；长度系数 {am['length_coef']:+.3f}）：" +
          "；".join(f"{k.replace('_minus_', '−')} {r['estimate']:+.2f}（Holm p {_p(r['p_holm'])}）" for k, r in am["contrasts"].items())]
    L += ["", "## 次要分析：各评分维度（全部病例）"]
    for c, fam in res["secondary_scores"].items():
        L += _score_block(fam, EP_CN[c])
    L += ["", "## 次要分析：关键安全错误发生率"]
    for k, fam in res["safety_error"].items():
        L += _binary_block(fam, {"all": "全部病例", "taskA": "任务A", "taskB": "任务B"}[k])
    L += ["", "## 按任务的亚组（探索性，Holm 仅在各亚组内校正）"]
    for t, fams in res["by_task"].items():
        fam = fams[pl["primary_endpoint"]]
        L.append(f"- {t}（n={fam['n_cases']}）：Friedman p = {_p(fam['omnibus']['p'])}；" + "；".join(f"{a} {fam['means'][a]:.2f}" for a in fam["means"]))
    if "reference_arm_A6_descriptive" in res:
        r6 = res["reference_arm_A6_descriptive"]
        L += ["", f"## 医师记录（A6，仅描述，不参加检验）：综合分 {r6['mean_overall']}，安全性 {r6['mean_safety']}，准确性 {r6['mean_accuracy']}，可用性 {r6['mean_usability']}，关键安全错误率 {r6['se_rate']*100:.1f}%"]
    L += ["", "## 评分者信度", ""]
    for k, v in res["reliability"].items():
        L.append(f"- {k}: {json.dumps(v, ensure_ascii=False)}")
    L += ["", f"## 支持性混合模型（R）：{res['mixed_models_R']['status']}", "", "全部数字（含按任务的各维度两两比较、有序指标、亚组）见 RESULTS.json。"]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/study.yaml")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    work = Path(cfg["paths"]["work_dir"])
    state = StudyState(work)
    state.require("unblinded")
    fr = json.loads((work / "STUDY_FREEZE.json").read_text(encoding="utf-8"))
    pl = analysis.plan(cfg)
    if fr.get("analysis_plan") != pl:
        raise SystemExit("[ERROR] the stats block of configs/study.yaml differs from the frozen analysis plan (work/STUDY_FREEZE.json: analysis_plan)")
    ap_csv = work / "SEALED" / "advisory_autometrics.csv"
    autom = pd.read_csv(ap_csv, encoding="utf-8-sig") if ap_csv.exists() else None
    res = analysis.run_all(work / "analysis", cfg, pl, Path("R/mixed_models.R"), autom)
    (work / "analysis/RESULTS.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    text = md(res)
    (work / "analysis/RESULTS.md").write_text(text, encoding="utf-8")
    state.advance("analyzed", {"primary_endpoint": pl["primary_endpoint"], "n_contrasts": len(pl["contrasts"])})
    print(text)


if __name__ == "__main__":
    main()
