from __future__ import annotations


import itertools
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import stats as ss

SYSTEM_ARMS = ["A1", "A2", "A3", "A4", "A5"]
SCORES = ["safety", "accuracy", "usability"]
ENDPOINTS = ["overall"] + SCORES
ORD_COLS = ["ord_诊断一致", "ord_证型一致", "ord_选穴策略一致", "ord_操作合规", "ord_选穴临床可接受", "ord_信息缺失识别"]
ARM_LABEL = {"A1": "4B", "A2": "4B+RAG", "A3": "4B+LoRA", "A4": "4B+LoRA+RAG", "A5": "大模型+RAG", "A6": "医师记录"}


def plan(cfg: dict) -> dict:

    st = cfg["stats"]
    arms = list(st.get("compare_arms") or SYSTEM_ARMS)
    con = st.get("contrasts", "all")
    pairs = [list(p) for p in itertools.combinations(arms, 2)] if con in (None, "all") else [list(p) for p in con]
    for a, b in pairs:
        if a not in arms or b not in arms or a == b:
            raise SystemExit(f"[ERROR] stats.contrasts contains an invalid pair: {a} vs {b}")
    ep = st.get("primary_endpoint", "overall")
    if ep not in ENDPOINTS:
        raise SystemExit(f"[ERROR] stats.primary_endpoint must be one of {ENDPOINTS}")
    pop = str(st.get("primary_population", "all"))
    if pop not in ("all", "A", "B"):
        raise SystemExit("[ERROR] stats.primary_population must be all | A | B")
    return {"design": "paired two-sided difference tests", "compare_arms": arms, "contrasts": pairs, "primary_endpoint": ep, "primary_population": pop,
            "alpha_two_sided": float(st.get("alpha_two_sided", 0.05)), "multiplicity": "holm", "omnibus": "friedman / cochran_q",
            "pairwise": "wilcoxon_signed_rank / mcnemar_exact", "bootstrap": int(st.get("bootstrap", 10000))}


def _prep(out: pd.DataFrame, rat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out, rat = out.copy(), rat.copy()
    out["overall"] = out[SCORES].mean(axis=1)
    rat["overall"] = rat[SCORES].mean(axis=1)
    return out, rat


def _wide(df: pd.DataFrame, col: str, arms: list[str]) -> pd.DataFrame:
    return df[df["arm"].isin(arms)].pivot(index="case_id", columns="arm", values=col)[arms].dropna()


def _subset(out: pd.DataFrame, pop: str, exclude: list[str] | None = None) -> pd.DataFrame:
    d = out if pop == "all" else out[out["task"] == pop]
    return d[~d["case_id"].isin(exclude or [])]


def score_family(out: pd.DataFrame, col: str, pl: dict, pop: str = "all", exclude: list[str] | None = None, seed: int = 20260920) -> dict:

    w = _wide(_subset(out, pop, exclude), col, pl["compare_arms"])
    res = {"endpoint": col, "population": pop, "n_cases": len(w), "means": {a: round(float(w[a].mean()), 3) for a in w.columns},
           "sds": {a: round(float(w[a].std(ddof=1)), 3) for a in w.columns}, "omnibus": ss.friedman(w.to_numpy())}
    pw, pv = {}, {}
    for a, b in pl["contrasts"]:
        r = ss.paired_score_contrast(w[b].to_numpy(), w[a].to_numpy(), boot=pl["bootstrap"], seed=seed)
        k = f"{b}_minus_{a}"
        pw[k], pv[k] = r, r["p"]
    adj = ss.holm(pv)
    for k in pw:
        pw[k]["p_holm"] = adj[k]
        pw[k]["significant"] = bool(adj[k] < pl["alpha_two_sided"])
    res["pairwise"] = pw
    return res


def binary_family(out: pd.DataFrame, pl: dict, pop: str = "all", exclude: list[str] | None = None, col: str = "se_any_final") -> dict:
    w = _wide(_subset(out, pop, exclude), col, pl["compare_arms"]).astype(int)
    res = {"endpoint": col, "population": pop, "n_cases": len(w), "events": {a: int(w[a].sum()) for a in w.columns},
           "rates": {a: round(float(w[a].mean()), 4) for a in w.columns}, "wilson95": {a: [round(x, 4) for x in ss.wilson_ci(int(w[a].sum()), len(w))] for a in w.columns},
           "omnibus": ss.cochran_q(w.to_numpy())}
    pw, pv = {}, {}
    for a, b in pl["contrasts"]:
        r = ss.paired_binary_contrast(w[b].to_numpy(), w[a].to_numpy())
        k = f"{b}_minus_{a}"
        pw[k], pv[k] = r, r["p"]
    adj = ss.holm(pv)
    for k in pw:
        pw[k]["p_holm"] = adj[k]
        pw[k]["significant"] = bool(adj[k] < pl["alpha_two_sided"])
    res["pairwise"] = pw
    return res


def adjusted_model(rat: pd.DataFrame, col: str, pl: dict) -> dict:

    arms = pl["compare_arms"]
    d = rat[rat["arm"].isin(arms)].copy()
    cats = sorted(d["cat7"].unique())[1:]
    d["len_z"] = (d["n_chars"] - d["n_chars"].mean()) / (d["n_chars"].std() or 1)
    ref, others = arms[0], arms[1:]
    X = np.column_stack([np.ones(len(d))] + [(d["arm"] == a).astype(float) for a in others] + [(d["task"] == "B").astype(float)]
                        + [(d["cat7"] == c).astype(float) for c in cats] + [d["len_z"]])
    fit = ss.ols_cluster_robust(d[col].to_numpy(), X, d["case_id"].to_numpy())
    pos = {a: 1 + i for i, a in enumerate(others)}
    res, pv = {}, {}
    for a, b in pl["contrasts"]:
        c = np.zeros(X.shape[1])
        if b != ref:
            c[pos[b]] += 1
        if a != ref:
            c[pos[a]] -= 1
        k = f"{b}_minus_{a}"
        res[k] = ss.wald_contrast(fit, c)
        pv[k] = res[k]["p"]
    adj = ss.holm(pv)
    for k in res:
        res[k]["p_holm"] = adj[k]
    return {"model": f"{col} ~ arm + task + cat7 + output_length_z (rater-level rows, case-cluster robust OLS)", "n_rows": len(d),
            "length_coef": float(fit["coef"][-1]), "contrasts": res}


def reliability(rat: pd.DataFrame) -> dict:

    res = {}
    d = rat.copy()
    d["_panel"] = d.groupby("盲评编码")["评分者"].transform(lambda s: "-".join(sorted(s)))
    for s in SCORES:
        panels = {}
        for panel, g in d.groupby("_panel"):
            m = g.pivot(index="盲评编码", columns="评分者", values=s).dropna().to_numpy(float)
            if m.shape[0] >= 4 and m.shape[1] >= 2:
                v = ss.icc2k(m)
                v1 = ss.icc2_single(m)
                if np.isfinite(v):
                    panels[panel] = {"icc2_1": v1, "icc2_k": v, "k": int(m.shape[1]), "n_outputs": int(m.shape[0])}
        allm = d.assign(_k=d.groupby("盲评编码").cumcount()).pivot(index="盲评编码", columns="_k", values=s).dropna().to_numpy(float)
        one = ss.icc1(allm)
        ws = [p["n_outputs"] for p in panels.values()]
        res[s] = {"per_panel": panels, "n_panels": len(panels),
                  "icc2_k_panel_weighted_mean": float(np.average([p["icc2_k"] for p in panels.values()], weights=ws)) if panels else None,
                  "icc2_1_panel_weighted_mean": float(np.average([p["icc2_1"] for p in panels.values()], weights=ws)) if panels else None,
                  "icc_one_way_single": one["icc1_1"], "icc_one_way_mean_of_k": one["icc1_k"], "k": one["k"], "n_outputs": one["n_targets"],
                  "target_threshold": 0.75,
                  "note": "primary reliability index = two-way random ICC(2,1)/ICC(2,k) within each fully crossed panel; one-way ICC over all outputs is the pooled index"}
    m = d.pivot_table(index="盲评编码", columns=d.groupby("盲评编码").cumcount(), values="se_any").dropna().to_numpy(int)
    k = m.shape[1]
    kappas = [ss.cohen_kappa(m[:, i], m[:, j]) for i in range(k) for j in range(i + 1, k)]
    res["never_event_any"] = {"n_outputs": len(m), "k": int(k), "raw_agreement": float((m.min(axis=1) == m.max(axis=1)).mean()), "gwet_ac1": ss.gwet_ac1(m),
                              "cohen_kappa_mean_pairwise": float(np.nanmean(kappas)) if kappas else None, "target_threshold": 0.70}
    return res


def subgroups(out: pd.DataFrame) -> dict:
    res = {}
    for by in ("cat7", "task", "provenance", "visit_type"):
        t = {}
        for (g, arm), s in out.groupby([by, "arm"]):
            k, n = int(s["se_any_final"].sum()), len(s)
            lo, hi = ss.wilson_ci(k, n)
            t[f"{g}|{arm}"] = {"n": n, "mean_overall": round(float(s["overall"].mean()), 2), "mean_safety": round(float(s["safety"].mean()), 2),
                               "mean_accuracy": round(float(s["accuracy"].mean()), 2), "mean_usability": round(float(s["usability"].mean()), 2),
                               "se_events": f"{k}/{n}", "se_rate": round(k / n, 3), "wilson95": [round(lo, 3), round(hi, 3)]}
        res[by] = t
    return res


def ordinal_descriptives(rat: pd.DataFrame) -> dict:
    res = {}
    for c in ORD_COLS:
        if c not in rat.columns:
            continue
        d = rat[rat[c].notna()]
        res[c[4:]] = {arm: {"n": len(s), "mean_0_2": round(float(s[c].mean()), 3), "share_yes": round(float((s[c] == 2).mean()), 3), "share_no": round(float((s[c] == 0).mean()), 3),
                            "partial_as_yes": round(float((s[c] >= 1).mean()), 3)} for arm, s in d.groupby("arm")}
    return res


def run_r(ana_dir: Path, r_script: Path, arms: list[str]) -> dict:
    if not shutil.which("Rscript") or not r_script.exists():
        return {"status": "SKIPPED (Rscript/R script not available): run  Rscript R/mixed_models.R work/analysis/rater_level.csv"}
    p = subprocess.run(["Rscript", str(r_script), str(ana_dir / "rater_level.csv"), ",".join(arms)], capture_output=True, text=True)
    (ana_dir / "mixed_models_R_output.txt").write_text(p.stdout + "\n" + p.stderr, encoding="utf-8")
    return {"status": "OK" if p.returncode == 0 else "FAILED", "output": str(ana_dir / "mixed_models_R_output.txt")}


def run_all(ana_dir: Path, cfg: dict, pl: dict, r_script: Path, autom: pd.DataFrame | None = None) -> dict:
    st = cfg["stats"]
    out = pd.read_csv(ana_dir / "output_level.csv", encoding="utf-8-sig")
    rat = pd.read_csv(ana_dir / "rater_level.csv", encoding="utf-8-sig")
    out, rat = _prep(out, rat)
    excl = (st.get("sensitivity") or {}).get("exclude_case_ids") or []
    seed, boot = int(cfg["study"]["seed"]), pl["bootstrap"]
    arms, ep, pop = pl["compare_arms"], pl["primary_endpoint"], pl["primary_population"]
    missing = [a for a in arms if a not in set(out["arm"])]
    if missing:
        raise SystemExit(f"[ERROR] compared arms missing from the rated outputs: {missing}")
    res = {"plan": pl, "n_outputs": len(out), "n_cases": int(out["case_id"].nunique()), "arms_rated": sorted(out["arm"].unique())}
    res["arm_summary"] = {}
    for (arm, tk), s in out.groupby(["arm", "task"]):
        k, n = int(s["se_any_final"].sum()), len(s)
        res["arm_summary"][f"{arm}|task{tk}"] = {"n": n, **{f"mean_{c}": round(float(s[c].mean()), 2) for c in ENDPOINTS}, "sd_overall": round(float(s["overall"].std(ddof=1)), 2),
                                                "se_events": f"{k}/{n}", "se_rate": round(k / n, 4), "wilson95": [round(x, 4) for x in ss.wilson_ci(k, n)],
                                                "mean_chars": round(float(s["n_chars"].mean()), 1)}
    # ---- primary
    res["primary"] = score_family(out, ep, pl, pop, seed=seed)
    res["primary_sensitivity_exclude_cases"] = score_family(out, ep, pl, pop, excl, seed=seed) if excl else {"status": "no cases flagged"}
    res["primary_sensitivity_adjusted_model"] = adjusted_model(rat if pop == "all" else rat[rat["task"] == pop], ep, pl)
    # ---- secondary: every score endpoint on all cases and by task
    res["secondary_scores"] = {c: score_family(out, c, pl, "all", seed=seed) for c in ENDPOINTS if not (c == ep and pop == "all")}
    res["by_task"] = {f"task{t}": {c: score_family(out, c, pl, t, seed=seed) for c in ENDPOINTS} for t in ("A", "B") if (out["task"] == t).any()}
    # ---- critical safety error (binary)
    res["safety_error"] = {"all": binary_family(out, pl, "all")}
    for t in ("A", "B"):
        if (out["task"] == t).any():
            res["safety_error"][f"task{t}"] = binary_family(out, pl, t)
    # ---- descriptive / quality
    if "A6" in set(out["arm"]):
        s = out[out["arm"] == "A6"]
        res["reference_arm_A6_descriptive"] = {"note": "clinician record; not part of any hypothesis test", "n": len(s),
                                               **{f"mean_{c}": round(float(s[c].mean()), 2) for c in ENDPOINTS}, "se_rate": round(float(s["se_any_final"].mean()), 4)}
    res["ordinal_descriptives"] = ordinal_descriptives(rat)
    res["reliability"] = reliability(rat)
    res["subgroups_descriptive"] = subgroups(out)
    res["length_check_by_arm"] = {a: round(float(s["n_chars"].mean()), 1) for a, s in out.groupby("arm")}
    k = int(cfg.get("rating", {}).get("raters_per_output", 3))
    res["missing_ratings"] = {"raters_per_output": k, "expected_rater_rows": int(len(out) * k), "observed": len(rat), "missing_proportion": round(1 - len(rat) / (k * len(out)), 4)}
    if autom is not None:
        m = out.merge(autom, on=["case_id", "arm"], how="inner")
        m = m[m["arm"].isin(arms)]
        res["spearman_autometric_vs_expert"] = {k: ss.spearman(m[k], m["accuracy"]) for k in ("f1", "weighted_f1", "jaccard") if k in m and len(m) > 5}
    res["mixed_models_R"] = run_r(ana_dir, r_script, arms)
    return res
