from __future__ import annotations


import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

CATS7 = ["腰背痛", "颈肩上肢痛", "下肢关节痛", "头面痛", "内脏/盆腔痛", "神经病理性痛", "广泛/其他痛"]
ARMS = ["A1", "A2", "A3", "A4", "A5"]
CASE_ID_RE = re.compile(r"^[A-Z]{1,3}\d?-\d{2,4}$")
TONGUE_RE = re.compile(r"舌[质体苔]?[淡红暗紫绛胖瘦边尖]|舌苔|苔[白黄薄厚腻燥少剥润滑灰黑]")
PULSE_RE = re.compile(r"脉[弦细滑沉数缓紧濡涩浮迟虚实结代洪微弱]")
TP_RE = re.compile(TONGUE_RE.pattern + "|" + PULSE_RE.pattern)


def rater_columns(asg: pd.DataFrame) -> list[str]:
   
    return sorted([c for c in asg.columns if re.match(r"^评分者\d+$", str(c))], key=lambda c: int(str(c)[3:]))


def task_letter(s: str) -> str:
    s = str(s)
    return "A" if s.startswith("任务A") else "B" if s.startswith("任务B") else ""


def arm_id(s: str) -> str:
    m = re.match(r"^(A[1-6])", str(s).strip())   
    return m.group(1) if m else ""


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def user_text(messages: list[dict]) -> str:
    return "\n".join(m["content"] for m in messages if m["role"] == "user")


class Package:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"package directory not found: {self.root}")

    def _one(self, pattern: str) -> Path:
        hits = sorted(p for p in self.root.glob(pattern) if not p.name.startswith(("~", ".")))
        if len(hits) != 1:
            raise FileNotFoundError(f"expected exactly one file matching {pattern!r} in {self.root}, found {len(hits)}")
        return hits[0]

    # ---- files -------------------------------------------------------------------
    @property
    def files(self) -> dict[str, Path]:
        return {
            "train": self.root / "train.jsonl",
            "test_initial_inputs": self.root / "test_initial_inputs.jsonl",
            "test_initial_reference": self.root / "test_initial_reference.jsonl",
            "test_followup_inputs": self.root / "test_followup_inputs.jsonl",
            "test_followup_reference": self.root / "test_followup_reference.jsonl",
            "testlist": self._one("测试集名单*.csv"),
            "annotation": self._one("*分类标注表*.xlsx"),
            "handbook": self._one("*专家评分手册*.xlsx"),
            "acupoint_lib": self._one("*穴位标准库*.xlsx"),
            "followup_xlsx": self._one("复诊病例整理*.xlsx"),
            "pilot": self._one("预跑与界值*.xlsx"),
        }

    # ---- tables ----------------------------------------------------------------------
    def annotation(self) -> pd.DataFrame:
        return pd.read_excel(self.files["annotation"], sheet_name="01_病例标注总表", dtype=str, keep_default_na=False)

    def testlist(self) -> pd.DataFrame:
        return pd.read_csv(self.files["testlist"], encoding="utf-8-sig", dtype=str, keep_default_na=False)

    def assignment(self) -> pd.DataFrame:
        df = pd.read_excel(self.files["handbook"], sheet_name="07_评分者分配", header=3, dtype=str, keep_default_na=False)
        return df[df["case_id"].map(lambda x: bool(CASE_ID_RE.match(str(x))))].reset_index(drop=True)

    def panel_of(self) -> dict[str, list[str]]:
        """case_id -> the raters (one panel) who rate every arm of that case."""
        asg = self.assignment()
        return {r["case_id"]: [r[c] for c in rater_columns(asg)] for r in asg.to_dict("records")}

    def blind_map(self) -> pd.DataFrame:
        df = pd.read_excel(self.files["handbook"], sheet_name="10_揭盲对照表_评分完成前勿开", header=3, dtype=str, keep_default_na=False)
        df = df[df["case_id"].map(lambda x: bool(CASE_ID_RE.match(str(x))))].reset_index(drop=True)
        df["arm"] = df["实际臂"].map(arm_id)
        return df

    def entry_form(self) -> pd.DataFrame:
        return pd.read_excel(self.files["handbook"], sheet_name="06_评分录入表", dtype=str, keep_default_na=False)

    # ---- jsonl ------------------------------------------------------------------------
    def train(self) -> list[dict]:
        return read_jsonl(self.files["train"])

    def test_initial(self) -> list[dict]:
        ref = {r["case_id"]: r["reference"] for r in read_jsonl(self.files["test_initial_reference"])}
        ann = self.annotation().set_index("case_id")
        out = []
        for r in read_jsonl(self.files["test_initial_inputs"]):
            cid = r["meta"]["case_id"]
            a = ann.loc[cid] if cid in ann.index else None
            out.append({"case_id": cid, "messages": r["messages"], "reference": ref.get(cid, ""),
                        "cat7": a["主分类_7类"] if a is not None else "", "task": task_letter(a["任务模式"]) if a is not None else "",
                        "source": a["来源"] if a is not None else "", "visit_type": a["诊次"] if a is not None else ""})
        return out

    def test_followup(self) -> list[dict]:
        ref = {r["case_id"]: r for r in read_jsonl(self.files["test_followup_reference"])}
        return [{"case_id": r["meta"]["case_id"], "visit": r["meta"].get("visit", ""), "messages": r["messages"],
                 "reference": ref[r["meta"]["case_id"]]["reference"], "adjust_type": ref[r["meta"]["case_id"]].get("adjust_type", "")}
                for r in read_jsonl(self.files["test_followup_inputs"])]

    def fingerprint(self) -> str:
        ids = ",".join(sorted(self.testlist()["case_id"]))
        return hashlib.sha256(ids.encode()).hexdigest()

    # ---- audit ---------------------------------------------------------------------------
    def verify(self, expect: dict | None = None) -> dict:
        exp = {"n_test": 55, "n_task_a": 20, "n_task_b": 35, "n_train": 786, "n_followup": 21, "n_arms": 5, "n_raters": 6, "raters_per_output": 3}
        exp.update(expect or {})
        errors, notes, warnings = [], [], []
        tl = self.testlist()
        ids = list(tl["case_id"])
        if len(ids) != exp["n_test"] or len(set(ids)) != len(ids):
            errors.append(f"test list must have {exp['n_test']} unique ids, got {len(ids)}/{len(set(ids))}")
        tasks = Counter(task_letter(x) for x in tl["task"])
        if tasks.get("A") != exp["n_task_a"] or tasks.get("B") != exp["n_task_b"]:
            errors.append(f"task split expected A={exp['n_task_a']}/B={exp['n_task_b']}, got {dict(tasks)}")
        train = self.train()
        tr_ids = {r["meta"]["case_id"] for r in train}
        if len(train) != exp["n_train"]:
            errors.append(f"train instances expected {exp['n_train']}, got {len(train)}")
        if tr_ids & set(ids):
            errors.append(f"LEAKAGE: {len(tr_ids & set(ids))} test case ids appear in train.jsonl: {sorted(tr_ids & set(ids))[:5]}")
        kinds = Counter(r["meta"].get("task") for r in train)
        notes.append(f"train instances by task: {dict(kinds)}; distinct train cases: {len(tr_ids)}")
        ti = self.test_initial()
        if sorted(r["case_id"] for r in ti) != sorted(ids):
            errors.append("test_initial_inputs ids differ from the test list")
        if any(not r["reference"] for r in ti):
            errors.append("some initial references are empty")
        fu = self.test_followup()
        if len(fu) != exp["n_followup"] or not {r["case_id"] for r in fu} <= set(ids):
            errors.append(f"follow-up tasks expected {exp['n_followup']} (subset of test ids), got {len(fu)}")
        ann = self.annotation()
        a_test = ann[ann["case_id"].isin(ids)]
        if set(a_test["划分建议"]) != {"测试集"}:
            errors.append(f"annotation 划分建议 for test ids: {sorted(set(a_test['划分建议']))}")
        bad_pain = a_test[a_test["主分类_7类"] == "非疼痛"]
        if len(bad_pain):
            errors.append("non-pain cases in the test set")
        # task-mode consistency: Task B inputs must not contain tongue/pulse text; Task A must
        a_full = sum(1 for r in ti if r["task"] == "A" and TONGUE_RE.search(user_text(r["messages"])) and PULSE_RE.search(user_text(r["messages"])))
        b_both = [r["case_id"] for r in ti if r["task"] == "B" and TONGUE_RE.search(user_text(r["messages"])) and PULSE_RE.search(user_text(r["messages"]))]
        b_part = [r["case_id"] for r in ti if r["task"] == "B" and bool(TONGUE_RE.search(user_text(r["messages"]))) != bool(PULSE_RE.search(user_text(r["messages"])))]
        if a_full != exp["n_task_a"]:
            errors.append(f"Task A inputs with BOTH tongue and pulse: {a_full}/{exp['n_task_a']}")
        if b_both:
            warnings.append(f"DATA-QUALITY: Task-B inputs that contain BOTH tongue and pulse text (annotation says 'no pulse'): {b_both}. "
                            "By its own input such a case is Task A; the test list is frozen (hashed), so keep it but report it and run the sensitivity analysis without it (analysis does this automatically via configs/study.yaml: sensitivity.exclude_case_ids).")
        if b_part:
            warnings.append(f"{len(b_part)} Task-B inputs contain only ONE of tongue/pulse (annotation '部分(仅舌或仅脉)'): {b_part}. "
                         "Protocol says Task B = no tongue/pulse; the fabrication metric is therefore computed per finding against the input, and these cases should be declared/analysed as a sensitivity subgroup.")
        notes.append(f"Task A with tongue+pulse: {a_full}/{exp['n_task_a']}; Task B with no tongue/pulse: {exp['n_task_b'] - len(b_part) - len(b_both)}/{exp['n_task_b']}")
        bm = self.blind_map()
        if len(bm) != exp["n_test"] * exp["n_arms"] or not bm["盲评编码"].is_unique:
            errors.append(f"blind map should have {exp['n_test'] * exp['n_arms']} unique codes, got {len(bm)}")
        per_case = bm.groupby("case_id")["arm"].apply(lambda s: sorted(s))
        if any(v != ARMS for v in per_case):
            errors.append("some cases in the blind map do not have exactly arms A1..A5")
        asg = self.assignment()
        rcols = rater_columns(asg)
        k = exp["raters_per_output"]
        if len(asg) != exp["n_test"] or len(rcols) != k or any(len({r[c] for c in rcols}) != k for _, r in asg.iterrows()):
            errors.append(f"assignment must list every test case once with {k} DIFFERENT raters (columns 评分者1..{k})")
        load = Counter(x for c in rcols for x in asg[c])
        if len(load) != exp["n_raters"]:
            errors.append(f"expected {exp['n_raters']} raters in the assignment, got {len(load)}")
        panels = Counter("-".join(sorted(r[c] for c in rcols)) for _, r in asg.iterrows())
        notes.append(f"raters={len(load)} cases/rater={dict(sorted(load.items()))}; panels={dict(panels)} (every output rated by {k} raters of one panel)")
        form = self.entry_form()
        if len(form) != exp["n_test"] * exp["n_arms"] * k:
            errors.append(f"entry form should have {exp['n_test'] * exp['n_arms'] * k} rows, got {len(form)}")
        dist = pd.crosstab(a_test["主分类_7类"], a_test["任务模式"].map(task_letter)).reindex(CATS7).fillna(0).astype(int)
        return {"ok": not errors, "errors": errors, "warnings": warnings, "notes": notes, "fingerprint_sha256": self.fingerprint(), "distribution": dist,
                "n_test": len(ids), "n_train_instances": len(train), "n_train_cases": len(tr_ids)}
