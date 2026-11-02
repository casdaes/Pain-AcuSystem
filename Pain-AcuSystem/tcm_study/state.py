from __future__ import annotations


import hashlib
import json
import time
from pathlib import Path

STAGES = ["verified", "frozen", "generated", "sheets_made", "scored", "locked", "stage2_made", "stage2_scored", "unblinded", "analyzed", "followup_generated"]


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(obj) -> str:
    return sha256_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str))


def hash_tree(path: str | Path, mode: str = "full", patterns=("*",)) -> dict:

    p = Path(path)
    files = [p] if p.is_file() else sorted(x for pat in patterns for x in p.rglob(pat) if x.is_file() and "checkpoint-" not in str(x))
    out = {}
    for f in files:
        rel = f.name if p.is_file() else str(f.relative_to(p))
        out[rel] = sha256_file(f) if mode == "full" else f"{f.stat().st_size}:{int(f.stat().st_mtime)}"
    return {"digest": sha256_json(out), "files": out}


class StudyState:
    def __init__(self, study_dir: str | Path):
        self.dir = Path(study_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "STUDY_STATE.json"
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {"stages": {}, "audit_log": []}

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def done(self, stage: str) -> bool:
        return stage in self.data["stages"]

    def require(self, *stages: str) -> None:
        missing = [s for s in stages if not self.done(s)]
        if missing:
            raise SystemExit(f"[STUDY][BLOCKED] prerequisite stage(s) not completed: {missing}. Required order: {' -> '.join(STAGES)}")

    def forbid(self, *stages: str, force: bool = False, reason: str = "") -> None:
        """Abort when a later stage already exists (order violation) unless forced (logged)."""
        later = [s for s in stages if self.done(s)]
        if later and not force:
            raise SystemExit(f"[STUDY][BLOCKED] stage(s) {later} already completed; this step is one-shot. Use --force --reason '...' only with statistician approval (it is logged).")
        if later and force:
            self.log("FORCED_RERUN", {"stages": later, "reason": reason})

    def advance(self, stage: str, meta: dict | None = None) -> None:
        self.data["stages"][stage] = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "meta": meta or {}}
        self.save()

    def log(self, event: str, meta: dict | None = None) -> None:
        self.data["audit_log"].append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event, "meta": meta or {}})
        self.save()
