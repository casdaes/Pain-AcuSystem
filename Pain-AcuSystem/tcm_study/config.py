from __future__ import annotations

from pathlib import Path

import yaml

_PATH_SECTIONS = ("paths", "models", "outputs")


def load_cfg(path: str | Path = "configs/study.yaml") -> dict:
    p = Path(path).expanduser().resolve()
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    root = Path(cfg.get("project_dir", p.parent.parent)).expanduser()
    root = root if root.is_absolute() else (p.parent / root).resolve()
    cfg["project_dir"] = str(root)
    for sec in _PATH_SECTIONS:
        for k, v in (cfg.get(sec) or {}).items():
            if isinstance(v, str) and v.strip():
                q = Path(v).expanduser()
                cfg[sec][k] = str(q if q.is_absolute() else (root / q).resolve())
    return cfg
