"""Load config.yaml once and resolve paths relative to the project root."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def p(rel: str) -> Path:
    """Absolute path for a project-relative path."""
    return ROOT / rel


def out_dir(kind: str = "outputs_dir") -> Path:
    d = p(load_config()["paths"][kind])
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=float)
