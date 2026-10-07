"""Minimal file-based model registry.

models/<name>/<version>/model.joblib + metadata.json (version, created_at, data fingerprint,
config hash, metrics, features). models/<name>/LATEST points at the served version, so a
rollback is a one-line change. Same layout maps 1:1 to an MLflow registry if adopted later.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.config import ROOT, load_config, out_dir


def file_sha256(path: Path, limit_mb: int = 50) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(limit_mb * 1024 * 1024))
    return h.hexdigest()[:16]


def config_hash() -> str:
    return hashlib.sha256((ROOT / "config.yaml").read_bytes()).hexdigest()[:16]


def save(name: str, obj, metrics: dict, data_files: list[Path], extra: dict | None = None) -> Path:
    version = load_config()["mlops"]["model_version"]
    d = out_dir("models_dir") / name / version
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, d / "model.joblib")
    meta = {
        "name": name,
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_sha256": config_hash(),
        "data_fingerprints": {Path(f).name: file_sha256(Path(f)) for f in data_files},
        "metrics": metrics,
        **(extra or {}),
    }
    with open(d / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=float)
    (d.parent / "LATEST").write_text(version, encoding="utf-8")
    return d


def load(name: str, version: str | None = None):
    base = out_dir("models_dir") / name
    version = version or (base / "LATEST").read_text(encoding="utf-8").strip()
    d = base / version
    with open(d / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    return joblib.load(d / "model.joblib"), meta
