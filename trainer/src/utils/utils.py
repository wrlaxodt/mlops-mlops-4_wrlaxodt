from __future__ import annotations

import hashlib
import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def model_dir(model_name: str) -> Path:
    d = project_root() / "models" / model_name
    ensure_dir(d)
    return d


def calculate_hash(filename: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def save_hash(dst: str) -> None:
    hash_ = calculate_hash(dst)
    base, _ = os.path.splitext(dst)
    with open(f"{base}.sha256", "w") as f:
        f.write(hash_)


def read_hash(dst: str) -> str:
    base, _ = os.path.splitext(dst)
    with open(f"{base}.sha256", "r") as f:
        return f.read()


def get_latest_model(model_name: str, ext: str = "pkl") -> str:
    d = model_dir(model_name)
    candidates = sorted(d.glob(f"*.{ext}"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No model found in {d} with ext={ext}")
    return str(candidates[0])
