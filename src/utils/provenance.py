"""Provenance sidecar writer.

Every result/figure gets a JSON recording git hash, environment versions,
seed, input checksums, and runtime.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


def _git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or "not_a_git_repo"
    except (FileNotFoundError, subprocess.SubprocessError):
        return "git_unavailable"


def _file_sha256(path: Path | str) -> str:
    p = Path(path)
    if not p.exists():
        return "file_missing"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_provenance(
    path: Path | str, *,
    seed: int | None = None,
    runtime_s: float | None = None,
    input_files: Mapping[str, Path | str] | None = None,
    extras: Mapping[str, Any] | None = None,
) -> dict:
    """Write a provenance sidecar JSON."""
    try:
        import torch
        torch_version = torch.__version__
        cuda_version = torch.version.cuda or "cpu_only"
    except ImportError:
        torch_version = "not_installed"
        cuda_version = "unknown"

    record: dict[str, Any] = {
        "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
        "git_hash": _git_hash(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "seed": seed,
        "runtime_s": runtime_s,
    }

    if input_files:
        record["input_sha256"] = {
            name: _file_sha256(p) for name, p in input_files.items()
        }
    if extras:
        record["extras"] = dict(extras)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(record, f, indent=2)
    return record
