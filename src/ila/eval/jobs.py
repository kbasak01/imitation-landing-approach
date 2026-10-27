"""Background-job bookkeeping: ``status.json`` (state, progress, PID) and repo-relative paths.

Every long Phase 2 job (reproduction, throughput, determinism) writes
``artifacts/runs/<job>/status.json`` atomically on every update, so ``/job-status`` can read it
while the job runs. Times are UTC ISO-8601 and wall seconds.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any

from ila import p8_adapter

__all__ = ["JobStatus", "host_info", "now_utc", "rel", "write_json"]


def now_utc() -> str:
    """Return the current UTC time, ISO-8601, seconds."""
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def rel(path: Path | str) -> str:
    """Return ``path`` relative to this repository's root (POSIX), else unchanged."""
    p = Path(path)
    try:
        return p.resolve().relative_to(p8_adapter.REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def write_json(path: Path, record: dict[str, Any]) -> None:
    """Write ``record`` as indented JSON, atomically (write a temp file, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=p8_adapter.REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def host_info() -> dict[str, Any]:
    """Return the machine, interpreter, commits and thread variables, for ``run_info.json``."""
    import torch

    cuda = torch.cuda.is_available()
    return {
        "host": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "p8_commit": p8_adapter.p8_commit(),
        "ila_commit": _git("rev-parse", "HEAD"),
        "ila_dirty_paths": [ln for ln in _git("status", "--porcelain").splitlines() if ln][:50],
        "env_threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
    }


class JobStatus:
    """The live ``status.json`` of one background job.

    Attributes:
        path: ``<run_dir>/status.json``.
        record: The current content.
    """

    def __init__(self, run_dir: Path, job: str) -> None:
        """Start the record (state ``starting``) and write it."""
        run_dir.mkdir(parents=True, exist_ok=True)
        self.path = run_dir / "status.json"
        started = now_utc()
        self.record: dict[str, Any] = {
            "job": job,
            "state": "starting",
            "pid": os.getpid(),
            "started_utc": started,
            "updated_utc": started,
            "progress": None,
            "log": rel(run_dir / "run.log"),
        }
        write_json(self.path, self.record)

    def update(self, state: str | None = None, **fields: Any) -> None:
        """Merge ``fields`` (and the state) into the record and write it."""
        if state is not None:
            self.record["state"] = state
        self.record.update(fields)
        self.record["updated_utc"] = now_utc()
        write_json(self.path, self.record)
