"""Shared experiment plumbing: provenance stamps, parallel sweeps, artifact writing.

Rule R5 says a claim without a committed artifact does not enter a scorecard,
so every runner here writes a summary JSON carrying the git commit and the
config that produced it. ``write_artifact`` refuses to write a dict it cannot
serialize rather than emitting a partial file.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = REPO_ROOT / "CSC_RESULTS"


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_artifact(path: Path, payload: dict) -> Path:
    """Serialize first, then write — a half-written artifact is worse than none."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"git_commit": git_commit(), **payload}
    text = json.dumps(payload, indent=2, sort_keys=False)
    path.write_text(text)
    return path


def _worker_init() -> None:
    # Hide the GPU from sweep workers. These phases are CPU-sized, and a pool
    # of workers each opening its own CUDA context is wasteful at best; if the
    # device is already allocated by anything else it OOMs outright, as
    # measured. Set before torch is imported in the worker.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import torch

    # One thread per worker: these models are tiny and intra-op parallelism
    # only makes the workers fight each other for cores.
    torch.set_num_threads(1)


def parallel_map(fn: Callable, items: Iterable, max_workers: int | None = None) -> list:
    """Run ``fn`` over ``items`` in a process pool, preserving input order."""
    items = list(items)
    workers = max_workers or max(1, min(len(items), (os.cpu_count() or 4) // 2))
    if workers == 1:
        _worker_init()
        return [fn(item) for item in items]
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        return list(pool.map(fn, items))
