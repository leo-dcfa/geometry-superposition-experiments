"""Rule R5 enforcement: no source module may be swallowed by .gitignore.

The parent program lost a whole data-pipeline module because `.gitignore`
contained an unanchored `data/`, which matched `kosmoslm/data/` as well as the
top-level corpus directory. Nobody noticed until an external auditor tried to
reproduce from a clean clone. This check makes that failure mode loud: every
`.py` file under the source roots must be tracked by git.

Run standalone (`python tools/check_tracked_sources.py`) or via the CI job.
Exits non-zero and names the offending files.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SOURCE_ROOTS = ("csc", "experiments", "tools", "tests")
REPO_ROOT = Path(__file__).resolve().parent.parent


def tracked_files() -> set[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {Path(p) for p in out.split("\0") if p}


def main() -> int:
    tracked = tracked_files()
    missing = []
    for root in SOURCE_ROOTS:
        root_dir = REPO_ROOT / root
        if not root_dir.is_dir():
            continue
        for path in sorted(root_dir.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(REPO_ROOT)
            if rel not in tracked:
                missing.append(rel)

    if missing:
        print("R5 VIOLATION — source modules present on disk but not tracked by git:")
        for rel in missing:
            print(f"  {rel}")
        print("\nCheck .gitignore for an unanchored directory pattern.")
        return 1

    print(f"R5 ok — every .py under {', '.join(SOURCE_ROOTS)} is tracked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
