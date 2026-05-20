#!/usr/bin/env python3
"""Register vecs as a vecs project so it can index its own source.

Idempotent. Edits ~/.vecs/config.yaml. Adds a `vecs` project entry whose
`code_dirs` points at this repo's src/ tree. Skips if `vecs` already
registered.

Usage:
    python scripts/register_self.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

CONFIG_PATH = Path.home() / ".vecs" / "config.yaml"
REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    if not CONFIG_PATH.exists():
        print(f"missing: {CONFIG_PATH}", file=sys.stderr)
        return 1
    data = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    projects = data.setdefault("projects", {})
    if "vecs" in projects:
        print(f"vecs already registered at {CONFIG_PATH}")
        return 0
    projects["vecs"] = {
        "code_dirs": [
            {
                "path": str(REPO_ROOT),
                "extensions": [".py", ".md"],
                "include_dirs": ["src", "docs", "tests", "scripts"],
                "exclude_dirs": [
                    ".venv",
                    ".uv",
                    ".worktrees",
                    "__pycache__",
                    ".pytest_cache",
                    ".ruff_cache",
                ],
            }
        ],
        "docs_dir": str(REPO_ROOT / "docs"),
    }
    CONFIG_PATH.write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"registered vecs at {REPO_ROOT} -> {CONFIG_PATH}")
    print("next: vecs index -p vecs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
