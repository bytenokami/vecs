#!/usr/bin/env python3
"""Acceptance checklist gate for workflow Phase 3.

Reads docs/features/<feature-name>/acceptance.md, finds checklist items
of the form `- [ ] text` / `- [x] text`, prints each, asks operator
to confirm. Exits 0 if all pass, 1 otherwise, 2 on usage error.

Usage:
    python scripts/check_acceptance.py <feature-name>
    python scripts/check_acceptance.py <feature-name> --non-interactive  # require all [x]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CHECKLIST_RE = re.compile(r"^\s*[-*]\s*\[([ xX])\]\s*(.+?)\s*$", re.MULTILINE)


def main(argv: list[str]) -> int:
    if len(argv) < 2 or len(argv) > 3:
        print("usage: check_acceptance.py <feature-name> [--non-interactive]", file=sys.stderr)
        return 2
    feature = argv[1]
    non_interactive = len(argv) == 3 and argv[2] == "--non-interactive"
    path = Path("docs/features") / feature / "acceptance.md"
    if not path.exists():
        print(f"acceptance file missing: {path}", file=sys.stderr)
        return 2
    items = CHECKLIST_RE.findall(path.read_text())
    if not items:
        print(f"no checklist items found in {path}", file=sys.stderr)
        return 2
    print(f"acceptance: {feature} ({len(items)} items)")
    print(f"source:     {path}\n")
    failed = 0
    for i, (mark, text) in enumerate(items, 1):
        if non_interactive:
            ok = mark.lower() == "x"
            print(f"  [{i}/{len(items)}] {text} -> {'pass' if ok else 'FAIL'}")
            if not ok:
                failed += 1
            continue
        ans = input(f"  [{i}/{len(items)}] {text} -- pass? [y/N] ").strip().lower()
        if ans != "y":
            failed += 1
    print()
    if failed:
        print(f"FAIL: {failed}/{len(items)} items failed")
        return 1
    print(f"PASS: all {len(items)} items confirmed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
