#!/usr/bin/env python3
"""Verify the immutable code and figure snapshot against SHA-256 hashes."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "SNAPSHOT_SHA256SUMS"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def snapshot_files() -> list[Path]:
    files: list[Path] = []
    for directory in (ROOT / "code", ROOT / "assets" / "figures"):
        files.extend(path for path in directory.rglob("*") if path.is_file())
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix())


def write_manifest() -> int:
    lines = [
        "# SHA-256 for the unmodified V22 code subset, checkpoints, test pairs and supplied figures.",
        *(f"{digest(path)}  {path.relative_to(ROOT).as_posix()}" for path in snapshot_files()),
    ]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {MANIFEST.name}: {len(lines) - 1} file(s)")
    return 0


def main() -> int:
    failures = 0
    checked = 0
    for number, raw_line in enumerate(MANIFEST.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        checked += 1
        if not path.is_file():
            failures += 1
            print(f"[MISSING] {relative}")
        elif digest(path) != expected:
            failures += 1
            print(f"[CHANGED] {relative}")
    if failures:
        print(f"Snapshot verification failed: {failures}/{checked} file(s)")
        return 1
    print(f"Snapshot verified: {checked} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(write_manifest() if sys.argv[1:] == ["--write"] else main())
