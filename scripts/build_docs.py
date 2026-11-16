"""Build the documentation corpus for baseline B1: one file per library version.

For every environment in ``.envs/`` runs ``vakg/docdump.py`` inside it and writes
``docs_corpus/<lib>/<version>.jsonl`` (signature + start of the docstring of every
function, method and class of that version). Needs ``vakg.envs`` to have been run.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCDUMP = ROOT / "src" / "vakg" / "docdump.py"


def build_one(lib: str, version: str, max_chars: int, force: bool) -> str:
    out = ROOT / "docs_corpus" / lib / f"{version}.jsonl"
    if out.exists() and not force:
        return "cached"
    python = ROOT / ".envs" / f"{lib}-{version}" / "bin" / "python"
    snapshot = ROOT / "snapshots" / lib / f"{version}.json"
    if not python.exists() or not snapshot.exists():
        return "missing environment or snapshot (run vakg.envs first)"
    out.parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run([str(python), "-I", str(DOCDUMP), str(snapshot), str(out),
                          "--max-chars", str(max_chars)], capture_output=True, text=True)
    if res.returncode:
        out.unlink(missing_ok=True)
        return "FAILED: " + res.stderr.strip()[-300:]
    return res.stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--max-chars", type=int, default=800)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    subset = json.load(open(ROOT / "data" / "subset.json"))
    specs = sorted({(e["library"], e["version"]) for e in subset["environments"]})
    start, failed = time.time(), 0
    with ThreadPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(build_one, l, v, args.max_chars, args.force): (l, v) for l, v in specs}
        for n, fut in enumerate(as_completed(futures), 1):
            lib, version = futures[fut]
            status = fut.result()
            failed += status.startswith(("FAILED", "missing"))
            print(f"[{n:2d}/{len(specs)}] {lib} {version}: {status}  ({time.time() - start:.0f}s)", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
