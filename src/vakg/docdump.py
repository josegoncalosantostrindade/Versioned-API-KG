"""Dump the docstrings of one installed library version (the B1 documentation corpus).

Like ``introspect.py``, this runs *inside* the environment of one library version and
only uses the standard library. It reads the snapshot written by ``introspect.py`` and,
for every function, method and class in it, stores the signature and the beginning of
the docstring, so the text is exactly the documentation shipped with that version.

Usage (inside the target environment):
    python -I docdump.py <snapshot.json> <output.jsonl> [--max-chars N]
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import warnings


def _resolve(path: str):
    """Import the longest module prefix of ``path`` and walk the rest with getattr."""
    parts = path.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except BaseException:  # noqa: BLE001 - some submodules exit or need extras
            continue
        for name in parts[cut:]:
            obj = getattr(obj, name)
        return obj
    raise ImportError(path)


def dump(snapshot_path: str, max_chars: int) -> list[dict]:
    warnings.filterwarnings("ignore")
    entities = json.load(open(snapshot_path))["entities"]
    docs, seen = [], set()
    for path, ent in entities.items():
        if ent.get("kind") not in ("function", "method", "class") or "alias_of" in ent:
            continue
        try:
            obj = _resolve(path)
            doc = inspect.getdoc(obj)
        except BaseException:  # noqa: BLE001 - proxies and descriptors may raise
            continue
        if not doc or not isinstance(doc, str):
            continue
        try:
            signature = str(inspect.signature(obj))
        except (TypeError, ValueError):
            signature = ""
        text = doc.strip()[:max_chars]
        if (path.rsplit(".", 1)[-1], text) in seen:
            continue  # the same object reachable through two paths
        seen.add((path.rsplit(".", 1)[-1], text))
        docs.append({"path": path, "kind": ent["kind"], "signature": signature[:300], "doc": text})
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("output")
    ap.add_argument("--max-chars", type=int, default=800)
    args = ap.parse_args()
    docs = dump(args.snapshot, args.max_chars)
    with open(args.output, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    print(f"{len(docs)} documented entities")


if __name__ == "__main__":
    main()
