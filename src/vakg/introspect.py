"""Snapshot the public API of an installed library.

This module is executed *inside* the virtual environment of one library version,
so it must only depend on the standard library. It walks the package's public
submodules and records every public attribute reachable as ``module.attr`` (or
``module.Class.method``), together with the parameter names of callables.

Usage (inside the target environment):
    python -I introspect.py <import_name> <output.json> [--seeds a.b,c.d] [--max-depth N]
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import pkgutil
import sys
import warnings

SKIP_PARTS = {"tests", "test", "testing", "conftest", "setup", "benchmarks", "_build"}


def _is_public(name: str) -> bool:
    return not name.startswith("_")


def _params(obj):
    """Return (list of parameter descriptions, accepts_var_keyword) or (None, None)."""
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None, None
    params, var_kw = [], False
    for p in sig.parameters.values():
        if p.kind is p.VAR_KEYWORD:
            var_kw = True
            continue
        if p.kind is p.VAR_POSITIONAL or p.name in ("self", "cls"):
            continue
        params.append({"name": p.name, "kind": p.kind.name.lower(),
                       "required": p.default is p.empty})
    return params, var_kw


def _kind(obj) -> str:
    if inspect.ismodule(obj):
        return "module"
    if inspect.isclass(obj):
        return "class"
    if callable(obj):
        return "function"
    return "attribute"


def _record(entities: dict, path: str, obj, kind: str | None = None) -> None:
    kind = kind or _kind(obj)
    entry = {"kind": kind}
    if kind in ("function", "method", "class"):
        params, var_kw = _params(obj)
        if params is not None:
            entry["params"] = params
            entry["var_keyword"] = var_kw
    entities[path] = entry


def _qualified(cls) -> str:
    """Where a class is really defined, e.g. ``sympy.core.basic.Basic``."""
    return f"{getattr(cls, '__module__', '?')}.{getattr(cls, '__qualname__', cls.__name__)}"


def _walk_class(entities: dict, path: str, cls) -> None:
    """Record only the members a class defines itself; inherited ones are reached
    through its bases (``bases`` -> rdfs:subClassOf in the KG)."""
    entities[path]["defined_as"] = _qualified(cls)
    entities[path]["bases"] = [_qualified(b) for b in getattr(cls, "__bases__", ())
                               if b is not object]
    for name in list(vars(cls)):
        if not _is_public(name):
            continue
        try:
            member = getattr(cls, name)
            if callable(member) and not inspect.isclass(member):
                _record(entities, f"{path}.{name}", member, "method")
            else:
                entities[f"{path}.{name}"] = {"kind": "attribute"}
        except Exception:  # noqa: BLE001 - properties and descriptors may raise
            entities[f"{path}.{name}"] = {"kind": "attribute"}


def _walk_module(entities: dict, path: str, mod, classes_seen: dict) -> list[str]:
    """Record a module's public members; return names of public submodules found.

    A class re-exported by several modules is walked once, at the first path where
    it is seen; later paths only point to it (``alias_of`` -> vakg:aliasOf)."""
    entities.setdefault(path, {"kind": "module"})
    root = path.split(".")[0]
    # __all__ can omit names that are still importable (e.g. networkx.community)
    names = set(dir(mod)) | set(getattr(mod, "__all__", None) or [])
    subs = []
    for name in sorted(n for n in names if isinstance(n, str)):
        if not _is_public(name):
            continue
        full = f"{path}.{name}"
        try:
            member = getattr(mod, name)
            if inspect.ismodule(member):
                real = getattr(member, "__name__", "")
                if real.split(".")[0] != root:
                    continue  # e.g. ``numpy.sys``: not part of this library's API
                if real != full:
                    # a module re-exported under another name: nx.community -> networkx.algorithms.community
                    entities[full] = {"kind": "module", "alias_of": real}
                subs.append(real)
                continue
            if inspect.isclass(member):
                canonical = classes_seen.get(id(member))
                if canonical:
                    entities[full] = {"kind": "class", "alias_of": canonical}
                    continue
                classes_seen[id(member)] = full
                _record(entities, full, member)
                _walk_class(entities, full, member)
            else:
                _record(entities, full, member)
        except Exception:  # noqa: BLE001 - e.g. Flask's context-local proxies raise on access
            entities[full] = {"kind": "attribute"}
    if hasattr(mod, "__path__"):
        for info in pkgutil.iter_modules(mod.__path__):
            if _is_public(info.name) and info.name not in SKIP_PARTS:
                subs.append(f"{path}.{info.name}")
    return subs


def snapshot(import_name: str, seeds: list[str], max_depth: int) -> dict:
    warnings.filterwarnings("ignore")
    entities: dict = {}
    errors: dict = {}
    classes_seen: dict = {}
    queue = [(import_name, 0)] + [(s, 0) for s in seeds]
    seen = set()
    while queue:
        name, depth = queue.pop(0)
        if name in seen:
            continue
        seen.add(name)
        try:
            mod = importlib.import_module(name)
        except BaseException as exc:  # noqa: BLE001 - some submodules exit or need extras
            errors[name] = f"{type(exc).__name__}: {exc}"[:200]
            continue
        subs = _walk_module(entities, name, mod, classes_seen)
        if depth < max_depth:
            queue.extend((s, depth + 1) for s in subs)
        else:
            # beyond the depth limit: record that the submodule exists, without walking it
            for s in subs:
                entities.setdefault(s, {"kind": "module"})
    top = importlib.import_module(import_name)
    return {
        "import_name": import_name,
        "version": getattr(top, "__version__", "unknown"),
        "python": ".".join(map(str, sys.version_info[:3])),
        "n_entities": len(entities),
        "entities": entities,
        "import_errors": errors,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("import_name")
    ap.add_argument("output")
    ap.add_argument("--seeds", default="", help="extra modules to walk, comma-separated")
    ap.add_argument("--max-depth", type=int, default=2)
    args = ap.parse_args()
    seeds = [s for s in args.seeds.split(",") if s]
    snap = snapshot(args.import_name, seeds, args.max_depth)
    with open(args.output, "w") as f:
        json.dump(snap, f)
    print(f"{args.import_name} {snap['version']}: {snap['n_entities']} entities, "
          f"{len(snap['import_errors'])} import errors")


if __name__ == "__main__":
    main()
