"""Check library calls in Python code against one version of the API knowledge graph.

The checker parses the code (``ast``), resolves every attribute chain that starts at
the target library (``np.convolve``, ``df.append`` with ``df: pd.DataFrame``,
``stats.norm``) to a fully qualified path, and asks the KG index whether that
member, and every keyword argument passed to it, exists in the target version.
Inherited members are found by following ``rdfs:subClassOf``.

Findings are only reported when the KG can say so with confidence: an attribute of
a value whose type is unknown (e.g. the result of a function call) is not checked.

Usage:
    vakg-check numpy 1.25.0 file.py
"""
from __future__ import annotations

import argparse
import ast
import difflib
import json
from dataclasses import asdict, dataclass

from .query import load_index

IMPORT_NAMES = {"scikit-learn": "sklearn", "pillow": "PIL"}


@dataclass
class Issue:
    line: int
    kind: str          # "missing_member" | "missing_param"
    symbol: str        # fully qualified path (or path(param=))
    message: str
    exists_in: list    # versions where it does exist (may be empty)
    suggestions: list  # close names that do exist in the target version
    # "error": the KG has evidence (the symbol exists in another version, or the
    #          callable's signature is known and lacks the parameter);
    # "warning": never seen in any version, but introspection cannot see everything
    #            (instance attributes set in __init__, unwalked submodules).
    severity: str = "error"


class VersionView:
    """The API of one library version, as seen through the KG index."""

    def __init__(self, library: str, version: str):
        self.idx = load_index(library)
        if version not in self.idx["versions"]:
            raise ValueError(f"{library} {version} not in KG; known: {self.idx['versions']}")
        self.version = version

    # -- basic facts -------------------------------------------------------
    def canonical(self, path: str) -> str:
        """Follow re-exports (vakg:aliasOf), also for members of an alias."""
        alias = self.idx["alias"]
        if path in alias:
            return alias[path]
        head, _, tail = path.rpartition(".")
        if head in alias:
            return f"{alias[head]}.{tail}"
        return path

    def kind(self, path: str) -> str | None:
        e = self.idx["entities"].get(path)
        return e["kind"] if e and self.version in e["versions"] else None

    def history(self, path: str) -> list:
        e = self.idx["entities"].get(path)
        return e["versions"] if e else []

    # -- class members, following rdfs:subClassOf ----------------------------
    def lookup_member(self, cls: str, name: str, _seen=None) -> tuple[str, str | None]:
        """Return ("found", path) | ("missing", None) | ("unknown", None)."""
        _seen = _seen or set()
        cls = self.canonical(cls)
        if cls in _seen:
            return "missing", None
        _seen.add(cls)
        cand = f"{cls}.{name}"
        if self.kind(cand):
            return "found", cand
        unknown = bool(self.idx["external_bases"].get(cls))
        for base in self.idx["superclasses"].get(cls, []):
            status, path = self.lookup_member(base, name, _seen)
            if status == "found":
                return status, path
            unknown |= status == "unknown"
        return ("unknown", None) if unknown else ("missing", None)

    def siblings(self, parent: str) -> list[str]:
        prefix = parent + "."
        return [p[len(prefix):] for p, e in self.idx["entities"].items()
                if p.startswith(prefix) and "." not in p[len(prefix):]
                and self.version in e["versions"]]

    # -- parameters -----------------------------------------------------------
    def params(self, path: str) -> tuple[set, bool]:
        ps = self.idx["params"].get(path, {})
        names = {n for n, vs in ps.items() if self.version in vs}
        var_kw = self.version in self.idx["var_keyword"].get(path, [])
        return names, var_kw

    def param_history(self, path: str, name: str) -> list:
        return self.idx["params"].get(path, {}).get(name, [])


class _Resolver(ast.NodeVisitor):
    def __init__(self, view: VersionView, import_name: str):
        self.v = view
        self.root = import_name
        self.names: dict[str, str] = {}   # local name -> module/entity path
        self.types: dict[str, str] = {}   # local variable -> class path
        self.issues: list[Issue] = []
        self._reported: set = set()

    # -- imports and simple type inference ----------------------------------
    def _ours(self, path: str) -> bool:
        return path == self.root or path.startswith(self.root + ".")

    def visit_Import(self, node):
        for a in node.names:
            if self._ours(a.name):
                local = a.asname or a.name.split(".")[0]
                self.names[local] = a.name if a.asname else a.name.split(".")[0]

    def visit_ImportFrom(self, node):
        if node.module and self._ours(node.module) and node.level == 0:
            for a in node.names:
                if a.name != "*":
                    self.names[a.asname or a.name] = f"{node.module}.{a.name}"
                    self._check_path(f"{node.module}.{a.name}", node.lineno)

    def _annotation_type(self, ann) -> str | None:
        path = self._path_of(ann) if ann is not None else None
        if path and self.v.kind(self.v.canonical(path)) == "class":
            return self.v.canonical(path)
        return None

    def visit_FunctionDef(self, node):
        for a in node.args.args + node.args.kwonlyargs:
            t = self._annotation_type(a.annotation)
            if t:
                self.types[a.arg] = t
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name):
            t = self._annotation_type(node.annotation)
            if t:
                self.types[node.target.id] = t
        self.generic_visit(node)

    def visit_Assign(self, node):
        self.generic_visit(node)
        # x = SomeClass(...)  ->  x has type SomeClass
        if isinstance(node.value, ast.Call) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            path = self._path_of(node.value.func)
            if path and self.v.kind(self.v.canonical(path)) == "class":
                self.types[node.targets[0].id] = self.v.canonical(path)

    # -- resolution ---------------------------------------------------------
    def _chain(self, node) -> tuple[str | None, list[str]]:
        """``a.b.c`` -> ("a", ["b", "c"])."""
        attrs = []
        while isinstance(node, ast.Attribute):
            attrs.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            return node.id, attrs[::-1]
        return None, []

    def _path_of(self, node) -> str | None:
        """Resolve an expression to a KG path without reporting (None if unknown)."""
        base, attrs = self._chain(node)
        if base in self.names:
            return ".".join([self.names[base], *attrs])
        return None

    def _report(self, issue: Issue):
        key = (issue.line, issue.symbol)
        if key not in self._reported:
            self._reported.add(key)
            self.issues.append(issue)

    def _check_path(self, full: str, line: int) -> str | None:
        """Walk ``mod.sub.Class.member`` step by step; report the first missing step.
        Returns the resolved path of the last step, or None if it cannot be resolved."""
        parts = full.split(".")
        cur = parts[0]
        if not self.v.kind(cur):
            return None
        for name in parts[1:]:
            if name.startswith("_"):
                return None  # private names are not part of the recorded API
            cur = self.v.canonical(cur)
            kind = self.v.kind(cur)
            if kind == "module":
                cand = f"{cur}.{name}"
                if self.v.kind(cand):
                    cur = cand
                    continue
                if self.v.kind(self.v.canonical(cand)):
                    cur = cand
                    continue
                self._missing(cand, cur, line)
                return None
            if kind == "class":
                status, path = self.v.lookup_member(cur, name)
                if status == "found":
                    cur = path
                    continue
                if status == "missing":
                    self._missing(f"{cur}.{name}", cur, line)
                return None
            return None  # function / attribute: what it returns is unknown
        return self.v.canonical(cur)

    def _missing(self, symbol: str, parent: str, line: int):
        hist = self.v.history(symbol)
        sugg = difflib.get_close_matches(symbol.rsplit(".", 1)[1], self.v.siblings(parent), n=3)
        if hist:
            msg = (f"`{symbol}` does not exist in {self.v.idx['library']} {self.v.version}. "
                   f"It exists in versions: {', '.join(hist)}.")
            severity = "error"
        else:
            msg = (f"`{symbol}` was not found in {self.v.idx['library']} {self.v.version} "
                   f"(nor in any other known version).")
            severity = "warning"
        self._report(Issue(line, "missing_member", symbol, msg, hist, sugg, severity))

    def visit_Attribute(self, node):
        base, attrs = self._chain(node)
        if base in self.names:
            self._check_path(".".join([self.names[base], *attrs]), node.lineno)
        elif base in self.types and attrs:
            self._check_path(".".join([self.types[base], *attrs]), node.lineno)
        else:
            self.generic_visit(node)

    def visit_Call(self, node):
        self.generic_visit(node)
        base, attrs = self._chain(node.func)
        if base in self.names:
            full = ".".join([self.names[base], *attrs])
        elif base in self.types and attrs:
            full = ".".join([self.types[base], *attrs])
        else:
            return
        resolved = self._resolved_quiet(full)
        if not resolved:
            return
        names, var_kw = self.v.params(resolved)
        if var_kw or not self.v.idx["params"].get(resolved):
            return  # accepts **kwargs, or no signature known
        for kw in node.keywords:
            if kw.arg and kw.arg not in names:
                hist = self.v.param_history(resolved, kw.arg)
                where = f" It exists in versions: {', '.join(hist)}." if hist else ""
                self._report(Issue(node.lineno, "missing_param", f"{resolved}({kw.arg}=)",
                                   f"`{resolved}` has no parameter `{kw.arg}` in "
                                   f"{self.v.idx['library']} {self.v.version}.{where}",
                                   hist, difflib.get_close_matches(kw.arg, sorted(names), n=3)))

    def _resolved_quiet(self, full: str) -> str | None:
        saved, self.issues = self.issues, []
        saved_rep = set(self._reported)
        try:
            return self._check_path(full, 0)
        finally:
            self.issues, self._reported = saved, saved_rep


def check_code(code: str, library: str, version: str) -> list[Issue]:
    view = VersionView(library, version)
    resolver = _Resolver(view, IMPORT_NAMES.get(library, library))
    tree = ast.parse(code)
    # imports first, so names are known wherever they are used
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            resolver.visit(node)
    resolver.visit(tree)
    return sorted(resolver.issues, key=lambda i: (i.line, i.symbol))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("library")
    ap.add_argument("version")
    ap.add_argument("file")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    issues = check_code(open(args.file).read(), args.library, args.version)
    if args.json:
        print(json.dumps([asdict(i) for i in issues], indent=1))
        return
    for i in issues:
        extra = f"  (did you mean: {', '.join(i.suggestions)}?)" if i.suggestions else ""
        print(f"line {i.line} [{i.severity}]: {i.message}{extra}")
    print(f"{len(issues)} issue(s)")


if __name__ == "__main__":
    main()
