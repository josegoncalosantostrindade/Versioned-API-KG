"""Build a versioned RDF knowledge graph from API snapshots.

For each library, every entity (module, class, function, method, attribute) and
every parameter seen in any version becomes a node. Versions are explicit nodes,
ordered by release, and each node is linked to:

    vakg:validIn    every version where it exists
    vakg:addedIn    the first version where it appears (if not in the oldest)
    vakg:removedIn  the first version where it is missing after having existed

Classes store only the members they define; inheritance is explicit:

    rdfs:subClassOf     base class inside the same library (inherited members
                        are found by following it, as in RDFS reasoning)
    vakg:externalBase   base class outside the library (e.g. ``builtins.dict``)
    vakg:aliasOf        a re-export of a class walked at another path

Output: ``kg/<library>.nt`` (N-Triples), queried with SPARQL by ``vakg.query``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote

from packaging.version import Version
from rdflib import RDF, RDFS, BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import XSD

ROOT = Path(__file__).resolve().parents[2]
VAKG = Namespace("http://example.org/vakg#")
API = Namespace("http://example.org/api/")

KIND_CLASS = {
    "module": VAKG.Module, "class": VAKG.Class, "function": VAKG.Function,
    "method": VAKG.Method, "attribute": VAKG.Attribute,
}


def entity_uri(library: str, path: str) -> URIRef:
    return API[f"{quote(library)}/{quote(path, safe='.')}"]


def param_uri(library: str, path: str, param: str) -> URIRef:
    return URIRef(f"{entity_uri(library, path)}#{quote(param)}")


def version_uri(library: str, version: str) -> URIRef:
    return API[f"{quote(library)}@{quote(version)}"]


def load_snapshots(library: str) -> list[dict]:
    snaps = [json.load(open(p)) for p in (ROOT / "snapshots" / library).glob("*.json")]
    # order by the version GitChameleon asks for (the installed one may be a patch substitute)
    return sorted(snaps, key=lambda s: Version(s.get("requested_version", s["version"])))


def _lifecycle(g: Graph, node, present: list[bool], versions: list[URIRef]) -> None:
    """Attach validIn / addedIn / removedIn given presence per ordered version."""
    for v, here in zip(versions, present):
        if here:
            g.add((node, VAKG.validIn, v))
    first = present.index(True)
    if first > 0:
        g.add((node, VAKG.addedIn, versions[first]))
    for i in range(first + 1, len(present)):
        if present[i - 1] and not present[i]:
            g.add((node, VAKG.removedIn, versions[i]))


def build_graph(library: str) -> Graph:
    snaps = load_snapshots(library)
    if not snaps:
        raise SystemExit(f"no snapshots for {library}; run vakg.envs first")
    g = Graph()
    g.bind("vakg", VAKG)
    g.bind("api", API)

    lib_node = API[quote(library)]
    g.add((lib_node, RDF.type, VAKG.Library))
    g.add((lib_node, RDFS.label, Literal(library)))

    versions = []
    for i, s in enumerate(snaps):
        requested = s.get("requested_version", s["version"])
        v = version_uri(library, requested)
        versions.append(v)
        g.add((v, RDF.type, VAKG.Version))
        g.add((v, VAKG.ofLibrary, lib_node))
        g.add((v, VAKG.versionString, Literal(requested)))
        g.add((v, VAKG.installedVersion, Literal(s["version"])))
        g.add((v, VAKG.order, Literal(i, datatype=XSD.integer)))
        if i:
            g.add((versions[i - 1], VAKG.nextVersion, v))

    all_paths = sorted(set().union(*(s["entities"] for s in snaps)))
    # where each class is really defined -> the path at which it was walked
    defined_at = {e["defined_as"]: p for s in snaps for p, e in s["entities"].items()
                  if "defined_as" in e}
    for path in all_paths:
        node = entity_uri(library, path)
        entries = [s["entities"].get(path) for s in snaps]
        kind = next(e["kind"] for e in entries if e)
        g.add((node, RDF.type, KIND_CLASS[kind]))
        g.add((node, VAKG.path, Literal(path)))
        g.add((node, VAKG.name, Literal(path.rsplit(".", 1)[-1])))
        g.add((node, VAKG.ofLibrary, lib_node))
        parent = path.rsplit(".", 1)[0]
        if parent != path:
            g.add((node, VAKG.memberOf, entity_uri(library, parent)))
        _lifecycle(g, node, [e is not None for e in entries], versions)

        # re-exports and inheritance (union over versions; validity lives on the nodes)
        for target in {e["alias_of"] for e in entries if e and "alias_of" in e}:
            g.add((node, VAKG.aliasOf, entity_uri(library, target)))
        for base in {b for e in entries if e for b in e.get("bases", [])}:
            if base in defined_at:
                g.add((node, RDFS.subClassOf, entity_uri(library, defined_at[base])))
            else:
                g.add((node, VAKG.externalBase, Literal(base)))

        # parameters of callables
        if any(e and "params" in e for e in entries):
            for e, v in zip(entries, versions):
                if e and e.get("var_keyword"):
                    g.add((node, VAKG.acceptsVarKeywordIn, v))
            names = sorted({p["name"] for e in entries if e for p in e.get("params", [])})
            for pname in names:
                pnode = param_uri(library, path, pname)
                g.add((pnode, RDF.type, VAKG.Parameter))
                g.add((pnode, VAKG.name, Literal(pname)))
                g.add((node, VAKG.hasParam, pnode))
                present = [bool(e) and any(p["name"] == pname for p in e.get("params", []))
                           for e in entries]
                _lifecycle(g, pnode, present, versions)
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("libraries", nargs="*", help="default: every library with snapshots")
    args = ap.parse_args()
    libs = args.libraries or sorted(p.name for p in (ROOT / "snapshots").iterdir() if p.is_dir())
    (ROOT / "kg").mkdir(exist_ok=True)
    for lib in libs:
        g = build_graph(lib)
        # N-Triples: same RDF as Turtle, but serialises/parses in seconds for large graphs
        out = ROOT / "kg" / f"{lib}.nt"
        g.serialize(out, format="nt", encoding="utf-8")
        removed = len(list(g.subjects(VAKG.removedIn, None)))
        print(f"{lib}: {len(g)} triples, {removed} entities/params removed across versions -> {out}")


if __name__ == "__main__":
    main()
