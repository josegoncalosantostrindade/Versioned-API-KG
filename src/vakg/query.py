"""SPARQL access to the versioned API knowledge graph.

The checker needs fast lookups ("does ``numpy.product`` exist in 1.25.0?"), so for
each library the KG is queried once with SPARQL and the answers are cached as a
compact index (``kg/<library>.index.json``). The KG stays the single source of
truth: delete the cache and it is rebuilt from the graph.
"""
from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from rdflib import Graph

ROOT = Path(__file__).resolve().parents[2]
KG_DIR = ROOT / "kg"

PREFIXES = """
PREFIX vakg: <http://example.org/vakg#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""

# Every entity with its kind and the versions in which it exists.
Q_ENTITIES = PREFIXES + """
SELECT ?path ?kind ?ver WHERE {
  ?e vakg:path ?path ; a ?kind ; vakg:validIn ?v .
  ?v vakg:versionString ?ver .
}"""

# Parameters of callables, per version.
Q_PARAMS = PREFIXES + """
SELECT ?path ?pname ?ver WHERE {
  ?e vakg:path ?path ; vakg:hasParam ?p .
  ?p vakg:name ?pname ; vakg:validIn ?v .
  ?v vakg:versionString ?ver .
}"""

# Callables that accept **kwargs, per version.
Q_VARKW = PREFIXES + """
SELECT ?path ?ver WHERE {
  ?e vakg:path ?path ; vakg:acceptsVarKeywordIn ?v .
  ?v vakg:versionString ?ver .
}"""

# Re-exports, inheritance inside the library, and bases outside it.
Q_ALIAS = PREFIXES + """
SELECT ?path ?target WHERE { ?e vakg:path ?path ; vakg:aliasOf ?t . ?t vakg:path ?target . }"""
Q_SUPER = PREFIXES + """
SELECT ?path ?base WHERE { ?e vakg:path ?path ; rdfs:subClassOf ?b . ?b vakg:path ?base . }"""
Q_EXTBASE = PREFIXES + """
SELECT ?path ?base WHERE { ?e vakg:path ?path ; vakg:externalBase ?base . }"""

# Versions in release order.
Q_VERSIONS = PREFIXES + """
SELECT ?ver ?order WHERE { ?v a vakg:Version ; vakg:versionString ?ver ; vakg:order ?order . }
ORDER BY ?order"""


def build_index(library: str) -> dict:
    g = Graph()
    g.parse(KG_DIR / f"{library}.nt", format="nt")
    versions = [str(r.ver) for r in g.query(Q_VERSIONS)]
    entities: dict = defaultdict(lambda: {"kind": None, "versions": []})
    for r in g.query(Q_ENTITIES):
        e = entities[str(r.path)]
        e["kind"] = str(r.kind).rsplit("#", 1)[-1].lower()
        e["versions"].append(str(r.ver))
    params: dict = defaultdict(lambda: defaultdict(list))
    for r in g.query(Q_PARAMS):
        params[str(r.path)][str(r.pname)].append(str(r.ver))
    varkw: dict = defaultdict(list)
    for r in g.query(Q_VARKW):
        varkw[str(r.path)].append(str(r.ver))
    alias = {str(r.path): str(r.target) for r in g.query(Q_ALIAS)}
    supers: dict = defaultdict(list)
    for r in g.query(Q_SUPER):
        supers[str(r.path)].append(str(r.base))
    ext: dict = defaultdict(list)
    for r in g.query(Q_EXTBASE):
        ext[str(r.path)].append(str(r.base))
    return {"library": library, "versions": versions, "entities": dict(entities),
            "params": {k: dict(v) for k, v in params.items()}, "var_keyword": dict(varkw),
            "alias": alias, "superclasses": dict(supers), "external_bases": dict(ext)}


@lru_cache(maxsize=None)
def load_index(library: str) -> dict:
    cache = KG_DIR / f"{library}.index.json"
    ttl = KG_DIR / f"{library}.nt"
    if cache.exists() and cache.stat().st_mtime >= ttl.stat().st_mtime:
        return json.load(open(cache))
    idx = build_index(library)
    json.dump(idx, open(cache, "w"))
    return idx
