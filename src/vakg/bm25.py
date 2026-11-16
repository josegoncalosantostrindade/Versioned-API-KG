"""A small BM25 (Okapi) retriever, used by baseline B1 to fetch documentation.

Written from scratch (standard library only) so that the baseline has no hidden
behaviour: ``score(q, d) = sum_t idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))``.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_SPLIT = re.compile(r"_+|(?<=[a-z0-9])(?=[A-Z])")

STOPWORDS = frozenset("""
a an and are as at be by for from has have if in is it its of on or that the this to
was were which will with not can do does may must should than then they we you your
def return import class self cls none true false pass lambda else elif while try except
""".split())


def tokenize(text: str) -> list[str]:
    """Lower-cased words; identifiers are kept whole *and* split on ``_``/camelCase,
    so ``greedy_modularity_communities`` matches both itself and ``modularity``."""
    out = []
    for ident in _IDENT.findall(text):
        parts = [p for p in _SPLIT.split(ident) if p]
        if len(parts) > 1:
            out.append(ident.lower())
        out.extend(p.lower() for p in parts)
    return [t for t in out if len(t) > 1 and t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.lengths = [len(d) for d in docs]
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                self.postings[term].append((i, tf))

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def top(self, query: list[str], k: int = 3) -> list[tuple[int, float]]:
        """The k best (document index, score) pairs, best first."""
        scores: dict[int, float] = defaultdict(float)
        for term in set(query):
            if term not in self.postings:
                continue
            idf = self.idf(term)
            for i, tf in self.postings[term]:
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
                scores[i] += idf * tf * (self.k1 + 1) / norm
        return sorted(scores.items(), key=lambda s: (-s[1], s[0]))[:k]
