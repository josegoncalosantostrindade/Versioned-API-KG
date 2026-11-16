"""Write the prompts for the generation baselines.

    data/prompts_b0.jsonl   B0: the model alone
    data/prompts_b1.jsonl   B1: + documentation of the target version retrieved with BM25

One JSON line per problem of the subset: example_id, library, version, messages
(chat format), and for B1 the retrieved entries. B1 needs ``scripts/build_docs.py``.
The gold documentation links of the benchmark are *not* used: retrieval only sees the
task description and the starting code, as it would in practice.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vakg.bm25 import BM25, tokenize  # noqa: E402
from vakg.prompts import messages  # noqa: E402


def load_rows() -> list[dict]:
    ids = set(json.load(open(ROOT / "data" / "subset.json"))["example_ids"])
    rows = [json.loads(l) for l in open(ROOT / "external/GitChameleonBenchmark/dataset/dataset.jsonl")]
    return sorted((r for r in rows if r["example_id"] in ids), key=lambda r: int(r["example_id"]))


class Corpus:
    """The documentation of one library version, indexed with BM25."""

    def __init__(self, library: str, version: str):
        path = ROOT / "docs_corpus" / library / f"{version}.jsonl"
        self.docs = [json.loads(l) for l in open(path)]
        # the path is repeated so that a match on the API name weighs more than one in the prose
        self.index = BM25([tokenize(f"{d['path']} {d['path']} {d['signature']} {d['doc']}") for d in self.docs])

    def retrieve(self, row: dict, k: int) -> list[dict]:
        query = tokenize(row["problem"] + "\n" + row["starting_code"])
        return [{**self.docs[i], "score": round(score, 3)} for i, score in self.index.top(query, k)]


def write(name: str, records: list[dict]) -> None:
    out = ROOT / "data" / f"prompts_{name}.jsonl"
    with open(out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    chars = sum(len(m["content"]) for r in records for m in r["messages"]) / len(records)
    print(f"{out.relative_to(ROOT)}: {len(records)} prompts, {chars:.0f} characters on average")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top-k", type=int, default=3, help="documentation entries per prompt (B1)")
    ap.add_argument("--only", choices=["b0", "b1"], help="write only this baseline")
    args = ap.parse_args()
    rows = load_rows()
    base = lambda r: {"example_id": r["example_id"], "library": r["library"], "version": r["version"]}  # noqa: E731

    if args.only != "b1":
        write("b0", [{**base(r), "run": "b0", "messages": messages(r)} for r in rows])
    if args.only == "b0":
        return

    corpora: dict = {}
    records, hits, with_name = [], 0, 0
    for r in rows:
        key = (r["library"], r["version"])
        if key not in corpora:
            try:
                corpora[key] = Corpus(*key)
            except FileNotFoundError:
                sys.exit(f"no documentation for {key[0]} {key[1]}: run scripts/build_docs.py first")
        docs = corpora[key].retrieve(r, args.top_k)
        # diagnostic only: did retrieval find the API the reference solution is about?
        target = (r.get("name_of_class_or_func") or "").split(".")[-1].strip()
        if target:
            with_name += 1
            hits += any(d["path"].split(".")[-1] == target for d in docs)
        records.append({**base(r), "run": "b1", "messages": messages(r, docs),
                        "retrieved": [{"path": d["path"], "score": d["score"]} for d in docs]})
    write("b1", records)
    if with_name:
        print(f"retrieval diagnostic: the target API is in the top {args.top_k} for "
              f"{hits}/{with_name} problems that name one")


if __name__ == "__main__":
    main()
