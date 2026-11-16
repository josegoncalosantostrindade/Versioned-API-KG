"""Compare evaluated runs side by side (a Markdown table, ready for slides or the report).

    python scripts/compare_runs.py b0 b1
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pct(stats: dict, key: str) -> str:
    return f"{stats[key] / stats['n']:.1%}" if stats["n"] else "-"


def main() -> None:
    runs = sys.argv[1:] or ["b0", "b1"]
    data = {r: json.load(open(ROOT / "results" / f"eval_{r}.json")) for r in runs}
    first = data[runs[0]]["summary"]
    rows = [("all", lambda s: s["all"])]
    rows += [(lib, lambda s, lib=lib: s["by_library"][lib]) for lib in first["by_library"]]
    rows += [(f"change: {g}", lambda s, g=g: s["by_group"][g]) for g in first["by_group"]]
    print("| subset | n | " + " | ".join(f"pass@1 {r} | invalid calls {r}" for r in runs) + " |")
    print("|---|---|" + "---|---|" * len(runs))
    for label, pick in rows:
        cells = [f"{pct(pick(data[r]['summary']), 'passed')} | {pct(pick(data[r]['summary']), 'invalid')}" for r in runs]
        print(f"| {label} | {pick(first)['n']} | " + " | ".join(cells) + " |")
    if len(runs) == 2:
        a, b = ({rec["example_id"]: rec["passed"] for rec in data[r]["records"]} for r in runs)
        ids = a.keys() & b.keys()
        print(f"\npaired on {len(ids)} problems: both pass {sum(a[i] and b[i] for i in ids)}, "
              f"only {runs[0]} {sum(a[i] and not b[i] for i in ids)}, "
              f"only {runs[1]} {sum(b[i] and not a[i] for i in ids)}, "
              f"neither {sum(not a[i] and not b[i] for i in ids)}")


if __name__ == "__main__":
    main()
