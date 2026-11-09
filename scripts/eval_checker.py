"""Evaluate the checker on GitChameleon 2.0 ground-truth solutions.

For every problem in the subset, the reference code (starting code + solution) is
checked against:
  * its own version   -> it is correct there, so any issue is a false positive;
  * every other version of the same library -> issues flag a version incompatibility.

Writes results/checker_eval.json and prints a summary per library.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vakg.checker import check_code  # noqa: E402
from vakg.query import load_index  # noqa: E402


def main() -> None:
    subset = json.load(open(ROOT / "data" / "subset.json"))
    ids = set(subset["example_ids"])
    rows = [json.loads(l) for l in open(ROOT / "external/GitChameleonBenchmark/dataset/dataset.jsonl")]
    rows = [r for r in rows if r["example_id"] in ids]
    libs = sorted({r["library"] for r in rows})
    if len(sys.argv) > 1:
        libs = [l for l in libs if l in sys.argv[1:]]

    records, summary = [], {}
    for lib in libs:
        versions = load_index(lib)["versions"]
        own = defaultdict(int)
        cross_total = cross_flagged = 0
        for r in (r for r in rows if r["library"] == lib):
            code = r["starting_code"] + r["solution"]
            try:
                own_issues = check_code(code, lib, r["version"])
            except SyntaxError:
                own["syntax_error"] += 1
                continue
            own["problems"] += 1
            own["false_positive_problems"] += any(i.severity == "error" for i in own_issues)
            own["warning_problems"] += any(i.severity == "warning" for i in own_issues)
            rec = {"example_id": r["example_id"], "library": lib, "version": r["version"],
                   "type_of_change": r["type_of_change"],
                   "own_version_issues": [asdict(i) for i in own_issues], "other_versions": {}}
            for v in versions:
                if v == r["version"]:
                    continue
                issues = check_code(code, lib, v)
                cross_total += 1
                cross_flagged += any(i.severity == "error" for i in issues)
                rec["other_versions"][v] = [asdict(i) for i in issues]
            records.append(rec)
        summary[lib] = {**own, "cross_version_checks": cross_total, "cross_version_flagged": cross_flagged}
        print(f"{lib:13s} problems={own['problems']:3d}  "
              f"false-positive problems={own['false_positive_problems']:2d}  "
              f"warnings={own['warning_problems']:2d}  "
              f"cross-version flagged={cross_flagged}/{cross_total}")
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    json.dump({"summary": summary, "records": records}, open(out / "checker_eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
