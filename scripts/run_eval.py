"""Evaluate one set of generated answers: hidden tests (pass@1) and the KG checker.

    python scripts/run_eval.py gold    # reference solutions: validates the harness itself
    python scripts/run_eval.py b0      # reads results/completions_b0.jsonl (from the Colab notebook)

For every problem the code is extracted from the answer, written next to the hidden
test of GitChameleon 2.0 and run with pytest inside ``.envs/<lib>-<version>`` (as the
official harness does). The same code is given to the checker for the target version.
Writes ``results/eval_<run>.json`` and prints a summary.

Metrics
    pass@1          fraction of problems whose hidden tests pass (one greedy answer each)
    invalid calls   fraction of problems where the checker reports at least one *error*
                    (a member or parameter that the KG says does not exist in that version)
"""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vakg.checker import check_code  # noqa: E402
from vakg.prompts import extract_code  # noqa: E402

BENCH = ROOT / "external" / "GitChameleonBenchmark" / "dataset"
PYTEST = "pytest==7.2.0"  # the version the official harness uses for Python 3.10
BEHAVIOUR = {"output behaviour", "output change", "gh\noutput behaviour"}
OTHER = {"other library or new feature"}


def change_group(type_of_change: str) -> str:
    """signature: visible in the API surface (names, parameters); behaviour: it is not."""
    if type_of_change in BEHAVIOUR:
        return "behaviour"
    return "other" if type_of_change in OTHER else "signature"


def load_rows() -> dict:
    ids = set(json.load(open(ROOT / "data" / "subset.json"))["example_ids"])
    rows = (json.loads(l) for l in open(BENCH / "dataset.jsonl"))
    return {r["example_id"]: r for r in rows if r["example_id"] in ids}


# -- environments ---------------------------------------------------------------
def _uv_install(python: Path, reqs: list[str]) -> bool:
    cmd = ["uv", "pip", "install", "-q", "-p", str(python), "--only-binary", ":all:", *reqs]
    return subprocess.run(cmd, capture_output=True, text=True).returncode == 0


def _relaxed(req: str) -> str:
    """numpy==1.21.2 -> numpy>=1.21,<1.22 (newest patch of the same minor version)."""
    name, _, version = req.partition("==")
    parts = version.split(".")
    if len(parts) < 2:
        return req
    return f"{name}>={parts[0]}.{parts[1]},<{parts[0]}.{int(parts[1]) + 1}"


def prepare_env(envs_dir: Path, lib: str, version: str, extras: set[str]) -> dict:
    """Install pytest and the problems' additional dependencies; done once per environment."""
    env = envs_dir / f"{lib}-{version}"
    python = env / "bin" / "python"
    if not python.exists():
        return {"ok": False, "missing": ["environment not built (run vakg.envs)"]}
    wanted = sorted(extras)
    marker = env / ".vakg_test_deps.json"
    if marker.exists():
        state = json.load(open(marker))
        if state.get("wanted") == wanted:
            return state
    snapshot = ROOT / "snapshots" / lib / f"{version}.json"
    installed = json.load(open(snapshot))["version"] if snapshot.exists() else version
    pin = f"{lib}=={installed}"  # keep the library fixed while its neighbours are installed
    missing, substituted = [], {}
    if not _uv_install(python, [pin, PYTEST, *wanted]):
        if not _uv_install(python, [pin, PYTEST]):
            missing.append(PYTEST)
        for req in wanted:
            if _uv_install(python, [pin, req]):
                continue
            if _relaxed(req) != req and _uv_install(python, [pin, _relaxed(req)]):
                substituted[req] = _relaxed(req)
            else:
                missing.append(req)
    state = {"ok": PYTEST not in missing, "wanted": wanted, "missing": missing, "substituted": substituted}
    json.dump(state, open(marker, "w"))
    return state


# -- one problem ----------------------------------------------------------------
def run_tests(python: Path, example_id: str, code: str, timeout: int) -> tuple[bool, str]:
    test = (BENCH / "hidden_tests" / f"test_sample_{example_id}.py").read_text()
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, f"sample_{example_id}.py").write_text(code)
        Path(tmp, "test_sample.py").write_text(test)
        cmd = [str(python), "-m", "pytest", "--disable-warnings", "-q", "-p", "no:cacheprovider", tmp]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=tmp)
        except subprocess.TimeoutExpired:
            return False, f"timeout after {timeout}s"
        return proc.returncode == 0, (proc.stdout + proc.stderr)[-1500:]


def evaluate(row: dict, code: str, envs_dir: Path, env_state: dict, timeout: int) -> dict:
    lib, version = row["library"], row["version"]
    rec = {"example_id": row["example_id"], "library": lib, "version": version,
           "type_of_change": row["type_of_change"], "group": change_group(row["type_of_change"]),
           "code": code, "compiles": True, "checker_errors": [], "checker_warnings": []}
    try:
        ast.parse(code)
        for issue in check_code(code, lib, version):
            entry = {"line": issue.line, "symbol": issue.symbol, "message": issue.message,
                     "suggestions": issue.suggestions}
            rec["checker_errors" if issue.severity == "error" else "checker_warnings"].append(entry)
    except SyntaxError as exc:
        rec["compiles"] = False
        rec["syntax_error"] = str(exc)
    needs = {d.lower() for d in (row["additional_dependencies"] or "").split()}
    rec["env_complete"] = env_state["ok"] and not (needs & set(env_state["missing"]))
    if not env_state["ok"]:
        rec["passed"], rec["test_output"] = False, "environment unavailable: " + ", ".join(env_state["missing"])
    else:
        python = envs_dir / f"{lib}-{version}" / "bin" / "python"
        rec["passed"], rec["test_output"] = run_tests(python, row["example_id"], code, timeout)
    return rec


# -- summary --------------------------------------------------------------------
def _line(label: str, recs: list[dict]) -> str:
    n = len(recs)
    passed = sum(r["passed"] for r in recs)
    invalid = sum(bool(r["checker_errors"]) for r in recs)
    return f"{label:14s} n={n:3d}  pass@1={passed / n:6.1%} ({passed:3d})  invalid calls={invalid / n:6.1%} ({invalid:3d})"


def summarise(recs: list[dict]) -> dict:
    def stats(group: list[dict]) -> dict:
        return {"n": len(group), "passed": sum(r["passed"] for r in group),
                "invalid": sum(bool(r["checker_errors"]) for r in group),
                "syntax_errors": sum(not r["compiles"] for r in group)}

    by_lib, by_group = defaultdict(list), defaultdict(list)
    for r in recs:
        by_lib[r["library"]].append(r)
        by_group[r["group"]].append(r)
    print()
    print(_line("ALL", recs))
    for name, group in sorted(by_lib.items()):
        print(_line(name, group))
    print("by kind of change:")
    for name, group in sorted(by_group.items()):
        print(_line("  " + name, group))
    # does a checker error predict a failing test? (rows: checker, columns: tests)
    table = {(flag, ok): 0 for flag in (True, False) for ok in (True, False)}
    for r in recs:
        table[(bool(r["checker_errors"]), r["passed"])] += 1
    print("checker vs tests:   flagged+failed={}  flagged+passed={}  clean+failed={}  clean+passed={}".format(
        table[(True, False)], table[(True, True)], table[(False, False)], table[(False, True)]))
    incomplete = [r["example_id"] for r in recs if not r["env_complete"]]
    syntax = sum(not r["compiles"] for r in recs)
    print(f"syntax errors: {syntax}   problems run in an incomplete environment: {len(incomplete)} {incomplete}")
    return {"all": stats(recs), "by_library": {k: stats(v) for k, v in sorted(by_lib.items())},
            "by_group": {k: stats(v) for k, v in sorted(by_group.items())},
            "checker_vs_tests": {"flagged_failed": table[(True, False)], "flagged_passed": table[(True, True)],
                                 "clean_failed": table[(False, False)], "clean_passed": table[(False, True)]},
            "incomplete_env": incomplete}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="'gold' or the name in results/completions_<run>.jsonl")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=120, help="seconds per problem")
    ap.add_argument("--libraries", default="", help="only these libraries, comma-separated")
    ap.add_argument("--envs-dir", default=str(ROOT / ".envs"))
    args = ap.parse_args()
    envs_dir = Path(args.envs_dir)
    rows = load_rows()
    wanted_libs = set(filter(None, args.libraries.split(",")))
    if wanted_libs:
        rows = {i: r for i, r in rows.items() if r["library"] in wanted_libs}

    if args.run == "gold":
        codes = {i: r["starting_code"] + r["solution"] for i, r in rows.items()}
    else:
        path = ROOT / "results" / f"completions_{args.run}.jsonl"
        answers = {a["example_id"]: a["answer"] for a in map(json.loads, open(path))}
        absent = sorted(set(rows) - set(answers), key=int)
        if absent:
            print(f"warning: no answer for {len(absent)} problems (counted as failed): {absent}")
        codes = {i: extract_code(answers.get(i, ""), r["starting_code"]) for i, r in rows.items()}

    extras = defaultdict(set)
    for r in rows.values():
        extras[(r["library"], r["version"])].update(d.lower() for d in (r["additional_dependencies"] or "").split())
    print(f"preparing {len(extras)} environments (pytest + additional dependencies)...", flush=True)
    states = {}
    for n, ((lib, version), deps) in enumerate(sorted(extras.items()), 1):
        states[(lib, version)] = prepare_env(envs_dir, lib, version, deps)
        note = "" if not states[(lib, version)]["missing"] else f"  MISSING {states[(lib, version)]['missing']}"
        sub = states[(lib, version)].get("substituted")
        print(f"  [{n:2d}/{len(extras)}] {lib}-{version}{note}{'  substituted ' + str(sub) if sub else ''}", flush=True)

    start, recs = time.time(), []
    with ThreadPoolExecutor(args.jobs) as pool:
        futures = [pool.submit(evaluate, r, codes[i], envs_dir, states[(r["library"], r["version"])], args.timeout)
                   for i, r in rows.items()]
        for n, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            recs.append(rec)
            flag = "" if not rec["checker_errors"] else "  checker: " + ", ".join(e["symbol"] for e in rec["checker_errors"])
            print(f"[{n:3d}/{len(futures)}] {rec['example_id']:>3s} {rec['library']:12s} {rec['version']:7s} "
                  f"{'PASS' if rec['passed'] else 'FAIL'}{flag}  ({time.time() - start:.0f}s)", flush=True)
    recs.sort(key=lambda r: int(r["example_id"]))
    summary = summarise(recs)
    out = ROOT / "results" / f"eval_{args.run}.json"
    out.parent.mkdir(exist_ok=True)
    json.dump({"run": args.run, "environments": {f"{l}-{v}": s for (l, v), s in states.items()},
               "summary": summary, "records": recs}, open(out, "w"), indent=1)
    print(f"\nwritten {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
