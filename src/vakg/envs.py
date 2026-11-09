"""Create one isolated environment per (library, version) and snapshot its API.

Reads ``data/subset.json`` (the GitChameleon 2.0 subset), installs each library
version with ``uv`` into ``.envs/<lib>-<version>`` and runs ``introspect.py``
there, writing ``snapshots/<lib>/<version>.json``.

If the exact version has no binary wheel for the chosen Python, the newest
patch release of the same minor version is used instead; both the requested
and the installed version are recorded, so the substitution is explicit.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTROSPECT = Path(__file__).with_name("introspect.py")

IMPORT_NAMES = {"scikit-learn": "sklearn", "pillow": "PIL"}


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def _minor_range(version: str) -> str:
    parts = version.split(".")
    if len(parts) == 1:
        return f"=={parts[0]}.*"
    major, minor = int(parts[0]), int(parts[1])
    return f">={major}.{minor},<{major}.{minor + 1}"


def _install(env: Path, python: str, reqs: list[str]) -> subprocess.CompletedProcess:
    return _run(["uv", "pip", "install", "-q", "-p", str(env / "bin" / "python"),
                 "--only-binary", ":all:", *reqs])


def _seeds(library: str, import_name: str) -> list[str]:
    """Modules referenced by the subset's problems, so they are always walked."""
    rows = [json.loads(l) for l in open(ROOT / "external/GitChameleonBenchmark/dataset/dataset.jsonl")] \
        if (ROOT / "external/GitChameleonBenchmark/dataset/dataset.jsonl").exists() else []
    seeds = set()
    for r in rows:
        if r["library"] != library:
            continue
        for call in r.get("api_calls") or []:
            parts = call.split(".")
            if parts[0] in (import_name, library) and len(parts) > 2:
                seeds.add(".".join([import_name] + parts[1:-1]))
    return sorted(seeds)


def build_one(spec: dict, python: str, max_depth: int, force: bool) -> dict:
    lib, version = spec["library"], spec["version"]
    import_name = IMPORT_NAMES.get(lib, lib)
    out = ROOT / "snapshots" / lib / f"{version}.json"
    if out.exists() and not force:
        return {"library": lib, "requested": version, "status": "cached"}
    env = ROOT / ".envs" / f"{lib}-{version}"
    if not env.exists():
        res = _run(["uv", "venv", "-q", "-p", python, str(env)])
        if res.returncode:
            return {"library": lib, "requested": version, "status": "venv-failed", "error": res.stderr[-300:]}
    extras = [e for e in (spec.get("extra") or "").split() if e]
    res = _install(env, python, [f"{lib}=={version}", *extras])
    substituted = False
    if res.returncode:
        # No wheel for the exact version: fall back to the newest patch of the same minor.
        res = _install(env, python, [f"{lib}{_minor_range(version)}"])
        substituted = True
        if res.returncode:
            return {"library": lib, "requested": version, "status": "install-failed", "error": res.stderr[-300:]}
    out.parent.mkdir(parents=True, exist_ok=True)
    seeds = ",".join(_seeds(lib, import_name))
    res = _run([str(env / "bin" / "python"), "-I", str(INTROSPECT), import_name, str(out),
                "--seeds", seeds, "--max-depth", str(max_depth)])
    if res.returncode:
        return {"library": lib, "requested": version, "status": "introspect-failed", "error": res.stderr[-300:]}
    snap = json.load(open(out))
    snap.update({"library": lib, "requested_version": version, "substituted": substituted})
    json.dump(snap, open(out, "w"))
    return {"library": lib, "requested": version, "installed": snap["version"],
            "substituted": substituted, "entities": snap["n_entities"], "status": "ok"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", default=str(ROOT / "data" / "subset.json"))
    ap.add_argument("--python", default="3.10", help="Python used for every environment")
    ap.add_argument("--libraries", default="", help="only these libraries, comma-separated")
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    subset = json.load(open(args.subset))
    wanted = set(filter(None, args.libraries.split(",")))
    # one environment per (library, version): extra pins only matter for running tests
    specs = {}
    for env in subset["environments"]:
        if wanted and env["library"] not in wanted:
            continue
        specs.setdefault((env["library"], env["version"]), env)
    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda s: build_one(s, args.python, args.max_depth, args.force),
                                specs.values()))
    for r in results:
        print(json.dumps(r))
    report = ROOT / "snapshots" / "build_report.json"
    report.parent.mkdir(exist_ok=True)
    json.dump(results, open(report, "w"), indent=1)
    if any(r["status"] not in ("ok", "cached") for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
