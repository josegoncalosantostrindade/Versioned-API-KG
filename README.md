# versioned-api-kg

Versioned API knowledge graph (RDF) for detecting and repairing hallucinated or
version-incompatible library calls in LLM-generated code.
Knowledge and Language 2026/27 — José Gonçalo Trindade, Guilherme António Bento.

## Pipeline

| Step | Command | Output |
|---|---|---|
| 1. One environment per library version + API introspection | `python -m vakg.envs` | `snapshots/<lib>/<version>.json` |
| 2. Versioned KG (validIn / addedIn / removedIn, rdfs:subClassOf, aliasOf) | `python -m vakg.kg` | `kg/<lib>.nt` (N-Triples) |
| 3. Check code against one version (AST + SPARQL-built index) | `python -m vakg.checker numpy 1.25.0 file.py` | issues (error / warning) |
| 4. Evaluate the checker on GitChameleon 2.0 ground truth | `python scripts/eval_checker.py` | `results/checker_eval.json` |

Setup:

```bash
uv venv .venv -p 3.10 && uv pip install -p .venv/bin/python -e ".[dev]"
git clone https://github.com/mrcabbage972/GitChameleonBenchmark external/GitChameleonBenchmark
.venv/bin/python -m vakg.envs --jobs 6   # ~5 min, needs uv
.venv/bin/python -m vakg.kg              # ~30 s
.venv/bin/python -m pytest -q
```

`snapshots/`, `.envs/` and `kg/*.nt` are generated and not versioned.

## Subset

`data/subset.json`: 141 GitChameleon 2.0 problems — scipy, sympy, flask, numpy,
scikit-learn, networkx (signature changes) and pandas (control: mostly behaviour
changes) — over 29 library versions.
