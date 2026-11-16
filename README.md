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
| 5. Documentation corpus per version (docstrings), for baseline B1 | `python scripts/build_docs.py` | `docs_corpus/<lib>/<version>.jsonl` |
| 6. Prompts: B0 (model alone) and B1 (+ BM25-retrieved documentation) | `python scripts/make_prompts.py` | `data/prompts_b0.jsonl`, `data/prompts_b1.jsonl` |
| 7. Generation (Qwen2.5-Coder-7B-Instruct, 4-bit, greedy) on a Colab T4 | `notebooks/generate_colab.ipynb` | `results/completions_<run>.jsonl` |
| 8. Hidden tests (pass@1) + checker (invalid calls) | `python scripts/run_eval.py gold` / `b0` / `b1` | `results/eval_<run>.json` |
| 9. Side-by-side table | `python scripts/compare_runs.py b0 b1` | Markdown table |

Setup:

```bash
uv venv .venv -p 3.10 && uv pip install -p .venv/bin/python -e ".[dev]"
git clone https://github.com/mrcabbage972/GitChameleonBenchmark external/GitChameleonBenchmark
.venv/bin/python -m vakg.envs --jobs 6   # ~5 min, needs uv
.venv/bin/python -m vakg.kg              # ~30 s
.venv/bin/python -m pytest -q
```

`snapshots/`, `.envs/`, `docs_corpus/` and `kg/*.nt` are generated and not versioned.

## Subset

`data/subset.json`: 141 GitChameleon 2.0 problems — scipy, sympy, flask, numpy,
scikit-learn, networkx (signature changes) and pandas (control: mostly behaviour
changes) — over 29 library versions.
