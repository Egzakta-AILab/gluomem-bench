# gluomem-bench

Benchmark harness, offline Optuna retrieval tuner, ablation runner, and DSPy prompt optimization suite for [GluoMem](https://github.com/UkisAI-Egzakta/GluoMem).

## Scope

`gluomem-bench` separates evaluation, literature baselines (`bm25`, `vector`, `full_context`, `mem0`), dataset adapters (`LoCoMo`, `LongMemEval`, `MemoryAgentBench` replay fixtures, `MINJA` adversarial suites, bilingual `SR/EN` fixtures), Optuna retrieval parameter tuning (`gluomem_bench.tune`), and offline DSPy prompt compilation (`gluomem_bench.llm.optimize`, `gluomem_bench.llm.campaign`, `gluomem_bench.llm.extraction_eval`) from the core `gluomem` runtime library.

## CLI

```bash
# Inspect runtime + benchmark component readiness
uv run gluomem-bench components --json

# Ingest a LoCoMo JSON file into a .mind SQLite database
uv run gluomem-bench ingest --input benchmarks_data/locomo10.json --db out.mind

# Fetch / verify benchmark datasets and run evaluations
uv run gluomem-bench bench fetch --dataset locomo
uv run gluomem-bench bench run --dataset locomo --retriever gluomem --db-dir runs/stores
uv run gluomem-bench bench rescore --run-dir runs/<run_id> --check

# Offline Optuna tuning and ablations over cached candidate pools
uv run gluomem-bench tune --cache-dir runs/cache --out runs/tune.json
uv run gluomem-bench ablate --cache-dir runs/cache --out runs/ablate.md
```

## BEAM 128K (adapted rubric protocol)

Use `uv run gluomem-bench beam --help` for the isolated rack-only
`profile`, `prepare`, `ingest`, `run`, and offline `rescore` commands.
Dataset source, license and immutable revision pins are in
`benchmarks_data/beam_128k_source.json`. Configure your own model endpoint;
machine-specific profiles and internal run receipts do not belong in this repository.
No full BEAM campaign or native paper-score reproduction is claimed.

## Checks

```bash
make check
```
