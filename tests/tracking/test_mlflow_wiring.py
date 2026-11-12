"""Inventory test: every existing experiment boundary remains on the old path."""

from __future__ import annotations

import ast
from pathlib import Path

# Module -> (family, campaign functions, condition functions). Pure scorers,
# read-only verification/rescore, source selectors and prepare/admission commands
# intentionally do not create experiment runs.
ENTRYPOINTS = {
    "bench/parallel_campaign.py": ("campaign", ("run_jobs",), ()),
    "bench/lme_first.py": ("longmemeval", ("run_isolated_jobs",), ()),
    "bench/comprehensive_driver.py": ("longmemeval", ("execute_longmemeval",), ()),
    "bench/lme_first_execute.py": (
        "longmemeval",
        ("execute_candidate",),
        ("run_native_job", "run_authored_qualification"),
    ),
    "bench/suite_campaign.py": ("longmemeval", (), ("run_history",)),
    "bench/runner.py": ("qa", (), ("run_benchmark", "ingest_per_conversation")),
    "bench/mab_campaign.py": ("memoryagentbench", ("execute",), ("run_history",)),
    "bench/beam_driver.py": ("beam", ("execute",), ("run_history",)),
    "bench/beam_eval.py": ("beam", (), ("run_beam",)),
    "bench/beam_ingest.py": ("beam", (), ("ingest_beam",)),
    "bench/v2_driver.py": ("longmemeval_v2", ("execute",), ("run_group", "qualify_vision")),
    "bench/arena_driver.py": (
        "memoryarena_formal",
        ("execute_formal", "execute_matrix"),
        ("run_native_task", "qualify_authored_memory"),
    ),
    "bench/arena_campaign.py": ("memoryarena_math", (), ("run_math_task", "run_offline_smoke")),
    "bench/arena_travel_driver.py": (
        "memoryarena_travel",
        ("execute_approved", "execute_continuation", "execute_matrix"),
        ("run_task", "smoke"),
    ),
    "bench/arena_shopping_driver.py": (
        "memoryarena_shopping",
        ("execute_shopping", "execute_continuation", "execute_matrix"),
        ("run_task",),
    ),
    "bench/arena_search_driver.py": ("memoryarena_search", ("execute",), ("run_task",)),
    "bench/cognee_driver.py": (
        "longmemeval",
        ("execute_cognee", "run_admitted"),
        ("run_graph_job",),
    ),
    "bench/native_backend_qualification.py": ("native_qualification", (), ("run",)),
    "bench/remediation_study.py": ("longmemeval", ("execute",), ()),
    "bench/trajectory_suite.py": ("trajectory_bridge", (), ("run_v2_memory", "run_arena")),
    "bench/replay.py": ("replay_fixture", (), ("run_replay_fixture",)),
    "bench/adversarial.py": ("adversarial_fixture", (), ("run_adversarial_fixture",)),
    "bench/mem0.py": ("qa", (), ("ingest_mem0_per_conversation",)),
    "ingest.py": ("ingest", (), ("ingest_path",)),
    "tune/objective.py": ("retrieval_tuning", (), ("run_optuna_tune",)),
    "tune/ablate.py": ("retrieval_ablation", (), ("run_ablation",)),
    "llm/optimize.py": ("dspy_optimization", ("main",), ("optimize_task",)),
}


def test_all_existing_experiment_boundaries_have_opt_in_hooks():
    root = Path(__file__).parents[2] / "gluomem_bench"
    missing = []
    for filename, (family, campaigns, conditions) in ENTRYPOINTS.items():
        module = ast.parse((root / filename).read_text(encoding="utf-8"))
        definitions = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef)}
        for kind, names in (("campaign", campaigns), ("condition", conditions)):
            for name in names:
                node = definitions[name]
                decorators = [
                    d
                    for d in node.decorator_list
                    if isinstance(d, ast.Call)
                    and isinstance(d.func, ast.Name)
                    and d.func.id == "track_experiment"
                ]
                if not any(
                    d.args
                    and ast.literal_eval(d.args[0]) == family
                    and any(
                        k.arg == "kind" and ast.literal_eval(k.value) == kind for k in d.keywords
                    )
                    for d in decorators
                ):
                    missing.append(filename + ":" + name)
    assert not missing, "Missing existing execution hooks: " + ", ".join(missing)
