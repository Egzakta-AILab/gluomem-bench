"""gluomem-bench CLI entrypoint (`gluomem-bench`).

Provides benchmark, dataset ingestion, and offline tuning/ablation commands:
- `gluomem-bench ingest`
- `gluomem-bench components`
- `gluomem-bench bench {run,rescore,gate,fetch,sample,ingest,verify,replay,registry}`
- `gluomem-bench tune`
- `gluomem-bench ablate`
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from gluomem.cli import _open_existing_db, _parse_override

if TYPE_CHECKING:
    from gluomem_bench.config import AppConfig


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--override",
        action="append",
        type=_parse_override,
        metavar="K=V",
        help="dotted config override, e.g. dspy.tasks.answer.model=sonnet5",
    )


def _config_from(args: argparse.Namespace) -> AppConfig:
    from gluomem_bench.config import load_config

    return load_config(overrides=dict(args.override or []))


def _cmd_ingest(args: argparse.Namespace) -> int:
    from gluomem.llm import make_extractors

    from gluomem_bench.ingest import ingest_path

    cfg = _config_from(args)
    facade = make_extractors(cfg)
    try:
        report = ingest_path(args.path, cfg, facade=facade, db_path=args.db)
    except (FileNotFoundError, ValueError) as exc:
        print(f"gluomem ingest: {exc}", file=sys.stderr)
        return 1
    print(
        f"ingested {report.conversations} conversation(s), "
        f"{report.sessions} session(s), {report.turns} turn(s): "
        f"I={report.i_frames} P={report.p_frames} B={report.b_frames} "
        f"deduped={report.deduped} quarantined={report.quarantined} "
        f"rejected={report.rejected} entities={report.entities} "
        f"relations={report.relations} retired_edges={report.retired_edges} "
        f"demoted={report.demoted}"
    )
    for err in report.extraction_errors:
        print(f"extraction error: {err}", file=sys.stderr)
    return 0


def _cmd_components(args: argparse.Namespace) -> int:
    import json

    from gluomem_bench.components import component_report

    cfg = _config_from(args)
    db = None
    if args.db:
        try:
            db = _open_existing_db(args.db, cfg)
        except FileNotFoundError as exc:
            print(f"gluomem components: {exc}", file=sys.stderr)
            return 1
    try:
        report = component_report(cfg, db=db)
    finally:
        if db is not None:
            db.close()
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_ready:
        store_ok = report.get("store", {}).get("embedder_matches") is not False
        return 0 if report["ready"] and store_ok else 3
    return 0


def _cmd_bench_run(args: argparse.Namespace) -> int:
    from gluomem_bench.bench.adapters.locomo import DatasetError
    from gluomem_bench.bench.runner import run_benchmark

    cfg = _config_from(args)
    try:
        paths = run_benchmark(
            cfg,
            dataset=args.dataset,
            retriever=args.retriever,
            db_dir=args.db_dir,
            out_root=args.out,
            limit=args.limit,
        )
    except (DatasetError, FileNotFoundError, ValueError) as exc:
        print(f"gluomem bench: {exc}", file=sys.stderr)
        return 1
    print(f"run dir: {paths.root}")
    if args.gate:
        import json

        entries = json.loads(paths.scores.read_text(encoding="utf-8"))
        return _gate_verdict(entries, cfg)
    return 0


def _gate_verdict(entries: list[dict], cfg: AppConfig) -> int:
    from gluomem_bench.bench.report import check_gate

    gate = cfg.benchmark.ci_gate
    try:
        ok, entry = check_gate(entries, gate)
    except ValueError as exc:
        print(f"gluomem bench: {exc}", file=sys.stderr)
        return 1
    if not ok:
        print(
            f"GATE FAIL metric={gate.metric} value={entry['value']:.4f} min={gate.min}",
            file=sys.stderr,
        )
        return 1
    print(f"GATE PASS metric={gate.metric} value={entry['value']:.4f} min={gate.min}")
    return 0


def _cmd_bench_rescore(args: argparse.Namespace) -> int:
    from gluomem_bench.bench.rescore import RescoreMismatchError, rescore_run

    try:
        entries = rescore_run(args.run_dir, check=args.check)
    except (FileNotFoundError, RescoreMismatchError) as exc:
        print(f"gluomem bench: {exc}", file=sys.stderr)
        return 1
    verb = "verified" if args.check else "rewrote"
    print(f"rescore {verb} {len(entries)} score entries from {args.run_dir}")
    return 0


def _cmd_bench_gate(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    cfg = _config_from(args)
    scores_path = Path(args.run_dir) / "scores.json"
    try:
        entries = json.loads(scores_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        print(f"gluomem bench: {exc}", file=sys.stderr)
        return 1
    return _gate_verdict(entries, cfg)


def _cmd_bench_fetch(args: argparse.Namespace) -> int:
    from gluomem_bench.bench.adapters import make_adapter
    from gluomem_bench.bench.adapters.locomo import DatasetError

    cfg = _config_from(args)
    try:
        adapter = make_adapter(args.dataset, cfg)
        path = adapter.fetch(force=args.force)
    except (DatasetError, ValueError, OSError) as exc:
        print(f"gluomem bench: {exc}", file=sys.stderr)
        return 1
    print(f"fetched {args.dataset} -> {path} (sha256 {adapter.sha})")
    return 0


def _cmd_bench_sample(args: argparse.Namespace) -> int:
    from pathlib import Path

    from gluomem_bench.bench.adapters import make_adapter
    from gluomem_bench.bench.adapters.locomo import (
        DatasetError,
        stratified_sample,
        write_sample_manifest,
    )

    cfg = _config_from(args)
    try:
        adapter = make_adapter(args.dataset, cfg)
        adapter.verify()
        samples = stratified_sample(
            adapter.load(),
            per_category=cfg.benchmark.sample.per_category,
            categories=cfg.benchmark.sample.categories,
            seed=cfg.benchmark.sample.seed,
        )
        manifest_path = Path(cfg.benchmark.sample.manifest)
        write_sample_manifest(samples, adapter=adapter, cfg=cfg, path=manifest_path)
    except (DatasetError, ValueError) as exc:
        print(f"gluomem bench: {exc}", file=sys.stderr)
        return 1
    counts = adapter.eligible_counts()
    print(f"eligible per category: {counts}")
    print(f"wrote {len(samples)}-sample manifest -> {manifest_path}")
    return 0


def _conv_ids_from_manifest(path: str | None) -> set[str] | None:
    if path is None:
        return None
    import json
    from pathlib import Path

    try:
        ids = json.loads(Path(path).read_text(encoding="utf-8"))["ids"]
        return {i.split(":qa:")[0] for i in ids}
    except OSError as exc:
        raise ValueError(f"cannot read conv-ids manifest {path}: {exc}") from exc
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(
            f"invalid conv-ids manifest {path} (need a JSON object with an "
            f"'ids' list of strings): {exc!r}"
        ) from exc


def _cmd_bench_ingest(args: argparse.Namespace) -> int:
    from pathlib import Path

    from gluomem_bench.bench.adapters.locomo import DatasetError

    cfg = _config_from(args)
    try:
        conv_ids = _conv_ids_from_manifest(args.conv_ids_from)
    except ValueError as exc:
        print(f"gluomem bench: {exc}", file=sys.stderr)
        return 1
    if args.system == "mem0":
        from gluomem_bench.bench.mem0 import ingest_mem0_per_conversation

        try:
            mem0_reports = ingest_mem0_per_conversation(
                cfg,
                dataset=args.dataset,
                db_dir=Path(args.db_dir),
                skip_existing=args.skip_existing,
                conv_ids=conv_ids,
            )
        except (DatasetError, ValueError, RuntimeError) as exc:
            print(f"gluomem bench: {exc}", file=sys.stderr)
            return 1
        print(f"ingested {len(mem0_reports)} conversation(s) into {args.db_dir} (mem0)")
        return 0

    from gluomem.llm import make_extractors

    from gluomem_bench.bench.runner import ingest_per_conversation

    facade = make_extractors(cfg)
    try:
        reports = ingest_per_conversation(
            cfg,
            dataset=args.dataset,
            db_dir=Path(args.db_dir),
            facade=facade,
            skip_existing=args.skip_existing,
            conv_ids=conv_ids,
        )
    except (DatasetError, ValueError) as exc:
        print(f"gluomem bench: {exc}", file=sys.stderr)
        return 1
    total_errors = [e for r in reports for e in r.extraction_errors]
    print(f"ingested {len(reports)} conversation(s) into {args.db_dir}")
    for err in total_errors:
        print(f"extraction error: {err}", file=sys.stderr)
    return 0


def _cmd_bench_verify(args: argparse.Namespace) -> int:
    from pathlib import Path

    from gluomem_bench.bench.adapters.locomo import DatasetError
    from gluomem_bench.bench.runner import verify_ingest_completeness

    cfg = _config_from(args)
    try:
        rows = verify_ingest_completeness(
            cfg,
            dataset=args.dataset,
            db_dir=Path(args.db_dir),
            write_sentinels=args.write_sentinels,
        )
    except (DatasetError, ValueError) as exc:
        print(f"gluomem bench: {exc}", file=sys.stderr)
        return 1
    incomplete = [r for r in rows if r["status"] == "incomplete"]
    complete = sum(1 for r in rows if r["status"] == "complete")
    absent = sum(1 for r in rows if r["status"] == "absent")
    for r in incomplete:
        print(
            f"INCOMPLETE {r['conv_id']}: {r['found_sessions']}/{r['expected_sessions']} sessions",
            file=sys.stderr,
        )
    print(
        f"verify {args.db_dir}: {complete} complete, {len(incomplete)} incomplete, {absent} absent"
    )
    if incomplete:
        print(
            "delete the incomplete .mind files and re-ingest with --skip-existing",
            file=sys.stderr,
        )
        return 1
    return 0


def _cmd_bench_replay(args: argparse.Namespace) -> int:
    from gluomem.embedding.providers import make_embedder

    from gluomem_bench.bench.registry import RegistryError
    from gluomem_bench.bench.replay import (
        ReplayFixtureError,
        ReplayNotReadyError,
        run_replay_fixture,
    )

    cfg = _config_from(args)
    try:
        receipt, digest = run_replay_fixture(
            cfg,
            benchmark_id=args.dataset,
            embedder=make_embedder(cfg),
            work_dir=args.work_dir,
            out_dir=args.out,
            candidate=args.candidate,
        )
    except (RegistryError, ReplayFixtureError, ReplayNotReadyError, FileExistsError) as exc:
        print(f"gluomem bench: {exc}", file=sys.stderr)
        return 1
    print(receipt["label"])
    print(f"receipt {args.out}/receipt.json sha256 {digest}")
    return 0


def _cmd_bench_registry(args: argparse.Namespace) -> int:  # noqa: ARG001
    import json

    from gluomem_bench.bench.registry import (
        REGISTRY,
        manifest_record,
        overlap_report,
        registry_digest,
    )

    print(
        json.dumps(
            {
                "digest": registry_digest(),
                "entries": [manifest_record(e) for _, e in sorted(REGISTRY.items())],
                "overlaps": overlap_report(REGISTRY),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _add_bench_parsers(sub: argparse._SubParsersAction) -> None:
    bench = sub.add_parser("bench", help="run / rescore benchmark evaluations")
    bench_sub = bench.add_subparsers(dest="bench_command")

    run = bench_sub.add_parser("run", help="run a benchmark end-to-end")
    run.add_argument("--dataset", required=True, help="dataset name (locomo)")
    run.add_argument(
        "--retriever",
        required=True,
        choices=["gluomem", "full_context", "vector_rag", "bm25", "mem0"],
    )
    run.add_argument("--db-dir", default=None, help="per-conversation .mind dir")
    run.add_argument("--out", default=None, help="run-dir root (default: config)")
    run.add_argument("--limit", type=int, default=None, help="truncate sample")
    run.add_argument("--gate", action="store_true", help="apply the CI gate to this run")
    _add_config_args(run)

    rescore = bench_sub.add_parser(
        "rescore", help="re-derive scores/report offline (no model calls)"
    )
    rescore.add_argument("run_dir", help="a bench run directory")
    rescore.add_argument(
        "--check",
        action="store_true",
        help="verify committed artifacts instead of rewriting",
    )

    gate = bench_sub.add_parser("gate", help="apply the CI gate to a run dir")
    gate.add_argument("run_dir", help="a bench run directory")
    _add_config_args(gate)

    fetch = bench_sub.add_parser("fetch", help="download + SHA-verify a dataset")
    fetch.add_argument("--dataset", required=True)
    fetch.add_argument("--force", action="store_true", help="re-download")
    _add_config_args(fetch)

    sample = bench_sub.add_parser("sample", help="write the committed stratified sample manifest")
    sample.add_argument("--dataset", required=True)
    _add_config_args(sample)

    ingest = bench_sub.add_parser(
        "ingest", help="ingest a dataset into per-conversation memory stores"
    )
    ingest.add_argument("--dataset", required=True)
    ingest.add_argument("--db-dir", required=True)
    ingest.add_argument(
        "--system",
        choices=["gluomem", "mem0"],
        default="gluomem",
        help="memory system to ingest into (mem0 needs `uv sync --extra mem0`)",
    )
    ingest.add_argument(
        "--conv-ids-from",
        default=None,
        help=(
            "sample-manifest JSON; restrict ingest to the conversations its "
            "ids reference (LongMemEval: id == conv_id; LoCoMo: id prefix)"
        ),
    )
    ingest.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip conversations already ingested (resume after interruption)",
    )
    _add_config_args(ingest)

    verify = bench_sub.add_parser(
        "verify",
        help="audit a per-conversation .mind dir for silently-partial ingests",
    )
    verify.add_argument("--dataset", required=True)
    verify.add_argument("--db-dir", required=True)
    verify.add_argument(
        "--write-sentinels",
        action="store_true",
        help=(
            "write the completion sentinel for verified-complete conversations "
            "(bless a pre-sentinel ingest dir for --skip-existing resume)"
        ),
    )
    _add_config_args(verify)

    replay = bench_sub.add_parser(
        "replay",
        help="replay a registered fixture in event order and write a receipt (offline)",
    )
    replay.add_argument("--dataset", default="mab_fixture", help="registry fixture id")
    replay.add_argument("--work-dir", required=True, help="fresh dir for per-history stores")
    replay.add_argument("--out", required=True, help="receipt output dir")
    replay.add_argument("--candidate", default="baseline", help="candidate label")
    _add_config_args(replay)

    bench_sub.add_parser(
        "registry", help="print the benchmark registry, digests and source overlaps"
    )


def _resolve_tune_cache(args: argparse.Namespace, cfg: AppConfig):  # noqa: ANN202
    from pathlib import Path

    from gluomem_bench.tune.candidate_cache import CandidateCache, build_cache_for_benchmark

    cache_dir = Path(args.cache) if getattr(args, "cache", None) else None
    rebuild = bool(getattr(args, "rebuild_cache", False))
    if cache_dir is not None and (cache_dir / "manifest.json").is_file() and not rebuild:
        return CandidateCache.load(cache_dir)
    if getattr(args, "db_dir", None):
        cache = build_cache_for_benchmark(
            cfg,
            db_dir=args.db_dir,
            dataset_name=getattr(args, "dataset", "locomo") or "locomo",
            sample_manifest=getattr(args, "sample_manifest", None),
            lane_fetch=getattr(args, "lane_fetch", None),
            limit=int(getattr(args, "limit", 10)),
        )
        if cache_dir is not None:
            cache.save(cache_dir)
        return cache
    if cache_dir is not None:
        return CandidateCache.load(cache_dir)
    raise ValueError(
        "provide --cache <dir> (existing candidate cache) or --db-dir <dir> to build one"
    )


def _cmd_tune(args: argparse.Namespace) -> int:
    from gluomem_bench import extras
    from gluomem_bench.tune.objective import TuneConfig, run_optuna_tune

    extras.require("tune", "Offline retrieval parameter tuning")
    cfg = _config_from(args)
    try:
        cache = _resolve_tune_cache(args, cfg)
        tune_cfg = TuneConfig(
            n_trials=int(args.trials),
            k=int(args.k),
            metric=str(args.metric),
            profile=str(args.profile or cfg.search.default_profile),
            seed=int(args.seed),
            held_out_fraction=float(args.held_out_fraction),
        )
        result = run_optuna_tune(
            cache,
            cfg,
            tune_cfg=tune_cfg,
            out_dir=args.out,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"gluomem tune: {exc}", file=sys.stderr)
        return 1

    print(
        f"tuned {result.n_trials} trial(s) on {len(cache.query_ids())} query(s): "
        f"incumbent {result.metric}@{result.k}={result.incumbent_score:.4f} -> "
        f"best={result.best_score:.4f} (+{result.improvement:.4f}, "
        f"trial #{result.best_trial_number}, model_calls=0)"
    )
    return 0


def _cmd_ablate(args: argparse.Namespace) -> int:
    import json

    from gluomem_bench.tune.ablate import format_ablation_table, run_ablation

    cfg = _config_from(args)
    try:
        cache = _resolve_tune_cache(args, cfg)
        report = run_ablation(
            cache,
            cfg,
            profile=args.profile,
            k=int(args.k),
            out_dir=args.out,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"gluomem ablate: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_ablation_table(report))
    return 0


def _add_tune_and_ablate_parsers(sub: argparse._SubParsersAction) -> None:
    tune = sub.add_parser(
        "tune", help="offline Tier-1 retrieval parameter tuning (seeded Optuna TPE)"
    )
    tune.add_argument("--cache", default=None, help="candidate cache directory")
    tune.add_argument("--db-dir", default=None, help="per-conversation .mind dir to build cache")
    tune.add_argument("--dataset", default="locomo", help="dataset name when building cache")
    tune.add_argument("--sample-manifest", default=None, help="optional sample manifest JSON")
    tune.add_argument(
        "--lane-fetch",
        type=int,
        default=None,
        help="per-lane pool (default: limit × multiplier)",
    )
    tune.add_argument("--rebuild-cache", action="store_true", help="force cache rebuild")
    tune.add_argument("--out", default=None, help="output dir for manifest/trials/best_params")
    tune.add_argument("--trials", type=int, default=200, help="number of Optuna trials")
    tune.add_argument("--k", type=int, default=10, help="cutoff k for metric@k")
    tune.add_argument(
        "--metric",
        choices=["ndcg", "recall", "hit"],
        default="ndcg",
        help="optimization objective metric",
    )
    tune.add_argument("--profile", default=None, help="scoring profile to tune")
    tune.add_argument("--seed", type=int, default=42, help="deterministic RNG seed")
    tune.add_argument(
        "--held-out-fraction", type=float, default=0.3, help="held-out query fraction"
    )
    tune.add_argument("--limit", type=int, default=10, help="cache retrieval limit")
    _add_config_args(tune)

    ablate = sub.add_parser(
        "ablate", help="run deterministic retrieval signal and lane ablations over a cache"
    )
    ablate.add_argument("--cache", default=None, help="candidate cache directory")
    ablate.add_argument("--db-dir", default=None, help="per-conversation .mind dir to build cache")
    ablate.add_argument("--dataset", default="locomo", help="dataset name when building cache")
    ablate.add_argument("--sample-manifest", default=None, help="optional sample manifest JSON")
    ablate.add_argument(
        "--lane-fetch",
        type=int,
        default=None,
        help="per-lane pool (default: limit × multiplier)",
    )
    ablate.add_argument("--limit", type=int, default=10, help="cache retrieval limit")
    ablate.add_argument("--rebuild-cache", action="store_true", help="force cache rebuild")
    ablate.add_argument("--out", default=None, help="output dir for ablation_report.json")
    ablate.add_argument("--k", type=int, default=10, help="cutoff k for metrics@k")
    ablate.add_argument("--profile", default=None, help="scoring profile to ablate")
    ablate.add_argument("--json", action="store_true", help="emit JSON instead of table")
    _add_config_args(ablate)


_BENCH_HANDLERS = {
    "replay": _cmd_bench_replay,
    "registry": _cmd_bench_registry,
    "run": _cmd_bench_run,
    "rescore": _cmd_bench_rescore,
    "gate": _cmd_bench_gate,
    "fetch": _cmd_bench_fetch,
    "sample": _cmd_bench_sample,
    "ingest": _cmd_bench_ingest,
    "verify": _cmd_bench_verify,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gluomem-bench",
        description="GluoMem — benchmark harness and offline tuner.",
    )
    sub = parser.add_subparsers(dest="command")

    ingest = sub.add_parser("ingest", help="ingest a LoCoMo-format benchmark file into memory")
    ingest.add_argument("path", help="path to a LoCoMo-format JSON file")
    ingest.add_argument("--db", default=None, help="mind db path (default: config)")
    _add_config_args(ingest)

    _add_bench_parsers(sub)
    from gluomem_bench.bench.beam_cli import add_parsers

    add_parsers(sub)

    components = sub.add_parser(
        "components",
        help="print effective component implementations and readiness (no secrets)",
    )
    components.add_argument("--db", default=None, help="also check this mind db's fingerprint")
    components.add_argument(
        "--require-ready",
        action="store_true",
        help="exit 3 unless every component is real and available",
    )
    _add_config_args(components)

    _add_tune_and_ablate_parsers(sub)

    args = parser.parse_args(argv)
    from gluomem.components import FakeComponentError, ProviderUnavailableError
    from gluomem.embedding.compat import EmbeddingMismatchError
    from gluomem.extras import MissingExtraError

    try:
        return _dispatch(args, parser)
    except (
        MissingExtraError,
        ProviderUnavailableError,
        FakeComponentError,
        EmbeddingMismatchError,
    ) as exc:
        print(f"gluomem: {exc}", file=sys.stderr)
        return 2


def _dispatch(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "beam":
        from gluomem_bench.bench.beam_cli import dispatch

        return dispatch(args)
    if args.command == "ingest":
        return _cmd_ingest(args)
    if args.command == "components":
        return _cmd_components(args)
    if args.command == "tune":
        return _cmd_tune(args)
    if args.command == "ablate":
        return _cmd_ablate(args)
    if args.command == "bench":
        handler = _BENCH_HANDLERS.get(args.bench_command or "")
        if handler is None:
            print(
                f"gluomem bench: pick a subcommand ({', '.join(sorted(_BENCH_HANDLERS))})",
                file=sys.stderr,
            )
            return 1
        return handler(args)
    print(f"gluomem: '{args.command}' is not implemented yet.", file=sys.stderr)
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
