"""Isolated BEAM commands; generic QA scorer intentionally not reused."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

from gluomem_bench.bench.beam_eval import prepare_beam, rescore_beam, run_beam
from gluomem_bench.bench.rack import RackClient, RackProfile, controlled_config


def add_parsers(sub):
    parser = sub.add_parser("beam", help="pinned BEAM 128K adapted rubric evaluation")
    commands = parser.add_subparsers(dest="beam_command", required=True)
    commands.add_parser("profile", help="print the fail-closed rack-only profile")
    prepare = commands.add_parser("prepare", help="verify official parquet and export JSON")
    prepare.add_argument("--parquet", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    rescore = commands.add_parser("rescore", help="offline artifact verification and rescore")
    rescore.add_argument("run_dir", type=Path)
    rescore.add_argument("--check", action="store_true")
    run = commands.add_parser("run", help="sequential rack-only evaluation")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--sha256", required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--profile", type=Path, required=True)
    run.add_argument("--limit", type=int, default=1, help="explicitly set for larger campaigns")
    run.add_argument("--retriever", choices=["full_context", "bm25", "gluomem"], default="bm25")
    run.add_argument("--db-dir", type=Path)
    ingest = commands.add_parser("ingest", help="production SourceEvent extraction/consolidation")
    ingest.add_argument("--data", type=Path, required=True)
    ingest.add_argument("--sha256", required=True)
    ingest.add_argument("--db-dir", type=Path, required=True)
    ingest.add_argument("--profile", type=Path, required=True)
    ingest.add_argument(
        "--conv-id",
        action="append",
        required=True,
        help="explicit history IDs; repeat to select more",
    )


def dispatch(args: argparse.Namespace) -> int:
    try:
        if args.beam_command == "profile":
            result = RackProfile(
                base_url="https://rack.example.com/v1", model="example/served-model"
            ).model_dump()
        elif args.beam_command == "prepare":
            result = prepare_beam(args.parquet, args.out)
        elif args.beam_command == "rescore":
            result = rescore_beam(args.run_dir, check=args.check)
        else:
            profile = RackProfile.model_validate_json(args.profile.read_text(encoding="utf-8"))
            cfg = controlled_config(profile, data=args.data, sha256=args.sha256)
            from gluomem_bench.bench.rack_trace import RackRouter, WorkloadBudget

            budget = WorkloadBudget(
                max_calls=profile.max_generation_calls, max_seconds=profile.max_run_seconds
            )
            if args.beam_command == "ingest":
                from gluomem_bench.bench.beam_ingest import ingest_beam, select_fresh_beam_ingest

                select_fresh_beam_ingest(
                    cfg, args.db_dir, set(args.conv_id), include_shared_traces=True
                )
                from gluomem.llm.modules import DspyExtractors

                client = RackClient(profile, receipt=args.db_dir / "calls.jsonl", budget=budget)
                try:
                    client.preflight()
                    router = RackRouter(
                        cfg,
                        client=client,
                        trace=args.db_dir / "production-raw.jsonl",
                        budget=budget,
                    )
                    result = ingest_beam(
                        cfg,
                        db_dir=args.db_dir,
                        extractors=DspyExtractors(cfg, router=router),
                        conv_ids=set(args.conv_id),
                        completion_guard=budget.check,
                    )
                finally:
                    client.close()
            else:
                client = RackClient(profile, receipt=args.out / "calls.jsonl", budget=budget)
                try:
                    result = run_beam(
                        cfg,
                        client=client,
                        out=args.out,
                        db_dir=args.db_dir,
                        retriever=args.retriever,
                        limit=args.limit,
                    )
                finally:
                    client.close()
    except (ValueError, OSError, RuntimeError, httpx.HTTPError) as exc:
        print(f"BEAM: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
