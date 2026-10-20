"""Byte-approved native graph driver: explicit complete LME-500 or legacy slice.

The root result after separate post-exit verification is the sole completion
 authority. Complete LongMemEval is one condition, never an entire multi-dataset suite.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROTOCOL = "cognee-graph-lme-seven@v1"
FULL_PROTOCOL = "cognee-graph-lme-complete@v2"
SAMPLE_IDS = (
    "51a45a95",
    "6222b6eb",
    "b0479f84",
    "bb7c3b45",
    "gpt4_fa19884d",
    "0e4e4c46",
    "f4f1d8a4_abs",
)


def verify_reference(spec):
    reference = spec["reference_spec"]
    path = Path(reference["path"])
    if (
        not path.is_absolute()
        or hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]
    ):
        raise ValueError("reference specification bytes differ")
    base = json.loads(path.read_bytes())
    if spec.get("protocol") == FULL_PROTOCOL:
        receipt = spec.get("reference_approval", {})
        review_path = Path(receipt.get("path", ""))
        if (
            not review_path.is_absolute()
            or not review_path.is_file()
            or hashlib.sha256(review_path.read_bytes()).hexdigest() != receipt.get("sha256")
        ):
            raise ValueError("reference independent byte-bound approval required")
        review = json.loads(review_path.read_bytes())
        if (
            review.get("approved") is not True
            or review.get("reviewer_kind") not in {"human", "AI-assisted"}
            or not review.get("reviewer_id")
            or review["reviewer_id"] == spec.get("producer_id")
            or review.get("spec_sha256") != reference["sha256"]
        ):
            raise ValueError("reference independent byte-bound approval required")
    elif reference["sha256"] != "626a08d36733a658bb14047af4f31f6e0abcc7ce16f5c9e61a091fd85f0360c9":
        raise ValueError("reference specification bytes differ")
    return base


def verify_execution_spec(spec_path, approval_path, *, post_exit=False):
    material = Path(spec_path).read_bytes()
    spec = json.loads(material)
    approval = json.loads(Path(approval_path).read_bytes())
    if (
        spec.get("protocol") not in {PROTOCOL, FULL_PROTOCOL}
        or approval.get("approved") is not True
        or approval.get("reviewer_kind") not in {"human", "AI-assisted"}
        or not approval.get("reviewer_id")
        or not spec.get("producer_id")
        or approval.get("reviewer_id") == spec.get("producer_id")
        or approval.get("spec_sha256") != hashlib.sha256(material).hexdigest()
    ):
        raise ValueError("exact byte-bound independent execution approval required")
    files = spec.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("nonempty frozen producer inventory required")
    frozen = {}
    for item in files:
        path = Path(item["path"])
        key = str(path.resolve())
        if (
            not path.is_absolute()
            or key in frozen
            or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]
        ):
            raise ValueError("frozen inventory identity/hash differs")
        frozen[key] = item["sha256"]
    reference = spec["reference_spec"]
    ref_path = Path(reference["path"])
    base = verify_reference(spec)
    package = Path(__file__).resolve().parent
    required = {str(Path(item["path"]).resolve()): item["sha256"] for item in base["files"]}
    extras = [
        package / name
        for name in (
            "cognee_driver.py",
            "cognee_retriever.py",
            "cognee_graph.py",
            "cognee_campaign.py",
        )
    ]
    extras += [
        package.parents[1] / "tests/bench" / name
        for name in ("test_cognee_driver.py", "test_cognee_retriever.py")
    ]
    for path in extras:
        required[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    required[str(ref_path.resolve())] = reference["sha256"]
    if spec.get("protocol") == FULL_PROTOCOL:
        item = spec["reference_approval"]
        required[str(Path(item["path"]).resolve())] = item["sha256"]
    if any(frozen.get(key) != value for key, value in required.items()):
        raise ValueError("frozen producer inventory omits actual closure/reference bytes")
    validate_scope(spec, base, post_exit=post_exit)
    verify_qualifications(spec, approval)
    return spec


def verify_qualifications(spec, approval):
    """Independent evidence admission, not self-certification by an adapter label.

    Each qualification JSON has qualified=true, condition, evidence_files (each
    path/sha256) and the named assertions below. The independent approval binds
    approved_gates {gate: sha256}; ALL files also belong to the frozen inventory.
    """
    from gluomem_bench.bench.cognee_graph import GRAPH_CONDITION

    gates = {
        "real_qwen_graph_scope",
        "shared_encoder_roles_and_vector_parity",
        "independent_protocol_review",
    }
    records = spec.get("qualification_files", [])
    if len(records) != len(gates) or {r["gate"] for r in records} != gates:
        raise ValueError("real-model/encoder/protocol qualification evidence required")
    inventory = {str(Path(r["path"]).resolve()): r["sha256"] for r in spec["files"]}

    def checked(item):
        path = Path(item["path"])
        if (
            not path.is_absolute()
            or inventory.get(str(path.resolve())) != item["sha256"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]
        ):
            raise ValueError("qualification frozen evidence differs")
        return path

    for row in records:
        material = json.loads(checked(row).read_bytes())
        if (
            approval.get("approved_gates", {}).get(row["gate"]) != row["sha256"]
            or material.get("qualified") is not True
            or material.get("condition") != GRAPH_CONDITION
            or not material.get("evidence_files")
        ):
            raise ValueError("independent qualification approval/evidence missing")
        for file in material["evidence_files"]:
            checked(file)
        if row["gate"] == "real_qwen_graph_scope" and (
            material.get("synthetic_only") is not True
            or material.get("real_model") != spec["profile"]["model"]
            or material.get("native_graph_context") is not True
            or material.get("native_final_completion_used") is not False
        ):
            raise ValueError("real Qwen native graph qualification incomplete")
        if row["gate"] == "shared_encoder_roles_and_vector_parity" and (
            material.get("wire_roles_qualified") is not True
            or material.get("vector_parity_qualified") is not True
            or material.get("recipe") != spec["embedding_recipe"]
        ):
            raise ValueError("shared-encoder qualification incomplete")


def validate_scope(spec, base, *, post_exit=False):
    complete = spec.get("protocol") == FULL_PROTOCOL
    ids = spec.get("sample_ids", [])
    expected_count = 500 if complete else 7
    if (
        not isinstance(base.get("protocol"), str)
        or not base["protocol"]
        or not isinstance(ids, list)
        or len(ids) != expected_count
        or any(not isinstance(sid, str) or not sid for sid in ids)
        or len(set(ids)) != expected_count
        or ids != base.get("sample_ids")
        or (complete and spec.get("scope") != "complete-longmemeval-s")
        or (
            not complete
            and (tuple(ids) != SAMPLE_IDS or base["protocol"] != "local-qwen-development-pilot@v1")
        )
        or spec.get("evidence_token_budget") != 4096
    ):
        raise ValueError(
            "exact complete/legacy ordered scope and 4096 served-token ceiling required"
        )
    for key in (
        "profile",
        "dataset",
        "config_overrides",
        "embedding_recipe",
        "embedding_url",
        "judge_source",
    ):
        if spec.get(key) != base.get(key):
            raise ValueError("shared pinned reference configuration differs: " + key)
    root = Path(spec["output_root"])
    project = Path(__file__).resolve().parents[2]
    if (
        not root.is_absolute()
        or root.is_symlink()
        or any(root.resolve().is_relative_to(p) for p in (project, project.parent / "GluoMem"))
    ):
        raise ValueError("explicit external fresh output root required")
    if not post_exit and root.exists():
        raise FileExistsError("fresh root required")
    jobs = spec.get("jobs", [])
    if (
        len(jobs) != expected_count
        or [r["sample_id"] for r in jobs] != ids
        or len({r["job_id"] for r in jobs}) != expected_count
        or len({str(Path(r["output"]).resolve()) for r in jobs}) != expected_count
    ):
        raise ValueError("exact ordered unique graph matrix required")
    outputs = []
    for row in jobs:
        out = Path(row["output"])
        if (
            row.get("system") != "cognee"
            or row["call_budget"] != 1024
            or row["seconds_budget"] != 1200
            or row["history_id"] != "longmemeval_s:" + row["sample_id"]
            or not out.is_absolute()
            or not out.resolve().is_relative_to(root.resolve())
            or out.resolve() == root.resolve()
            or out.is_symlink()
        ):
            raise ValueError("native+reader+judge require one bounded 1024-call/1200-second job")
        outputs.append(out.resolve())
    if any(
        a.is_relative_to(b) or b.is_relative_to(a)
        for i, a in enumerate(outputs)
        for b in outputs[i + 1 :]
    ):
        raise ValueError("job roots must not overlap")
    policy = spec["campaign_policy"]
    if (
        policy["max_workers"] != 1
        or policy["max_in_flight"] != 1
        or not 1 <= policy["max_calls"] <= expected_count * 1024
        or not 1 <= policy["max_seconds"] <= expected_count * 1200
        or policy["stop_on_failure"] is not True
    ):
        raise ValueError("explicit sequential globally bounded campaign required")


def reconcile_job(
    *,
    out,
    documents,
    sample,
    profile,
    cfg,
    adapter,
    judge_source,
    expected_calls,
    require_real=True,
):
    """Read closed native DB, raw wire traces and common-reader input independently."""
    from dataclasses import asdict

    from gluomem_bench.bench.cognee_campaign import (
        verify_generation_evidence,
        verify_native_coverage,
    )
    from gluomem_bench.bench.cognee_graph import GRAPH_CONDITION, NATIVE_SOURCE_PINS
    from gluomem_bench.bench.comparison_backends import COGNEE_WHEEL_SHA256
    from gluomem_bench.bench.suite_campaign import verify_history

    out = Path(out)
    native_out, reader_out = out / "graph", out / "reader"
    ledger = json.loads((native_out / "native-ledger.json").read_bytes())
    wanted_pins = {
        "version": "1.6.5",
        "wheel_sha256": COGNEE_WHEEL_SHA256,
        "source_sha256": NATIVE_SOURCE_PINS,
    }
    if (
        ledger["status"] != "closed_verified"
        or ledger["condition"] != GRAPH_CONDITION
        or ledger["native_worker_closed"] is not True
        or ledger["proxy_closed"] is not True
        or ledger["source_pins"] != wanted_pins
        or ledger["source_inputs"] != [asdict(item) for item in documents]
    ):
        raise ValueError("closed native source/accounting evidence differs")
    actual_files = {
        p.relative_to(native_out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in native_out.rglob("*")
        if p.is_file() and p.name != "native-ledger.json"
    }
    if ledger["artifact_sha256"] != actual_files:
        raise ValueError("closed native artifact inventory/hash differs")
    coverage = ledger["verification"]["coverage"]
    native_coverage = verify_native_coverage(
        native_out / "native",
        dataset=ledger["verification"]["coverage"]["dataset_name"]
        if "dataset_name" in coverage
        else ledger["dataset"],
        expected=documents,
        quiescent=True,
    )
    coverage_fields = {
        key: value
        for key, value in native_coverage.items()
        if key not in {"condition", "graph_retrieval_qualified", "graph_retrieval_gap"}
    }
    if coverage_fields != coverage:
        raise ValueError("closed native processing coverage differs")
    generation = verify_generation_evidence(
        native_out / "generation.jsonl",
        profile=profile,
        expected_calls=ledger["generation_calls"],
        starting_calls=0,
    )
    if generation != ledger["generation"]:
        raise ValueError("native wire generation ledger differs")
    expected = {"dataset": adapter.name, "system": "cognee", "sample_ids": [sample.id]}
    summary = verify_history(
        reader_out, expected_job=expected, require_real=require_real, judge_source=judge_source
    )
    if json.loads((reader_out / "effective-config.json").read_bytes()) != cfg.model_dump(
        mode="json"
    ):
        raise ValueError("effective reader configuration differs")
    rows = json.loads((reader_out / "predictions.json").read_bytes())
    if rows[0]["sample"] != json.loads(json.dumps(asdict(sample))):
        raise ValueError("actual sample differs from pinned selected data")
    contexts = json.loads((native_out / "graph-contexts.json").read_bytes())
    readers = json.loads((reader_out / "reader-evidence.json").read_bytes())
    if len(contexts) != 1 or ledger["graph_context_receipts"] != 1 or len(readers) != 1:
        raise ValueError("native graph/common-reader context receipt coverage differs")
    context = contexts[0]
    if (
        context["sample_id"] != sample.id
        or context["query"] != sample.question
        or context["native_final_completion_used"] is not False
        or readers[0]["question"] != sample.reader_question
        or readers[0]["budget"] != 4096
        or readers[0]["original_text"] != "\n".join(h["text"] for h in context["hits"])
        or [d["text"] for d in rows[0]["prediction"]["retrieved"]]
        != [h["text"] for h in context["hits"]]
    ):
        raise ValueError("native graph evidence differs from actual common-reader evidence")
    for hit in context["hits"]:
        p = hit["provenance"]
        if (
            p["condition"] != GRAPH_CONDITION
            or p["dataset_name"] != ledger["dataset"]
            or str(p["dataset_id"]).replace("-", "") != str(coverage["dataset_id"]).replace("-", "")
            or not any(e["kind"] == "graph_edge" for e in p["native_evidence"])
        ):
            raise ValueError("native graph context lacks scoped graph-edge lineage")
    wire = [
        json.loads(line) for line in (reader_out / "calls.jsonl").read_bytes().splitlines() if line
    ]
    attempts = [r for r in wire if r["url"].endswith("/chat/completions")]
    reader_calls = json.loads((reader_out / "usage.json").read_bytes())["generation_calls"]
    if (
        len(attempts) != reader_calls
        or reader_calls != 2
        or expected_calls != generation["calls"] + reader_calls
    ):
        raise ValueError("ALL native+reader+judge attempts must reconcile with outer job")
    prompt = completion = 0
    for r in attempts:
        usage = r["response"]["usage"]
        if (
            r["status"] != 200
            or type(usage["prompt_tokens"]) is not int
            or type(usage["completion_tokens"]) is not int
            or min(usage["prompt_tokens"], usage["completion_tokens"]) < 0
        ):
            raise ValueError("actual reader/judge token usage unavailable")
        prompt += usage["prompt_tokens"]
        completion += usage["completion_tokens"]
    return {
        "quality_status": "verified_pending_post_exit",
        "generation_calls": expected_calls,
        "native_generation_calls": generation["calls"],
        "reader_judge_generation_calls": reader_calls,
        "prompt_tokens": generation["prompt_tokens"] + prompt,
        "completion_tokens": generation["completion_tokens"] + completion,
        "source_inputs_sha256": coverage["inputs_sha256"],
        "summary": summary,
        "reliability": {
            "failed_attempts": 0,
            "expected_questions": 1,
            "completed_questions": 1,
            "native_worker_closed": True,
            "proxy_closed": True,
        },
    }


def run_graph_job(
    *,
    job,
    context,
    cfg,
    adapter,
    sample,
    profile,
    judge_source,
    settings,
    transport=None,
    embedder=None,
    require_real=True,
):
    """Actual run_history common reader/scorer, not a second QA implementation."""
    from gluomem_bench.bench.cognee_retriever import (
        CogneeRetrieverFactory,
        save_json,
        session_documents,
    )
    from gluomem_bench.bench.suite_campaign import run_history

    factory = CogneeRetrieverFactory(
        settings=settings,
        profile=profile,
        outer_budget=context,
        out=job.output / "graph",
        transport=transport,
    )
    initiating = None
    try:
        try:
            run_history(
                cfg=cfg,
                adapter=adapter,
                samples=[sample],
                profile=profile,
                out=job.output / "reader",
                system="cognee",
                judge_source=judge_source,
                embedder=embedder,
                outer_budget=context,
                evidence_token_budget=4096,
                retriever_factory=factory,
                transport=transport,
            )
        except BaseException as exc:
            initiating = exc
            raise
        finally:
            try:
                factory.close()
            except BaseException:
                if initiating is None:
                    raise
        context.check()
        conv = next(c for c in adapter.conversations() if c.conv_id == sample.conv_id)
        result = reconcile_job(
            out=job.output,
            documents=session_documents(conv),
            sample=sample,
            profile=profile,
            cfg=cfg,
            adapter=adapter,
            judge_source=judge_source,
            expected_calls=context.calls,
            require_real=require_real,
        )
        context.check()
        save_json(job.output / "pipeline.json", result)
        return result
    except BaseException as exc:
        context.failed = True
        native_calls = factory.budget.calls if factory.budget is not None else None
        # None means not started, never an inferred zero for unavailable accounting.
        reader_calls = context.calls - native_calls if native_calls is not None else context.calls
        exc.__dict__.update(
            generation_calls=context.calls,
            native_generation_calls=native_calls,
            reader_judge_generation_calls=reader_calls,
        )
        if job.output.exists():
            failure = {
                "complete": False,
                "generation_calls": context.calls,
                "native_generation_calls": native_calls,
                "native_accounting_status": "recorded"
                if native_calls is not None
                else "not_started",
                "reader_judge_generation_calls": reader_calls,
                "error_type": "native_pipeline_failure",
                "completion_authority": "root-only",
            }
            try:
                from gluomem_bench.bench.cognee_native_boundary import clean_diagnostic
                failure['native_diagnostic'] = clean_diagnostic(
                    getattr(exc, 'native_diagnostic', None))
                save_json(job.output / "pipeline-failure.json", failure)
            except BaseException:
                exc.add_note("failure receipt unavailable; actual spend attached to exception")
        raise


def _save(path, value):
    """Atomic UTF-8 byte write; failed replacement does not publish success."""
    path = Path(path)
    pending = path.with_name(path.name + ".pending")
    material = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n"
    ).encode("utf-8")
    import os

    with pending.open("xb") as stream:
        stream.write(material)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def _artifact_hashes(root):
    excluded = {"campaign-result.json", "postexit-verification.json"}
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(root).rglob("*"))
        if p.is_file() and p.relative_to(root).as_posix() not in excluded
    }


def finalize_root(root, result, *, spec_sha256, protocol=PROTOCOL):
    import os

    record = {
        **result,
        "protocol": protocol,
        "complete": False,
        "status": "pending_post_exit_verification"
        if all(row["status"] == "complete" for row in result["jobs"])
        else "incomplete",
        "completion_authority": "root-post-exit-only",
        "producer_pid": os.getpid(),
        "spec_sha256": spec_sha256,
        "full_suite": {
            "status": "unrun",
            "other_datasets": "unrun",
            "other_systems": "unrun",
            "unselected_longmemeval": "unrun",
            "scope": "complete LongMemEval-S Cognee graph condition"
            if protocol == FULL_PROTOCOL
            else "seven selected Cognee graph cases only",
        },
    }
    try:
        record["artifact_sha256"] = _artifact_hashes(Path(root))
        _save(Path(root) / "campaign-result.json", record)
    except BaseException as exc:
        error = RuntimeError(
            "root result persistence failed; incomplete, coordination retains spend"
        )
        error.generation_calls = result["accounting"]["calls"]
        error.receipt = record
        raise error from exc
    return record


def assert_producer_exited(result):
    import os

    pid = result["producer_pid"]
    if type(pid) is not int or pid < 1:
        raise ValueError("actual producer PID required")
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            if ctypes.get_last_error() != 87:  # Invalid PID only; access denied remains blocked.
                raise RuntimeError("cannot establish producer has exited")
            return
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value == 259:
                raise RuntimeError("producer has not exited; post-exit verification refused")
        finally:
            kernel.CloseHandle(handle)
    else:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        raise RuntimeError("producer has not exited; post-exit verification refused")


def run_admitted(spec, *, cfg, adapter, profile, encoder, spec_sha256):
    """Run the frozen matrix with ONE JobContext for both native and reader ledgers."""
    from gluomem_bench.bench.comparison_backends import CogneeSettings
    from gluomem_bench.bench.parallel_campaign import CampaignPolicy, Job, run_jobs

    root = Path(spec["output_root"])
    canonical = {s.id: s for s in adapter.load()}
    jobs = [
        Job(r["job_id"], r["history_id"], Path(r["output"]), r["call_budget"], r["seconds_budget"])
        for r in spec["jobs"]
    ]
    records = {r["job_id"]: r for r in spec["jobs"]}

    def run(job, context):
        r = records[job.job_id]
        native = spec["native"]
        settings = CogneeSettings(
            python_executable=native["python_executable"],
            root=job.output / "graph/native",
            dataset="history-" + job.job_id,
            llm_url=profile.base_url,
            llm_model=profile.model,
            llm_api_key="guard-bound-placeholder",
            embedding_url=spec["embedding_url"],
            embedding_model=spec["embedding_recipe"]["model"],
            embedding_dimensions=spec["embedding_recipe"]["dim"],
            embedding_api_key="local-placeholder",
            max_completion_tokens=profile.output_budget("cognify"),
            chunk_size=native["chunk_size"],
            timeout_seconds=min(native["timeout_seconds"], context.remaining()),
        )
        return run_graph_job(
            job=job,
            context=context,
            cfg=cfg,
            adapter=adapter,
            sample=canonical[r["sample_id"]],
            profile=profile,
            judge_source=Path(spec["judge_source"]),
            settings=settings,
            embedder=encoder,
        )

    result = run_jobs(
        jobs,
        policy=CampaignPolicy(**spec["campaign_policy"]),
        out=root / "coordination",
        run_job=run,
    )
    return finalize_root(root, result, spec_sha256=spec_sha256, protocol=spec["protocol"])


def original_history(raw):
    """Common adapter projection, with opaque IDs only; no Cognee-only text."""
    from gluomem_bench.bench.adapters.longmemeval import LongMemEvalAdapter
    from gluomem_bench.bench.source_identifiers import ordinal_turn_ids

    # Delegate text stripping and empty/session filtering to the comparison seam.
    adapter = object.__new__(LongMemEvalAdapter)
    adapter._raw = [raw]
    conversation = adapter.conversations()[0]
    return ordinal_turn_ids(conversation)[0]


def select_inputs(adapter, sample_ids, *, complete=False):
    """Explicit ordered IDs, common allowed-history projection, actual input counts."""
    from collections import Counter
    from dataclasses import replace

    from gluomem_bench.bench.source_identifiers import ordinal_turn_ids

    ids = list(sample_ids)
    loaded = adapter.load()
    canonical = {s.id: s for s in loaded}
    if (
        not ids
        or len(set(ids)) != len(ids)
        or len(canonical) != len(loaded)
        or any(sid not in canonical for sid in ids)
    ):
        raise ValueError("unique explicit reference IDs required")
    if complete and (len(ids) != 500 or ids != [s.id for s in loaded]):
        raise ValueError("complete 500 IDs in pinned dataset/reference order required")
    samples = [canonical[sid] for sid in ids]
    if any(s.question_date is None for s in samples):
        raise ValueError("native question date required")
    wanted = {s.conv_id for s in samples}
    projected = {}
    for conv in adapter.conversations():
        if conv.conv_id in wanted:
            if conv.conv_id in projected:
                raise ValueError("duplicate source conversation")
            conv = ordinal_turn_ids(conv)[0]
            projected[conv.conv_id] = replace(
                conv, sessions=sorted(conv.sessions, key=lambda s: (s.date_time, s.index))
            )
    if set(projected) != wanted:
        raise ValueError("selected pinned histories incomplete")
    conversations = [projected[s.conv_id] for s in samples]
    roles = Counter()
    sessions = turns = chars = utf8_bytes = 0
    for conv in conversations:
        sessions += len(conv.sessions)
        for session in conv.sessions:
            turns += len(session.turns)
            for turn in session.turns:
                roles[turn.speaker] += 1
                chars += len(turn.text)
                utf8_bytes += len(turn.text.encode("utf-8"))
    counts = {
        "questions": len(samples),
        "histories": len(conversations),
        "sessions": sessions,
        "turns": turns,
        "documents": sessions,
        "roles": dict(sorted(roles.items())),
        "text_characters": chars,
        "text_utf8_bytes": utf8_bytes,
    }
    return samples, conversations, counts


def prepare_inputs(spec):
    """Read-only dataset/config/scorer admission before ANY client/store construction."""
    dataset = spec["dataset"]
    source = Path(dataset["path"])
    if (
        dataset["name"] != "longmemeval_s"
        or dataset["sha256"] != "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
        or hashlib.sha256(source.read_bytes()).hexdigest() != dataset["sha256"]
    ):
        raise ValueError("exact pinned LongMemEval-S dataset bytes required")
    from types import SimpleNamespace

    from gluomem_bench.bench.adapters.longmemeval import LongMemEvalAdapter
    from gluomem_bench.bench.rack import RackProfile, controlled_config
    from gluomem_bench.bench.upstream_scoring import load_longmemeval_prompt

    load_longmemeval_prompt(Path(spec["judge_source"]))
    profile = RackProfile.model_validate(spec["profile"])
    if profile.profile_version not in {"role-budgets-v5", "role-budgets-v6"}:
        raise ValueError("explicit matched role-budget Qwen profile required")
    cfg = controlled_config(profile, data=source, sha256=dataset["sha256"])
    cfg = cfg.with_overrides(spec["config_overrides"])
    if cfg.benchmark.answer.context_k != 50:
        raise ValueError("matched effective native graph triplet top_k=50 required")
    recipe = spec["embedding_recipe"]
    if (
        recipe["dim"] != 768
        or recipe["model"] != "google/embeddinggemma-2"
        or recipe["revision"] != "914f7f89142e33e77833254d9c9b90c3cef7303b"
        or recipe["max_tokens"] != 8192
        or recipe["precision"] != "bfloat16"
        or recipe["text_only"] is not True
    ):
        raise ValueError("pinned shared Gemma encoder recipe differs")
    if not Path(spec["native"]["python_executable"]).is_file():
        raise ValueError("isolated native Python executable required")
    if (
        type(spec["native"]["chunk_size"]) is not int
        or spec["native"]["chunk_size"] < 1
        or not 1 <= spec["native"]["timeout_seconds"] <= 1200
    ):
        raise ValueError("explicit native chunk/deadline limits required")
    adapter = LongMemEvalAdapter(cfg, dataset["name"])
    adapter.verify()
    complete = spec.get("protocol") == FULL_PROTOCOL
    selected, conversations, counts = select_inputs(adapter, spec["sample_ids"], complete=complete)
    if complete and spec.get("input_counts") != counts:
        raise ValueError("frozen actual complete input counts differ")
    active = SimpleNamespace(
        name=adapter.name,
        sha=adapter.sha,
        path=adapter.path,
        verify=adapter.verify,
        load=lambda: selected,
        conversations=lambda: conversations,
        input_counts=counts,
    )
    return cfg, active, profile


def verify_root_evidence(*, root, spec, cfg, adapter, profile, spec_sha256=None, require_real=True):
    root = Path(root)
    prior = json.loads((root / "campaign-result.json").read_bytes())
    assert_producer_exited(prior)
    try:
        return _verify_root_evidence(
            root=root,
            spec=spec,
            cfg=cfg,
            adapter=adapter,
            profile=profile,
            spec_sha256=spec_sha256,
            require_real=require_real,
        )
    except BaseException as exc:
        # Revoke a previously verified sentinel on any newly discovered corruption.
        failure = {
            **prior,
            "complete": False,
            "status": "failed_post_exit_verification",
            "verification_error_type": type(exc).__name__,
        }
        exc.__dict__.update(generation_calls=prior["accounting"]["calls"], receipt=failure)
        try:
            _save(root / "campaign-result.json", failure)
        except BaseException:
            exc.add_note("root failure persistence unavailable; retained spend attached")
        raise


def _verify_root_evidence(
    *, root, spec, cfg, adapter, profile, spec_sha256=None, require_real=True
):
    """Post-exit exact readback. A missing/failed root cannot be rescued by leaves."""
    from gluomem_bench.bench.cognee_retriever import session_documents
    from gluomem_bench.bench.parallel_campaign import _json_bytes

    root = Path(root)
    result = json.loads((root / "campaign-result.json").read_bytes())
    assert_producer_exited(result)
    if (
        result["protocol"] != spec.get("protocol", PROTOCOL)
        or result["completion_authority"] != "root-post-exit-only"
        or (spec_sha256 is not None and result["spec_sha256"] != spec_sha256)
    ):
        raise ValueError("root authority/spec identity differs")
    if result["artifact_sha256"] != _artifact_hashes(root):
        raise ValueError("post-exit root artifact inventory/hash differs")
    progress = json.loads((root / "coordination/progress.json").read_bytes())
    plan = json.loads((root / "coordination/plan.json").read_bytes())
    if (
        result["jobs"] != progress["jobs"]
        or result["accounting"] != progress["accounting"]
        or result["plan_sha256"] != hashlib.sha256(_json_bytes(plan)).hexdigest()
        or plan["policy"] != spec["campaign_policy"]
    ):
        raise ValueError("root/coordination plan and actual spend differ")
    expected_jobs = [
        {
            "job_id": r["job_id"],
            "history_id": r["history_id"],
            "output": str(Path(r["output"]).resolve()),
            "call_budget": r["call_budget"],
            "seconds_budget": r["seconds_budget"],
        }
        for r in spec["jobs"]
    ]
    if plan["jobs"] != expected_jobs or len(result["jobs"]) != len(expected_jobs):
        raise ValueError("post-exit job matrix differs")
    canonical = {s.id: s for s in adapter.load()}
    conversations = {c.conv_id: c for c in adapter.conversations()}
    total = 0
    verified = []
    for row, expected in zip(result["jobs"], spec["jobs"], strict=True):
        if row["status"] != "complete" or row["job_id"] != expected["job_id"]:
            raise ValueError("incomplete root; failed/unrun jobs remain unmeasured")
        sample = canonical[expected["sample_id"]]
        value = reconcile_job(
            out=Path(expected["output"]),
            documents=session_documents(conversations[sample.conv_id]),
            sample=sample,
            cfg=cfg,
            adapter=adapter,
            profile=profile,
            judge_source=Path(spec["judge_source"]),
            expected_calls=row["calls"],
            require_real=require_real,
        )
        if value != row["result"] or value != json.loads(
            (Path(expected["output"]) / "pipeline.json").read_bytes()
        ):
            raise ValueError("root and actual pipeline accounting differ")
        total += value["generation_calls"]
        verified.append({"job_id": row["job_id"], **value})
    if total != result["accounting"]["calls"] or result["artifact_sha256"] != _artifact_hashes(
        root
    ):
        raise ValueError("post-exit global spend/hash reconciliation differs")
    receipt = {
        "protocol": result["protocol"],
        "complete": True,
        "spec_sha256": result["spec_sha256"],
        "generation_calls": total,
        "jobs": verified,
        "artifact_sha256": result["artifact_sha256"],
    }
    # Receipt publication first; root publication last is the only completion authority.
    _save(root / "postexit-verification.json", receipt)
    complete = {
        **result,
        "complete": True,
        "status": "complete_verified_post_exit",
        "postexit_verification_sha256": hashlib.sha256(
            (root / "postexit-verification.json").read_bytes()
        ).hexdigest(),
    }
    try:
        _save(root / "campaign-result.json", complete)
    except BaseException as exc:
        error = RuntimeError("post-exit root publication failed; incomplete")
        error.__dict__.update(generation_calls=total, receipt=result)
        raise error from exc
    return complete


def verify_post_exit(spec_path, approval_path):
    spec = verify_execution_spec(spec_path, approval_path, post_exit=True)
    root = Path(spec["output_root"])
    result = json.loads((root / "campaign-result.json").read_bytes())
    assert_producer_exited(result)
    cfg, adapter, profile = prepare_inputs(spec)
    return verify_root_evidence(
        root=root,
        spec=spec,
        cfg=cfg,
        adapter=adapter,
        profile=profile,
        spec_sha256=hashlib.sha256(Path(spec_path).read_bytes()).hexdigest(),
    )


def main(argv=None):
    import sys

    try:
        return _main(argv)
    except Exception as exc:
        failure = {"complete": False, "error_type": type(exc).__name__}
        if "generation_calls" in exc.__dict__:
            failure["generation_calls"] = exc.__dict__["generation_calls"]
        print(json.dumps(failure), file=sys.stderr)
        raise SystemExit(1) from None


def _main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="read-only admission; no clients/stores")
    mode.add_argument(
        "--verify-post-exit", action="store_true", help="verify after producing process exits"
    )
    args = parser.parse_args(argv)
    if args.check:
        spec = verify_execution_spec(args.spec, args.approval)
        prepare_inputs(spec)
        print(json.dumps({"admission": "verified", "inference": "unrun", "full_suite": "unrun"}))
        return
    result = (
        verify_post_exit(args.spec, args.approval)
        if args.verify_post_exit
        else execute_cognee(args.spec, args.approval)
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "complete": result["complete"],
                "accounting": result["accounting"],
                "full_suite": "unrun",
            }
        )
    )


def execute_cognee(spec_path, approval_path):
    spec = verify_execution_spec(spec_path, approval_path)
    cfg, adapter, profile = prepare_inputs(spec)
    # All refusal gates above are read-only; the shared encoder remains lazy.
    from gluomem_bench.bench.embedding_runtime import EmbeddingRecipe, RemoteEmbedder

    encoder = RemoteEmbedder(
        spec["embedding_url"], recipe=EmbeddingRecipe(**spec["embedding_recipe"])
    )
    try:
        return run_admitted(
            spec,
            cfg=cfg,
            adapter=adapter,
            profile=profile,
            encoder=encoder,
            spec_sha256=hashlib.sha256(Path(spec_path).read_bytes()).hexdigest(),
        )
    finally:
        close = getattr(encoder, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
