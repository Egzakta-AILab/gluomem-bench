"""Pinned Cognee graph context for a common external reader, never native answers.

This is a separate condition from the frozen CHUNKS producer. Context is one
native assembled retrieval unit; top_k counts native triplets, NOT evidence
paragraphs. Native formatting is retained, never re-ranked or locally rebuilt.
Real-model qualification is intentionally not run by import or by unit tests.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

if __package__:
    from gluomem_bench.bench import comparison_backends as _base
else:
    # Absolute-file -I entry: load ONLY this inspected stdlib-only sibling.
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "_cognee_graph_base", Path(__file__).with_name("comparison_backends.py")
    )
    _base = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _base
    _spec.loader.exec_module(_base)
Evidence = _base.Evidence
CogneeEvidenceBackend = _base.CogneeEvidenceBackend
_search_limit = _base._search_limit


class CogneeGraphEvidenceBackend(CogneeEvidenceBackend):
    """Native add/cognify, then scoped GraphCompletionRetriever context only.

    Supply CogneeGenerationBridge.bind(settings), keeping that SAME bridge/budget
    alive through ingestion, graph retrieval and the caller's common Qwen reader.
    No query-time native LLM helper is disabled or substituted.
    """

    _worker_file = "cognee_graph.py"
    _worker_flag = "--graph-worker"

    async def source_pins(self):
        """Read actual isolated native source hashes (also enforced at init)."""
        if self.closed:
            raise RuntimeError("backend is closed")
        return await self._rpc("source_pins")

    async def search(self, query, *, top_k=None, limit=None):
        if self.closed:
            raise RuntimeError("backend is closed")
        top_k = _search_limit(top_k, limit)
        if top_k == 0 or not query.strip():
            return []
        return graph_context_evidence(
            await self._rpc("search", query=query, top_k=top_k),
            dataset=self.settings.dataset,
        )


GRAPH_CONDITION = "cognee-1.6.5-native-graph-context-only-v1"
# Hashes of the inspected files shipped in cognee==1.6.5, not a local graph implementation.
NATIVE_SOURCE_PINS = {
    "modules/retrieval/graph_completion_retriever.py": (
        "f02aa6990dbafdc3cf8446b042a1cb5b9bb923afca41e1c3efe1f0785abcec99"
    ),
    "modules/retrieval/utils/brute_force_triplet_search.py": (
        "cd7b92eded8561c06a59f436a5afc42558167275b104f363184f2b73bfa157e6"
    ),
    "modules/graph/utils/resolve_edges_to_text.py": (
        "a37c8593cfb5f077cf6a652ac73cf1aa3b8efdd83dabbc06dca33388f89ae1e9"
    ),
    "modules/retrieval/utils/evidence.py": (
        "3b792ddf82b20aafe01257ad516f85bae31ca81ba63fa362fd1a8f4ccda70cad"
    ),
    "context_global_variables.py": (
        "5dea856133d2850065f53fede90ac7defe256b5a7d853a7ef851c1f046656b4d"
    ),
    "modules/data/methods/get_authorized_existing_datasets.py": (
        "3504feadd4f0a482514d6ad94cd380bdf8aa7cad48ded5130e8f5df4d57329c7"
    ),
    "infrastructure/llm/structured_output_framework/litellm_native/native_adapter.py": (
        "45a110ebf798a2a06eb180caa45824c41a1411ad0b1af4fb18178ef97298d637"
    ),
}


def _native_source_pins():
    import importlib.metadata

    dist = importlib.metadata.distribution("cognee")
    if dist.version != _base.COGNEE_VERSION:
        raise RuntimeError("requires cognee==1.6.5")
    actual = {
        name: hashlib.sha256(Path(dist.locate_file("cognee/" + name)).read_bytes()).hexdigest()
        for name in NATIVE_SOURCE_PINS
    }
    if actual != NATIVE_SOURCE_PINS:
        raise RuntimeError("Cognee native source pin mismatch")
    return {
        "version": dist.version,
        "wheel_sha256": _base.COGNEE_WHEEL_SHA256,
        "source_sha256": actual,
    }


async def _native_graph_context(settings, query, top_k, embedding_config):
    from cognee.context_global_variables import (
        set_database_global_context_variables,
        set_session_user_context_variable,
    )
    from cognee.modules.data.methods import get_authorized_existing_datasets
    from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
    from cognee.modules.retrieval.utils.evidence import graph_source_evidence
    from cognee.modules.users.methods import get_default_user

    user = await get_default_user()
    await set_session_user_context_variable(user)
    datasets = await get_authorized_existing_datasets(
        [settings["dataset"]], "read", user, strict=True
    )
    if len(datasets) != 1 or datasets[0].name != settings["dataset"]:
        raise ValueError("exact native dataset scope required")
    dataset = datasets[0]
    async with set_database_global_context_variables(
        dataset.id,
        dataset.owner_id,
        embedding_config=embedding_config,
    ):
        retriever = GraphCompletionRetriever(top_k=top_k)
        objects = await retriever.get_retrieved_objects(query=query)
        context = await retriever.get_context_from_objects(query=query, retrieved_objects=objects)
        refs = retriever.get_context_evidence(objects, dataset_id=dataset.id)
        refs.extend(await graph_source_evidence(refs, dataset.id))
        return {
            "dataset_id": str(dataset.id),
            "dataset_name": dataset.name,
            "context": context,
            "evidence": [ref.model_dump(mode="json") for ref in refs],
        }


async def _graph_worker():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_cognee_native_boundary", Path(__file__).with_name("cognee_native_boundary.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    boundary = module.NativeBoundary(Path.cwd() / "native-diagnostics.json")
    boundary.install(network=True)
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    # All Python/C-level native output is non-durable; pre-format categorical
    # observers above retain warning/error admission without plaintext logging.
    with open(os.devnull, "w") as discard:
        os.dup2(discard.fileno(), sys.stdout.fileno())
        os.dup2(discard.fileno(), sys.stderr.fileno())
    native = settings = pins = None
    for line in sys.stdin:
        stage = "unknown"
        try:
            request = json.loads(line)
            operation = request["operation"]
            result = None
            if operation == "init":
                stage = "init"
                settings = request["settings"]
                boundary.approve_proxies(settings)
                pins = _native_source_pins()
                _base._native_environment(settings)
                os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "true"
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                # Native preflight/dataset setup can construct an adapter before
                # per-call overrides; its default role must be document too.
                os.environ["EMBEDDING_INPUT_TYPE"] = "document"
                # Native 1.6.5's LiteLLM-native adapter forwards llm_args, not
                # its max_completion_tokens attribute; bind the actual wire cap.
                args = json.loads(os.environ["LLM_ARGS"])
                args["max_tokens"] = settings["max_completion_tokens"]
                os.environ["LLM_ARGS"] = json.dumps(args)
                import cognee

                native = cognee
                result = pins
            elif operation == "source_pins":
                stage = "source_pins"
                result = _native_source_pins()
            elif operation == "close":
                protocol.write(json.dumps({"ok": True, "result": None}) + "\n")
                break
            else:
                if native is None or settings is None:
                    raise RuntimeError("worker is not initialized")
                from cognee.infrastructure.databases.vector.create_vector_engine import (
                    _create_vector_engine,
                )
                from cognee.infrastructure.databases.vector.embeddings.config import EmbeddingConfig

                role = "query" if operation == "search" else "document"
                embedding = EmbeddingConfig(**_base._embedding_kwargs(settings, role))
                _base._refresh_embedding_role(_create_vector_engine)
                if operation == "add":
                    stage = "add"
                    with (Path(settings["root"]) / "inputs.jsonl").open(
                        "a", encoding="utf-8"
                    ) as stream:
                        for chunk in request["chunks"]:
                            stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    await native.add(
                        data=[chunk["text"] for chunk in request["chunks"]],
                        dataset_name=settings["dataset"],
                        run_in_background=False,
                        embedding_config=embedding,
                    )
                    stage = "cognify"
                    await native.cognify(
                        datasets=[settings["dataset"]],
                        chunk_size=settings["chunk_size"],
                        chunks_per_batch=1,
                        data_per_batch=1,
                        run_in_background=False,
                        raise_on_error=True,
                        extractor="llm",
                        embedding_config=embedding,
                    )
                elif operation == "search":
                    stage = "search"
                    result = await _native_graph_context(
                        settings, request["query"], request["top_k"], embedding
                    )
                else:
                    raise ValueError("unknown graph worker operation")
            stage = "diagnostic_admission"
            boundary.check()
            boundary.startup = False
            protocol.write(json.dumps({"ok": True, "result": result}, allow_nan=False) + "\n")
        except Exception as exc:
            diagnostic = (
                module.clean_diagnostic(exc.native_diagnostic)
                if hasattr(exc, "native_diagnostic")
                else module.native_diagnostic(exc, stage)
            )
            try:
                boundary.capture("native_failure", exc, diagnostic=diagnostic)
                boundary.save()
            except Exception:
                pass  # Receipt failure must not replace the initiating RPC diagnostic.
            protocol.write(
                json.dumps({"ok": False, "error_type": "native_failure", "diagnostic": diagnostic})
                + "\n"
            )
            break


def verify_graph_evidence(root, *, dataset, expected, bridge, quiescent=False):
    """Reuse committed coverage + exact generation gate, never promote quality.

    Run before common-reader calls and again after close (quiescent=True).
    Nested coverage is the unchanged historical verifier receipt; its CHUNKS
    condition label is NOT a statement about this adapter's retrieval algorithm.
    Real shared-encoder/Qwen scope qualification remains a separate admission.
    """
    from gluomem_bench.bench.cognee_campaign import verify_native_evidence

    verified = verify_native_evidence(
        root,
        dataset=dataset,
        expected=expected,
        bridge=bridge,
        quiescent=quiescent,
    )
    return {
        **verified,
        "condition": GRAPH_CONDITION,
        "graph_retrieval_qualified": False,
        "complete_campaign_admission": False,
        "remaining_qualification": (
            "Real shared-encoder roles/parity, scoped synthetic Qwen graph retrieval, "
            "independent protocol review."
        ),
    }


def graph_context_evidence(payload, *, dataset):
    """Map source-only RPC output to one deterministic common-reader Evidence."""
    if payload.get("completion") is not None:
        raise ValueError("native completion must never be evidence")
    if payload.get("dataset_name") != dataset or not payload.get("dataset_id"):
        raise ValueError("native dataset scope mismatch")
    context = payload["context"]
    if not context:
        return []
    if not any(ref.get("kind") == "graph_edge" for ref in payload["evidence"]):
        raise ValueError("nonempty native graph context lacks graph-edge lineage")
    text = (
        context
        if isinstance(context, str)
        else json.dumps(
            context, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    )
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return [
        Evidence(
            "cognee-graph:" + str(payload["dataset_id"]) + ":" + digest,
            text,
            {
                "backend": "cognee",
                "condition": GRAPH_CONDITION,
                "dataset_id": str(payload["dataset_id"]),
                "dataset_name": dataset,
                "native_evidence": payload["evidence"],
                "score_kind": "unscored_native_context",
            },
        )
    ]


if __name__ == "__main__" and sys.argv[1:] == ["--graph-worker"]:
    asyncio.run(_graph_worker())
