"""Pinned MemoryArena Search. All QA and adapted building require byte-bound approval.

No ambient credentials, encoder substitution, generated-code execution, or implicit
benchmark admission. Offline native source loading executes only pinned definitions.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Optional

from gluomem_bench.bench.error_privacy import safe_error

from . import arena_campaign as shared
from .embedding_runtime import EmbeddingRecipe, RemoteEmbedder
from .rack import FatalRackError, RackClient, SharedRackError

PIN = shared.PIN
WEB = "env/env_systems/web_search_env/"
SOURCE_PINS = {
    "run_search.py": "f58ee0eb1e7ad209f251f11ee57949a9e6077e4296b04f1177479e89041ac431",
    "env/env_systems/base_env.py": (
        "3c7a2dcd6595f3ec39c06e1fefcaaf8903d0848fb17e8bdf56daaf35c46a6278"
    ),
    WEB + "search_agent/openai_client.py": (
        "c6c0b280b01c3ee6294eda2f2854bd752d5a106f6aef88380c39d9db956692c5"
    ),
    WEB
    + "search_agent/prompts.py": "cf974be9bd61d52919411e89a37776d6e8365ab22f42262b19fe0d6c4125f8a4",
    WEB
    + "search_agent/utils.py": "3a6d9bc4956deeafce4a4515c9d025e3bf1d3e1dbc97945413b84e87d2bc3e8f",
    WEB
    + "evaluate_with_openai.py": "eebc3d0bd8ab8d39e7240a9e60714e26789b391814d5f9c6309102088cf3a760",
    "env/env_systems/browsecomp_plus_env.py": (
        "d74c1eabf7d5555915c2d07606c4e51784527c56d460a3def4c73270b80a82f6"
    ),
    WEB + "searcher/searchers/openai_searcher.py": (
        "0de0263e9b60c5a29b5c6b81c37b0787df8e983c4f8b1d702749bcf9d39f0a12"
    ),
    WEB + "searcher/searchers/base.py": (
        "870e646f9dba8f2f99b1bbf596e6e8aa3ed9ba52127a60ffbc64b781bae815e0"
    ),
    WEB + "scripts_build_index/build_openai_embedding_index.py": (
        "c4dd386cf078cbd4159b23ae443dff0dcff0e42ebaa8c978c5781c3177728a64"
    ),
}
PROTOCOL = "memoryarena-search-full@v1"
SYSTEMS = ("no_memory", "bm25", "dense", "hybrid", "full_context", "gluomem")
DATA_SHA = "b445ee36fa3ccb9ad08eae9e7adda86bbc64f14f1e2a0682a8b2085cdb8e4c0e"
CORPUS_SHA = "a7d63ac2095e6bc21d6aa644fa5a9881a78a11634d4f4d2568dab3f98ab259f4"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_bytes(shared._json(value))


def read_json(path):
    return json.loads(Path(path).read_bytes())


def sources(root):
    result = {
        name: subprocess.check_output(["git", "-C", str(Path(root)), "show", PIN + ":" + name])
        for name in SOURCE_PINS
    }
    if any(hashlib.sha256(result[n]).hexdigest() != h for n, h in SOURCE_PINS.items()):
        raise ValueError("pinned Search source mismatch")
    return result


def _definitions(mod, raw, name, *, assignments=False, names=None):
    """Execute unchanged trusted native definitions, omitting import-time clients/CLI."""
    tree = ast.parse(raw, filename="pinned-git:" + PIN + "/" + name)
    tree.body = [
        n
        for n in tree.body
        if (isinstance(n, (ast.ClassDef, ast.FunctionDef)) and (names is None or n.name in names))
        or (assignments and isinstance(n, (ast.Assign, ast.AnnAssign)))
    ]
    exec(compile(tree, "pinned-git:" + PIN + "/" + name, "exec"), mod.__dict__)


def load_native(root):
    import copy
    from abc import ABC, abstractmethod

    import numpy as np
    import openai

    mod = ModuleType("_pinned_arena_search")
    mod.__dict__.update(
        json=json,
        re=re,
        Path=Path,
        Dict=dict,
        List=list,
        Any=Any,
        Optional=Optional,
        Tuple=tuple,
        copy=copy,
        np=np,
        openai=openai,
        ABC=ABC,
        abstractmethod=abstractmethod,
    )
    raw = sources(root)
    _definitions(
        mod, raw[WEB + "search_agent/prompts.py"], WEB + "search_agent/prompts.py", assignments=True
    )
    _definitions(mod, raw[WEB + "search_agent/utils.py"], WEB + "search_agent/utils.py")
    _definitions(mod, raw[WEB + "searcher/searchers/base.py"], WEB + "searcher/searchers/base.py")
    _definitions(
        mod,
        raw[WEB + "searcher/searchers/openai_searcher.py"],
        WEB + "searcher/searchers/openai_searcher.py",
    )
    import os
    from datetime import datetime

    mod.__dict__.update(os=os, datetime=datetime)
    _definitions(
        mod,
        raw[WEB + "search_agent/openai_client.py"],
        WEB + "search_agent/openai_client.py",
        names={
            "SearchToolHandler",
            "build_request",
            "run_conversation_with_tools",
            "_persist_response",
        },
    )
    _definitions(mod, raw[WEB + "evaluate_with_openai.py"], WEB + "evaluate_with_openai.py")
    _definitions(mod, raw["run_search.py"], "run_search.py", names={"_summarize_single_result"})
    import builtins
    import sys

    mod.__file__ = str(Path(root) / "env/env_systems/browsecomp_plus_env.py")
    real_import = builtins.__import__

    def native_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "evaluate_with_openai":
            return mod
        if name == "agent.search":
            return SimpleNamespace(run_query_with_agent_and_memory=mod.query_runner)
        if name == "sys":
            return SimpleNamespace(path=list(sys.path))
        return real_import(name, globals, locals, fromlist, level)

    mod.__dict__["__builtins__"] = {**vars(builtins), "__import__": native_import}
    _definitions(mod, raw["env/env_systems/base_env.py"], "env/env_systems/base_env.py")
    _definitions(
        mod, raw["env/env_systems/browsecomp_plus_env.py"], "env/env_systems/browsecomp_plus_env.py"
    )
    mod.source = raw
    return mod


def project_document(document, plan):
    text = document["text"]
    if (
        document["docid"] != plan["docid"]
        or hashlib.sha256(text.encode()).hexdigest() != plan["text_sha256"]
    ):
        raise ValueError("document identity differs from token audit")
    result, cursor = [], 0
    for start, end, count in plan["spans"]:
        if start != cursor or not start < end <= len(text) or not 0 < count <= 8192:
            raise ValueError("document span coverage/limit mismatch")
        result.append(text[start:end])
        cursor = end
    if not result or cursor != len(text):
        raise ValueError("document span coverage mismatch")
    return result


def approval_for(spec_path, approval_path, *, purpose):
    approval = read_json(approval_path)
    if (
        approval.get("spec_sha256") != sha(spec_path)
        or approval.get("decision") != "approve"
        or approval.get("purpose") != purpose
        or not approval.get("reviewer")
        or approval.get("reviewer") == "producer"
        or approval.get("independent") is not True
    ):
        raise ValueError("independent byte-bound approval required")
    return read_json(spec_path)


class SearchAdmission:
    """Constructed only by complete read-only byte/asset/approval verification."""

    def __init__(self, spec_path, approval_path):
        self.spec_path, self.approval_path = Path(spec_path), Path(approval_path)
        self.spec = verify_spec(approval_for(spec_path, approval_path, purpose="search-quality"))
        self.spec_sha256 = sha(spec_path)

    def validate(self):
        spec = approval_for(self.spec_path, self.approval_path, purpose="search-quality")
        if sha(self.spec_path) != self.spec_sha256 or spec != self.spec:
            raise ValueError("Search admission changed; independent reapproval required")
        return verify_spec(spec, allow_started=True, assets=False)


def execute(spec_path, approval_path):
    admission = SearchAdmission(spec_path, approval_path)
    return _execute_verified(
        admission.spec, spec_path=spec_path, approval_path=approval_path, admission=admission
    )


class SearchTransport(RackClient):
    """Scoped SDK seam. Native cumulative actor allowance; no retry/fallback."""

    def __init__(self, profile, *, receipt, budget, admission):
        if type(admission) is not SearchAdmission:
            raise ValueError("complete byte-bound Search asset admission required")
        spec = admission.validate()
        if profile.model_dump(mode="json") != spec["profile"]:
            raise ValueError("actual native role profile differs from frozen admission")
        from .parallel_campaign import JobContext

        if (
            type(budget) is not JobContext
            or budget.calls != 0
            or budget.job.call_budget != spec["bounds"]["calls_per_task_method"]
            or budget.job.seconds_budget != spec["bounds"]["seconds_per_task_method"]
            or budget.limiter.max_in_flight != 1
        ):
            raise ValueError("fresh exact whole-task native/production budget required")
        self.actor_cap = spec["bounds"]["actor_output_tokens"]
        self.judge_cap = spec["bounds"]["judge_output_tokens"]
        self._identity_ready = False
        self._admitted_spec = spec
        self._offline_authored = False
        self._active_path = None
        super().__init__(profile, receipt=receipt, budget=budget)

    @classmethod
    def authored(cls, profile, *, receipt, budget, transport):
        import httpx

        if (
            type(transport) is not httpx.MockTransport
            or budget.calls != 0
            or budget.job.call_budget > 8
        ):
            raise ValueError("authored offline transport and <=8 fresh calls required")
        obj = cls.__new__(cls)
        obj.actor_cap, obj.judge_cap = 15000, 8000
        obj._identity_ready, obj._active_path = False, None
        obj._offline_authored = True
        obj._admitted_spec = None
        RackClient.__init__(obj, profile, receipt=receipt, budget=budget, transport=transport)
        return obj

    def _latch(self, exc):
        # A request timeout is local. Only explicit shared evidence cancels matrix.
        if self._failure is None:
            self._failure = safe_error(exc)
        if self.budget is not None:
            self.budget.failed = True
            if isinstance(exc, SharedRackError) or self._active_path in {"/models", "/tokenize"}:
                self.budget.budget.failed = True

    def _request(self, method, path, payload=None):
        self._active_path = path
        if path != "/chat/completions":
            return super()._request(method, path, payload)
        record = {
            "ordinal": self.budget.calls,
            "role": getattr(self, "_dispatch_role", "production"),
            "output_allowance": payload.get("max_completion_tokens", payload.get("max_tokens")),
            "complete": False,
            "usage": None,
        }
        self._dispatch_role = "production"
        started = time.monotonic()
        try:
            result = super()._request(method, path, payload)
            record.update(complete=True, usage=result.get("usage"), response_id=result.get("id"))
            return result
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            self.receipt.parent.mkdir(parents=True, exist_ok=True)
            shared._append(self.receipt.parent / "generation-receipts.jsonl", record)

    def native_create(self, role, **request):
        from openai.types.chat import ChatCompletion

        try:
            self.check()
            if not self._identity_ready:
                self.preflight()
                self._identity_ready = True
            if role not in {"actor", "judge"} or request.get("model") != self.profile.model:
                raise SharedRackError("undeclared native role/model")
            if (
                not self._offline_authored
                and role == "actor"
                and request.get("tools") != self._admitted_spec["search_tool_definitions"]
            ):
                raise FatalRackError("native search/get_document schema differs from frozen source")
            cap = request.get("max_completion_tokens")
            if type(cap) is not int or not 1 <= cap <= (
                self.actor_cap if role == "actor" else self.judge_cap
            ):
                raise FatalRackError("native output allowance differs from freeze")
            if request.get("temperature", 0) != 0 or request.get("top_p", 1) != 1:
                raise FatalRackError("undeclared native sampling")
            if set(request) - {
                "model",
                "messages",
                "max_completion_tokens",
                "temperature",
                "top_p",
                "tools",
                "tool_choice",
            }:
                raise FatalRackError("undeclared native SDK request fields")
            messages = json.loads(
                json.dumps(request["messages"], default=lambda x: x.model_dump(exclude_none=True))
            )
            payload = {
                **request,
                "messages": messages,
                "temperature": 0,
                "stream": False,
                "n": 1,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            token_request = {
                "model": self.profile.model,
                "messages": messages,
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            if "tools" in request:
                token_request["tools"] = request["tools"]
            tokens = self._request("POST", "/tokenize", token_request).get("count")
            self.admit_context(
                tokens,
                output_budget=cap,
                input_chars=len(json.dumps(token_request, ensure_ascii=False)),
            )
            with self.generation_lease(tokens + cap + self.profile.schema_reserve_tokens):
                self.budget.begin_call()
                self._dispatch_role = role
                result = self._request("POST", "/chat/completions", payload)
            if result.get("model") != self.profile.model:
                raise SharedRackError("native response model mismatch")
            usage = result.get("usage", {})
            if usage.get("prompt_tokens") != tokens:
                raise SharedRackError("native tokenizer/wire token parity mismatch")
            if any(
                type(usage.get(k)) is not int or usage[k] < 0
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            ):
                raise SharedRackError("native response accounting unavailable")
            choices = result.get("choices", [])
            if len(choices) != 1 or choices[0].get("finish_reason") not in {"stop", "tool_calls"}:
                raise FatalRackError("native response incomplete")
            if role == "judge" and choices[0].get("finish_reason") != "stop":
                raise FatalRackError("judge cannot call tools")
            calls = choices[0]["message"].get("tool_calls") or []
            if bool(calls) != (choices[0]["finish_reason"] == "tool_calls"):
                raise FatalRackError("native tool-call finish/schema mismatch")
            schemas = {
                t["function"]["name"]: t["function"]["parameters"] for t in request.get("tools", [])
            }
            for call in calls:
                function = call.get("function", {})
                schema = schemas.get(function.get("name"))
                if schema is None:
                    raise FatalRackError("undeclared native response tool")
                arguments = json.loads(function["arguments"])
                if (
                    not isinstance(arguments, dict)
                    or set(arguments) != set(schema["required"])
                    or any(not isinstance(arguments[k], str) for k in arguments)
                ):
                    raise FatalRackError("malformed native tool arguments; no fallback turn")
            return ChatCompletion.model_validate(result)
        except (Exception, shared.NativeAbort) as exc:
            self._latch(exc)
            raise shared.NativeAbort(safe_error(exc)) from exc

    def sdk(self, role):
        if role == "actor":
            return SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(
                        create=lambda **kw: self.native_create("actor", **kw)
                    )
                )
            )
        if role == "judge":

            def respond(**kw):
                if set(kw) != {"model", "max_output_tokens", "input"}:
                    raise shared.NativeAbort("unexpected native Responses judge fields")
                result = self.native_create(
                    "judge",
                    model=kw["model"],
                    max_completion_tokens=kw["max_output_tokens"],
                    messages=[{"role": "user", "content": kw["input"]}],
                )
                return SimpleNamespace(output_text=result.choices[0].message.content)

            return SimpleNamespace(responses=SimpleNamespace(create=respond))
        raise shared.NativeAbort("undeclared SDK role")


def build_adapted(config_path, approval_path):
    config = approval_for(config_path, approval_path, purpose="search-adapted-index-build")
    return _build_adapted_verified(config, config_path=config_path, approval_path=approval_path)


def dataset_plan(data_root, bridge_path, legacy_path):
    from .arena_driver import task_digest
    from .trajectory_suite import load_arena

    bridge = read_json(bridge_path)
    if sha(legacy_path) != "6f6b1f6c40ae37196e23fe4053747568ed2031bffd3da3733748f99c6631b46f":
        raise ValueError("native legacy data hash mismatch")
    legacy = {
        str(r["id"]): r
        for r in (json.loads(line) for line in Path(legacy_path).read_bytes().splitlines())
    }
    tasks = load_arena(
        Path(data_root),
        domain="progressive_search",
        pins={"progressive_search/data.jsonl": DATA_SHA},
    )
    mapping = {r["memoryarena_task_id"]: r["native_query_id"] for r in bridge["bridge"]}
    if (
        len(mapping) != len(bridge["bridge"])
        or len(tasks) != 221
        or sum(len(t.sessions) for t in tasks) != 1641
    ):
        raise ValueError("full 221 task / 1641 session coverage required")
    rows = []
    for task in tasks:
        qid = mapping.get(task.id)
        old = legacy.get(qid)
        if (
            old is None
            or old["question"] != [s.question for s in task.sessions]
            or old["answer"] != list(task.gold)
        ):
            raise ValueError("complete ordered native ID bridge mismatch")
        rows.append(
            {
                "id": task.id,
                "native_query_id": qid,
                "sessions": len(task.sessions),
                "input_sha256": task_digest(task),
            }
        )
    return {
        "tasks": rows,
        "sessions": 1641,
        "data_sha256": DATA_SHA,
        "bridge_sha256": sha(bridge_path),
        "legacy_sha256": sha(legacy_path),
    }


def _check_record(record):
    if (
        not isinstance(record, dict)
        or not Path(record.get("path", "")).is_file()
        or sha(record["path"]) != record.get("sha256")
    ):
        raise ValueError("frozen byte-bound file differs or missing")
    return Path(record["path"])


def verify_index_admission(admission):
    if admission.get("kind") not in {"original", "adapted-gemma2"}:
        raise ValueError("explicit original/adapted encoder identity admission required")
    if not admission.get("identity_evidence"):
        raise ValueError("encoder identity evidence required; dimensions/defaults are not identity")
    evidence = read_json(_check_record(admission["identity_evidence"]))
    if admission["kind"] == "original":
        if (
            admission.get("dimension") != 1536
            or evidence.get("protocol") != "original-search-encoder-identity@v1"
            or evidence.get("index_build_manifest_verified") is not True
            or evidence.get("exact_text_projection_verified") is not True
            or evidence.get("query_recipe_verified") is not True
            or evidence.get("no_cost_local_endpoint_qualified") is not True
            or not evidence.get("archived_build_manifest")
            or not evidence.get("vector_parity_receipt")
        ):
            raise ValueError("original encoder identity/recipe/local qualification incomplete")
        _check_record(evidence["archived_build_manifest"])
        _check_record(evidence["vector_parity_receipt"])
    else:
        from .embedding_runtime import EmbeddingRecipe

        if (
            admission.get("dimension") != 768
            or evidence.get("protocol") != "adapted-search-gemma2-index@v1"
            or evidence.get("recipe") != EmbeddingRecipe().as_dict()
            or evidence.get("rows") != 100195
            or evidence.get("corpus_sha256") != CORPUS_SHA
            or evidence.get("complete") is not True
            or evidence.get("projection") != "fulltext-contiguous-chunk-mean-l2@v1"
        ):
            raise ValueError("adapted complete corpus/recipe admission incomplete")
        _check_record(evidence["token_audit"])
        _check_record(evidence["token_rows"])
        _check_record(evidence["wire"])
        config_path = _check_record(evidence["build_config"])
        approval_path = _check_record(evidence["build_approval"])
        approval_for(config_path, approval_path, purpose="search-adapted-index-build")
        for record in evidence["producer_files"]:
            _check_record(record)
    if evidence.get("index_files") != admission.get("index_files"):
        raise ValueError("index identity evidence differs from admitted files")
    for record in admission["index_files"]:
        _check_record(record)
    import glob

    bound = {str(Path(r["path"]).resolve()) for r in admission["index_files"]}
    resolved = {
        str(Path(p).resolve())
        for pattern in (admission["index_path"], admission["id_map_path"])
        for p in glob.glob(pattern)
    }
    if bound != resolved:
        raise ValueError("native index/ID-map globs differ from byte-bound admitted files")
    _local_url(admission["embedding_url"])
    if len(admission["index_files"]) != 8:
        raise ValueError("four exact shards and four ID maps required")
    if admission.get("corpus_sha256") != CORPUS_SHA or sha(admission["corpus_path"]) != CORPUS_SHA:
        raise ValueError("complete native corpus required")
    return evidence


def _local_url(url):
    from urllib.parse import urlsplit

    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.path != "/v1"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("explicit authless loopback encoder /v1 required")
    return url


def _build_adapted_verified(config, *, config_path, approval_path):
    """Entirely separate adapted corpus build; never called by freeze/run."""
    import httpx
    import numpy as np

    recipe = EmbeddingRecipe()
    if (
        config.get("protocol") != "adapted-search-gemma2-build@v1"
        or config.get("recipe") != recipe.as_dict()
        or config.get("projection") != "fulltext-contiguous-chunk-mean-l2@v1"
        or config.get("batch_size") != 32
    ):
        raise ValueError("explicit adapted recipe/projection/batch contract required")
    if sha(config["corpus_path"]) != CORPUS_SHA:
        raise ValueError("complete corpus SHA mismatch")
    audit_path = _check_record(config["token_audit"])
    audit = read_json(audit_path)
    qualification = read_json(_check_record(config["encoder_qualification"]))
    parity = read_json(_check_record(config["tokenizer_parity"]))
    if qualification.get("model") != recipe.as_dict() or qualification.get("finite") is not True:
        raise ValueError("existing real shared encoder qualification required")
    if (
        parity.get("complete_backend_json_equal") is not True
        or parity.get("tokenizer_sha256") != audit["tokenizer_sha256"]
    ):
        raise ValueError("exact served-runtime tokenizer parity required")
    if (
        config["max_embedding_calls"] != (audit["chunks"] + 31) // 32
        or config["max_document_tokens"] != audit["document_dispatch_tokens"]
    ):
        raise ValueError("exact measured adapted work allowance required")
    rows_path = _check_record(config["token_rows"])
    if (
        audit["rows"] != 100195
        or audit["rows_sha256"] != sha(rows_path)
        or audit["corpus_sha256"] != CORPUS_SHA
    ):
        raise ValueError("exact full corpus tokenizer audit required")
    if audit["document_prefix"] != recipe.prefix("document") or audit[
        "query_prefix"
    ] != recipe.prefix("query"):
        raise ValueError("actual prefixed tokenizer recipe required")
    for record in config["producer_files"]:
        _check_record(record)
    if not any(
        Path(r["path"]).resolve() == Path(__file__).resolve() for r in config["producer_files"]
    ):
        raise ValueError("builder source byte pin required")
    base = _local_url(config["embedding_url"])
    out = Path(config["output_root"])
    if out.exists():
        raise FileExistsError("fresh adapted index root required")
    out.mkdir(parents=True)
    started = time.monotonic()
    calls = 0
    tokens = 0
    sums = np.zeros((100195, 768), dtype=np.float64)
    counts = np.zeros(100195, dtype=np.int64)
    ids = []
    pending = []
    expected_counts = []
    owners = []
    wire = out / "embedding-wire.jsonl"

    def flush(client):
        nonlocal calls, tokens
        if not pending:
            return
        if (
            calls >= config["max_embedding_calls"]
            or tokens + sum(expected_counts) > config["max_document_tokens"]
            or time.monotonic() - started >= config["max_build_seconds"]
        ):
            raise ValueError("frozen adapted whole-build work/deadline allowance exhausted")
        payload = {"model": recipe.model, "input": list(pending), "input_type": "document"}
        receipt = {
            "role": "document",
            "items": len(pending),
            "shape": [len(pending), 768],
            "input_sha256": hashlib.sha256(
                json.dumps(pending, ensure_ascii=False).encode()
            ).hexdigest(),
            "prepared_sha256": hashlib.sha256(
                json.dumps(
                    [recipe.prefix("document") + s for s in pending], ensure_ascii=False
                ).encode()
            ).hexdigest(),
            "row_ordinals": list(owners),
            "expected_token_counts": list(expected_counts),
            "internal_batch_size": 32,
            "complete": False,
        }
        calls += 1
        try:
            result = client.post(
                base + "/embeddings", json=payload, timeout=config["request_timeout_seconds"]
            )
            result.raise_for_status()
            result = result.json()
            if (
                result.get("recipe") != recipe.as_dict()
                or result.get("role") != "document"
                or result.get("token_counts") != expected_counts
            ):
                raise ValueError("adapted actual role/recipe/tokenization mismatch")
            data = result["data"]
            if [r["index"] for r in data] != list(range(len(pending))):
                raise ValueError("embedding batch order mismatch")
            matrix = np.asarray([r["embedding"] for r in data], dtype=np.float32)
            if (
                matrix.shape != (len(pending), 768)
                or not np.isfinite(matrix).all()
                or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4)
            ):
                raise ValueError("adapted vector shape/norm mismatch; no resizing")
            for owner, vector in zip(owners, matrix, strict=True):
                sums[owner] += vector
                counts[owner] += 1
            actual = result.get("usage", {}).get("prompt_tokens")
            if actual != sum(expected_counts):
                raise ValueError("embedding usage/token count mismatch")
            tokens += actual
            receipt.update(
                complete=True,
                usage=result["usage"],
                token_counts=result["token_counts"],
                recipe=result["recipe"],
            )
        finally:
            shared._append(wire, receipt)
        pending.clear()
        expected_counts.clear()
        owners.clear()

    try:
        with httpx.Client(trust_env=False, follow_redirects=False) as client:
            models = client.get(base + "/models", timeout=30)
            models.raise_for_status()
            if not any(
                r.get("id") == recipe.model and r.get("recipe") == recipe.as_dict()
                for r in models.json()["data"]
            ):
                raise ValueError("actual embedding service recipe not admitted")
            with Path(config["corpus_path"]).open("rb") as corpus, rows_path.open("rb") as plans:
                for ordinal, (line, planline) in enumerate(zip(corpus, plans, strict=True)):
                    document, plan = json.loads(line), json.loads(planline)
                    if plan["ordinal"] != ordinal:
                        raise ValueError("original corpus row order differs")
                    ids.append(document["docid"])
                    chunks = project_document(document, plan)
                    for chunk, span in zip(chunks, plan["spans"], strict=True):
                        pending.append(chunk)
                        expected_counts.append(span[2])
                        owners.append(ordinal)
                        if len(pending) == 32:
                            flush(client)
                flush(client)
        if (
            len(ids) != 100195
            or len(set(ids)) != 100195
            or (counts <= 0).any()
            or int(counts.sum()) != audit["chunks"]
        ):
            raise ValueError("all original corpus rows/IDs/chunks required")
        if tokens != audit["document_dispatch_tokens"]:
            raise ValueError("exact full-token work differs")
        matrix = sums / counts[:, None]
        norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        if (norm <= 0).any():
            raise ValueError("zero pooled document vector")
        matrix = (matrix / norm).astype("float32")
        np.save(out / "pooled-vectors.npy", matrix)
        write_json(out / "original-ids.json", ids)
        worker = _check_record(config["native_worker"]["source"])
        subprocess.run(
            [
                config["native_worker"]["python"],
                "-I",
                str(worker),
                "build",
                "--matrix",
                str(out / "pooled-vectors.npy"),
                "--ids",
                str(out / "original-ids.json"),
                "--out",
                str(out),
            ],
            check=True,
            capture_output=True,
            timeout=config["index_finalization_seconds"],
        )
        files = []
        for shard in range(4):
            index_path = out / f"shard{shard}.index"
            map_path = out / f"shard{shard}_id_map.json"
            if read_json(map_path)["ids"] != ids[shard::4]:
                raise ValueError("adapted disk ID readback mismatch")
            files.extend(
                [
                    {"path": str(index_path), "sha256": sha(index_path)},
                    {"path": str(map_path), "sha256": sha(map_path)},
                ]
            )
        result = {
            "protocol": "adapted-search-gemma2-index@v1",
            "build_config": {"path": str(config_path), "sha256": sha(config_path)},
            "build_approval": {"path": str(approval_path), "sha256": sha(approval_path)},
            "producer_files": config["producer_files"],
            "complete": True,
            "rows": len(ids),
            "corpus_sha256": CORPUS_SHA,
            "recipe": recipe.as_dict(),
            "projection": config["projection"],
            "token_audit": config["token_audit"],
            "token_rows": config["token_rows"],
            "wire": {"path": str(wire), "sha256": sha(wire)},
            "index_files": files,
            "embedding_calls": calls,
            "prompt_tokens": tokens,
            "quality_calls": 0,
            "elapsed_seconds": time.monotonic() - started,
            "document_prefix": recipe.prefix("document"),
            "query_prefix": recipe.prefix("query"),
            "batch_size": 32,
        }
        for record in config["producer_files"]:
            _check_record(record)
        if time.monotonic() - started > config["max_build_seconds"]:
            raise ValueError("frozen whole-build deadline expired during finalization")
        write_json(out / "index-evidence.json", result)
        return result
    except BaseException as exc:
        write_json(
            out / "failure.json",
            {
                "status": "failed",
                "error": safe_error(exc),
                "embedding_attempts": calls,
                "known_prompt_tokens": tokens,
                "currency_cost": 0,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise


class GuardedTools:
    def __init__(self, tools, budget):
        self.native, self.budget = tools, budget

    def get_tool_definitions(self):
        return self.native.get_tool_definitions()

    def execute_tool(self, name, arguments):
        try:
            self.budget.check()
            key = {"search": "query", "get_document": "docid"}.get(name)
            if (
                key is None
                or not isinstance(arguments, dict)
                or set(arguments) != {key}
                or not isinstance(arguments[key], str)
            ):
                raise ValueError("undeclared native tool/arguments; generated code forbidden")
            result = self.native.execute_tool(name, arguments)
            self.budget.check()
            return result
        except Exception as exc:
            self.budget.failed = True
            raise shared.NativeAbort(safe_error(exc)) from exc


def run_task(
    task, *, native, client, tools, memory, out, native_query_id, gold_path, qrels, max_iterations
):
    from .arena_driver import task_digest

    out = Path(out)
    if type(client) is not SearchTransport or (
        not callable(memory) and memory.budget is not client.budget
    ):
        raise ValueError("one native/production whole-task budget required")
    if client._offline_authored:
        if task.id >= 0 or not all(s.question.startswith("Authored") for s in task.sessions):
            raise ValueError("offline authored task only")
    else:
        matches = [r for r in client._admitted_spec["tasks"] if r["id"] == task.id]
        if (
            len(matches) != 1
            or matches[0]["input_sha256"] != task_digest(task)
            or matches[0]["native_query_id"] != native_query_id
        ):
            raise ValueError("task not admitted by full frozen input/native ID bridge")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    steps = []
    memories = []
    closed = False
    guarded = GuardedTools(tools, client.budget)

    def run_query(query, memory_client=None, output_dir=None, **kwargs):
        context = memory.query(query)
        wrapped = (
            query
            if not context
            else "<memory_context>\n" + context + "\n</memory_context>\n\n" + query
        )
        # Match native agent/search TSV projection, but refuse native silent truncation.
        projected = wrapped.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        if len(projected) > 130000:
            raise shared.NativeAbort("native TSV prompt limit exceeded; no silent truncation")
        request = native.build_request(
            projected, client.profile.model, client.actor_cap, guarded, temperature=0
        )
        response, trace, usage, outputs = native.run_conversation_with_tools(
            client.sdk("actor"), request, guarded, max_iterations=max_iterations
        )
        if response.choices[0].finish_reason != "stop" or response.choices[0].message.tool_calls:
            raise shared.NativeAbort("native loop exhausted without complete final answer")
        target = out / f"session-{len(steps):02d}"
        target.mkdir()
        native._persist_response(
            str(target), request, response, trace, usage, outputs, query_id=native_query_id
        )
        files = list(target.glob("run_*.json"))
        if len(files) != 1:
            raise ValueError("one native session artifact required")
        full = read_json(files[0])
        answer = response.choices[0].message.content
        if not isinstance(answer, str):
            raise ValueError("native complete answer must be text")
        record = {
            "question_sha256": hashlib.sha256(query.encode()).hexdigest(),
            "answer": answer,
            "trace": full["result"],
            "tool_call_counts": full["tool_call_counts"],
            "retrieved_docids": full["retrieved_docids"],
            "full_result": full,
            "memory_context": context,
            "query_with_memory": projected,
        }
        steps.append(record)
        shared._append(out / "steps.jsonl", record)
        return record

    native.query_runner = run_query
    try:
        if callable(memory):
            memory = memory(client.budget, out)
        if memory.budget is not client.budget:
            raise ValueError("production memory budget differs")
        for i, session in enumerate(task.sessions[:-1]):
            item = run_query(session.question)
            content = f"Subquery {i + 1}: {session.question}\n\nPredicted Answer: {item['answer']}"
            if item["trace"]:
                content += "\n\nTrace: " + json.dumps(item["trace"], ensure_ascii=False)
            receipt = memory.insert(
                {
                    "source_ref": f"progressive_search:{task.id}:session:{i}",
                    "content": content,
                    "authorship": "agent",
                    "feedback_in_memory": False,
                }
            )
            if receipt.get("readback_verified") is not True:
                raise ValueError("native experience commit unverified")
            memories.append(receipt)
            shared._append(out / "memory-receipts.jsonl", receipt)
        final = native.BrowseCompPlusEnvironment.run_final_query(
            query_id=native_query_id,
            original_query=task.sessions[-1].question,
            subqueries=[s.question for s in task.sessions[:-1]],
            output_dir=out,
            memory_client=None,
            script_path="",
            index_path="",
            corpus_path="",
            agent_model=client.profile.model,
            model_name="",
            mcp_url=None,
            provider=None,
            mcp_name="",
            gpu=None,
            qrel_data=qrels,
            client=client.sdk("judge"),
            judge_model=client.profile.model,
            ground_truth_path=gold_path,
            step_memory=False,
            store_eval_in_memory=False,
        )
        judgement = final["judgement"]
        if judgement.get("parse_error") or type(judgement.get("correct")) is not bool:
            raise shared.NativeAbort("native judge parse error; no fabricated zero")
        if len(steps) != len(task.sessions):
            raise ValueError("complete session coverage required")
        client.budget.check()
        memory.close()
        closed = True
        (out / "native-evaluator-source.py").write_bytes(
            native.source[WEB + "evaluate_with_openai.py"]
        )
        result = {
            **final,
            "protocol": PROTOCOL,
            "status": "complete",
            "task_id": task.id,
            "native_query_id": native_query_id,
            "query_id": native_query_id,
            "completed_sessions": len(steps),
            "expected_sessions": len(task.sessions),
            "calls": client.budget.calls,
            "input_sha256": task_digest(task),
            "elapsed_seconds": time.monotonic() - started,
            "currency_cost": None,
            "scoring_relevant_docids": qrels.get(native_query_id, []),
            "native_summary": native._summarize_single_result(
                {**final, "query_id": native_query_id}, qrel_data=qrels
            ),
            "completion_authority": "root campaign-result.json only",
        }
        files = [
            {"path": str(p.relative_to(out)), "sha256": sha(p)}
            for p in sorted(out.rglob("*"))
            if p.is_file()
        ]
        result["files"] = files
        write_json(out / "result.json", result)
        verify_task(out)
        return result
    except BaseException as exc:
        client.budget.failed = True
        if (out / "result.json").exists():
            (out / "result.json").unlink()
        write_json(
            out / "failure.json",
            {
                "status": "failed",
                "error": safe_error(exc),
                "completed_sessions": len(steps),
                "expected_sessions": len(task.sessions),
                "calls": client.budget.calls,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise
    finally:
        if not closed:
            try:
                memory.close()
            except Exception as exc:
                shared._append(
                    out / "cleanup-failures.jsonl",
                    {"error": safe_error(exc), "calls": client.budget.calls},
                )


def verify_task(out):
    out = Path(out)
    if (out / "failure.json").exists():
        raise ValueError("failed task cannot be complete")
    result = read_json(out / "result.json")
    if (
        result["status"] != "complete"
        or result["completed_sessions"] != result["expected_sessions"]
    ):
        raise ValueError("incomplete native task")
    required = {
        "steps.jsonl",
        "http.jsonl",
        "generation-receipts.jsonl",
        "native-evaluator-source.py",
    }
    if result["expected_sessions"] > 1:
        required.add("memory-receipts.jsonl")
    if not required <= {r["path"] for r in result["files"]}:
        raise ValueError("complete named native task evidence inventory required")
    for record in result["files"]:
        path = (out / record["path"]).resolve()
        if not path.is_relative_to(out.resolve()) or sha(path) != record["sha256"]:
            raise ValueError("task artifact byte mismatch")
    steps = [json.loads(line) for line in (out / "steps.jsonl").read_bytes().splitlines()]
    if (
        len(steps) != result["completed_sessions"]
        or steps[-1]["answer"] != result["predicted_answer"]
    ):
        raise ValueError("native saved session/answer mismatch")
    receipts = [
        json.loads(line) for line in (out / "generation-receipts.jsonl").read_bytes().splitlines()
    ]
    wire = [json.loads(line) for line in (out / "http.jsonl").read_bytes().splitlines()]
    generations = [r for r in wire if r["url"].endswith("/chat/completions")]
    if (
        len(generations) != result["calls"]
        or len(receipts) != result["calls"]
        or [r["ordinal"] for r in receipts] != list(range(1, result["calls"] + 1))
        or any(not r["complete"] for r in receipts)
    ):
        raise ValueError("native actual generation accounting mismatch")
    judge_indices = [i for i, r in enumerate(receipts) if r["role"] == "judge"]
    if len(judge_indices) != 1:
        raise ValueError("one native final judge required")
    parser = (out / "native-evaluator-source.py").read_bytes()
    if hashlib.sha256(parser).hexdigest() != SOURCE_PINS[WEB + "evaluate_with_openai.py"]:
        raise ValueError("native saved judge parser source mismatch")
    mod = ModuleType("_offline_saved_search_judge")
    mod.re = re
    _definitions(mod, parser, WEB + "evaluate_with_openai.py", names={"parse_judge_response"})
    raw = generations[judge_indices[0]]["response"]["choices"][0]["message"]["content"]
    parsed = mod.parse_judge_response(raw)
    if any(parsed.get(k) != v for k, v in result["judgement"].items()):
        raise ValueError("saved native judge verdict differs from actual raw response")
    relevant = set(result["scoring_relevant_docids"])
    recall = len(set(result["retrieved_docids"]) & relevant) / len(relevant) if relevant else None
    if recall != result["recall"]:
        raise ValueError("native saved recall differs from evidence IDs")
    result["cost_accounting"] = cost_accounting(out)
    return result


class WorkerSearch:
    """Native FAISS/corpus algorithm in isolated existing CPython3.12, no installs."""

    def __init__(
        self, config, admission, *, native_root, native, embedder, budget, expected_rows=100195
    ):
        import os

        self.embedder, self.budget = embedder, budget
        self.search_description = lambda k: native.BaseSearcher.search_description(self, k)
        self.get_document_description = lambda: native.BaseSearcher.get_document_description(self)
        worker = _check_record(config["source"])
        env = {
            k: v
            for k, v in os.environ.items()
            if k.upper()
            in {
                "SYSTEMROOT",
                "WINDIR",
                "PATH",
                "LOCALAPPDATA",
                "USERPROFILE",
                "TMPDIR",
                "TEMP",
                "TMP",
            }
        }
        self.process = subprocess.Popen(
            [
                config["python"],
                "-I",
                str(worker),
                "serve",
                "--native-root",
                str(native_root),
                "--corpus",
                admission["corpus_path"],
                "--index-path",
                admission["index_path"],
                "--id-map-path",
                admission["id_map_path"],
                "--dimension",
                str(admission["dimension"]),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=env,
        )
        try:
            ready = self._receive()
            if ready != {
                "ready": True,
                "rows": expected_rows,
                "vectors": expected_rows,
                "dimension": admission["dimension"],
            }:
                raise ValueError("native worker full corpus/index identity differs")
        except BaseException:
            self.close()
            raise

    def _receive(self):
        import queue
        import threading

        q = queue.Queue()

        def read():
            try:
                q.put(self.process.stdout.readline())
            except BaseException as exc:
                q.put(exc)

        threading.Thread(target=read, daemon=True).start()
        try:
            line = q.get(timeout=min(300, self.budget.remaining()))
        except queue.Empty:
            self.close()
            raise TimeoutError("native local FAISS worker deadline") from None
        if isinstance(line, BaseException):
            raise line
        if not line:
            raise RuntimeError("native local worker exited: " + self.process.stderr.read()[-2000:])
        result = json.loads(line)
        if "error" in result:
            raise shared.NativeAbort(result["error"])
        return result

    def _call(self, request):
        self.budget.check()
        self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        result = self._receive()["result"]
        self.budget.check()
        return result

    def search(self, query, k=10, allowed_docids=None):
        vector = self.embedder.embed_query(query)
        return self._call(
            {
                "command": "search",
                "query": query,
                "vector": vector.tolist(),
                "k": k,
                "allowed_docids": allowed_docids,
            }
        )

    def get_document(self, docid):
        return self._call({"command": "get_document", "docid": docid})

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"command":"close"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=10)
            except Exception:
                self.process.kill()
                self.process.wait(timeout=10)
        for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
            if pipe:
                pipe.close()


def freeze_spec(config):
    """Freeze candidate bytes without approving encoder, endpoints, or any QA."""
    from .arena_driver import _code_inventory
    from .rack import RackProfile

    profile = RackProfile.model_validate(config["profile"])
    plan = dataset_plan(config["data_root"], config["bridge_path"], config["legacy_path"])
    native = load_native(config["native_root"])
    searcher = native.OpenAISearcher.__new__(native.OpenAISearcher)
    tools = native.SearchToolHandler(searcher, snippet_max_tokens=None, k=10).get_tool_definitions()
    return {
        **config,
        **plan,
        "protocol": PROTOCOL,
        "native_commit": PIN,
        "native_source_sha256": SOURCE_PINS,
        "search_tool_definitions": tools,
        "systems": list(SYSTEMS),
        "optional_methods": {
            "mem0": "unqualified; no stock-document admission",
            "cognee": "unqualified; no native-document admission",
        },
        "roles": {
            r: profile.model for r in ("actor", "nativejudge", "extraction", "consolidation")
        },
        "files": _code_inventory(),
        "quality_calls": 0,
        "adaptations": [
            "Explicit shared six-method online experience-memory interface; "
            "agent documents, never user-chat coercion.",
            "Native Responses grader prompt/parser transported through guarded Chat Completions.",
            "Matched deterministic temperature=0; native 15000 cumulative actor output "
            "and 8000 judge output.",
            "Native TSV whitespace projection; overflow fails instead of "
            "native silent character truncation.",
            "An admitted Gemma2 corpus index is adapted retrieval, "
            "never archived-index reproduction.",
            "Native Qwen3-0.6B tokenizer/512-token snippets require "
            "separately byte-frozen local tokenizer assets.",
        ],
    }


def verify_gold_projection(binding, receipt_binding):
    """Preserve native official evaluator answers, not legacy generated answer traces."""
    projection = _check_record(binding)
    receipt = read_json(_check_record(receipt_binding))
    expected = "0374946363a5c59f4d3707fa9c31b9e5152269c0f024668a71687dda4e48a85e"
    if (
        receipt.get("source_sha256") != expected
        or receipt.get("projection_sha256") != sha(projection)
        or receipt.get("rows") != 830
    ):
        raise ValueError("pinned native evaluator projection required")
    projected = [json.loads(line) for line in projection.read_bytes().splitlines()]
    if len(projected) != 830 or len({r["query_id"] for r in projected}) != 830:
        raise ValueError("complete unique native final gold IDs required")
    digest = hashlib.sha256()
    count = 0
    with Path(receipt["source_path"]).open("rb") as stream:
        for line, saved in zip(stream, projected, strict=True):
            digest.update(line)
            row = json.loads(line)
            actual = {
                "query_id": row["query_id"],
                "query": row["query"],
                "answer": row["answer"],
                "evidence_docids": [d["docid"] for d in row["evidence_docs"]],
            }
            if actual != saved:
                raise ValueError("native final gold/evidence projection changed")
            count += 1
    if digest.hexdigest() != expected or count != 830:
        raise ValueError("actual pinned native gold byte hash/row coverage mismatch")
    return {
        "rows": count,
        "query_ids": [r["query_id"] for r in projected],
        "source_sha256": expected,
    }


def verify_spec(spec, *, allow_started=False, assets=True):
    from .arena_driver import _code_inventory, verify_memory_qualification
    from .rack import RackProfile

    if (
        spec.get("protocol") != PROTOCOL
        or spec.get("native_commit") != PIN
        or spec.get("native_source_sha256") != SOURCE_PINS
    ):
        raise ValueError("Search full protocol/source pin mismatch")
    if spec.get("systems") != list(SYSTEMS):
        raise ValueError("exact six real method matrix required; Mem0/Cognee unqualified")
    # Encoder evidence is checked before clients, output roots or native dispatch.
    if assets:
        verify_index_admission(spec.get("index_admission", {}))
    profile = RackProfile.model_validate(spec["profile"])
    if (
        spec.get("roles")
        != {r: profile.model for r in ("actor", "nativejudge", "extraction", "consolidation")}
        or profile.profile_version != "role-budgets-v6"
    ):
        raise ValueError("explicit matched full role profile required")
    plan = dataset_plan(spec["data_root"], spec["bridge_path"], spec["legacy_path"])
    if any(spec.get(k) != v for k, v in plan.items()):
        raise ValueError("frozen full task/session/input bridge mismatch")
    b = spec["bounds"]
    if (
        b.get("actor_output_tokens") != 15000
        or b.get("judge_output_tokens") != 8000
        or b.get("max_iterations") != 30
        or b.get("top_k") != 10
        or b.get("snippet_tokens") != 512
        or b.get("feedback_in_memory") is not False
        or b.get("calls_per_task_method") != profile.max_generation_calls
        or b.get("seconds_per_task_method") != profile.max_run_seconds
        or b.get("calls_per_task_method", 0) < max(t["sessions"] for t in plan["tasks"]) * 30 + 1
        or type(b.get("campaign_seconds")) not in (int, float)
        or b["campaign_seconds"] <= 0
    ):
        raise ValueError("native frozen role/tool/output/whole-task bounds differ")
    if spec.get("files") != _code_inventory():
        raise ValueError("producer source inventory changed; re-freeze/review required")
    sources(spec["native_root"])
    root = Path(spec["output_root"]).resolve()
    project = Path(__file__).resolve().parents[3]
    if root.is_relative_to(project):
        raise ValueError("external output root required")
    if not allow_started and root.exists():
        raise FileExistsError("fresh Search root required")
    _local_url(spec["memory_embedding_url"])
    if assets:
        native = load_native(spec["native_root"])
        searcher = native.OpenAISearcher.__new__(native.OpenAISearcher)
        tools = native.SearchToolHandler(
            searcher, snippet_max_tokens=None, k=10
        ).get_tool_definitions()
        if spec.get("search_tool_definitions") != tools:
            raise ValueError("exact pinned native Search tool definitions required")
        _check_record(spec["native_worker"]["source"])
        _check_record(spec["native_worker"]["python_binary"])
        if str(Path(spec["native_worker"]["python"]).resolve()) != str(
            Path(spec["native_worker"]["python_binary"]["path"]).resolve()
        ):
            raise ValueError("native worker interpreter differs from frozen identity")
        gold = verify_gold_projection(spec["gold_projection"], spec["gold_projection_receipt"])
        if not {t["native_query_id"] for t in plan["tasks"]} <= set(gold["query_ids"]):
            raise ValueError("canonical bridge native IDs missing official final gold")
        tokenizer = spec["snippet_tokenizer"]
        if tokenizer.get("model_id") != "Qwen/Qwen3-0.6B" or not re.fullmatch(
            "[0-9a-f]{40}", tokenizer.get("revision", "")
        ):
            raise ValueError("native snippet tokenizer immutable identity required")
        for record in tokenizer["files"]:
            _check_record(record)
        verify_memory_qualification(spec.get("production_memory_qualification"), profile)
    return spec


class SearchEmbedding(RemoteEmbedder):
    """Explicit Gemma2 query/document wire role, bytes, shape and usage receipts."""

    def __init__(self, base_url, *, budget, receipt):
        super().__init__(_local_url(base_url))
        self.budget, self.receipt = budget, Path(receipt)

    def _encode(self, texts, role):
        import httpx
        import numpy as np

        self.budget.check()
        started = time.monotonic()
        receipt = {
            "role": role,
            "items": len(texts),
            "shape": [len(texts), self.dim],
            "internal_batch_size": 32,
            "input_sha256": hashlib.sha256(
                json.dumps(texts, ensure_ascii=False).encode()
            ).hexdigest(),
            "prepared_sha256": hashlib.sha256(
                json.dumps(
                    [self.recipe.prefix(role) + t for t in texts], ensure_ascii=False
                ).encode()
            ).hexdigest(),
            "complete": False,
        }
        try:
            with httpx.Client(trust_env=False, follow_redirects=False) as client:
                response = client.post(
                    self.base_url + "/embeddings",
                    json={"model": self.recipe.model, "input": texts, "input_type": role},
                    timeout=min(120, self.budget.remaining()),
                )
                response.raise_for_status()
                result = response.json()
            if result.get("recipe") != self.recipe.as_dict() or result.get("role") != role:
                raise SharedRackError("actual Search embedding recipe/role mismatch")
            data = result["data"]
            if [r["index"] for r in data] != list(range(len(texts))):
                raise SharedRackError("Search embedding response batch order mismatch")
            values = np.asarray([r["embedding"] for r in data], dtype=np.float32)
            if (
                values.shape != (len(texts), self.dim)
                or not np.isfinite(values).all()
                or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-4)
            ):
                raise SharedRackError("Search embedding dimensions/norm mismatch; no resizing")
            counts = result.get("token_counts")
            if (
                not isinstance(counts, list)
                or len(counts) != len(texts)
                or any(type(n) is not int or not 0 < n <= 8192 for n in counts)
            ):
                raise SharedRackError("actual embedding token counts required")
            if result.get("usage", {}).get("prompt_tokens") != sum(counts):
                raise SharedRackError("Search embedding accounting mismatch")
            receipt.update(
                complete=True, recipe=result["recipe"], token_counts=counts, usage=result["usage"]
            )
            self.budget.check()
            return list(values)
        except Exception as exc:
            self.budget.failed = True
            if isinstance(exc, SharedRackError):
                self.budget.budget.failed = True
            receipt["error"] = safe_error(exc)
            raise shared.NativeAbort(safe_error(exc)) from exc
        finally:
            receipt["elapsed_seconds"] = time.monotonic() - started
            self.receipt.parent.mkdir(parents=True, exist_ok=True)
            shared._append(self.receipt, receipt)


class OriginalQueryEncoder:
    """Only a separately admitted exact original local endpoint; no vector mutation."""

    def __init__(self, admission, *, budget, receipt):
        self.admission, self.budget, self.receipt = admission, budget, Path(receipt)
        self.url = _local_url(admission["embedding_url"])

    def embed_query(self, text):
        import httpx
        import numpy as np

        self.budget.check()
        a = self.admission
        with httpx.Client(trust_env=False, follow_redirects=False) as client:
            response = client.post(
                self.url + "/embeddings",
                json={"model": a["model"], "input": [text], "input_type": "query"},
                timeout=min(120, self.budget.remaining()),
            )
            response.raise_for_status()
            result = response.json()
        if (
            result.get("recipe") != a["query_recipe"]
            or result.get("role") != "query"
            or [r["index"] for r in result["data"]] != [0]
        ):
            raise SharedRackError("original exact query recipe/role differs")
        vector = np.asarray(result["data"][0]["embedding"], dtype=np.float32)
        if vector.shape != (1536,) or not np.isfinite(vector).all():
            raise SharedRackError("original exact 1536 dimensions required")
        self.receipt.parent.mkdir(parents=True, exist_ok=True)
        shared._append(
            self.receipt,
            {
                "complete": True,
                "role": "query",
                "items": 1,
                "shape": [1, 1536],
                "input_sha256": hashlib.sha256(
                    json.dumps([text], ensure_ascii=False).encode()
                ).hexdigest(),
                "recipe": result["recipe"],
                "usage": result.get("usage"),
                "post_normalization": "none; preserve returned float32",
            },
        )
        self.budget.check()
        return vector


def _execute_verified(spec, *, spec_path, approval_path, admission):
    import zipfile

    from gluomem.llm.modules import DspyExtractors
    from transformers import AutoTokenizer

    from .arena_driver import (
        ArenaProductionRouter,
        GluoExperienceMemory,
        RawExperienceMemory,
        execute_matrix,
    )
    from .rack import RackProfile, controlled_config
    from .trajectory_suite import load_arena

    profile = RackProfile.model_validate(spec["profile"])
    root = Path(spec["output_root"])
    native = load_native(spec["native_root"])
    tokenizer = AutoTokenizer.from_pretrained(
        spec["snippet_tokenizer"]["path"], local_files_only=True
    )
    native.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda name: (
            tokenizer
            if name == "Qwen/Qwen3-0.6B"
            else (_ for _ in ()).throw(ValueError("unfrozen tokenizer"))
        )
    )
    tasks = load_arena(
        Path(spec["data_root"]),
        domain="progressive_search",
        pins={"progressive_search/data.jsonl": DATA_SHA},
    )
    mapping = {r["id"]: r["native_query_id"] for r in spec["tasks"]}
    gold = [
        json.loads(line) for line in Path(spec["gold_projection"]["path"]).read_bytes().splitlines()
    ]
    qrels = {r["query_id"]: r["evidence_docids"] for r in gold}
    root.mkdir(parents=True, exist_ok=False)
    write_json(
        root / "campaign-result.json",
        {
            "protocol": PROTOCOL,
            "status": "running",
            "quality_complete": False,
            "calls": None,
            "completion_authority": "this root only",
        },
    )
    (root / "execution-spec.json").write_bytes(Path(spec_path).read_bytes())
    (root / "execution-approval.json").write_bytes(Path(approval_path).read_bytes())
    result = None
    try:
        with zipfile.ZipFile(root / "frozen-source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
            for i, record in enumerate(spec["files"]):
                archive.write(record["path"], arcname=f"{i:04d}/" + Path(record["path"]).name)
            archive.write(
                spec["native_worker"]["source"]["path"], arcname="external/native_search_worker.py"
            )
            for name, raw in native.source.items():
                archive.writestr("MemoryArena/" + name, raw)

        def run_one(task, system, context, target):
            client = None
            corpus = None
            try:
                client = SearchTransport(
                    profile, receipt=target / "http.jsonl", budget=context, admission=admission
                )
                encoder = SearchEmbedding(
                    spec["memory_embedding_url"],
                    budget=context,
                    receipt=target / "memory-embedding-wire.jsonl",
                )
                query_encoder = (
                    SearchEmbedding(
                        spec["index_admission"]["embedding_url"],
                        budget=context,
                        receipt=target / "corpus-query-wire.jsonl",
                    )
                    if spec["index_admission"]["kind"] == "adapted-gemma2"
                    else OriginalQueryEncoder(
                        spec["index_admission"],
                        budget=context,
                        receipt=target / "corpus-query-wire.jsonl",
                    )
                )
                corpus = WorkerSearch(
                    spec["native_worker"],
                    spec["index_admission"],
                    native_root=spec["native_root"],
                    native=native,
                    embedder=query_encoder,
                    budget=context,
                )
                tools = native.SearchToolHandler(
                    corpus, snippet_max_tokens=512, k=10, include_get_document=True
                )

                def memory_factory(budget, out):
                    if system != "gluomem":
                        return RawExperienceMemory(
                            system, budget=budget, embedder=encoder, top_k=10
                        )
                    cfg = controlled_config(
                        profile,
                        data=Path(spec["data_root"]) / "progressive_search/data.jsonl",
                        sha256=DATA_SHA,
                        embedder=encoder,
                    )
                    write_json(out / "effective-config.json", cfg.model_dump(mode="json"))
                    router = ArenaProductionRouter(
                        cfg, client=client, trace=out / "production-raw.jsonl", budget=budget
                    )
                    return GluoExperienceMemory(
                        cfg=cfg,
                        extractors=DspyExtractors(cfg, router=router),
                        embedder=encoder,
                        budget=budget,
                        db_path=out / "production.mind",
                        evidence_path=out / "memory-events.jsonl",
                        top_k=10,
                    )

                output = run_task(
                    task,
                    native=native,
                    client=client,
                    tools=tools,
                    memory=memory_factory,
                    out=target,
                    native_query_id=mapping[task.id],
                    gold_path=Path(spec["gold_projection"]["path"]),
                    qrels=qrels,
                    max_iterations=30,
                )
                corpus.close()
                corpus = None
                client.close()
                client = None
                return output
            except BaseException as exc:
                target.mkdir(parents=True, exist_ok=True)
                if (target / "result.json").exists():
                    (target / "result.json").unlink()
                prior = (
                    read_json(target / "failure.json") if (target / "failure.json").exists() else {}
                )
                write_json(
                    target / "failure.json",
                    {
                        **prior,
                        "status": "failed",
                        "calls": context.calls,
                        "error": safe_error(exc),
                    },
                )
                raise
            finally:
                if corpus is not None:
                    corpus.close()
                if client is not None:
                    client.close()

        b = spec["bounds"]
        result = execute_matrix(
            tasks,
            systems=SYSTEMS,
            out=root / "jobs",
            call_budget=b["calls_per_task_method"],
            seconds_budget=b["seconds_per_task_method"],
            campaign_seconds=b["campaign_seconds"],
            run_one=run_one,
        )
        metrics = {}
        for system in SYSTEMS:
            rows = [r for r in result["tasks"] if r["system"] == system]
            complete = [verify_task(r["output"]) for r in rows if r["status"] == "complete"]
            admitted = len(complete) == 221
            summaries = [r["native_summary"] for r in complete]
            tool_names = {tool for s in summaries for tool in s["average_tool_calls"]}
            calibration = [
                s["calibration_error"] for s in summaries if s["calibration_error"] is not None
            ]
            metrics[system] = {
                "status": "complete" if admitted else "incomplete-no-full-score",
                "expected_tasks": 221,
                "completed_tasks": len(complete),
                "average_tool_calls": {
                    tool: sum(s["average_tool_calls"].get(tool, 0) for s in summaries) / 221
                    for tool in tool_names
                }
                if admitted
                else None,
                "calibration_error": sum(calibration) / len(calibration)
                if admitted and calibration
                else None,
                "accuracy": sum(r["judgement"]["correct"] for r in complete) / 221
                if admitted
                else None,
                "recall": sum(r["recall"] for r in complete) / 221
                if admitted and all(r["recall"] is not None for r in complete)
                else None,
            }
        for row in result["tasks"]:
            if row.get("output"):
                row["cost_accounting"] = cost_accounting(row["output"])
                if row["status"] == "complete":
                    row["result_sha256"] = sha(Path(row["output"]) / "result.json")
        costs = cost_accounting(root / "jobs")
        costs["budget_calls"] = result["calls"]
        costs["unreceipted_budget_calls"] = result["calls"] - costs["attempts"]
        result["cost_accounting"] = costs
        if result["quality_complete"] and (
            costs["unknown_usage_attempts"] or costs["unreceipted_budget_calls"]
        ):
            raise ValueError("complete root generation accounting mismatch")
        result.update(
            protocol=PROTOCOL,
            status="complete" if result["quality_complete"] else "incomplete",
            metrics=metrics,
            optional_methods=spec["optional_methods"],
            spec_sha256=sha(spec_path),
            completion_authority="this root campaign-result.json only",
        )
        verify_spec(spec, allow_started=True, assets=False)
        write_json(root / "campaign-result.json", result)
        if read_json(root / "campaign-result.json") != result:
            raise ValueError("root result readback mismatch")
        return result
    except BaseException as exc:
        observed_cost = None
        accounting_error = None
        try:
            observed_cost = cost_accounting(root / "jobs")
        except Exception as accounting_exc:
            accounting_error = safe_error(accounting_exc)
        write_json(
            root / "campaign-result.json",
            {
                "protocol": PROTOCOL,
                "status": "failed",
                "quality_complete": False,
                "calls": None if result is None else result["calls"],
                "error": safe_error(exc),
                "retained_coordinator": str(root / "jobs"),
                "cost_accounting": observed_cost,
                "accounting_error": accounting_error,
                "completion_authority": "this root only",
            },
        )
        raise


def cost_accounting(out):
    """Count each actual generation once; failed unreported usage remains unknown."""
    records = []
    for path in Path(out).rglob("generation-receipts.jsonl"):
        records.extend(json.loads(line) for line in path.read_bytes().splitlines())
    by_role = {}
    for record in records:
        role = record["role"]
        value = by_role.setdefault(
            role,
            {
                "attempts": 0,
                "known_prompt_tokens": 0,
                "known_completion_tokens": 0,
                "unknown_usage_attempts": 0,
            },
        )
        value["attempts"] += 1
        usage = record.get("usage")
        if not isinstance(usage, dict) or any(
            type(usage.get(k)) is not int for k in ("prompt_tokens", "completion_tokens")
        ):
            value["unknown_usage_attempts"] += 1
        else:
            value["known_prompt_tokens"] += usage["prompt_tokens"]
            value["known_completion_tokens"] += usage["completion_tokens"]
    total = {
        key: sum(role[key] for role in by_role.values())
        for key in (
            "attempts",
            "known_prompt_tokens",
            "known_completion_tokens",
            "unknown_usage_attempts",
        )
    }
    embeddings = []
    for pattern in ("*embedding-wire.jsonl", "corpus-query-wire.jsonl"):
        for path in Path(out).rglob(pattern):
            embeddings.extend(json.loads(line) for line in path.read_bytes().splitlines())
    total.update(
        by_role=by_role,
        currency_cost=None,
        embedding_attempts=len(embeddings),
        known_embedding_prompt_tokens=sum(
            (r.get("usage") or {}).get("prompt_tokens", 0) for r in embeddings
        ),
        unknown_embedding_usage_attempts=sum(
            not isinstance(r.get("usage"), dict) for r in embeddings
        ),
    )
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--data-root", type=Path, required=True)
    plan.add_argument("--bridge", type=Path, required=True)
    plan.add_argument("--legacy", type=Path, required=True)
    plan.add_argument("--out", type=Path, required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    for command in ("run", "verify-admission", "build-adapted"):
        p = commands.add_parser(command)
        p.add_argument(
            "--spec" if command != "build-adapted" else "--config",
            dest="spec",
            type=Path,
            required=True,
        )
        p.add_argument("--approval", type=Path, required=True)
    rescore = commands.add_parser("verify-task")
    rescore.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = dataset_plan(args.data_root, args.bridge, args.legacy)
        if args.out.exists():
            raise FileExistsError("fresh plan file required")
        write_json(args.out, result)
        print(
            json.dumps(
                {"tasks": len(result["tasks"]), "sessions": result["sessions"], "quality_calls": 0}
            )
        )
    elif args.command == "freeze":
        if args.out.exists():
            raise FileExistsError("fresh frozen specification required")
        write_json(args.out, freeze_spec(read_json(args.config)))
        print(json.dumps({"spec_sha256": sha(args.out), "quality_calls": 0, "approved": False}))
    elif args.command == "verify-admission":
        verify_spec(approval_for(args.spec, args.approval, purpose="search-quality"))
        print('{"admitted":true,"quality_calls":0}')
    elif args.command == "run":
        result = execute(args.spec, args.approval)
        print(json.dumps({"status": result["status"], "calls": result["calls"]}))
        return 0 if result["quality_complete"] else 2
    elif args.command == "build-adapted":
        result = build_adapted(args.spec, args.approval)
        print(
            json.dumps({"rows": result["rows"], "complete": result["complete"], "quality_calls": 0})
        )
    else:
        result = verify_task(args.out)
        print(
            json.dumps(
                {"status": result["status"], "completed_sessions": result["completed_sessions"]}
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
