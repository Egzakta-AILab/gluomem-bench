"""Full paired MemoryArena formal reasoning driver; no implicit QA permission.

``plan`` is read-only; ``run`` requires an independently byte-bound approval.
The older arena_campaign authored <=8/120s permit is deliberately unchanged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType

from gluomem_bench.bench.error_privacy import safe_error

from . import arena_campaign as native
from .embedding_runtime import RemoteEmbedder
from .rack_trace import RackRouter
from .trajectory_suite import load_arena

FORMAL_DOMAINS = ("formal_reasoning_math", "formal_reasoning_phys")
SYSTEMS = ("no_memory", "bm25", "dense", "hybrid", "full_context", "gluomem")
PROTOCOL = "memoryarena-full-formal-native@v1"


class ArenaProductionRouter(RackRouter):
    """Scoped authless rack placeholder, never read or mutate ambient credentials."""

    def assert_api_key(self, model_name):
        entry = self._registry.resolve(model_name)
        if (
            entry.api_base != self.client.profile.base_url
            or entry.provider_model != "openai/" + self.client.profile.model
        ):
            raise ValueError("production document role model/endpoint differs")

    def lm_for(self, task):
        lm = super().lm_for(task)
        lm.kwargs["api_key"] = "local-endpoint-placeholder"
        return lm


class BudgetedArenaEmbedder(RemoteEmbedder):
    """Query/document embeddings share the same whole-task deadline as generation."""

    def __init__(self, base_url, *, budget):
        super().__init__(base_url)
        self.budget = budget

    def _encode(self, texts, role):
        self.budget.check()
        self.timeout = min(120, self.budget.remaining())
        result = super()._encode(texts, role)
        self.budget.check()
        return result


@dataclass(frozen=True)
class FrozenNativeToolCapability(native.FrozenToolCapability):
    """Actual model/tokenizer/native SDK tools wire, not a schema-lookalike probe."""

    def validate(self, profile):
        return super().validate(profile, exact_native=True)


def task_digest(task):
    return hashlib.sha256(native._json(asdict(task))).hexdigest()


def plan_dataset(data_root, pins):
    return {
        "protocol": PROTOCOL,
        "quality_calls": 0,
        "inventory": native.inventory_domains(data_root, pins),
        "tasks": {
            domain: [
                {"id": t.id, "sessions": len(t.sessions), "input_sha256": task_digest(t)}
                for t in load_arena(Path(data_root), domain=domain, pins=pins)
            ]
            for domain in FORMAL_DOMAINS
        },
    }


def _code_inventory():
    project = Path(__file__).resolve().parents[2]
    roots = (
        project / "gluomem_bench",
        project.parent / "GluoMem" / "gluomem",
        project.parent / "GluoMem" / "artifacts" / "dspy",
    )
    return [
        {"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for root in roots
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in {".py", ".sql", ".json", ".mako"}
    ]


def build_spec(
    *,
    data_root,
    pins,
    native_root,
    profile,
    output_root,
    embedding_url,
    call_budget,
    seconds_budget,
    max_tool_iterations,
    campaign_seconds,
):
    plan = plan_dataset(data_root, pins)
    native._sources(native_root)
    return {
        **plan,
        "data_root": str(Path(data_root).resolve()),
        "data_pins": pins,
        "native_root": str(Path(native_root).resolve()),
        "native_commit": native.PIN,
        "native_source_sha256": native.SOURCE_PINS,
        "output_root": str(Path(output_root).resolve()),
        "profile": profile.model_dump(mode="json"),
        "profile_sha256": hashlib.sha256(native._json(profile.model_dump(mode="json"))).hexdigest(),
        "roles": {r: profile.model for r in ("actor", "reasoning", "nativejudge", "extraction")},
        "systems": list(SYSTEMS),
        "embedding_url": embedding_url,
        "bounds": {
            "calls_per_task_method": call_budget,
            "seconds_per_task_method": seconds_budget,
            "max_tool_iterations": max_tool_iterations,
            "campaign_seconds": campaign_seconds,
            "feedback_in_memory": False,
        },
        "files": _code_inventory(),
    }


def verify_spec(spec_path, approval_path, *, allow_started=False):
    """No clients, writes or generations until ALL immutable inputs are verified."""
    from .rack import RackProfile

    raw = Path(spec_path).read_bytes()
    spec, approval = json.loads(raw), json.loads(Path(approval_path).read_bytes())
    if (
        approval.get("approved") is not True
        or approval.get("reviewer_kind") not in {"parent", "human"}
        or approval.get("spec_sha256") != hashlib.sha256(raw).hexdigest()
    ):
        raise ValueError("independent byte-bound parent approval required")
    profile = RackProfile.model_validate(spec["profile"])
    if (
        spec.get("protocol") != PROTOCOL
        or spec.get("native_commit") != native.PIN
        or spec.get("native_source_sha256") != native.SOURCE_PINS
        or spec.get("roles")
        != {r: profile.model for r in ("actor", "reasoning", "nativejudge", "extraction")}
        or spec.get("profile_sha256") != hashlib.sha256(native._json(spec["profile"])).hexdigest()
        or profile.profile_version != "role-budgets-v6"
        or profile.output_budget("answer") != 4096
        or profile.output_budget("rubric") != 4096
    ):
        raise ValueError("full native role model/profile/source contract mismatch")
    plan = plan_dataset(spec["data_root"], spec["data_pins"])
    if any(spec.get(k) != plan[k] for k in ("tasks", "inventory", "quality_calls")):
        raise ValueError("full dataset bytehash/input IDs/session coverage mismatch")
    systems = spec.get("systems", [])
    optional = systems[len(SYSTEMS) :]
    if (
        systems[: len(SYSTEMS)] != list(SYSTEMS)
        or len(set(systems)) != len(systems)
        or not set(optional) <= {"mem0", "cognee"}
    ):
        raise ValueError("complete paired method matrix required")
    bindings = spec.get("native_document_factories", {})
    if set(bindings) != set(optional):
        raise ValueError("native document factory byte-bound source/qualification required")
    for system, binding in bindings.items():
        if binding.get("kind") != "native-document-factory@v1" or binding.get("backend") != system:
            raise ValueError("native document factory real backend contract required")
        for prefix in ("source", "qualification"):
            path, digest = binding.get(prefix + "_path"), binding.get(prefix + "_sha256")
            if (
                not isinstance(path, str)
                or not Path(path).is_absolute()
                or not isinstance(digest, str)
                or len(digest) != 64
                or hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest
            ):
                raise ValueError("native document factory exact source/qualification hash required")
    bounds = spec["bounds"]
    if (
        type(bounds["calls_per_task_method"]) is not int
        or bounds["calls_per_task_method"] < 1
        or bounds["calls_per_task_method"] != profile.max_generation_calls
        or bounds["seconds_per_task_method"] != profile.max_run_seconds
        or type(bounds["max_tool_iterations"]) is not int
        or not 1 <= bounds["max_tool_iterations"] <= 8
        or type(bounds["campaign_seconds"]) not in (int, float)
        or bounds["campaign_seconds"] <= 0
        or bounds["feedback_in_memory"] is not False
    ):
        raise ValueError("explicit equal whole-task budgets required")
    if spec.get("files") != _code_inventory():
        raise ValueError("full source bytehash inventory mismatch")
    native._sources(spec["native_root"])
    if not allow_started and Path(spec["output_root"]).exists():
        raise FileExistsError("fresh full campaign root required")
    project = Path(__file__).resolve().parents[3]
    if Path(spec["output_root"]).resolve().is_relative_to(project):
        raise ValueError("external artifacts required")
    return spec


@dataclass(frozen=True)
class FrozenDatasetTask:
    """Independent approval is reread at task admission; no self-attested boolean."""

    spec_path: Path
    approval_path: Path

    def validate(self, task, profile, max_tool_iterations):
        spec = verify_spec(self.spec_path, self.approval_path, allow_started=True)
        if spec["profile"] != profile.model_dump(mode="json"):
            raise ValueError("frozen profile differs")
        if spec["bounds"]["max_tool_iterations"] != max_tool_iterations:
            raise ValueError("frozen tool iteration bound differs")
        admitted = spec["tasks"].get(task.domain, [])
        if {
            "id": task.id,
            "sessions": len(task.sessions),
            "input_sha256": task_digest(task),
        } not in admitted:
            raise ValueError("task differs from frozen native dataset input")
        return spec


class _AuthoredMemoryPermit:
    def validate(self, task, profile, max_tool_iterations):
        if task != native.AUTHORED_LIVE_TASK or max_tool_iterations != 1:
            raise ValueError("authored-only memory binding qualification; no dataset QA")


class FullArenaTransport(native.ArenaTransport):
    """Separate full permit: native guard unchanged, not the legacy 8/120 gate.

    This controls the native OpenAI SDK call surface through scoped injection;
    only declared reasoning calls reach the native loop, so coding cannot execute.
    """

    def __init__(
        self,
        profile,
        *,
        receipt,
        budget,
        permit,
        task,
        capability,
        transport=None,
        max_tool_iterations=3,
    ):
        from .parallel_campaign import JobContext
        from .rack import RackClient

        if (
            type(permit) is not FrozenDatasetTask
            or type(capability) is not FrozenNativeToolCapability
        ):
            raise ValueError(
                "frozen full dataset approval and exact native tool capability required"
            )
        spec = permit.validate(task, profile, max_tool_iterations)
        verify_memory_qualification(spec.get("production_memory_qualification"), profile)
        if (
            type(budget) is not JobContext
            or budget.calls != 0
            or budget.job.call_budget != spec["bounds"]["calls_per_task_method"]
            or budget.job.seconds_budget != spec["bounds"]["seconds_per_task_method"]
            or budget.limiter.max_in_flight != 1
        ):
            raise ValueError("full whole-task budget differs from frozen common bounds")
        self.permit, self.live, self.live_transport_kind = (
            permit,
            True,
            "direct-http" if transport is None else "injected-test-transport",
        )
        self.qualification, self._identity_ready = capability.validate(profile), False
        RackClient.__init__(self, profile, receipt=receipt, budget=budget, transport=transport)

    @classmethod
    def authored(cls, profile, *, receipt, budget, transport=None):
        from .parallel_campaign import JobContext
        from .rack import RackClient

        if (
            type(budget) is not JobContext
            or budget.job.call_budget > 8
            or budget.job.seconds_budget > 300
            or budget.budget.max_calls > 8
            or budget.budget.max_seconds > 300
            or budget.calls != 0
            or budget.limiter.max_in_flight != 1
        ):
            raise ValueError("authored qualification requires fresh <=8 calls/300s shared envelope")
        obj = cls.__new__(cls)
        obj.permit, obj.live, obj.qualification, obj._identity_ready = (
            _AuthoredMemoryPermit(),
            True,
            None,
            False,
        )
        obj.live_transport_kind = "direct-http" if transport is None else "injected-test-transport"
        RackClient.__init__(obj, profile, receipt=receipt, budget=budget, transport=transport)
        return obj

    def _request(self, method, path, payload=None):
        if path == "/chat/completions":
            payload = {**payload, "stream": False, "n": 1}
        return super()._request(method, path, payload)

    def _latch(self, exc):
        from .rack import SharedRackError

        super()._latch(exc)
        if isinstance(exc, SharedRackError) or getattr(exc, "shared_failure", False):
            # JobContext deliberately isolates normal job failures. Shared
            # infrastructure/identity faults must additionally stop admission,
            # without clearing the failed job or manufacturing later QA zeros.
            self.budget.budget.failed = True


class RawExperienceMemory:
    """Online native agent experience controls, isolated per task and method."""

    def __init__(self, mode, *, budget, embedder=None, top_k=10):
        import sqlite3

        if mode not in set(SYSTEMS) - {"gluomem"} or type(top_k) is not int or top_k < 1:
            raise ValueError("unsupported raw memory mode")
        if mode in {"dense", "hybrid"} and embedder is None:
            raise ValueError("explicit real shared encoder required")
        self.mode, self.budget, self.entries = mode, budget, []
        self.embedder, self.top_k, self.vectors = embedder, top_k, []
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE VIRTUAL TABLE docs USING fts5(content)")

    def query(self, question):
        self.budget.check()
        if self.mode == "no_memory" or not self.entries:
            return ""
        if self.mode == "full_context":
            return "\n\n".join(e["content"] for e in self.entries)
        import re

        tokens = re.findall(r"\w+", question)
        query = " OR ".join('"' + t.replace('"', '""') + '"' for t in tokens)
        bm25 = (
            [
                row[0] - 1
                for row in self.db.execute(
                    "SELECT rowid FROM docs WHERE docs MATCH ? ORDER BY bm25(docs), rowid", (query,)
                )
            ]
            if query
            else []
        )
        ranked = bm25
        if self.mode in {"dense", "hybrid"}:
            import numpy as np

            matrix = np.asarray(self.vectors, dtype=float)
            q = np.asarray(self.embedder.embed_query(question), dtype=float)
            norm = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
            sims = np.divide(matrix @ q, norm, out=np.zeros(len(matrix)), where=norm != 0)
            dense = sorted(range(len(matrix)), key=lambda i: (-sims[i], i))
            if self.mode == "dense":
                ranked = dense
            else:
                scores = {}
                for ranks in (bm25, dense):
                    for rank, i in enumerate(ranks):
                        scores[i] = scores.get(i, 0) + 1 / (60 + rank + 1)
                ranked = sorted(scores, key=lambda i: (-scores[i], i))
        return "\n\n".join(self.entries[i]["content"] for i in ranked[: self.top_k])

    def insert(self, experience):
        self.budget.check()
        self.entries.append(deepcopy(experience))
        self.db.execute("INSERT INTO docs(content) VALUES (?)", (experience["content"],))
        self.db.commit()
        if self.mode in {"dense", "hybrid"}:
            self.vectors.append(self.embedder.embed_batch([experience["content"]])[0])
        readback = self.db.execute(
            "SELECT content FROM docs WHERE rowid=?", (len(self.entries),)
        ).fetchone()
        if readback != (experience["content"],):
            raise ValueError("raw experience readback mismatch")
        return {
            "event_id": experience["source_ref"],
            "content_sha256": hashlib.sha256(experience["content"].encode()).hexdigest(),
            "readback_verified": True,
        }

    def close(self):
        self.db.close()


def official_metrics(source, tasks, out):
    """Unmodified native per-task progress, final-task success and passrate_at_k."""
    native._verify_sources(source)
    mod = ModuleType("_arena_full_metrics")
    exec(
        compile(source[native.EVAL], "pinned-git:" + native.PIN + "/" + native.EVAL, "exec"),
        mod.__dict__,
    )
    logger = logging.getLogger(__name__)
    data = {key: (rows, len(rows)) for key, rows in tasks.items()}
    if not data or any(not rows or len(rows) > 20 for rows in tasks.values()):
        raise ValueError("complete 1..20 session native task logs required")
    papers, passed, counts, minimum = mod.get_passrate_at_k(data, logger=logger)
    mod.print_result(papers, passed, counts, minimum, str(out), logger=logger)
    return json.loads((Path(out) / "all_results.json").read_bytes())


def run_native_task(task, *, source, transport, memory, out, max_tool_iterations):
    """Real native MathAgent+MathEnvironment for both upstream formal configs.

    Gold goes only to env.step; memory receives native build_memory_entry output.
    Failure never becomes a score. The campaign owns continuation to the next task.
    """
    native._verify_sources(source)
    if (
        task.domain not in FORMAL_DOMAINS
        or not 1 <= len(task.sessions) <= 20
        or len(task.sessions) != len(task.gold)
        or type(max_tool_iterations) is not int
        or not 1 <= max_tool_iterations <= 8
    ):
        raise ValueError("aligned full native task and equal whole-task budget required")
    if transport.live:
        if not isinstance(transport, FullArenaTransport):
            raise ValueError("legacy authored permit cannot admit full benchmark tasks")
        transport.permit.validate(task, transport.profile, max_tool_iterations)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for name, material in source.items():
        path = out / "native-source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(material)
    (out / "input-task.json").write_bytes(native._json(asdict(task)))
    (out / "effective-profile.json").write_bytes(
        native._json(transport.profile.model_dump(mode="json"))
    )
    rows, closed, attempted_index = [], False, None
    try:
        if callable(memory):
            memory = memory(transport.budget, out)
        if memory.budget is not transport.budget:
            raise ValueError("shared whole-task memory budget required")
        with native.native_math(source, transport, max_tool_iterations=max_tool_iterations) as (
            actor,
            env,
        ):
            try:
                env.reset(seed=0)
                transport.preflight()
                for index, (session, gold) in enumerate(zip(task.sessions, task.gold, strict=True)):
                    attempted_index = index
                    started = time.perf_counter()
                    transport.check()
                    query = actor.build_prompt(session.question, background=session.background)
                    context = memory.query(query)
                    action = actor.act(
                        actor.build_prompt(
                            session.question, background=session.background, memory_context=context
                        )
                    )
                    transport.check()
                    observation, reward, info = env.step(action, ground_truth=gold, need_judge=True)
                    entry = actor.build_memory_entry(
                        session.question, action, observation, reward=None
                    )
                    experience = {
                        "content": entry,
                        "authorship": "agent",
                        "source_ref": f"memoryarena/{task.domain}/{task.id}/session/{index}",
                        "session_index": index,
                        "feedback_in_memory": False,
                        "native_source_commit": native.PIN,
                    }
                    readback = memory.insert(experience)
                    if readback.get("readback_verified") is not True:
                        raise ValueError("memory committed readback proof missing")
                    row = {
                        "query_id": index,
                        "response": observation["final"],
                        "is_correct": reward,
                        "memory_context": context,
                        "time": time.perf_counter() - started,
                        "action": action,
                        "observation": observation,
                        "experience": experience,
                        "memory_readback": readback,
                        "post_insert_readback": memory.query(query),
                    }
                    native._append(out / "steps.jsonl", row)
                    rows.append(row)
            finally:
                env.close()
                memory.close()
                closed = env.history == [] and env._current_observation is None
        transport.check()
        metrics = official_metrics(source, {str(task.id): rows}, out / "scores")
        result = {
            "protocol": PROTOCOL,
            "status": "complete",
            "task_id": task.id,
            "domain": task.domain,
            "input_sha256": task_digest(task),
            "completed_sessions": len(rows),
            "expected_sessions": len(task.sessions),
            "calls": transport.budget.calls,
            "native_closed": closed,
            "metrics": metrics,
            "native_verifier": "MathEnvironment.judge",
            "max_tool_iterations": max_tool_iterations,
            "feedback_in_memory": False,
            "transport": "live" if transport.live else "offline-test-only",
            "artifact_sha256": {
                str(p.relative_to(out)).replace("\\", "/"): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in sorted(out.rglob("*"))
                if p.is_file()
            },
        }
        transport.check()
        (out / "result.pending.json").write_bytes(native._json(result))
        verify_task(out, manifest_name="result.pending.json")
        transport.check()
        (out / "result.pending.json").replace(out / "result.json")
        return result
    except BaseException as exc:
        for name in ("result.json", "result.pending.json"):
            (out / name).unlink(missing_ok=True)
        (out / "failure.json").write_bytes(
            native._json(
                {
                    "status": "failed",
                    "completed_sessions": len(rows),
                    "expected_sessions": len(task.sessions),
                    "calls": transport.budget.calls,
                    "native_closed": closed,
                    "error": safe_error(exc),
                    "sessions": [
                        {
                            "id": i,
                            "status": "complete"
                            if i < len(rows)
                            else ("failed" if i == attempted_index else "unrun"),
                        }
                        for i in range(len(task.sessions))
                    ],
                }
            )
        )
        raise RuntimeError("full native task failed; see failure.json") from exc


def verify_task(out, *, manifest_name="result.json"):
    """Dataset/network-free source, wire, step, readback and official scorer replay."""
    out = Path(out)
    if manifest_name not in {"result.json", "result.pending.json"}:
        raise ValueError("unsupported native completion manifest")
    result = json.loads((out / manifest_name).read_bytes())
    if (
        result["protocol"] != PROTOCOL
        or result["status"] != "complete"
        or not result["native_closed"]
        or (out / "failure.json").exists()
    ):
        raise ValueError("incomplete full task bundle")
    for name, digest in result["artifact_sha256"].items():
        path = (out / name).resolve()
        if (
            not path.is_relative_to(out.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("artifact hash mismatch: " + name)
    source = {name: (out / "native-source" / name).read_bytes() for name in native.SOURCE_PINS}
    native._verify_sources(source)
    raw = json.loads((out / "input-task.json").read_bytes())
    if hashlib.sha256(native._json(raw)).hexdigest() != result["input_sha256"]:
        raise ValueError("input hash mismatch")
    rows = [json.loads(line) for line in (out / "steps.jsonl").read_bytes().splitlines()]
    if (
        len(rows) != result["expected_sessions"]
        or len(rows) != result["completed_sessions"]
        or len(rows) != len(raw["sessions"])
        or [r["query_id"] for r in rows] != list(range(len(rows)))
    ):
        raise ValueError("full session coverage mismatch")
    wire = [json.loads(line) for line in (out / "http.jsonl").read_bytes().splitlines()]
    generations = [w for w in wire if w.get("url", "").endswith("/chat/completions")]
    production = (
        [json.loads(line) for line in (out / "production-raw.jsonl").read_bytes().splitlines()]
        if (out / "production-raw.jsonl").exists()
        else []
    )
    if (
        len(generations) + len(production) != result["calls"]
        or any("error" in r or r.get("status") != 200 for r in generations)
        or any("error" in r or not r.get("response") for r in production)
    ):
        raise ValueError("actual native + production generation accounting mismatch")
    judges = [
        w
        for w in generations
        if w["request"]["messages"][0]["content"]
        == (
            "You are a helpful assistant that judges the equivalence "
            "of two mathematical expressions."
        )
    ]
    if len(judges) != len(rows):
        raise ValueError("native judge wire coverage mismatch")
    # Rebuild entries from original native method without constructing any SDK.
    with native.native_math(
        source, _OfflineIdentity(json.loads((out / "effective-profile.json").read_bytes()))
    ) as (actor, env):
        for row, judge in zip(rows, judges, strict=True):
            verdict = judge["response"]["choices"][0]["message"]["content"].strip().lower()
            experience = row["experience"]
            if (
                verdict not in {"yes", "no"}
                or type(row["is_correct"]) is not bool
                or row["is_correct"] != (verdict == "yes")
                or row["observation"]["judge_result"] != verdict
                or row["observation"]["final"] != row["response"]
                or row["memory_readback"].get("readback_verified") is not True
                or experience["authorship"] != "agent"
                or experience["feedback_in_memory"] is not False
                or experience["content"]
                != actor.build_memory_entry(
                    raw["sessions"][row["query_id"]]["question"],
                    row["action"],
                    row["observation"],
                    reward=None,
                )
            ):
                raise ValueError("native feedback/entry/readback differs")
        env.close()
    replay = out / ("rescore-" + uuid.uuid4().hex)
    metrics = official_metrics(source, {str(result["task_id"]): rows}, replay)
    if metrics != result["metrics"]:
        raise ValueError("official full scorer differs")
    return metrics


class _OfflineIdentity:
    """Construct pinned verifier methods without network clients during rescore."""

    def __init__(self, profile):
        from .rack import RackProfile

        self.profile = RackProfile.model_validate(profile)

    def sdk(self, role):
        return None


class GluoExperienceMemory:
    """Production document SourceEvents, DspyExtractors, admission and HybridSearch.

    Native agent authorship is retained as a document, never relabeled as a user.
    The readback includes committed envelope hashes and frame provenance, even
    when extraction legitimately produces no admissible claims.
    """

    def __init__(self, *, cfg, extractors, embedder, budget, db_path, evidence_path, top_k=10):
        from gluomem.frames.store import FrameStore
        from gluomem.kg.graph import KnowledgeGraph
        from gluomem.llm.modules import DspyExtractors
        from gluomem.search.hybrid import HybridSearch
        from gluomem.storage.db import MindDB

        if type(extractors) is not DspyExtractors:
            raise ValueError("real production DspyExtractors required; no Null/Fake extractor")
        self.cfg, self.extractors, self.embedder, self.budget = cfg, extractors, embedder, budget
        self.path, self.evidence_path, self.top_k = Path(db_path), Path(evidence_path), top_k
        if self.path.exists() or self.evidence_path.exists():
            raise FileExistsError("fresh online memory and evidence required")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = MindDB(self.path, dim=cfg.storage.vector_dim)
        self.store = FrameStore(
            self.db, embedder=embedder, chunking=cfg.chunking, lifecycle=cfg.lifecycle
        )
        self.kg = KnowledgeGraph(self.db)
        self.search = HybridSearch(
            self.db, self.store, cfg, embedder=embedder, reranker=None, kg=self.kg
        )
        self.events, self.receipts = [], []

    def query(self, question):
        self.budget.check()
        if not self.events:
            return ""
        result = self.search.search(question, limit=self.top_k)
        self.budget.check()
        return "" if result.abstained else "\n\n".join(hit.frame.content for hit in result.hits)

    def insert(self, experience):
        import sqlite3

        from gluomem.write.events import ingest_source_event

        from .beam_ingest import _read_rows, _validate_rows

        self.budget.check()
        if (
            experience.get("authorship") != "agent"
            or experience.get("feedback_in_memory") is not False
        ):
            raise ValueError("native agent-authored document without evaluator feedback required")
        event = native.document_experience(experience)
        result = ingest_source_event(
            event, store=self.store, kg=self.kg, cfg=self.cfg, extractors=self.extractors
        )
        events = [*self.events, event]
        committed, retention = _validate_rows(_read_rows(self.path), events)
        if (
            result.deduplicated
            or committed[:-1] != self.receipts
            or result.event_row_id != committed[-1]["row_id"]
        ):
            raise ValueError("production source-event commit/readback mismatch")
        reader = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True)
        reader.row_factory = sqlite3.Row
        try:
            provenance = [
                dict(r)
                for r in reader.execute(
                    "SELECT frame_id, source_event_id, source_type, source_ref, "
                    "message_id, verbatim_span "
                    "FROM memory_provenance WHERE source_event_id=? ORDER BY frame_id",
                    (result.event_row_id,),
                )
            ]
            frame_ids = {f.id for f in result.frames}
            if not frame_ids <= {r["frame_id"] for r in provenance} or any(
                r["source_type"] != "document"
                or r["source_ref"] != event.source_ref
                or r["message_id"] is not None
                for r in provenance
            ):
                raise ValueError("production document frame provenance mismatch")
        finally:
            reader.close()
        receipt = {
            **committed[-1],
            "readback_verified": True,
            "provenance_count": len(provenance),
            "provenance": provenance,
            "content_sha256": hashlib.sha256(experience["content"].encode()).hexdigest(),
            "frames": [
                {"id": f.id, "content": f.content, "source": f.source, "kind": f.kind}
                for f in result.frames
            ],
        }
        native._append(
            self.evidence_path, {"event": event.model_dump(mode="json"), "receipt": receipt}
        )
        self.events, self.receipts = events, committed
        self.budget.check()
        return receipt

    def close(self):
        self.db.close()


def execute_matrix(
    tasks,
    *,
    systems,
    out,
    call_budget,
    seconds_budget,
    campaign_seconds,
    run_one,
    completion_owner="matrix",
):
    """Pair whole tasks; only the explicit completion owner publishes authority.

    Standalone/authored callers retain matrix-owned completion by default. Formal
    execution defers authority until aggregate scores and evidence are verified.
    """
    if completion_owner not in {"matrix", "formal"}:
        raise ValueError("explicit matrix or formal completion owner required")
    from .parallel_campaign import CampaignBudget, CampaignStopped, Job, JobContext, RequestLimiter

    tasks, systems, out = list(tasks), list(systems), Path(out)
    keys = [(t.domain, t.id) for t in tasks]
    if (
        not tasks
        or len(keys) != len(set(keys))
        or not systems
        or len(systems) != len(set(systems))
        or not set(systems) <= set(SYSTEMS) | {"mem0", "cognee"}
    ):
        raise ValueError("unique paired task/method matrix required")
    outer = CampaignBudget(
        max_calls=len(tasks) * len(systems) * call_budget, max_seconds=campaign_seconds
    )
    limiter = RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072)
    out.mkdir(parents=True, exist_ok=False)
    matrix = [
        {
            "task_id": t.id,
            "domain": t.domain,
            "system": system,
            "input_sha256": task_digest(t),
            "expected_sessions": len(t.sessions),
            "completed_sessions": 0,
            "status": "unrun",
            "calls": 0,
        }
        for t in tasks
        for system in systems
    ]
    (out / "frozen-matrix.json").write_bytes(native._json(matrix))
    for row in matrix:
        task = next(t for t in tasks if (t.domain, t.id) == (row["domain"], row["task_id"]))
        target = out / row["domain"] / str(row["task_id"]) / row["system"]
        job = Job(
            f"{row['domain']}:{row['task_id']}:{row['system']}",
            f"{row['domain']}:{row['task_id']}",
            target,
            call_budget,
            seconds_budget,
        )
        context = None
        try:
            outer.check()
            if not outer.reserve(job):
                raise CampaignStopped("campaign whole-task reservation unavailable")
            context = JobContext(job, outer, limiter)
            result = run_one(task, row["system"], context, target)
            if result.get("status") != "complete" or result.get("completed_sessions") != len(
                task.sessions
            ):
                raise ValueError("incomplete task result cannot become score")
            row.update(status="complete", completed_sessions=len(task.sessions), output=str(target))
        except CampaignStopped as exc:
            row.update(error=safe_error(exc), status="failed" if context is not None else "unrun")
        except (Exception, native.NativeAbort) as exc:
            row.update(
                status="failed", error=safe_error(exc), output=str(target)
            )
            if (target / "failure.json").is_file():
                row["completed_sessions"] = json.loads((target / "failure.json").read_bytes()).get(
                    "completed_sessions", 0
                )
        finally:
            if context is not None:
                row["calls"] = context.calls
                outer.release(context)
            (out / "task-statuses.json").write_bytes(native._json(matrix))
    result = {
        "protocol": PROTOCOL,
        "tasks": matrix,
        "completion_owner": completion_owner,
        "authoritative": completion_owner == "matrix",
        "matrix_complete": all(r["status"] == "complete" for r in matrix),
        "quality_complete": completion_owner == "matrix"
        and all(r["status"] == "complete" for r in matrix),
        "expected_sessions": sum(r["expected_sessions"] for r in matrix),
        "completed_sessions": sum(r["completed_sessions"] for r in matrix),
        "calls": outer.calls,
        "whole_budget": {
            "calls_per_task_method": call_budget,
            "seconds_per_task_method": seconds_budget,
            "campaign_seconds": campaign_seconds,
        },
    }
    (out / "campaign-result.json").write_bytes(native._json(result))
    return result


def qualify_authored_memory(*, native_root, out, profile, embedding_url):
    """One fixed authored task, real native loop + production memory, <=8/300s.

    No dataset questions are loaded or generated. All native and production
    extraction/consolidation calls spend the same fresh sequential counter.
    """
    if (
        profile.profile_version != "role-budgets-v6"
        or profile.output_budget("answer") != 4096
        or profile.output_budget("rubric") != 4096
    ):
        raise ValueError("role-budgets-v6 reader/nativejudge4096 qualification profile required")
    from gluomem.llm.modules import DspyExtractors

    from .parallel_campaign import CampaignBudget, Job, JobContext, RequestLimiter
    from .rack import controlled_config

    source, out = native._sources(native_root), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    task_out = out / "task"
    job = Job("authored-real-memory", "authored-real-memory", task_out, 8, 300)
    outer = CampaignBudget(max_calls=8, max_seconds=300)
    outer.reserve(job)
    budget = JobContext(job, outer, RequestLimiter(max_in_flight=1, max_in_flight_tokens=131072))
    client = FullArenaTransport.authored(profile, receipt=task_out / "http.jsonl", budget=budget)
    encoder = BudgetedArenaEmbedder(embedding_url, budget=budget)

    def factory(context, target):
        cfg = controlled_config(
            profile,
            data=target / "authored.txt",
            sha256=task_digest(native.AUTHORED_LIVE_TASK),
            embedder=encoder,
        )
        (target / "effective-config.json").write_bytes(native._json(cfg.model_dump(mode="json")))
        (target / "encoder.json").write_bytes(native._json(encoder.fingerprint.as_dict()))
        router = ArenaProductionRouter(
            cfg, client=client, trace=target / "production-raw.jsonl", budget=context
        )
        return GluoExperienceMemory(
            cfg=cfg,
            extractors=DspyExtractors(cfg, router=router),
            embedder=encoder,
            budget=context,
            db_path=target / "production.mind",
            evidence_path=target / "memory-events.jsonl",
        )

    try:
        run_native_task(
            native.AUTHORED_LIVE_TASK,
            source=source,
            transport=client,
            memory=factory,
            out=task_out,
            max_tool_iterations=1,
        )
        row = json.loads((task_out / "steps.jsonl").read_bytes().splitlines()[0])
        if (
            not row["action"].get("tool_trace")
            or row["action"]["tool_trace"][0]["tool"] != "reasoning"
            or not row["post_insert_readback"]
            or row["memory_readback"]["provenance_count"] < 1
        ):
            raise ValueError(
                "authored native reasoning and production retrieval/provenance proof required"
            )
        wire = [json.loads(line) for line in (task_out / "http.jsonl").read_bytes().splitlines()]
        probe = wire[:3]
        if len(probe) != 3 or not probe[-1]["url"].endswith("/chat/completions"):
            raise ValueError("first exact native tool wire absent")
        usage = probe[-1]["response"]["usage"]
        proof = {
            "generation_attempts": 1,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "finish_reason": probe[-1]["response"]["choices"][0]["finish_reason"],
            "tool_names": [
                c["function"]["name"]
                for c in probe[-1]["response"]["choices"][0]["message"]["tool_calls"]
            ],
            "tool_was_executed": False,
            "tools_sha256": hashlib.sha256(
                json.dumps(probe[-1]["request"]["tools"], sort_keys=True).encode()
            ).hexdigest(),
        }
        receipt_path, wire_path = out / "native-tool-receipt.json", out / "native-tool-wire.jsonl"
        # This extracted first-call probe did not execute its tool yet; the full
        # task bundle separately proves the later native tool execution and spend.
        receipt_path.write_bytes(
            native._json(
                {
                    "accounting": {"calls": 1},
                    "jobs": [{"status": "complete", "calls": 1, "result": proof}],
                    "scope": "first native actor call only",
                }
            )
        )
        wire_path.write_bytes(b"".join(json.dumps(r).encode() + b"\n" for r in probe))
        cap = FrozenNativeToolCapability(
            receipt_path,
            hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            wire_path,
            hashlib.sha256(wire_path.read_bytes()).hexdigest(),
        )
        cap.validate(profile)
        qualification = {
            "protocol": PROTOCOL,
            "kind": "authored-native-production-memory-NOT-benchmark",
            "complete": True,
            "calls": budget.calls,
            "max_calls": 8,
            "max_seconds": 300,
            "task_bundle": str(task_out.resolve()),
            "task_result_sha256": hashlib.sha256(
                (task_out / "result.json").read_bytes()
            ).hexdigest(),
            "capability": {
                "receipt_path": str(receipt_path.resolve()),
                "receipt_sha256": cap.receipt_sha256,
                "wire_path": str(wire_path.resolve()),
                "wire_sha256": cap.wire_sha256,
            },
            "profile": profile.model_dump(mode="json"),
            "production_readback": row["memory_readback"],
            "retrieved_context": row["post_insert_readback"],
        }
        verify_task(task_out)
        (out / "qualification.json").write_bytes(native._json(qualification))
        return qualification
    except (Exception, native.NativeAbort) as exc:
        (out / "qualification-failure.json").write_bytes(
            native._json(
                {
                    "complete": False,
                    "calls": budget.calls,
                    "error": safe_error(exc),
                    "kind": "authored-native-production-memory-NOT-benchmark",
                }
            )
        )
        raise
    finally:
        client.close()


def verify_memory_qualification(binding, profile):
    """Read-only prerequisite proof, not permission to run a dataset."""
    if not isinstance(binding, dict) or not {"path", "sha256"} <= binding.keys():
        raise ValueError("production memory qualification byte binding required")
    raw = Path(binding["path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != binding["sha256"]:
        raise ValueError("production memory qualification hash differs")
    proof = json.loads(raw)
    if (
        proof.get("protocol") != PROTOCOL
        or proof.get("complete") is not True
        or proof.get("kind") != "authored-native-production-memory-NOT-benchmark"
        or type(proof.get("calls")) is not int
        or not 1 <= proof["calls"] <= 8
        or not proof.get("retrieved_context")
        or proof["production_readback"].get("provenance_count", 0) < 1
    ):
        raise ValueError("real production memory qualification incomplete")
    admitted = proof["profile"]
    actual = profile.model_dump(mode="json")
    if any(
        admitted[k] != actual[k]
        for k in (
            "model",
            "base_url",
            "context_window",
            "temperature",
            "generation_budgets",
            "profile_version",
        )
    ):
        raise ValueError("qualified model/role profile differs")
    bundle = Path(proof["task_bundle"])
    if (
        hashlib.sha256((bundle / "result.json").read_bytes()).hexdigest()
        != proof["task_result_sha256"]
    ):
        raise ValueError("production memory task bundle hash differs")
    verify_task(bundle)
    record = json.loads((bundle / "result.json").read_bytes())
    rows = [json.loads(line) for line in (bundle / "steps.jsonl").read_bytes().splitlines()]
    if (
        record["transport"] != "live"
        or record["calls"] != proof["calls"]
        or rows[0]["memory_readback"] != proof["production_readback"]
        or rows[0]["post_insert_readback"] != proof["retrieved_context"]
    ):
        raise ValueError("production memory qualification contradicts actual steps")
    # SourceEvent database and provenance are independently reread, not trusted
    # from a claimed readback_verified field in the step/qualification JSON.
    from .beam_ingest import _read_rows, _validate_rows

    events = [native.document_experience(row["experience"]) for row in rows]
    committed, _ = _validate_rows(_read_rows(bundle / "production.mind"), events)
    saved = [
        json.loads(line) for line in (bundle / "memory-events.jsonl").read_bytes().splitlines()
    ]
    import sqlite3

    conn = sqlite3.connect((bundle / "production.mind").resolve().as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        for row, event, commit, envelope in zip(rows, events, committed, saved, strict=True):
            if (
                envelope["event"] != event.model_dump(mode="json")
                or envelope["receipt"] != row["memory_readback"]
            ):
                raise ValueError("production document envelope/readback differs")
            if any(row["memory_readback"].get(k) != v for k, v in commit.items()):
                raise ValueError("production committed source receipt differs")
            provenance = [
                dict(r)
                for r in conn.execute(
                    "SELECT frame_id, source_event_id, source_type, source_ref, "
                    "message_id, verbatim_span "
                    "FROM memory_provenance WHERE source_event_id=? ORDER BY frame_id",
                    (commit["row_id"],),
                )
            ]
            if provenance != row["memory_readback"]["provenance"]:
                raise ValueError("production frame provenance differs")
    finally:
        conn.close()
    cap = proof["capability"]
    FrozenNativeToolCapability(
        Path(cap["receipt_path"]), cap["receipt_sha256"], Path(cap["wire_path"]), cap["wire_sha256"]
    ).validate(profile)
    return proof


def execute_formal(spec_path, approval_path):
    """Execute ALL frozen Math+Phys tasks/all sessions paired across methods.

    No quality calls occur until approval, source/data pins and the actual
    native+production qualification have all been verified, read-only.
    """
    spec = verify_spec(spec_path, approval_path)
    from gluomem.llm.modules import DspyExtractors

    from .rack import RackProfile, controlled_config

    profile = RackProfile.model_validate(spec["profile"])
    proof = verify_memory_qualification(spec.get("production_memory_qualification"), profile)
    # Optional real Mem0 stock-document / Cognee bridges are explicit interfaces,
    # never fabricated stand-ins. Independently frozen code MUST expose
    # verify_qualification(path, profile=...) that rereads native stores and wire,
    # plus create_memory(profile, client, budget, embedder, out). Returned memory
    # implements query(str), insert(native_experience)->committed readback,
    # close(), budget. Assistant-to-user chat conversion is forbidden.
    factories = {}
    for system, binding in spec.get("native_document_factories", {}).items():
        import importlib.util

        module_spec = importlib.util.spec_from_file_location(
            "_arena_native_document_" + system, binding["source_path"]
        )
        if module_spec is None or module_spec.loader is None:
            raise ValueError("native document factory import specification unavailable")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        module.verify_qualification(Path(binding["qualification_path"]), profile=profile)
        if not callable(getattr(module, "create_memory", None)):
            raise ValueError("native document factory create_memory interface missing")
        factories[system] = module.create_memory
    cap = proof["capability"]
    capability = FrozenNativeToolCapability(
        Path(cap["receipt_path"]), cap["receipt_sha256"], Path(cap["wire_path"]), cap["wire_sha256"]
    )
    source = native._sources(spec["native_root"])
    tasks = [
        t
        for domain in FORMAL_DOMAINS
        for t in load_arena(Path(spec["data_root"]), domain=domain, pins=spec["data_pins"])
    ]

    permit = FrozenDatasetTask(Path(spec_path), Path(approval_path))
    root = Path(spec["output_root"])

    def run_one(task, system, context, target):
        encoder = BudgetedArenaEmbedder(spec["embedding_url"], budget=context)
        if not (root / "execution-spec.json").exists():
            (root / "execution-spec.json").write_bytes(Path(spec_path).read_bytes())
            (root / "execution-approval.json").write_bytes(Path(approval_path).read_bytes())
            import zipfile

            with zipfile.ZipFile(root / "frozen-source.zip", "x", zipfile.ZIP_DEFLATED) as archive:
                for record in spec["files"]:
                    path = Path(record["path"])
                    archive.write(path, arcname=str(path).split("egzakta-org", 1)[-1].lstrip("/\\"))
        client = FullArenaTransport(
            profile,
            receipt=target / "http.jsonl",
            budget=context,
            permit=permit,
            task=task,
            capability=capability,
            max_tool_iterations=spec["bounds"]["max_tool_iterations"],
        )

        def factory(budget, output):
            if system in factories:
                return factories[system](
                    profile=profile, client=client, budget=budget, embedder=encoder, out=output
                )
            if system != "gluomem":
                return RawExperienceMemory(system, budget=budget, embedder=encoder, top_k=10)
            cfg = controlled_config(
                profile,
                data=Path(spec["data_root"]) / task.domain / "data.jsonl",
                sha256=spec["data_pins"][task.domain + "/data.jsonl"],
                embedder=encoder,
            )
            (output / "effective-config.json").write_bytes(
                native._json(cfg.model_dump(mode="json"))
            )
            (output / "encoder.json").write_bytes(native._json(encoder.fingerprint.as_dict()))
            router = ArenaProductionRouter(
                cfg, client=client, trace=output / "production-raw.jsonl", budget=budget
            )
            return GluoExperienceMemory(
                cfg=cfg,
                extractors=DspyExtractors(cfg, router=router),
                embedder=encoder,
                budget=budget,
                db_path=output / "production.mind",
                evidence_path=output / "memory-events.jsonl",
            )

        try:
            return run_native_task(
                task,
                source=source,
                transport=client,
                memory=factory,
                out=target,
                max_tool_iterations=spec["bounds"]["max_tool_iterations"],
            )
        finally:
            client.close()

    bounds = spec["bounds"]
    result = execute_matrix(
        tasks,
        systems=spec["systems"],
        out=root,
        call_budget=bounds["calls_per_task_method"],
        seconds_budget=bounds["seconds_per_task_method"],
        campaign_seconds=bounds["campaign_seconds"],
        run_one=run_one,
        completion_owner="formal",
    )
    try:
        metrics = {}
        for domain in FORMAL_DOMAINS:
            metrics[domain] = {}
            for system in spec["systems"]:
                rows = [
                    r for r in result["tasks"] if r["domain"] == domain and r["system"] == system
                ]
                complete = bool(rows) and all(r["status"] == "complete" for r in rows)
                record = {
                    "status": "complete" if complete else "incomplete-no-full-score",
                    "expected_tasks": len(rows),
                    "completed_tasks": sum(r["status"] == "complete" for r in rows),
                }
                if complete:
                    logs = {}
                    for row in rows:
                        verify_task(row["output"])
                        logs[str(row["task_id"])] = [
                            json.loads(line)
                            for line in (Path(row["output"]) / "steps.jsonl")
                            .read_bytes()
                            .splitlines()
                        ]
                    score_out = root / "scores" / domain / system
                    record["official_metrics"] = official_metrics(source, logs, score_out)
                    if (
                        json.loads((score_out / "all_results.json").read_bytes())
                        != record["official_metrics"]
                    ):
                        raise ValueError("aggregate official score readback differs")
                metrics[domain][system] = record
        # Reread independent admission and the exact execution evidence before
        # binding it. No leaf completion is itself whole-campaign authority.
        if verify_spec(spec_path, approval_path, allow_started=True) != spec:
            raise ValueError("final frozen specification differs")
        spec_bytes = Path(spec_path).read_bytes()
        if (root / "execution-spec.json").read_bytes() != spec_bytes or (
            root / "execution-approval.json"
        ).read_bytes() != Path(approval_path).read_bytes():
            raise ValueError("final execution spec/approval binding differs")
        import zipfile

        with zipfile.ZipFile(root / "frozen-source.zip") as archive:
            for record in spec["files"]:
                name = str(Path(record["path"])).split("egzakta-org", 1)[-1].lstrip("/\\")
                name = zipfile.ZipInfo.from_file(record["path"], arcname=name).filename
                if hashlib.sha256(archive.read(name)).hexdigest() != record["sha256"]:
                    raise ValueError("final frozen source archive binding differs")
        evidence = [
            root / name
            for name in (
                "execution-spec.json",
                "execution-approval.json",
                "frozen-source.zip",
                "frozen-matrix.json",
                "task-statuses.json",
            )
        ]
        evidence += [
            Path(r["output"]) / "result.json" for r in result["tasks"] if r["status"] == "complete"
        ]
        evidence += [
            root / "scores" / domain / system / "all_results.json"
            for domain in FORMAL_DOMAINS
            for system in spec["systems"]
            if metrics[domain][system]["status"] == "complete"
        ]
        result.update(
            metrics=metrics,
            inventory=spec["inventory"],
            spec_sha256=hashlib.sha256(spec_bytes).hexdigest(),
            artifact_sha256={
                p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in evidence
            },
            optional_adapters={
                "mem0": (
                    "unrun: stock-document factory qualification required; "
                    "chat-user coercion forbidden"
                ),
                "cognee": "unrun: stable native document factory qualification required",
            },
            coverage_scope=(
                "full formal_reasoning_math + formal_reasoning_phys; "
                "other native domains require separate runtimes"
            ),
            authoritative=True,
            quality_complete=result["matrix_complete"],
        )
        # Stage and read back the final bound payload before the sole atomic
        # authority promotion. A failed final readback also revokes the root.
        pending = root / "campaign-result.pending.json"
        payload = native._json(result)
        pending.write_bytes(payload)
        if pending.read_bytes() != payload:
            raise ValueError("final campaign pending readback differs")
        for name, digest in result["artifact_sha256"].items():
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
                raise ValueError("final campaign evidence readback differs: " + name)
        pending.replace(root / "campaign-result.json")
        if (root / "campaign-result.json").read_bytes() != payload:
            raise ValueError("final campaign root readback differs")
        return result
    except BaseException as exc:
        # Remove even a successfully renamed complete root if final readback
        # fails. Preserve the actual matrix spend and leaves, never rescore or
        # promote leaves during recovery. Fresh-root admission remains required.
        (root / "campaign-result.json").unlink(missing_ok=True)
        (root / "campaign-result.pending.json").unlink(missing_ok=True)
        failure = {
            **result,
            "authoritative": False,
            "quality_complete": False,
            "error": safe_error(exc),
            "finalization_status": "failed",
        }
        (root / "campaign-failure.json").write_bytes(native._json(failure))
        try:
            (root / "campaign-result.json").write_bytes(native._json(failure))
        except OSError:
            # A persistently unwritable root still has separate failure evidence
            # and task-statuses.json with actual spend; no complete root remains.
            pass
        raise


def main(argv=None):
    import argparse

    from .rack import RackProfile

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="freeze every native Math+Phys input; zero generations")
    for key in ("data-root", "pins", "native-root", "profile", "output-root", "spec-out"):
        plan.add_argument("--" + key, type=Path, required=True)
    plan.add_argument("--embedding-url", required=True)
    plan.add_argument("--production-memory-qualification", type=Path)
    plan.add_argument(
        "--native-document-factories",
        type=Path,
        help="optional byte-pinned real Mem0/Cognee document bridge bindings",
    )
    plan.add_argument("--calls-per-task", type=int, required=True)
    plan.add_argument("--seconds-per-task", type=int, required=True)
    plan.add_argument("--campaign-seconds", type=int, required=True)
    plan.add_argument("--tool-iterations", type=int, required=True)
    run = commands.add_parser(
        "run", help="full native execution; independent byte-bound approval mandatory"
    )
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--approval", type=Path, required=True)
    verify = commands.add_parser(
        "verify-task", help="offline native scorer/source/step/readback replay"
    )
    verify.add_argument("--bundle", type=Path, required=True)
    qualify = commands.add_parser(
        "qualify-authored-memory",
        help="fixed authored only, real production binding, <=8 calls/300s",
    )
    for key in ("native-root", "profile", "out"):
        qualify.add_argument("--" + key, type=Path, required=True)
    qualify.add_argument("--embedding-url", required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        spec = build_spec(
            data_root=args.data_root,
            pins=json.loads(args.pins.read_bytes()),
            native_root=args.native_root,
            profile=RackProfile.model_validate(json.loads(args.profile.read_bytes())),
            output_root=args.output_root,
            embedding_url=args.embedding_url,
            call_budget=args.calls_per_task,
            seconds_budget=args.seconds_per_task,
            campaign_seconds=args.campaign_seconds,
            max_tool_iterations=args.tool_iterations,
        )
        if args.production_memory_qualification is not None:
            spec["production_memory_qualification"] = {
                "path": str(args.production_memory_qualification.resolve()),
                "sha256": hashlib.sha256(
                    args.production_memory_qualification.read_bytes()
                ).hexdigest(),
            }
        if args.native_document_factories is not None:
            bindings = json.loads(args.native_document_factories.read_bytes())
            spec["native_document_factories"] = bindings
            spec["systems"].extend(sorted(bindings))
        with args.spec_out.open("xb") as stream:
            stream.write(native._json(spec))
        print(
            json.dumps(
                {
                    "spec_sha256": hashlib.sha256(args.spec_out.read_bytes()).hexdigest(),
                    "inventory": spec["inventory"],
                    "quality_calls": 0,
                }
            )
        )
    elif args.command == "verify-task":
        print(json.dumps(verify_task(args.bundle)))
    elif args.command == "qualify-authored-memory":
        result = qualify_authored_memory(
            native_root=args.native_root,
            out=args.out,
            profile=RackProfile.model_validate(json.loads(args.profile.read_bytes())),
            embedding_url=args.embedding_url,
        )
        print(
            json.dumps(
                {"complete": result["complete"], "calls": result["calls"], "kind": result["kind"]}
            )
        )
    else:
        result = execute_formal(args.spec, args.approval)
        print(
            json.dumps(
                {
                    "quality_complete": result["quality_complete"],
                    "calls": result["calls"],
                    "expected_sessions": result["expected_sessions"],
                    "completed_sessions": result["completed_sessions"],
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
