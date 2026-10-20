"""Offline contracts for Arena memory ownership and explicit reservation lifetime."""

import ast
import hashlib
import importlib.util
from pathlib import Path

import pytest

from gluomem_bench.bench import arena_driver as formal

MEMORY_AST = {
    "ArenaProductionRouter": "1341076acf62838f473039e769ad86ce36b5db51d4f7969e7bd3747496d4dbcc",
    "BudgetedArenaEmbedder": "a2b2231b107d68877bb413b95bb000e2361a2d6aba61ad75a144846551abbccb",
    "RawExperienceMemory": "0fbb6bdf608bb28d1660ce783b2736cde02cb86d888a51c69f6dec65fd33b201",
    "GluoExperienceMemory": "3a196e5608fcd84b37690d5fee5548738652fd8fa87a79507950f7af75fe7f72",
}


def test_memory_has_neutral_owner_with_exact_compatible_definitions():
    assert importlib.util.find_spec("gluomem_bench.bench.experience_memory") is not None
    from gluomem_bench.bench import experience_memory as memory

    definitions = {
        n.name: n
        for n in ast.parse(Path(memory.__file__).read_bytes()).body
        if isinstance(n, ast.ClassDef)
    }
    for name, digest in MEMORY_AST.items():
        cls = getattr(memory, name)
        assert getattr(formal, name) is cls
        assert cls.__module__ == memory.__name__
        assert (
            hashlib.sha256(
                ast.dump(definitions[name], include_attributes=False).encode()
            ).hexdigest()
            == digest
        )
    inventory = {Path(r["path"]).resolve() for r in formal._code_inventory()}
    assert Path(memory.__file__).resolve() in inventory


@pytest.mark.parametrize("domain", ["shopping", "search"])
def test_domain_file_hashes_use_shared_byte_primitive(tmp_path, domain):
    import importlib

    from gluomem_bench.bench.artifacts import sha256_file

    driver = importlib.import_module("gluomem_bench.bench.arena_" + domain + "_driver")
    path = tmp_path / "binary"
    path.write_bytes(b"\x00\xff\r\n")
    assert driver.sha(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert driver.sha is sha256_file


def test_production_memory_assembly_preserves_roles_and_owned_paths(tmp_path, monkeypatch):
    from gluomem.llm import modules

    from gluomem_bench.bench import experience_memory as memory

    assert hasattr(memory, "production_memory")
    cfg, client, encoder, budget = object(), object(), object(), object()
    calls, router, extractors, result = [], object(), object(), object()

    def make_router(config, **kwargs):
        calls.append(("router", config, kwargs))
        return router

    def make_extractors(config, **kwargs):
        calls.append(("extractors", config, kwargs))
        return extractors

    def make_memory(**kwargs):
        calls.append(("memory", kwargs))
        return result

    monkeypatch.setattr(memory, "ArenaProductionRouter", make_router)
    monkeypatch.setattr(modules, "DspyExtractors", make_extractors)
    monkeypatch.setattr(memory, "GluoExperienceMemory", make_memory)
    assert (
        memory.production_memory(cfg, client=client, embedder=encoder, budget=budget, out=tmp_path)
        is result
    )
    assert calls == [
        (
            "router",
            cfg,
            {"client": client, "trace": tmp_path / "production-raw.jsonl", "budget": budget},
        ),
        ("extractors", cfg, {"router": router}),
        (
            "memory",
            {
                "cfg": cfg,
                "extractors": extractors,
                "embedder": encoder,
                "budget": budget,
                "db_path": tmp_path / "production.mind",
                "evidence_path": tmp_path / "memory-events.jsonl",
                "top_k": 10,
            },
        ),
    ]


@pytest.mark.parametrize("outcome", ["complete", "failure", "cancel", "refused"])
def test_explicit_reservation_retains_debits_until_after_handler(tmp_path, outcome):
    assert importlib.util.find_spec("gluomem_bench.bench.arena_runtime") is not None
    from gluomem_bench.bench.arena_runtime import release_context, reserve_context
    from gluomem_bench.bench.parallel_campaign import (
        CampaignBudget,
        CampaignStopped,
        Job,
        RequestLimiter,
    )

    outer = CampaignBudget(max_calls=3, max_seconds=10, clock=lambda: 0)
    job = Job("arena", "history", tmp_path, 3, 5)
    if outcome == "refused":
        outer.calls = 1
    row, context, events = {"calls": 0}, None, []
    try:
        outer.check()
        context = reserve_context(
            outer, job, RequestLimiter(max_in_flight=1, max_in_flight_tokens=32), "refusal text"
        )
        context.begin_call()
        if outcome == "failure":
            raise ValueError("task failure")
        if outcome == "cancel":
            context.failed = True
            context.check()
        row["status"] = "complete"
    except (ValueError, CampaignStopped) as exc:
        row["status"] = "failed" if context is not None else "unrun"
        events.append(("handler", outer.reserved_calls, row["calls"]))
        if outcome == "refused":
            assert str(exc) == "refusal text"
    finally:
        release_context(context, row)
        events.append(("write", outer.reserved_calls, row["calls"]))
    assert row == {
        "calls": 0 if outcome == "refused" else 1,
        "status": "unrun"
        if outcome == "refused"
        else ("complete" if outcome == "complete" else "failed"),
    }
    assert outer.calls == 1
    assert events[-1] == ("write", 0, row["calls"])
    if outcome in {"failure", "cancel"}:
        assert events[0] == ("handler", 2, 0)


@pytest.mark.parametrize("domain", ["travel", "shopping"])
def test_historical_bridge_resolves_actual_class_origin_and_rejects_mutation(
    tmp_path, monkeypatch, domain
):
    import json
    import zipfile

    from gluomem_bench.bench import arena_shopping_driver as shopping
    from gluomem_bench.bench import arena_travel_driver as travel
    from gluomem_bench.bench import experience_memory as memory
    from gluomem_bench.bench import rack

    # Authored synthetic old source, never imports or executes archived producers.
    memory_source = Path(memory.__file__).read_text(encoding="utf-8")
    tree = ast.parse(memory_source)
    old_memory = "\n\n".join(
        ast.unparse(n) for n in tree.body if getattr(n, "name", None) in MEMORY_AST
    ).encode()
    sources = {
        "arena_driver.py": old_memory,
        "arena_travel_driver.py": Path(travel.__file__).read_bytes(),
        "arena_shopping_driver.py": Path(shopping.__file__).read_bytes(),
        "rack.py": Path(rack.__file__).read_bytes(),
    }
    records = [
        {"path": name, "sha256": hashlib.sha256(raw).hexdigest()} for name, raw in sources.items()
    ]
    archive_path = tmp_path / "frozen-source.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("inventory.json", json.dumps(records))
        for index, (name, raw) in enumerate(sources.items()):
            archive.writestr(f"source/{index:06d}/{name}", raw)
            archive.writestr(f"producer/{index}/{name}", raw)
    before = archive_path.read_bytes()
    for name in MEMORY_AST:
        monkeypatch.setattr(formal, name, getattr(memory, name))

    def bridge():
        if domain == "travel":
            return travel._travel_semantic_bridge(
                [{"path": str(archive_path), "sha256": hashlib.sha256(before).hexdigest()}]
            )
        return shopping._shopping_bridge({"files": records, "output_root": str(tmp_path)})

    assert set(bridge()["unchanged_memory_classes"]) == set(MEMORY_AST)
    changed = tmp_path / "experience_memory.py"
    changed.write_text(memory_source.replace("return lm", "return None", 1), encoding="utf-8")
    monkeypatch.setattr(memory, "__file__", str(changed))
    with pytest.raises(ValueError, match="memory|production|algorithm"):
        bridge()
    assert archive_path.read_bytes() == before


@pytest.mark.parametrize(
    "filename,digest",
    [
        ("arena_driver.py", "e1e0952641d6988867c9834dbc2ce7da699442344eb79628326e80de0da33272"),
        (
            "arena_travel_driver.py",
            "dca02b00db2fca88d1306c9ff96db3adfbfa0ba05150c0fa4bb5a067ffe8117d",
        ),
        (
            "arena_shopping_driver.py",
            "e42e56b4b657de9cc66a3aa0a1ce5b4458f86d7f5e3b0563dc3b9a2cca74ba80",
        ),
    ],
)
def test_matrix_reservation_extraction_is_exact_ast_substitution(filename, digest):
    tree = ast.parse(Path(formal.__file__).with_name(filename).read_bytes())
    matrix = next(n for n in tree.body if getattr(n, "name", None) == "execute_matrix")
    assert any(isinstance(n, ast.Name) and n.id == "reserve_context" for n in ast.walk(matrix))

    class Inline(ast.NodeTransformer):
        def visit_Assign(self, node):
            call = node.value
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "reserve_context"
            ):
                budget, job, limiter, message = map(ast.unparse, call.args)
                return ast.parse(
                    f"if not {budget}.reserve({job}):\n    raise CampaignStopped({message})\n"
                    f"context = JobContext({job}, {budget}, {limiter})"
                ).body
            return self.generic_visit(node)

        def visit_Expr(self, node):
            call = node.value
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "release_context"
            ):
                return ast.parse(
                    'if context is not None:\n    row["calls"] = context.calls\n'
                    "    outer.release(context)"
                ).body
            return self.generic_visit(node)

    restored = Inline().visit(matrix)
    assert (
        hashlib.sha256(ast.dump(restored, include_attributes=False).encode()).hexdigest() == digest
    )
