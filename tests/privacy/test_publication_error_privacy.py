"""Authored offline probes of trusted publication diagnostics, not model execution.

Expression probes execute the actual source expression at each collector boundary;
they do not claim whole-driver execution. The SearchEmbedding probe executes its
whole published method with in-memory clients and intercepted receipt appends.
"""
import ast
import hashlib
import importlib.util
import json
import sys
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "gluomem_bench" / "bench"


def diagnostics(tree):
    parents = {child: parent for parent in ast.walk(tree)
               for child in ast.iter_child_nodes(parent)}
    variables = {"exc", "cleanup", "accounting_exc", "cleanup_exc", "other",
                 "cleanup_error", "persistence", "persistence_error"}

    def controlled(node):
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Name):
                if node.type.id in {"DatasetError", "ContextNotApplicable", "CampaignStopped"}:
                    return True
        return False

    def atom(node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"safe_error", "safe_kind", "safe_notes"}:
                return True
            if node.func.id == "str" and len(node.args) == 1:
                return isinstance(node.args[0], ast.Name) and node.args[0].id in variables
            if node.func.id == "list" and node.args:
                return "__notes__" in ast.unparse(node)
        if isinstance(node, ast.Attribute) and node.attr == "__name__":
            value = node.value
            return (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                    and value.func.id == "type" and len(value.args) == 1
                    and isinstance(value.args[0], ast.Name) and value.args[0].id in variables)
        if isinstance(node, ast.FormattedValue):
            return isinstance(node.value, ast.Name) and node.value.id in variables
        return False

    roots = set()
    for node in ast.walk(tree):
        if not atom(node) or controlled(node):
            continue
        while node in parents and isinstance(parents[node], (ast.BinOp, ast.JoinedStr,
                                                              ast.FormattedValue)):
            node = parents[node]
        roots.add(node)
    return sorted(roots, key=lambda node: (node.lineno, node.col_offset))


def helpers():
    path = SOURCE / "error_privacy.py"
    if not path.exists():
        return {}
    spec = importlib.util.spec_from_file_location("owned_error_privacy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {name: getattr(module, name) for name in ("safe_error", "safe_kind", "safe_notes")}


class PrivateValue:
    str_calls = 0
    repr_calls = 0

    def __str__(self):
        self.str_calls += 1
        return "authored-opaque-private-value"

    def __repr__(self):
        self.repr_calls += 1
        return "authored-opaque-private-value"


class OpaqueFailure(Exception):
    str_calls = 0
    repr_calls = 0

    def __str__(self):
        self.str_calls += 1
        return "authored-opaque-private-value"

    def __repr__(self):
        self.repr_calls += 1
        return "authored-opaque-private-value"


OpaqueFailure.__name__ = "authored-opaque-private-value"


def authored_error():
    value = PrivateValue()
    error = OpaqueFailure(value)
    error.add_note("authored-opaque-private-value")
    error.__cause__ = ValueError("authored-opaque-private-value")
    return error, value


def assert_private_absent(case, payload, error, value):
    # Never let failing assertion introspection expose even authored markers.
    if "authored-opaque-private-value" in json.dumps(payload):
        case.fail("private text retained in diagnostic")
    case.assertEqual(error.str_calls, 0, "exception string conversion")
    case.assertEqual(error.repr_calls, 0, "exception repr conversion")
    case.assertEqual(value.str_calls, 0, "private argument string conversion")
    case.assertEqual(value.repr_calls, 0, "private argument repr conversion")


class CollectorPrivacyTests(unittest.TestCase):
    pass


def collector_probe(filename, index):
    def test(self):
        error, value = authored_error()
        tree = ast.parse((SOURCE / (filename + ".py")).read_text(encoding="utf-8"))
        expression = diagnostics(tree)[index]
        namespace = {name: error for name in (
            "exc", "cleanup", "accounting_exc", "cleanup_exc", "other",
            "cleanup_error", "persistence", "persistence_error"
        )}
        namespace.update(helpers())
        namespace.update(budget=SimpleNamespace(calls=7), spend={"calls": 7},
                         name="result.json")
        result = eval(compile(ast.Expression(expression), "trusted-collector", "eval"), namespace)
        assert_private_absent(self, result, error, value)
        self.assertIsNotNone(error.__cause__)
    return test


BOUNDARIES = [
    ('arena_search_driver', 0, 274),
    ('arena_search_driver', 1, 400),
    ('arena_search_driver', 2, 809),
    ('arena_search_driver', 3, 842),
    ('arena_search_driver', 4, 1010),
    ('arena_search_driver', 5, 1025),
    ('arena_search_driver', 6, 1446),
    ('arena_search_driver', 7, 1447),
    ('arena_search_driver', 8, 1658),
    ('arena_search_driver', 9, 1740),
    ('arena_search_driver', 10, 1748),
    ('arena_campaign', 0, 270),
    ('arena_campaign', 1, 322),
    ('arena_campaign', 2, 479),
    ('arena_campaign', 3, 803),
    ('arena_driver', 0, 558),
    ('arena_driver', 1, 861),
    ('arena_driver', 2, 1029),
    ('arena_driver', 3, 1364),
    ('arena_shopping_driver', 0, 253),
    ('arena_shopping_driver', 1, 698),
    ('arena_shopping_driver', 2, 752),
    ('arena_shopping_driver', 3, 761),
    ('arena_shopping_driver', 4, 763),
    ('arena_shopping_driver', 5, 1422),
    ('arena_shopping_driver', 6, 1589),
    ('arena_shopping_driver', 7, 2472),
    ('arena_shopping_driver', 8, 2475),
    ('arena_shopping_driver', 9, 2531),
    ('arena_shopping_driver', 10, 2553),
    ('arena_shopping_driver', 11, 2554),
    ('arena_shopping_driver', 12, 2561),
    ('arena_travel_driver', 0, 454),
    ('arena_travel_driver', 1, 498),
    ('arena_travel_driver', 2, 1007),
    ('arena_travel_driver', 3, 1494),
    ('arena_travel_driver', 4, 1508),
    ('arena_travel_driver', 5, 1531),
    ('arena_travel_driver', 6, 1596),
    ('arena_travel_driver', 7, 2740),
    ('arena_travel_driver', 8, 2760),
    ('arena_travel_driver', 9, 2772),
    ('arena_travel_driver', 10, 2805),
    ('arena_travel_driver', 11, 2818),
    ('beam_driver', 0, 375),
    ('beam_driver', 1, 614),
    ('beam_driver', 2, 633),
    ('beam_driver', 3, 676),
    ('beam_driver', 4, 1055),
    ('beam_driver', 5, 1055),
    ('beam_driver', 6, 1457),
    ('mab_campaign', 0, 113),
    ('mab_campaign', 1, 116),
    ('mab_campaign', 2, 118),
    ('mab_campaign', 3, 765),
    ('mab_campaign', 4, 973),
    ('mab_campaign', 5, 978),
    ('mab_campaign', 6, 980),
    ('mab_campaign', 7, 983),
    ('mab_campaign', 8, 1515),
    ('mab_campaign', 9, 1518),
    ('mab_campaign', 10, 1520),
    ('mab_campaign', 11, 1687),
    ('mab_campaign', 12, 1740),
    ('mab_campaign', 13, 1743),
    ('mab_campaign', 14, 1745),
    ('v2_driver', 0, 248),
    ('v2_driver', 1, 439),
    ('v2_driver', 2, 677),
    ('v2_driver', 3, 857),
    ('v2_driver', 4, 869),
    ('v2_driver', 5, 874),
    ('v2_driver', 6, 877),
    ('v2_driver', 7, 1340),
    ('v2_driver', 8, 1367),
]

for _filename, _index, _original_line in BOUNDARIES:
    setattr(CollectorPrivacyTests, f"test_{_filename}_{_index:02d}_line_{_original_line}",
            collector_probe(_filename, _index))


class SearchEmbeddingPrivacyTests(unittest.TestCase):
    def test_failed_http_preserves_receipt_latch_cause_and_finally(self):
        for shared_failure in (False, True):
            with self.subTest(shared_failure=shared_failure):
                self._run_failure(shared_failure)

    def test_actual_http_error_receipt_and_forward_are_categorical(self):
        self._run_failure(False, http_error=True)

    def _run_failure(self, shared_failure, http_error=False):
        error, value = authored_error()
        if http_error:
            error = HTTPError("https://fixture.invalid/opaque", 400, value, {}, None)
            error.str_calls = error.repr_calls = 0
            error.__cause__ = ValueError("authored cause")
            # HTTPError construction renders its reason on this interpreter;
            # counters cover only the production method boundary below.
            value.str_calls = value.repr_calls = 0
        class SharedRackError(OpaqueFailure):
            pass
        class NativeAbort(BaseException):
            pass
        if shared_failure:
            error = SharedRackError(value)
            error.__cause__ = ValueError("authored cause")
        receipts = []
        directories = []
        checks = []
        class Client:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                checks.append("client_closed")
            def post(self, *args, **kwargs):
                raise error
        httpx = ModuleType("httpx")
        httpx.Client = Client
        numpy = ModuleType("numpy")
        budget = SimpleNamespace(failed=False, budget=SimpleNamespace(failed=False), calls=7,
                                 check=lambda: checks.append("checked"), remaining=lambda: 20)
        instance = SimpleNamespace(
            budget=budget, dim=768, base_url="https://fixture.invalid/v1",
            recipe=SimpleNamespace(prefix=lambda role: "authored ", model="authored"),
            receipt=SimpleNamespace(parent=SimpleNamespace(
                mkdir=lambda **kwargs: directories.append(True))),
        )
        tree = ast.parse((SOURCE / "arena_search_driver.py").read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "SearchEmbedding")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_encode")
        namespace = dict(time=time, hashlib=hashlib, json=json, SharedRackError=SharedRackError,
                         shared=SimpleNamespace(NativeAbort=NativeAbort,
                             _append=lambda path, row: receipts.append(dict(row))))
        namespace.update(helpers())
        exec(compile(ast.Module(body=[method], type_ignores=[]), "trusted-search-method", "exec"),
             namespace)
        original_cause = error.__cause__
        with patch.dict(sys.modules, httpx=httpx, numpy=numpy):
            try:
                namespace["_encode"](instance, ["authored input"], "query")
            except NativeAbort as forwarded:
                self.assertIs(forwarded.__cause__, error)
                assert_private_absent(self, forwarded.args, error, value)
            else:
                self.fail("failure must propagate")
        self.assertIs(error.__cause__, original_cause)
        self.assertEqual(len(receipts), 1)
        assert_private_absent(self, receipts, error, value)
        self.assertFalse(receipts[0]["complete"])
        self.assertTrue(budget.failed)
        self.assertEqual(budget.budget.failed, shared_failure)
        self.assertEqual(budget.calls, 7)
        self.assertEqual(directories, [True])
        self.assertIn("client_closed", checks)
        self.assertIn("elapsed_seconds", receipts[0])

    def test_http_error_payload_never_reaches_forwarded_string(self):
        # All collectors share this helper; URL/reason/headers/args stay in memory.
        safe = helpers().get("safe_error")
        if safe is None:
            self.fail("categorical helper missing")
        error = HTTPError("https://fixture.invalid/opaque", 400,
                          "authored-opaque-private-value", {}, None)
        self.assertNotIn("authored-opaque-private-value", safe(error))


class HandlerPersistenceTests(unittest.TestCase):
    def _run_handler(self, filename, diagnostic_index, *, cleanup_failure=False):
        import tempfile

        error, value = authored_error()
        cleanup, cleanup_value = authored_error()
        tree = ast.parse((SOURCE / (filename + ".py")).read_text(encoding="utf-8"))
        expression = diagnostics(tree)[diagnostic_index]
        parents = {child: parent for parent in ast.walk(tree)
                   for child in ast.iter_child_nodes(parent)}
        handler = expression
        while not isinstance(handler, ast.ExceptHandler):
            handler = parents[handler]
        closed, latched = [], []
        def close():
            closed.append(True)
            if cleanup_failure:
                raise cleanup
        def write(path, payload):
            path.write_text(json.dumps(payload), encoding="utf-8")
        class ResourceNotAdmitted(Exception):
            pass
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            for name in ("result.json", "result.pending.json", "root-status.json",
                         "root-status.pending.json"):
                (out / name).write_text("non-authoritative authored candidate", encoding="utf-8")
            budget = SimpleNamespace(calls=7, snapshot=lambda: {"calls": 7})
            client = SimpleNamespace(budget=budget, close=close, _latch=latched.append)
            namespace = dict(
                injected=error, out=out, root=out, target=out, rows=[{}], row={},
                raw={"questions": [1, 2, 3]}, task=SimpleNamespace(sessions=[1, 2, 3]),
                transport=client, client=client, budget=budget, mem=SimpleNamespace(close=close),
                closed=True, attempted_index=1, attempted=1, initially_absent=True,
                result={}, stage="authored_failure", ResourceNotAdmitted=ResourceNotAdmitted,
                _latch_group_failure=lambda client, error: latched.append(error),
                initial_calls=2, json=json, _json=lambda row: json.dumps(row).encode(),
                native=SimpleNamespace(_json=lambda row: json.dumps(row).encode()),
                common=SimpleNamespace(_json=lambda row: json.dumps(row).encode()),
                write=write, _write=write,
            )
            namespace["row"] = {"questions": [1, 2, 3]}
            namespace.update(helpers())
            # Execute the actual trusted catch body, including writes/revocation.
            caught = ast.ExceptHandler(type=ast.Name(id="BaseException", ctx=ast.Load()),
                                      name="exc", body=handler.body)
            stmt = ast.Try(body=[ast.Raise(exc=ast.Name(id="injected", ctx=ast.Load()))],
                           handlers=[caught], orelse=[], finalbody=[])
            module = ast.fix_missing_locations(ast.Module(body=[stmt], type_ignores=[]))
            try:
                exec(compile(module, "trusted-publication-handler", "exec"), namespace)
            except BaseException as raised:
                self.assertTrue(raised is error or raised.__cause__ is error)
            if filename in {"arena_travel_driver", "v2_driver"}:
                payload = (namespace["row"] if filename == "arena_travel_driver"
                           else namespace["result"])
                write(out / "status.json", payload)
            else:
                payload = json.loads((out / "failure.json").read_text(encoding="utf-8"))
            persisted = [path.read_text(encoding="utf-8") for path in out.glob("*.json")]
            assert_private_absent(self, persisted, error, value)
            assert_private_absent(self, persisted, cleanup, cleanup_value)
            if filename not in {"beam_driver", "v2_driver", "arena_travel_driver"}:
                self.assertEqual(payload.get("calls", payload.get("generation_calls")), 7)
            if filename in {"arena_driver", "arena_shopping_driver"}:
                self.assertEqual([row["status"] for row in payload["sessions"]],
                                 ["complete", "failed", "unrun"])
                self.assertFalse((out / "result.json").exists())
            if filename == "arena_shopping_driver":
                self.assertEqual(len(closed), 1 if cleanup_failure else 2)
                self.assertEqual(len(payload["cleanup_diagnostics"]), 1)
                self.assertEqual(payload["cleanup_error"] is not None, cleanup_failure)
            if filename == "mab_campaign":
                self.assertEqual(error.generation_calls, 7)
                self.assertEqual(error.accounting, {"calls": 7})
                self.assertFalse((out / "root-status.json").exists())
            self.assertEqual(budget.calls, 7)

    def test_arena_campaign_persisted_failure(self):
        self._run_handler("arena_campaign", 3)

    def test_arena_driver_persisted_failure(self):
        self._run_handler("arena_driver", 0)

    def test_shopping_origin_and_cleanup_persisted_failure(self):
        self._run_handler("arena_shopping_driver", 3, cleanup_failure=True)

    def test_travel_failed_status_persistence(self):
        self._run_handler("arena_travel_driver", 3)

    def test_beam_persisted_failure(self):
        self._run_handler("beam_driver", 1)

    def test_mab_persisted_failure(self):
        self._run_handler("mab_campaign", 9)

    def test_v2_failed_status_persistence(self):
        self._run_handler("v2_driver", 3)

    def test_v2_cleanup_status_persistence(self):
        self._run_handler("v2_driver", 4)

    def test_v2_accounting_status_persistence(self):
        self._run_handler("v2_driver", 6)


class CategoricalHelperTests(unittest.TestCase):
    def test_exact_lookup_error_label_without_private_args(self):
        safe = helpers()["safe_kind"]
        self.assertEqual(safe(LookupError(PrivateValue())), "LookupError")

    def test_custom_subclass_cannot_supply_a_builtin_label(self):
        safe = helpers()["safe_kind"]
        class Custom(ValueError):
            pass
        self.assertEqual(safe(Custom(PrivateValue())), "Exception")

    def test_missing_or_nonlist_notes_are_not_serialized(self):
        safe = helpers()["safe_notes"]
        error = RuntimeError(PrivateValue())
        self.assertEqual(safe(error), [])
        error.__notes__ = PrivateValue()
        self.assertEqual(safe(error), [])


class NarrowCollectorPrivacyTests(unittest.TestCase):
    def _assert_expression(self, filename, selector):
        tree = ast.parse((SOURCE / (filename + ".py")).read_text(encoding="utf-8"))
        expression = selector(tree)
        error, value = authored_error()
        namespace = {"exc": error, **helpers()}
        payload = eval(compile(ast.Expression(expression), "trusted-narrow-collector", "eval"),
                       namespace)
        assert_private_absent(self, payload, error, value)

    def _campaign_expression(self, tree):
        handler = next(node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)
                       and isinstance(node.type, ast.Name) and node.type.id == "CampaignStopped")
        call = handler.body[0].value
        return next(keyword.value for keyword in call.keywords if keyword.arg == "error")

    def test_arena_campaign_stopped_from_callback_is_categorical(self):
        self._assert_expression("arena_driver", self._campaign_expression)

    def test_travel_campaign_stopped_from_callback_is_categorical(self):
        self._assert_expression("arena_travel_driver", self._campaign_expression)

    def test_beam_selection_blocker_never_persists_dataset_identifier(self):
        def select(tree):
            return next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                        and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id == "blocker" and isinstance(node.value, ast.Call))
        self._assert_expression("beam_driver", select)


if __name__ == "__main__":
    unittest.main()
