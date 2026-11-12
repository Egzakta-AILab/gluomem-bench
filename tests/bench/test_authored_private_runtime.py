"""Portable authored runtime positives do not bypass real admission guards."""

import copy
import sys
from pathlib import Path
from types import ModuleType

import pytest
from test_comprehensive_driver_setup import approved_setup as approved_setup


def test_private_runtime_keeps_inventory_and_import_refusals(approved_setup, monkeypatch):
    driver, spec, _, _ = approved_setup
    runtime = Path(spec["runtime_root"])
    assert runtime.name == "runtime"
    assert Path(driver.__file__).is_relative_to(runtime)
    assert driver.verify_execution_spec
    pins = {item["path"] for item in spec["files"]}
    for name, module in list(sys.modules.items()):
        if name.split(".")[0] in {"gluomem", "gluomem_bench"}:
            assert module.__file__ is not None
            assert Path(module.__file__).is_relative_to(runtime)
            assert str(Path(module.__file__).resolve()) in pins
    incomplete = copy.deepcopy(spec)
    incomplete["files"].pop()
    with pytest.raises(ValueError, match="inventory/source binding differs"):
        driver.validate_chained_continuation_spec(incomplete)
    rogue = ModuleType("gluomem_bench.authored_outside_runtime")
    rogue.__file__ = __file__
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, rogue.__name__, rogue)
        with pytest.raises(ValueError, match="actual import escapes private runtime"):
            driver.validate_chained_continuation_spec(spec)
    assert not Path(spec["output_root"]).exists()


def test_ambient_checkout_remains_ineligible(tmp_path):
    from gluomem_bench.bench import comprehensive_driver as driver

    with pytest.raises(ValueError, match="requires a private runtime"):
        driver._chain_launch_metadata({}, tmp_path / "out")
    assert not (tmp_path / "out").exists()
