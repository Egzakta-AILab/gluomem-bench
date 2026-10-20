"""Native metric semantics on authored examples, not benchmark quality evidence."""

import pytest


@pytest.mark.parametrize("prediction,gold,expected", [
    ("Answer: The New-York.", ["newyork"], True),
    ("old value; new value", ["new value"], True),
    ("wrong", [["correct"], ["wrong"]], True),
    ("concatenate", ["cat"], True),
    ("unknown", ["Paris"], False),
    ("first line\nAnswer: Paris", ["Paris"], True),
])
def test_mab_native_substring_semantics(prediction, gold, expected):
    from gluomem_bench.bench.upstream_scoring import mab_substring_score

    assert mab_substring_score(prediction, gold) is expected


def test_unverified_upstream_prompt_source_is_not_executed(tmp_path):
    from gluomem_bench.bench.upstream_scoring import load_longmemeval_prompt

    path = tmp_path / "scorer.py"
    path.write_text("raise RuntimeError('must never execute')", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_longmemeval_prompt(path)
