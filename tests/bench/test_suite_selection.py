"""Selection contracts use generated rows, never claim benchmark quality."""

import pytest

from gluomem_bench.bench.types import Sample


def test_stratified_selection_is_order_independent_and_gold_blind():
    from dataclasses import replace

    from gluomem_bench.bench.suite import select_samples

    pool = [Sample(str(i), str(i), "question", "gold", i % 2 + 1, "task") for i in range(20)]
    selected = select_samples(pool, categories=(1, 2), per_category=2, seed=42)
    changed = [replace(s, gold="changed", evidence=("changed",)) for s in reversed(pool)]
    assert [s.id for s in selected] == [
        s.id for s in select_samples(changed, categories=(1, 2), per_category=2, seed=42)
    ]
    assert len(selected) == 4
    assert {c: sum(s.category == c for s in selected) for c in (1, 2)} == {1: 2, 2: 2}


@pytest.mark.parametrize("per_category", [0, -1, True])
def test_selection_refuses_invalid_counts(per_category):
    from gluomem_bench.bench.suite import select_samples

    with pytest.raises(ValueError):
        select_samples([], categories=(1,), per_category=per_category, seed=42)


def test_selection_refuses_duplicate_ids_and_insufficient_strata():
    from gluomem_bench.bench.suite import select_samples

    s = Sample("same", "history", "q", "a", 1, "task")
    with pytest.raises(ValueError, match="duplicate"):
        select_samples([s, s], categories=(1,), per_category=1, seed=42)
    with pytest.raises(ValueError, match="stratum"):
        select_samples([s], categories=(1, 2), per_category=1, seed=42)
