"""Gold-blind deterministic sampling for paired, source-pinned suite slices.

Selection is independent of input order and platform PRNG versions. This is
an adapted sampling protocol, not an upstream benchmark split. Keep manifests
and all dataset-derived material in an operator-local external directory.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from gluomem_bench.bench.types import Sample

SELECTION_PROTOCOL = "sha256-stratified-id@v1"


def select_samples(
    samples: Sequence[Sample], *, categories: Sequence[int], per_category: int, seed: int
) -> list[Sample]:
    """Rank IDs per stratum; never use gold, evidence, length or model output."""
    if type(per_category) is not int or per_category < 1 or type(seed) is not int:
        raise ValueError("positive integer count and integer seed required")
    if not categories or len(set(categories)) != len(categories):
        raise ValueError("nonempty distinct categories required")
    ids = [sample.id for sample in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sample IDs")

    def rank(sample: Sample) -> tuple[str, str]:
        material = json.dumps(
            [SELECTION_PROTOCOL, seed, sample.category, sample.id], separators=(",", ":")
        )
        return hashlib.sha256(material.encode()).hexdigest(), sample.id

    selected = []
    for category in categories:
        bucket = sorted((s for s in samples if s.category == category), key=rank)
        if len(bucket) < per_category:
            raise ValueError(f"stratum {category} has {len(bucket)} rows; need {per_category}")
        selected.extend(bucket[:per_category])
    return selected
