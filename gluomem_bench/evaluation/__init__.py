"""Bounded offline evaluator for prospective authored memory contracts.

Not an LLM judge, semantic-absence test, campaign admission or benchmark score.
Keep checks, supplements and source authorities outside every model prompt.
"""

from .grader import Refusal, evaluate
from .origin import bind_origin_checks
from .state_contracts import bind_report_sources, validate_report_projection
from .supplement import apply_supplement

__all__ = [
    "Refusal",
    "apply_supplement",
    "bind_origin_checks",
    "bind_report_sources",
    "evaluate",
    "validate_report_projection",
]
