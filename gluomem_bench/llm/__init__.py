"""gluomem_bench.llm — QA answerer, judge, extraction evaluator, and prompt optimizer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gluomem.llm import (
    CachedExtractors,
    CognifyResult,
    ContradictionVerdict,
    ExtractedEvent,
    ExtractedFact,
    ExtractionCacheMiss,
    ExtractionError,
    Extractors,
    FakeExtractors,
    KGEntity,
    KGRelation,
    NullExtractors,
    make_extractors,
)

from gluomem_bench.llm.outputs import AnswerResult, Verdict

if TYPE_CHECKING:
    from gluomem_bench.llm.answer import Answerer, build_answer_module
    from gluomem_bench.llm.judge import QAJudge, build_judge_module

__all__ = [
    "AnswerResult",
    "Answerer",
    "CachedExtractors",
    "CognifyResult",
    "ContradictionVerdict",
    "ExtractedEvent",
    "ExtractedFact",
    "ExtractionCacheMiss",
    "ExtractionError",
    "Extractors",
    "FakeExtractors",
    "KGEntity",
    "KGRelation",
    "NullExtractors",
    "QAJudge",
    "Verdict",
    "build_answer_module",
    "build_judge_module",
    "make_extractors",
]


def __getattr__(name: str) -> Any:
    if name in ("Answerer", "build_answer_module"):
        from gluomem_bench.extras import require

        require("llm", f"gluomem_bench.llm.{name}")
        from gluomem_bench.llm import answer

        return getattr(answer, name)
    if name in ("QAJudge", "build_judge_module"):
        from gluomem_bench.extras import require

        require("llm", f"gluomem_bench.llm.{name}")
        from gluomem_bench.llm import judge

        return getattr(judge, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
