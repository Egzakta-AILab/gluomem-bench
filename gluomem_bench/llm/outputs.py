"""Benchmark QA answer and judge output models, plus re-exports of Core outputs."""

from __future__ import annotations

from gluomem.llm.outputs import (
    CognifyResult,
    ContradictionVerdict,
    ExtractedEvent,
    ExtractedFact,
    ExtractionError,
    Extractors,
    KGEntity,
    KGRelation,
)
from pydantic import BaseModel, Field


class Verdict(BaseModel):
    """Whether the prediction answers the question in agreement with gold."""

    correct: bool
    score: float = Field(ge=0, le=1)
    rationale: str


# Answerer return value for the bench harness; never referenced by a dspy
# OutputField (AnswerQA outputs a plain str), so it is not prompt-visible.
class AnswerResult(BaseModel):
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


__all__ = [
    "AnswerResult",
    "CognifyResult",
    "ContradictionVerdict",
    "ExtractedEvent",
    "ExtractedFact",
    "ExtractionError",
    "Extractors",
    "KGEntity",
    "KGRelation",
    "Verdict",
]
