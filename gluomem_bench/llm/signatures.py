"""gluomem_bench.llm.signatures — QA answerer and judge DSPy signatures.

Signature class docstrings and field ``desc`` strings are LIVE PROMPT text
(hashed into run manifests) — any edit changes model behaviour.
"""

from __future__ import annotations

import dspy
from gluomem.llm.signatures import (
    TASK_SIGNATURES,
    ClaimConsolidation,
    CognifyExtraction,
    ConsolidateClaimSignature,
    ContradictionCheck,
    EventClaimExtraction,
    EventExtraction,
    ExtractEventClaimsSignature,
    FactExtraction,
    make_cognify_signature,
)

from gluomem_bench.llm.outputs import Verdict


class JudgeQA(dspy.Signature):
    """Judge whether a predicted answer matches the gold answer for a question.
    Grade leniently: mark correct when the prediction conveys the gold
    information, even with different wording, extra detail, or a different
    date format."""

    question: str = dspy.InputField()
    gold: str = dspy.InputField()
    prediction: str = dspy.InputField()
    verdict: Verdict = dspy.OutputField()


# Rubric-carrying judge for benchmarks whose official grading differs per
# question type (LongMemEval: temporal off-by-one forgiveness, knowledge-update
# accepts old+new, abstention correct iff declined). The rubric is an INPUT so
# one signature serves every type; the caller supplies the per-type text.
class JudgeQATyped(dspy.Signature):
    """Judge whether a predicted answer matches the gold answer for a question.
    Apply the grading rubric exactly; it defines what counts as correct for
    this question type."""

    question: str = dspy.InputField()
    gold: str = dspy.InputField()
    prediction: str = dspy.InputField()
    rubric: str = dspy.InputField(desc="Grading rubric for this question type; follow it exactly.")
    verdict: Verdict = dspy.OutputField()


# Benchmark answerer (the experiment subject). NOT in TASK_SIGNATURES:
# `optimize --task all` iterates that map and would demand an answer trainset;
# the answer prompt is deliberately left unoptimized.
class AnswerQA(dspy.Signature):
    """Answer the question using only the provided memory context. Give the
    shortest answer that fully resolves the question; say "I don't know" when
    the context does not contain the answer."""

    question: str = dspy.InputField()
    context: str = dspy.InputField(
        desc="Retrieved memories, one per line, each prefixed with its date."
    )
    answer: str = dspy.OutputField()


# Preference-aware variant (benchmark.answer.style='preference_aware'). The
# LongMemEval ss-pref failure mode is synthesis, not retrieval: the base
# answerer surfaces the user's own prior experiment but phrases it as a fresh
# suggestion, which per-type preference rubrics grade as ignoring the user's
# history. Only the docstring differs from AnswerQA.
class AnswerQAPreferenceAware(dspy.Signature):
    """Answer the question using only the provided memory context. Give the
    shortest answer that fully resolves the question; say "I don't know" when
    the context does not contain the answer. When the context records the
    user's own preferences, past choices, or prior experiments relevant to
    the question, explicitly acknowledge them as the user's own history and
    build the answer on them — never re-present something the user already
    tried or stated as if it were a new suggestion."""

    question: str = dspy.InputField()
    context: str = dspy.InputField(
        desc="Retrieved memories, one per line, each prefixed with its date."
    )
    answer: str = dspy.OutputField()


__all__ = [
    "TASK_SIGNATURES",
    "AnswerQA",
    "AnswerQAPreferenceAware",
    "ClaimConsolidation",
    "CognifyExtraction",
    "ConsolidateClaimSignature",
    "ContradictionCheck",
    "EventClaimExtraction",
    "EventExtraction",
    "ExtractEventClaimsSignature",
    "FactExtraction",
    "JudgeQA",
    "JudgeQATyped",
    "make_cognify_signature",
]
