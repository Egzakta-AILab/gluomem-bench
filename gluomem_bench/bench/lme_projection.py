"""Fresh matched LongMemEval input projection, separate from historical adapters.

Use the returned conversation for ALL compared backends. Source text (including
empty turns and boundary whitespace), roles and timestamps are retained exactly;
only session order and model-facing identifiers change. Never feed answer/session
annotations or the raw row to a memory backend. This does not change native
memory algorithms, extraction policy, or the native Mem0 observation clock.
"""

from __future__ import annotations

from gluomem_bench.bench.adapters.longmemeval import LongMemEvalAdapter, _parse_session_date
from gluomem.write.conversation import Conversation, Session, Turn

PROJECTION_POLICY = "lme-exact-text-empty-retained-chronological-opaque@v1"


class ExactLongMemEvalAdapter(LongMemEvalAdapter):
    """Explicit fresh-run adapter; native sample/scorer/clock loading is inherited."""

    projection_policy = PROJECTION_POLICY

    def conversations(self) -> list[Conversation]:
        return [project_history(row) for row in self._raw_questions()]


def project_history(raw: dict) -> Conversation:
    """Project one released row without model calls or content normalization.

    Original array coordinates produce opaque turn IDs before chronological
    sorting; equal timestamps retain original session order. Gold and evidence
    annotations are never accessed. The caller retains the raw row out of band.
    """
    if not (len(raw["haystack_dates"]) == len(raw["haystack_session_ids"]) == len(raw["haystack_sessions"])):
        raise ValueError("parallel source arrays differ")
    for turns in raw["haystack_sessions"]:
        for turn in turns:
            if type(turn.get("content")) is not str or turn.get("role") not in ("user", "assistant"):
                raise ValueError("source role/content must be explicit text")
    sessions = []
    for si, (date, turns) in enumerate(zip(raw["haystack_dates"], raw["haystack_sessions"], strict=True)):
        sessions.append(Session(
            index=si + 1,
            date_time=_parse_session_date(date, where="matched LME history"),
            turns=[Turn(speaker=t["role"], text=t["content"], dia_id=f"s{si}t{ti}")
                   for ti, t in enumerate(turns)],
        ))
    return Conversation(
        conv_id=raw["question_id"], speakers=("user", "assistant"),
        sessions=sorted(sessions, key=lambda s: (s.date_time, s.index)),
    )
