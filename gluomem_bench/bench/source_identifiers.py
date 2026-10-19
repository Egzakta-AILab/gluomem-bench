"""Model-facing ordinal IDs; original source IDs remain an external audit mapping.

This bridge changes identifiers only, never roles, text, timestamps or order.
Do not index or pass its bindings into a model: source IDs can encode evidence
annotations. Applying this bridge is a versioned protocol change, not a repair
of predictions already generated with a different input representation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from gluomem.write.conversation import Conversation


@dataclass(frozen=True)
class SourceIdBinding:
    session_ordinal: int
    turn_ordinal: int
    original_id: str | None
    model_id: str


def ordinal_turn_ids(
    conversation: Conversation,
) -> tuple[Conversation, tuple[SourceIdBinding, ...]]:
    """Return a separate conversation and coordinate-qualified provenance mapping."""
    sessions = []
    bindings = []
    for session_ordinal, session in enumerate(conversation.sessions):
        turns = []
        for turn_ordinal, turn in enumerate(session.turns):
            identifier = f"s{session_ordinal}t{turn_ordinal}"
            bindings.append(SourceIdBinding(session_ordinal, turn_ordinal, turn.dia_id, identifier))
            turns.append(replace(turn, dia_id=identifier))
        sessions.append(replace(session, turns=turns))
    return replace(conversation, sessions=sessions), tuple(bindings)
