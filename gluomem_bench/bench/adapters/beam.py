"""BEAM official 128K tier (upstream storage split is literally ``100K``).

Input is a deterministic JSON export of HF rows, not a quality corpus authored
here. Only ``chat`` enters ingestion; probes, plans and profiles never do.
"""

from __future__ import annotations

import ast
import json
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING

from gluomem_bench.bench.adapters.locomo import DatasetError
from gluomem_bench.bench.adapters.longmemeval import LongMemEvalAdapter
from gluomem_bench.bench.types import Sample
from gluomem_bench.ingest import Conversation, Session, Turn

if TYPE_CHECKING:
    from gluomem_bench.config import AppConfig

HF_REVISION = "3205395e897e7318c7b094ef4e6047b9b82dbb03"
CODE_REVISION = "b2da22eac88bb0874c64665f13457eb99835774a"
PARQUET_SHA256 = "c0519be25907005ba873c927c50877471d550873039d96c041554d0075a78ace"
PARQUET_URL = (
    "https://huggingface.co/datasets/Mohammadta/BEAM/resolve/"
    f"{HF_REVISION}/data/100K-00000-of-00001.parquet"
)
CATEGORIES = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)


class BeamAdapter(LongMemEvalAdapter):
    """Reuse SHA-verified JSON transport, not LongMemEval QA semantics."""

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__(cfg, "beam_128k")

    def verify(self) -> None:
        super().verify()
        try:
            rows = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(rows, list) or not rows:
                raise ValueError("expected nonempty HF row list")
            ids = [str(row["conversation_id"]) for row in rows]
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate conversation_id")
            self._raw = rows
            for row in rows:
                seen = set()
                if not row["chat"]:
                    raise ValueError("empty chat")
                for batch in row["chat"]:
                    if not batch:
                        raise ValueError("empty chat batch")
                    for message in batch:
                        if message["role"] not in ("user", "assistant"):
                            raise ValueError("unsupported role")
                        if message["id"] in seen or not isinstance(message["content"], str):
                            raise ValueError("duplicate message id or non-text content")
                        seen.add(message["id"])
            probes = list(self._probes())
            if not probes:
                raise ValueError("no probing questions")
            for _, _, cat, question in probes:
                if cat not in CATEGORIES or not question["question"].strip():
                    raise ValueError("unknown category or empty question")
                rubric = question["rubric"]
                if (
                    not isinstance(rubric, list)
                    or not rubric
                    or not all(isinstance(item, str) and item.strip() for item in rubric)
                ):
                    raise ValueError("nonempty string rubric nuggets required")
            self.conversations()
        except (KeyError, TypeError, ValueError) as exc:
            self._raw = None
            raise DatasetError(f"invalid BEAM data: {exc}") from exc

    def _probes(self):
        for raw in self._raw_questions():
            conv_id = f"beam-{raw['conversation_id']}"
            probes = raw["probing_questions"]
            if isinstance(probes, str):
                try:
                    probes = json.loads(probes)
                except json.JSONDecodeError:
                    probes = ast.literal_eval(probes)
            idx = 0
            for category, questions in probes.items():
                for question in questions:
                    yield conv_id, idx, category, question
                    idx += 1

    def load(self) -> list[Sample]:
        return [
            Sample(
                id=f"{cid}:qa:{idx}",
                conv_id=cid,
                question=q["question"],
                gold=str(
                    next(
                        (q[k] for k in ("ideal_response", "ideal_answer", "answer") if q.get(k)), ""
                    )
                ),
                category=CATEGORIES.index(cat) + 1,
                category_name=cat,
            )
            for cid, idx, cat, q in self._probes()
        ]

    def rubrics(self) -> dict[str, tuple[str, ...]]:
        return {f"{cid}:qa:{idx}": tuple(q["rubric"]) for cid, idx, _, q in self._probes()}

    def conversations(self) -> list[Conversation]:
        conversations = []
        for raw in self._raw_questions():
            cid = f"beam-{raw['conversation_id']}"
            sessions = []
            anchor = None
            for index, batch in enumerate(raw["chat"], 1):
                turns = []
                for message in batch:
                    anchor = message.get("time_anchor") or anchor
                    turns.append(
                        Turn(
                            speaker=message["role"],
                            text=message["content"],
                            dia_id=f"{cid}:{message['id']}",
                        )
                    )
                if not anchor:
                    raise DatasetError(f"{cid}: missing time_anchor; no invented timestamp")
                sessions.append(
                    Session(
                        index=index, date_time=datetime.strptime(anchor, "%B-%d-%Y"), turns=turns
                    )
                )
            conversations.append(
                Conversation(conv_id=cid, speakers=("user", "assistant"), sessions=sessions)
            )
        return conversations

    def dia_content(self) -> dict[str, str]:
        return {
            str(t.dia_id): t.text for c in self.conversations() for s in c.sessions for t in s.turns
        }

    def content_to_dias(self) -> dict[str, tuple[str, ...]]:
        bridge = defaultdict(list)
        for dia, text in self.dia_content().items():
            bridge[text].append(dia)
        return {text: tuple(ids) for text, ids in bridge.items()}
