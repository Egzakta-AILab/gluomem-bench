"""Official MAB FactConsolidation bytes, adapted 6K document-chunk protocol.

Reuses existing SHA-pinned download/verification and conversation contracts.
Not the synthetic mab_fixture; not a claim of whole-benchmark/native execution.
4096-character lossless chunks differ from upstream token chunks. The epoch
session date is a format placeholder only; production document events omit it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

from gluomem_bench.bench.adapters.locomo import DatasetError
from gluomem_bench.bench.adapters.longmemeval import LongMemEvalAdapter
from gluomem_bench.bench.types import Sample
from gluomem_bench.ingest import Conversation, Session, Turn

SOURCES = {"factconsolidation_sh_6k": 1, "factconsolidation_mh_6k": 2}
CONFLICT_INSTRUCTIONS = (
    "Use only the provided knowledge pool, not real-world facts. "
    "Each fact begins with a serial number; a larger serial number means a newer fact. "
    "Resolve conflicting facts by choosing the fact with the largest serial number. "
    "Answer the question concisely using that rule."
)


class MemoryAgentBenchConflictAdapter(LongMemEvalAdapter):
    def __init__(self, cfg, *, path=None, chunk_chars=4096):
        super().__init__(cfg, "memoryagentbench_conflict", path=path)
        if type(chunk_chars) is not int or chunk_chars < 1:
            raise ValueError("positive integer chunk_chars required")
        self.chunk_chars = chunk_chars

    def _raw_questions(self):
        if self._raw is None:
            self.verify()
            import pyarrow.parquet as pq

            rows = pq.read_table(self.path).to_pylist()
            self._raw = [r for r in rows if r.get("metadata", {}).get("source") in SOURCES]
            if sorted(r["metadata"]["source"] for r in self._raw) != sorted(SOURCES):
                raise DatasetError("exactly one row per selected MAB source required")
            for row in self._raw:
                if not isinstance(row["context"], str) or not row["context"]:
                    raise DatasetError("nonempty MAB context required")
                lengths = [len(row[k]) for k in ("questions", "answers")]
                lengths.append(len(row["metadata"]["qa_pair_ids"]))
                if len(set(lengths)) != 1 or not lengths[0]:
                    raise DatasetError("parallel nonempty MAB QA arrays required")
        return self._raw

    @staticmethod
    def _history_id(row):
        return "mab-" + hashlib.sha256(row["context"].encode()).hexdigest()

    def load(self):
        samples = []
        for row in self._raw_questions():
            source = row["metadata"]["source"]
            for qid, question, gold in zip(row["metadata"]["qa_pair_ids"], row["questions"],
                                            row["answers"], strict=True):
                if (not qid or not question or not gold
                        or not all(isinstance(a, str) and a for a in gold)):
                    raise DatasetError("nonempty MAB IDs, questions and gold alternatives required")
                samples.append(Sample(str(qid), self._history_id(row), question,
                                      json.dumps(gold), SOURCES[source], source,
                                      task_instructions=CONFLICT_INSTRUCTIONS))
        if len({s.id for s in samples}) != len(samples):
            raise DatasetError("duplicate MAB question IDs")
        return samples

    def conversations(self):
        histories = {}
        for row in self._raw_questions():
            cid = self._history_id(row)
            context = row["context"]
            if cid in histories:
                continue
            turns = [Turn(speaker="document", text=context[i:i + self.chunk_chars],
                          dia_id=f"{cid}:chunk:{i // self.chunk_chars}")
                     for i in range(0, len(context), self.chunk_chars)]
            histories[cid] = Conversation(
                conv_id=cid, speakers=("document", "document"),
                sessions=[Session(index=1, date_time=datetime(1970, 1, 1), turns=turns)],
            )
        return list(histories.values())
