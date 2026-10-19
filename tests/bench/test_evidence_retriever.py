"""Matched raw evidence uses allowed turns, not answer/evidence annotations."""

from datetime import datetime
from types import SimpleNamespace

from gluomem_bench.bench.types import Sample
from gluomem_bench.ingest import Conversation, Session, Turn
from gluomem_bench.llm.outputs import AnswerResult


def test_raw_retriever_preserves_roles_and_dates_and_excludes_gold():
    from gluomem_bench.bench.evidence_retriever import RawCampaignRetriever

    conv = Conversation(conv_id="authored", speakers=("user", "assistant"), sessions=[
        Session(index=1, date_time=datetime(2026, 1, 1), turns=[
            Turn(speaker="assistant", text="The editor is Neovim.", dia_id="source-turn-1"),
            Turn(speaker="user", text="My city is Prague.", dia_id="source-turn-2"),
        ])])
    calls = []

    def answer(question, context):
        calls.append((question, context))
        return AnswerResult(text="Neovim", prompt_tokens=30, completion_tokens=3)

    cfg = SimpleNamespace(benchmark=SimpleNamespace(answer=SimpleNamespace(context_k=50)))
    retriever = RawCampaignRetriever(cfg, conv, answer, mode="bm25")
    try:
        sample = Sample("q", "authored", "What editor?", "GOLD_ONLY_POISON", 1,
                        "synthetic", evidence=("ANNOTATION_ONLY_POISON",),
                        question_date="2026/01/02")
        prediction = retriever.answer(sample)
        assert prediction.answer == "Neovim"
        assert prediction.retrieved[0].dia_id == "source-turn-1"
        assert "assistant" in calls[0][1]
        assert "2026-01-01" in calls[0][1]
        assert "GOLD_ONLY_POISON" not in str(calls)
        assert "ANNOTATION_ONLY_POISON" not in str(calls)
        assert "2026/01/02" in calls[0][0]
    finally:
        retriever.close()


def test_dense_and_hybrid_share_chunks_and_asymmetric_encoder_calls():
    import numpy as np

    from gluomem_bench.bench.evidence_retriever import RawCampaignRetriever

    class Encoder:
        def __init__(self):
            self.documents = []
            self.queries = []

        def embed_batch(self, texts):
            self.documents.append(list(texts))
            return [np.array([1.0, 0.0]) for _ in texts]

        def embed_query(self, text):
            self.queries.append(text)
            return np.array([1.0, 0.0])

    conv = Conversation(conv_id="authored", speakers=("user", "assistant"), sessions=[
        Session(index=1, date_time=datetime(2026, 1, 1), turns=[
            Turn(speaker="user", text="Neovim is my editor.", dia_id="source-turn")])])
    cfg = SimpleNamespace(benchmark=SimpleNamespace(answer=SimpleNamespace(context_k=50)))
    encoder = Encoder()
    sample = Sample("q", "authored", "What editor?", "unused", 1, "synthetic")
    retrievers = [RawCampaignRetriever(
        cfg, conv, lambda question, context: AnswerResult(text="Neovim"),
        mode=mode, embedder=encoder,
    ) for mode in ("dense", "hybrid")]
    try:
        for retriever in retrievers:
            assert retriever.answer(sample).retrieved[0].dia_id == "source-turn"
        assert encoder.documents[0] == encoder.documents[1]
        assert encoder.queries == [sample.question, sample.question]
    finally:
        for retriever in retrievers:
            retriever.close()
