"""Contract tests; authored vectors are not embedding-quality evidence."""

import pytest


def test_recipe_rejects_fp16_and_records_asymmetric_prompts():
    from gluomem_bench.bench.embedding_runtime import EmbeddingRecipe

    with pytest.raises(ValueError, match="precision"):
        EmbeddingRecipe(precision="float16")
    recipe = EmbeddingRecipe()
    assert recipe.dim == 768
    assert recipe.prefix("query") == "task: search result | query: "
    assert recipe.prefix("document") == "title: none | text: "
    assert recipe.revision == "914f7f89142e33e77833254d9c9b90c3cef7303b"


def test_encoder_checks_prefixed_lengths_before_model_and_normalizes():
    import numpy as np

    from gluomem_bench.bench.embedding_runtime import EmbeddingRecipe, LocalEncoder

    class Model:
        def __init__(self):
            self.inputs = []

        def encode(self, texts, **kwargs):
            self.inputs.append((texts, kwargs))
            return np.tile(np.array([3.0, 4.0] + [0.0] * 766), (len(texts), 1))

    model = Model()
    encoder = LocalEncoder(EmbeddingRecipe(), model=model,
                           token_count=lambda text: len(text))
    result = encoder.encode(["hello"], role="query")
    assert result["token_counts"] == [len("task: search result | query: hello")]
    assert result["data"][0]["embedding"][:2] == pytest.approx([0.6, 0.8])
    assert model.inputs[0][0] == ["task: search result | query: hello"]
    assert model.inputs[0][1]["prompt"] == ""
    before = len(model.inputs)
    with pytest.raises(ValueError, match="overflow"):
        encoder.encode(["x" * 8192], role="document")
    assert len(model.inputs) == before


@pytest.mark.parametrize("receipt_delay", [0, 0.1])
def test_loopback_service_client_preserves_roles_and_fingerprint(
    tmp_path, monkeypatch, receipt_delay
):
    import threading
    import time
    from pathlib import Path

    import numpy as np

    from gluomem_bench.bench.embedding_runtime import (
        EmbeddingRecipe,
        LocalEncoder,
        RemoteEmbedder,
        make_server,
    )

    class Model:
        def encode(self, texts, **kwargs):
            result = np.zeros((len(texts), 768), dtype=np.float32)
            for i, text in enumerate(texts):
                result[i, 0 if text.startswith("task:") else 1] = 1
            return result

    original_open = Path.open

    def delayed_receipt_open(path, mode="r", *args, **kwargs):
        if path.name == "embedding-calls.jsonl" and mode == "a":
            time.sleep(receipt_delay)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", delayed_receipt_open)
    encoder = LocalEncoder(EmbeddingRecipe(), model=Model(), token_count=len)
    server = make_server(encoder, port=0, receipt=tmp_path / "embedding-calls.jsonl")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = RemoteEmbedder(f"http://127.0.0.1:{server.server_port}/v1")
        assert client.embed("fact")[1] == 1
        assert client.embed_query("fact")[0] == 1
        assert len(client.embed_batch(["one", "two"])) == 2
        assert client.fingerprint.revision == EmbeddingRecipe().revision
        assert client.dim == 768
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    # HTTP responses precede receipt writes; close joins all request handlers.
    assert (tmp_path / "embedding-calls.jsonl").read_text().count("\n") == 3
