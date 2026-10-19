"""Cached CPU-only token counting for an explicitly adapted Mem0 condition."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

from gluomem_bench.bench.embedding_runtime import EmbeddingRecipe

TOKENIZER_SHA256 = "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"


@lru_cache(maxsize=1)
def _decode(raw):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def mem0_token_counter(cfg):
    """Authenticate bytes BEFORE decode; no model, service, hub or path defaults.

    The operator path and expected hash come from the frozen effective config.
    A fresh hash check on each construction prevents cached path-mutation bypass.
    The parsed Rust tokenizer alone is reused; no Torch/Transformers imports.
    """
    policy = cfg.benchmark.mem0
    if policy.max_batch_input_tokens is None:
        return None
    recipe = EmbeddingRecipe()
    for key in (cfg.embedding.model, policy.embedder_model):
        entry = cfg.models[key]
        if (entry.provider_model, entry.revision, entry.dim) != (
                recipe.model, recipe.revision, recipe.dim):
            raise ValueError("Mem0 token counter requires pinned encoder recipe")
    if (not policy.tokenizer_path or policy.tokenizer_sha256 != TOKENIZER_SHA256):
        raise ValueError("Mem0 registered tokenizer path and SHA256 required")
    path = Path(policy.tokenizer_path)
    if not path.is_absolute():
        raise ValueError("Mem0 tokenizer requires an absolute operator path")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != policy.tokenizer_sha256:
        raise ValueError("Mem0 tokenizer byte hash mismatch")
    library = version("tokenizers")
    if library != "0.23.3":
        raise ValueError("Mem0 token counter library version differs")
    tokenizer = _decode(raw)

    def count(text):
        return len(tokenizer.encode(recipe.prefix("query") + text,
                                    add_special_tokens=True).ids)

    count.metadata = {"protocol": "cached-rust-query-tokenizer@v1",
                      "tokenizer_sha256": policy.tokenizer_sha256,
                      "library": "tokenizers", "library_version": library,
                      "recipe": recipe.as_dict(), "query_prefix": recipe.prefix("query"),
                      "include_special_tokens": True, "truncation": False,
                      "padding": False,
                      "counter_source_sha256": hashlib.sha256(
                          Path(__file__).read_bytes()).hexdigest()}
    count.metadata["recipe_sha256"] = hashlib.sha256(json.dumps(
        count.metadata, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return count
