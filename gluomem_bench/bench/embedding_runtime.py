"""Pinned local embedding recipe and shared, explicit text-role transport.

This runtime is independent of memory-product dependency environments. Benchmark
admission must exercise the actual model; unit-test vectors are not quality data.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def safe_failure_diagnostic(exc):
    """JSON-safe failure contract; never format untrusted exceptions or text.

    Callers must use this at every collector, not only the originating encoder.
    The original exception (including cause/traceback) stays in memory only.
    """
    kind = "HTTPError" if isinstance(exc, HTTPError) else "Exception"
    if kind == "Exception":
        for builtin, label in (
            (OSError, "OSError"),
            (TimeoutError, "TimeoutError"),
            (ConnectionError, "ConnectionError"),
            (ValueError, "ValueError"),
            (TypeError, "TypeError"),
            (RuntimeError, "RuntimeError"),
        ):
            if type(exc) is builtin:
                kind = label
                break
    result = {"type": kind,
              "message": "encoder_http_error" if kind == "HTTPError" else "operation_failed"}
    try:
        source = exc.__dict__.get("encoder_failure")
        if type(source) is dict:
            metadata = {}
            status = source.get("status")
            if type(status) is int and 100 <= status <= 599:
                metadata["status"] = status
            role = source.get("role")
            if type(role) is str and role in ("query", "document"):
                metadata["role"] = role
            for key in ("input_sha256", "body_sha256"):
                value = source.get(key)
                if (type(value) is str and len(value) == 64
                        and all(char in "0123456789abcdef" for char in value)):
                    metadata[key] = value
            count = source.get("body_bytes_captured")
            if type(count) is int and 0 <= count <= 4096:
                metadata["body_bytes_captured"] = count
            truncated = source.get("body_truncated")
            if type(truncated) is bool:
                metadata["body_truncated"] = truncated
            capture = source.get("body_capture")
            if type(capture) is str and capture == "unavailable":
                metadata["body_capture"] = "unavailable"
            if metadata:
                result["encoder_failure"] = metadata
    except Exception:
        pass
    return result


@dataclass(frozen=True)
class EmbeddingRecipe:
    model: str = "google/embeddinggemma-2"
    revision: str = "914f7f89142e33e77833254d9c9b90c3cef7303b"
    dim: int = 768
    precision: str = "bfloat16"
    max_tokens: int = 8192
    text_only: bool = True

    def __post_init__(self):
        if self.precision not in {"bfloat16", "float32"}:
            raise ValueError("precision must be bfloat16 or float32; float16 is unsafe")
        if self.dim != 768 or self.max_tokens != 8192:
            raise ValueError("this qualified recipe requires native dimensions and context")

    def prefix(self, role):
        if role == "query":
            return "task: search result | query: "
        if role == "document":
            return "title: none | text: "
        raise ValueError("role must be query or document")

    def as_dict(self):
        return asdict(self)


class LocalEncoder:
    """One loaded checkpoint; explicit prefixes, no truncation, serialized GPU use."""

    def __init__(self, recipe, *, model, token_count):
        self.recipe = recipe
        self.model = model
        self.token_count = token_count
        self._lock = Lock()

    @classmethod
    def load(cls, recipe, *, device="cuda", cache_folder=None):
        import torch
        from sentence_transformers import SentenceTransformer

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; refusing silent CPU fallback")
        dtype = getattr(torch, recipe.precision)
        if recipe.precision == "bfloat16" and device == "cuda":
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("BF16 requested but unsupported")
        config = {"vision_config": None, "audio_config": None} if recipe.text_only else {}
        model = SentenceTransformer(
            recipe.model, revision=recipe.revision, device=device, cache_folder=cache_folder,
            model_kwargs={"torch_dtype": dtype}, config_kwargs=config,
        )
        tokenizer = model.tokenizer

        def count(text):
            return len(tokenizer.encode(text, add_special_tokens=True, truncation=False))

        return cls(recipe, model=model, token_count=count)

    def encode(self, texts, *, role="document"):
        import numpy as np

        if not isinstance(texts, list) or not texts or not all(
            isinstance(text, str) and text.strip() for text in texts
        ):
            raise ValueError("nonempty list of nonempty text inputs required")
        prefix = self.recipe.prefix(role)
        prepared = [prefix + text for text in texts]
        counts = [self.token_count(text) for text in prepared]
        if any(count > self.recipe.max_tokens for count in counts):
            raise ValueError("embedding context overflow; no truncation")
        with self._lock:
            values = self.model.encode(
                prepared, prompt="", convert_to_numpy=True, normalize_embeddings=True,
                show_progress_bar=False, batch_size=32,
            )
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.shape != (len(texts), self.recipe.dim) or not np.isfinite(matrix).all():
            raise ValueError("invalid embedding dimensions or nonfinite values")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if (norms <= 0).any():
            raise ValueError("zero embedding norm")
        matrix = matrix / norms
        return {
            "object": "list", "model": self.recipe.model,
            "data": [{"object": "embedding", "index": i, "embedding": row.tolist()}
                     for i, row in enumerate(matrix)],
            "usage": {"prompt_tokens": sum(counts), "total_tokens": sum(counts)},
            "token_counts": counts, "recipe": self.recipe.as_dict(), "role": role,
        }


def make_server(encoder, *, port, receipt):
    """Loopback-only OpenAI-shaped endpoint with explicit role extension.

    Native callers omitting input_type receive document encoding. Query adapters
    must request query encoding explicitly; qualification verifies that contract.
    """
    receipt = Path(receipt)
    receipt_lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, status, value):
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/models":
                self.send_json(200, {"data": [{"id": encoder.recipe.model,
                                             "recipe": encoder.recipe.as_dict()}]})
            else:
                self.send_json(404, {"error": "unknown endpoint"})

        def do_POST(self):
            started = time.monotonic()
            record = {"complete": False}
            try:
                if self.path != "/v1/embeddings":
                    self.send_json(404, {"error": "unknown endpoint"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 4_000_000:
                    raise ValueError("invalid embedding request size")
                payload = json.loads(self.rfile.read(length))
                if payload.get("model") != encoder.recipe.model:
                    raise ValueError("embedding model differs from frozen recipe")
                values = payload["input"]
                texts = [values] if isinstance(values, str) else values
                role = payload.get("input_type", "document")
                record.update(input_sha256=hashlib.sha256(
                    json.dumps(texts, ensure_ascii=False).encode()).hexdigest(), role=role)
                result = encoder.encode(texts, role=role)
                record.update(complete=True, items=len(texts),
                              token_counts=result["token_counts"], recipe=result["recipe"])
                self.send_json(200, result)
            except Exception as exc:
                record["error"] = safe_failure_diagnostic(exc)
                if record.get("role") not in ("query", "document"):
                    record.pop("role", None)
                self.send_json(400, {"error": record["error"]})
            finally:
                record["elapsed_seconds"] = time.monotonic() - started
                receipt.parent.mkdir(parents=True, exist_ok=True)
                with receipt_lock, receipt.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


class RemoteEmbedder:
    """GluoMem Embedder protocol backed by the shared local encoder."""

    def __init__(self, base_url, *, recipe=None, timeout=120):
        self.base_url = base_url.rstrip("/")
        self.recipe = recipe or EmbeddingRecipe()
        self.dim = self.recipe.dim
        self.timeout = timeout

    @property
    def fingerprint(self):
        from gluomem.embedding.fingerprint import EmbedderFingerprint

        return EmbedderFingerprint(
            implementation="shared-local-sentence-transformers",
            model=self.recipe.model, revision=self.recipe.revision, dim=self.dim,
            preprocessing=json.dumps(self.recipe.as_dict(), sort_keys=True)
            + ";query=SearchQuery;document=Document;l2=true",
        )

    def _encode(self, texts, role):
        import numpy as np

        payload = {"model": self.recipe.model, "input": texts, "input_type": role}
        request = Request(self.base_url + "/embeddings", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except HTTPError as exc:
            # Untrusted reason, headers, URL, model and body never become text
            # diagnostics. Hash only the bounded prefix, not a historical body.
            try:
                metadata = {
                    "input_sha256": hashlib.sha256(
                        json.dumps(texts, ensure_ascii=False).encode()).hexdigest(),
                }
                if type(exc.code) is int and 100 <= exc.code <= 599:
                    metadata["status"] = exc.code
                if type(role) is str and role in ("query", "document"):
                    metadata["role"] = role
                try:
                    captured = exc.read(4097)
                    metadata.update(body_sha256=hashlib.sha256(captured[:4096]).hexdigest(),
                                    body_truncated=len(captured) > 4096,
                                    body_bytes_captured=min(len(captured), 4096))
                except Exception:
                    metadata["body_capture"] = "unavailable"
                exc.encoder_failure = metadata
            except Exception:
                # Diagnostic computation/attachment must not mask the cause.
                pass
            raise
        if result.get("recipe") != self.recipe.as_dict() or result.get("role") != role:
            raise ValueError("embedding response recipe or input role mismatch")
        rows = result["data"]
        if [row["index"] for row in rows] != list(range(len(texts))):
            raise ValueError("embedding response coverage/order mismatch")
        values = [np.asarray(row["embedding"], dtype=np.float32) for row in rows]
        if any(v.shape != (self.dim,) or not np.isfinite(v).all()
               or not np.isclose(np.linalg.norm(v), 1, atol=1e-4) for v in values):
            raise ValueError("embedding response invalid vector")
        return values

    def embed(self, text):
        return self._encode([text], "document")[0]

    def embed_batch(self, texts):
        return self._encode(texts, "document") if texts else []

    def embed_query(self, text):
        return self._encode([text], "query")[0]


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--precision", choices=["bfloat16", "float32"], default="bfloat16")
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError("fresh embedding receipts required")
    encoder = LocalEncoder.load(EmbeddingRecipe(precision=args.precision), device=args.device)
    print(json.dumps({"ready": True, "recipe": encoder.recipe.as_dict(),
                      "device": str(encoder.model.device), "port": args.port}), flush=True)
    server = make_server(encoder, port=args.port, receipt=args.receipt)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
