.PHONY: install test lint typecheck check optimize-prompts

# Sync from local cache / lock without downloading model weights.
install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run pyrefly check

check: lint typecheck test

# Offline DSPy compile -> artifacts/ (requires OPENROUTER_API_KEY and trainsets; spends real tokens).
optimize-prompts:
	uv run python -m gluomem_bench.llm.optimize --task all --trainset $(or $(TRAINSETS),benchmarks_data/trainsets)
