# Bounded authored evaluation

`gluomem_bench.evaluation` is a portable, **offline** evaluator, separate from
native BEAM, LongMemEval, MemoryAgentBench and MemoryArena scoring. It evaluates
prospectively authored literal answers and actual evidence against an independent
store snapshot. It does not call a model, open a store, select an endpoint, grant
execution authority, or change existing benchmark defaults.

## API and ordering

```python
from hashlib import sha256
from gluomem_bench.evaluation import (
    apply_supplement, bind_origin_checks, bind_report_sources, evaluate,
    validate_report_projection,
)

# checks, source_bytes and scope are independently authored evaluator inputs.
# Optional: compose aliases BEFORE ingestion/answers, using an independently
# supplied digest and every original input's raw bytes. Never derive aliases
# from responses; never put these checks or accepted values in model prompts.
checks = apply_supplement(
    checks, supplement_bytes=supplement_bytes,
    supplement_sha256=independent_supplement_sha256, history_id="authored",
    original_inputs=original_input_bytes,
)
source_authority = bind_report_sources(
    checks, source_bytes=source_bytes, source_sha256=independent_source_sha256,
    scope=scope,
)
# Save materializations at source ingestion, BEFORE a later correction chooses
# its target. This binds the intended original, not the writer's chosen ID.
checks, binding_receipt = bind_origin_checks(checks, materializations)
# Validate any replayed projection before inspecting saved results/state.
validate_report_projection(checks, source_authority, effective_events)
rows = evaluate(results, checks, independent_snapshot,
                correction_sources=source_authority)
```

The source digests above must come from independent caller pins, not a result,
store snapshot or model-generated declaration. `apply_supplement` is optional;
without it no aliases are added. The retained supplement wire protocol is
`q30-prospective-evaluator-only@v1`: `original_inputs` maps authored names to raw
SHA-256 pins, `histories` maps the selected history to one update per check ID,
and `label_catalog`, `authorship`, `semantic_stale_positive_absence` are explicit
metadata. Only `accepted_values` and `correction_context_policy` can be updated;
query text, scope and source selectors cannot be rewritten by a supplement.
The absence marker must be `not_measured`. Native benchmark runners do not
implicitly load a supplement or dispatch this evaluator.

`evaluate(results, checks, snapshot)` remains available for plain literal checks.
A check using `canonical-value-only-v1` must declare `accepted_values`; only exact
trimmed values pass. No semantic equivalence, response-derived normalization or
cross-field answer guessing is performed. Missing results are `unrun`, not model
quality zeros. Invalid authority/contracts raise `Refusal` (a `ValueError`);
failed predicates return `passed: false` with diagnostics.

Both `checks` and `results` require unique IDs within their own collection. An ID
must be an exact built-in string containing non-whitespace text. Missing, null,
numeric, boolean, container, string-subclass, empty and whitespace-only IDs are
refused before grading. Case and all other whitespace remain significant; IDs
are never trimmed, stringified or otherwise repaired. Empty `results` are valid
and leave checks `unrun`. The API and CLI share this preflight; the CLI applies it
before reading source authorities or binding origins. Direct source/origin
helpers may raise ordinary `ValueError`; `evaluate` wraps its contract refusals
in `Refusal`.

## Correction and report boundary

- `source-bound-state-v1` resolves the intended target/replacement using authored
  source/message/scope selectors. `source-materialization-binding-v1` freezes the
  original content hash from its pre-correction materialization.
- Correction requires reciprocal supersession pointers, a typed correct/supersede
  edge, retirement of the intended original throughout the store (including
  active same-hash clones outside recall), and eligible replacement evidence.
- `source-bound-verbatim-report-v1` is an explicit correction-only exception to
  the eligible-payload old-literal check. Every allowed report is the **whole**
  exact declared UTF-8 span. Message/source/event/revision/role/scope, message and
  span hashes, byte boundaries, committed source/provenance and a returned
  `span_verified is True` witness must agree. A shorter genuine quote is not
  proof for a larger report payload. The target and replacement cannot exempt
  themselves as reports; the replacement needs its own committed source proof.
- Positive integer identities are type checked before joins; booleans, floats
  and missing IDs cannot authenticate one another through Python equality.
  An omitted source revision alone has the declared integer default of 1.
- A model's assertion/report label grants no authority. Captured or unseen
  paraphrases are not supported report exceptions. Literal-free semantic stale
  assertions are **not measured**, not certified absent.
- Hard-forget literal exclusion stays absolute. Do not apply report exceptions
  to withdrawal/forget lifecycle checks; a full-reader-context exclusion rejects
  this correction policy. No erased text is reconstructed or restored.

## Working offline entrypoints

The authored cases in `tests/bench/test_evaluation_contracts.py` exercise the
public API, independent source pins, report/retirement/identity failures, aliases,
and the real module entrypoint. They contain synthetic source text, not retained
conversations or captured model outputs.

```sh
python -m pytest tests/bench/test_evaluation_contracts.py
python -m gluomem_bench.evaluation case.json
```

The JSON case has `checks`, `results`, `snapshot`; optionally `materializations`
and `source: {"path": "sources.json", "sha256": "<independent pin>", "scope": {...}}`.
Source paths are relative to the case file. The CLI grades explicitly supplied
observations only, prints JSON rows, and exits 1 for predicate failures. Use the
API to compose pinned supplements prospectively before producing observations.
These commands are not a network/process sandbox: operator automation must
install offline/environment/write guards before test collection/imports.

## Evidence limits and follow-up

This is a minimal port of bounded diagnostic predicates and source-origin checks,
not the historical qualification operators, approval certificates, raw captures,
fixed campaign inputs or an additional readiness framework. Historical scenario-ID
and gold-literal fallback branches were excluded; public state checks must be
explicitly authored. Existing native scorer/admission contracts remain independent.

Offline authored contract success is not model quality, native-SDK qualification,
comparative accuracy, or permission to spend. Current runner integration and
source-bound operator activation still need separately reviewed evidence.
MetaVerifier catalogue/origin work remains **TODO**; this publication does not
claim it repaired. See [publication status](PUBLICATION_STATUS.md).
