"""Byte-pinned BEAM scorer execution; no upstream imports or model construction.

Native prompt and metric fidelity is distinct from judge-model fidelity. A Qwen
judge is an adapted model, NOT reproduction of upstream gpt-4.1-mini judgments.
The native rubric prompt does NOT interpolate probing_question, despite its
responsiveness instruction; preserve this omission rather than repair the task.

Public API:
* NativeBeam.from_directory(root): verify all SOURCE_SHA256 bytes, then extract
  only the native scorer definitions/template without importing upstream code.
* native.evaluate(category, rubric, llm_response, probing_question, invoke):
  offline conformance seam; invoke(kind, messages) supplies authored raw text.
* score_case(native, case, guarded_rack_client, judge_model=..., judge_output_tokens=...):
  operator-admitted execution only; both explicit values must match the profile.
* rescore(bundle): dataset/network-free replay from embedded source bytes and
  ordered request/response bindings. Saved numeric scores are not trusted.
* report(bundles, expected_case_sha256=[...]): exact frozen-plan coverage and
  native history-macro reporting, with tau_norm (not F1-weighted final_score).

Cases retain every original field and require case_id, system, conversation_id,
storage_split (literal 100K/500K/1M/10M), category, question, rubric, llm_response,
and dataset provenance including the original CC-BY-SA-4.0 license. No history,
timestamp or missing 10M anchor is fabricated. This module does not admit an
inference campaign, full-dataset coverage, Qwen judge quality or reference-model
reproduction. External frozen manifests must authenticate saved bundle hashes.

Upstream source: mohammadtavakoli78/BEAM, revision below.
Code license: MIT; dataset license: CC-BY-SA-4.0 (original identifiers retained).

MIT License
Copyright (c) 2025 Mohammad Tavakoli
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import ast
import base64
import copy
import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

CODE_REVISION = "b2da22eac88bb0874c64665f13457eb99835774a"
SOURCE_SHA256 = {
    "src/prompts.py": "81fe51ab2bcd7e8ad218b3a9a969a340c38d31f90ac2d93fa78a3bb890ba157b",
    "src/evaluation/compute_metrics.py": (
        "e72bf25d36d521d166862063ec94f8e7875328193142d1edddb9b4677d47780d"
    ),
    "src/evaluation/report_results.py": (
        "1486a44fac721717c0a9373b32f4487c277d2bf28044cf28f649a6a74e692c96"
    ),
    "src/evaluation/run_evaluation.py": (
        "518894ab65860043a24259457ddf3b1ab8ae18427b6e01e5aa90cd29c4256a3b"
    ),
    "src/llm.py": "1ce32f8bf67935fc0fcf169963cce79b5f7ee4e63e25baf72b174c497d364c54",
    "src/llms_config.json": "f630bb1b9cdd9f7dd4e681ba47af900faa1c333121237a74578e30d085f1bb67",
    "LICENSE": "60c7b74ffaacaae383c3f06a48bc79d204b25bc9c1856df9711b3d290febd474",
}
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


class NativeBeam:
    """Compile only selected definitions after verifying *all* source bytes.

    Never imports the upstream package (which downloads NLTK data, constructs
    clients and loads encoders). ``invoke(kind, exact_messages) -> raw_text`` is
    an offline seam; use ``score_case`` for guarded execution and bound artifacts.
    """

    @classmethod
    def from_directory(cls, root: Path):
        return cls({name: (Path(root) / name).read_bytes() for name in SOURCE_SHA256})

    def __init__(self, sources: dict[str, bytes]):
        if set(sources) != set(SOURCE_SHA256):
            raise ValueError("incomplete native BEAM source inventory")
        for name, expected in SOURCE_SHA256.items():
            if hashlib.sha256(sources[name]).hexdigest() != expected:
                raise ValueError(f"native BEAM source hash mismatch: {name}")
        self.sources = dict(sources)
        self.source_receipt = {
            "repository": "mohammadtavakoli78/BEAM",
            "revision": CODE_REVISION,
            "code_license": "MIT",
            "dataset_license": "CC-BY-SA-4.0",
            "sha256": dict(SOURCE_SHA256),
        }
        env = {"json": json, "re": re, "List": list, "Tuple": tuple}
        tree = ast.parse(sources["src/prompts.py"])
        template = next(
            ast.literal_eval(n.value)
            for n in tree.body
            if isinstance(n, ast.Assign)
            and getattr(n.targets[0], "id", "") == "unified_llm_judge_base_prompt"
        )
        env["unified_llm_judge_base_prompt"] = template
        tree = ast.parse(sources["src/evaluation/compute_metrics.py"])
        names = {
            "parse_json_response",
            "llm_equivalence",
            "align_with_llm",
            "event_ordering_score",
        } | {f"evaluate_{c}" for c in CATEGORIES}
        selected = ast.Module(
            body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names],
            type_ignores=[],
        )
        exec(compile(selected, "<byte-pinned BEAM compute_metrics>", "exec"), env)
        self._native = env

    def evaluate(self, category, rubric, llm_response, probing_question, invoke):
        if category == "event_ordering":
            # Optional dependency required only for the native Kendall tau-b path.
            from scipy.stats import kendalltau

            self._native["kendalltau"] = kendalltau

        class Bridge:
            def invoke(self, prompt):
                kind = "rubric" if isinstance(prompt, str) else "equivalence"
                messages = [{"role": "user", "content": prompt}] if kind == "rubric" else prompt
                raw = invoke(kind, messages)
                if kind == "rubric":
                    _validate_rubric(raw)
                elif not isinstance(raw, str) or raw.strip().upper() not in ("YES", "NO"):
                    # Deliberate fail-closed narrowing of native substring-'yes'.
                    raise ValueError("invalid equivalence judgment")
                return SimpleNamespace(content=raw)

        return self._native[f"evaluate_{category}"](
            rubric=rubric,
            llm_response=llm_response,
            probing_question=probing_question,
            model=Bridge(),
        )


def _validate_rubric(raw):
    """Fail-closed narrowing: no JSON repair/coercion or prose salvage.

    Native valid JSON/fenced-JSON outputs retain their native score semantics.
    This guard deliberately rejects upstream's permissive malformed-output paths.
    """

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    try:
        text = raw.strip()
        fence = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fence:
            text = fence.group(1)
        value = json.loads(text, object_pairs_hook=unique)
        if (
            not isinstance(value, dict)
            or set(value) != {"score", "reason"}
            or type(value["score"]) not in (int, float)
            or value["score"] not in (0, 0.5, 1)
            or not isinstance(value["reason"], str)
            or not value["reason"].strip()
        ):
            raise ValueError("schema")
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("invalid rubric judgment") from exc


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _request(case, source_receipt, profile, index, kind, messages):
    return {
        "case_sha256": _digest(case),
        "source_receipt_sha256": _digest(source_receipt),
        "profile_sha256": _digest(profile),
        "index": index,
        "kind": kind,
        "task": "rubric",
        "messages": copy.deepcopy(messages),
    }


def _admit_case(case):
    try:
        if (
            case["category"] not in CATEGORIES
            or not isinstance(case["question"], str)
            or not isinstance(case["llm_response"], str)
            or not isinstance(case["rubric"], list)
            or not case["rubric"]
            or any(not isinstance(item, str) or not item.strip() for item in case["rubric"])
            or any(
                not isinstance(case[key], str) or not case[key]
                for key in ("case_id", "system", "conversation_id")
            )
            or case["storage_split"] not in ("100K", "500K", "1M", "10M")
            or case["dataset"]["license"] != "CC-BY-SA-4.0"
        ):
            raise ValueError("invalid case")
        _digest(case)  # Reject non-JSON/nonfinite metadata without dropping fields.
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("BEAM scoring admission: invalid lossless case") from exc


def _admit_profile(profile, judge_model, judge_output_tokens):
    from gluomem_bench.bench.rack import RackProfile

    checked = RackProfile.model_validate(profile)
    if (
        checked.model != judge_model
        or type(judge_output_tokens) is not int
        or checked.generation_budgets is None
        or checked.output_budget("rubric") != judge_output_tokens
    ):
        raise ValueError("BEAM scoring admission: explicit model/output must match guarded profile")
    return checked


def _validate_response(response, profile, output_tokens):
    try:
        envelope = response["response"]
        choices = envelope["choices"]
        counts = [
            response[k] for k in ("admitted_prompt_tokens", "prompt_tokens", "completion_tokens")
        ]
        if (
            envelope["model"] != profile["model"]
            or len(choices) != 1
            or choices[0]["finish_reason"] != "stop"
            or choices[0]["message"]["content"] != response["text"]
            or not isinstance(response["text"], str)
            or any(type(v) is not int or v < 0 for v in counts)
            or counts[0] != counts[1]
            or envelope["usage"]["prompt_tokens"] != counts[1]
            or envelope["usage"]["completion_tokens"] != counts[2]
            or counts[2] > output_tokens
            or counts[0] + output_tokens + profile["schema_reserve_tokens"]
            > profile["context_window"]
        ):
            raise ValueError("inconsistent response")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("invalid BEAM response envelope") from exc


def _result(scores, category):
    # Serialization-only deviation: preserve undefined native NaN as null,
    # explicitly flagged. Never convert an undefined metric to a numeric zero.
    scores = copy.deepcopy(scores)
    undefined = [
        key
        for key, value in scores.items()
        if isinstance(value, float) and not math.isfinite(value)
    ]
    for key in undefined:
        scores[key] = None
    metric = "tau_norm" if category == "event_ordering" else "llm_judge_score"
    return {
        "scores": scores,
        "report_metric": metric,
        "report_score": scores[metric],
        "undefined_metrics": undefined,
    }


def score_case(
    native: NativeBeam, case: dict, client, *, judge_model: str, judge_output_tokens: int
) -> dict:
    """Guarded, sequential native scoring; no default endpoint/model/output budget.

    Operator must explicitly admit inference before invoking this function. The
    returned JSON-serializable bundle retains source bytes, lossless case fields,
    exact ordered requests and raw rack responses for dataset-free replay.
    It is conformance-capable evidence, not execution/quality admission.
    """
    from gluomem_bench.bench.rack import RackClient

    _admit_case(case)
    if not isinstance(client, RackClient) or client.budget is None:
        raise ValueError("BEAM scoring admission: guarded RackClient with shared budget required")
    case = copy.deepcopy(case)
    profile = client.profile.model_dump(mode="json")
    _admit_profile(profile, judge_model, judge_output_tokens)
    judgments = []

    def invoke(kind, messages):
        request = _request(case, native.source_receipt, profile, len(judgments), kind, messages)
        row = {"request": request, "request_sha256": _digest(request), "status": "request_failed"}
        judgments.append(row)
        response = client.generate_messages("rubric", messages)
        row.update(response=response, response_sha256=_digest(response), status="response_received")
        row["binding_sha256"] = _digest([row["request_sha256"], row["response_sha256"]])
        _validate_response(response, profile, judge_output_tokens)
        return response["text"]

    try:
        client.preflight()
        scores = native.evaluate(
            case["category"], case["rubric"], case["llm_response"], case["question"], invoke
        )
        client.check()
    except Exception as exc:
        client._latch(exc)
        exc.beam_progress = {
            "complete": False,
            "case": case,
            "judgments": judgments,
            "generation_calls": client.budget.calls,
        }
        raise
    return {
        "protocol": "beam-native-scoring-transport@v1",
        "case": case,
        "case_sha256": _digest(case),
        "source_receipt": copy.deepcopy(native.source_receipt),
        "source_bytes_base64": {
            n: base64.b64encode(b).decode("ascii") for n, b in native.sources.items()
        },
        "judge_profile": profile,
        "judge_model": judge_model,
        "judge_output_tokens": judge_output_tokens,
        "judge_fidelity": "adapted-model; not reference-model reproduction",
        "reference_judge_model": "gpt-4.1-mini",
        "judgments": judgments,
        **_result(scores, case["category"]),
    }


def rescore(bundle: dict) -> dict:
    """Replay exact greedy calls from saved raw judgments, with no file/network IO.

    Saved numeric scores are deliberately ignored. Hashes bind bytes and roles,
    not authenticity: an external frozen manifest must authenticate the bundle
    and planned case coverage before any campaign/result admission.
    """
    native = NativeBeam(
        {
            name: base64.b64decode(value, validate=True)
            for name, value in bundle["source_bytes_base64"].items()
        }
    )
    case, profile = bundle["case"], bundle["judge_profile"]
    _admit_case(case)
    _admit_profile(profile, bundle["judge_model"], bundle["judge_output_tokens"])
    if (
        bundle["protocol"] != "beam-native-scoring-transport@v1"
        or bundle["case_sha256"] != _digest(case)
        or bundle["source_receipt"] != native.source_receipt
        or bundle["judge_fidelity"] != "adapted-model; not reference-model reproduction"
        or bundle["reference_judge_model"] != "gpt-4.1-mini"
    ):
        raise ValueError("invalid BEAM replay binding")
    index = 0

    def invoke(kind, messages):
        nonlocal index
        expected = _request(case, native.source_receipt, profile, index, kind, messages)
        if index >= len(bundle["judgments"]):
            raise ValueError("missing BEAM replay judgment")
        row = bundle["judgments"][index]
        if (
            row.get("status") != "response_received"
            or row["request_sha256"] != _digest(expected)
            or row["request_sha256"] != _digest(row["request"])
            or row["response_sha256"] != _digest(row["response"])
            or row["binding_sha256"] != _digest([row["request_sha256"], row["response_sha256"]])
        ):
            raise ValueError("invalid BEAM replay request/response binding")
        index += 1
        _validate_response(row["response"], profile, bundle["judge_output_tokens"])
        return row["response"]["text"]

    scores = native.evaluate(
        case["category"], case["rubric"], case["llm_response"], case["question"], invoke
    )
    if index != len(bundle["judgments"]):
        raise ValueError("extra BEAM replay judgments")
    return _result(scores, case["category"])


def report(bundles: list[dict], *, expected_case_sha256: list[str]) -> dict:
    """Native per-category per-history means, then equal-history averaging.

    Coverage expectations must come from the operator's frozen plan, not from
    whichever cases completed. No pooled cross-split/metric/model headline.
    Undefined tau propagates as null, never drops a case or becomes zero.
    """
    from collections import defaultdict

    if (
        not bundles
        or len(expected_case_sha256) != len(set(expected_case_sha256))
        or sorted(b["case_sha256"] for b in bundles) != sorted(expected_case_sha256)
    ):
        raise ValueError("BEAM report exact expected coverage mismatch")
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    identities, metadata = set(), {}
    for bundle in bundles:
        replay = rescore(bundle)
        case = bundle["case"]
        key = (case["system"], case["storage_split"], _digest(case["dataset"]))
        identity = (key, case["case_id"])
        if identity in identities:
            raise ValueError("duplicate BEAM report case identity")
        identities.add(identity)
        if key in metadata and metadata[key]["judge_profile"] != bundle["judge_profile"]:
            raise ValueError("mixed BEAM report judge profiles")
        metadata[key] = bundle
        grouped[key][case["category"]][case["conversation_id"]].append(replay["report_score"])
    rows = []
    for key, categories in sorted(grouped.items()):
        by_category = {}
        for category in CATEGORIES:
            if category not in categories:
                continue
            histories = categories[category]
            means = [
                sum(values) / len(values) if None not in values else None
                for values in histories.values()
            ]
            by_category[category] = {
                "metric": "tau_norm" if category == "event_ordering" else "llm_judge_score",
                "score": sum(means) / len(means) if None not in means else None,
                "histories": len(histories),
                "questions": sum(map(len, histories.values())),
                "undefined_questions": sum(v is None for vs in histories.values() for v in vs),
            }
        rows.append(
            {
                "system": key[0],
                "storage_split": key[1],
                "dataset": copy.deepcopy(metadata[key]["case"]["dataset"]),
                "by_category": by_category,
            }
        )
    return {
        "protocol": "beam-native-report@v1",
        "cases": len(bundles),
        "rows": rows,
        "scope": "exact operator-planned coverage; not full-dataset admission",
        "aggregation": "mean of per-history per-category means; splits separate",
        "judge_fidelity": "adapted-model; not reference-model reproduction",
    }
