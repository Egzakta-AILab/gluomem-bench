"""Pinned native scoring primitives; no judges, provider defaults or dataset I/O.

MAB normalization/substring semantics adapted from HUST-AI-HYZ/MemoryAgentBench
(MIT), utils/eval_other_utils.py at 538026089d1a8a8eff05121d0db89b388f360eba.
LongMemEval's exact prompt function is loaded only from verified upstream bytes,
without executing module imports or its commercial-client entry point.
"""

from __future__ import annotations

import ast
import hashlib
import re
import string
from pathlib import Path

LONGMEMEVAL_SCORER_SHA256 = "ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251"
MAB_SCORER_SHA256 = "d77976be409298970614d477a9d8003850caddb0510e56a7e821a037d98493a2"


def load_longmemeval_prompt(path: Path):
    """Return the exact pinned get_anscheck_prompt, not paraphrased rubrics."""
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != LONGMEMEVAL_SCORER_SHA256:
        raise ValueError("upstream LongMemEval scorer byte hash mismatch")
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == "get_anscheck_prompt"]
    if len(functions) != 1:
        raise ValueError("missing upstream prompt function")
    body: list[ast.stmt] = [functions[0]]
    module = ast.Module(body=body, type_ignores=[])
    namespace = {}
    exec(compile(module, "<pinned-longmemeval-prompt>", "exec"), namespace)
    return namespace["get_anscheck_prompt"]


def _normalize(text: str) -> str:
    text = text.lower()
    text = "".join(c for c in text if c not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def mab_substring_score(prediction: str, gold: str | list) -> bool:
    """Native substring match with max over alternatives and parsed/raw output.

    This intentionally permits within-word matches and correct-plus-wrong lists.
    Those upstream metric weaknesses must not be repaired in the native score;
    use a separately named diagnostic for stale/contradictory extra answers.
    """
    if isinstance(gold, str):
        alternatives = [gold]
    elif gold and isinstance(gold[0], list):
        alternatives = [answer for group in gold for answer in group]
    else:
        alternatives = gold
    candidates = [prediction]
    for pattern in (
        re.compile(r"(?:Answer:)(.*)(?:\n|$)", flags=re.IGNORECASE),
        re.compile(r"(?:^)(.*)(?:\n|$)"),
    ):
        match = pattern.search(prediction)
        if match:
            candidates.append(re.sub(r"^Answer:", "", match[1].strip(),
                                     flags=re.IGNORECASE).strip())
            break
    return bool(max(_normalize(answer) in _normalize(candidate)
                    for candidate in candidates for answer in alternatives))
