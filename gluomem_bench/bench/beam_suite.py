"""BEAM all-tier chat-only adapter and explicitly gated scoring seams.

Never use fixture admission as evidence of a pinned upstream dataset. Existing
BeamAdapter supplies QA/rubric/bridge semantics; this adapter qualifies split IDs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gluomem_bench.bench.adapters.beam import CATEGORIES, BeamAdapter
from gluomem_bench.bench.adapters.locomo import DatasetError

SPLITS = ("100K", "500K", "1M", "10M")
HF_REVISION = "3205395e897e7318c7b094ef4e6047b9b82dbb03"
HF_10M_REVISION = "9b2096193fe74e2837e4713e483351e19817773c"
CODE_REVISION = "b2da22eac88bb0874c64665f13457eb99835774a"
# Sizes and LFS SHA256s independently inspected through the pinned HF trees.
_FILES = {
    "100K": [
        (
            "100K-00000-of-00001.parquet",
            5429768,
            "c0519be25907005ba873c927c50877471d550873039d96c041554d0075a78ace",
        )
    ],
    "500K": [
        (
            "500K-00000-of-00001.parquet",
            33956263,
            "af05921c979355038e1761b7cde3d2dd713200dd3071b278de0200f6c7f30122",
        )
    ],
    "1M": [
        (
            "1M-00000-of-00001.parquet",
            66156374,
            "41b5acbbb55a586b1305514ef9d9fb03365d9b3331b598a1c2dd7603d93ef533",
        )
    ],
    "10M": [
        (
            "10M-00000-of-00002.parquet",
            153844664,
            "31d96fd47ec56221d202e68792f26c00e49467dd4b36ee105c36ebd19ef78ad5",
        ),
        (
            "10M-00001-of-00002.parquet",
            189980875,
            "a4f13fe25af51d57405ae41008689c31d1421377f3efde56a024b441deb2ee65",
        ),
    ],
}
SOURCE_SHA256 = {
    "src/prompts.py": "81fe51ab2bcd7e8ad218b3a9a969a340c38d31f90ac2d93fa78a3bb890ba157b",
    "src/beam/download_dataset.py": (
        "bf8f70f5c3866766599b8b5112d3cbc5a68543dcd08d47bc508ec562f8e98ce7"
    ),
    "src/evaluation/compute_metrics.py": (
        "e72bf25d36d521d166862063ec94f8e7875328193142d1edddb9b4677d47780d"
    ),
    "src/evaluation/report_results.py": (
        "1486a44fac721717c0a9373b32f4487c277d2bf28044cf28f649a6a74e692c96"
    ),
}


def dataset_spec(split: str) -> dict:
    """Frozen actual upstream inventory. The advertised 128K split is 100K."""
    if split not in SPLITS:
        raise DatasetError(f"unknown literal BEAM split: {split}")
    repo = "Mohammadta/BEAM-10M" if split == "10M" else "Mohammadta/BEAM"
    revision = HF_10M_REVISION if split == "10M" else HF_REVISION
    return {
        "repository": repo,
        "revision": revision,
        "split": split,
        "license": "CC-BY-SA-4.0",
        "expected_conversation_ids": [
            str(i) for i in range(1, {"100K": 20, "500K": 35, "1M": 35, "10M": 10}[split] + 1)
        ],
        "histories": {"100K": 20, "500K": 35, "1M": 35, "10M": 10}[split],
        "files": [
            {
                "file": file,
                "size_bytes": size,
                "sha256": sha,
                "url": f"https://huggingface.co/datasets/{repo}/resolve/{revision}/data/{file}",
            }
            for file, size, sha in _FILES[split]
        ],
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def upstream_source_receipt(root: Path) -> dict:
    """Validate actual archived scorer/prompt/download bytes, not revision labels."""
    sources = []
    for name, expected in SOURCE_SHA256.items():
        path = Path(root) / name
        if not path.is_file() or file_sha256(path) != expected:
            raise DatasetError(f"upstream source missing or hash mismatch: {name}")
        sources.append({"file": name, "size_bytes": path.stat().st_size, "sha256": expected})
    return {"repository": "mohammadtavakoli78/BEAM", "revision": CODE_REVISION, "sources": sources}


def iter_parquet_adapters(split: str, data_dir: Path, *, require_timestamps: bool = True):
    """Verify every shard before yielding chat-only history adapters in file order.

    Streaming a row at a time keeps 10M practical; partial consumption is NOT
    whole-split admission. ``admission_receipt`` consumes and validates all rows.
    No download, model construction, profile/plan reads or inference occurs.
    """
    spec = dataset_spec(split)
    paths = []
    for item in spec["files"]:
        path = Path(data_dir) / item["file"]
        if not path.is_file():
            raise DatasetError(f"missing BEAM shard: {path.name}")
        if path.stat().st_size != item["size_bytes"] or file_sha256(path) != item["sha256"]:
            raise DatasetError(f"BEAM shard size/hash mismatch: {path.name}")
        paths.append(path)
    import pyarrow.parquet as pq

    files = [pq.ParquetFile(path) for path in paths]
    if sum(f.metadata.num_rows for f in files) != spec["histories"]:
        raise DatasetError("BEAM split history count mismatch")
    seen = set()
    sha = hashlib.sha256(json.dumps(spec, sort_keys=True).encode("utf-8")).hexdigest()
    for path, parquet in zip(paths, files, strict=True):
        for batch in parquet.iter_batches(
            batch_size=1, columns=["conversation_id", "chat", "probing_questions"]
        ):
            row = batch.to_pylist()[0]
            if "expected_conversation_ids" in spec:
                expected_id = spec["expected_conversation_ids"][len(seen)]
                if row["conversation_id"] != expected_id:
                    raise DatasetError("conversation_id differs from pinned split order")
            if row["conversation_id"] in seen:
                raise DatasetError("duplicate conversation_id across shards")
            seen.add(row["conversation_id"])
            adapter = BeamSuiteAdapter.from_rows(split, [row])
            adapter._path = path
            adapter._sha = sha
            adapter.verify(require_timestamps=require_timestamps)
            adapter._source_files = [
                (Path(data_dir) / f["file"], f["sha256"], f["size_bytes"]) for f in spec["files"]
            ]
            yield adapter
    if len(seen) != spec["histories"]:
        raise DatasetError("incomplete BEAM split")


def admission_receipt(split: str, data_dir: Path, *, upstream_dir: Path | None = None) -> dict:
    """Full parse coverage and chat-only input hashes, not benchmark results."""
    from collections import Counter
    from dataclasses import asdict

    spec = dataset_spec(split)
    source = upstream_source_receipt(upstream_dir) if upstream_dir is not None else None
    history_receipts = []
    categories, roles = Counter(), Counter()
    samples = []
    for adapter in iter_parquet_adapters(split, data_dir, require_timestamps=False):
        qa = adapter.load()
        samples.extend(q.id for q in qa)
        categories.update(q.category_name for q in qa)
        chat = adapter.chat_messages()
        roles.update(m["role"] for m in chat)
        blocker = None
        try:
            sessions = len(adapter.conversations()[0].sessions)
        except DatasetError as exc:
            sessions = None
            blocker = str(exc)
        chat_bytes = json.dumps(
            chat, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        history_receipts.append(
            {
                "conv_id": qa[0].conv_id,
                "questions": len(qa),
                "messages": len(chat),
                "sessions": sessions,
                "typed_ingestion_admitted": blocker is None,
                "ingestion_blocker": blocker,
                "chat_characters": sum(len(m["content"]) for m in chat),
                "roles": dict(Counter(m["role"] for m in chat)),
                "duplicate_source_id_occurrences": len(chat)
                - len(set(m["source_id"] for m in chat)),
                "chat_sha256": hashlib.sha256(chat_bytes).hexdigest(),
                "sample_sha256": hashlib.sha256(
                    json.dumps([asdict(q) for q in qa], sort_keys=True, ensure_ascii=False).encode(
                        "utf-8"
                    )
                ).hexdigest(),
            }
        )
    if len(samples) != len(set(samples)) or len(history_receipts) != spec["histories"]:
        raise DatasetError("split coverage mismatch")
    # Hashes of recoverable producer files, alongside source/data byte hashes.
    producer = {
        "beam_suite.py": file_sha256(Path(__file__)),
        "adapters/beam.py": file_sha256(Path(__file__).parent / "adapters" / "beam.py"),
    }
    return {
        "protocol": "admission:beam_all_tiers@v1",
        "complete": True,
        "storage_split": split,
        "dataset": spec,
        "sources": spec["files"],
        "upstream_code": source,
        "upstream_code_verified": source is not None,
        "producer_sha256": producer,
        "histories": len(history_receipts),
        "questions": len(samples),
        "messages": sum(r["messages"] for r in history_receipts),
        "roles": dict(roles),
        "by_category": dict(sorted(categories.items())),
        "sample_ids": samples,
        "history_receipts": history_receipts,
        "typed_ingestion_admitted": all(r["typed_ingestion_admitted"] for r in history_receipts),
        "scoring_gates": scoring_gates(),
        "inference_admitted": False,
    }


def main(argv=None):
    """python -m gluomem_bench.bench.beam_suite admit --split 10M ..."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["admit"])
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    receipt = admission_receipt(args.split, args.data_dir, upstream_dir=args.upstream_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as stream:
        stream.write(
            (
                json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                + "\n"
            ).encode("utf-8")
        )
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in (
                    "storage_split",
                    "histories",
                    "questions",
                    "messages",
                    "inference_admitted",
                )
            }
        )
    )


def scoring_gates() -> dict:
    return {
        "native_headline_admitted": False,
        "upstream_semantic_judge_implemented": False,
        "upstream_rubric_judge_implemented": False,
        "event_ordering_report_metric": "tau_norm",
        "event_ordering_objective_tail": "upstream-conformant Kendall tau-b on union ranks",
        "adapted_rubric_available": True,
        "blockers": [
            "pinned upstream LLM equivalence transport/prompt execution not implemented",
            "pinned upstream rubric judge transport/prompt execution not implemented",
            "no inference admission implied by dataset parsing",
        ],
    }


def score_rubric(scores: list[float]) -> dict:
    """Saved nugget verdicts only; not a native BEAM judge implementation."""
    if not scores or any(type(v) not in (int, float) or v not in (0, 0.5, 1) for v in scores):
        raise ValueError("rubric verdicts must be nonempty 0/0.5/1 numbers")
    return {
        "protocol": "adapted:beam_rubric@v1",
        "rubric_compliance": sum(scores) / len(scores),
        "nuggets": len(scores),
    }


def score_event_ordering(reference: list[str], answer: str, *, equivalent=None) -> dict:
    """Pinned scorer's greedy alignment + objective tau-b tail.

    ``equivalent(reference_item, system_line) -> bool`` is an explicit semantic
    judge seam, not an implementation of the upstream LLM. No implicit exact
    matching fallback. Newline splitting includes blank lines as upstream does.
    Upstream NaN is retained as JSON-safe None, with defined=False (a deviation
    in serialization, not a fabricated score). Reporting uses tau_norm, NOT the
    scorer's F1-weighted final_score. Always adapted until judge admission.
    """
    import math

    if equivalent is None:
        raise RuntimeError("an explicit semantic equivalence judge is required")
    if (
        not reference
        or any(not isinstance(r, str) for r in reference)
        or not isinstance(answer, str)
    ):
        raise ValueError("nonempty string reference list and string answer required")
    used = set()
    canonical = []
    for line in answer.split("\n"):
        matched = None
        for index, item in enumerate(reference):
            if index not in used:
                verdict = equivalent(item, line)
                if type(verdict) is not bool:
                    raise ValueError("semantic equivalence judge must return bool")
                if verdict:
                    matched = index
                    break
        if matched is None:
            canonical.append(line)
        else:
            used.add(matched)
            canonical.append(reference[matched])
    tp = len(set(reference) & set(canonical))
    fp = len([s for s in canonical if s not in reference])
    fn = len([s for s in reference if s not in canonical])
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    union = list(dict.fromkeys(reference + canonical))

    def ranks(seq):
        positions = {item: i + 1 for i, item in enumerate(seq)}
        return [positions.get(item, len(union) + 1) for item in union]

    x, y = ranks(reference), ranks(canonical)
    concordance = untied_x = untied_y = 0
    for i in range(len(union)):
        for j in range(i):
            dx, dy = x[i] - x[j], y[i] - y[j]
            untied_x += dx != 0
            untied_y += dy != 0
            concordance += (dx * dy > 0) - (dx * dy < 0)
    denominator = math.sqrt(untied_x * untied_y)
    tau_norm = (concordance / denominator + 1) / 2 if denominator else None
    return {
        "protocol": "adapted:beam_event_ordering_semantic_seam@v1",
        "report_metric": "tau_norm",
        "system_canonical": canonical,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tau_norm": tau_norm,
        "final_score": tau_norm * f1 if tau_norm is not None else None,
        "defined": tau_norm is not None,
    }


class BeamSuiteAdapter(BeamAdapter):
    """Executable adapter; only chat is exposed to conversation ingestion."""

    @classmethod
    def from_rows(cls, split: str, rows: list[dict]) -> BeamSuiteAdapter:
        if split not in SPLITS:
            raise DatasetError(f"unknown literal BEAM split: {split}")
        self = cls.__new__(cls)
        self.split = split
        self.name = f"beam_{split}"
        self._path = Path("<in-memory-fixture>")
        self._sha = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        self._raw = []
        for r in rows:
            batches = r["chat"]
            if split == "10M":
                flattened = []
                for plan in batches:
                    active = [(k, v) for k, v in plan.items() if v is not None]
                    if len(active) != 1 or not active[0][0].startswith("plan-"):
                        raise DatasetError("10M requires exactly one active chat plan per entry")
                    for batch in active[0][1]:
                        flattened.append([m for turn in batch["turns"] for m in turn])
                batches = flattened
            from collections import Counter

            counts = Counter(m["id"] for batch in batches for m in batch)
            cid = f"beam-{split}-{r['conversation_id']}"
            # Actual 1M upstream histories reset numeric IDs across batches.
            # Preserve that source token; disambiguate only the harness bridge.
            batches = [
                [
                    dict(
                        m,
                        _dia_id=f"{cid}:{m['id']}"
                        + (f":batch:{bi}:message:{mi}" if counts[m["id"]] > 1 else ""),
                    )
                    for mi, m in enumerate(batch)
                ]
                for bi, batch in enumerate(batches)
            ]
            self._raw.append(
                {
                    "conversation_id": f"{split}-{r['conversation_id']}",
                    "chat": batches,
                    "probing_questions": r["probing_questions"],
                }
            )
        return self

    def fetch(self, *, force: bool = False) -> Path:
        """Offline transport: operators must stage pinned shards deliberately."""
        if force or not self.path.is_file():
            raise DatasetError("offline BEAM adapter cannot download or replace sources")
        self.verify()
        return self.path

    def conversations(self):
        """Carry anchors forward only; preserve changes even inside a batch."""
        from datetime import datetime

        from gluomem_bench.ingest import Conversation, Session, Turn

        result = []
        for r in self._raw_questions():
            cid = f"beam-{r['conversation_id']}"
            sessions = []
            anchor = None
            for batch in r["chat"]:
                current = None
                for m in batch:
                    anchor = m.get("time_anchor") or anchor
                    if anchor is None:
                        raise DatasetError(f"{cid}: missing time_anchor; no invented timestamp")
                    try:
                        dt = datetime.strptime(anchor, "%B-%d-%Y")
                    except (TypeError, ValueError) as exc:
                        raise DatasetError(f"{cid}: invalid time_anchor") from exc
                    if current is None or current.date_time != dt:
                        current = Session(index=len(sessions) + 1, date_time=dt, turns=[])
                        sessions.append(current)
                    current.turns.append(
                        Turn(speaker=m["role"], text=m["content"], dia_id=m["_dia_id"])
                    )
            result.append(
                Conversation(conv_id=cid, speakers=("user", "assistant"), sessions=sessions)
            )
        return result

    def chat_anchors(self):
        """Original per-message nullable anchors, never profile/plan annotations."""
        return {
            m["_dia_id"]: m.get("time_anchor")
            for r in self._raw_questions()
            for batch in r["chat"]
            for m in batch
        }

    def chat_message_ids(self):
        """Harness bridge -> exact original source ID (which need not be unique)."""
        return {
            m["_dia_id"]: m["id"]
            for r in self._raw_questions()
            for batch in r["chat"]
            for m in batch
        }

    def chat_messages(self):
        """Lossless ordered chat-only projection; annotations cannot enter ingestion."""
        return [
            {
                "dia_id": m["_dia_id"],
                "source_id": m["id"],
                "role": m["role"],
                "content": m["content"],
                "time_anchor": m.get("time_anchor"),
            }
            for r in self._raw_questions()
            for batch in r["chat"]
            for m in batch
        ]

    def verify(self, *, require_timestamps: bool = True) -> None:
        for path, expected, size in getattr(self, "_source_files", []):
            if not path.is_file() or path.stat().st_size != size or file_sha256(path) != expected:
                raise DatasetError(f"BEAM source size/hash mismatch: {path.name}")
        rows = self._raw_questions()
        ids = [r["conversation_id"] for r in rows]
        if not ids or len(ids) != len(set(ids)):
            raise DatasetError("empty dataset or duplicate conversation_id")
        for r in rows:
            if not r["chat"]:
                raise DatasetError("empty chat")
            for batch in r["chat"]:
                seen = set()
                if not batch:
                    raise DatasetError("empty chat batch")
                for m in batch:
                    if m["role"] not in ("user", "assistant") or not isinstance(m["content"], str):
                        raise DatasetError("invalid message role/content")
                    if m["id"] in seen:
                        raise DatasetError("duplicate message id")
                    seen.add(m["id"])
        probes = list(self._probes())
        if not probes:
            raise DatasetError("no probing questions")
        for _, _, cat, q in probes:
            if cat not in CATEGORIES or not q["question"].strip():
                raise DatasetError("unknown category or empty question")
            if (
                not isinstance(q["rubric"], list)
                or not q["rubric"]
                or not all(isinstance(x, str) and x.strip() for x in q["rubric"])
            ):
                raise DatasetError("nonempty string rubric nuggets required")
        if require_timestamps:
            self.conversations()


if __name__ == "__main__":
    main()
