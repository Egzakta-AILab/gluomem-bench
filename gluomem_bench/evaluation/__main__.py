"""Explicit offline JSON grader, separate from native benchmark scoring.

Usage: python -m gluomem_bench.evaluation CASE.json
The case contains checks/results/snapshot and may contain materializations plus
source={path,sha256,scope}. Source bytes are always read independently, never
reconstructed from a result or store snapshot. Paths are relative to CASE.json.
Check/result IDs must be unique exact built-in strings containing non-whitespace
text. Whitespace is preserved, never trimmed or normalized into another identity.
"""

import argparse
import json
from pathlib import Path

from . import bind_origin_checks, bind_report_sources, evaluate
from .grader import _validate_observation_ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    args = parser.parse_args(argv)
    case = json.loads(args.case.read_bytes())
    checks = case["checks"]
    _validate_observation_ids(case["results"], checks)
    authority = None
    if "source" in case:
        source = case["source"]
        authority = bind_report_sources(
            checks,
            source_bytes=(args.case.parent / source["path"]).read_bytes(),
            source_sha256=source["sha256"],
            scope=source["scope"],
        )
    if "materializations" in case:
        checks, _ = bind_origin_checks(checks, case["materializations"])
    rows = evaluate(case["results"], checks, case["snapshot"], correction_sources=authority)
    print(json.dumps(rows, ensure_ascii=False, sort_keys=True))
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
