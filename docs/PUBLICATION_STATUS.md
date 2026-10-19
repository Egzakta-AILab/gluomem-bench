# Publication status and reproducibility

## Scope

This source publication combines multi-family MAB, BEAM, V2 and MemoryArena
harnesses with native-comparison, accounting, privacy, resource-lifecycle and
structured-schema repairs. Contract tests with authored inputs are not measured
benchmark quality, installed-SDK parity, or permission to execute a campaign.
The companion GluoMem core changes are required for structured adapters,
field-length constraints and the current configuration contract.

The latest retained GluoMem development attempt remains failed and paused:
0 of 3 histories completed, all 3 failed, with 816 generation calls spent.
Whitespace-only generation remains unresolved. Do not infer a quality ranking,
reuse failed outcomes as successful results, refund historical spend, relabel
older runs with newer protocols, or restart inference from this publication.
These figures describe the operator's historical record, not a new run performed
while assembling this change. Complete Mem0, Cognee and other baseline histories
under separate reviewed authority before making comparative claims.

## Method boundaries

- Mem0's lossless token-block input policy is an **adapted** condition, using
  4096-token blocks. Preserve source timestamps as metadata. Native execution-day
  observations do not prove historical source-clock execution; the retained
  Mem0 2.2.1 timestamp/reference-date API is not historical-clock support.
- Cognee's operation budget derives from the same absolute job deadline,
  retaining cleanup reserve. Disclose request (600 seconds), preflight
  (30 seconds), and native LLM (90 seconds) timeouts separately. Cleanup reserve
  is an explicit slice inside the operation deadline, not an extra phase budget.
  The native adapter enforces Cognee 1.6.5; the
  optional dependency declaration does not certify an installed environment.
- The opt-in v7 operating point permits extraction 16384, consolidation 32768,
  reader 4096 and judge 4096 output tokens, HTTP timeout 1800 seconds, history
  21600 seconds, 16384 calls, context 131072 and schema reserve 2048 tokens.
  These are explicit profile inputs, not new defaults. Older profile ceilings
  remain enforced and old experimental conditions remain unchanged.
- Resource accounting must retain the shared native group 4096 MiB ceiling,
  host reserve 8192 MiB and local 12288 MiB boundary. An unknown measurement is
  not zero RSS. Failure to establish resource identity must fail closed.
- Native LM15 evidence is a typed response projection, not raw wire capture.
  Length failures must not silently drop claims, default to ADD, truncate data,
  recover after length exhaustion or reuse old outcomes.

## Offline checks and operator inputs

The PR's test additions have been reduced: 70 newly added test/fixture files
(23,370 lines) were removed. Existing upstream tests are preserved. Retained
coverage focuses on diagnostic privacy, budget accounting, source/scoring
integrity, resource membership and native backend boundaries. Tests referenced
by production frozen inventories, plus their helper dependencies, remain so
that removing tests does not break inventory construction or admission checks.
This is intentionally reduced coverage, not evidence that every removed test
was redundant or that the broader harness is now fully verified.

Use an isolated environment with the declared development dependencies and the
companion core checkout. Disable dotenv autoload and external network access;
use authored loopback fixtures and newly owned scratch SQLite stores only.
Collection and fake-backed regression checks do not require model downloads,
Docker, native runtime workers, retained data or live endpoints.

```sh
python -B -I -c "import sys; sys.path.insert(0, '.'); import pytest; raise SystemExit(pytest.main(['--collect-only', '-q']))"
```

Run checks behind an offline boundary in operator automation; the command above
is collection syntax, not itself a network/process sandbox. Some older tests
still require separately supplied native source fixtures or retained evidence;
the publication handoff reports actual passes, failures and exclusions. Do not
interpret collection success as a green full suite or real-model qualification.
No environment was installed or upgraded for the publication checks, and the
lockfile has not yet been regenerated for the added Cognee optional extra.

Endpoint/model profiles must be supplied explicitly and kept outside the repo.
Do not publish API keys, workstation paths, private hostnames, raw questions,
gold answers, generated responses, datasets, stores or frozen experiment trees.
Campaign-specific continuation operators and sealed-evidence replay tests are
not included; the reusable continuation accounting boundary is included without
historical pins. Upstream licenses and scoring attribution remain in NOTICE.
