"""Runtime data must be frozen with the code which executes it."""
from pathlib import Path

import pytest


@pytest.mark.parametrize('driver_name', ['comprehensive_driver', 'beam_driver', 'arena_driver'])
def test_full_inventory_covers_core_sql_and_dspy_prompts(tmp_path, driver_name):
    import importlib
    import gluomem

    driver = importlib.import_module('gluomem_bench.bench.' + driver_name)
    core = Path(gluomem.__file__).resolve().parent
    resources = {core / 'storage/schema.sql'}
    resources.update((core.parent / 'artifacts/dspy').glob('*.json'))
    assert len(resources) >= 4
    if driver_name == 'comprehensive_driver':
        source, judge = tmp_path / 'data.json', tmp_path / 'judge.py'
        source.write_bytes(b'[]')
        judge.write_bytes(b'# authored inventory fixture\n')
        inventory = driver.frozen_inventory(source, judge)
    elif driver_name == 'arena_driver':
        inventory = driver._code_inventory()
    else:
        from gluomem_bench.bench.beam_scoring import SOURCE_SHA256
        from gluomem_bench.bench.beam_suite import SOURCE_SHA256 as ADAPTER_SOURCES
        from gluomem_bench.bench.beam_suite import dataset_spec

        for relative in set(SOURCE_SHA256) | set(ADAPTER_SOURCES):
            path = tmp_path / 'upstream' / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'# authored inventory fixture\n')
        for row in dataset_spec('100K')['files']:
            path = tmp_path / 'data' / row['file']
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'authored file-inventory boundary fixture')
        inventory = driver.frozen_inventory('100K', tmp_path / 'data',
                                            tmp_path / 'upstream', ['no_memory'])
    frozen = {Path(item['path']) for item in inventory}
    assert resources <= frozen, f'Unfrozen runtime inputs: {resources - frozen}'
