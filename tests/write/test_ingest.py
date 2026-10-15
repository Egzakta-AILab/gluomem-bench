"""LoCoMo parser and file ingest e2e tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from gluomem.llm import FakeExtractors
from gluomem.write.gate import InjectionScanner

from gluomem_bench.ingest import (
    _parse_locomo_datetime,
    ingest_conversation,
    ingest_path,
    load_locomo,
)

FIXTURE = Path(__file__).parent.parent / "fixtures" / "locomo_tiny.json"


def test_locomo_parse_and_ingest_e2e(env, tmp_path: Path) -> None:
    assert _parse_locomo_datetime("1:56 pm on 8 May, 2023").hour == 13
    with pytest.raises(ValueError, match="unrecognized"):
        _parse_locomo_datetime("mai huitieme, treize heures")

    bad = tmp_path / "bad.json"
    bad.write_text('[{"sample_id": "x"}]', encoding="utf-8")
    with pytest.raises(ValueError, match="missing sample_id/conversation"):
        load_locomo(bad)

    convs = load_locomo(FIXTURE)
    assert len(convs) == 1
    conv = convs[0]
    assert conv.conv_id == "conv-tiny-1" and conv.speakers == ("Alice", "Bob")
    assert "shares photo: a golden retriever puppy" in conv.sessions[0].turns[3].text

    report = ingest_conversation(
        conv,
        facade=FakeExtractors(banner=False),
        scanner=InjectionScanner(env.cfg.safety),
        store=env.store,
        kg=env.kg,
        cfg=env.cfg,
    )
    assert (report.conversations, report.sessions, report.turns, report.quarantined) == (
        1,
        3,
        9,
        1,
    )
    assert report.deduped > 0

    leaked = env.db.conn.execute(
        "SELECT COUNT(*) FROM memory_frames "
        "WHERE content LIKE '%ignore all previous%' AND gop_id NOT LIKE '%:session:%'"
    ).fetchone()[0]
    assert leaked == 0

    latest_i, _ = env.store.reconstruct_state("conv-tiny-1:profile:Alice")
    assert latest_i is not None

    # Re-ingest is idempotent.
    count_before = env.db.conn.execute("SELECT COUNT(*) FROM memory_frames").fetchone()[0]
    report2 = ingest_conversation(
        conv,
        facade=FakeExtractors(banner=False),
        scanner=InjectionScanner(env.cfg.safety),
        store=env.store,
        kg=env.kg,
        cfg=env.cfg,
    )
    count_after = env.db.conn.execute("SELECT COUNT(*) FROM memory_frames").fetchone()[0]
    assert count_after == count_before and report2.i_frames == 0

    # ingest_path e2e
    path_report = ingest_path(
        FIXTURE,
        env.cfg,
        facade=FakeExtractors(banner=False),
        db_path=tmp_path / "path.mind",
        embedder=env.store.embedder,
    )
    assert path_report.conversations == 1 and path_report.turns == 9
