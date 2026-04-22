"""Unit tests for per-role BM25 memory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("rank_bm25")

from aegean.investment.memory import RoleMemory, RoleMemoryRegistry


def test_recall_ranks_lexically_similar_entries_higher():
    mem = RoleMemory(key="global:bull")
    mem.add(
        "AAPL bullish earnings beat strong revenue guidance raised",
        "BUY — momentum intact, raise target",
    )
    mem.add(
        "AAPL weak guidance slowdown demand concerns",
        "HOLD — wait for next quarter",
    )
    hits = mem.recall("AAPL earnings beat guidance raised strong", n_matches=2)
    assert len(hits) == 2
    assert "BUY" in hits[0]["recommendation"]
    assert hits[0]["similarity_score"] >= hits[1]["similarity_score"]


def test_empty_memory_returns_empty_list():
    mem = RoleMemory(key="global:bear")
    assert mem.recall("any situation", n_matches=3) == []


def test_registry_scopes_by_group_and_role():
    reg = RoleMemoryRegistry()
    reg.record("bull", "situation A positive catalyst", "BUY", group_id="grp-1")
    reg.record("bear", "situation B macro shock", "SELL", group_id="grp-1")
    reg.record("bull", "different group situation", "HOLD", group_id="grp-2")

    bulls_g1 = reg.recall("bull", "positive catalyst", group_id="grp-1")
    bears_g1 = reg.recall("bear", "macro shock", group_id="grp-1")
    bulls_g2 = reg.recall("bull", "different group situation", group_id="grp-2")

    assert bulls_g1 and "BUY" in bulls_g1[0]["recommendation"]
    assert bears_g1 and "SELL" in bears_g1[0]["recommendation"]
    assert bulls_g2 and "HOLD" in bulls_g2[0]["recommendation"]


def test_registry_falls_back_to_global_when_group_empty():
    reg = RoleMemoryRegistry()
    reg.record("bull", "generic pullback value setup", "BUY on weakness", group_id=None)
    hits = reg.recall("bull", "generic pullback value setup", group_id="new-group")
    assert hits and "BUY on weakness" in hits[0]["recommendation"]


def test_persistence_round_trip(tmp_path: Path):
    persist_dir = tmp_path / "mem"
    reg1 = RoleMemoryRegistry(persist_dir=persist_dir)
    reg1.record("bull", "AAPL tailwind iPhone cycle", "BUY", group_id="g1")
    reg1.record("bull", "AAPL margin compression", "HOLD", group_id="g1")

    files = list(persist_dir.glob("*.json"))
    assert files, "memory should persist to disk"
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert len(payload["entries"]) == 2

    reg2 = RoleMemoryRegistry(persist_dir=persist_dir)
    hits = reg2.recall("bull", "AAPL iPhone cycle", group_id="g1")
    assert hits and "BUY" in hits[0]["recommendation"]
