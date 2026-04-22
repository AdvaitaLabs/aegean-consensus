"""Per-role episodic memory using BM25 lexical retrieval.

Each investment role (fundamental_specialist, valuation_specialist, bull,
bear, ...) keeps its own store of (situation, recommendation, outcome)
triples. Retrieval runs BM25 against past situations — no embeddings,
no API calls, fully offline.

Inspired by TradingAgents' FinancialSituationMemory, adapted to aegean's
group-scoped investment analysis pipeline.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class MemoryEntry:
    situation: str
    recommendation: str
    outcome: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


_TOKEN_RE = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


class RoleMemory:
    """BM25-backed memory for a single role scope.

    Scope key is opaque — typically ``{group_id}:{role}`` or ``global:{role}``.
    """

    def __init__(self, key: str, persist_path: Optional[Path] = None):
        self.key = key
        self.persist_path = Path(persist_path) if persist_path else None
        self._entries: List[MemoryEntry] = []
        self._bm25: Any = None
        self._lock = threading.Lock()
        if self.persist_path and self.persist_path.exists():
            self._load()

    def add(
        self,
        situation: str,
        recommendation: str,
        outcome: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: str = "",
    ) -> None:
        if not situation or not recommendation:
            return
        with self._lock:
            self._entries.append(
                MemoryEntry(
                    situation=situation,
                    recommendation=recommendation,
                    outcome=outcome or {},
                    metadata=metadata or {},
                    created_at=created_at,
                )
            )
            self._rebuild()
            self._save()

    def recall(self, situation: str, n_matches: int = 2) -> List[Dict[str, Any]]:
        if not situation or not self._entries or self._bm25 is None:
            return []
        query = _tokenize(situation)
        if not query:
            return []
        scores = self._bm25.get_scores(query)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top = order[: max(1, n_matches)]
        max_score = max(scores) if len(scores) and max(scores) > 0 else 1.0
        results: List[Dict[str, Any]] = []
        for idx in top:
            if scores[idx] <= 0:
                continue
            entry = self._entries[idx]
            results.append(
                {
                    "matched_situation": entry.situation,
                    "recommendation": entry.recommendation,
                    "outcome": entry.outcome,
                    "metadata": entry.metadata,
                    "created_at": entry.created_at,
                    "similarity_score": float(scores[idx] / max_score),
                }
            )
        return results

    def update_outcome(self, index: int, outcome: Dict[str, Any]) -> None:
        with self._lock:
            if 0 <= index < len(self._entries):
                self._entries[index].outcome = outcome
                self._save()

    def entries(self) -> List[MemoryEntry]:
        return list(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries = []
            self._bm25 = None
            self._save()

    def _rebuild(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            self._bm25 = None
            return
        if not self._entries:
            self._bm25 = None
            return
        self._bm25 = BM25Okapi([_tokenize(e.situation) for e in self._entries])

    def _save(self) -> None:
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"key": self.key, "entries": [asdict(e) for e in self._entries]}
        tmp = self.persist_path.with_suffix(self.persist_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.persist_path)

    def _load(self) -> None:
        try:
            payload = json.loads(self.persist_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        raw_entries = payload.get("entries") or []
        self._entries = [
            MemoryEntry(
                situation=e.get("situation", ""),
                recommendation=e.get("recommendation", ""),
                outcome=e.get("outcome", {}) or {},
                metadata=e.get("metadata", {}) or {},
                created_at=e.get("created_at", ""),
            )
            for e in raw_entries
            if e.get("situation") and e.get("recommendation")
        ]
        self._rebuild()


class RoleMemoryRegistry:
    """Holds one :class:`RoleMemory` per (group_id, role) pair."""

    GLOBAL_GROUP = "global"

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self._memories: Dict[str, RoleMemory] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _scope_key(role: str, group_id: Optional[str]) -> str:
        return f"{group_id or RoleMemoryRegistry.GLOBAL_GROUP}:{role}"

    def get(self, role: str, group_id: Optional[str] = None) -> RoleMemory:
        key = self._scope_key(role, group_id)
        with self._lock:
            memory = self._memories.get(key)
            if memory is None:
                persist_path = None
                if self.persist_dir:
                    safe = key.replace("/", "_").replace(":", "__")
                    persist_path = self.persist_dir / f"{safe}.json"
                memory = RoleMemory(key=key, persist_path=persist_path)
                self._memories[key] = memory
            return memory

    def recall(
        self,
        role: str,
        situation: str,
        group_id: Optional[str] = None,
        n_matches: int = 2,
        include_global_fallback: bool = True,
    ) -> List[Dict[str, Any]]:
        results = self.get(role, group_id).recall(situation, n_matches=n_matches)
        if results or not include_global_fallback or not group_id:
            return results
        return self.get(role, None).recall(situation, n_matches=n_matches)

    def record(
        self,
        role: str,
        situation: str,
        recommendation: str,
        group_id: Optional[str] = None,
        outcome: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: str = "",
    ) -> None:
        self.get(role, group_id).add(
            situation=situation,
            recommendation=recommendation,
            outcome=outcome,
            metadata=metadata,
            created_at=created_at,
        )
