"""
ChatAgent: aggregates crowd chat into a consensus signal (V1).

Workflow:
  1. Fetch the last N minutes of chat from the chat service.
  2. Summarise the chat into a structured sentiment payload via LLM.
  3. Output a probability distribution that reflects crowd lean.

Notes:
  - The crowd is informative but herding-prone; this agent's capability_weight
    is set low (0.20) so it cannot dominate consensus.
  - The HTTP call to the chat service is wrapped so that a missing or down
    chat service degrades to "no signal" rather than blowing up consensus.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from aegean.agents.sports.base_sports_agent import (
    BaseSportsAgent,
    OUTPUT_CONTRACT,
    _extract_json,
    _normalize_probs,
)
from aegean.core.models import Solution, TokenUsage

logger = logging.getLogger(__name__)


DEFAULT_CHAT_SERVICE_URL = os.getenv(
    "CHAT_SERVICE_URL", "http://localhost:9100"
).rstrip("/")
DEFAULT_WINDOW_MINUTES = 30


class ChatFetcher:
    """
    Pluggable chat-history fetcher.

    Each call optionally targets a single user-created room. When
    `room_id` is None the chat service returns the aggregate-across-rooms
    view for the match (useful for the "global pulse" demo). When set,
    only messages from that room are returned - this is what powers
    per-room chat experiences without making the consensus engine
    "global-by-default".
    """

    def __init__(
        self,
        base_url: str = DEFAULT_CHAT_SERVICE_URL,
        timeout: float = 5.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def fetch(
        self,
        match_id: str,
        window_minutes: int = DEFAULT_WINDOW_MINUTES,
        room_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the chat window payload. On any error returns an empty window."""
        try:
            import requests
        except ImportError:
            logger.warning("requests not installed; returning empty chat window")
            return _empty_window(match_id, room_id)

        params: Dict[str, Any] = {"match_id": match_id, "minutes": window_minutes}
        if room_id:
            params["room_id"] = room_id

        try:
            r = requests.get(
                f"{self.base_url}/api/v1/chat/window",
                params=params,
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning("ChatFetcher failed (room=%s): %s; returning empty", room_id, e)
            return _empty_window(match_id, room_id)


def _empty_window(match_id: str, room_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "match_id": match_id,
        "room_id": room_id,
        "total_messages": 0,
        "messages": [],
    }


# ----------------------------- ChatAgent -----------------------------


class ChatAgent(BaseSportsAgent):
    """
    Crowd-sentiment aggregator agent.

    Differs from the other sports agents:
      - Takes a ChatFetcher (or a callable) instead of relying purely on the
        match-context prompt.
      - Pulls the chat window at predict time and folds it into the prompt.
      - Has a much lower capability_weight so the crowd cannot dominate.
    """

    ROLE = "chat_specialist"
    DEFAULT_WEIGHT = 0.20
    SPECIALIZATION_HINT = (
        "You are a crowd-sentiment analyst for football matches.\n"
        "You will receive recent user chat messages from a fan community.\n"
        "Your job: convert collective sentiment into a probability "
        "distribution.\n"
        "Important calibration notes:\n"
        "  - Crowds herd; do not blindly follow noisy majorities.\n"
        "  - Look for SPECIFIC concerns (injury rumours, lineup leaks) more "
        "    than generic cheering.\n"
        "  - If chat is sparse (<10 messages) lean toward uniform.\n"
        "  - Account for language mix; if both sides have strong fan "
        "    presence the signal is weaker."
    )

    def __init__(
        self,
        agent_id: Optional[str] = None,
        llm_client: Any = None,
        model_name: str = "",
        capability_weight: Optional[float] = None,
        chat_fetcher: Optional[ChatFetcher] = None,
        chat_fetch_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        window_minutes: int = DEFAULT_WINDOW_MINUTES,
    ):
        super().__init__(
            agent_id=agent_id,
            llm_client=llm_client,
            model_name=model_name,
            capability_weight=capability_weight,
        )
        # Either fetcher object OR raw callable. Callable wins if both given.
        # The callable signature can be either fn(match_id) or
        # fn(match_id, room_id=...); we adapt below.
        self._chat_fetcher = chat_fetcher
        self._chat_fetch_fn = chat_fetch_fn
        self.window_minutes = window_minutes

    # ----------------- override prompt building -----------------

    def _resolve_match_id(self, task: str) -> Optional[str]:
        """
        Extract match_id from the task prompt.

        Convention: tasks generated by AegeanBench's prompt builder contain
        a line like 'Match ID: <id>' or include the match metadata. We look
        for either format; absent both, return None and skip chat fetch.
        """
        import re
        m = re.search(r"match[_ ]?id[:=]\s*([A-Za-z0-9\-_]+)", task, re.IGNORECASE)
        if m:
            return m.group(1)
        m = re.search(r"\b(WC2026-[A-Z0-9]+)\b", task)
        if m:
            return m.group(1)
        return None

    @staticmethod
    def _resolve_room_id(task: str) -> Optional[str]:
        """
        Extract optional room_id from the task prompt.

        Convention used by AegeanBench: a single line like
            Room ID: room_abc123
        injected into the prompt just before the chat section.
        """
        import re
        m = re.search(
            r"room[_ ]?id[:=]\s*([A-Za-z0-9\-_]+)", task, re.IGNORECASE
        )
        return m.group(1) if m else None

    def _fetch_chat(self, match_id: str, room_id: Optional[str] = None) -> Dict[str, Any]:
        if self._chat_fetch_fn is not None:
            try:
                # Support both legacy fn(match_id) and new fn(match_id, room_id=...)
                try:
                    return self._chat_fetch_fn(match_id, room_id=room_id)
                except TypeError:
                    return self._chat_fetch_fn(match_id)
            except Exception as e:
                logger.warning("chat_fetch_fn failed: %s", e)
                return _empty_window(match_id, room_id)
        if self._chat_fetcher is not None:
            return self._chat_fetcher.fetch(match_id, self.window_minutes, room_id=room_id)
        # Default: instantiate a fresh fetcher
        return ChatFetcher().fetch(match_id, self.window_minutes, room_id=room_id)

    def _format_chat_section(self, chat: Dict[str, Any]) -> str:
        messages: List[Dict[str, Any]] = chat.get("messages", [])
        if not messages:
            return "(no recent chat messages in window)"

        lines: List[str] = []
        # Cap to last 30 messages to keep prompt size bounded
        for msg in messages[-30:]:
            user_name = msg.get("user_name", "anon")
            text = (msg.get("text") or "").strip()
            lang = msg.get("language", "")
            tag = f"[{lang}]" if lang else ""
            lines.append(f"  - {user_name}{tag}: {text}")
        header = (
            f"Recent {len(messages)} chat messages "
            f"(last {self.window_minutes} min):"
        )
        return header + "\n" + "\n".join(lines)

    def _build_prompt(self, task: str) -> str:
        match_id = self._resolve_match_id(task) or "unknown"
        room_id = self._resolve_room_id(task)
        chat = self._fetch_chat(match_id, room_id=room_id)
        chat_section = self._format_chat_section(chat)
        if room_id:
            chat_section = f"(room: {room_id})\n" + chat_section

        return (
            f"{self.SPECIALIZATION_HINT}\n\n"
            f"--- MATCH CONTEXT ---\n{task}\n\n"
            f"--- CROWD CHAT (community sentiment) ---\n{chat_section}\n\n"
            f"{OUTPUT_CONTRACT}"
        )
