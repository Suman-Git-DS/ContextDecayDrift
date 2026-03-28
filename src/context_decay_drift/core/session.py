"""Session management for tracking conversation history and drift over time."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Turn:
    """A single conversation turn."""

    role: str  # "user", "assistant", or "system"
    content: str
    turn_number: int


@dataclass
class Session:
    """Tracks a conversation session with its system prompt and turn history.

    A session represents a single continuous conversation with an LLM.
    It stores the system prompt (the ground truth for drift measurement)
    and all conversation turns.

    Args:
        system_prompt: The original system prompt / instructions for the LLM.
        session_id: Optional identifier. Auto-generated if not provided.
    """

    system_prompt: str
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    turns: list[Turn] = field(default_factory=list)
    _turn_counter: int = field(default=0, repr=False)

    def add_user_message(self, content: str) -> Turn:
        """Record a user message."""
        return self._add_turn("user", content)

    def add_assistant_message(self, content: str) -> Turn:
        """Record an assistant response."""
        return self._add_turn("assistant", content)

    def _add_turn(self, role: str, content: str) -> Turn:
        self._turn_counter += 1
        turn = Turn(role=role, content=content, turn_number=self._turn_counter)
        self.turns.append(turn)
        return turn

    @property
    def turn_count(self) -> int:
        """Total number of turns in the session."""
        return self._turn_counter

    @property
    def assistant_turns(self) -> list[Turn]:
        """All assistant responses in order."""
        return [t for t in self.turns if t.role == "assistant"]

    @property
    def user_turns(self) -> list[Turn]:
        """All user messages in order."""
        return [t for t in self.turns if t.role == "user"]

    def get_recent_context(self, n: int = 5) -> list[Turn]:
        """Get the last n turns of conversation."""
        return self.turns[-n:]

    def get_full_text(self) -> str:
        """Concatenate all turn content into a single string."""
        return " ".join(t.content for t in self.turns)

    def reset(self) -> None:
        """Clear all turns but keep the system prompt and session ID."""
        self.turns.clear()
        self._turn_counter = 0
