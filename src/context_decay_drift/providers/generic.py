"""Generic drift wrapper for any LLM client.

Use this when you have a custom LLM integration that doesn't fit
the OpenAI or Anthropic patterns. You manually feed user messages
and assistant responses, and the wrapper tracks drift.

Usage:
    from context_decay_drift.providers.generic import GenericDriftTracker

    tracker = GenericDriftTracker(
        system_prompt="You are a helpful assistant..."
    )

    # After each LLM call in your pipeline:
    tracker.record_turn(
        user_message="What is Python?",
        assistant_response="Python is a programming language..."
    )

    drift = tracker.get_drift()
    print(f"Drift score: {drift.score:.1f}")
"""

from __future__ import annotations

from typing import Optional

from context_decay_drift.core.analyzer import DriftAnalyzer
from context_decay_drift.core.scorer import DriftScore
from context_decay_drift.core.session import Session
from context_decay_drift.providers.base import BaseProvider
from context_decay_drift.strategies.base import BaseStrategy


class GenericDriftTracker(BaseProvider):
    """Provider-agnostic drift tracker.

    Use this to add drift tracking to any LLM pipeline by manually
    recording turns.

    Args:
        system_prompt: The system prompt to track drift against.
        analyzer: Optional pre-configured DriftAnalyzer.
        strategies: Optional list of strategies.
        session: Optional pre-existing session.
        decay_rate: Exponential decay rate.
        window_size: Turn window size.
    """

    def __init__(
        self,
        system_prompt: str,
        analyzer: Optional[DriftAnalyzer] = None,
        strategies: Optional[list[BaseStrategy]] = None,
        session: Optional[Session] = None,
        decay_rate: float = 0.95,
        window_size: int = 5,
    ):
        super().__init__(
            system_prompt=system_prompt,
            analyzer=analyzer,
            strategies=strategies,
            session=session,
            decay_rate=decay_rate,
            window_size=window_size,
        )

    def record_turn(
        self,
        user_message: str,
        assistant_response: str,
    ) -> DriftScore:
        """Record a conversation turn and get the updated drift score.

        Args:
            user_message: What the user said.
            assistant_response: What the assistant replied.

        Returns:
            Updated DriftScore after this turn.
        """
        self.session.add_user_message(user_message)
        self.session.add_assistant_message(assistant_response)
        return self.analyzer.analyze(self.session)

    def record_assistant_response(self, response: str) -> DriftScore:
        """Record only an assistant response (if user message already tracked)."""
        self.session.add_assistant_message(response)
        return self.analyzer.analyze(self.session)

    def record_user_message(self, message: str) -> None:
        """Record only a user message (drift not recalculated until assistant responds)."""
        self.session.add_user_message(message)
