"""Base provider wrapper with drift tracking."""

from __future__ import annotations

from typing import Any, Optional

from context_decay_drift.core.analyzer import DriftAnalyzer
from context_decay_drift.core.scorer import DriftScore
from context_decay_drift.core.session import Session
from context_decay_drift.strategies.base import BaseStrategy


class DriftAwareResponse:
    """Wraps an LLM response with drift information.

    Attributes:
        response: The original LLM response object (provider-specific).
        content: The extracted text content from the response.
        drift: The DriftScore computed after this response.
    """

    def __init__(self, response: Any, content: str, drift: DriftScore):
        self.response = response
        self.content = content
        self.drift = drift

    @property
    def drift_score(self) -> float:
        """Shortcut to the numeric drift score (0-100)."""
        return self.drift.score

    @property
    def drift_verdict(self) -> str:
        """Shortcut to the drift verdict string."""
        return self.drift.verdict.value

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "content": self.content,
            "drift": self.drift.to_dict(),
        }

    def __repr__(self) -> str:
        return (
            f"DriftAwareResponse(drift_score={self.drift_score:.1f}, "
            f"verdict={self.drift_verdict!r})"
        )


class BaseProvider:
    """Base class for LLM provider wrappers with drift tracking.

    Subclasses implement `_call_llm()` for their specific provider SDK.

    Args:
        system_prompt: The system prompt to track drift against.
        analyzer: Optional pre-configured DriftAnalyzer.
        strategies: Optional list of strategies (used if analyzer is None).
        session: Optional pre-existing session to continue.
        decay_rate: Exponential decay rate (passed to DriftAnalyzer).
        window_size: Turn window size (passed to DriftAnalyzer).
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
        self.system_prompt = system_prompt
        self.session = session or Session(system_prompt=system_prompt)
        self.analyzer = analyzer or DriftAnalyzer(
            strategies=strategies,
            decay_rate=decay_rate,
            window_size=window_size,
        )

    def get_drift(self) -> DriftScore:
        """Get current drift score without making an LLM call."""
        return self.analyzer.analyze(self.session)

    def reset_session(self) -> None:
        """Reset the session while keeping the system prompt."""
        self.session.reset()
