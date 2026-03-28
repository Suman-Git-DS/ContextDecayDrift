"""Base strategy interface for drift measurement."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """Abstract base class for drift measurement strategies.

    Each strategy implements a specific way to measure how far the
    conversation has drifted from the original system prompt.

    Subclasses must implement `score()` which returns:
        - A float score in [0, 100] (100 = no drift)
        - A dict mapping strategy name to its score (for composite tracking)
    """

    @property
    def name(self) -> str:
        """Human-readable name for this strategy."""
        return self.__class__.__name__

    @abstractmethod
    def score(
        self,
        system_prompt: str,
        assistant_responses: list[str],
    ) -> tuple[float, dict[str, float]]:
        """Calculate drift score.

        Args:
            system_prompt: The original system prompt text.
            assistant_responses: Recent assistant response texts.

        Returns:
            Tuple of (combined_score, {strategy_name: score}).
            Scores are in [0, 100] where 100 means no drift.
        """
        ...
