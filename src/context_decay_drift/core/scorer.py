"""Drift scoring data structures and verdict classification."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional


class DriftVerdict(enum.Enum):
    """Classification of drift severity."""

    FRESH = "fresh"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"

    @classmethod
    def from_score(cls, score: float) -> DriftVerdict:
        """Classify a 0-100 score into a drift verdict.

        Score ranges:
            90-100: FRESH   - Context is well-preserved
            75-89:  MILD    - Minor drift, still effective
            55-74:  MODERATE - Noticeable drift, monitor closely
            35-54:  SEVERE  - Significant drift, consider resetting
            0-34:   CRITICAL - Context largely lost, reset recommended
        """
        if score >= 90:
            return cls.FRESH
        elif score >= 75:
            return cls.MILD
        elif score >= 55:
            return cls.MODERATE
        elif score >= 35:
            return cls.SEVERE
        else:
            return cls.CRITICAL


@dataclass(frozen=True)
class DriftScore:
    """Immutable drift measurement result.

    Attributes:
        score: Context preservation score from 0 (fully drifted) to 100 (fresh).
        verdict: Categorical severity classification.
        turn_number: The conversation turn at which this was measured.
        session_id: Identifier for the session.
        strategy_scores: Per-strategy breakdown of scores.
        metadata: Additional strategy-specific data.
    """

    score: float
    verdict: DriftVerdict
    turn_number: int
    session_id: str
    strategy_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def is_effective(self) -> bool:
        """Whether the context is still considered effective (score >= 55)."""
        return self.score >= 55.0

    @property
    def needs_reset(self) -> bool:
        """Whether a context reset is recommended (score < 35)."""
        return self.score < 35.0

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "score": round(self.score, 2),
            "verdict": self.verdict.value,
            "is_effective": self.is_effective,
            "needs_reset": self.needs_reset,
            "turn_number": self.turn_number,
            "session_id": self.session_id,
            "strategy_scores": {
                k: round(v, 2) for k, v in self.strategy_scores.items()
            },
            "metadata": self.metadata,
        }
