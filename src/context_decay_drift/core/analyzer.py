"""Central drift analyzer that orchestrates strategies and produces scores."""

from __future__ import annotations

from typing import Optional

from context_decay_drift.core.scorer import DriftScore, DriftVerdict
from context_decay_drift.core.session import Session
from context_decay_drift.strategies.base import BaseStrategy
from context_decay_drift.strategies.composite import CompositeStrategy
from context_decay_drift.strategies.keyword import KeywordStrategy
from context_decay_drift.strategies.token_overlap import TokenOverlapStrategy


class DriftAnalyzer:
    """Analyzes context drift in LLM conversations.

    The analyzer accepts one or more strategies that each produce a sub-score.
    Scores are combined (weighted average) into a single 0-100 drift score.

    Args:
        strategies: List of strategies to use. If None, uses default set.
        decay_rate: Exponential decay factor applied per turn (0-1).
            Lower values mean faster decay. Default 0.95.
        window_size: Number of recent assistant turns to consider
            for drift calculation. Default 5 (0 = use all turns).
    """

    def __init__(
        self,
        strategies: Optional[list[BaseStrategy]] = None,
        decay_rate: float = 0.95,
        window_size: int = 5,
    ):
        if decay_rate <= 0 or decay_rate > 1:
            raise ValueError("decay_rate must be in (0, 1]")
        if window_size < 0:
            raise ValueError("window_size must be >= 0")

        self.decay_rate = decay_rate
        self.window_size = window_size

        if strategies is not None:
            self.strategy = (
                strategies[0]
                if len(strategies) == 1
                else CompositeStrategy(strategies)
            )
        else:
            self.strategy = CompositeStrategy(
                [
                    KeywordStrategy(),
                    TokenOverlapStrategy(),
                ]
            )

    def analyze(self, session: Session) -> DriftScore:
        """Compute drift score for the current state of a session.

        Returns a DriftScore with the combined score and per-strategy breakdown.
        """
        if not session.assistant_turns:
            return DriftScore(
                score=100.0,
                verdict=DriftVerdict.FRESH,
                turn_number=0,
                session_id=session.session_id,
                strategy_scores={},
                metadata={"reason": "no_assistant_turns"},
            )

        # Select the window of assistant turns to evaluate
        assistant_turns = session.assistant_turns
        if self.window_size > 0:
            assistant_turns = assistant_turns[-self.window_size :]

        # Gather the text from the windowed assistant turns
        recent_texts = [t.content for t in assistant_turns]

        # Calculate raw strategy score
        raw_score, strategy_scores = self.strategy.score(
            system_prompt=session.system_prompt,
            assistant_responses=recent_texts,
        )

        # Apply exponential decay based on total turn count
        total_turns = session.turn_count
        decay_factor = self.decay_rate ** (total_turns / 2)
        decayed_score = raw_score * decay_factor

        # Clamp to [0, 100]
        final_score = max(0.0, min(100.0, decayed_score))
        verdict = DriftVerdict.from_score(final_score)

        return DriftScore(
            score=final_score,
            verdict=verdict,
            turn_number=session.turn_count,
            session_id=session.session_id,
            strategy_scores=strategy_scores,
            metadata={
                "raw_score": round(raw_score, 2),
                "decay_factor": round(decay_factor, 4),
                "decay_rate": self.decay_rate,
                "window_size": self.window_size,
                "turns_evaluated": len(recent_texts),
            },
        )

    def is_effective(self, session: Session) -> bool:
        """Quick check: is the session context still effective?"""
        return self.analyze(session).is_effective

    def needs_reset(self, session: Session) -> bool:
        """Quick check: should the session context be reset?"""
        return self.analyze(session).needs_reset
