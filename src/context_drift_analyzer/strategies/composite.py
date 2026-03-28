"""Composite strategy that combines multiple sub-strategies."""

from __future__ import annotations

from typing import Optional

from context_drift_analyzer.strategies.base import BaseStrategy


class CompositeStrategy(BaseStrategy):
    """Weighted combination of multiple drift strategies.

    Args:
        strategies: List of strategies to combine.
        weights: Optional weight for each strategy. If None, equal weights.
            Weights are normalized to sum to 1.0.
    """

    def __init__(
        self,
        strategies: list[BaseStrategy],
        weights: Optional[list[float]] = None,
    ):
        if not strategies:
            raise ValueError("At least one strategy is required")

        self.strategies = strategies

        if weights is None:
            self.weights = [1.0 / len(strategies)] * len(strategies)
        else:
            if len(weights) != len(strategies):
                raise ValueError("weights must have same length as strategies")
            total = sum(weights)
            if total <= 0:
                raise ValueError("Sum of weights must be positive")
            self.weights = [w / total for w in weights]

    @property
    def name(self) -> str:
        return "composite"

    def score(
        self,
        system_prompt: str,
        assistant_responses: list[str],
    ) -> tuple[float, dict[str, float]]:
        combined_score = 0.0
        all_scores: dict[str, float] = {}

        for strategy, weight in zip(self.strategies, self.weights):
            sub_score, sub_scores = strategy.score(system_prompt, assistant_responses)
            combined_score += sub_score * weight
            all_scores.update(sub_scores)

        all_scores[self.name] = round(combined_score, 2)
        return combined_score, all_scores
