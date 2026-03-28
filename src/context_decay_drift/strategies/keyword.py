"""Keyword-based drift strategy.

Measures what fraction of the system prompt's key terms still appear
in recent assistant responses. A dropping keyword hit rate signals
that the assistant is drifting away from its instructions.
"""

from __future__ import annotations

from context_decay_drift.strategies.base import BaseStrategy
from context_decay_drift.utils.text import extract_keywords, tokenize


class KeywordStrategy(BaseStrategy):
    """Tracks system-prompt keyword presence in assistant responses.

    Args:
        top_n: Number of top keywords to extract from the system prompt.
            0 means use all unique keywords. Default 30.
    """

    def __init__(self, top_n: int = 30):
        self.top_n = top_n

    @property
    def name(self) -> str:
        return "keyword"

    def score(
        self,
        system_prompt: str,
        assistant_responses: list[str],
    ) -> tuple[float, dict[str, float]]:
        if not assistant_responses:
            return 100.0, {self.name: 100.0}

        # Extract the reference keywords from the system prompt
        prompt_keywords = set(extract_keywords(system_prompt, top_n=self.top_n))
        if not prompt_keywords:
            return 100.0, {self.name: 100.0}

        # Combine all recent assistant responses into one token set
        response_tokens = set()
        for resp in assistant_responses:
            response_tokens.update(tokenize(resp, remove_stopwords=True))

        # Calculate hit rate
        hits = prompt_keywords & response_tokens
        hit_rate = len(hits) / len(prompt_keywords)

        # Scale to 0-100
        score = hit_rate * 100.0
        return score, {self.name: round(score, 2)}
