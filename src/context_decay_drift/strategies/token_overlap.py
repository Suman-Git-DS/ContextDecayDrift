"""Token-overlap drift strategy using cosine similarity.

Compares the term-frequency vectors of the system prompt and recent
assistant responses. High cosine similarity means the assistant is
staying on-topic; low similarity means drift.
"""

from __future__ import annotations

from context_decay_drift.strategies.base import BaseStrategy
from context_decay_drift.utils.text import (
    cosine_similarity,
    term_frequency,
    tokenize,
)


class TokenOverlapStrategy(BaseStrategy):
    """Cosine similarity between system-prompt and response term-frequency vectors.

    Args:
        include_stopwords: Whether to include stop words in the comparison.
            Default False (removes stop words for cleaner signal).
    """

    def __init__(self, include_stopwords: bool = False):
        self.include_stopwords = include_stopwords

    @property
    def name(self) -> str:
        return "token_overlap"

    def score(
        self,
        system_prompt: str,
        assistant_responses: list[str],
    ) -> tuple[float, dict[str, float]]:
        if not assistant_responses:
            return 100.0, {self.name: 100.0}

        remove_sw = not self.include_stopwords
        prompt_tokens = tokenize(system_prompt, remove_stopwords=remove_sw)
        if not prompt_tokens:
            return 100.0, {self.name: 100.0}

        # Combine all response texts
        combined_response = " ".join(assistant_responses)
        response_tokens = tokenize(combined_response, remove_stopwords=remove_sw)
        if not response_tokens:
            return 0.0, {self.name: 0.0}

        # Build TF vectors and compute similarity
        prompt_tf = term_frequency(prompt_tokens)
        response_tf = term_frequency(response_tokens)

        similarity = cosine_similarity(prompt_tf, response_tf)

        # Scale to 0-100
        score = similarity * 100.0
        return score, {self.name: round(score, 2)}
