"""Base class for embedding-based semantic drift strategies.

All embedding strategies share the same scoring logic:
1. Embed the initial context (system prompt + few-shots) → reference vector
2. Embed the recent assistant responses → current vector
3. Cosine similarity between them → drift score (0-100)

Subclasses only need to implement `embed()` for their specific backend.

IMPORTANT: Cosine similarity between instruction-style text (system prompts)
and response-style text (assistant replies) typically ranges from 0.1 to 0.5,
not 0 to 1. A raw cosine of 0.3 between a system prompt and an on-topic
response is actually good alignment. We use calibrated scaling to map the
realistic range [0, 0.6] → [0, 100] so drift scores are meaningful.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import Optional

from context_drift_analyzer.strategies.base import BaseStrategy

# Calibration: in practice, cosine similarity between a system prompt
# (instruction text) and an on-topic response rarely exceeds 0.55-0.60.
# We scale the [0, MAX_EXPECTED_SIM] range to [0, 100].
MAX_EXPECTED_SIMILARITY = 0.55


class EmbeddingStrategy(BaseStrategy):
    """Abstract base for embedding-powered drift detection.

    Subclasses implement `embed(text) -> list[float]` using any backend:
    sentence-transformers, OpenAI API, Cohere, Voyage, or any custom model.

    The scoring logic is shared: cosine similarity between the reference
    embedding (initial context) and the current conversation embedding,
    with calibrated scaling so scores are meaningful.

    Args:
        cache_reference: If True, caches the reference embedding after first
            computation so it's not re-embedded on every call. Default True.
        max_similarity: Expected maximum cosine similarity for perfect
            alignment. Used to scale scores. Default 0.55.
    """

    def __init__(
        self,
        cache_reference: bool = True,
        max_similarity: float = MAX_EXPECTED_SIMILARITY,
    ):
        self._cache_reference = cache_reference
        self._max_similarity = max_similarity
        self._ref_cache: dict[str, list[float]] = {}

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Convert text to an embedding vector.

        Args:
            text: Input text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        ...

    def score(
        self,
        system_prompt: str,
        assistant_responses: list[str],
    ) -> tuple[float, dict[str, float]]:
        if not assistant_responses:
            return 100.0, {self.name: 100.0}

        # Get reference embedding (with optional caching)
        ref_embedding = self._get_reference_embedding(system_prompt)

        # Embed the current conversation state (all recent responses combined)
        current_text = " ".join(assistant_responses)
        current_embedding = self.embed(current_text)

        # Compute cosine similarity
        raw_similarity = self._cosine_similarity(ref_embedding, current_embedding)

        # Calibrated scaling: map [0, max_similarity] → [0, 100]
        # This ensures that realistic on-topic responses score high (70-95)
        # rather than low (20-40) due to the instruction-vs-response gap.
        scaled = (raw_similarity / self._max_similarity) * 100.0
        score = max(0.0, min(100.0, scaled))

        return score, {self.name: round(score, 2)}

    def _get_reference_embedding(self, text: str) -> list[float]:
        """Get embedding for reference text, using cache if enabled."""
        if self._cache_reference:
            cache_key = str(hash(text))
            if cache_key not in self._ref_cache:
                self._ref_cache[cache_key] = self.embed(text)
            return self._ref_cache[cache_key]
        return self.embed(text)

    def clear_cache(self) -> None:
        """Clear the reference embedding cache."""
        self._ref_cache.clear()

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two dense vectors."""
        if len(vec_a) != len(vec_b):
            raise ValueError(
                f"Vector dimension mismatch: {len(vec_a)} vs {len(vec_b)}"
            )

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)
