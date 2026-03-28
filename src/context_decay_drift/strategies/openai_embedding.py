"""OpenAI Embedding API strategy for semantic drift detection.

Uses OpenAI's embedding models to measure drift. Requires an OpenAI API key.

Install:
    pip install context-decay-drift[openai]

Models:
    - "text-embedding-3-small" : Cheap, fast, 1536 dims (default)
    - "text-embedding-3-large" : Better quality, 3072 dims
    - "text-embedding-ada-002" : Legacy, 1536 dims
"""

from __future__ import annotations

from typing import Any, Optional

from context_decay_drift.strategies.embedding_base import EmbeddingStrategy


class OpenAIEmbeddingStrategy(EmbeddingStrategy):
    """Semantic drift detection using OpenAI's embedding API.

    Args:
        client: An initialized OpenAI client instance.
        model: Embedding model name. Default "text-embedding-3-small".
        cache_reference: Cache the reference embedding. Default True.
    """

    def __init__(
        self,
        client: Any,
        model: str = "text-embedding-3-small",
        cache_reference: bool = True,
    ):
        super().__init__(cache_reference=cache_reference)
        self.client = client
        self.model = model

    @property
    def name(self) -> str:
        return "openai_embedding"

    def embed(self, text: str) -> list[float]:
        """Embed text using OpenAI's embedding API."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model,
        )
        return response.data[0].embedding
