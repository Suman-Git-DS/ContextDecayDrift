"""Callable embedding strategy — bring your own embedding function.

Use this when you have a custom embedding pipeline (Cohere, Voyage,
Google, a local model, or anything else). Just pass a function that
takes a string and returns a list of floats.

Usage:
    from context_drift_analyzer.strategies.callable_embedding import CallableEmbeddingStrategy

    # With Cohere
    import cohere
    co = cohere.Client("your-api-key")

    def cohere_embed(text: str) -> list[float]:
        response = co.embed(texts=[text], model="embed-english-v3.0")
        return response.embeddings[0]

    strategy = CallableEmbeddingStrategy(embed_fn=cohere_embed, name="cohere")

    # With Google Vertex AI
    def vertex_embed(text: str) -> list[float]:
        # your vertex AI embedding logic
        ...

    strategy = CallableEmbeddingStrategy(embed_fn=vertex_embed, name="vertex_ai")
"""

from __future__ import annotations

from typing import Callable

from context_drift_analyzer.strategies.embedding_base import EmbeddingStrategy


class CallableEmbeddingStrategy(EmbeddingStrategy):
    """Bring-your-own embedding function for drift detection.

    Wraps any `(str) -> list[float]` function into a drift strategy.
    Use this to plug in Cohere, Voyage, Google, HuggingFace Inference API,
    or any custom embedding backend.

    Args:
        embed_fn: A callable that takes a string and returns a list of floats.
        strategy_name: Name for this strategy in score reports. Default "custom_embedding".
        cache_reference: Cache the reference embedding. Default True.
    """

    def __init__(
        self,
        embed_fn: Callable[[str], list[float]],
        strategy_name: str = "custom_embedding",
        cache_reference: bool = True,
    ):
        super().__init__(cache_reference=cache_reference)
        self._embed_fn = embed_fn
        self._strategy_name = strategy_name

    @property
    def name(self) -> str:
        return self._strategy_name

    def embed(self, text: str) -> list[float]:
        """Embed text using the provided callable."""
        return self._embed_fn(text)
