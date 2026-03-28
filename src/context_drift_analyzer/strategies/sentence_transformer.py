"""Sentence Transformer strategy for semantic drift detection.

Uses HuggingFace sentence-transformers to embed text into dense vectors
and measure semantic similarity between initial context and conversation.

Install:
    pip install context-drift-analyzer[semantic]
    # or: pip install sentence-transformers

Models (pick based on your tradeoff):
    - "all-MiniLM-L6-v2"        : Fast, 80MB, good quality (default)
    - "all-mpnet-base-v2"       : Best quality, 420MB, slower
    - "paraphrase-MiniLM-L3-v2" : Fastest, 60MB, decent quality
"""

from __future__ import annotations

from typing import Optional

from context_drift_analyzer.strategies.embedding_base import EmbeddingStrategy


class SentenceTransformerStrategy(EmbeddingStrategy):
    """Semantic drift detection using sentence-transformers.

    Embeds the initial context and recent responses, then measures
    cosine similarity. A score of 100 means semantically identical;
    0 means completely unrelated.

    Args:
        model_name: HuggingFace model name. Default "all-MiniLM-L6-v2".
        device: Device to run on ("cpu", "cuda", "mps"). Default None (auto).
        cache_reference: Cache the reference embedding. Default True.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_reference: bool = True,
    ):
        super().__init__(cache_reference=cache_reference)
        self.model_name = model_name
        self._device = device
        self._model = None  # Lazy-loaded

    @property
    def name(self) -> str:
        return "sentence_transformer"

    @property
    def model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for SentenceTransformerStrategy. "
                    "Install it with: pip install context-drift-analyzer[semantic] "
                    "or: pip install sentence-transformers"
                )
            self._model = SentenceTransformer(
                self.model_name, device=self._device
            )
        return self._model

    def embed(self, text: str) -> list[float]:
        """Embed text using the sentence transformer model."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
