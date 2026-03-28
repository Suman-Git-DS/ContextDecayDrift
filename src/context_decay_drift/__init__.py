"""
context-decay-drift: Measure and monitor context drift in LLM conversations.

Provides drift scoring (0-100) that tracks how far an LLM conversation
has drifted from its original system prompt and few-shot context across
sessions and turns. Uses semantic embeddings for accurate drift detection.
"""

from context_decay_drift.core.session import Session, FewShotExample
from context_decay_drift.core.analyzer import DriftAnalyzer
from context_decay_drift.core.scorer import DriftScore, DriftVerdict
from context_decay_drift.strategies.keyword import KeywordStrategy
from context_decay_drift.strategies.token_overlap import TokenOverlapStrategy
from context_decay_drift.strategies.composite import CompositeStrategy
from context_decay_drift.strategies.embedding_base import EmbeddingStrategy
from context_decay_drift.strategies.callable_embedding import CallableEmbeddingStrategy

__version__ = "0.2.0"

__all__ = [
    "Session",
    "FewShotExample",
    "DriftAnalyzer",
    "DriftScore",
    "DriftVerdict",
    "KeywordStrategy",
    "TokenOverlapStrategy",
    "CompositeStrategy",
    "EmbeddingStrategy",
    "CallableEmbeddingStrategy",
    "__version__",
]


# Lazy imports for optional dependencies
def __getattr__(name: str):
    if name == "SentenceTransformerStrategy":
        from context_decay_drift.strategies.sentence_transformer import (
            SentenceTransformerStrategy,
        )
        return SentenceTransformerStrategy
    if name == "OpenAIEmbeddingStrategy":
        from context_decay_drift.strategies.openai_embedding import (
            OpenAIEmbeddingStrategy,
        )
        return OpenAIEmbeddingStrategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
