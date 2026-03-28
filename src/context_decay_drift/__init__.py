"""
context-decay-drift: Measure and monitor context drift in LLM conversations.

Provides drift scoring (0-100) that tracks how far an LLM conversation
has drifted from its original system prompt across sessions and turns.
"""

from context_decay_drift.core.session import Session
from context_decay_drift.core.analyzer import DriftAnalyzer
from context_decay_drift.core.scorer import DriftScore, DriftVerdict
from context_decay_drift.strategies.keyword import KeywordStrategy
from context_decay_drift.strategies.token_overlap import TokenOverlapStrategy
from context_decay_drift.strategies.composite import CompositeStrategy

__version__ = "0.1.0"

__all__ = [
    "Session",
    "DriftAnalyzer",
    "DriftScore",
    "DriftVerdict",
    "KeywordStrategy",
    "TokenOverlapStrategy",
    "CompositeStrategy",
    "__version__",
]
