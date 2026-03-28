"""Drift explainer — generates 1-2 line human-readable explanations of why drift occurred.

Supports two modes:
1. Local (default): Score-based explanation with topic analysis — free, no API calls.
2. Custom callable: Provide your own explanation function (e.g., LLM-powered).
"""

from __future__ import annotations

from typing import Callable, Optional

from context_drift_analyzer.utils.text import extract_keywords, tokenize


class DriftExplainer:
    """Generates human-readable explanations for detected drift.

    Args:
        explain_fn: Optional custom explanation function.
            Signature: (original_context: str, recent_text: str, score: float) -> str
            If not provided, uses a local rule-based explainer (no API calls).
    """

    def __init__(
        self,
        explain_fn: Optional[Callable[[str, str, float], str]] = None,
    ):
        self._explain_fn = explain_fn or self._local_explain

    def explain(
        self,
        original_context: str,
        recent_text: str,
        score: float,
    ) -> str:
        """Generate a 1-2 line explanation of why drift occurred.

        Args:
            original_context: The original system prompt + few-shot context.
            recent_text: The recent assistant response text.
            score: The drift score (0-100).

        Returns:
            A short explanation string.
        """
        return self._explain_fn(original_context, recent_text, score)

    @staticmethod
    def _local_explain(
        original_context: str,
        recent_text: str,
        score: float,
    ) -> str:
        """Score-based explanation with topic analysis — no API calls needed."""
        if not recent_text.strip():
            return "No response to evaluate."

        # Extract meaningful topics from both texts
        original_topics = set(extract_keywords(original_context, top_n=15))
        response_topics = set(extract_keywords(recent_text, top_n=10))

        # Identify shared and divergent topics
        shared = original_topics & response_topics
        new_topics = response_topics - original_topics

        if score >= 90:
            return "Context is well-preserved. Responses closely align with original instructions."

        if score >= 75:
            if shared:
                return f"Mild drift detected, but core topics still present ({', '.join(sorted(shared)[:3])})."
            return "Mild drift: response is related but uses different terminology than the original context."

        if score >= 55:
            if shared and new_topics:
                return (
                    f"Moderate drift: some original topics present ({', '.join(sorted(shared)[:3])}), "
                    f"but new topics emerging ({', '.join(sorted(new_topics)[:3])})."
                )
            if new_topics:
                return f"Moderate drift: conversation shifting toward new topics ({', '.join(sorted(new_topics)[:4])})."
            return "Moderate drift: response is loosely related to original context."

        if score >= 35:
            if new_topics:
                return (
                    f"Significant drift: conversation has moved away from original purpose. "
                    f"Now focused on: {', '.join(sorted(new_topics)[:4])}."
                )
            return "Significant drift: response shows weak alignment with original context. Consider resetting."

        # Critical
        if new_topics:
            return (
                f"Critical drift: conversation has largely departed from its original purpose. "
                f"Current topics ({', '.join(sorted(new_topics)[:4])}) are unrelated to original context."
            )
        return "Critical drift: response has very low alignment with original context. Reset recommended."
