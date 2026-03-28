"""Tests for drift measurement strategies."""

import pytest
from context_decay_drift.strategies.keyword import KeywordStrategy
from context_decay_drift.strategies.token_overlap import TokenOverlapStrategy
from context_decay_drift.strategies.composite import CompositeStrategy


SYSTEM_PROMPT = (
    "You are a Python programming tutor. Help students learn Python concepts "
    "including variables, functions, loops, classes, and error handling. "
    "Always provide code examples and explain step by step."
)


class TestKeywordStrategy:
    def setup_method(self):
        self.strategy = KeywordStrategy(top_n=20)

    def test_perfect_alignment(self):
        """Response using all system prompt keywords should score high."""
        responses = [
            "Python variables store data. Functions organize code. "
            "Loops iterate over items. Classes define objects. "
            "Error handling uses try/except. Here is a code example "
            "that explains step by step how to learn Python programming concepts."
        ]
        score, scores = self.strategy.score(SYSTEM_PROMPT, responses)
        assert score > 50.0
        assert "keyword" in scores

    def test_off_topic_response(self):
        """Response about unrelated topic should score low."""
        responses = [
            "The weather today is sunny with temperatures around 72 degrees. "
            "Tomorrow there might be rain in the forecast."
        ]
        score, _ = self.strategy.score(SYSTEM_PROMPT, responses)
        assert score < 30.0

    def test_empty_responses(self):
        score, _ = self.strategy.score(SYSTEM_PROMPT, [])
        assert score == 100.0

    def test_gradual_drift(self):
        """Score should decrease as responses drift off-topic."""
        on_topic = [
            "Python functions are defined with def keyword. "
            "Classes use class keyword. Variables store values. "
            "Loops include for and while. Error handling uses try except."
        ]
        off_topic = [
            "The stock market went up today. "
            "Investors are excited about new tech companies."
        ]

        score_on, _ = self.strategy.score(SYSTEM_PROMPT, on_topic)
        score_off, _ = self.strategy.score(SYSTEM_PROMPT, off_topic)
        assert score_on > score_off

    def test_empty_system_prompt(self):
        score, _ = self.strategy.score("", ["Hello world"])
        assert score == 100.0


class TestTokenOverlapStrategy:
    def setup_method(self):
        self.strategy = TokenOverlapStrategy()

    def test_high_overlap(self):
        responses = [
            "Let me help you learn Python programming. We'll cover "
            "variables, functions, loops, classes, and error handling "
            "with code examples explained step by step."
        ]
        score, scores = self.strategy.score(SYSTEM_PROMPT, responses)
        assert score > 40.0
        assert "token_overlap" in scores

    def test_no_overlap(self):
        responses = [
            "Basketball scores were incredible last night. "
            "The championship game drew millions of viewers."
        ]
        score, _ = self.strategy.score(SYSTEM_PROMPT, responses)
        assert score < 30.0

    def test_empty_responses(self):
        score, _ = self.strategy.score(SYSTEM_PROMPT, [])
        assert score == 100.0

    def test_empty_prompt(self):
        score, _ = self.strategy.score("", ["Hello world"])
        assert score == 100.0


class TestCompositeStrategy:
    def test_equal_weights(self):
        kw = KeywordStrategy()
        to = TokenOverlapStrategy()
        comp = CompositeStrategy([kw, to])

        responses = [
            "Python programming with functions, variables, and loops. "
            "Code examples step by step for learning concepts."
        ]
        score, scores = comp.score(SYSTEM_PROMPT, responses)
        assert 0 <= score <= 100
        assert "keyword" in scores
        assert "token_overlap" in scores
        assert "composite" in scores

    def test_custom_weights(self):
        kw = KeywordStrategy()
        to = TokenOverlapStrategy()
        comp = CompositeStrategy([kw, to], weights=[0.7, 0.3])

        responses = ["Python functions and variables with code examples."]
        score, _ = comp.score(SYSTEM_PROMPT, responses)
        assert 0 <= score <= 100

    def test_empty_strategies_raises(self):
        with pytest.raises(ValueError, match="At least one strategy"):
            CompositeStrategy([])

    def test_mismatched_weights_raises(self):
        kw = KeywordStrategy()
        with pytest.raises(ValueError, match="same length"):
            CompositeStrategy([kw], weights=[0.5, 0.5])

    def test_zero_weights_raises(self):
        kw = KeywordStrategy()
        with pytest.raises(ValueError, match="positive"):
            CompositeStrategy([kw], weights=[0.0])

    def test_weights_normalized(self):
        kw = KeywordStrategy()
        to = TokenOverlapStrategy()
        comp = CompositeStrategy([kw, to], weights=[2.0, 8.0])
        assert abs(sum(comp.weights) - 1.0) < 1e-9
