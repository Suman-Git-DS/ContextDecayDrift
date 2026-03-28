"""Tests for text utility functions."""

import pytest
from context_drift_analyzer.utils.text import (
    tokenize,
    extract_keywords,
    ngrams,
    cosine_similarity,
    term_frequency,
)


class TestTokenize:
    def test_basic(self):
        tokens = tokenize("Hello World", remove_stopwords=False)
        assert tokens == ["hello", "world"]

    def test_removes_punctuation(self):
        tokens = tokenize("Hello, world! How are you?", remove_stopwords=False)
        assert "hello" in tokens
        assert "," not in tokens
        assert "!" not in tokens

    def test_removes_stopwords(self):
        tokens = tokenize("The quick brown fox jumps over the lazy dog")
        assert "the" not in tokens
        assert "over" not in tokens
        assert "quick" in tokens

    def test_keeps_stopwords_when_disabled(self):
        tokens = tokenize("the cat is here", remove_stopwords=False)
        assert "the" in tokens
        assert "is" in tokens

    def test_empty_string(self):
        assert tokenize("") == []


class TestExtractKeywords:
    def test_basic(self):
        text = "python python java python java go"
        keywords = extract_keywords(text)
        assert keywords[0] == "python"  # most frequent

    def test_top_n(self):
        text = "a b c d e f g h i j k"  # all stopwords removed that are stopwords
        keywords = extract_keywords(text, top_n=3)
        assert len(keywords) <= 3

    def test_empty_text(self):
        assert extract_keywords("") == []


class TestNgrams:
    def test_bigrams(self):
        tokens = ["a", "b", "c", "d"]
        result = ngrams(tokens, n=2)
        assert result == [("a", "b"), ("b", "c"), ("c", "d")]

    def test_trigrams(self):
        tokens = ["a", "b", "c", "d"]
        result = ngrams(tokens, n=3)
        assert result == [("a", "b", "c"), ("b", "c", "d")]

    def test_empty(self):
        assert ngrams([], n=2) == []

    def test_too_short(self):
        assert ngrams(["a"], n=2) == []


class TestCosineSimilarity:
    def test_identical(self):
        vec = {"a": 1.0, "b": 2.0}
        assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-9

    def test_orthogonal(self):
        vec_a = {"a": 1.0}
        vec_b = {"b": 1.0}
        assert cosine_similarity(vec_a, vec_b) == 0.0

    def test_partial_overlap(self):
        vec_a = {"a": 1.0, "b": 1.0}
        vec_b = {"b": 1.0, "c": 1.0}
        sim = cosine_similarity(vec_a, vec_b)
        assert 0.0 < sim < 1.0

    def test_empty_vectors(self):
        assert cosine_similarity({}, {"a": 1.0}) == 0.0
        assert cosine_similarity({}, {}) == 0.0


class TestTermFrequency:
    def test_basic(self):
        tokens = ["a", "a", "b"]
        tf = term_frequency(tokens)
        assert abs(tf["a"] - 2 / 3) < 1e-9
        assert abs(tf["b"] - 1 / 3) < 1e-9

    def test_empty(self):
        assert term_frequency([]) == {}

    def test_single_token(self):
        tf = term_frequency(["x"])
        assert tf == {"x": 1.0}
