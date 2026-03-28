"""Tests for drift scoring and verdicts."""

import pytest
from context_decay_drift.core.scorer import DriftScore, DriftVerdict


class TestDriftVerdict:
    def test_fresh(self):
        assert DriftVerdict.from_score(100) == DriftVerdict.FRESH
        assert DriftVerdict.from_score(90) == DriftVerdict.FRESH
        assert DriftVerdict.from_score(95.5) == DriftVerdict.FRESH

    def test_mild(self):
        assert DriftVerdict.from_score(89) == DriftVerdict.MILD
        assert DriftVerdict.from_score(75) == DriftVerdict.MILD

    def test_moderate(self):
        assert DriftVerdict.from_score(74) == DriftVerdict.MODERATE
        assert DriftVerdict.from_score(55) == DriftVerdict.MODERATE

    def test_severe(self):
        assert DriftVerdict.from_score(54) == DriftVerdict.SEVERE
        assert DriftVerdict.from_score(35) == DriftVerdict.SEVERE

    def test_critical(self):
        assert DriftVerdict.from_score(34) == DriftVerdict.CRITICAL
        assert DriftVerdict.from_score(0) == DriftVerdict.CRITICAL


class TestDriftScore:
    def _make_score(self, score: float = 80.0) -> DriftScore:
        return DriftScore(
            score=score,
            verdict=DriftVerdict.from_score(score),
            turn_number=5,
            session_id="test-123",
            strategy_scores={"keyword": 75.0, "token_overlap": 85.0},
        )

    def test_is_effective_true(self):
        ds = self._make_score(80.0)
        assert ds.is_effective is True

    def test_is_effective_false(self):
        ds = self._make_score(40.0)
        assert ds.is_effective is False

    def test_needs_reset_true(self):
        ds = self._make_score(20.0)
        assert ds.needs_reset is True

    def test_needs_reset_false(self):
        ds = self._make_score(50.0)
        assert ds.needs_reset is False

    def test_to_dict(self):
        ds = self._make_score(72.5)
        d = ds.to_dict()
        assert d["score"] == 72.5
        assert d["verdict"] == "moderate"
        assert d["is_effective"] is True
        assert d["needs_reset"] is False
        assert d["turn_number"] == 5
        assert d["session_id"] == "test-123"
        assert "keyword" in d["strategy_scores"]

    def test_frozen(self):
        ds = self._make_score(80.0)
        with pytest.raises(AttributeError):
            ds.score = 50.0  # type: ignore
