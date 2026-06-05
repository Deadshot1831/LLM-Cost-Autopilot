import pytest

from autopilot.quality import (
    EXACT_MATCH_THRESHOLD,
    JUDGE_THRESHOLD,
    SEMANTIC_THRESHOLD,
    QualityVerdict,
    VerdictResult,
    exact_match_score,
    is_short_prompt,
    score_exact_match,
    score_semantic,
)


class TestScoreSemantic:
    def test_high_score_passes(self):
        r = score_semantic(0.91)
        assert r.verdict == QualityVerdict.PASS
        assert r.method == "semantic"
        assert "cosine=0.91" in r.detail

    def test_low_score_fails(self):
        r = score_semantic(0.20)
        assert r.verdict == QualityVerdict.FAIL
        assert r.method == "semantic"

    def test_threshold_boundary_passes(self):
        # >= SEMANTIC_THRESHOLD is PASS
        assert score_semantic(SEMANTIC_THRESHOLD).verdict == QualityVerdict.PASS
        # just under is FAIL
        assert score_semantic(SEMANTIC_THRESHOLD - 0.01).verdict == QualityVerdict.FAIL


class TestExactMatchScore:
    def test_identical_strings_perfect(self):
        assert exact_match_score("hello world", "hello world") == 1.0

    def test_partial_overlap(self):
        s = exact_match_score("the cat sat", "the dog sat")
        assert 0.0 < s < 1.0

    def test_zero_overlap(self):
        assert exact_match_score("apple banana", "xyz qrs") == 0.0

    def test_case_insensitive(self):
        assert exact_match_score("Hello", "hello") == 1.0

    def test_empty_reference_returns_zero(self):
        assert exact_match_score("anything", "") == 0.0

    def test_empty_candidate_returns_zero(self):
        assert exact_match_score("", "anything") == 0.0


class TestScoreExactMatch:
    def test_pass_when_above_threshold(self):
        result = score_exact_match("hello world", "hello world")
        assert result.verdict == QualityVerdict.PASS
        assert result.method == "exact_match"

    def test_fail_when_below_threshold(self):
        result = score_exact_match("apple banana", "xyz qrs")
        assert result.verdict == QualityVerdict.FAIL

    def test_skip_when_reference_empty(self):
        result = score_exact_match("anything", "")
        assert result.verdict == QualityVerdict.SKIP


class TestIsShortPrompt:
    def test_short_prompt(self):
        assert is_short_prompt("What is 2 + 2?") is True

    def test_long_prompt(self):
        assert is_short_prompt("word " * 50) is False


class TestThresholds:
    def test_thresholds_documented(self):
        assert 0.0 < EXACT_MATCH_THRESHOLD < 1.0
        assert 1.0 <= JUDGE_THRESHOLD <= 5.0


class TestVerdictResult:
    def test_construction(self):
        r = VerdictResult(QualityVerdict.PASS, 0.9, "exact_match", "ok")
        assert r.verdict == QualityVerdict.PASS
        assert r.score == 0.9
