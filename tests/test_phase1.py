"""Tests for Phase 1 components: unified reward routing, info_value, defense_penalty, Judge parsing.

Run with: pytest tests/test_phase1.py -v
No GPU required. No API calls (Judge is tested via parse logic, not live API).
"""

import json
import pytest


# =========================================================================
# Info-value tests
# =========================================================================

class TestInfoValue:
    """info_value: 4*p*(1-p), maximised at p=0.5."""

    def test_perfect_split(self):
        from grubrics_science.rewards.alignment import compute_info_value

        # 3 above threshold, 3 below → p=0.5 → info=1.0
        scores = [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]
        assert abs(compute_info_value(scores) - 1.0) < 0.01

    def test_all_pass(self):
        from grubrics_science.rewards.alignment import compute_info_value

        # All above threshold → p=1.0 → info=0.0
        scores = [0.9, 0.8, 0.7, 0.6]
        assert compute_info_value(scores) == 0.0

    def test_all_fail(self):
        from grubrics_science.rewards.alignment import compute_info_value

        # All below threshold → p=0.0 → info=0.0
        scores = [0.1, 0.2, 0.3, 0.4]
        assert compute_info_value(scores) == 0.0

    def test_single_element(self):
        from grubrics_science.rewards.alignment import compute_info_value

        assert compute_info_value([0.5]) == 0.0

    def test_custom_threshold(self):
        from grubrics_science.rewards.alignment import compute_info_value

        # With threshold=0.3, scores [0.1, 0.5] → p=0.5 → info=1.0
        assert abs(compute_info_value([0.1, 0.5], threshold=0.3) - 1.0) < 0.01


# =========================================================================
# Defense penalty tests
# =========================================================================

class TestDefensePenalty:
    """defense_penalty: detects degenerate rubrics (all same scores)."""

    def test_identical_scores_max_penalty(self):
        from grubrics_science.rewards.alignment import compute_defense_penalty

        pen = compute_defense_penalty([0.5, 0.5, 0.5, 0.5])
        assert pen == 1.0

    def test_high_variance_no_penalty(self):
        from grubrics_science.rewards.alignment import compute_defense_penalty

        pen = compute_defense_penalty([0.0, 0.5, 1.0])
        assert pen == 0.0

    def test_low_variance_partial_penalty(self):
        from grubrics_science.rewards.alignment import compute_defense_penalty

        # Small variance → some penalty
        pen = compute_defense_penalty([0.5, 0.52, 0.48, 0.51])
        assert pen > 0.5

    def test_single_element(self):
        from grubrics_science.rewards.alignment import compute_defense_penalty

        assert compute_defense_penalty([0.5]) == 1.0


# =========================================================================
# Unified reward routing tests
# =========================================================================

class TestGrubricsRewardRouting:
    """grubrics_reward.compute_score routes based on data_source."""

    def test_verifiable_uses_local_reward(self):
        from grubrics_science.rewards.grubrics_reward import compute_score

        good_rubric = (
            "Points: 3.0, Item: The answer correctly computes the result\n"
            "Points: 3.0, Item: The solution shows step-by-step reasoning\n"
            "Points: 2.0, Item: The answer identifies the correct operation\n"
            "Points: 2.0, Item: The final answer is clearly stated"
        )

        # All verifiable sources should use local reward (no API)
        for source in ["gsm8k", "math", "olympiad_math"]:
            score = compute_score(
                data_source=source,
                solution_str=good_rubric,
                extra_info={"question": "What is 2+2?"},
            )
            assert isinstance(score, float)
            assert score > 0.5, f"{source}: good rubric should score > 0.5, got {score}"

    def test_bad_rubric_low_score_for_verifiable(self):
        from grubrics_science.rewards.grubrics_reward import compute_score

        score = compute_score(
            data_source="gsm8k",
            solution_str="this is not a rubric",
            extra_info={"question": "What is 2+2?"},
        )
        assert score < 0.2

    def test_open_domain_without_cache_falls_back(self):
        """FrontierScience without precomputed data falls back to local reward."""
        from grubrics_science.rewards.grubrics_reward import compute_score

        good_rubric = (
            "Points: 5.0, Item: The answer derives the equation correctly\n"
            "Points: 5.0, Item: The explanation is physically sound"
        )

        # No answers/gold_scores → should fallback, not crash
        score = compute_score(
            data_source="frontierscience",
            solution_str=good_rubric,
            extra_info={
                "question": "Derive E=mc^2",
                "answers": [],
                "gold_scores": [],
            },
        )
        assert isinstance(score, float)
        assert score >= 0.0


# =========================================================================
# Compute_reward with new components
# =========================================================================

class TestComputeRewardExtended:
    """compute_reward with info_value and defense_penalty."""

    def test_info_value_bonus_increases_reward(self):
        from grubrics_science.rewards.alignment import compute_reward

        # Same alignment, but one has info_value bonus
        base = compute_reward(
            scores=[1, 2, 3],
            gold_scores=[10, 20, 30],
            rubric_text="short",
            lambda_info=0.0,
        )
        with_info = compute_reward(
            scores=[1, 2, 3],
            gold_scores=[10, 20, 30],
            rubric_text="short",
            lambda_info=0.5,
        )
        # info_value for [1,2,3] with threshold 0.5: all below → 0.0
        # So no difference here. Use scores that trigger info bonus:
        base2 = compute_reward(
            scores=[0.3, 0.7],
            gold_scores=[0.3, 0.7],
            rubric_text="short",
            lambda_info=0.0,
        )
        with_info2 = compute_reward(
            scores=[0.3, 0.7],
            gold_scores=[0.3, 0.7],
            rubric_text="short",
            lambda_info=0.5,
        )
        # p=0.5 → info=1.0, bonus=0.5*1.0=0.5
        assert with_info2 > base2

    def test_defense_penalty_reduces_reward(self):
        from grubrics_science.rewards.alignment import compute_reward

        # Degenerate scores (all same) → high defense penalty
        base = compute_reward(
            scores=[0.5, 0.5, 0.5],
            gold_scores=[1, 2, 3],
            rubric_text="short",
            lambda_defense=0.0,
        )
        with_defense = compute_reward(
            scores=[0.5, 0.5, 0.5],
            gold_scores=[1, 2, 3],
            rubric_text="short",
            lambda_defense=0.5,
        )
        assert with_defense < base


# =========================================================================
# Judge response parsing tests (no API calls)
# =========================================================================

class TestJudgeParsing:
    """Test Judge._parse_response and extract_json_from_response."""

    def test_extract_json_from_code_block(self):
        from grubrics_science.judge.judge import extract_json_from_response

        response = '```json\n{"rubric_evaluations": [{"rubric_id": "r1", "total_score": 0.8}]}\n```'
        result = extract_json_from_response(response)
        assert result is not None
        assert "rubric_evaluations" in result

    def test_extract_json_direct(self):
        from grubrics_science.judge.judge import extract_json_from_response

        response = '{"rubric_evaluations": [{"rubric_id": "r1", "total_score": 0.7}]}'
        result = extract_json_from_response(response)
        assert result is not None
        assert result["rubric_evaluations"][0]["total_score"] == 0.7

    def test_extract_json_garbage_returns_none(self):
        from grubrics_science.judge.judge import extract_json_from_response

        assert extract_json_from_response("this is not json at all") is None

    def test_cache_key_deterministic(self):
        from grubrics_science.judge.judge import _cache_key

        k1 = _cache_key("q1", "a1", ["r1"])
        k2 = _cache_key("q1", "a1", ["r1"])
        k3 = _cache_key("q1", "a1", ["r2"])

        assert k1 == k2  # same inputs → same key
        assert k1 != k3  # different rubric → different key

    def test_cache_key_order_matters(self):
        from grubrics_science.judge.judge import _cache_key

        k1 = _cache_key("q1", "a1", ["r1", "r2"])
        k2 = _cache_key("q1", "a1", ["r2", "r1"])

        assert k1 != k2  # rubric order matters
