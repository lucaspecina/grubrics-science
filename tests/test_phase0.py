"""Tests for Phase 0 components: data adapters, reward function, alignment metrics.

Run with: pytest tests/test_phase0.py -v
No GPU required. No API calls.
"""

import json
import tempfile
from pathlib import Path

import pytest


# =========================================================================
# Fixtures: reusable test data
# =========================================================================

GOOD_RUBRIC = (
    "Points: 3.0, Item: The answer correctly computes the final numerical result\n"
    "Points: 3.0, Item: The solution shows clear step-by-step arithmetic reasoning\n"
    "Points: 2.0, Item: The answer identifies and uses the correct mathematical operation\n"
    "Points: 2.0, Item: The final answer is clearly stated with proper units"
)

BAD_RUBRIC_NO_FORMAT = "This is just random text with no rubric format at all."

BAD_RUBRIC_WRONG_TOTAL = (
    "Points: 5.0, Item: The answer is correct\n"
    "Points: 5.0, Item: The reasoning is clear\n"
    "Points: 5.0, Item: Extra item that breaks the total"
)

BAD_RUBRIC_TRIVIAL_ITEMS = (
    "Points: 5.0, Item: ok\n"
    "Points: 5.0, Item: yes"
)

MATH_QUESTION = "If Juan has 5 apples and buys 3 more, how many does he have in total?"


# =========================================================================
# Data adapter tests
# =========================================================================

class TestAdapterSchema:
    """Each adapter's to_verl_format must produce the correct schema."""

    REQUIRED_KEYS = {"data_source", "prompt", "reward_model", "extra_info"}

    def _check_row(self, row, expected_source, expected_domain):
        """Common assertions for any adapter row."""
        # All required keys present
        assert self.REQUIRED_KEYS.issubset(row.keys()), (
            f"Missing keys: {self.REQUIRED_KEYS - row.keys()}"
        )
        # Correct data_source
        assert row["data_source"] == expected_source

        # prompt is a list of chat messages (before serialization)
        assert isinstance(row["prompt"], list)
        assert len(row["prompt"]) >= 2
        assert row["prompt"][0]["role"] == "system"
        assert row["prompt"][1]["role"] == "user"

        # reward_model and extra_info are dicts
        assert isinstance(row["reward_model"], dict)
        assert isinstance(row["extra_info"], dict)

        # extra_info has domain_type
        assert row["extra_info"]["domain_type"] == expected_domain

        # extra_info has question text
        assert "question" in row["extra_info"]
        assert len(row["extra_info"]["question"]) > 0

    def test_gsm8k_adapter(self):
        from grubrics_science.data.adapters.gsm8k import GSM8KAdapter

        adapter = GSM8KAdapter()
        row = adapter.to_verl_format({
            "question": MATH_QUESTION,
            "solution": "5 + 3 = 8 #### 8",
            "final_answer": "8",
        })

        self._check_row(row, "gsm8k", "verifiable")
        assert row["reward_model"]["ground_truth"] == "8"

    def test_math_hendrycks_adapter(self):
        from grubrics_science.data.adapters.math_hendrycks import MATHAdapter

        adapter = MATHAdapter()
        # load_raw outputs "question" (mapped from HF "problem" field)
        row = adapter.to_verl_format({
            "question": "Solve: 2x + 4 = 10",
            "solution": "2x = 6, x = \\boxed{3}",
            "final_answer": "3",
            "level": "Level 1",
            "subject": "Algebra",
        })

        self._check_row(row, "math", "verifiable")
        assert "ground_truth" in row["reward_model"]

    def test_frontierscience_adapter(self):
        from grubrics_science.data.adapters.frontierscience import FrontierScienceAdapter

        adapter = FrontierScienceAdapter()
        row = adapter.to_verl_format({
            "question_id": "0",
            "problem": "Derive the mass-energy relation for a relativistic particle.",
            "golden_rubric": "Points: 5.0, Item: Correct derivation\nPoints: 5.0, Item: Clear explanation",
            "subject": "physics",
            "task_group_id": "group_1",
        })

        self._check_row(row, "frontierscience", "open_rubric")

        # FrontierScience-specific fields
        assert "golden_rubric" in row["extra_info"]
        assert "answers" in row["extra_info"]  # empty list without cache
        assert "gold_scores" in row["extra_info"]
        assert row["reward_model"]["ground_truth"] == ""  # open-ended

class TestParquetSerialization:
    """Base adapter serialization: prompt→JSON string, dicts stay dicts."""

    def test_prompt_becomes_json_string(self):
        from grubrics_science.data.adapters.gsm8k import GSM8KAdapter

        adapter = GSM8KAdapter()
        row = adapter.to_verl_format({
            "question": "What is 2 + 2?",
            "solution": "2 + 2 = 4 #### 4",
            "final_answer": "4",
        })

        # Simulate base.py serialization
        if not isinstance(row["prompt"], str):
            row["prompt"] = json.dumps(row["prompt"], ensure_ascii=False)

        # prompt is now a JSON string
        assert isinstance(row["prompt"], str)
        parsed = json.loads(row["prompt"])
        assert isinstance(parsed, list)
        assert parsed[0]["role"] == "system"

        # reward_model and extra_info must NOT be serialized to strings
        assert isinstance(row["reward_model"], dict)
        assert isinstance(row["extra_info"], dict)

    def test_to_parquet_end_to_end(self):
        """to_parquet writes a valid parquet file with correct types."""
        import pandas as pd
        from grubrics_science.data.adapters.gsm8k import GSM8KAdapter

        adapter = GSM8KAdapter()

        # Monkey-patch load_raw to avoid HuggingFace download
        fake_items = [
            {"question": f"Question {i}?", "solution": f"sol #### {i}", "final_answer": str(i)}
            for i in range(5)
        ]
        adapter.load_raw = lambda path=None: fake_items

        with tempfile.TemporaryDirectory() as tmpdir:
            parquet_path = adapter.to_parquet(output_dir=tmpdir, split="test")
            assert parquet_path.exists()

            df = pd.read_parquet(parquet_path)
            assert len(df) == 5
            assert "data_source" in df.columns
            assert "prompt" in df.columns

            # prompt should be JSON string in parquet
            first_prompt = df.iloc[0]["prompt"]
            assert isinstance(first_prompt, str)
            parsed = json.loads(first_prompt)
            assert isinstance(parsed, list)


# =========================================================================
# Reward function tests
# =========================================================================

class TestFormatScore:
    """format_score: checks rubric format (Points/Item lines, sum to 10)."""

    def test_good_rubric_scores_high(self):
        from grubrics_science.rewards.gsm8k_reward import format_score

        score = format_score(GOOD_RUBRIC)
        assert score >= 0.8, f"Good rubric should score >= 0.8, got {score}"

    def test_no_format_scores_zero(self):
        from grubrics_science.rewards.gsm8k_reward import format_score

        score = format_score(BAD_RUBRIC_NO_FORMAT)
        assert score == 0.0, f"No format should score 0.0, got {score}"

    def test_wrong_total_gets_partial_credit(self):
        from grubrics_science.rewards.gsm8k_reward import format_score

        score = format_score(BAD_RUBRIC_WRONG_TOTAL)
        assert 0.1 < score < 0.8, f"Wrong total should get partial credit, got {score}"

    def test_empty_string_scores_zero(self):
        from grubrics_science.rewards.gsm8k_reward import format_score

        assert format_score("") == 0.0

    def test_close_to_ten_gets_partial(self):
        from grubrics_science.rewards.gsm8k_reward import format_score

        # Sums to 9.5 — close but not exact
        rubric = (
            "Points: 3.0, Item: Correct computation of the result\n"
            "Points: 3.0, Item: Clear step-by-step solution\n"
            "Points: 3.5, Item: Proper identification of the approach"
        )
        score = format_score(rubric)
        assert score > 0.5, f"Close-to-10 rubric should get decent score, got {score}"


class TestCoherenceScore:
    """coherence_score: checks item quality and uniqueness."""

    def test_substantive_items_score_high(self):
        from grubrics_science.rewards.gsm8k_reward import coherence_score

        score = coherence_score(GOOD_RUBRIC, MATH_QUESTION)
        assert score >= 0.6, f"Substantive items should score >= 0.6, got {score}"

    def test_trivial_items_score_low(self):
        from grubrics_science.rewards.gsm8k_reward import coherence_score

        score = coherence_score(BAD_RUBRIC_TRIVIAL_ITEMS, MATH_QUESTION)
        assert score < 0.5, f"Trivial items should score < 0.5, got {score}"

    def test_duplicate_items_penalized(self):
        from grubrics_science.rewards.gsm8k_reward import coherence_score

        duplicated = (
            "Points: 5.0, Item: The answer correctly identifies the solution\n"
            "Points: 5.0, Item: The answer correctly identifies the solution"
        )
        unique = (
            "Points: 5.0, Item: The answer correctly identifies the solution\n"
            "Points: 5.0, Item: The answer shows clear mathematical derivation"
        )
        dup_score = coherence_score(duplicated, MATH_QUESTION)
        uniq_score = coherence_score(unique, MATH_QUESTION)
        assert uniq_score > dup_score, (
            f"Unique items ({uniq_score}) should score higher than duplicates ({dup_score})"
        )

    def test_no_format_scores_zero(self):
        from grubrics_science.rewards.gsm8k_reward import coherence_score

        assert coherence_score(BAD_RUBRIC_NO_FORMAT, MATH_QUESTION) == 0.0


class TestComputeScore:
    """compute_score: the function veRL calls. Must discriminate."""

    def test_good_beats_bad(self):
        from grubrics_science.rewards.gsm8k_reward import compute_score

        good = compute_score(
            data_source="gsm8k",
            solution_str=GOOD_RUBRIC,
            extra_info={"question": MATH_QUESTION},
        )
        bad = compute_score(
            data_source="gsm8k",
            solution_str=BAD_RUBRIC_NO_FORMAT,
            extra_info={"question": MATH_QUESTION},
        )

        assert good > bad, f"Good ({good}) must beat bad ({bad})"
        assert good > 0.5
        assert bad < 0.2

    def test_returns_float_in_range(self):
        from grubrics_science.rewards.gsm8k_reward import compute_score

        score = compute_score(
            data_source="gsm8k",
            solution_str=GOOD_RUBRIC,
            extra_info={"question": MATH_QUESTION},
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_works_without_extra_info(self):
        """compute_score should not crash when extra_info is missing."""
        from grubrics_science.rewards.gsm8k_reward import compute_score

        score = compute_score(
            data_source="gsm8k",
            solution_str=GOOD_RUBRIC,
        )
        assert isinstance(score, float)


# =========================================================================
# Alignment metric tests
# =========================================================================

class TestSpearmanCorrelation:
    def test_perfect_positive(self):
        from grubrics_science.rewards.alignment import spearman_correlation

        corr = spearman_correlation([1, 2, 3, 4, 5], [10, 20, 30, 40, 50])
        assert abs(corr - 1.0) < 0.01

    def test_perfect_negative(self):
        from grubrics_science.rewards.alignment import spearman_correlation

        corr = spearman_correlation([5, 4, 3, 2, 1], [1, 2, 3, 4, 5])
        assert abs(corr - (-1.0)) < 0.01

    def test_weak_correlation(self):
        from grubrics_science.rewards.alignment import spearman_correlation

        # These rankings are weakly correlated at best
        corr = spearman_correlation([1, 2, 3, 4, 5], [2, 4, 1, 5, 3])
        assert abs(corr) < 0.8  # not strongly correlated

    def test_constant_scores_return_zero(self):
        from grubrics_science.rewards.alignment import spearman_correlation

        corr = spearman_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
        assert corr == 0.0

    def test_single_element(self):
        from grubrics_science.rewards.alignment import spearman_correlation

        corr = spearman_correlation([1.0], [5.0])
        assert corr == 1.0


class TestPairwiseAccuracy:
    def test_perfect_ordering(self):
        from grubrics_science.rewards.alignment import pairwise_accuracy

        acc = pairwise_accuracy([1, 2, 3], [10, 20, 30])
        assert abs(acc - 1.0) < 0.01

    def test_fully_inverted(self):
        from grubrics_science.rewards.alignment import pairwise_accuracy

        acc = pairwise_accuracy([3, 2, 1], [1, 2, 3])
        assert abs(acc) < 0.01

    def test_partial_ordering(self):
        from grubrics_science.rewards.alignment import pairwise_accuracy

        # 2 out of 3 pairs correct
        acc = pairwise_accuracy([1, 3, 2], [1, 2, 3])
        assert 0.3 < acc < 0.9

    def test_all_ties_in_gold(self):
        from grubrics_science.rewards.alignment import pairwise_accuracy

        acc = pairwise_accuracy([1, 2, 3], [5, 5, 5])
        assert acc == 1.0  # no non-tie pairs to get wrong


class TestComputeReward:
    def test_high_alignment_positive_reward(self):
        from grubrics_science.rewards.alignment import compute_reward

        reward = compute_reward(
            scores=[1, 2, 3],
            gold_scores=[10, 20, 30],
            rubric_text="short",
            alignment_metric="spearman",
            lambda_len=0.001,
        )
        assert reward > 0.9

    def test_low_alignment_low_reward(self):
        from grubrics_science.rewards.alignment import compute_reward

        reward = compute_reward(
            scores=[3, 1, 2],
            gold_scores=[1, 2, 3],
            rubric_text="short",
            alignment_metric="spearman",
            lambda_len=0.0,
        )
        assert reward < 0.5

    def test_length_penalty_reduces_reward(self):
        from grubrics_science.rewards.alignment import compute_reward

        short = compute_reward(
            scores=[1, 2, 3],
            gold_scores=[10, 20, 30],
            rubric_text="short",
            lambda_len=0.01,
        )
        long = compute_reward(
            scores=[1, 2, 3],
            gold_scores=[10, 20, 30],
            rubric_text="a" * 1000,
            lambda_len=0.01,
        )
        assert short > long
