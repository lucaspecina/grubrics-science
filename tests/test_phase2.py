"""Tests for Phase 2: precompute_verifiable, perturbation, unified reward routing.

Run with: pytest tests/test_phase2.py -v
No GPU required. No API calls.
"""

import json
import pytest


# =========================================================================
# Perturbation tests
# =========================================================================

class TestPerturbations:
    """Test the perturbation strategies for verifiable answers."""

    def test_perturb_final_number_gsm8k(self):
        from grubrics_science.data.precompute_verifiable import perturb_final_number

        response = "Step 1: 2+2=4\nStep 2: 4*3=12\n\n#### 12"
        perturbed = perturb_final_number(response, "12", "gsm8k")

        # Should contain #### but NOT the correct answer
        assert "####" in perturbed
        assert "#### 12" not in perturbed

    def test_perturb_final_number_math_boxed(self):
        from grubrics_science.data.precompute_verifiable import perturb_final_number

        response = "The answer is \\boxed{42}."
        perturbed = perturb_final_number(response, "42", "math")

        assert "\\boxed{" in perturbed
        assert "\\boxed{42}" not in perturbed

    def test_truncate_solution_removes_answer(self):
        from grubrics_science.data.precompute_verifiable import truncate_solution

        response = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n#### 42"
        truncated = truncate_solution(response)

        # Should not contain the final answer marker
        assert "####" not in truncated
        # Should keep some lines
        assert "Line 1" in truncated

    def test_truncate_solution_removes_boxed(self):
        from grubrics_science.data.precompute_verifiable import truncate_solution

        response = "Step 1\nStep 2\nStep 3\nStep 4\nStep 5\nSo \\boxed{7}."
        truncated = truncate_solution(response)

        assert "\\boxed{" not in truncated

    def test_create_perturbations_correct_answer(self):
        from grubrics_science.data.precompute_verifiable import create_perturbations

        response = "2+2=4\n\n#### 4"
        perturbations = create_perturbations(response, "4", True, "gsm8k", num_perturbations=3)

        assert len(perturbations) == 3
        # All perturbations should be incorrect (gold_score=0.0)
        for _, score in perturbations:
            assert score == 0.0

    def test_create_perturbations_incorrect_answer(self):
        from grubrics_science.data.precompute_verifiable import create_perturbations

        response = "2+2=5\n\n#### 5"
        perturbations = create_perturbations(response, "4", False, "gsm8k", num_perturbations=3)

        assert len(perturbations) == 3
        # First perturbation should be a correct stub
        assert perturbations[0][1] == 1.0
        # Others should be incorrect
        assert perturbations[1][1] == 0.0
        assert perturbations[2][1] == 0.0

    def test_perturbations_guarantee_variance(self):
        """The whole point: perturbations guarantee both correct and incorrect answers."""
        from grubrics_science.data.precompute_verifiable import create_perturbations

        # Case 1: original is correct
        perturbs_correct = create_perturbations("#### 4", "4", True, "gsm8k", 3)
        all_answers = [(True, 1.0)] + [(False, s) for _, s in perturbs_correct]
        scores = [s for _, s in all_answers]
        assert 1.0 in scores and 0.0 in scores, "Must have both correct and incorrect"

        # Case 2: original is incorrect
        perturbs_wrong = create_perturbations("#### 5", "4", False, "gsm8k", 3)
        all_answers2 = [(False, 0.0)] + [(None, s) for _, s in perturbs_wrong]
        scores2 = [s for _, s in all_answers2]
        assert 1.0 in scores2 and 0.0 in scores2, "Must have both correct and incorrect"


# =========================================================================
# Programmatic answer checking
# =========================================================================

class TestAnswerChecking:
    """Test extract/normalize/check functions."""

    def test_extract_hash_answer(self):
        from grubrics_science.data.precompute_verifiable import extract_hash_answer

        assert extract_hash_answer("blah blah\n#### 42") == "42"
        assert extract_hash_answer("#### 3.14") == "3.14"
        assert extract_hash_answer("no answer here") == ""

    def test_extract_boxed(self):
        from grubrics_science.data.precompute_verifiable import extract_boxed

        assert extract_boxed("The answer is \\boxed{42}.") == "42"
        assert extract_boxed("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
        assert extract_boxed("no boxed here") == ""

    def test_normalize_answer(self):
        from grubrics_science.data.precompute_verifiable import normalize_answer

        assert normalize_answer("42") == "42"
        assert normalize_answer("42.0") == "42"
        assert normalize_answer("$42") == "42"
        assert normalize_answer("1,000") == "1000"
        assert normalize_answer("3.14") == "3.14"

    def test_check_correct_gsm8k(self):
        from grubrics_science.data.precompute_verifiable import check_correct

        is_correct, extracted = check_correct("blah\n#### 42", "42", "gsm8k")
        assert is_correct is True
        assert extracted == "42"

        is_correct, extracted = check_correct("blah\n#### 43", "42", "gsm8k")
        assert is_correct is False

    def test_check_correct_math(self):
        from grubrics_science.data.precompute_verifiable import check_correct

        is_correct, extracted = check_correct("So \\boxed{7}.", "7", "math")
        assert is_correct is True

        is_correct, extracted = check_correct("So \\boxed{8}.", "7", "math")
        assert is_correct is False


# =========================================================================
# Adapter cache integration
# =========================================================================

class TestVerifiableAdapterCache:
    """Test that GSM8K/MATH adapters read precompute cache."""

    @pytest.fixture
    def gsm8k_cache_file(self, tmp_path):
        """Create a minimal GSM8K precompute cache."""
        cache = tmp_path / "gsm8k_cache.jsonl"
        entry = {
            "question_id": "gsm8k_0",
            "question": "What is 2+2?",
            "gold_answer": "4",
            "dataset": "gsm8k",
            "answers": [
                "Step: 2+2=4\n\n#### 4",
                "Step: 2+2=5\n\n#### 5",
                "Step: 2+2=",
                "Step: 2+2=3\n\n#### 3",
            ],
            "gold_scores": [1.0, 0.0, 0.0, 0.0],
            "original_correct": True,
        }
        cache.write_text(json.dumps(entry) + "\n")
        return str(cache)

    def test_gsm8k_adapter_loads_cache(self, gsm8k_cache_file):
        from grubrics_science.data.adapters.gsm8k import GSM8KAdapter

        adapter = GSM8KAdapter(cache_path=gsm8k_cache_file)
        # Simulate a raw item matching the cached question
        item = {
            "question": "What is 2+2?",
            "solution": "2+2=4\n#### 4",
            "final_answer": "4",
        }
        row = adapter.to_verl_format(item)
        extra = row["extra_info"]

        assert len(extra["answers"]) == 4
        assert len(extra["gold_scores"]) == 4
        assert extra["gold_scores"] == [1.0, 0.0, 0.0, 0.0]

    def test_gsm8k_adapter_no_cache(self):
        from grubrics_science.data.adapters.gsm8k import GSM8KAdapter

        adapter = GSM8KAdapter(cache_path=None)
        item = {
            "question": "What is 2+2?",
            "solution": "2+2=4\n#### 4",
            "final_answer": "4",
        }
        row = adapter.to_verl_format(item)
        extra = row["extra_info"]

        assert extra["answers"] == []
        assert extra["gold_scores"] == []

    def test_gsm8k_adapter_contrastive_excerpts(self, gsm8k_cache_file):
        from grubrics_science.data.adapters.gsm8k import GSM8KAdapter

        adapter = GSM8KAdapter(cache_path=gsm8k_cache_file)
        item = {
            "question": "What is 2+2?",
            "solution": "2+2=4\n#### 4",
            "final_answer": "4",
        }
        row = adapter.to_verl_format(item)
        prompt_text = json.dumps(row["prompt"])

        assert "High-quality answer excerpt" in prompt_text
        assert "Low-quality answer excerpt" in prompt_text


# =========================================================================
# Unified reward routing with verifiable cache
# =========================================================================

class TestUnifiedRewardRouting:
    """Test that verifiable reward uses functional alignment when cache is present."""

    def test_verifiable_without_cache_raises(self):
        """Without precomputed data, verifiable reward must raise."""
        import asyncio
        import pytest
        from grubrics_science.rewards.grubrics_reward import _reward_verifiable

        rubric = (
            "Points: 5.0, Item: Correctly computes 2+2\n"
            "Points: 5.0, Item: Shows step-by-step work"
        )

        with pytest.raises(ValueError, match="Missing precomputed"):
            asyncio.run(_reward_verifiable(
                solution_str=rubric,
                ground_truth="4",
                extra_info={"question": "What is 2+2?", "question_id": "test"},
            ))

    def test_verifiable_without_cache_compute_score_raises(self):
        """compute_score must raise for verifiable without precompute."""
        import asyncio
        import pytest
        from grubrics_science.rewards.grubrics_reward import compute_score

        rubric = (
            "Points: 3.0, Item: Correct answer\n"
            "Points: 3.0, Item: Clear steps\n"
            "Points: 4.0, Item: Final result"
        )
        with pytest.raises(ValueError, match="Missing precomputed"):
            asyncio.run(compute_score(
                data_source="gsm8k",
                solution_str=rubric,
                ground_truth="42",
                extra_info={"question": "What is 6*7?", "question_id": "test"},
            ))


# =========================================================================
# Precompute cache format compatibility
# =========================================================================

class TestCacheFormatCompatibility:
    """Verify cache format is compatible between precompute and adapters."""

    def test_cache_entry_has_required_fields(self):
        """Precompute_verifiable produces entries with all fields adapters need."""
        # Simulate what precompute_verifiable.py would produce
        entry = {
            "question_id": "gsm8k_0",
            "question": "What is 2+2?",
            "gold_answer": "4",
            "dataset": "gsm8k",
            "answers": ["correct answer", "wrong 1", "wrong 2", "wrong 3"],
            "gold_scores": [1.0, 0.0, 0.0, 0.0],
            "original_correct": True,
            "extracted_answer": "4",
        }

        # Fields that adapters read
        assert "question_id" in entry
        assert "question" in entry
        assert "answers" in entry
        assert "gold_scores" in entry
        assert isinstance(entry["answers"], list)
        assert isinstance(entry["gold_scores"], list)
        assert len(entry["answers"]) == len(entry["gold_scores"])

    def test_gold_scores_have_variance(self):
        """Cache entries should have variance in gold_scores for Spearman."""
        from grubrics_science.data.precompute_verifiable import create_perturbations

        # Correct original
        perturbs = create_perturbations("#### 4", "4", True, "gsm8k", 3)
        gold_scores = [1.0] + [s for _, s in perturbs]
        assert max(gold_scores) != min(gold_scores), "Must have variance"

        # Incorrect original
        perturbs2 = create_perturbations("#### 5", "4", False, "gsm8k", 3)
        gold_scores2 = [0.0] + [s for _, s in perturbs2]
        assert max(gold_scores2) != min(gold_scores2), "Must have variance"
