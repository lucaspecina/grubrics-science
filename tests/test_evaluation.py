"""Tests for the evaluation module.

Run with: pytest tests/test_evaluation.py -v
No GPU required. No API calls.
"""

import json
import pytest


# =========================================================================
# Metrics tests
# =========================================================================

class TestAlignmentScore:
    """Test alignment_score (Spearman)."""

    def test_perfect_correlation(self):
        from grubrics_science.evaluation.metrics import alignment_score
        assert alignment_score([0.1, 0.5, 0.9], [0.2, 0.6, 1.0]) == pytest.approx(1.0)

    def test_inverse_correlation(self):
        from grubrics_science.evaluation.metrics import alignment_score
        score = alignment_score([0.9, 0.5, 0.1], [0.1, 0.5, 0.9])
        assert score == pytest.approx(-1.0)

    def test_no_correlation(self):
        from grubrics_science.evaluation.metrics import alignment_score
        # With only 2 values, correlation is either +1 or -1
        # Use 4+ values for a more meaningful test
        score = alignment_score([0.1, 0.9, 0.1, 0.9], [0.5, 0.5, 0.5, 0.5])
        assert score == pytest.approx(0.0)

    def test_too_few_values(self):
        from grubrics_science.evaluation.metrics import alignment_score
        assert alignment_score([0.5], [0.5]) == 0.0

    def test_mismatched_lengths(self):
        from grubrics_science.evaluation.metrics import alignment_score
        assert alignment_score([0.1, 0.2], [0.1]) == 0.0


class TestDiscriminationScore:
    """Test discrimination_score (std)."""

    def test_degenerate(self):
        from grubrics_science.evaluation.metrics import discrimination_score
        assert discrimination_score([0.5, 0.5, 0.5]) == pytest.approx(0.0)

    def test_high_variance(self):
        from grubrics_science.evaluation.metrics import discrimination_score
        score = discrimination_score([0.0, 1.0])
        assert score > 0.4

    def test_single_value(self):
        from grubrics_science.evaluation.metrics import discrimination_score
        assert discrimination_score([0.5]) == 0.0


class TestFormatValidity:
    """Test format_validity (rubric line matching)."""

    def test_all_valid(self):
        from grubrics_science.evaluation.metrics import format_validity
        rubric = (
            "Points: 3.0, Item: Correct derivation\n"
            "Points: 2.0, Item: Clear explanation\n"
            "Points: 5.0, Item: Uses proper units"
        )
        assert format_validity(rubric) == pytest.approx(1.0)

    def test_none_valid(self):
        from grubrics_science.evaluation.metrics import format_validity
        rubric = "This is just random text\nAnother line"
        assert format_validity(rubric) == pytest.approx(0.0)

    def test_partial_valid(self):
        from grubrics_science.evaluation.metrics import format_validity
        rubric = (
            "Points: 5.0, Item: Good item\n"
            "This line is invalid\n"
            "Points: 5.0, Item: Another good item"
        )
        assert format_validity(rubric) == pytest.approx(2 / 3, abs=0.01)

    def test_empty_rubric(self):
        from grubrics_science.evaluation.metrics import format_validity
        assert format_validity("") == 0.0


class TestPointsSum:
    """Test points_sum."""

    def test_sums_to_ten(self):
        from grubrics_science.evaluation.metrics import points_sum
        rubric = (
            "Points: 3.0, Item: A\n"
            "Points: 3.0, Item: B\n"
            "Points: 4.0, Item: C"
        )
        assert points_sum(rubric) == pytest.approx(10.0)

    def test_non_standard_sum(self):
        from grubrics_science.evaluation.metrics import points_sum
        rubric = "Points: 2.5, Item: A\nPoints: 1.5, Item: B"
        assert points_sum(rubric) == pytest.approx(4.0)


class TestInfoValue:
    """Test info_value (discriminativeness)."""

    def test_max_at_half_split(self):
        from grubrics_science.evaluation.metrics import info_value
        # 3 above, 3 below threshold = p=0.5 → 4*0.5*0.5 = 1.0
        assert info_value([0.3, 0.4, 0.45, 0.55, 0.6, 0.7]) == pytest.approx(1.0)

    def test_all_above(self):
        from grubrics_science.evaluation.metrics import info_value
        # All above threshold = p=1.0 → 0.0
        assert info_value([0.6, 0.7, 0.8, 0.9]) == pytest.approx(0.0)


class TestComputeAllMetrics:
    """Test compute_all_metrics aggregation."""

    def test_returns_all_keys(self):
        from grubrics_science.evaluation.metrics import compute_all_metrics
        rubric = "Points: 5.0, Item: A\nPoints: 5.0, Item: B"
        scores = [0.3, 0.7, 0.5]
        gold = [0.2, 0.8, 0.4]
        result = compute_all_metrics(rubric, scores, gold)
        expected_keys = {"alignment", "discrimination", "format_validity",
                         "points_sum", "info_value", "length"}
        assert expected_keys == set(result.keys())


# =========================================================================
# Baselines tests
# =========================================================================

class TestGoldenRubricBaseline:
    """Test B0: golden rubric baseline."""

    def test_returns_golden(self):
        from grubrics_science.evaluation.baselines import golden_rubric
        entry = {"golden_rubric": "Points: 10.0, Item: Test"}
        assert golden_rubric(entry) == "Points: 10.0, Item: Test"

    def test_missing_golden(self):
        from grubrics_science.evaluation.baselines import golden_rubric
        assert golden_rubric({}) == ""


class TestRandomRubricBaseline:
    """Test B3: random rubric baseline."""

    def test_format(self):
        from grubrics_science.evaluation.baselines import random_rubric
        rubric = random_rubric({"question": "test"}, seed=42)
        lines = rubric.strip().split("\n")
        assert len(lines) >= 5
        assert len(lines) <= 8
        for line in lines:
            assert line.startswith("Points:")
            assert "Item:" in line

    def test_points_sum_to_ten(self):
        from grubrics_science.evaluation.baselines import random_rubric
        from grubrics_science.evaluation.metrics import points_sum
        rubric = random_rubric({"question": "test"}, seed=42)
        assert points_sum(rubric) == pytest.approx(10.0, abs=0.2)

    def test_deterministic(self):
        from grubrics_science.evaluation.baselines import random_rubric
        r1 = random_rubric({"question": "test"}, seed=42)
        r2 = random_rubric({"question": "test"}, seed=42)
        assert r1 == r2

    def test_seeded_baseline_varies(self):
        from grubrics_science.evaluation.baselines import SeededRandomBaseline
        baseline = SeededRandomBaseline(base_seed=42)
        r1 = baseline({"question_id": "0"})
        r2 = baseline({"question_id": "1"})
        assert r1 != r2


# =========================================================================
# Holdout tests
# =========================================================================

class TestHoldout:
    """Test holdout data loading and splitting."""

    @pytest.fixture
    def mock_dataset(self, tmp_path):
        """Create a mock FrontierScience dataset."""
        ds_path = tmp_path / "test.jsonl"
        cache_path = tmp_path / "cache.jsonl"

        records = []
        cache_entries = []
        for i in range(10):
            records.append({
                "problem": f"Question {i}",
                "answer": f"Points: 10.0, Item: Answer for Q{i}",
                "subject": "physics",
            })
            cache_entries.append({
                "question_id": str(i),
                "question": f"Question {i}",
                "golden_rubric": f"Points: 10.0, Item: Answer for Q{i}",
                "answers": [f"Answer A for Q{i}", f"Answer B for Q{i}"],
                "gold_scores": [0.8, 0.4],
            })

        ds_path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        cache_path.write_text("\n".join(json.dumps(e) for e in cache_entries) + "\n")

        return str(ds_path), str(cache_path)

    def test_load_with_cache(self, mock_dataset):
        from grubrics_science.evaluation.holdout import load_frontierscience_with_cache
        ds_path, cache_path = mock_dataset
        data = load_frontierscience_with_cache(ds_path, cache_path)
        assert len(data) == 10
        assert "answers" in data[0]
        assert "gold_scores" in data[0]
        assert "golden_rubric" in data[0]

    def test_split_holdout(self, mock_dataset):
        from grubrics_science.evaluation.holdout import (
            load_frontierscience_with_cache,
            split_holdout,
        )
        ds_path, cache_path = mock_dataset
        data = load_frontierscience_with_cache(ds_path, cache_path)
        train, holdout = split_holdout(data, holdout_size=3, seed=42)
        assert len(train) == 7
        assert len(holdout) == 3

    def test_split_deterministic(self, mock_dataset):
        from grubrics_science.evaluation.holdout import (
            load_frontierscience_with_cache,
            split_holdout,
        )
        ds_path, cache_path = mock_dataset
        data = load_frontierscience_with_cache(ds_path, cache_path)
        _, h1 = split_holdout(data, holdout_size=3, seed=42)
        _, h2 = split_holdout(data, holdout_size=3, seed=42)
        ids1 = [q["question_id"] for q in h1]
        ids2 = [q["question_id"] for q in h2]
        assert ids1 == ids2

    def test_no_overlap(self, mock_dataset):
        from grubrics_science.evaluation.holdout import (
            load_frontierscience_with_cache,
            split_holdout,
        )
        ds_path, cache_path = mock_dataset
        data = load_frontierscience_with_cache(ds_path, cache_path)
        train, holdout = split_holdout(data, holdout_size=3, seed=42)
        train_ids = {q["question_id"] for q in train}
        holdout_ids = {q["question_id"] for q in holdout}
        assert train_ids.isdisjoint(holdout_ids)


# =========================================================================
# Eval pipeline tests (with mock Judge)
# =========================================================================

class TestEvalPipeline:
    """Test the evaluation pipeline with a mock Judge."""

    def test_evaluate_rubric_on_question(self):
        """Test evaluation with a mock judge."""
        from grubrics_science.evaluation.eval_rubrics import evaluate_rubric_on_question
        from unittest.mock import AsyncMock

        # Mock judge that returns fixed scores
        mock_judge = AsyncMock()
        mock_judge.evaluate_answers_batched = AsyncMock(
            return_value=[0.8, 0.4, 0.6]
        )

        result = evaluate_rubric_on_question(
            rubric_text="Points: 5.0, Item: A\nPoints: 5.0, Item: B",
            question="Test question",
            answers=["a1", "a2", "a3"],
            gold_scores=[0.9, 0.3, 0.7],
            judge=mock_judge,
        )

        assert "alignment" in result
        assert "discrimination" in result
        assert "rubric_scores" in result
        assert result["alignment"] > 0.5  # Should be well correlated

    def test_evaluate_on_holdout_with_mock(self):
        """Test full holdout evaluation with mock judge."""
        from grubrics_science.evaluation.eval_rubrics import evaluate_on_holdout
        from unittest.mock import AsyncMock

        mock_judge = AsyncMock()
        mock_judge.evaluate_answers_batched = AsyncMock(
            return_value=[0.8, 0.4, 0.6]
        )

        holdout = [
            {
                "question_id": "0",
                "question": "Q0",
                "answers": ["a1", "a2", "a3"],
                "gold_scores": [0.9, 0.3, 0.7],
                "golden_rubric": "Points: 10.0, Item: Test",
            },
            {
                "question_id": "1",
                "question": "Q1",
                "answers": ["a1", "a2", "a3"],
                "gold_scores": [0.2, 0.8, 0.5],
                "golden_rubric": "Points: 10.0, Item: Test2",
            },
        ]

        def gen_fn(entry):
            return entry["golden_rubric"]

        result = evaluate_on_holdout(
            rubric_generator_fn=gen_fn,
            holdout_data=holdout,
            judge=mock_judge,
            verbose=False,
        )

        assert result["num_questions"] == 2
        assert "aggregated" in result
        assert "per_question" in result
        assert "alignment_mean" in result["aggregated"]
