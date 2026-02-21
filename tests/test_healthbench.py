"""Tests for HealthBench adapter, precompute, holdout, and Judge validation.

Run with: pytest tests/test_healthbench.py -v
No GPU required. No API calls (uses mock data).
"""

import json
import os
import tempfile

import pytest


# =========================================================================
# Rubric parsing
# =========================================================================

class TestRubricParsing:
    """HealthBench rubric JSON -> text conversion."""

    def test_basic_rubric(self):
        from grubrics_science.data.adapters.healthbench import _rubrics_to_text

        rubrics = [
            {"criterion": "Mentions correct diagnosis", "points": 10, "tags": ["accuracy"]},
            {"criterion": "Explains treatment options", "points": 5, "tags": ["completeness"]},
        ]
        text = _rubrics_to_text(rubrics)
        assert "Points: 10, Item: Mentions correct diagnosis" in text
        assert "Points: 5, Item: Explains treatment options" in text

    def test_negative_points(self):
        from grubrics_science.data.adapters.healthbench import _rubrics_to_text

        rubrics = [
            {"criterion": "Gives dangerous advice", "points": -8, "tags": ["safety"]},
        ]
        text = _rubrics_to_text(rubrics)
        assert "Points: -8, Item: Gives dangerous advice" in text

    def test_empty_rubrics(self):
        from grubrics_science.data.adapters.healthbench import _rubrics_to_text

        assert _rubrics_to_text([]) == ""

    def test_missing_criterion(self):
        from grubrics_science.data.adapters.healthbench import _rubrics_to_text

        rubrics = [{"points": 5}]
        assert _rubrics_to_text(rubrics) == ""

    def test_zero_points(self):
        from grubrics_science.data.adapters.healthbench import _rubrics_to_text

        rubrics = [{"criterion": "Optional note", "points": 0, "tags": []}]
        text = _rubrics_to_text(rubrics)
        assert "Points: 0, Item: Optional note" in text


# =========================================================================
# Question text extraction
# =========================================================================

class TestQuestionExtraction:
    """Multi-turn prompt -> flat question text."""

    def test_single_turn(self):
        from grubrics_science.data.adapters.healthbench import _extract_question_text

        prompt = [{"role": "user", "content": "What is diabetes?"}]
        text = _extract_question_text(prompt)
        assert "user: What is diabetes?" in text

    def test_multi_turn(self):
        from grubrics_science.data.adapters.healthbench import _extract_question_text

        prompt = [
            {"role": "user", "content": "I have a headache"},
            {"role": "assistant", "content": "How long have you had it?"},
            {"role": "user", "content": "About 3 days"},
        ]
        text = _extract_question_text(prompt)
        assert "user: I have a headache" in text
        assert "assistant: How long have you had it?" in text
        assert "user: About 3 days" in text

    def test_empty_prompt(self):
        from grubrics_science.data.adapters.healthbench import _extract_question_text

        assert _extract_question_text([]) == ""


# =========================================================================
# HealthBenchAdapter
# =========================================================================

class TestHealthBenchAdapter:
    """HealthBenchAdapter: load, parse, cache, veRL format."""

    @pytest.fixture
    def sample_oss_eval(self, tmp_path):
        """Create a minimal oss_eval.jsonl for testing."""
        records = [
            {
                "prompt_id": "hb_001",
                "prompt": [{"role": "user", "content": "What causes migraines?"}],
                "rubrics": [
                    {"criterion": "Mentions triggers", "points": 5, "tags": ["completeness"]},
                    {"criterion": "Mentions treatment", "points": 10, "tags": ["accuracy"]},
                ],
                "category": "neurology",
                "example_tags": ["headache"],
                "ideal_completions_data": {
                    "ideal_completion": "Migraines are caused by...",
                    "ideal_completions_ref_completions": ["Reference answer 1"],
                },
            },
            {
                "prompt_id": "hb_002",
                "prompt": [{"role": "user", "content": "Is aspirin safe?"}],
                "rubrics": [
                    {"criterion": "Mentions side effects", "points": 8, "tags": ["safety"]},
                ],
                "category": "pharmacology",
                "example_tags": [],
                "ideal_completions_data": None,
            },
        ]
        path = tmp_path / "oss_eval.jsonl"
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return str(path)

    @pytest.fixture
    def sample_meta_eval(self, tmp_path):
        """Create a minimal meta_eval.jsonl for testing."""
        entries = [
            {"prompt_id": "hb_001", "completion": "Answer from model A about migraines"},
            {"prompt_id": "hb_001", "completion": "Answer from model B about migraines"},
            {"prompt_id": "hb_002", "completion": "Aspirin is generally safe but..."},
        ]
        path = tmp_path / "meta_eval.jsonl"
        with open(path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        return str(path)

    @pytest.fixture
    def sample_cache(self, tmp_path):
        """Create a minimal precompute cache."""
        entries = [
            {
                "prompt_id": "hb_001",
                "answers": ["Good answer", "Bad answer"],
                "gold_scores": [0.85, 0.25],
            },
        ]
        path = tmp_path / "cache.jsonl"
        with open(path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        return str(path)

    def test_load_from_jsonl(self, sample_oss_eval):
        from grubrics_science.data.adapters.healthbench import HealthBenchAdapter

        adapter = HealthBenchAdapter(dataset_path=sample_oss_eval)
        items = adapter.load_raw()

        assert len(items) == 2
        assert items[0]["prompt_id"] == "hb_001"
        assert items[1]["prompt_id"] == "hb_002"
        assert "Mentions triggers" in items[0]["golden_rubric"]
        assert "Mentions treatment" in items[0]["golden_rubric"]

    def test_parse_record_fields(self, sample_oss_eval):
        from grubrics_science.data.adapters.healthbench import HealthBenchAdapter

        adapter = HealthBenchAdapter(dataset_path=sample_oss_eval)
        items = adapter.load_raw()

        item = items[0]
        assert item["category"] == "neurology"
        assert item["ideal_completion"] == "Migraines are caused by..."
        assert len(item["ref_completions"]) == 1
        assert "What causes migraines?" in item["question"]

    def test_meta_eval_answers(self, sample_oss_eval, sample_meta_eval):
        from grubrics_science.data.adapters.healthbench import HealthBenchAdapter

        adapter = HealthBenchAdapter(
            dataset_path=sample_oss_eval,
            meta_eval_path=sample_meta_eval,
        )
        items = adapter.load_raw()

        answers = adapter._get_answers_for_question("hb_001", items[0])
        assert len(answers) == 2
        assert "model A" in answers[0]
        assert "model B" in answers[1]

    def test_cache_takes_priority(self, sample_oss_eval, sample_meta_eval, sample_cache):
        from grubrics_science.data.adapters.healthbench import HealthBenchAdapter

        adapter = HealthBenchAdapter(
            dataset_path=sample_oss_eval,
            meta_eval_path=sample_meta_eval,
            cache_path=sample_cache,
        )
        items = adapter.load_raw()

        answers = adapter._get_answers_for_question("hb_001", items[0])
        assert answers == ["Good answer", "Bad answer"]

    def test_fallback_to_ideal_completions(self, sample_oss_eval):
        from grubrics_science.data.adapters.healthbench import HealthBenchAdapter

        adapter = HealthBenchAdapter(dataset_path=sample_oss_eval)
        items = adapter.load_raw()

        answers = adapter._get_answers_for_question("hb_001", items[0])
        assert "Migraines are caused by..." in answers
        assert "Reference answer 1" in answers

    def test_to_verl_format(self, sample_oss_eval, sample_cache):
        from grubrics_science.data.adapters.healthbench import HealthBenchAdapter

        adapter = HealthBenchAdapter(
            dataset_path=sample_oss_eval,
            cache_path=sample_cache,
        )
        items = adapter.load_raw()
        verl_row = adapter.to_verl_format(items[0])

        assert verl_row["data_source"] == "healthbench"
        assert "prompt" in verl_row
        assert verl_row["reward_model"]["style"] == "rubric_for_open"
        assert verl_row["extra_info"]["domain_type"] == "open_rubric"
        assert verl_row["extra_info"]["prompt_id"] == "hb_001"
        assert verl_row["extra_info"]["gold_scores"] == [0.85, 0.25]

    def test_to_verl_format_no_cache(self, sample_oss_eval):
        from grubrics_science.data.adapters.healthbench import HealthBenchAdapter

        adapter = HealthBenchAdapter(dataset_path=sample_oss_eval)
        items = adapter.load_raw()
        verl_row = adapter.to_verl_format(items[0])

        assert verl_row["extra_info"]["gold_scores"] == []


# =========================================================================
# Holdout integration
# =========================================================================

class TestHealthBenchHoldout:
    """Holdout split with HealthBench data."""

    @pytest.fixture
    def sample_data(self):
        """Create sample merged data (simulating load_healthbench_with_cache output)."""
        return [
            {
                "question_id": f"hb_{i:03d}",
                "question": f"Question {i}",
                "golden_rubric": f"Points: 5, Item: Criterion {i}",
                "category": "test",
                "answers": [f"ans_a_{i}", f"ans_b_{i}"],
                "gold_scores": [0.8, 0.3],
            }
            for i in range(100)
        ]

    def test_split_sizes(self, sample_data):
        from grubrics_science.evaluation.holdout import split_holdout

        train, holdout = split_holdout(sample_data, holdout_size=20, seed=42)
        assert len(holdout) == 20
        assert len(train) == 80
        assert len(train) + len(holdout) == 100

    def test_split_reproducible(self, sample_data):
        from grubrics_science.evaluation.holdout import split_holdout

        _, h1 = split_holdout(sample_data, holdout_size=20, seed=42)
        _, h2 = split_holdout(sample_data, holdout_size=20, seed=42)
        assert [d["question_id"] for d in h1] == [d["question_id"] for d in h2]

    def test_split_different_seed(self, sample_data):
        from grubrics_science.evaluation.holdout import split_holdout

        _, h1 = split_holdout(sample_data, holdout_size=20, seed=42)
        _, h2 = split_holdout(sample_data, holdout_size=20, seed=123)
        assert [d["question_id"] for d in h1] != [d["question_id"] for d in h2]

    def test_no_overlap(self, sample_data):
        from grubrics_science.evaluation.holdout import split_holdout

        train, holdout = split_holdout(sample_data, holdout_size=20, seed=42)
        train_ids = {d["question_id"] for d in train}
        holdout_ids = {d["question_id"] for d in holdout}
        assert train_ids.isdisjoint(holdout_ids)

    def test_default_holdout_sizes(self):
        from grubrics_science.evaluation.holdout import DEFAULT_HOLDOUT_SIZES

        assert DEFAULT_HOLDOUT_SIZES["healthbench"] == 500
        assert DEFAULT_HOLDOUT_SIZES["frontierscience"] == 12


# =========================================================================
# Registry
# =========================================================================

class TestRegistry:
    """HealthBench is registered in the adapter registry."""

    def test_healthbench_in_registry(self):
        from grubrics_science.data.adapters import ADAPTERS

        assert "healthbench" in ADAPTERS

    def test_get_adapter(self):
        from grubrics_science.data.adapters import get_adapter

        adapter = get_adapter("healthbench")
        assert adapter.data_source == "healthbench"
        assert adapter.domain_type == "open_rubric"

    def test_get_adapter_with_paths(self, tmp_path):
        from grubrics_science.data.adapters import get_adapter

        adapter = get_adapter(
            "healthbench",
            cache_path=str(tmp_path / "cache.jsonl"),
            meta_eval_path=str(tmp_path / "meta.jsonl"),
        )
        assert adapter._cache_path == str(tmp_path / "cache.jsonl")
        assert adapter._meta_eval_path == str(tmp_path / "meta.jsonl")


# =========================================================================
# Validate Judge metrics
# =========================================================================

class TestValidateJudgeMetrics:
    """Unit tests for the metrics in validate_judge.py."""

    def test_perfect_agreement(self):
        from scripts.validate_judge import compute_metrics

        y_true = [True, True, False, False]
        y_pred = [True, True, False, False]
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == 1.0
        assert m["kappa"] == 1.0
        assert m["f1"] == 1.0

    def test_no_agreement(self):
        from scripts.validate_judge import compute_metrics

        y_true = [True, True, False, False]
        y_pred = [False, False, True, True]
        m = compute_metrics(y_true, y_pred)
        assert m["accuracy"] == 0.0

    def test_partial_agreement(self):
        from scripts.validate_judge import compute_metrics

        y_true = [True, True, False, False, True]
        y_pred = [True, False, False, True, True]
        m = compute_metrics(y_true, y_pred)
        assert 0.0 < m["accuracy"] < 1.0
        assert m["n"] == 5

    def test_empty(self):
        from scripts.validate_judge import compute_metrics

        m = compute_metrics([], [])
        assert m["n"] == 0
        assert m["accuracy"] == 0.0
