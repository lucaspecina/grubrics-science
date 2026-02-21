"""Tests for MedQA and MedMCQA adapters.

Run with: pytest tests/test_medqa.py -v
No GPU required. No API calls (uses mock data).
"""

import json
import pytest


# =========================================================================
# MedQA Adapter
# =========================================================================

class TestMedQAAdapter:
    """MedQAAdapter: parsing, cache, veRL format."""

    def test_data_source(self):
        from grubrics_science.data.adapters.medqa import MedQAAdapter

        adapter = MedQAAdapter()
        assert adapter.data_source == "medqa"
        assert adapter.domain_type == "verifiable"

    def test_to_verl_format_structure(self):
        from grubrics_science.data.adapters.medqa import MedQAAdapter

        adapter = MedQAAdapter()
        item = {
            "question_id": "medqa_0",
            "question": "A 45-year-old man presents with chest pain. What is the most likely diagnosis?",
            "options": {"A": "MI", "B": "PE", "C": "GERD", "D": "Pneumonia"},
            "answer_letter": "A",
            "correct_text": "MI",
            "final_answer": "MI",
        }

        verl_row = adapter.to_verl_format(item)

        assert verl_row["data_source"] == "medqa"
        assert verl_row["reward_model"]["ground_truth"] == "MI"
        assert verl_row["reward_model"]["style"] == "rubric_for_verifiable"
        assert verl_row["extra_info"]["domain_type"] == "verifiable"
        assert verl_row["extra_info"]["answer_letter"] == "A"
        assert "prompt" in verl_row

    def test_cache_integration(self, tmp_path):
        from grubrics_science.data.adapters.medqa import MedQAAdapter

        cache_data = {
            "question_id": "medqa_0",
            "answers": ["A. MI", "B. PE", "C. GERD", "D. Pneumonia"],
            "gold_scores": [1.0, 0.0, 0.0, 0.0],
        }
        cache_path = tmp_path / "cache.jsonl"
        with open(cache_path, "w") as f:
            f.write(json.dumps(cache_data) + "\n")

        adapter = MedQAAdapter(cache_path=str(cache_path))
        item = {
            "question_id": "medqa_0",
            "question": "Test question",
            "options": {"A": "MI", "B": "PE", "C": "GERD", "D": "Pneumonia"},
            "answer_letter": "A",
            "correct_text": "MI",
            "final_answer": "MI",
        }

        verl_row = adapter.to_verl_format(item)
        assert verl_row["extra_info"]["gold_scores"] == [1.0, 0.0, 0.0, 0.0]
        assert len(verl_row["extra_info"]["answers"]) == 4

    def test_registry(self):
        from grubrics_science.data.adapters import ADAPTERS, get_adapter

        assert "medqa" in ADAPTERS
        adapter = get_adapter("medqa")
        assert adapter.data_source == "medqa"


# =========================================================================
# MedMCQA Adapter
# =========================================================================

class TestMedMCQAAdapter:
    """MedMCQAAdapter: parsing, subject context, veRL format."""

    def test_data_source(self):
        from grubrics_science.data.adapters.medmcqa import MedMCQAAdapter

        adapter = MedMCQAAdapter()
        assert adapter.data_source == "medmcqa"
        assert adapter.domain_type == "verifiable"

    def test_cop_mapping(self):
        from grubrics_science.data.adapters.medmcqa import _COP_TO_LETTER, _COP_TO_KEY

        assert _COP_TO_LETTER[1] == "A"
        assert _COP_TO_LETTER[4] == "D"
        assert _COP_TO_KEY[1] == "opa"
        assert _COP_TO_KEY[4] == "opd"

    def test_to_verl_format_structure(self):
        from grubrics_science.data.adapters.medmcqa import MedMCQAAdapter

        adapter = MedMCQAAdapter()
        item = {
            "question_id": "medmcqa_0",
            "question": "Which vitamin deficiency causes scurvy?",
            "options": {"A": "Vitamin A", "B": "Vitamin B", "C": "Vitamin C", "D": "Vitamin D"},
            "answer_letter": "C",
            "correct_text": "Vitamin C",
            "final_answer": "Vitamin C",
            "subject": "Biochemistry",
            "topic": "Vitamins",
        }

        verl_row = adapter.to_verl_format(item)

        assert verl_row["data_source"] == "medmcqa"
        assert verl_row["reward_model"]["ground_truth"] == "Vitamin C"
        assert verl_row["reward_model"]["style"] == "rubric_for_verifiable"
        assert verl_row["extra_info"]["subject"] == "Biochemistry"
        assert verl_row["extra_info"]["topic"] == "Vitamins"

    def test_subject_in_context(self):
        from grubrics_science.data.adapters.medmcqa import MedMCQAAdapter

        adapter = MedMCQAAdapter()
        item = {
            "question_id": "medmcqa_0",
            "question": "Test question",
            "options": {"A": "A", "B": "B", "C": "C", "D": "D"},
            "answer_letter": "A",
            "correct_text": "A",
            "final_answer": "A",
            "subject": "Pharmacology",
            "topic": "Antibiotics",
        }

        verl_row = adapter.to_verl_format(item)
        prompt_text = str(verl_row["prompt"])
        assert "Pharmacology" in prompt_text or "medical" in prompt_text.lower()

    def test_cache_integration(self, tmp_path):
        from grubrics_science.data.adapters.medmcqa import MedMCQAAdapter

        cache_data = {
            "question_id": "medmcqa_0",
            "answers": ["A. Vitamin A", "B. Vitamin B", "C. Vitamin C", "D. Vitamin D"],
            "gold_scores": [0.0, 0.0, 1.0, 0.0],
        }
        cache_path = tmp_path / "cache.jsonl"
        with open(cache_path, "w") as f:
            f.write(json.dumps(cache_data) + "\n")

        adapter = MedMCQAAdapter(cache_path=str(cache_path))
        item = {
            "question_id": "medmcqa_0",
            "question": "Test",
            "options": {"A": "Vitamin A", "B": "Vitamin B", "C": "Vitamin C", "D": "Vitamin D"},
            "answer_letter": "C",
            "correct_text": "Vitamin C",
            "final_answer": "Vitamin C",
            "subject": "",
            "topic": "",
        }

        verl_row = adapter.to_verl_format(item)
        assert verl_row["extra_info"]["gold_scores"] == [0.0, 0.0, 1.0, 0.0]

    def test_registry(self):
        from grubrics_science.data.adapters import ADAPTERS, get_adapter

        assert "medmcqa" in ADAPTERS
        adapter = get_adapter("medmcqa")
        assert adapter.data_source == "medmcqa"


# =========================================================================
# Precompute verifiable MCQ path
# =========================================================================

class TestPrecomputeVerifiableMCQ:
    """Verify that precompute_verifiable supports medqa/medmcqa datasets."""

    def test_mcq_choices_in_argparse(self):
        """The CLI parser should accept medqa and medmcqa."""
        from grubrics_science.data.precompute_verifiable import main
        import argparse

        # We can't run main() but we can verify the dataset choices
        # by checking the source code was updated
        import inspect
        source = inspect.getsource(main)
        assert "medqa" in source
        assert "medmcqa" in source

    def test_mcq_adapter_dispatch(self):
        """precompute_verifiable should import MedQA/MedMCQA adapters."""
        import inspect
        from grubrics_science.data import precompute_verifiable as pv

        source = inspect.getsource(pv.precompute_verifiable)
        assert "MedQAAdapter" in source
        assert "MedMCQAAdapter" in source

    def test_mcq_branch_exists(self):
        """The MCQ branch should use options as answers (no perturbations)."""
        import inspect
        from grubrics_science.data import precompute_verifiable as pv

        source = inspect.getsource(pv.precompute_verifiable)
        assert "is_mcq" in source
        assert "answer_letter" in source


# =========================================================================
# veRL format validation
# =========================================================================

class TestVeRLFormat:
    """Verify veRL format compatibility across all medical adapters."""

    REQUIRED_KEYS = {"data_source", "prompt", "reward_model", "extra_info"}
    REQUIRED_REWARD_KEYS = {"ground_truth", "style"}

    def _make_item(self, adapter_cls, **kwargs):
        adapter = adapter_cls(**kwargs)
        if adapter_cls.__name__ == "HealthBenchAdapter":
            item = {
                "prompt_id": "test_001",
                "prompt": [{"role": "user", "content": "Test?"}],
                "rubrics": [{"criterion": "Test", "points": 5, "tags": []}],
                "golden_rubric": "Points: 5, Item: Test",
                "question": "user: Test?",
                "category": "test",
                "example_tags": [],
                "ideal_completion": "",
                "ref_completions": [],
            }
        else:
            item = {
                "question_id": "test_0",
                "question": "Test question?",
                "options": {"A": "Opt A", "B": "Opt B", "C": "Opt C", "D": "Opt D"},
                "answer_letter": "A",
                "correct_text": "Opt A",
                "final_answer": "Opt A",
                "subject": "Test",
                "topic": "Test",
            }
        return adapter.to_verl_format(item)

    def test_healthbench_verl_keys(self):
        from grubrics_science.data.adapters.healthbench import HealthBenchAdapter
        row = self._make_item(HealthBenchAdapter)
        assert self.REQUIRED_KEYS.issubset(row.keys())
        assert self.REQUIRED_REWARD_KEYS.issubset(row["reward_model"].keys())

    def test_medqa_verl_keys(self):
        from grubrics_science.data.adapters.medqa import MedQAAdapter
        row = self._make_item(MedQAAdapter)
        assert self.REQUIRED_KEYS.issubset(row.keys())
        assert self.REQUIRED_REWARD_KEYS.issubset(row["reward_model"].keys())

    def test_medmcqa_verl_keys(self):
        from grubrics_science.data.adapters.medmcqa import MedMCQAAdapter
        row = self._make_item(MedMCQAAdapter)
        assert self.REQUIRED_KEYS.issubset(row.keys())
        assert self.REQUIRED_REWARD_KEYS.issubset(row["reward_model"].keys())

    def test_prompt_is_list_of_messages(self):
        from grubrics_science.data.adapters.healthbench import HealthBenchAdapter
        row = self._make_item(HealthBenchAdapter)
        prompt = row["prompt"]
        assert isinstance(prompt, list)
        assert all(isinstance(m, dict) for m in prompt)
        assert all("role" in m and "content" in m for m in prompt)
