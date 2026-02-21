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
        for source in ["gsm8k", "math", "medqa", "medmcqa"]:
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

    def test_parse_batched_response(self):
        from grubrics_science.judge.judge import Judge

        judge = Judge.__new__(Judge)  # skip __init__ (no API client needed)

        # Good response
        response = '{"evaluations": [{"answer_id": "a1", "total_score": 0.65}, {"answer_id": "a2", "total_score": 0.43}]}'
        scores = judge._parse_batched_response(response, 2)
        assert len(scores) == 2
        assert scores[0] == 0.65
        assert scores[1] == 0.43

    def test_parse_batched_response_missing_answer(self):
        from grubrics_science.judge.judge import Judge

        judge = Judge.__new__(Judge)

        # Response only has a1, but we expect 3 answers
        response = '{"evaluations": [{"answer_id": "a1", "total_score": 0.70}]}'
        scores = judge._parse_batched_response(response, 3)
        assert len(scores) == 3
        assert scores[0] == 0.70
        assert scores[1] == 0.0  # missing → 0.0
        assert scores[2] == 0.0

    def test_parse_batched_response_garbage(self):
        from grubrics_science.judge.judge import Judge

        judge = Judge.__new__(Judge)
        scores = judge._parse_batched_response("not json at all", 4)
        assert scores == [0.0, 0.0, 0.0, 0.0]


# =========================================================================
# Integration: adapter → cache → reward (no API, no GPU)
# =========================================================================

class TestAdapterCacheIntegration:
    """Verify FrontierScienceAdapter reads precompute cache and produces
    extra_info that grubrics_reward can consume."""

    @pytest.fixture
    def cache_file(self, tmp_path):
        """Create a minimal precompute cache JSONL."""
        cache = tmp_path / "cache.jsonl"
        entry = {
            "question_id": "0",
            "question": "Derive the partition function for a 2D ideal gas.",
            "subject": "physics",
            "golden_rubric": "Points: 5.0, Item: Correct derivation\nPoints: 5.0, Item: Units",
            "answers": [
                "Answer A: detailed derivation with correct steps...",
                "Answer B: superficial treatment, missing key steps...",
                "Answer C: wrong approach but long text...",
            ],
            "gold_scores": [0.85, 0.45, 0.20],
        }
        cache.write_text(json.dumps(entry, ensure_ascii=False) + "\n")
        return str(cache)

    @pytest.fixture
    def dataset_file(self, tmp_path):
        """Create a minimal FrontierScience dataset JSONL."""
        ds = tmp_path / "test.jsonl"
        record = {
            "problem": "Derive the partition function for a 2D ideal gas.",
            "answer": "Points: 5.0, Item: Correct derivation\nPoints: 5.0, Item: Units",
            "subject": "physics",
        }
        ds.write_text(json.dumps(record, ensure_ascii=False) + "\n")
        return str(ds)

    def test_adapter_loads_cache_into_extra_info(self, cache_file, dataset_file):
        """Adapter reads cache and populates answers + gold_scores in extra_info."""
        from grubrics_science.data.adapters.frontierscience import FrontierScienceAdapter

        adapter = FrontierScienceAdapter(cache_path=cache_file)
        items = adapter.load_raw(path=dataset_file)
        assert len(items) == 1

        row = adapter.to_verl_format(items[0])
        extra = row["extra_info"]

        assert len(extra["answers"]) == 3
        assert len(extra["gold_scores"]) == 3
        assert extra["gold_scores"] == [0.85, 0.45, 0.20]
        assert extra["question"] == "Derive the partition function for a 2D ideal gas."
        assert extra["golden_rubric"] != ""

    def test_adapter_without_cache_gives_empty_lists(self, dataset_file):
        """Without cache, adapter still works but answers/gold_scores are empty."""
        from grubrics_science.data.adapters.frontierscience import FrontierScienceAdapter

        adapter = FrontierScienceAdapter(cache_path=None)
        items = adapter.load_raw(path=dataset_file)
        row = adapter.to_verl_format(items[0])
        extra = row["extra_info"]

        assert extra["answers"] == []
        assert extra["gold_scores"] == []

    def test_adapter_prompt_has_contrastive_excerpts_when_cached(self, cache_file, dataset_file):
        """When cache is available, prompt should include best/worst answer excerpts."""
        from grubrics_science.data.adapters.frontierscience import FrontierScienceAdapter

        adapter = FrontierScienceAdapter(cache_path=cache_file)
        items = adapter.load_raw(path=dataset_file)
        row = adapter.to_verl_format(items[0])

        # prompt is a list of messages
        prompt_text = json.dumps(row["prompt"])
        assert "High-quality answer excerpt" in prompt_text
        assert "Low-quality answer excerpt" in prompt_text

    def test_extra_info_compatible_with_reward_function(self, cache_file, dataset_file):
        """extra_info from adapter has the keys grubrics_reward expects."""
        from grubrics_science.data.adapters.frontierscience import FrontierScienceAdapter

        adapter = FrontierScienceAdapter(cache_path=cache_file)
        items = adapter.load_raw(path=dataset_file)
        row = adapter.to_verl_format(items[0])
        extra = row["extra_info"]

        # These are the keys _reward_open_sync reads from extra_info
        assert "answers" in extra
        assert "gold_scores" in extra
        assert "question" in extra
        assert isinstance(extra["answers"], list)
        assert isinstance(extra["gold_scores"], list)
        assert isinstance(extra["question"], str)
        assert len(extra["answers"]) == len(extra["gold_scores"])

    def test_reward_verifiable_path_no_api(self):
        """Verifiable reward path works end-to-end without any API."""
        from grubrics_science.rewards.grubrics_reward import compute_score

        rubric = (
            "Points: 3.0, Item: Correctly identifies the operation\n"
            "Points: 4.0, Item: Shows clear step-by-step work\n"
            "Points: 3.0, Item: Arrives at the correct final answer"
        )
        score = compute_score(
            data_source="gsm8k",
            solution_str=rubric,
            ground_truth="42",
            extra_info={"question": "What is 6*7?", "domain_type": "verifiable"},
        )
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # well-formed rubric

    def test_reward_open_fallback_when_no_cache(self):
        """Open domain without cache data falls back to format-only reward."""
        from grubrics_science.rewards.grubrics_reward import compute_score

        rubric = (
            "Points: 5.0, Item: Correct derivation of partition function\n"
            "Points: 5.0, Item: Proper treatment of boundary conditions"
        )
        score = compute_score(
            data_source="frontierscience",
            solution_str=rubric,
            extra_info={
                "question": "Derive the partition function.",
                "answers": [],
                "gold_scores": [],
            },
        )
        assert 0.0 <= score <= 1.0

    def test_parquet_roundtrip_preserves_cache_data(self, cache_file, dataset_file, tmp_path):
        """Parquet write/read preserves answers and gold_scores in extra_info."""
        import pandas as pd
        from grubrics_science.data.adapters.frontierscience import FrontierScienceAdapter

        adapter = FrontierScienceAdapter(cache_path=cache_file)
        pq_path = adapter.to_parquet(
            output_dir=str(tmp_path / "parquet"),
            path=dataset_file,
            split="test",
        )

        df = pd.read_parquet(pq_path)
        assert len(df) == 1

        extra = df.iloc[0]["extra_info"]
        # extra_info is stored as a dict in parquet
        assert len(extra["answers"]) == 3
        # parquet may return numpy arrays, so compare element-wise
        assert len(extra["gold_scores"]) == 3
        assert list(extra["gold_scores"]) == [0.85, 0.45, 0.20]
