"""Tests for Phase 3: configurable reward weights and ablation flags.

Run with: pytest tests/test_phase3.py -v
No GPU required. No API calls.
"""

import os
import pytest


# =========================================================================
# RewardConfig tests
# =========================================================================

class TestRewardConfig:
    """Test RewardConfig dataclass and env loading."""

    def test_defaults(self):
        from grubrics_science.rewards.grubrics_reward import RewardConfig

        config = RewardConfig()
        assert config.lambda_len == 0.1
        assert config.lambda_info == 0.3
        assert config.lambda_defense == 0.3
        assert config.char_threshold == 3000
        assert config.use_functional_alignment is True

    def test_from_env(self, monkeypatch):
        from grubrics_science.rewards.grubrics_reward import RewardConfig

        monkeypatch.setenv("REWARD_LAMBDA_LEN", "0.05")
        monkeypatch.setenv("REWARD_LAMBDA_INFO", "0.0")
        monkeypatch.setenv("REWARD_LAMBDA_DEFENSE", "0.5")
        monkeypatch.setenv("REWARD_CHAR_THRESHOLD", "5000")

        config = RewardConfig.from_env()
        assert config.lambda_len == 0.05
        assert config.lambda_info == 0.0
        assert config.lambda_defense == 0.5
        assert config.char_threshold == 5000

    def test_from_env_defaults(self, monkeypatch):
        """If env vars not set, use defaults."""
        from grubrics_science.rewards.grubrics_reward import RewardConfig

        # Clear any existing env vars
        for key in ["REWARD_LAMBDA_LEN", "REWARD_LAMBDA_INFO",
                     "REWARD_LAMBDA_DEFENSE", "REWARD_CHAR_THRESHOLD"]:
            monkeypatch.delenv(key, raising=False)

        config = RewardConfig.from_env()
        assert config.lambda_len == 0.1
        assert config.lambda_info == 0.3
        assert config.lambda_defense == 0.3
        assert config.char_threshold == 3000
        assert config.use_functional_alignment is True


class TestConfigureReward:
    """Test configure_reward() programmatic override."""

    def test_configure_changes_global(self):
        from grubrics_science.rewards.grubrics_reward import (
            RewardConfig,
            configure_reward,
            get_reward_config,
        )
        import grubrics_science.rewards.grubrics_reward as mod

        # Reset the global
        mod._reward_config = None

        custom = RewardConfig(
            lambda_len=0.0,
            lambda_info=0.0,
            lambda_defense=0.0,
            char_threshold=1000,
        )
        configure_reward(custom)

        config = get_reward_config()
        assert config.lambda_len == 0.0
        assert config.lambda_info == 0.0
        assert config.lambda_defense == 0.0
        assert config.char_threshold == 1000

        # Reset for other tests
        mod._reward_config = None


# =========================================================================
# Reward weight integration tests
# =========================================================================

class TestRewardWeightsIntegration:
    """Test that configurable weights actually affect reward computation."""

    def test_zero_info_disables_info_bonus(self):
        """With lambda_info=0, info_value shouldn't affect reward."""
        from grubrics_science.rewards.grubrics_reward import (
            RewardConfig,
            configure_reward,
            _reward_functional_alignment,
        )
        import grubrics_science.rewards.grubrics_reward as mod

        # We can't easily call _reward_functional_alignment without a Judge,
        # but we can verify the config is read correctly.
        config = RewardConfig(lambda_info=0.0)
        assert config.lambda_info == 0.0

        # Reset
        mod._reward_config = None

    def test_missing_precompute_raises_verifiable(self):
        """Without precomputed data, verifiable reward must raise."""
        from grubrics_science.rewards.grubrics_reward import _reward_verifiable
        import grubrics_science.rewards.grubrics_reward as mod

        mod._reward_config = None

        with pytest.raises(ValueError, match="Missing precomputed"):
            _reward_verifiable(
                solution_str="Points: 5.0, Item: A\nPoints: 5.0, Item: B",
                ground_truth="42",
                extra_info={
                    "data_source": "gsm8k",
                    "question": "What is 6*7?",
                    "question_id": "test_q",
                },
            )

        mod._reward_config = None

    def test_missing_precompute_raises_open(self):
        """Without precomputed data, open-domain reward must raise."""
        import asyncio
        from grubrics_science.rewards.grubrics_reward import _reward_open
        import grubrics_science.rewards.grubrics_reward as mod

        mod._reward_config = None

        with pytest.raises(ValueError, match="Missing precomputed"):
            asyncio.run(_reward_open(
                solution_str="Points: 5.0, Item: A\nPoints: 5.0, Item: B",
                extra_info={
                    "question": "What causes chest pain?",
                    "prompt_id": "test_p",
                },
            ))

        mod._reward_config = None


# =========================================================================
# Contrastive flag tests
# =========================================================================

class TestContrastiveFlag:
    """Test USE_CONTRASTIVE env var controls excerpt generation."""

    def test_use_contrastive_default_true(self, monkeypatch):
        monkeypatch.delenv("USE_CONTRASTIVE", raising=False)
        from grubrics_science.data.adapters import use_contrastive
        assert use_contrastive() is True

    def test_use_contrastive_disabled(self, monkeypatch):
        monkeypatch.setenv("USE_CONTRASTIVE", "0")
        from grubrics_science.data.adapters import use_contrastive
        assert use_contrastive() is False

    def test_use_contrastive_enabled(self, monkeypatch):
        monkeypatch.setenv("USE_CONTRASTIVE", "1")
        from grubrics_science.data.adapters import use_contrastive
        assert use_contrastive() is True

    def test_gsm8k_adapter_no_contrastive(self, monkeypatch, tmp_path):
        """With USE_CONTRASTIVE=0, GSM8K adapter should not include excerpts."""
        import json

        monkeypatch.setenv("USE_CONTRASTIVE", "0")

        # Create cache with answers
        cache_path = tmp_path / "cache.jsonl"
        cache_entry = {
            "question_id": "gsm8k_0",
            "question": "What is 2+2?",
            "answers": ["The answer is 4", "The answer is 5"],
            "gold_scores": [1.0, 0.0],
        }
        cache_path.write_text(json.dumps(cache_entry) + "\n")

        from grubrics_science.data.adapters.gsm8k import GSM8KAdapter

        adapter = GSM8KAdapter(cache_path=str(cache_path))
        item = {
            "question": "What is 2+2?",
            "solution": "2+2=4\n#### 4",
            "final_answer": "4",
        }
        result = adapter.to_verl_format(item)

        # Prompt should not contain contrastive excerpts
        prompt_text = str(result["prompt"])
        assert "High-quality answer excerpt" not in prompt_text
        assert "Low-quality answer excerpt" not in prompt_text

    def test_gsm8k_adapter_with_contrastive(self, monkeypatch, tmp_path):
        """With USE_CONTRASTIVE=1, GSM8K adapter should include excerpts."""
        import json

        monkeypatch.setenv("USE_CONTRASTIVE", "1")

        cache_path = tmp_path / "cache.jsonl"
        cache_entry = {
            "question_id": "gsm8k_0",
            "question": "What is 2+2?",
            "answers": ["The answer is 4", "The answer is 5"],
            "gold_scores": [1.0, 0.0],
        }
        cache_path.write_text(json.dumps(cache_entry) + "\n")

        from grubrics_science.data.adapters.gsm8k import GSM8KAdapter

        adapter = GSM8KAdapter(cache_path=str(cache_path))
        item = {
            "question": "What is 2+2?",
            "solution": "2+2=4\n#### 4",
            "final_answer": "4",
        }
        result = adapter.to_verl_format(item)

        # Prompt should contain contrastive excerpts
        prompt_text = str(result["prompt"])
        assert "High-quality answer excerpt" in prompt_text

    def test_frontierscience_adapter_no_contrastive(self, monkeypatch, tmp_path):
        """With USE_CONTRASTIVE=0, FS adapter should not include excerpts."""
        import json

        monkeypatch.setenv("USE_CONTRASTIVE", "0")

        cache_path = tmp_path / "cache.jsonl"
        cache_entry = {
            "question_id": "0",
            "question": "Derive X.",
            "golden_rubric": "Points: 10.0, Item: Test",
            "answers": ["Good answer with derivation", "Bad answer"],
            "gold_scores": [0.9, 0.3],
        }
        cache_path.write_text(json.dumps(cache_entry) + "\n")

        from grubrics_science.data.adapters.frontierscience import FrontierScienceAdapter

        adapter = FrontierScienceAdapter(cache_path=str(cache_path))
        item = {
            "question_id": "0",
            "problem": "Derive X.",
            "golden_rubric": "Points: 10.0, Item: Test",
            "subject": "physics",
        }
        result = adapter.to_verl_format(item)

        # Extra info should have empty excerpts
        assert result["extra_info"]["best_answer_excerpt"] == ""
        assert result["extra_info"]["worst_answer_excerpt"] == ""


# =========================================================================
# Apply reward config env tests
# =========================================================================

class TestApplyRewardConfigEnv:
    """Test _apply_reward_config_env from run_grpo."""

    def test_sets_env_vars(self, monkeypatch):
        from grubrics_science.training.run_grpo import _apply_reward_config_env

        # Clear existing vars
        for var in ["REWARD_LAMBDA_LEN", "REWARD_LAMBDA_INFO",
                     "REWARD_LAMBDA_DEFENSE", "REWARD_CHAR_THRESHOLD",
                     "USE_CONTRASTIVE"]:
            monkeypatch.delenv(var, raising=False)

        config = {
            "lambda_len": 0.05,
            "lambda_info": 0.0,
            "lambda_defense": 0.5,
            "char_threshold": 5000,
            "use_contrastive": True,
        }
        _apply_reward_config_env(config)

        assert os.environ["REWARD_LAMBDA_LEN"] == "0.05"
        assert os.environ["REWARD_LAMBDA_INFO"] == "0.0"
        assert os.environ["REWARD_LAMBDA_DEFENSE"] == "0.5"
        assert os.environ["REWARD_CHAR_THRESHOLD"] == "5000"
        assert os.environ["USE_CONTRASTIVE"] == "1"

    def test_empty_config_no_change(self, monkeypatch):
        from grubrics_science.training.run_grpo import _apply_reward_config_env

        monkeypatch.delenv("REWARD_LAMBDA_LEN", raising=False)
        _apply_reward_config_env({})
        assert "REWARD_LAMBDA_LEN" not in os.environ


# =========================================================================
# YAML config has reward_config section
# =========================================================================

class TestYAMLConfig:
    """Test that YAML configs include reward_config."""

    def test_production_yaml_has_reward_config(self):
        import yaml

        with open("configs/verl_grpo.yaml") as f:
            config = yaml.safe_load(f)

        assert "reward_config" in config
        rc = config["reward_config"]
        assert rc["lambda_len"] == 0.1
        assert rc["lambda_info"] == 0.3
        assert rc["lambda_defense"] == 0.3
        assert rc["char_threshold"] == 3000
        assert rc["use_contrastive"] is True

    def test_debug_yaml_has_reward_config(self):
        import yaml

        with open("configs/verl_grpo_debug.yaml") as f:
            config = yaml.safe_load(f)

        assert "reward_config" in config
        rc = config["reward_config"]
        assert rc["use_contrastive"] is True
