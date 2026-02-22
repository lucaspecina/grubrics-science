"""Curriculum training library functions.

The CLI entry point is now at the repo root: ``python run_grpo.py``.
This module re-exports the key functions so existing imports keep working.
"""

# Re-export from root run_grpo — but since root is not a package,
# keep the actual library helpers here for importability.

import logging
import os
from typing import Optional, List

from .curriculum import CurriculumScheduler, parse_phases

logger = logging.getLogger(__name__)


def _apply_reward_config_env(reward_config: dict) -> None:
    """Bridge YAML reward_config values to env vars read by grubrics_reward.py."""
    env_map = {
        "lambda_len": "REWARD_LAMBDA_LEN",
        "lambda_info": "REWARD_LAMBDA_INFO",
        "lambda_defense": "REWARD_LAMBDA_DEFENSE",
        "char_threshold": "REWARD_CHAR_THRESHOLD",
        "use_contrastive": "USE_CONTRASTIVE",
    }
    for key, env_var in env_map.items():
        if key in reward_config:
            val = reward_config[key]
            if isinstance(val, bool):
                val = "1" if val else "0"
            os.environ[env_var] = str(val)
            logger.info("  reward env: %s=%s", env_var, os.environ[env_var])


__all__ = [
    "CurriculumScheduler",
    "parse_phases",
    "_apply_reward_config_env",
]
