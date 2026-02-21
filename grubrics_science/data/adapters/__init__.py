"""Dataset adapter registry.

Flags:
    USE_CONTRASTIVE: env var or kwarg to control contrastive excerpts
        in prompts. Set ``USE_CONTRASTIVE=0`` to disable (ablation A1).
        Default: enabled ("1").
"""

import os
from typing import Dict, Optional, Type

from ..base import DatasetAdapter
from .gsm8k import GSM8KAdapter
from .math_hendrycks import MATHAdapter
from .frontierscience import FrontierScienceAdapter
from .healthbench import HealthBenchAdapter
from .medqa import MedQAAdapter
from .medmcqa import MedMCQAAdapter

ADAPTERS: Dict[str, Type[DatasetAdapter]] = {
    "gsm8k": GSM8KAdapter,
    "math": MATHAdapter,
    "frontierscience": FrontierScienceAdapter,
    "healthbench": HealthBenchAdapter,
    "medqa": MedQAAdapter,
    "medmcqa": MedMCQAAdapter,
}


def use_contrastive() -> bool:
    """Check if contrastive excerpts should be included in prompts.

    Reads ``USE_CONTRASTIVE`` env var. Default: True.
    """
    return os.environ.get("USE_CONTRASTIVE", "1") == "1"


def get_adapter(
    name: str,
    cache_path: Optional[str] = None,
    **kwargs,
) -> DatasetAdapter:
    """Instantiate an adapter by name.

    Args:
        name: Adapter name (e.g. "gsm8k", "math", "frontierscience",
            "healthbench", "medqa", "medmcqa").
        cache_path: Optional path to precompute cache JSONL. Adapters
            that support it will load answers + gold_scores from cache.
        **kwargs: Extra keyword arguments passed to the adapter constructor
            (e.g. ``meta_eval_path`` for HealthBenchAdapter).

    Raises:
        KeyError: If ``name`` is not in the registry.
    """
    if name not in ADAPTERS:
        available = ", ".join(sorted(ADAPTERS))
        raise KeyError(f"Unknown adapter '{name}'. Available: {available}")

    cls = ADAPTERS[name]
    # Pass cache_path and any extra kwargs to adapters that support them
    try:
        return cls(cache_path=cache_path, **kwargs)
    except TypeError:
        try:
            return cls(cache_path=cache_path)
        except TypeError:
            return cls()
