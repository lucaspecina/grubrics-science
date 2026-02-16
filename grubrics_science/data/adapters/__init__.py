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
from .verifiable_math import VerifiableMathAdapter

ADAPTERS: Dict[str, Type[DatasetAdapter]] = {
    "gsm8k": GSM8KAdapter,
    "math": MATHAdapter,
    "frontierscience": FrontierScienceAdapter,
    "olympiad_math": VerifiableMathAdapter,
}


def use_contrastive() -> bool:
    """Check if contrastive excerpts should be included in prompts.

    Reads ``USE_CONTRASTIVE`` env var. Default: True.
    """
    return os.environ.get("USE_CONTRASTIVE", "1") == "1"


def get_adapter(
    name: str,
    cache_path: Optional[str] = None,
) -> DatasetAdapter:
    """Instantiate an adapter by name.

    Args:
        name: Adapter name (e.g. "gsm8k", "math", "frontierscience").
        cache_path: Optional path to precompute cache JSONL. Adapters
            that support it will load answers + gold_scores from cache.

    Raises:
        KeyError: If ``name`` is not in the registry.
    """
    if name not in ADAPTERS:
        available = ", ".join(sorted(ADAPTERS))
        raise KeyError(f"Unknown adapter '{name}'. Available: {available}")

    cls = ADAPTERS[name]
    # Pass cache_path to adapters that support it
    try:
        return cls(cache_path=cache_path)
    except TypeError:
        # Adapter doesn't accept cache_path (e.g. VerifiableMathAdapter)
        return cls()
