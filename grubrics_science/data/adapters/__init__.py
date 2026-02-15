"""Dataset adapter registry."""

from typing import Dict, Type

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


def get_adapter(name: str, cache_path: str = None) -> DatasetAdapter:
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
