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


def get_adapter(name: str) -> DatasetAdapter:
    """Instantiate an adapter by name.

    Raises:
        KeyError: If ``name`` is not in the registry.
    """
    if name not in ADAPTERS:
        available = ", ".join(sorted(ADAPTERS))
        raise KeyError(f"Unknown adapter '{name}'. Available: {available}")
    return ADAPTERS[name]()
