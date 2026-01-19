"""Seed management utilities."""

import random
import numpy as np
import torch


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_deterministic_seed(base_seed: int, *args) -> int:
    """Generate a deterministic seed from base seed and additional args."""
    # Hash-like function to combine seeds
    combined = base_seed
    for arg in args:
        combined = hash((combined, arg)) & 0x7FFFFFFF  # Keep positive
    return combined

