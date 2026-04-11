"""Seed management utilities."""

import logging
import random
import torch
import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    logger.info(f"Random seed set to {seed}")


def get_random_seed() -> int:
    """
    Get a random seed value.

    Returns:
        Random integer seed
    """
    seed = random.randint(0, 2**31 - 1)
    logger.debug(f"Generated random seed: {seed}")
    return seed
