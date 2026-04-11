"""Device management utilities."""

import logging
import torch

logger = logging.getLogger(__name__)


def get_device() -> str:
    """
    Get the current device.

    Returns:
        "cuda" if GPU available, otherwise "cpu"
    """
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"CUDA device available: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("CUDA not available, using CPU")

    return device


def get_device_info() -> dict:
    """
    Get detailed device information.

    Returns:
        Dictionary with device info
    """
    info = {
        "device": get_device(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = torch.cuda.get_device_capability(0)
        try:
            info["cuda_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        except:
            pass

    return info


def print_device_info() -> None:
    """Print device information."""
    info = get_device_info()
    logger.info("Device Information:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")
