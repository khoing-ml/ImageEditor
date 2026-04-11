"""File I/O utilities."""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary containing configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info(f"Loading configuration from {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if not config:
        config = {}

    logger.info(f"Configuration loaded with {len(config)} keys")
    return config


def save_config(config: Dict[str, Any], output_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Path to save to
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving configuration to {output_path}")

    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info("Configuration saved successfully")


def ensure_dir(dir_path: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        dir_path: Path to directory

    Returns:
        Path object
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp_filename(base_name: str, extension: str = ".png") -> str:
    """
    Generate a timestamped filename.

    Args:
        base_name: Base name for the file
        extension: File extension

    Returns:
        Timestamped filename
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}{extension}"
