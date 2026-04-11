"""Input validation utilities."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def validate_number_range(
    value: float, min_val: float, max_val: float, name: str
) -> None:
    """
    Validate that a number is within a range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of parameter for error messages

    Raises:
        ValueError: If value is out of range
    """
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value)}")

    if value < min_val or value > max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")


def validate_positive_int(value: int, name: str) -> None:
    """
    Validate that a value is a positive integer.

    Args:
        value: Value to validate
        name: Name of parameter for error messages

    Raises:
        ValueError: If value is not a positive integer
    """
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {type(value)}")

    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_choice(value: Any, choices: list, name: str) -> None:
    """
    Validate that a value is one of allowed choices.

    Args:
        value: Value to validate
        choices: List of allowed values
        name: Name of parameter for error messages

    Raises:
        ValueError: If value is not in choices
    """
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value}")


def validate_generation_parameters(params: dict) -> None:
    """
    Validate generation parameters.

    Args:
        params: Dictionary of parameters

    Raises:
        ValueError: If any parameter is invalid
    """
    if "guidance_scale" in params:
        validate_number_range(params["guidance_scale"], 1.0, 20.0, "guidance_scale")

    if "strength" in params:
        validate_number_range(params["strength"], 0.0, 1.0, "strength")

    if "num_inference_steps" in params:
        validate_positive_int(params["num_inference_steps"], "num_inference_steps")

    if "height" in params:
        validate_positive_int(params["height"], "height")

    if "width" in params:
        validate_positive_int(params["width"], "width")

    logger.debug("Generation parameters validated")
