"""Tests for validators."""

import pytest
from src.utils.validators import (
    validate_number_range,
    validate_positive_int,
    validate_choice,
)


def test_validate_number_range_valid():
    """Test valid number range."""
    validate_number_range(5.0, 0.0, 10.0, "test_param")


def test_validate_number_range_invalid():
    """Test invalid number range."""
    with pytest.raises(ValueError):
        validate_number_range(15.0, 0.0, 10.0, "test_param")


def test_validate_positive_int_valid():
    """Test valid positive integer."""
    validate_positive_int(5, "test_param")


def test_validate_positive_int_invalid():
    """Test invalid positive integer."""
    with pytest.raises(ValueError):
        validate_positive_int(-5, "test_param")


def test_validate_choice_valid():
    """Test valid choice."""
    validate_choice("option1", ["option1", "option2"], "test_param")


def test_validate_choice_invalid():
    """Test invalid choice."""
    with pytest.raises(ValueError):
        validate_choice("invalid", ["option1", "option2"], "test_param")
