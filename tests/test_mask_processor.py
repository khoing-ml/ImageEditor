"""Tests for mask preparation workflow."""

import numpy as np
import pytest
from PIL import Image

from src.services.mask_processor import MaskProcessor


def test_validate_dimensions_mismatch_raises():
    """Dimension mismatch should fail fast."""
    processor = MaskProcessor()
    image = Image.new("RGB", (64, 64), color="black")
    mask = Image.new("L", (32, 32), color=255)

    with pytest.raises(ValueError):
        processor.validate_dimensions(image, mask)


def test_prepare_removes_small_islands():
    """Tiny disconnected components should be removed."""
    processor = MaskProcessor()
    mask_np = np.zeros((64, 64), dtype=np.uint8)
    mask_np[20:40, 20:40] = 255
    mask_np[5:7, 5:7] = 255
    mask = Image.fromarray(mask_np, mode="L")

    prepared = processor.prepare_for_inpainting(
        mask,
        image_size=(64, 64),
        options={
            "opening_kernel": 0,
            "closing_kernel": 0,
            "fill_holes": False,
            "min_component_area": 100,
            "dilate_kernel": 0,
            "erode_kernel": 0,
            "feather_radius": 0,
        },
    )

    prepared_np = np.array(prepared)
    assert prepared_np[6, 6] == 0
    assert prepared_np[25, 25] == 255


def test_prepare_fills_holes():
    """Interior holes should be filled when enabled."""
    processor = MaskProcessor()
    mask_np = np.zeros((64, 64), dtype=np.uint8)
    mask_np[15:50, 15:50] = 255
    mask_np[25:35, 25:35] = 0
    mask = Image.fromarray(mask_np, mode="L")

    prepared = processor.prepare_for_inpainting(
        mask,
        image_size=(64, 64),
        options={
            "opening_kernel": 0,
            "closing_kernel": 0,
            "fill_holes": True,
            "min_component_area": 0,
            "dilate_kernel": 0,
            "erode_kernel": 0,
            "feather_radius": 0,
        },
    )

    prepared_np = np.array(prepared)
    assert prepared_np[30, 30] == 255


def test_prepare_feather_produces_soft_mask_values():
    """Feathering should create soft grayscale values at boundaries."""
    processor = MaskProcessor()
    mask_np = np.zeros((64, 64), dtype=np.uint8)
    mask_np[20:44, 20:44] = 255
    mask = Image.fromarray(mask_np, mode="L")

    prepared = processor.prepare_for_inpainting(
        mask,
        image_size=(64, 64),
        options={
            "opening_kernel": 0,
            "closing_kernel": 0,
            "fill_holes": False,
            "min_component_area": 0,
            "dilate_kernel": 0,
            "erode_kernel": 0,
            "feather_radius": 5,
        },
    )

    prepared_np = np.array(prepared)
    edge_value = prepared_np[20, 20]
    assert 0 < edge_value < 255
