"""Mask preparation utilities for inpainting workflows."""

import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class MaskProcessor:
    """Prepare and refine masks before inpainting."""

    DEFAULT_OPTIONS: Dict[str, Any] = {
        "enabled": True,
        "threshold": 127,
        "opening_kernel": 0,
        "closing_kernel": 3,
        "fill_holes": True,
        "min_component_area": 64,
        "dilate_kernel": 5,
        "erode_kernel": 0,
        "feather_radius": 7,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def validate_dimensions(self, image: Image.Image, mask: Image.Image) -> None:
        """Ensure mask dimensions match source image dimensions exactly."""
        if image.size != mask.size:
            raise ValueError(
                f"Mask size {mask.size} must match image size {image.size} for inpainting"
            )

    def prepare_for_inpainting(
        self,
        mask: Image.Image,
        image_size: Optional[Tuple[int, int]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Image.Image:
        """Validate and refine a raw mask for inpainting input."""
        prepared_options = self._merged_options(options)
        mask_l = mask.convert("L")

        if image_size is not None and mask_l.size != image_size:
            raise ValueError(
                f"Mask size {mask_l.size} must match image size {image_size} for inpainting"
            )

        mask_np = np.array(mask_l, dtype=np.uint8)
        mask_np = self._to_binary(mask_np, prepared_options["threshold"])

        if prepared_options["opening_kernel"] > 0:
            mask_np = self._morph(mask_np, cv2.MORPH_OPEN, prepared_options["opening_kernel"])

        if prepared_options["closing_kernel"] > 0:
            mask_np = self._morph(mask_np, cv2.MORPH_CLOSE, prepared_options["closing_kernel"])

        if prepared_options["fill_holes"]:
            mask_np = self._fill_holes(mask_np)

        if prepared_options["min_component_area"] > 1:
            mask_np = self._remove_small_components(mask_np, prepared_options["min_component_area"])

        if prepared_options["dilate_kernel"] > 0:
            mask_np = self._morph(mask_np, cv2.MORPH_DILATE, prepared_options["dilate_kernel"])

        if prepared_options["erode_kernel"] > 0:
            mask_np = self._morph(mask_np, cv2.MORPH_ERODE, prepared_options["erode_kernel"])

        if prepared_options["feather_radius"] > 0:
            mask_np = self._feather(mask_np, prepared_options["feather_radius"])

        logger.debug("Mask prepared for inpainting")
        return Image.fromarray(mask_np, mode="L")

    def _merged_options(self, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        merged = dict(self.DEFAULT_OPTIONS)
        merged.update(self.config)
        if options:
            merged.update(options)

        merged["threshold"] = int(np.clip(merged["threshold"], 0, 255))
        for key in [
            "opening_kernel",
            "closing_kernel",
            "min_component_area",
            "dilate_kernel",
            "erode_kernel",
            "feather_radius",
        ]:
            merged[key] = max(0, int(merged[key]))

        return merged

    @staticmethod
    def _to_binary(mask_np: np.ndarray, threshold: int) -> np.ndarray:
        return np.where(mask_np >= threshold, 255, 0).astype(np.uint8)

    @staticmethod
    def _morph(mask_np: np.ndarray, operation: int, kernel_size: int) -> np.ndarray:
        kernel_size = max(1, int(kernel_size))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(mask_np, operation, kernel)

    @staticmethod
    def _fill_holes(mask_np: np.ndarray) -> np.ndarray:
        filled = mask_np.copy()
        h, w = filled.shape
        flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(filled, flood_mask, (0, 0), 255)
        holes = cv2.bitwise_not(filled)
        return cv2.bitwise_or(mask_np, holes)

    @staticmethod
    def _remove_small_components(mask_np: np.ndarray, min_area: int) -> np.ndarray:
        binary = (mask_np > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        cleaned = np.zeros_like(mask_np)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_area:
                cleaned[labels == label] = 255

        return cleaned

    @staticmethod
    def _feather(mask_np: np.ndarray, radius: int) -> np.ndarray:
        radius = max(1, int(radius))
        kernel_size = radius * 2 + 1
        return cv2.GaussianBlur(mask_np, (kernel_size, kernel_size), 0)
