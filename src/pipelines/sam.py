"""SAM-based segmentation pipeline."""

import logging
from typing import List, Optional, Tuple, Union
from PIL import Image

from src.pipelines.base import BasePipeline
from src.services.sam_segmentation import SAMSegmentation

logger = logging.getLogger(__name__)


class SAMPipeline(BasePipeline):
    """Segment Anything Model pipeline for mask generation."""

    def __init__(self, model_type: str = "vit_b", device: str = "cuda"):
        """
        Initialize SAM pipeline.

        Args:
            model_type: Type of SAM model ("vit_h", "vit_l", "vit_b", "mobile_sam")
            device: Device to load model on
        """
        super().__init__(model_id=f"facebook/sam-{model_type}", device=device)
        self.model_type = model_type
        self.sam: Optional[SAMSegmentation] = None

    def load_model(self) -> None:
        """Load the SAM model."""
        logger.info(f"Loading SAM pipeline with model type: {self.model_type}")
        self.sam = SAMSegmentation(model_type=self.model_type, device=self.device)
        self.sam.load_model()
        self.pipeline = self.sam
        logger.info("SAM pipeline loaded successfully")

    def generate(self, **kwargs) -> Union[Image.Image, List[Image.Image]]:
        """
        Generate masks using SAM.

        Supported kwargs:
            - image: PIL Image to segment
            - points: List of (x, y) tuples for point prompts
            - positive_points: bool, treat points as foreground (default: True)
            - box: Tuple of (x_min, y_min, x_max, y_max) for box prompt
            - mode: "points", "box", or "all" (default: "points")

        Returns:
            PIL Image mask or list of masks
        """
        if not self.sam:
            self.load_model()

        image = kwargs.get("image")
        if image is None:
            raise ValueError("'image' parameter is required")

        self.sam.set_image(image)
        mode = kwargs.get("mode", "points")

        if mode == "points":
            points = kwargs.get("points", [])
            if not points:
                raise ValueError("'points' parameter required for point mode")
            positive = kwargs.get("positive_points", True)
            return self.sam.generate_mask_from_points(points, positive_points=positive)

        elif mode == "box":
            box = kwargs.get("box")
            if box is None:
                raise ValueError("'box' parameter required for box mode")
            return self.sam.generate_mask_from_box(box)

        elif mode == "all":
            return self.sam.generate_all_masks()

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'points', 'box', or 'all'")
