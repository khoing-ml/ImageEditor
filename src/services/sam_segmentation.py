"""Segment Anything Model (SAM) for automatic mask generation."""

import logging
from typing import Optional, Tuple, List
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

logger = logging.getLogger(__name__)


class SAMSegmentation:
    """Segment Anything Model for mask generation."""

    # Available SAM model sizes on Hugging Face
    AVAILABLE_MODELS = {
        "vit_h": "facebook/sam-vit-huge",
        "vit_l": "facebook/sam-vit-large",
        "vit_b": "facebook/sam-vit-base",
        "mobile_sam": "facebook/mobile-sam",
    }

    def __init__(self, model_type: str = "vit_b", device: str = "cuda"):
        """
        Initialize SAM segmentation.

        Args:
            model_type: Type of SAM model ("vit_h", "vit_l", "vit_b", "mobile_sam")
            device: Device to load model on ("cuda" or "cpu")

        Raises:
            ImportError: If segment-anything is not installed
            ValueError: If model_type is not supported
        """
        if not SAM_AVAILABLE:
            raise ImportError(
                "segment-anything is not installed. "
                "Install it with: pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model type must be one of {list(self.AVAILABLE_MODELS.keys())}")

        self.model_type = model_type
        self.device = device
        self.predictor: Optional[SamPredictor] = None
        self.current_image: Optional[np.ndarray] = None

    def load_model(self) -> None:
        """Load the SAM model."""
        logger.info(f"Loading SAM model: {self.model_type}")
        try:
            checkpoint = self._get_checkpoint_path()
            sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            logger.info(f"SAM model '{self.model_type}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            raise

    def _get_checkpoint_path(self) -> str:
        """Get the checkpoint path, downloading if necessary."""
        # Try to use huggingface_hub to download
        try:
            from huggingface_hub import hf_hub_download
            
            model_name = {
                "vit_h": "sam_vit_h_4b8939.pth",
                "vit_l": "sam_vit_l_0b3195.pth",
                "vit_b": "sam_vit_b_01ec64.pth",
                "mobile_sam": "mobile_sam.pt",
            }
            
            checkpoint = hf_hub_download(
                repo_id=self.AVAILABLE_MODELS[self.model_type],
                filename=model_name[self.model_type],
            )
            return checkpoint
        except Exception as e:
            logger.warning(f"Could not download from Hugging Face: {e}")
            logger.info("Attempting to use local checkpoint or download directly...")
            raise

    def set_image(self, image: Image.Image) -> None:
        """
        Set the image for mask prediction.

        Args:
            image: PIL Image to process
        """
        if not self.predictor:
            self.load_model()

        # Convert PIL Image to numpy array
        image_np = np.array(image.convert("RGB"))
        self.predictor.set_image(image_np)
        self.current_image = image_np
        logger.debug(f"Image set for SAM prediction: {image_np.shape}")

    def generate_mask_from_points(
        self,
        points: List[Tuple[int, int]],
        positive_points: bool = True,
        return_confidence: bool = False,
    ) -> Image.Image:
        """
        Generate mask from point prompts.

        Args:
            points: List of (x, y) coordinates
            positive_points: If True, points are foreground; if False, background
            return_confidence: If True, also return confidence scores

        Returns:
            PIL Image mask (or tuple with confidence scores)
        """
        if not self.predictor or self.current_image is None:
            raise RuntimeError("Image not set. Call set_image() first.")

        points_array = np.array(points, dtype=np.float32)
        labels_array = np.ones(len(points), dtype=np.int32) if positive_points else np.zeros(len(points), dtype=np.int32)

        masks, scores, logits = self.predictor.predict(
            point_coords=points_array,
            point_labels=labels_array,
            multimask_output=False,
        )

        mask = masks[0]
        mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")

        if return_confidence:
            return mask_image, scores[0]
        return mask_image

    def generate_mask_from_box(
        self,
        box: Tuple[int, int, int, int],
        return_confidence: bool = False,
    ) -> Image.Image:
        """
        Generate mask from bounding box.

        Args:
            box: Bounding box as (x_min, y_min, x_max, y_max)
            return_confidence: If True, also return confidence scores

        Returns:
            PIL Image mask (or tuple with confidence scores)
        """
        if not self.predictor or self.current_image is None:
            raise RuntimeError("Image not set. Call set_image() first.")

        box_array = np.array([box], dtype=np.float32)

        masks, scores, logits = self.predictor.predict(
            box=box_array[0],
            multimask_output=False,
        )

        mask = masks[0]
        mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")

        if return_confidence:
            return mask_image, scores[0]
        return mask_image

    def generate_mask_from_text_prompt(
        self,
        prompt: str,
    ) -> Image.Image:
        """
        Generate mask from text prompt using CLIP-based segmentation.

        Note: This requires additional dependencies (openai CLIP + grounding DINO)
        For now, this raises NotImplementedError. Use point or box prompts instead.

        Args:
            prompt: Text description of region to segment

        Returns:
            PIL Image mask
        """
        raise NotImplementedError(
            "Text-based SAM segmentation requires additional models (CLIP, Grounding DINO). "
            "Use point or box prompts instead."
        )

    def generate_all_masks(self) -> List[Image.Image]:
        """
        Generate masks for all objects in the image automatically.

        Returns:
            List of PIL Image masks
        """
        if not self.predictor or self.current_image is None:
            raise RuntimeError("Image not set. Call set_image() first.")

        masks = self.predictor.get_image_embedding()
        
        # Generate masks for the entire image
        masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            multimask_output=True,
        )

        mask_images = [
            Image.fromarray((mask * 255).astype(np.uint8), mode="L")
            for mask in masks
        ]

        logger.info(f"Generated {len(mask_images)} masks from image")
        return mask_images

    def get_device(self) -> str:
        """Get the current device."""
        return self.device

    def to(self, device: str) -> None:
        """Move model to specified device."""
        if self.predictor and hasattr(self.predictor.model, 'to'):
            self.predictor.model.to(device)
            self.device = device
            logger.info(f"SAM model moved to {device}")
