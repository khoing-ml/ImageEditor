"""Inpainting pipeline implementation."""

import logging
from typing import List, Optional
from PIL import Image

from diffusers import StableDiffusionInpaintPipeline
from src.pipelines.base import BasePipeline

logger = logging.getLogger(__name__)


class InpaintPipeline(BasePipeline):
    """Inpainting pipeline for masked region replacement."""

    def load_model(self) -> None:
        """Load the inpainting model."""
        logger.info(f"Loading Inpaint model: {self.model_id}")
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            self.model_id, torch_dtype="auto"
        )
        self.pipeline.to(self.device)
        logger.info("Inpaint model loaded successfully")

    def generate(
        self,
        image: Image.Image,
        mask_image: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Inpaint a masked region in an image.

        Args:
            image: Input PIL Image
            mask_image: Mask image (white = inpaint, black = preserve)
            prompt: Text prompt for inpainting
            negative_prompt: Things to avoid
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            num_images_per_prompt: Number of images to generate
            seed: Random seed for reproducibility

        Returns:
            List of inpainted PIL Images
        """
        if not self.pipeline:
            self.load_model()

        logger.info("Starting inpainting operation")
        logger.debug(f"Prompt: {prompt}")

        generator = None
        if seed is not None:
            import torch

            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
        )

        logger.info("Inpainting completed")
        return result.images
