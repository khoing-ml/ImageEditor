"""Image-to-Image pipeline implementation."""

import logging
from typing import List, Optional
from PIL import Image

from diffusers import StableDiffusionImg2ImgPipeline
from src.pipelines.base import BasePipeline

logger = logging.getLogger(__name__)


class Img2ImgPipeline(BasePipeline):
    """Image-to-Image redesign pipeline using Stable Diffusion."""

    def load_model(self) -> None:
        """Load the image-to-image model."""
        logger.info(f"Loading Img2Img model: {self.model_id}")
        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_id, torch_dtype="auto"
        )
        self.pipeline.to(self.device)
        logger.info("Img2Img model loaded successfully")

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        strength: float = 0.75,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Redesign an image based on a text prompt.

        Args:
            image: Input PIL Image
            prompt: Text prompt for redesign
            negative_prompt: Things to avoid
            strength: How much to transform the image (0.0-1.0)
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            num_images_per_prompt: Number of images to generate
            seed: Random seed for reproducibility

        Returns:
            List of redesigned PIL Images
        """
        if not self.pipeline:
            self.load_model()

        logger.info(f"Transforming image with strength {strength}")
        logger.debug(f"Prompt: {prompt}")

        generator = None
        if seed is not None:
            import torch

            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipeline(
            prompt=prompt,
            image=image,
            negative_prompt=negative_prompt,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
        )

        logger.info("Image transformation completed")
        return result.images
