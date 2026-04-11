"""Text-to-Image pipeline implementation."""

import logging
from typing import List, Optional
from PIL import Image

from diffusers import StableDiffusionPipeline
from src.pipelines.base import BasePipeline

logger = logging.getLogger(__name__)


class Text2ImgPipeline(BasePipeline):
    """Text-to-Image generation pipeline using Stable Diffusion."""

    def _is_sdxl_model(self) -> bool:
        """Detect whether the configured model id is an SDXL checkpoint."""
        model = self.model_id.lower()
        return (
            "sdxl" in model
            or "stable-diffusion-xl" in model
            or "sd-xl" in model
        )

    def load_model(self) -> None:
        """Load the text-to-image model."""
        logger.info(f"Loading Text2Img model: {self.model_id}")

        # Use SDXL pipeline when the checkpoint is SDXL.
        if self._is_sdxl_model():
            try:
                from diffusers import StableDiffusionXLPipeline
                self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                    self.model_id, torch_dtype="auto"
                )
                logger.info("Using SDXL pipeline")
            except ImportError:
                raise RuntimeError(
                    "SDXL model detected but StableDiffusionXLPipeline is unavailable in "
                    "the installed diffusers version. Please upgrade diffusers."
                )
        else:
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id, torch_dtype="auto"
            )
        
        self.pipeline.to(self.device)
        logger.info("Text2Img model loaded successfully")

    def generate(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        height: int = 768,
        width: int = 768,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Generate images from text prompts.

        Args:
            prompt: Text prompt for generation (or main prompt for SDXL)
            prompt_2: Secondary prompt for SDXL dual prompting
            negative_prompt: Things to avoid in generation
            negative_prompt_2: Secondary negative prompt for SDXL
            height: Image height in pixels
            width: Image width in pixels
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            num_images_per_prompt: Number of images to generate
            seed: Random seed for reproducibility

        Returns:
            List of PIL Images
        """
        if not self.pipeline:
            self.load_model()

        logger.info(f"Generating {num_images_per_prompt} images from prompt")
        logger.debug(f"Prompt: {prompt}")
        
        if prompt_2:
            logger.debug(f"Prompt_2: {prompt_2}")

        generator = None
        if seed is not None:
            import torch

            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Check if this is SDXL pipeline (has prompt_2 support)
        is_sdxl = hasattr(self.pipeline, 'text_encoder_2')

        if is_sdxl and prompt_2:
            logger.info("Using SDXL dual prompt mode")
            result = self.pipeline(
                prompt=prompt,
                prompt_2=prompt_2,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2 or negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
            )
        else:
            # Fallback to single prompt mode
            if prompt_2:
                # Combine prompts if dual not supported
                combined_prompt = f"{prompt}, {prompt_2}"
                logger.warning("Dual prompt not supported, combining into single prompt")
            else:
                combined_prompt = prompt

            result = self.pipeline(
                prompt=combined_prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
            )

        logger.info("Image generation completed")
        return result.images
