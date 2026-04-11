"""ControlNet pipeline implementation."""

import logging
from typing import List, Optional
from PIL import Image

from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from src.pipelines.base import BasePipeline

logger = logging.getLogger(__name__)


class ControlNetPipeline(BasePipeline):
    """ControlNet pipeline for conditional image generation."""

    def __init__(
        self,
        model_id: str,
        controlnet_model_id: str,
        control_type: str = "canny",
        device: str = "cuda",
    ):
        """
        Initialize ControlNet pipeline.

        Args:
            model_id: Base model identifier
            controlnet_model_id: ControlNet model identifier
            control_type: Type of control ("canny" or "depth")
            device: Device to load model on
        """
        super().__init__(model_id, device)
        self.controlnet_model_id = controlnet_model_id
        self.control_type = control_type

    def load_model(self) -> None:
        """Load the ControlNet model and base pipeline."""
        logger.info(f"Loading ControlNet model: {self.controlnet_model_id}")
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_model_id, torch_dtype="auto"
        )

        logger.info(f"Loading base model with ControlNet: {self.model_id}")
        self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_id, controlnet=controlnet, torch_dtype="auto"
        )
        self.pipeline.to(self.device)
        logger.info("ControlNet pipeline loaded successfully")

    def generate(
        self,
        prompt: str,
        control_image: Image.Image,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Generate images with ControlNet conditioning.

        Args:
            prompt: Text prompt for generation
            control_image: Control image (canny edge map or depth map)
            negative_prompt: Things to avoid
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            controlnet_conditioning_scale: ControlNet conditioning strength
            num_images_per_prompt: Number of images to generate
            seed: Random seed for reproducibility

        Returns:
            List of generated PIL Images
        """
        if not self.pipeline:
            self.load_model()

        logger.info(f"Generating with ControlNet ({self.control_type})")
        logger.debug(f"Prompt: {prompt}")

        generator = None
        if seed is not None:
            import torch

            generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipeline(
            prompt=prompt,
            image=control_image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
        )

        logger.info("ControlNet generation completed")
        return result.images
