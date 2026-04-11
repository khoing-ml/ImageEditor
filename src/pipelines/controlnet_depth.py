"""ControlNet Depth pipeline for structure-preserving generation."""

import logging
from typing import List, Optional
from PIL import Image

from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, StableDiffusionControlNetPipeline
from src.pipelines.base import BasePipeline

logger = logging.getLogger(__name__)


class ControlNetDepthPipeline(BasePipeline):
    """ControlNet pipeline with depth conditioning for interior design."""

    def __init__(
        self,
        model_id: str,
        controlnet_model_id: Optional[str] = None,
        device: str = "cuda",
        use_sdxl: bool = True,
    ):
        """
        Initialize ControlNet Depth pipeline.

        Args:
            model_id: Base diffusion model ID
            controlnet_model_id: ControlNet depth model ID
            device: Device to run on
            use_sdxl: Whether to use SDXL pipeline
        """
        super().__init__(model_id, device)
        self.controlnet_model_id = controlnet_model_id or "lllyasviel/control_v11f1p_sd15_depth"
        self.use_sdxl = use_sdxl
        self.controlnet = None

    def load_model(self) -> None:
        """Load the ControlNet depth model and base pipeline."""
        logger.info(f"Loading ControlNet depth model: {self.controlnet_model_id}")

        try:
            self.controlnet = ControlNetModel.from_pretrained(
                self.controlnet_model_id, torch_dtype="auto"
            )
        except Exception as e:
            logger.error(f"Failed to load ControlNet model: {e}")
            logger.warning("Attempting to fall back to alternative model")
            self.controlnet_model_id = "lllyasviel/control_v11f1p_sd15_depth"
            self.controlnet = ControlNetModel.from_pretrained(
                self.controlnet_model_id, torch_dtype="auto"
            )

        logger.info(f"Loading base model with ControlNet: {self.model_id}")

        # Use SDXL if available and requested
        if self.use_sdxl and "sdxl" in self.model_id.lower():
            try:
                self.pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                    self.model_id, controlnet=self.controlnet, torch_dtype="auto"
                )
                logger.info("Using SDXL ControlNet pipeline")
            except Exception as e:
                logger.warning(f"SDXL ControlNet not available: {e}, falling back to SD1.5")
                self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                    self.model_id, controlnet=self.controlnet, torch_dtype="auto"
                )
        else:
            self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                self.model_id, controlnet=self.controlnet, torch_dtype="auto"
            )

        self.pipeline.to(self.device)
        logger.info("ControlNet depth pipeline loaded successfully")

    def generate(
        self,
        prompt: str,
        depth_image: Image.Image,
        prompt_2: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        negative_prompt_2: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Generate images with depth control.

        Args:
            prompt: Text prompt for generation
            depth_image: Depth map image for structural control
            prompt_2: Secondary prompt for SDXL dual prompting
            negative_prompt: Negative prompt
            negative_prompt_2: Secondary negative prompt for SDXL
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            controlnet_conditioning_scale: ControlNet strength (0-1, higher = stronger structure control)
            num_images_per_prompt: Number of images to generate
            seed: Random seed for reproducibility

        Returns:
            List of generated PIL Images
        """
        if not self.pipeline:
            self.load_model()

        logger.info(f"Generating with depth control (strength={controlnet_conditioning_scale})")
        logger.debug(f"Prompt: {prompt}")

        if prompt_2:
            logger.debug(f"Prompt_2: {prompt_2}")

        generator = None
        if seed is not None:
            import torch
            generator = torch.Generator(device=self.device).manual_seed(seed)

        # Check if this is SDXL pipeline
        is_sdxl = hasattr(self.pipeline, 'text_encoder_2')

        if is_sdxl and prompt_2:
            logger.info("Using SDXL dual prompt with depth control")
            result = self.pipeline(
                prompt=prompt,
                prompt_2=prompt_2,
                image=depth_image,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2 or negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
            )
        else:
            # Fallback to single prompt
            if prompt_2:
                combined_prompt = f"{prompt}, {prompt_2}"
                logger.warning("Dual prompt not supported, combining prompts")
            else:
                combined_prompt = prompt

            result = self.pipeline(
                prompt=combined_prompt,
                image=depth_image,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
            )

        logger.info("Depth-controlled generation completed")
        return result.images
