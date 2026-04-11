"""Generation service that orchestrates pipelines and utilities."""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from PIL import Image

from src.pipelines import (
    Text2ImgPipeline,
    Img2ImgPipeline,
    InpaintPipeline,
    ControlNetPipeline,
)
from src.services.image_loader import ImageLoader
from src.services.mask_processor import MaskProcessor
from src.services.prompt_builder import PromptBuilder
from src.services.metadata_logger import MetadataLogger

logger = logging.getLogger(__name__)


class GenerationService:
    """Orchestrates image generation pipelines and supporting services."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the generation service.

        Args:
            config: Dictionary containing configuration
        """
        self.config = config
        self.device = config.get("device", "cuda")
        self.pipelines = {}
        self.image_loader = ImageLoader()
        self.mask_processor = MaskProcessor(config.get("mask_preparation", {}))
        self.prompt_builder = PromptBuilder(config)
        self.metadata_logger = MetadataLogger(config)

    def get_text2img_pipeline(self) -> Text2ImgPipeline:
        """Get or create text-to-image pipeline."""
        if "text2img" not in self.pipelines:
            model_id = self.config.get("text2img_model_id", "runwayml/stable-diffusion-v1-5")
            self.pipelines["text2img"] = Text2ImgPipeline(model_id, self.device)
            self.pipelines["text2img"].load_model()
        return self.pipelines["text2img"]

    def get_img2img_pipeline(self) -> Img2ImgPipeline:
        """Get or create image-to-image pipeline."""
        if "img2img" not in self.pipelines:
            model_id = self.config.get("img2img_model_id", "runwayml/stable-diffusion-v1-5")
            self.pipelines["img2img"] = Img2ImgPipeline(model_id, self.device)
            self.pipelines["img2img"].load_model()
        return self.pipelines["img2img"]

    def get_inpaint_pipeline(self) -> InpaintPipeline:
        """Get or create inpainting pipeline."""
        if "inpaint" not in self.pipelines:
            model_id = self.config.get(
                "inpaint_model_id", "runwayml/stable-diffusion-inpainting"
            )
            self.pipelines["inpaint"] = InpaintPipeline(model_id, self.device)
            self.pipelines["inpaint"].load_model()
        return self.pipelines["inpaint"]

    def get_controlnet_pipeline(self, control_type: str = "canny") -> ControlNetPipeline:
        """Get or create ControlNet pipeline."""
        pipeline_key = f"controlnet_{control_type}"
        if pipeline_key not in self.pipelines:
            base_model = self.config.get("base_model_id", "runwayml/stable-diffusion-v1-5")
            controlnet_model = self.config.get(
                "controlnet_model_id", "lllyasviel/control_canny-fp16"
            )
            self.pipelines[pipeline_key] = ControlNetPipeline(
                base_model, controlnet_model, control_type, self.device
            )
            self.pipelines[pipeline_key].load_model()
        return self.pipelines[pipeline_key]

    def generate_text2img(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Generate images from text prompt."""
        logger.info("Starting text-to-image generation")
        pipeline = self.get_text2img_pipeline()
        images = pipeline.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            **kwargs,
        )
        logger.info(f"Generated {len(images)} images")
        return images

    def generate_img2img(
        self,
        image_path: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Redesign an image."""
        logger.info(f"Starting image-to-image generation from {image_path}")
        image = self.image_loader.load(image_path)
        pipeline = self.get_img2img_pipeline()
        images = pipeline.generate(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            **kwargs,
        )
        logger.info(f"Generated {len(images)} redesigned images")
        return images

    def generate_inpaint(
        self,
        image_path: str,
        mask_path: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Inpaint a masked region."""
        logger.info(f"Starting inpainting from {image_path} with mask {mask_path}")
        image = self.image_loader.load(image_path)
        mask = self.image_loader.load(mask_path, mode="L")  # Load as grayscale

        prepare_mask = kwargs.pop("prepare_mask", self.mask_processor.config.get("enabled", True))
        mask_options = kwargs.pop("mask_options", None)
        prepared_mask_output_path = kwargs.pop("prepared_mask_output_path", None)

        self.mask_processor.validate_dimensions(image, mask)
        if prepare_mask:
            mask = self.mask_processor.prepare_for_inpainting(
                mask,
                image_size=image.size,
                options=mask_options,
            )
            if prepared_mask_output_path:
                self.image_loader.save_image(mask, prepared_mask_output_path)

        pipeline = self.get_inpaint_pipeline()
        images = pipeline.generate(
            image=image,
            mask_image=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            **kwargs,
        )
        logger.info(f"Generated {len(images)} inpainted images")
        return images

    def generate_controlnet(
        self,
        control_image_path: str,
        prompt: str,
        control_type: str = "canny",
        negative_prompt: Optional[str] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """Generate images with ControlNet guidance."""
        logger.info(f"Starting ControlNet generation with {control_type}")
        control_image = self.image_loader.load(control_image_path)
        pipeline = self.get_controlnet_pipeline(control_type)
        images = pipeline.generate(
            prompt=prompt,
            control_image=control_image,
            negative_prompt=negative_prompt,
            **kwargs,
        )
        logger.info(f"Generated {len(images)} ControlNet images")
        return images
