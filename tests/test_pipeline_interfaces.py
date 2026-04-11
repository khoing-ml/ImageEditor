"""Tests for pipeline interfaces."""

import pytest
from src.pipelines import (
    BasePipeline,
    Text2ImgPipeline,
    Img2ImgPipeline,
    InpaintPipeline,
)


def test_text2img_initialization():
    """Test Text2Img pipeline initialization."""
    pipeline = Text2ImgPipeline("runwayml/stable-diffusion-v1-5")
    assert pipeline.model_id == "runwayml/stable-diffusion-v1-5"
    assert pipeline.device == "cuda"


def test_img2img_initialization():
    """Test Img2Img pipeline initialization."""
    pipeline = Img2ImgPipeline("runwayml/stable-diffusion-v1-5")
    assert pipeline.model_id == "runwayml/stable-diffusion-v1-5"
    assert pipeline.pipeline is None  # Model not loaded yet


def test_inpaint_initialization():
    """Test Inpaint pipeline initialization."""
    pipeline = InpaintPipeline("runwayml/stable-diffusion-inpainting")
    assert pipeline.model_id == "runwayml/stable-diffusion-inpainting"
    assert pipeline.device == "cuda"


def test_device_switching():
    """Test switching device."""
    pipeline = Text2ImgPipeline("runwayml/stable-diffusion-v1-5", device="cpu")
    assert pipeline.device == "cpu"

    pipeline.to("cuda")
    # Note: actual device movement requires loaded model
    assert pipeline.device == "cuda"
