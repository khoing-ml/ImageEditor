"""Base pipeline class for all diffusion models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BasePipeline(ABC):
    """
    Abstract base class for all diffusion pipelines.

    This class defines the interface that all pipeline implementations must follow.
    """

    def __init__(self, model_id: str, device: str = "cuda"):
        """
        Initialize the base pipeline.

        Args:
            model_id: Hugging Face model identifier
            device: Device to load model on ("cuda" or "cpu")
        """
        self.model_id = model_id
        self.device = device
        self.pipeline = None

    @abstractmethod
    def load_model(self) -> None:
        """Load the diffusion model from Hugging Face."""
        pass

    @abstractmethod
    def generate(self, **kwargs) -> Any:
        """
        Generate images using the pipeline.

        Returns:
            Pipeline output (typically PIL Image or list of images)
        """
        pass

    def to(self, device: str) -> None:
        """Move pipeline to specified device."""
        if self.pipeline:
            self.pipeline.to(device)
            self.device = device

    def enable_attention_slicing(self) -> None:
        """Enable attention slicing for lower memory usage."""
        if self.pipeline:
            self.pipeline.enable_attention_slicing()

    def disable_attention_slicing(self) -> None:
        """Disable attention slicing for faster generation."""
        if self.pipeline:
            self.pipeline.disable_attention_slicing()
