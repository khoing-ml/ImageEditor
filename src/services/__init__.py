"""Services module initialization with lazy exports.

Avoid eager imports here to prevent circular-import chains between
`src.services` and `src.pipelines` during package initialization.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "GenerationService",
    "ImageLoader",
    "MaskProcessor",
    "PromptBuilder",
    "PromptPair",
    "IPAdapterConfig",
    "PromptWeightingParser",
    "DualPrompt",
    "DualPromptBuilder",
    "InteriorDesignPromptBuilder",
    "MetadataLogger",
    "SAMSegmentation",
]


_SYMBOL_TO_MODULE = {
    "GenerationService": "src.services.generation_service",
    "ImageLoader": "src.services.image_loader",
    "MaskProcessor": "src.services.mask_processor",
    "PromptBuilder": "src.services.prompt_builder",
    "PromptPair": "src.services.prompt_builder",
    "IPAdapterConfig": "src.services.prompt_builder",
    "PromptWeightingParser": "src.services.prompt_weighting",
    "DualPrompt": "src.services.prompt_weighting",
    "DualPromptBuilder": "src.services.prompt_weighting",
    "InteriorDesignPromptBuilder": "src.services.prompt_weighting",
    "MetadataLogger": "src.services.metadata_logger",
    "SAMSegmentation": "src.services.sam_segmentation",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve service exports on first access."""
    module_path = _SYMBOL_TO_MODULE.get(name)
    if module_path is None:
        raise AttributeError(f"module 'src.services' has no attribute '{name}'")

    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value
