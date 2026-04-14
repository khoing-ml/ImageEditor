"""Pipeline module initialization with lazy exports."""

from importlib import import_module
from typing import Any

__all__ = [
    "BasePipeline",
    "Text2ImgPipeline",
    "Img2ImgPipeline",
    "InpaintPipeline",
    "ControlNetPipeline",
    "SAMPipeline",
]


_SYMBOL_TO_MODULE = {
    "BasePipeline": "src.pipelines.base",
    "Text2ImgPipeline": "src.pipelines.text2img",
    "Img2ImgPipeline": "src.pipelines.img2img",
    "InpaintPipeline": "src.pipelines.inpaint",
    "ControlNetPipeline": "src.pipelines.controlnet",
    "SAMPipeline": "src.pipelines.sam",
}


def __getattr__(name: str) -> Any:
    """Lazily resolve pipeline exports on first access."""
    module_path = _SYMBOL_TO_MODULE.get(name)
    if module_path is None:
        raise AttributeError(f"module 'src.pipelines' has no attribute '{name}'")

    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value
