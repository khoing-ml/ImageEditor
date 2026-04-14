"""Gradio web interface for Interior Design Diffusion."""

import logging
import tempfile
import uuid
import gradio as gr
import sys
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv

import numpy as np
from PIL import Image

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services import GenerationService
from src.utils.io import load_config

logger = logging.getLogger(__name__)

load_dotenv()


def create_interface(config_path: str = "configs/default.yaml") -> gr.Blocks:
    """
    Create Gradio interface for the application.

    Args:
        config_path: Path to configuration file

    Returns:
        Gradio Blocks interface
    """
    config = load_config(config_path)
    service = GenerationService(config)
    temp_dir = Path(tempfile.gettempdir()) / "image_editor_gradio"
    temp_dir.mkdir(parents=True, exist_ok=True)
    inpaint_output_root = Path(config.get("inpaint_output_dir", "outputs/inpaint"))
    inpaint_output_root.mkdir(parents=True, exist_ok=True)

    def _extract_background_and_mask(editor_data: Any) -> tuple[Image.Image, Image.Image]:
        """Extract base image and binary mask from Gradio ImageEditor payload."""
        if editor_data is None:
            raise ValueError("Please upload an image and paint the mask area first.")

        if isinstance(editor_data, Image.Image):
            # Fallback mode: no drawing data available.
            base = editor_data.convert("RGB")
            empty_mask = Image.new("L", base.size, 0)
            return base, empty_mask

        if not isinstance(editor_data, dict):
            raise ValueError("Unsupported editor input format.")

        background = editor_data.get("background")
        composite = editor_data.get("composite")
        layers = editor_data.get("layers") or []

        if background is None and composite is None:
            raise ValueError("Please upload an image before inpainting.")

        if background is None:
            background = composite
        if composite is None:
            composite = background

        background = background.convert("RGB")
        composite = composite.convert("RGB")
        bg_arr = np.array(background, dtype=np.int16)

        # Primary mask extraction from drawing layers.
        # Some Gradio versions emit layers with opaque alpha across the canvas,
        # so we combine multiple signals to isolate real brush strokes.
        layer_mask = np.zeros((background.height, background.width), dtype=np.uint8)
        white_brush_mask = np.zeros((background.height, background.width), dtype=np.uint8)
        for layer in layers:
            if layer is None:
                continue
            layer_arr = np.array(layer.convert("RGBA"))
            if layer_arr.shape[:2] != layer_mask.shape:
                continue
            alpha = layer_arr[..., 3].astype(np.uint8)
            layer_rgb = layer_arr[..., :3].astype(np.int16)
            color_delta = np.abs(layer_rgb - bg_arr).sum(axis=2)
            painted = np.where((alpha > 0) & (color_delta > 8), 255, 0).astype(np.uint8)
            layer_mask = np.maximum(layer_mask, painted)

            # Brush color is constrained to white in this UI, so explicit white
            # detection is a robust signal even when alpha is noisy.
            white_pixels = (
                (layer_arr[..., 0] >= 240)
                & (layer_arr[..., 1] >= 240)
                & (layer_arr[..., 2] >= 240)
            )
            painted_white = np.where((alpha > 0) & white_pixels, 255, 0).astype(np.uint8)
            white_brush_mask = np.maximum(white_brush_mask, painted_white)

        # Fallback mask extraction from visual differences between composite and background.
        comp_arr = np.array(composite, dtype=np.int16)
        diff = np.abs(comp_arr - bg_arr).sum(axis=2)
        diff_mask = (diff > 24).astype(np.uint8) * 255

        white_coverage = float((white_brush_mask > 0).mean())
        layer_coverage = float((layer_mask > 0).mean())
        diff_coverage = float((diff_mask > 0).mean())

        # Prefer explicit white brush signal when available.
        if 0.0 < white_coverage < 0.99:
            merged = white_brush_mask
            # Add layer-derived pixels only when they are not near full-canvas noise.
            if 0.0 < layer_coverage < 0.95:
                merged = np.maximum(merged, layer_mask)
        # If diff-based mask explodes to near-full-canvas while layer data is sane,
        # trust layer-derived mask to avoid all-white artifacts.
        elif layer_coverage > 0.0 and diff_coverage > 0.98 and layer_coverage < 0.95:
            merged = layer_mask
        else:
            merged = np.maximum(layer_mask, diff_mask).astype(np.uint8)
        binary_mask = Image.fromarray(merged, mode="L").point(lambda p: 255 if p > 10 else 0)

        return background, binary_mask

    def generate_text2img(
        prompt: str,
        negative_prompt: str,
        num_images: int,
        guidance_scale: float,
        inference_steps: int,
        height: int,
        width: int,
    ):
        """Generate images from text."""
        try:
            images = service.generate_text2img(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_images_per_prompt=num_images,
                guidance_scale=guidance_scale,
                num_inference_steps=inference_steps,
                height=height,
                width=width,
            )
            return images, "Generation successful!"
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None, f"Error: {str(e)}"

    def generate_inpaint(
        editor_data: Dict[str, Any],
        prompt: str,
        negative_prompt: str,
        num_images: int,
        guidance_scale: float,
        inference_steps: int,
        seed: int,
        mask_extract_threshold: int,
        invert_mask: bool,
        prepare_mask: bool,
        opening_kernel: int,
        closing_kernel: int,
        fill_holes: bool,
        min_component_area: int,
        dilate_kernel: int,
        erode_kernel: int,
        feather_radius: int,
    ):
        """Generate inpainted images from hand-drawn mask."""
        try:
            base_image, mask_image = _extract_background_and_mask(editor_data)

            mask_arr = np.array(mask_image, dtype=np.uint8)
            mask_arr = np.where(mask_arr > mask_extract_threshold, 255, 0).astype(np.uint8)
            if invert_mask:
                mask_arr = 255 - mask_arr
            mask_image = Image.fromarray(mask_arr, mode="L")

            white_ratio = float((mask_arr == 255).mean())
            if white_ratio > 0.995:
                logger.warning(
                    "Extracted mask is almost fully white (ratio=%.4f). Continuing with best-effort mask.",
                    white_ratio,
                )

            # Ensure mask contains editable region.
            if np.array(mask_image).max() == 0:
                return (
                    None,
                    mask_image,
                    None,
                    "Error: No painted mask detected. Draw over area to edit.",
                    [],
                )

            run_id = uuid.uuid4().hex[:8]
            input_path = temp_dir / f"inpaint_input_{run_id}.png"
            mask_path = temp_dir / f"inpaint_mask_{run_id}.png"
            base_image.save(input_path)
            mask_image.save(mask_path)

            run_output_dir = inpaint_output_root / f"run_{run_id}"
            run_output_dir.mkdir(parents=True, exist_ok=True)
            run_input_path = run_output_dir / "input.png"
            run_mask_path = run_output_dir / "mask_raw.png"
            run_prepared_mask_path = run_output_dir / "mask_prepared.png"
            base_image.save(run_input_path)
            mask_image.save(run_mask_path)

            images = service.generate_inpaint(
                image_path=str(input_path),
                mask_path=str(mask_path),
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                prepare_mask=prepare_mask,
                mask_options={
                    "opening_kernel": opening_kernel,
                    "closing_kernel": closing_kernel,
                    "fill_holes": fill_holes,
                    "min_component_area": min_component_area,
                    "dilate_kernel": dilate_kernel,
                    "erode_kernel": erode_kernel,
                    "feather_radius": feather_radius,
                },
                prepared_mask_output_path=str(run_prepared_mask_path),
                num_images_per_prompt=num_images,
                guidance_scale=guidance_scale,
                num_inference_steps=inference_steps,
                seed=seed if seed and seed >= 0 else None,
            )

            prepared_mask_preview = mask_image
            if prepare_mask and run_prepared_mask_path.exists():
                prepared_mask_preview = Image.open(run_prepared_mask_path).convert("L")

            download_paths: List[str] = [str(run_input_path), str(run_mask_path)]
            if run_prepared_mask_path.exists():
                download_paths.append(str(run_prepared_mask_path))
            for i, image in enumerate(images, start=1):
                out_path = run_output_dir / f"output_{i:02d}.png"
                image.save(out_path)
                download_paths.append(str(out_path))

            status = f"Inpainting successful! Saved artifacts to {run_output_dir}"
            if white_ratio > 0.995:
                status += (
                    " | Warning: mask was almost fully white; extraction may be unstable. "
                    "Try increasing Mask Extract Threshold if results look off."
                )
            return images, mask_image, prepared_mask_preview, status, download_paths
        except Exception as e:
            logger.error(f"Inpainting failed: {e}")
            return None, None, None, f"Error: {str(e)}", []

    def update_inpaint_tool_sizes(brush_size: int, eraser_size: int, canvas_height: int):
        """Update brush, eraser, and canvas size for the inpainting editor."""
        return gr.update(
            brush=gr.Brush(
                colors=["#ffffff"],
                default_color="#ffffff",
                default_size=int(brush_size),
            ),
            eraser=gr.Eraser(default_size=int(eraser_size)),
            height=int(canvas_height),
        )

    # Create interface with tabs
    with gr.Blocks(title="Interior Design Diffusion") as demo:
        gr.Markdown(
            """
            # Interior Design Diffusion
            Generate and edit interior design images using AI.
            """
        )

        with gr.Tabs():
            # Text-to-Image tab
            with gr.Tab("Text-to-Image"):
                with gr.Row():
                    with gr.Column():
                        t2i_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe your desired interior design...",
                            lines=3,
                        )
                        t2i_negative = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="Things to avoid...",
                            lines=2,
                        )

                        with gr.Row():
                            t2i_guidance = gr.Slider(
                                label="Guidance Scale",
                                minimum=1.0,
                                maximum=20.0,
                                value=7.5,
                                step=0.1,
                            )
                            t2i_steps = gr.Slider(
                                label="Inference Steps",
                                minimum=10,
                                maximum=100,
                                value=50,
                                step=10,
                            )

                        with gr.Row():
                            t2i_height = gr.Slider(
                                label="Height",
                                minimum=256,
                                maximum=1024,
                                value=768,
                                step=64,
                            )
                            t2i_width = gr.Slider(
                                label="Width",
                                minimum=256,
                                maximum=1024,
                                value=768,
                                step=64,
                            )

                        t2i_num_images = gr.Slider(
                            label="Number of Images",
                            minimum=1,
                            maximum=4,
                            value=1,
                            step=1,
                        )

                        t2i_button = gr.Button("Generate", size="lg")

                    with gr.Column():
                        t2i_output = gr.Gallery(label="Generated Images", columns=2)
                        t2i_status = gr.Textbox(label="Status", interactive=False)

                t2i_button.click(
                    generate_text2img,
                    inputs=[
                        t2i_prompt,
                        t2i_negative,
                        t2i_num_images,
                        t2i_guidance,
                        t2i_steps,
                        t2i_height,
                        t2i_width,
                    ],
                    outputs=[t2i_output, t2i_status],
                )

            # Coming soon placeholders
            with gr.Tab("Image-to-Image"):
                gr.Markdown("Image-to-Image redesign coming soon...")

            with gr.Tab("Inpainting"):
                gr.Markdown(
                    "Upload an image, paint over the area to edit, then generate inpainted results."
                )
                with gr.Row():
                    with gr.Column():
                        inpaint_editor = gr.ImageEditor(
                            label="Image + Hand-Drawn Mask",
                            type="pil",
                            image_mode="RGB",
                            brush=gr.Brush(colors=["#ffffff"], default_color="#ffffff", default_size=24),
                            eraser=gr.Eraser(default_size=20),
                            height=512,
                        )
                        inpaint_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe how the masked area should be modified...",
                            lines=3,
                        )
                        inpaint_negative = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="Things to avoid...",
                            lines=2,
                        )
                        with gr.Row():
                            inpaint_guidance = gr.Slider(
                                label="Guidance Scale",
                                minimum=1.0,
                                maximum=20.0,
                                value=7.5,
                                step=0.1,
                            )
                            inpaint_steps = gr.Slider(
                                label="Inference Steps",
                                minimum=10,
                                maximum=100,
                                value=50,
                                step=5,
                            )
                        with gr.Row():
                            inpaint_num_images = gr.Slider(
                                label="Number of Images",
                                minimum=1,
                                maximum=4,
                                value=1,
                                step=1,
                            )
                            inpaint_seed = gr.Number(
                                label="Seed (-1 for random)",
                                value=-1,
                                precision=0,
                            )

                        with gr.Accordion("Mask Tool Settings", open=False):
                            with gr.Row():
                                inpaint_brush_size = gr.Slider(
                                    label="Brush Size",
                                    minimum=1,
                                    maximum=128,
                                    value=24,
                                    step=1,
                                )
                                inpaint_eraser_size = gr.Slider(
                                    label="Eraser Size",
                                    minimum=1,
                                    maximum=128,
                                    value=20,
                                    step=1,
                                )
                            inpaint_canvas_height = gr.Slider(
                                label="Canvas Height",
                                minimum=256,
                                maximum=1024,
                                value=512,
                                step=32,
                            )
                            inpaint_apply_tool_sizes = gr.Button("Apply Tool Sizes")

                            inpaint_mask_extract_threshold = gr.Slider(
                                label="Mask Extract Threshold",
                                minimum=0,
                                maximum=255,
                                value=10,
                                step=1,
                            )
                            inpaint_invert_mask = gr.Checkbox(
                                label="Invert Mask (edit unpainted area)",
                                value=False,
                            )
                            inpaint_prepare_mask = gr.Checkbox(
                                label="Enable Mask Cleanup + Feathering",
                                value=True,
                            )
                            with gr.Row():
                                inpaint_opening_kernel = gr.Slider(
                                    label="Opening Kernel",
                                    minimum=0,
                                    maximum=31,
                                    value=0,
                                    step=1,
                                )
                                inpaint_closing_kernel = gr.Slider(
                                    label="Closing Kernel",
                                    minimum=0,
                                    maximum=31,
                                    value=3,
                                    step=1,
                                )
                            inpaint_fill_holes = gr.Checkbox(
                                label="Fill Holes",
                                value=True,
                            )
                            inpaint_min_component_area = gr.Slider(
                                label="Min Component Area",
                                minimum=0,
                                maximum=5000,
                                value=64,
                                step=1,
                            )
                            with gr.Row():
                                inpaint_dilate_kernel = gr.Slider(
                                    label="Dilation Kernel",
                                    minimum=0,
                                    maximum=31,
                                    value=5,
                                    step=1,
                                )
                                inpaint_erode_kernel = gr.Slider(
                                    label="Erosion Kernel",
                                    minimum=0,
                                    maximum=31,
                                    value=0,
                                    step=1,
                                )
                            inpaint_feather_radius = gr.Slider(
                                label="Feather Radius",
                                minimum=0,
                                maximum=25,
                                value=7,
                                step=1,
                            )

                        inpaint_button = gr.Button("Inpaint", size="lg")

                    with gr.Column():
                        inpaint_output = gr.Gallery(label="Inpainted Images", columns=2)
                        inpaint_mask_preview_raw = gr.Image(
                            label="Raw Mask (white = edited)",
                            type="pil",
                            image_mode="L",
                        )
                        inpaint_mask_preview_prepared = gr.Image(
                            label="Prepared Mask Used for Inpainting",
                            type="pil",
                            image_mode="L",
                        )
                        inpaint_downloads = gr.File(
                            label="Download Artifacts (input, raw mask, prepared mask, outputs)",
                            file_count="multiple",
                        )
                        inpaint_status = gr.Textbox(label="Status", interactive=False)

                inpaint_button.click(
                    generate_inpaint,
                    inputs=[
                        inpaint_editor,
                        inpaint_prompt,
                        inpaint_negative,
                        inpaint_num_images,
                        inpaint_guidance,
                        inpaint_steps,
                        inpaint_seed,
                        inpaint_mask_extract_threshold,
                        inpaint_invert_mask,
                        inpaint_prepare_mask,
                        inpaint_opening_kernel,
                        inpaint_closing_kernel,
                        inpaint_fill_holes,
                        inpaint_min_component_area,
                        inpaint_dilate_kernel,
                        inpaint_erode_kernel,
                        inpaint_feather_radius,
                    ],
                    outputs=[
                        inpaint_output,
                        inpaint_mask_preview_raw,
                        inpaint_mask_preview_prepared,
                        inpaint_status,
                        inpaint_downloads,
                    ],
                )

                inpaint_apply_tool_sizes.click(
                    update_inpaint_tool_sizes,
                    inputs=[inpaint_brush_size, inpaint_eraser_size, inpaint_canvas_height],
                    outputs=[inpaint_editor],
                )

            with gr.Tab("ControlNet"):
                gr.Markdown("ControlNet feature coming soon...")

    return demo


def launch(config_path: str = "configs/default.yaml", **kwargs):
    """
    Launch the Gradio interface.

    Args:
        config_path: Path to configuration file
        **kwargs: Additional arguments to pass to launch()
    """
    demo = create_interface(config_path)
    demo.launch(**kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    launch(share=False)
