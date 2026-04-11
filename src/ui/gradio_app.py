"""Gradio web interface for Interior Design Diffusion."""

import logging
import tempfile
import uuid
import gradio as gr
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
from PIL import Image

from src.services import GenerationService
from src.utils.io import load_config

logger = logging.getLogger(__name__)


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

        # Primary mask extraction from layer alpha channels when available.
        layer_mask = np.zeros((background.height, background.width), dtype=np.uint8)
        for layer in layers:
            if layer is None:
                continue
            layer_arr = np.array(layer.convert("RGBA"))
            if layer_arr.shape[:2] != layer_mask.shape:
                continue
            alpha = layer_arr[..., 3]
            layer_mask = np.maximum(layer_mask, alpha)

        # Fallback mask extraction from visual differences between composite and background.
        bg_arr = np.array(background, dtype=np.int16)
        comp_arr = np.array(composite, dtype=np.int16)
        diff = np.abs(comp_arr - bg_arr).sum(axis=2)
        diff_mask = (diff > 12).astype(np.uint8) * 255

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
    ):
        """Generate inpainted images from hand-drawn mask."""
        try:
            base_image, mask_image = _extract_background_and_mask(editor_data)

            # Ensure mask contains editable region.
            if np.array(mask_image).max() == 0:
                return (
                    None,
                    mask_image,
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
            run_mask_path = run_output_dir / "mask.png"
            base_image.save(run_input_path)
            mask_image.save(run_mask_path)

            images = service.generate_inpaint(
                image_path=str(input_path),
                mask_path=str(mask_path),
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_images_per_prompt=num_images,
                guidance_scale=guidance_scale,
                num_inference_steps=inference_steps,
                seed=seed if seed and seed >= 0 else None,
            )

            download_paths: List[str] = [str(run_input_path), str(run_mask_path)]
            for i, image in enumerate(images, start=1):
                out_path = run_output_dir / f"output_{i:02d}.png"
                image.save(out_path)
                download_paths.append(str(out_path))

            status = f"Inpainting successful! Saved artifacts to {run_output_dir}"
            return images, mask_image, status, download_paths
        except Exception as e:
            logger.error(f"Inpainting failed: {e}")
            return None, None, f"Error: {str(e)}", []

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

                        inpaint_button = gr.Button("Inpaint", size="lg")

                    with gr.Column():
                        inpaint_output = gr.Gallery(label="Inpainted Images", columns=2)
                        inpaint_mask_preview = gr.Image(
                            label="Detected Binary Mask (white = edited)",
                            type="pil",
                            image_mode="L",
                        )
                        inpaint_downloads = gr.File(
                            label="Download Artifacts (input, mask, outputs)",
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
                    ],
                    outputs=[
                        inpaint_output,
                        inpaint_mask_preview,
                        inpaint_status,
                        inpaint_downloads,
                    ],
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
