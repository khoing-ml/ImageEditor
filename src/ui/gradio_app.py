"""Gradio web interface for Interior Design Diffusion."""

import logging
import gradio as gr
from pathlib import Path
from typing import List, Tuple

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
                gr.Markdown("Inpainting feature coming soon...")

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
