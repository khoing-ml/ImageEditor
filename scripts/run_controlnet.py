"""ControlNet generation script."""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is importable when script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services import GenerationService
from src.utils.io import load_config, ensure_dir, get_timestamp_filename
from src.utils.seed import set_seed


def main():
    """Generate images with ControlNet guidance."""
    parser = argparse.ArgumentParser(
        description="Generate images with ControlNet structural guidance"
    )
    parser.add_argument(
        "--control-image", type=str, required=True, help="Control image path (edge/depth map)"
    )
    parser.add_argument("--prompt", type=str, required=True, help="Generation prompt")
    parser.add_argument(
        "--control-type", type=str, default="canny", help="Control type (canny or depth)"
    )
    parser.add_argument(
        "--negative-prompt", type=str, default=None, help="Negative prompt"
    )
    parser.add_argument(
        "--num-images", type=int, default=1, help="Number of images to generate"
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=7.5, help="Guidance scale"
    )
    parser.add_argument(
        "--controlnet-scale", type=float, default=1.0, help="ControlNet conditioning scale"
    )
    parser.add_argument(
        "--inference-steps", type=int, default=50, help="Number of inference steps"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/controlnet", help="Output directory"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Config file path"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        # Load configuration
        config = load_config(args.config)

        # Set seed if provided
        if args.seed is not None:
            set_seed(args.seed)

        # Create output directory
        output_dir = ensure_dir(args.output_dir)

        # Initialize generation service
        service = GenerationService(config)

        # Generate images with ControlNet
        logger.info(f"Generating with ControlNet ({args.control_type})...")
        images = service.generate_controlnet(
            control_image_path=args.control_image,
            prompt=args.prompt,
            control_type=args.control_type,
            negative_prompt=args.negative_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_scale,
            num_inference_steps=args.inference_steps,
            seed=args.seed,
        )

        # Save images
        for i, image in enumerate(images):
            filename = get_timestamp_filename(f"controlnet_{i:03d}", ".png")
            output_path = output_dir / filename
            image.save(output_path)
            logger.info(f"Saved ControlNet image to {output_path}")

        logger.info(f"Successfully generated and saved {len(images)} image(s)")

    except Exception as e:
        logger.error(f"ControlNet generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
