"""Inpainting script."""

import argparse
import logging
import sys
from pathlib import Path

from src.services import GenerationService
from src.utils.io import load_config, ensure_dir, get_timestamp_filename
from src.utils.seed import set_seed


def main():
    """Inpaint a masked region in an image."""
    parser = argparse.ArgumentParser(
        description="Inpaint a masked region in an image"
    )
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--mask", type=str, required=True, help="Mask image path")
    parser.add_argument("--prompt", type=str, required=True, help="Inpainting prompt")
    parser.add_argument(
        "--negative-prompt", type=str, default=None, help="Negative prompt"
    )
    parser.add_argument(
        "--num-images", type=int, default=1, help="Number of inpainted variants"
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=7.5, help="Guidance scale"
    )
    parser.add_argument(
        "--inference-steps", type=int, default=50, help="Number of inference steps"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/inpaint", help="Output directory"
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

        # Generate inpainted images
        logger.info("Starting inpainting operation...")
        images = service.generate_inpaint(
            image_path=args.image,
            mask_path=args.mask,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.inference_steps,
            seed=args.seed,
        )

        # Save images
        for i, image in enumerate(images):
            filename = get_timestamp_filename(f"inpaint_{i:03d}", ".png")
            output_path = output_dir / filename
            image.save(output_path)
            logger.info(f"Saved inpainted image to {output_path}")

        logger.info(f"Successfully generated and saved {len(images)} image(s)")

    except Exception as e:
        logger.error(f"Inpainting failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
