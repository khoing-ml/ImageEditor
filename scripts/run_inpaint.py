"""Inpainting script."""

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
        "--no-prepare-mask",
        action="store_true",
        help="Skip mask cleanup/dilation/feathering before inpainting",
    )
    parser.add_argument(
        "--mask-threshold", type=int, default=127, help="Mask binarization threshold (0-255)"
    )
    parser.add_argument(
        "--mask-opening-kernel", type=int, default=0, help="Opening kernel size"
    )
    parser.add_argument(
        "--mask-closing-kernel", type=int, default=3, help="Closing kernel size"
    )
    parser.add_argument(
        "--mask-min-component-area",
        type=int,
        default=64,
        help="Remove connected mask regions smaller than this area",
    )
    parser.add_argument(
        "--mask-dilate-kernel", type=int, default=5, help="Dilation kernel size"
    )
    parser.add_argument(
        "--mask-erode-kernel", type=int, default=0, help="Erosion kernel size"
    )
    parser.add_argument(
        "--mask-feather-radius", type=int, default=7, help="Gaussian feather radius"
    )
    parser.add_argument(
        "--prepared-mask-output",
        type=str,
        default=None,
        help="Optional path to save prepared mask preview",
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
            prepare_mask=not args.no_prepare_mask,
            mask_options={
                "threshold": args.mask_threshold,
                "opening_kernel": args.mask_opening_kernel,
                "closing_kernel": args.mask_closing_kernel,
                "min_component_area": args.mask_min_component_area,
                "dilate_kernel": args.mask_dilate_kernel,
                "erode_kernel": args.mask_erode_kernel,
                "feather_radius": args.mask_feather_radius,
            },
            prepared_mask_output_path=args.prepared_mask_output,
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
