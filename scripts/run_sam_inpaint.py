#!/usr/bin/env python3
"""
Example script for SAM-assisted inpainting.

This script generates a mask with SAM and uses it for inpainting.

Usage:
    # Point-based inpainting
    python scripts/run_sam_inpaint.py --image room.jpg --points "100,100 200,200" --prompt "add a modern painting"
    
    # Box-based inpainting
    python scripts/run_sam_inpaint.py --image room.jpg --box "50,50,300,300" --prompt "replace with wooden table"
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services import GenerationService
from src.utils.device import get_device
from src.utils.io import load_config


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def parse_points(points_str: str) -> list:
    """Parse point string like '100,100 200,200' into list of tuples."""
    points = []
    for point in points_str.split():
        x, y = map(int, point.split(","))
        points.append((x, y))
    return points


def parse_box(box_str: str) -> tuple:
    """Parse box string like '50,50,300,300' into tuple."""
    x_min, y_min, x_max, y_max = map(int, box_str.split(","))
    return (x_min, y_min, x_max, y_max)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SAM-assisted inpainting"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inpaint.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Inpainting prompt",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Things to avoid",
    )
    parser.add_argument(
        "--sam-mode",
        type=str,
        choices=["points", "box"],
        default="points",
        help="SAM segmentation mode",
    )
    parser.add_argument(
        "--points",
        type=str,
        help="Points for SAM (format: 'x1,y1 x2,y2 ...')",
    )
    parser.add_argument(
        "--box",
        type=str,
        help="Bounding box for SAM (format: 'x_min,y_min,x_max,y_max')",
    )
    parser.add_argument(
        "--positive-points",
        action="store_true",
        default=True,
        help="Treat SAM points as foreground",
    )
    parser.add_argument(
        "--negative-points",
        action="store_true",
        help="Treat SAM points as background",
    )
    parser.add_argument(
        "--sam-model-type",
        type=str,
        choices=["vit_h", "vit_l", "vit_b", "mobile_sam"],
        default="vit_b",
        help="SAM model type",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/sam_inpaint",
        help="Output directory",
    )
    parser.add_argument(
        "--save-mask",
        action="store_true",
        help="Save the generated mask",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Validate input
    if not Path(args.image).exists():
        logger.error(f"Image not found: {args.image}")
        sys.exit(1)

    logger.info(f"Device: {get_device()}")

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Configuration loaded from {args.config}")

    # Create service
    service = GenerationService(config)
    logger.info("Generation service initialized")

    # Prepare SAM parameters
    sam_kwargs = {"mode": args.sam_mode, "model_type": args.sam_model_type}

    if args.sam_mode == "points":
        if not args.points:
            logger.error("--points parameter required for point mode")
            sys.exit(1)
        sam_kwargs["points"] = parse_points(args.points)
        sam_kwargs["positive_points"] = not args.negative_points

    elif args.sam_mode == "box":
        if not args.box:
            logger.error("--box parameter required for box mode")
            sys.exit(1)
        sam_kwargs["box"] = parse_box(args.box)

    # Prepare inpainting parameters
    inpaint_kwargs = {
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "num_images_per_prompt": args.num_images,
    }

    if args.seed is not None:
        inpaint_kwargs["seed"] = args.seed

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save mask path if requested
    mask_output_path = None
    if args.save_mask:
        mask_output_path = str(output_dir / "mask.png")

    # Generate inpainted images
    try:
        logger.info("Starting SAM-assisted inpainting")
        logger.info(f"Prompt: {args.prompt}")

        images = service.generate_inpaint_with_sam(
            image_path=args.image,
            sam_model_type=args.sam_model_type,
            sam_mode=args.sam_mode,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt if args.negative_prompt else None,
            prepare_mask=True,
            prepared_mask_output_path=mask_output_path,
            **sam_kwargs,
            **inpaint_kwargs,
        )

        # Save images
        for i, image in enumerate(images):
            output_path = output_dir / f"inpainted_{i:02d}.png"
            image.save(output_path)
            logger.info(f"Saved inpainted image to {output_path}")

        logger.info(f"Generated {len(images)} inpainted images")

    except Exception as e:
        logger.error(f"Error during inpainting: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
