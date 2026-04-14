#!/usr/bin/env python3
"""
Example script for using SAM to generate masks.

Usage:
    python scripts/run_sam.py --image examples/sample_inputs/room.jpg --mode points --points 100,100 200,200
    python scripts/run_sam.py --image examples/sample_inputs/room.jpg --mode box --box 50,50,300,300
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
        description="SAM (Segment Anything Model) mask generation"
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
        default="configs/sam.yaml",
        help="Path to SAM config file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["points", "box", "all"],
        default="points",
        help="Segmentation mode",
    )
    parser.add_argument(
        "--points",
        type=str,
        help="Points for segmentation (format: 'x1,y1 x2,y2 ...')",
    )
    parser.add_argument(
        "--positive-points",
        action="store_true",
        default=True,
        help="Treat points as foreground (default: true)",
    )
    parser.add_argument(
        "--negative-points",
        action="store_true",
        help="Treat points as background",
    )
    parser.add_argument(
        "--box",
        type=str,
        help="Bounding box for segmentation (format: 'x_min,y_min,x_max,y_max')",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["vit_h", "vit_l", "vit_b", "mobile_sam"],
        default="vit_b",
        help="SAM model type",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/sam_mask.png",
        help="Output path for generated mask",
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
    sam_kwargs = {"mode": args.mode, "model_type": args.model_type}

    if args.mode == "points":
        if not args.points:
            logger.error("--points parameter required for point mode")
            sys.exit(1)
        sam_kwargs["points"] = parse_points(args.points)
        sam_kwargs["positive_points"] = not args.negative_points

    elif args.mode == "box":
        if not args.box:
            logger.error("--box parameter required for box mode")
            sys.exit(1)
        sam_kwargs["box"] = parse_box(args.box)

    # Generate mask
    try:
        logger.info(f"Generating mask with mode: {args.mode}")
        mask = service.generate_mask_with_sam(
            image_path=args.image,
            **sam_kwargs,
        )

        # Create output directory if needed
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save mask
        mask.save(output_path)
        logger.info(f"Mask saved to {args.output}")

    except Exception as e:
        logger.error(f"Error generating mask: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
