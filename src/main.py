"""Main entry point for the Interior Design Diffusion application."""

import argparse
import logging
from pathlib import Path

from src.services import GenerationService
from src.utils.device import get_device
from src.utils.io import load_config


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interior Design Diffusion - Generate and edit interior design images"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["text2img", "img2img", "inpaint", "controlnet"],
        default="text2img",
        help="Pipeline type to execute",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info(f"Starting Interior Design Diffusion - Pipeline: {args.pipeline}")
    logger.info(f"Device: {get_device()}")

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Configuration loaded from {args.config}")

    # Initialize generation service
    service = GenerationService(config)
    logger.info("Generation service initialized")

    logger.info("Ready for pipeline execution")


if __name__ == "__main__":
    main()
