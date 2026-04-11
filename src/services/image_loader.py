"""Image loading and preprocessing service."""

import logging
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)


class ImageLoader:
    """Handle image loading and validation."""

    SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def load(self, image_path: str, mode: str = "RGB") -> Image.Image:
        """
        Load an image and convert to RGB.

        Args:
            image_path: Path to the image file
            mode: Image mode to convert to ("RGB" or "L" for grayscale)

        Returns:
            PIL Image

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image format is not supported
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {path.suffix}")

        logger.info(f"Loading image from {image_path}")
        image = Image.open(path)

        if mode == "RGB":
            if image.mode != "RGB":
                image = image.convert("RGB")
        elif mode == "L":
            if image.mode != "L":
                image = image.convert("L")
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        logger.info(f"Image loaded successfully - Size: {image.size}, Mode: {image.mode}")
        return image

    def validate_image_size(
        self, image: Image.Image, expected_size: tuple = None
    ) -> bool:
        """
        Validate image dimensions.

        Args:
            image: PIL Image
            expected_size: Expected (width, height) tuple

        Returns:
            True if valid, raises ValueError otherwise
        """
        width, height = image.size

        if width < 64 or height < 64:
            raise ValueError(f"Image too small: {image.size}")

        if width > 2048 or height > 2048:
            logger.warning(f"Large image detected: {image.size}")

        if expected_size and image.size != expected_size:
            logger.warning(f"Image size mismatch: {image.size} vs {expected_size}")

        return True

    def save_image(self, image: Image.Image, output_path: str, quality: int = 95) -> None:
        """
        Save an image to disk.

        Args:
            image: PIL Image
            output_path: Path to save to
            quality: JPEG quality (1-100)
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix.lower() in {".jpg", ".jpeg"}:
            image.save(path, quality=quality)
        else:
            image.save(path)

        logger.info(f"Image saved to {output_path}")
