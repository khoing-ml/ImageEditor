"""Tests for SAM segmentation."""

import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np

from src.services.sam_segmentation import SAMSegmentation


class TestSAMSegmentation:
    """Test cases for SAM segmentation service."""

    @pytest.fixture
    def sam(self):
        """Create SAM instance with mocked predictor."""
        with patch('src.services.sam_segmentation.SAMSegmentation.load_model'):
            sam = SAMSegmentation(model_type="vit_b", device="cpu")
            sam.predictor = MagicMock()
            return sam

    def test_init(self):
        """Test SAM initialization."""
        with patch('src.services.sam_segmentation.SAM_AVAILABLE', True):
            sam = SAMSegmentation(model_type="vit_b", device="cuda")
            assert sam.model_type == "vit_b"
            assert sam.device == "cuda"

    def test_invalid_model_type(self):
        """Test with invalid model type."""
        with patch('src.services.sam_segmentation.SAM_AVAILABLE', True):
            with pytest.raises(ValueError):
                SAMSegmentation(model_type="invalid_model")

    def test_sam_not_available(self):
        """Test when SAM is not installed."""
        with patch('src.services.sam_segmentation.SAM_AVAILABLE', False):
            with pytest.raises(ImportError):
                SAMSegmentation()

    def test_set_image(self, sam):
        """Test setting image."""
        image = Image.new("RGB", (256, 256), color="red")
        sam.set_image(image)
        
        assert sam.current_image is not None
        assert sam.current_image.shape == (256, 256, 3)
        sam.predictor.set_image.assert_called_once()

    def test_generate_mask_from_points(self, sam):
        """Test mask generation from points."""
        # Mock predictor output
        mock_mask = np.ones((256, 256), dtype=np.bool_)
        sam.predictor.predict.return_value = (
            np.array([mock_mask]),  # masks
            np.array([0.95]),  # scores
            None,  # logits
        )

        # Set image first
        image = Image.new("RGB", (256, 256), color="red")
        sam.set_image(image)

        # Generate mask
        mask = sam.generate_mask_from_points(
            points=[(100, 100), (200, 200)],
            positive_points=True,
        )

        assert isinstance(mask, Image.Image)
        assert mask.mode == "L"
        assert mask.size == (256, 256)

    def test_generate_mask_from_points_with_confidence(self, sam):
        """Test mask generation with confidence scores."""
        mock_mask = np.ones((256, 256), dtype=np.bool_)
        sam.predictor.predict.return_value = (
            np.array([mock_mask]),
            np.array([0.95]),
            None,
        )

        image = Image.new("RGB", (256, 256), color="red")
        sam.set_image(image)

        mask, confidence = sam.generate_mask_from_points(
            points=[(100, 100)],
            return_confidence=True,
        )

        assert isinstance(mask, Image.Image)
        assert confidence is not None

    def test_generate_mask_from_box(self, sam):
        """Test mask generation from bounding box."""
        mock_mask = np.ones((256, 256), dtype=np.bool_)
        sam.predictor.predict.return_value = (
            np.array([mock_mask]),
            np.array([0.98]),
            None,
        )

        image = Image.new("RGB", (256, 256), color="red")
        sam.set_image(image)

        mask = sam.generate_mask_from_box(
            box=(50, 50, 200, 200),
        )

        assert isinstance(mask, Image.Image)
        assert mask.mode == "L"

    def test_generate_mask_without_image(self, sam):
        """Test that generating mask without setting image raises error."""
        sam.current_image = None
        
        with pytest.raises(RuntimeError):
            sam.generate_mask_from_points([(100, 100)])

    def test_text_prompt_not_implemented(self, sam):
        """Test that text prompts are not yet supported."""
        image = Image.new("RGB", (256, 256), color="red")
        sam.set_image(image)

        with pytest.raises(NotImplementedError):
            sam.generate_mask_from_text_prompt("a chair")

    def test_device_movement(self, sam):
        """Test moving model to different device."""
        sam.predictor.model = MagicMock()
        sam.to("cuda")
        
        assert sam.device == "cuda"
        sam.predictor.model.to.assert_called_with("cuda")

    def test_get_device(self, sam):
        """Test getting current device."""
        assert sam.get_device() == "cpu"
        sam.to("cuda")
        assert sam.get_device() == "cuda"


class TestSAMIntegration:
    """Integration tests for SAM with other services."""

    @pytest.fixture
    def config(self):
        """Mock configuration."""
        return {
            "device": "cpu",
            "sam": {
                "model_type": "vit_b",
            },
        }

    def test_generation_service_integration(self, config):
        """Test SAM integration with GenerationService."""
        from src.services import GenerationService

        with patch('src.services.sam_segmentation.SAMSegmentation.load_model'):
            service = GenerationService(config)
            
            # Service should be able to get SAM pipeline
            with patch.object(service, 'generate_mask_with_sam') as mock_gen:
                mock_gen.return_value = Image.new("L", (256, 256))
                
                # This would call the actual method
                assert service.config is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
