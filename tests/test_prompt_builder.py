"""Tests for prompt building with negative prompts and IP Adapter support."""

import pytest
from pathlib import Path
from src.services.prompt_builder import PromptBuilder, PromptPair, IPAdapterConfig


class TestPromptBuilder:
    """Test cases for PromptBuilder class."""

    @pytest.fixture
    def builder(self):
        """Create a PromptBuilder instance."""
        config = {}
        return PromptBuilder(config)

    def test_build_from_template(self, builder):
        """Test building a prompt pair from a template."""
        prompt_pair = builder.build_from_template(
            "living_room",
            style="modern",
            furniture="sleek furniture",
            lighting="warm lighting",
            decor="minimal decor",
        )

        assert isinstance(prompt_pair, PromptPair)
        assert "modern" in prompt_pair.positive
        assert "sleek furniture" in prompt_pair.positive
        assert prompt_pair.negative is not None  # Should have default negative prompt

    def test_unknown_template(self, builder):
        """Test attempting to build from unknown template."""
        with pytest.raises(ValueError):
            builder.build_from_template("unknown_room")

    def test_add_custom_template(self, builder):
        """Test adding a custom template."""
        custom_template = "A {adjective} space"
        builder.add_template("custom", custom_template)
        
        prompt_pair = builder.build_from_template("custom", adjective="beautiful")
        assert "beautiful" in prompt_pair.positive

    def test_missing_template_variable(self, builder):
        """Test template with missing required variable."""
        with pytest.raises(ValueError):
            builder.build_from_template("living_room", style="modern")

    def test_record_prompt(self, builder):
        """Test recording prompts."""
        builder.record_prompt(
            "test prompt",
            "text2img",
            {"guidance_scale": 7.5},
            negative_prompt="bad quality",
        )

        history = builder.get_prompt_history()
        assert len(history) == 1
        assert history[0]["prompt"] == "test prompt"
        assert history[0]["negative_prompt"] == "bad quality"

    def test_export_prompt_history(self, builder, tmp_path):
        """Test exporting prompt history."""
        builder.record_prompt("test", "text2img", {"param": "value"})
        
        output_file = tmp_path / "history.json"
        builder.export_prompt_history(str(output_file))
        
        assert output_file.exists()


class TestNegativePrompts:
    """Test cases for negative prompt functionality."""

    @pytest.fixture
    def builder(self):
        """Create a PromptBuilder instance."""
        return PromptBuilder({})

    def test_build_negative_prompt_with_preset(self, builder):
        """Test building negative prompt from preset."""
        negative = builder.build_negative_prompt(presets=["quality"])
        assert "blurry" in negative or "low quality" in negative

    def test_build_negative_prompt_multiple_presets(self, builder):
        """Test combining multiple negative presets."""
        negative = builder.build_negative_prompt(
            presets=["quality", "artifact"]
        )
        assert "quality" in negative.lower() or "blurry" in negative
        assert "watermark" in negative or "artifact" in negative

    def test_build_negative_prompt_with_custom_terms(self, builder):
        """Test building negative prompt with custom terms."""
        negative = builder.build_negative_prompt(
            custom_terms=["ugly", "deformed", "low resolution"]
        )
        assert "ugly" in negative
        assert "deformed" in negative

    def test_build_negative_prompt_combined(self, builder):
        """Test combining presets and custom terms."""
        negative = builder.build_negative_prompt(
            presets=["quality"],
            custom_terms=["ugly", "bad lighting"]
        )
        assert "quality" in negative.lower() or "blurry" in negative
        assert "ugly" in negative
        assert "bad lighting" in negative

    def test_add_custom_negative_preset(self, builder):
        """Test adding a custom negative preset."""
        builder.add_negative_preset("custom", "weird colors, saturated")
        negative = builder.build_negative_prompt(presets=["custom"])
        assert "weird colors" in negative or "saturated" in negative

    def test_build_prompt_pair(self, builder):
        """Test building a prompt pair."""
        pair = builder.build_prompt_pair(
            "beautiful living room",
            negative_presets=["quality"],
            negative_custom=["messy"]
        )

        assert pair.positive == "beautiful living room"
        assert pair.negative is not None
        assert "messy" in pair.negative

    def test_negative_prompt_deduplication(self, builder):
        """Test that duplicate terms are removed."""
        negative = builder.build_negative_prompt(
            presets=["quality", "quality"],
            custom_terms=["ugly", "ugly"]
        )
        # Count occurrences of "ugly" - should be 1
        assert negative.count("ugly") == 1

    def test_prompt_pair_str_representation(self, builder):
        """Test string representation of PromptPair."""
        pair = builder.build_prompt_pair(
            "test prompt",
            negative_presets=["quality"]
        )
        str_repr = str(pair)
        assert "test prompt" in str_repr
        assert "Positive:" in str_repr
        assert "Negative:" in str_repr


class TestIPAdapter:
    """Test cases for IP Adapter functionality."""

    @pytest.fixture
    def builder(self):
        """Create a PromptBuilder instance."""
        return PromptBuilder({})

    @pytest.fixture
    def temp_image(self, tmp_path):
        """Create a temporary test image."""
        img_path = tmp_path / "test_image.jpg"
        # Create a minimal image file
        img_path.write_bytes(b"fake_image_data")
        return str(img_path)

    def test_register_ip_adapter(self, builder, temp_image):
        """Test registering an IP Adapter."""
        config = builder.register_ip_adapter(
            "test_adapter",
            temp_image,
            weight=0.8,
            scale=1.2,
        )

        assert config.image_path == temp_image
        assert config.weight == 0.8
        assert config.scale == 1.2

    def test_register_ip_adapter_missing_file(self, builder):
        """Test registering IP Adapter with missing image."""
        with pytest.raises(FileNotFoundError):
            builder.register_ip_adapter(
                "bad_adapter",
                "/nonexistent/path/image.jpg"
            )

    def test_register_ip_adapter_invalid_weight(self, builder, temp_image):
        """Test IP Adapter with invalid weight."""
        with pytest.raises(ValueError):
            builder.register_ip_adapter("bad", temp_image, weight=1.5)

    def test_register_ip_adapter_invalid_type(self, builder, temp_image):
        """Test IP Adapter with invalid type."""
        with pytest.raises(ValueError):
            builder.register_ip_adapter(
                "bad",
                temp_image,
                ip_adapter_type="invalid_type"
            )

    def test_get_ip_adapter(self, builder, temp_image):
        """Test retrieving a registered IP Adapter."""
        builder.register_ip_adapter("test", temp_image)
        adapter = builder.get_ip_adapter("test")
        
        assert adapter is not None
        assert adapter.image_path == temp_image

    def test_list_ip_adapters(self, builder, temp_image):
        """Test listing all IP Adapters."""
        builder.register_ip_adapter("adapter1", temp_image)
        builder.register_ip_adapter("adapter2", temp_image)
        
        adapters = builder.list_ip_adapters()
        assert "adapter1" in adapters
        assert "adapter2" in adapters

    def test_remove_ip_adapter(self, builder, temp_image):
        """Test removing an IP Adapter."""
        builder.register_ip_adapter("test", temp_image)
        result = builder.remove_ip_adapter("test")
        
        assert result is True
        assert builder.get_ip_adapter("test") is None

    def test_build_with_ip_adapter(self, builder, temp_image):
        """Test building configuration with IP Adapter."""
        builder.register_ip_adapter(
            "test_adapter",
            temp_image,
            weight=0.9,
            conditioning_scale=1.5,
            ip_adapter_type="plus_face"
        )
        
        config = builder.build_with_ip_adapter(
            "Generate a portrait",
            "test_adapter",
            negative_presets=["artifact"],
            negative_custom=["blurry"]
        )

        assert config["prompt"] == "Generate a portrait"
        assert config["negative_prompt"] is not None
        assert "ip_adapter" in config
        assert config["ip_adapter"]["weight"] == 0.9
        assert config["ip_adapter"]["ip_adapter_type"] == "plus_face"

    def test_build_with_nonexistent_ip_adapter(self, builder):
        """Test building with non-registered IP Adapter."""
        with pytest.raises(ValueError):
            builder.build_with_ip_adapter(
                "test prompt",
                "nonexistent_adapter"
            )

    def test_ip_adapter_types(self, builder, temp_image):
        """Test different IP Adapter types."""
        types = ["plus", "plus_face", "full", "full_face"]
        
        for adapter_type in types:
            adapter = builder.register_ip_adapter(
                f"adapter_{adapter_type}",
                temp_image,
                ip_adapter_type=adapter_type
            )
            assert adapter.ip_adapter_type == adapter_type
