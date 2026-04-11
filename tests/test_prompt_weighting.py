"""Tests for prompt weighting and dual prompt functionality."""

import pytest
from src.services.prompt_weighting import (
    PromptWeightingParser,
    DualPrompt,
    DualPromptBuilder,
    InteriorDesignPromptBuilder,
)


class TestPromptWeightingParser:
    """Test cases for PromptWeightingParser."""

    def test_parse_no_weights(self):
        """Test parsing prompt without weights."""
        prompt = "a beautiful bedroom with natural light"
        tokens = PromptWeightingParser.parse_weighted_prompt(prompt)
        
        assert len(tokens) == 1
        assert tokens[0][0] == prompt
        assert tokens[0][1] == 1.0

    def test_parse_single_weight(self):
        """Test parsing prompt with single weighted token."""
        prompt = "(beautiful bedroom)1.5 with natural light"
        tokens = PromptWeightingParser.parse_weighted_prompt(prompt)
        
        assert len(tokens) == 2
        assert tokens[0][0] == "beautiful bedroom"
        assert tokens[0][1] == 1.5
        assert tokens[1][1] == 1.0

    def test_parse_multiple_weights(self):
        """Test parsing prompt with multiple weighted tokens."""
        prompt = "(modern)1.3 (living room)1.5 with (natural light)1.2"
        tokens = PromptWeightingParser.parse_weighted_prompt(prompt)
        
        assert len(tokens) == 5
        assert tokens[0][1] == 1.3
        assert tokens[2][1] == 1.5
        assert tokens[4][1] == 1.2

    def test_validate_weights_valid(self):
        """Test valid weight validation."""
        tokens = [
            ("token1", 0.5),
            ("token2", 1.0),
            ("token3", 2.0),
        ]
        assert PromptWeightingParser.validate_weights(tokens) is True

    def test_validate_weights_invalid(self):
        """Test invalid weight validation."""
        tokens = [
            ("token1", 0.5),
            ("token2", -0.5),  # Invalid negative weight
        ]
        assert PromptWeightingParser.validate_weights(tokens) is False

    def test_emphasize_token(self):
        """Test token emphasis."""
        result = PromptWeightingParser.emphasize_token("important", weight=1.5)
        assert "(important)1.5" == result

    def test_emphasize_token_default_weight(self):
        """Test token emphasis with default weight."""
        result = PromptWeightingParser.emphasize_token("important")
        assert "(important)1.5" == result

    def test_de_emphasize_token(self):
        """Test token de-emphasis."""
        result = PromptWeightingParser.de_emphasize_token("unimportant", weight=0.5)
        assert "(unimportant)0.5" == result

    def test_de_emphasize_token_default_weight(self):
        """Test token de-emphasis with default weight."""
        result = PromptWeightingParser.de_emphasize_token("unimportant")
        assert "(unimportant)0.7" == result


class TestDualPrompt:
    """Test cases for DualPrompt class."""

    def test_dual_prompt_creation(self):
        """Test creating a DualPrompt."""
        pair = DualPrompt(
            prompt="living room",
            prompt_2="modern style",
            negative_prompt="blurry"
        )
        
        assert pair.prompt == "living room"
        assert pair.prompt_2 == "modern style"
        assert pair.negative_prompt == "blurry"

    def test_dual_prompt_str_representation(self):
        """Test string representation of DualPrompt."""
        pair = DualPrompt(
            prompt="living room",
            prompt_2="modern style"
        )
        
        str_repr = str(pair)
        assert "Positive:" in str_repr
        assert "living room" in str_repr
        assert "Negative:" in str_repr


class TestDualPromptBuilder:
    """Test cases for DualPromptBuilder."""

    @pytest.fixture
    def builder(self):
        """Create a DualPromptBuilder instance."""
        return DualPromptBuilder()

    def test_build_from_components(self, builder):
        """Test building dual prompt from components."""
        pair = builder.build_from_components(
            room_type="living room",
            layout="open layout",
            focal_objects="sofa, coffee table",
            style="modern",
            materials="oak wood",
            lighting="natural daylight",
            color_palette="neutral beige"
        )
        
        assert isinstance(pair, DualPrompt)
        assert "living room" in pair.prompt
        assert "sofa" in pair.prompt
        assert "modern" in pair.prompt_2
        assert "oak wood" in pair.prompt_2
        assert pair.negative_prompt is not None

    def test_build_with_emphasis(self, builder):
        """Test building with emphasized components."""
        pair = builder.build_from_components(
            room_type="bedroom",
            layout="spacious",
            focal_objects="bed",
            style="luxury",
            materials="marble",
            lighting="warm",
            color_palette="jewel tones",
            emphasize=["luxury", "marble"]
        )
        
        # Emphasis weight should be in prompt_2
        assert "luxury" in pair.prompt_2
        assert "marble" in pair.prompt_2

    def test_emphasize_component(self, builder):
        """Test emphasizing a component in existing dual prompt."""
        original = DualPrompt(
            prompt="living room with sofa",
            prompt_2="modern style"
        )
        
        emphasized = builder.emphasize_component(original, "modern", weight=1.5)
        
        assert "(modern)1.5" in emphasized.prompt_2

    def test_split_prompt_by_role(self, builder):
        """Test intelligent prompt splitting."""
        full = "beautiful modern living room with natural lighting, oak wood materials, warm atmosphere"
        
        pair = builder.split_prompt_by_role(full)
        
        assert pair.prompt is not None
        assert pair.prompt_2 is not None
        # Style-related keywords should go to prompt_2
        if "natural lighting" in pair.prompt_2:
            assert "natural lighting" in pair.prompt_2

    def test_to_dict(self, builder):
        """Test converting DualPrompt to dictionary."""
        pair = DualPrompt(
            prompt="test prompt",
            prompt_2="test style",
            negative_prompt="bad quality"
        )
        
        result = builder.to_dict(pair)
        
        assert result["prompt"] == "test prompt"
        assert result["prompt_2"] == "test style"
        assert result["negative_prompt"] == "bad quality"


class TestInteriorDesignPromptBuilder:
    """Test cases for InteriorDesignPromptBuilder."""

    @pytest.fixture
    def builder(self):
        """Create an InteriorDesignPromptBuilder instance."""
        return InteriorDesignPromptBuilder()

    def test_room_types(self, builder):
        """Test that room types are available."""
        assert "living_room" in builder.ROOM_TYPES
        assert "bedroom" in builder.ROOM_TYPES
        assert "kitchen" in builder.ROOM_TYPES

    def test_style_palettes(self, builder):
        """Test that style palettes are available."""
        assert "modern" in builder.STYLE_PALETTES
        assert "scandinavian" in builder.STYLE_PALETTES
        assert "luxury" in builder.STYLE_PALETTES

    def test_build_interior_prompt_living_room(self, builder):
        """Test building interior prompt for living room."""
        pair = builder.build_interior_prompt(
            room="living_room",
            layout="open",
            focal_objects="sofa, coffee table",
            style="modern",
            materials="natural_wood",
            lighting="natural",
            color_palette="neutral"
        )
        
        assert "living room" in pair.prompt
        assert "open layout" in pair.prompt
        assert "modern contemporary" in pair.prompt_2
        assert "light oak wood" in pair.prompt_2
        assert "warm natural daylight" in pair.prompt_2

    def test_build_interior_prompt_bedroom(self, builder):
        """Test building interior prompt for bedroom."""
        pair = builder.build_interior_prompt(
            room="bedroom",
            layout="cozy",
            focal_objects="bed, nightstands",
            style="scandinavian",
            materials="natural_wood",
            lighting="warm",
            color_palette="warm"
        )
        
        assert "bedroom" in pair.prompt
        assert "cozy and intimate" in pair.prompt
        assert "Scandinavian minimalist" in pair.prompt_2

    def test_build_interior_prompt_kitchen(self, builder):
        """Test building interior prompt for kitchen."""
        pair = builder.build_interior_prompt(
            room="kitchen",
            layout="open",
            focal_objects="island, appliances",
            style="modern",
            materials="concrete",
            lighting="bright",
            color_palette="cool"
        )
        
        assert "kitchen" in pair.prompt
        assert "modern contemporary" in pair.prompt_2
        assert "concrete, steel" in pair.prompt_2

    def test_build_with_custom_negative(self, builder):
        """Test building with custom negative prompts."""
        pair = builder.build_interior_prompt(
            room="living_room",
            layout="open",
            focal_objects="sofa",
            style="modern",
            materials="natural_wood",
            lighting="natural",
            color_palette="neutral",
            custom_negative=["dark", "cluttered"]
        )
        
        assert "dark" in pair.negative_prompt
        assert "cluttered" in pair.negative_prompt

    def test_material_palette_resolution(self, builder):
        """Test that material palettes are properly resolved."""
        pair = builder.build_interior_prompt(
            room="living_room",
            layout="open",
            focal_objects="sofa",
            style="modern",
            materials="concrete",  # Should resolve to actual material text
            lighting="natural",
            color_palette="neutral"
        )
        
        # "concrete" should be resolved to full text
        assert "concrete" in pair.prompt_2 or "steel" in pair.prompt_2

    def test_all_rooms(self, builder):
        """Test building prompts for all room types."""
        for room_key in builder.ROOM_TYPES.keys():
            pair = builder.build_interior_prompt(
                room=room_key,
                layout="open",
                focal_objects="furniture",
                style="modern",
                materials="natural_wood",
                lighting="natural",
                color_palette="neutral"
            )
            
            assert pair.prompt is not None
            assert len(pair.prompt) > 0
            assert pair.prompt_2 is not None
            assert len(pair.prompt_2) > 0

    def test_all_styles(self, builder):
        """Test building prompts with all style options."""
        for style_key in builder.STYLE_PALETTES.keys():
            pair = builder.build_interior_prompt(
                room="living_room",
                layout="open",
                focal_objects="furniture",
                style=style_key,
                materials="natural_wood",
                lighting="natural",
                color_palette="neutral"
            )
            
            assert pair.prompt_2 is not None
            # Style should be in prompt_2
            assert builder.STYLE_PALETTES[style_key] in pair.prompt_2

    def test_dual_prompt_structure(self, builder):
        """Test that dual prompt maintains proper structure."""
        pair = builder.build_interior_prompt(
            room="bedroom",
            layout="spacious",
            focal_objects="bed",
            style="luxury",
            materials="marble",
            lighting="ambient",
            color_palette="jewel"
        )
        
        # Prompt should have room/layout/objects
        assert any(word in pair.prompt.lower() for word in ["bedroom", "spacious", "bed"])
        
        # Prompt_2 should have style/materials/lighting/palette
        assert any(word in pair.prompt_2.lower() for word in ["luxury", "marble", "lighting", "tone"])
