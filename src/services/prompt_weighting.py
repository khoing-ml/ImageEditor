"""Prompt weighting and advanced prompt conditioning."""

import logging
import re
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DualPrompt:
    """SDXL dual prompt structure."""
    prompt: str  # Main prompt (room type, layout, objects)
    prompt_2: str  # Secondary prompt (style, materials, lighting, realism)
    negative_prompt: Optional[str] = None
    negative_prompt_2: Optional[str] = None


class PromptWeightingParser:
    """Parse and process Compel-style prompt weighting syntax."""

    # Regex pattern for weighted syntax: (text)weight
    WEIGHT_PATTERN = r'\(([^)]+)\)(\d+\.?\d*)'

    @staticmethod
    def parse_weighted_prompt(prompt: str) -> List[Tuple[str, float]]:
        """
        Parse a weighted prompt string.

        Args:
            prompt: Prompt string with optional weights like "(important)1.5"

        Returns:
            List of (text, weight) tuples
        """
        tokens = []
        last_end = 0

        for match in re.finditer(PromptWeightingParser.WEIGHT_PATTERN, prompt):
            # Add unweighted text before this match
            if match.start() > last_end:
                unweighted = prompt[last_end:match.start()].strip()
                if unweighted:
                    tokens.append((unweighted, 1.0))

            # Add weighted text
            text = match.group(1).strip()
            weight = float(match.group(2))
            tokens.append((text, weight))

            last_end = match.end()

        # Add remaining unweighted text
        if last_end < len(prompt):
            remaining = prompt[last_end:].strip()
            if remaining:
                tokens.append((remaining, 1.0))

        return tokens

    @staticmethod
    def validate_weights(tokens: List[Tuple[str, float]]) -> bool:
        """
        Validate weight values.

        Args:
            tokens: List of (text, weight) tuples

        Returns:
            True if all weights are valid (> 0)
        """
        for text, weight in tokens:
            if weight <= 0:
                logger.warning(f"Weight must be positive: {weight}")
                return False
        return True

    @staticmethod
    def emphasize_token(text: str, weight: float = 1.5) -> str:
        """
        Add emphasis weight to a token.

        Args:
            text: Token text
            weight: Weight value (default 1.5 for strong emphasis)

        Returns:
            Weighted token string
        """
        if weight == 1.0:
            return text
        return f"({text}){weight}"

    @staticmethod
    def de_emphasize_token(text: str, weight: float = 0.7) -> str:
        """
        Add de-emphasis weight to a token.

        Args:
            text: Token text
            weight: Weight value (default 0.7 for de-emphasis)

        Returns:
            Weighted token string
        """
        if weight == 1.0:
            return text
        return f"({text}){weight}"


class DualPromptBuilder:
    """Build structured dual prompts for SDXL."""

    def __init__(self):
        """Initialize dual prompt builder."""
        self.weighting = PromptWeightingParser()

    def build_from_components(
        self,
        # Prompt components
        room_type: str,
        layout: str,
        focal_objects: str,
        style: str,
        materials: str,
        lighting: str,
        color_palette: str,
        photography_style: str = "interior photography",
        realism_tag: str = "ultra realistic",
        # Optional weighting
        emphasize: Optional[List[str]] = None,
        de_emphasize: Optional[List[str]] = None,
        # Negative prompts
        negative_prompts: Optional[List[str]] = None,
    ) -> DualPrompt:
        """
        Build a dual prompt from structured components.

        Args:
            room_type: Type of room (living room, bedroom, etc.)
            layout: Layout description (open, closed, spacious, etc.)
            focal_objects: Main objects (sofa, bed, desk, etc.)
            style: Design style (modern, scandinavian, luxury, etc.)
            materials: Materials (oak wood, marble, concrete, etc.)
            lighting: Lighting description (natural daylight, warm, dramatic, etc.)
            color_palette: Color scheme
            photography_style: How it should look (interior photography, etc.)
            realism_tag: Realism level
            emphasize: List of components to emphasize with weights
            de_emphasize: List of components to de-emphasize
            negative_prompts: Custom negative prompts

        Returns:
            DualPrompt object
        """
        # Build prompt (room, layout, objects)
        prompt_parts = [room_type, layout, focal_objects]
        prompt = ", ".join(p for p in prompt_parts if p).strip()

        # Apply emphasis to prompt components if needed
        if emphasize:
            for component in emphasize:
                if component in prompt_parts:
                    idx = prompt_parts.index(component)
                    prompt_parts[idx] = self.weighting.emphasize_token(component)
                    prompt = ", ".join(prompt_parts)

        # Build prompt_2 (style, materials, lighting, colors, realism)
        prompt_2_parts = [style, materials, lighting, color_palette, photography_style, realism_tag]
        prompt_2 = ", ".join(p for p in prompt_2_parts if p).strip()

        # Build negative prompt
        negative_prompt = self._build_negative_prompt(negative_prompts)

        logger.info(f"Built dual prompt pair")
        logger.debug(f"Prompt: {prompt}")
        logger.debug(f"Prompt_2: {prompt_2}")

        return DualPrompt(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=negative_prompt,
        )

    def _build_negative_prompt(self, custom_terms: Optional[List[str]] = None) -> str:
        """
        Build negative prompt with defaults.

        Args:
            custom_terms: Additional negative terms

        Returns:
            Negative prompt string
        """
        defaults = [
            "blurry",
            "low quality",
            "distorted perspective",
            "bad proportions",
            "unrealistic lighting",
            "duplicate furniture",
            "cluttered",
            "watermark",
        ]

        if custom_terms:
            defaults.extend(custom_terms)

        return ", ".join(defaults)

    def emphasize_component(
        self, dual_prompt: DualPrompt, component: str, weight: float = 1.5
    ) -> DualPrompt:
        """
        Add emphasis to a component in existing dual prompt.

        Args:
            dual_prompt: Existing DualPrompt
            component: Component to emphasize
            weight: Weight value

        Returns:
            Modified DualPrompt
        """
        # Try to emphasize in prompt
        if component in dual_prompt.prompt:
            prompt = dual_prompt.prompt.replace(
                component,
                self.weighting.emphasize_token(component, weight),
            )
        else:
            prompt = dual_prompt.prompt

        # Try to emphasize in prompt_2
        if component in dual_prompt.prompt_2:
            prompt_2 = dual_prompt.prompt_2.replace(
                component,
                self.weighting.emphasize_token(component, weight),
            )
        else:
            prompt_2 = dual_prompt.prompt_2

        return DualPrompt(
            prompt=prompt,
            prompt_2=prompt_2,
            negative_prompt=dual_prompt.negative_prompt,
            negative_prompt_2=dual_prompt.negative_prompt_2,
        )

    def split_prompt_by_role(self, full_prompt: str) -> DualPrompt:
        """
        Intelligently split a full prompt into dual prompts.

        Args:
            full_prompt: Complete prompt string

        Returns:
            Suggested DualPrompt split
        """
        # Heuristic: split on common style/quality keywords
        style_keywords = [
            "style", "aesthetic", "lighting", "material", "color",
            "photography", "realistic", "quality", "warm", "cool"
        ]

        parts = [p.strip() for p in full_prompt.split(",")]

        prompt_parts = []
        prompt_2_parts = []

        for part in parts:
            # If part contains style-related keywords, add to prompt_2
            if any(keyword in part.lower() for keyword in style_keywords):
                prompt_2_parts.append(part)
            else:
                prompt_parts.append(part)

        prompt = ", ".join(prompt_parts) if prompt_parts else full_prompt[:len(full_prompt)//2]
        prompt_2 = ", ".join(prompt_2_parts) if prompt_2_parts else full_prompt[len(full_prompt)//2:]

        logger.info("Split prompt into dual structure")

        return DualPrompt(prompt=prompt, prompt_2=prompt_2)

    def to_dict(self, dual_prompt: DualPrompt) -> Dict[str, Any]:
        """
        Convert DualPrompt to dictionary for pipeline use.

        Args:
            dual_prompt: DualPrompt object

        Returns:
            Dictionary with prompt keys
        """
        return {
            "prompt": dual_prompt.prompt,
            "prompt_2": dual_prompt.prompt_2,
            "negative_prompt": dual_prompt.negative_prompt or "",
            "negative_prompt_2": dual_prompt.negative_prompt_2 or "",
        }


class InteriorDesignPromptBuilder(DualPromptBuilder):
    """Specialized builder for interior design prompts."""

    # Interior design specific presets
    ROOM_TYPES = {
        "living_room": "living room",
        "bedroom": "bedroom",
        "kitchen": "kitchen",
        "bathroom": "bathroom",
        "office": "home office",
        "dining": "dining room",
        "entryway": "entryway",
        "study": "study room",
    }

    LAYOUT_OPTIONS = {
        "open": "open layout",
        "spacious": "spacious and bright",
        "cozy": "cozy and intimate",
        "minimalist": "minimal layout",
        "compartmented": "compartmented spaces",
        "multi_zone": "multi-zone layout",
    }

    STYLE_PALETTES = {
        "modern": "modern contemporary",
        "scandinavian": "Scandinavian minimalist",
        "industrial": "industrial modern",
        "luxury": "luxury elegant",
        "bohemian": "bohemian eclectic",
        "traditional": "traditional classic",
        "japandi": "Japanese Scandinavian",
        "mid_century": "mid-century modern",
        "coastal": "coastal maritime",
        "farmhouse": "farmhouse rustic",
    }

    MATERIAL_PALETTES = {
        "natural_wood": "light oak wood, natural materials",
        "concrete": "concrete, steel, industrial materials",
        "marble": "marble, stone, luxurious materials",
        "velvet": "velvet furnishings, soft textures",
        "wool": "wool textiles, natural fibers",
        "leather": "leather upholstery, refined textures",
        "glass": "glass surfaces, minimalist materials",
        "rattan": "rattan, wicker, natural weaves",
        "fabric": "linen fabrics, cotton textiles",
    }

    LIGHTING_OPTIONS = {
        "natural": "warm natural daylight",
        "ambient": "soft ambient lighting",
        "warm": "warm color temperature lighting",
        "cool": "cool modern lighting",
        "dramatic": "dramatic accent lighting",
        "moody": "moody atmospheric lighting",
        "bright": "bright task lighting",
        "mixed": "mixed natural and accent lighting",
    }

    COLOR_PALETTES = {
        "neutral": "neutral beige palette",
        "warm": "warm earth tones",
        "cool": "cool gray palette",
        "jewel": "jewel tones and rich colors",
        "monochrome": "monochrome grayscale",
        "pastel": "soft pastel tones",
        "bold": "bold saturated colors",
        "muted": "muted desaturated palette",
    }

    def build_interior_prompt(
        self,
        room: str,
        layout: str,
        focal_objects: str,
        style: str,
        materials: str,
        lighting: str,
        color_palette: str,
        custom_negative: Optional[List[str]] = None,
    ) -> DualPrompt:
        """
        Build an interior-specific dual prompt.

        Args:
            room: Room type key (living_room, bedroom, etc.)
            layout: Layout key (open, spacious, cozy, etc.)
            focal_objects: Comma-separated objects or custom text
            style: Style key (modern, scandinavian, etc.)
            materials: Materials key (natural_wood, concrete, etc.)
            lighting: Lighting key (natural, ambient, warm, etc.)
            color_palette: Color palette key (neutral, warm, etc.)
            custom_negative: Additional negative prompt terms

        Returns:
            DualPrompt
        """
        # Resolve presets to actual values
        room_text = self.ROOM_TYPES.get(room, room)
        layout_text = self.LAYOUT_OPTIONS.get(layout, layout)
        style_text = self.STYLE_PALETTES.get(style, style)
        materials_text = self.MATERIAL_PALETTES.get(materials, materials)
        lighting_text = self.LIGHTING_OPTIONS.get(lighting, lighting)
        color_text = self.COLOR_PALETTES.get(color_palette, color_palette)

        # Build using parent method
        return self.build_from_components(
            room_type=room_text,
            layout=layout_text,
            focal_objects=focal_objects,
            style=style_text,
            materials=materials_text,
            lighting=lighting_text,
            color_palette=color_text,
            photography_style="interior photography",
            realism_tag="ultra realistic",
            negative_prompts=custom_negative,
        )
