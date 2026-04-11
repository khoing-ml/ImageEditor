"""Prompt management and template building with negative prompts and IP Adapter support."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PromptPair:
    """Container for positive and negative prompt pair."""
    positive: str
    negative: Optional[str] = None
    
    def __str__(self) -> str:
        """String representation."""
        return f"Positive: {self.positive}\nNegative: {self.negative or 'None'}"


@dataclass
class IPAdapterConfig:
    """Configuration for IP Adapter."""
    image_path: str
    weight: float = 1.0
    scale: float = 1.0
    conditioning_scale: float = 1.0
    ip_adapter_type: str = "plus"  # 'plus', 'plus_face', 'full', 'full_face'
    
    def validate(self) -> None:
        """Validate IP Adapter configuration."""
        if not Path(self.image_path).exists():
            raise FileNotFoundError(f"IP Adapter image not found: {self.image_path}")
        
        if self.weight < 0 or self.weight > 1:
            raise ValueError(f"Weight must be between 0 and 1, got {self.weight}")
        
        if self.scale < 0:
            raise ValueError(f"Scale must be non-negative, got {self.scale}")
        
        valid_types = {"plus", "plus_face", "full", "full_face"}
        if self.ip_adapter_type not in valid_types:
            raise ValueError(f"Invalid IP Adapter type: {self.ip_adapter_type}")


class PromptBuilder:
    """Build and manage prompts with templates, negative prompts, and IP Adapter support."""

    DEFAULT_TEMPLATES = {
        "living_room": "A modern {style} living room with {furniture}, {lighting}, and {decor}",
        "bedroom": "A cozy {style} bedroom with {furniture}, {color_palette}, and {ambiance}",
        "kitchen": "A contemporary {style} kitchen with {cabinets}, {countertops}, and {appliances}",
        "office": "A productive {style} office with {furniture}, {lighting}, and {organization}",
        "bathroom": "A{article} {style} bathroom with {fixtures}, {tiles}, and {accessories}",
    }

    # Common negative prompt templates by issue
    NEGATIVE_PROMPT_PRESETS = {
        "quality": "blurry, low quality, pixelated, distorted, bad anatomy",
        "artifact": "watermark, text, logo, signature, border artifacts",
        "artistic": "cartoon, painted, sketch, drawing, illustration",
        "anatomy": "deformed, mutated, malformed, asymmetrical, broken",
        "professional": "amateurish, unprofessional, poorly lit, overexposed, underexposed",
        "style": "unrealistic, fake, artificial, CGI, 3D render",
        "furniture": "broken furniture, damaged decor, messy, cluttered",
        "all": "blurry, low quality, watermark, text, distorted, bad anatomy, deformed, cartoon, poorly lit"
    }

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize prompt builder.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.templates = self.DEFAULT_TEMPLATES.copy()
        self.negative_templates = self._create_default_negative_templates()
        self.prompt_history: List[Dict[str, Any]] = []
        self.ip_adapter_configs: Dict[str, IPAdapterConfig] = {}

    def build_from_template(self, room_type: str, **kwargs) -> PromptPair:
        """
        Build a prompt pair from templates.

        Args:
            room_type: Type of room (living_room, bedroom, etc.)
            **kwargs: Values to fill in the template. 
                     Can include 'negative_preset' key for negative template

        Returns:
            PromptPair with positive and negative prompts
        """
        if room_type not in self.templates:
            raise ValueError(f"Unknown room type: {room_type}")

        # Extract negative preset if provided
        negative_preset = kwargs.pop("negative_preset", "quality")
        
        template = self.templates[room_type]
        try:
            prompt = template.format(**kwargs)
            logger.info(f"Built prompt from template: {room_type}")
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")
        
        # Build negative prompt
        negative_prompt = self._build_negative_prompt(negative_preset)
        
        return PromptPair(positive=prompt, negative=negative_prompt)

    def add_template(self, name: str, template: str) -> None:
        """Add a custom prompt template."""
        self.templates[name] = template
        logger.info(f"Added custom template: {name}")

    def _create_default_negative_templates(self) -> Dict[str, str]:
        """Create default negative prompt templates."""
        return {
            "quality": "blurry, low quality, pixelated, distorted, noisy",
            "artifact": "watermark, text, logo, signature, jpeg artifacts",
            "anatomy": "deformed, malformed, asymmetrical, broken limbs, bad proportions",
            "professional": "amateurish, unprofessional, poorly lit, overexposed",
            "style": "cartoon, painted, sketch, drawing, illustrated style",
            "furniture": "broken, damaged, messy, cluttered, dilapidated",
        }

    def _build_negative_prompt(self, preset: str) -> str:
        """
        Build a negative prompt from a preset.

        Args:
            preset: Negative prompt preset name

        Returns:
            Negative prompt string
        """
        if preset not in self.negative_templates:
            logger.warning(f"Unknown negative preset: {preset}, using 'quality'")
            preset = "quality"
        
        return self.negative_templates[preset]

    def build_negative_prompt(
        self, 
        presets: Optional[List[str]] = None, 
        custom_terms: Optional[List[str]] = None
    ) -> str:
        """
        Build a comprehensive negative prompt from presets and custom terms.

        Args:
            presets: List of preset names to combine
            custom_terms: Additional custom negative terms

        Returns:
            Combined negative prompt string
        """
        parts = []
        
        if presets:
            for preset in presets:
                if preset in self.negative_templates:
                    parts.append(self.negative_templates[preset])
                else:
                    logger.warning(f"Unknown negative preset: {preset}")
        
        if custom_terms:
            parts.extend(custom_terms)
        
        if not parts:
            logger.warning("No negative prompt components provided")
            return self.negative_templates["quality"]
        
        # Combine with commas, removing duplicates while preserving order
        combined = ", ".join(parts)
        seen = set()
        result = []
        for item in combined.split(", "):
            item = item.strip()
            if item and item not in seen:
                result.append(item)
                seen.add(item)
        
        negative_prompt = ", ".join(result)
        logger.debug(f"Built negative prompt: {negative_prompt}")
        return negative_prompt

    def add_negative_preset(self, name: str, terms: str) -> None:
        """
        Add a custom negative prompt preset.

        Args:
            name: Preset name
            terms: Negative prompt terms (comma-separated or single string)
        """
        self.negative_templates[name] = terms
        logger.info(f"Added negative prompt preset: {name}")

    def build_prompt_pair(
        self, 
        positive: str, 
        negative_presets: Optional[List[str]] = None,
        negative_custom: Optional[List[str]] = None,
    ) -> PromptPair:
        """
        Build a prompt pair from positive text and negative components.

        Args:
            positive: Positive prompt text
            negative_presets: List of negative preset names
            negative_custom: Custom negative terms

        Returns:
            PromptPair object
        """
        negative = self.build_negative_prompt(negative_presets, negative_custom) if (negative_presets or negative_custom) else None
        return PromptPair(positive=positive, negative=negative)

    # IP Adapter Methods

    def register_ip_adapter(
        self,
        name: str,
        image_path: str,
        weight: float = 1.0,
        scale: float = 1.0,
        conditioning_scale: float = 1.0,
        ip_adapter_type: str = "plus",
    ) -> IPAdapterConfig:
        """
        Register an IP Adapter configuration.

        Args:
            name: Unique identifier for this IP Adapter
            image_path: Path to the IP Adapter image
            weight: Adapter weight (0-1)
            scale: Adapter scale factor
            conditioning_scale: Conditioning scale for IP Adapter
            ip_adapter_type: Type of IP Adapter

        Returns:
            IPAdapterConfig object
        """
        config = IPAdapterConfig(
            image_path=image_path,
            weight=weight,
            scale=scale,
            conditioning_scale=conditioning_scale,
            ip_adapter_type=ip_adapter_type,
        )
        config.validate()
        self.ip_adapter_configs[name] = config
        logger.info(f"Registered IP Adapter: {name}")
        return config

    def get_ip_adapter(self, name: str) -> Optional[IPAdapterConfig]:
        """Get a registered IP Adapter configuration."""
        return self.ip_adapter_configs.get(name)

    def list_ip_adapters(self) -> List[str]:
        """List all registered IP Adapter names."""
        return list(self.ip_adapter_configs.keys())

    def remove_ip_adapter(self, name: str) -> bool:
        """Remove an IP Adapter configuration."""
        if name in self.ip_adapter_configs:
            del self.ip_adapter_configs[name]
            logger.info(f"Removed IP Adapter: {name}")
            return True
        return False

    def build_with_ip_adapter(
        self,
        prompt: str,
        ip_adapter_name: str,
        negative_presets: Optional[List[str]] = None,
        negative_custom: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Build a complete generation configuration with IP Adapter.

        Args:
            prompt: Main text prompt
            ip_adapter_name: Name of registered IP Adapter
            negative_presets: Negative prompt presets
            negative_custom: Custom negative terms

        Returns:
            Dictionary with prompt and IP Adapter configuration
        """
        ip_adapter = self.get_ip_adapter(ip_adapter_name)
        if not ip_adapter:
            raise ValueError(f"IP Adapter not found: {ip_adapter_name}")
        
        prompt_pair = self.build_prompt_pair(prompt, negative_presets, negative_custom)
        
        config = {
            "prompt": prompt_pair.positive,
            "negative_prompt": prompt_pair.negative,
            "ip_adapter": {
                "image_path": ip_adapter.image_path,
                "weight": ip_adapter.weight,
                "scale": ip_adapter.scale,
                "conditioning_scale": ip_adapter.conditioning_scale,
                "ip_adapter_type": ip_adapter.ip_adapter_type,
            }
        }
        
        logger.info(f"Built generation config with IP Adapter: {ip_adapter_name}")
        return config

    def load_templates_from_file(self, file_path: str) -> None:
        """Load templates from a YAML file."""
        try:
            import yaml

            path = Path(file_path)
            if path.exists():
                with open(path, "r") as f:
                    data = yaml.safe_load(f)
                    
                    if data:
                        # Load positive prompts templates
                        if "templates" in data:
                            self.templates.update(data["templates"])
                            logger.info(f"Loaded {len(data['templates'])} templates")
                        
                        # Load negative prompt templates
                        if "negative_templates" in data:
                            self.negative_templates.update(data["negative_templates"])
                            logger.info(f"Loaded {len(data['negative_templates'])} negative templates")
        except Exception as e:
            logger.warning(f"Failed to load templates from {file_path}: {e}")

    def record_prompt(
        self,
        prompt: str,
        pipeline_type: str,
        parameters: Dict[str, Any],
        negative_prompt: Optional[str] = None,
        ip_adapter_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a prompt and its parameters for logging.

        Args:
            prompt: The positive prompt used
            pipeline_type: Type of pipeline (text2img, img2img, etc.)
            parameters: Generation parameters
            negative_prompt: Optional negative prompt used
            ip_adapter_config: Optional IP Adapter configuration
        """
        entry = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "pipeline": pipeline_type,
            "parameters": parameters,
        }
        
        if ip_adapter_config:
            entry["ip_adapter"] = ip_adapter_config
        
        self.prompt_history.append(entry)
        logger.debug(f"Recorded prompt for {pipeline_type}")

    def get_prompt_history(self) -> List[Dict[str, Any]]:
        """Get the history of all recorded prompts."""
        return self.prompt_history.copy()

    def export_prompt_history(self, output_path: str) -> None:
        """Export prompt history to a JSON file."""
        try:
            import json

            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w") as f:
                json.dump(self.prompt_history, f, indent=2)

            logger.info(f"Exported {len(self.prompt_history)} prompts to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export prompt history: {e}")
