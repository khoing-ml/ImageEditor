"""Examples: Dual Prompting, Prompt Weighting, and Interior Design Generation."""

import logging
from src.services import (
    DualPromptBuilder,
    InteriorDesignPromptBuilder,
    PromptWeightingParser,
)


logging.basicConfig(level=logging.INFO)


def example_prompt_weighting():
    """Example: Prompt weighting with emphasized tokens."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Prompt Weighting with Emphasized Tokens")
    print("="*70 + "\n")
    
    parser = PromptWeightingParser()
    
    # Parse a weighted prompt
    prompt = "(natural daylight)1.5 modern living room with (oak wood)1.3 furniture"
    print(f"Original: {prompt}\n")
    
    tokens = parser.parse_weighted_prompt(prompt)
    print("Parsed tokens:")
    for text, weight in tokens:
        print(f"  '{text}' -> weight {weight}")
    
    # Emphasize components
    print("\n\nEmphasis examples:")
    important = parser.emphasize_token("natural light", weight=1.7)
    print(f"  Emphasize 'natural light': {important}")
    
    secondary = parser.de_emphasize_token("clutter", weight=0.3)
    print(f"  De-emphasize 'clutter': {secondary}")


def example_dual_prompting():
    """Example: Building structured dual prompts."""
    print("\n" + "="*70)
    print("EXAMPLE 2: SDXL Dual Prompt Structure")
    print("="*70 + "\n")
    
    builder = DualPromptBuilder()
    
    # Build a dual prompt from structured components
    pair = builder.build_from_components(
        room_type="modern living room",
        layout="open spacious layout",
        focal_objects="low sofa, glass coffee table, accent wall art",
        style="minimalist contemporary",
        materials="light oak wood, white walls",
        lighting="warm natural daylight from large windows",
        color_palette="neutral beige with subtle gray accents",
        photography_style="interior photography",
        realism_tag="ultra realistic, high detail"
    )
    
    print("PROMPT (room, layout, objects):")
    print(f"  {pair.prompt}\n")
    
    print("PROMPT_2 (style, materials, lighting, realism):")
    print(f"  {pair.prompt_2}\n")
    
    print("NEGATIVE_PROMPT:")
    print(f"  {pair.negative_prompt}\n")
    
    # Get as dictionary for pipeline
    prompt_dict = builder.to_dict(pair)
    print("For pipeline use:")
    for key, value in prompt_dict.items():
        print(f"  {key}: {value[:60]}...")


def example_interior_design_builder():
    """Example: Interior design specialized builder."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Interior Design Prompt Builder")
    print("="*70 + "\n")
    
    builder = InteriorDesignPromptBuilder()
    
    # Example 1: Modern living room
    print("1. MODERN LIVING ROOM\n")
    pair = builder.build_interior_prompt(
        room="living_room",
        layout="open",
        focal_objects="sofa, coffee table, accent shelving",
        style="modern",
        materials="natural_wood",
        lighting="natural",
        color_palette="neutral",
        custom_negative=["cluttered", "dark"]
    )
    
    print(f"Prompt:\n  {pair.prompt}\n")
    print(f"Prompt_2:\n  {pair.prompt_2}\n")
    print(f"Negative:\n  {pair.negative_prompt}\n\n")
    
    # Example 2: Luxury bedroom
    print("2. LUXURY BEDROOM\n")
    pair = builder.build_interior_prompt(
        room="bedroom",
        layout="spacious",
        focal_objects="platform bed, upholstered headboard, night tables",
        style="luxury",
        materials="velvet",
        lighting="ambient",
        color_palette="jewel",
        custom_negative=["cheap looking", "plastic"]
    )
    
    print(f"Prompt:\n  {pair.prompt}\n")
    print(f"Prompt_2:\n  {pair.prompt_2}\n")
    print(f"Negative:\n  {pair.negative_prompt}\n\n")
    
    # Example 3: Scandinavian kitchen
    print("3. SCANDINAVIAN KITCHEN\n")
    pair = builder.build_interior_prompt(
        room="kitchen",
        layout="open",
        focal_objects="island, stove, dining nook",
        style="scandinavian",
        materials="natural_wood",
        lighting="bright",
        color_palette="warm"
    )
    
    print(f"Prompt:\n  {pair.prompt}\n")
    print(f"Prompt_2:\n  {pair.prompt_2}\n")


def example_dual_prompt_emphasis():
    """Example: Adding emphasis to existing prompts."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Emphasizing Components")
    print("="*70 + "\n")
    
    builder = DualPromptBuilder()
    
    # Start with base prompt
    base = builder.build_from_components(
        room_type="living room",
        layout="open layout",
        focal_objects="sofa, coffee table",
        style="contemporary",
        materials="oak wood",
        lighting="natural daylight",
        color_palette="warm tones"
    )
    
    print("BASE PROMPT:")
    print(f"  {base.prompt}\n")
    print("BASE PROMPT_2:")
    print(f"  {base.prompt_2}\n")
    
    # Emphasize lighting importance
    emphasized = builder.emphasize_component(base, "natural daylight", weight=1.7)
    
    print("AFTER EMPHASIZING 'natural daylight':")
    print(f"  {emphasized.prompt_2}\n")


def example_prompt_splitting():
    """Example: Intelligent prompt splitting."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Intelligent Prompt Splitting")
    print("="*70 + "\n")
    
    builder = DualPromptBuilder()
    
    # Full prompt that needs splitting
    full_prompt = """
    A beautiful modern bedroom with a large bed, nightstands, wooden flooring,
    contemporary style, soft warm lighting, luxurious materials, soft color palette,
    interior photography, ultra realistic, high detail, professional composition
    """
    
    print("ORIGINAL FULL PROMPT:")
    print(f"  {full_prompt.strip()}\n")
    
    pair = builder.split_prompt_by_role(full_prompt)
    
    print("SPLIT INTO DUAL PROMPTS:\n")
    print("Prompt (spatial/objects):")
    print(f"  {pair.prompt}\n")
    print("Prompt_2 (style/quality):")
    print(f"  {pair.prompt_2}\n")


def example_room_comparison():
    """Example: Comparing prompts across different rooms."""
    print("\n" + "="*70)
    print("EXAMPLE 6: Multi-Room Comparison")
    print("="*70 + "\n")
    
    builder = InteriorDesignPromptBuilder()
    
    rooms = ["living_room", "bedroom", "kitchen", "bathroom", "office"]
    
    for room in rooms:
        pair = builder.build_interior_prompt(
            room=room,
            layout="spacious",
            focal_objects="primary furniture",
            style="modern",
            materials="natural_wood",
            lighting="natural",
            color_palette="neutral"
        )
        
        print(f"\n{room.upper().replace('_', ' ')}:")
        print(f"  Prompt: {pair.prompt[:70]}...")
        print(f"  Style:  {pair.prompt_2[:70]}...")


def example_style_comparison():
    """Example: Comparing same room with different styles."""
    print("\n" + "="*70)
    print("EXAMPLE 7: Style Variations for Same Room")
    print("="*70 + "\n")
    
    builder = InteriorDesignPromptBuilder()
    
    styles = ["modern", "scandinavian", "luxury", "industrial", "bohemian"]
    
    print("LIVING ROOM WITH DIFFERENT STYLES:\n")
    
    for style in styles:
        pair = builder.build_interior_prompt(
            room="living_room",
            layout="open",
            focal_objects="sofa, coffee table",
            style=style,
            materials="natural_wood",
            lighting="natural",
            color_palette="neutral"
        )
        
        print(f"\n{style.upper()}:")
        # Extract just the style part from prompt_2
        style_part = pair.prompt_2.split(",")[0]
        print(f"  {style_part}")


def main():
    """Run all examples."""
    print("\n\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "DUAL PROMPT & WEIGHTING EXAMPLES" + " "*21 + "║")
    print("║" + " "*10 + "SDXL, Compel, and Interior Design" + " "*24 + "║")
    print("╚" + "="*68 + "╝")
    
    example_prompt_weighting()
    example_dual_prompting()
    example_interior_design_builder()
    example_dual_prompt_emphasis()
    example_prompt_splitting()
    example_room_comparison()
    example_style_comparison()
    
    print("\n\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*18 + "EXAMPLES COMPLETED" + " "*32 + "║")
    print("╚" + "="*68 + "╝")
    print("\n")
    
    print("NEXT STEPS:")
    print("  1. Use InteriorDesignPromptBuilder for structured prompts")
    print("  2. Pass results to SDXL text2img pipeline")
    print("  3. Use DualPrompt dictionary for prompt/prompt_2 parameters")
    print("  4. Add emphasis weights for critical components")
    print("  5. Combine with IP-Adapter for style reference images")
    print("  6. Use ControlNet depth for structure preservation\n")


if __name__ == "__main__":
    main()
