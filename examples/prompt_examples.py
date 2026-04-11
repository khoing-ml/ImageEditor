"""Example: Using negative prompts and IP Adapter."""

import logging
from src.services import PromptBuilder

logging.basicConfig(level=logging.INFO)


def example_negative_prompts():
    """Example showing negative prompt usage."""
    print("\n=== Negative Prompts Example ===\n")
    
    builder = PromptBuilder({})
    
    # Example 1: Using a single preset
    print("1. Single Preset:")
    negative = builder.build_negative_prompt(presets=["quality"])
    print(f"   {negative}\n")
    
    # Example 2: Combining multiple presets
    print("2. Multiple Presets:")
    negative = builder.build_negative_prompt(
        presets=["quality", "artifact", "professional"]
    )
    print(f"   {negative}\n")
    
    # Example 3: Mixing presets with custom terms
    print("3. Presets + Custom Terms:")
    negative = builder.build_negative_prompt(
        presets=["quality", "furniture"],
        custom_terms=["dark room", "small space", "overcrowded"]
    )
    print(f"   {negative}\n")
    
    # Example 4: Custom preset
    print("4. Custom Negative Preset:")
    builder.add_negative_preset(
        "interior_specific",
        "cramped, poky, narrow, confined, dingy, gloomy"
    )
    negative = builder.build_negative_prompt(presets=["interior_specific"])
    print(f"   {negative}\n")


def example_prompt_pairs():
    """Example showing prompt pair usage."""
    print("\n=== Prompt Pairs Example ===\n")
    
    builder = PromptBuilder({})
    
    # Example 1: Build from template (includes negative by default)
    print("1. From Template:")
    pair = builder.build_from_template(
        "living_room",
        style="scandinavian minimalist",
        furniture="light wood furniture",
        lighting="natural daylight",
        decor="plants and white textiles",
        negative_preset="quality"
    )
    print(f"   Positive: {pair.positive}")
    print(f"   Negative: {pair.negative}\n")
    
    # Example 2: Build pair from custom text
    print("2. From Custom Text:")
    pair = builder.build_prompt_pair(
        "A luxurious master bedroom with marble accents and elegant drapes",
        negative_presets=["quality", "artifact", "professional"],
        negative_custom=["cheap", "tacky", "overdone"]
    )
    print(f"   Positive: {pair.positive}")
    print(f"   Negative: {pair.negative}\n")


def example_ip_adapter():
    """Example showing IP Adapter functionality."""
    print("\n=== IP Adapter Example ===\n")
    
    # Note: This example assumes you have reference images
    # Create dummy files for demonstration
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy reference images
        ref1 = Path(tmpdir) / "modern_style.jpg"
        ref2 = Path(tmpdir) / "vintage_style.jpg"
        ref1.write_bytes(b"dummy")
        ref2.write_bytes(b"dummy")
        
        builder = PromptBuilder({})
        
        # Register multiple style references
        print("1. Registering IP Adapters:")
        builder.register_ip_adapter(
            "modern_style",
            str(ref1),
            weight=0.8,
            ip_adapter_type="plus"
        )
        builder.register_ip_adapter(
            "vintage_style",
            str(ref2),
            weight=0.7,
            ip_adapter_type="plus"
        )
        print(f"   Registered: {builder.list_ip_adapters()}\n")
        
        # Build with IP Adapter
        print("2. Building Configuration with IP Adapter:")
        config = builder.build_with_ip_adapter(
            prompt="A living room with similar design sensibility",
            ip_adapter_name="modern_style",
            negative_presets=["quality", "artifact"],
            negative_custom=["different style"]
        )
        print(f"   Prompt: {config['prompt']}")
        print(f"   Negative: {config['negative_prompt']}")
        print(f"   IP Adapter Type: {config['ip_adapter']['ip_adapter_type']}")
        print(f"   IP Adapter Weight: {config['ip_adapter']['weight']}\n")
        
        # Retrieve adapter
        print("3. Retrieving Adapter Config:")
        adapter = builder.get_ip_adapter("modern_style")
        print(f"   Image: {adapter.image_path}")
        print(f"   Weight: {adapter.weight}")
        print(f"   Scale: {adapter.scale}\n")


def example_recording():
    """Example showing prompt recording and export."""
    print("\n=== Prompt Recording Example ===\n")
    
    from pathlib import Path
    import tempfile
    import json
    
    builder = PromptBuilder({})
    
    # Record some prompts
    print("Recording prompts...")
    
    pair1 = builder.build_prompt_pair(
        "Modern living room",
        negative_presets=["quality"],
        negative_custom=["dated"]
    )
    
    builder.record_prompt(
        prompt=pair1.positive,
        pipeline_type="text2img",
        parameters={"guidance_scale": 7.5, "steps": 50, "seed": 42},
        negative_prompt=pair1.negative
    )
    
    pair2 = builder.build_prompt_pair(
        "Luxury bedroom",
        negative_presets=["quality", "artifact"],
        negative_custom=["tiny room", "cheap look"]
    )
    
    builder.record_prompt(
        prompt=pair2.positive,
        pipeline_type="text2img",
        parameters={"guidance_scale": 8.0, "steps": 50, "seed": 123},
        negative_prompt=pair2.negative
    )
    
    # Get history
    history = builder.get_prompt_history()
    print(f"\nRecorded {len(history)} prompts:")
    for i, entry in enumerate(history, 1):
        print(f"\n{i}. Prompt: {entry['prompt'][:50]}...")
        print(f"   Type: {entry['pipeline']}")
        print(f"   Negative: {entry.get('negative_prompt', 'None')[:40]}...")
    
    # Export history
    with tempfile.TemporaryDirectory() as tmpdir:
        export_path = Path(tmpdir) / "prompt_history.json"
        builder.export_prompt_history(str(export_path))
        
        print(f"\n\nExported to {export_path}")
        with open(export_path) as f:
            exported = json.load(f)
            print(f"Exported {len(exported)} entries")


def example_loading_from_file():
    """Example showing loading prompts from YAML file."""
    print("\n=== Loading from Configuration File ===\n")
    
    builder = PromptBuilder({})
    
    # Load templates from YAML
    print("Loading templates from configs/prompts.yaml...")
    builder.load_templates_from_file("configs/prompts.yaml")
    
    print(f"Loaded prompt templates")
    print(f"Loaded negative templates\n")
    
    # Use loaded templates
    print("Using loaded templates:")
    try:
        pair = builder.build_from_template(
            "modern_living",
            style="minimalist",
            furniture="contemporary",
            lighting="natural",
            decor="plants",
            negative_preset="quality_assurance"
        )
        print(f"Positive: {pair.positive}")
        print(f"Negative: {pair.negative}")
    except Exception as e:
        print(f"Note: Templates require matching variables: {e}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("PROMPT BUILDER EXAMPLES - Negative Prompts & IP Adapter")
    print("="*60)
    
    example_negative_prompts()
    example_prompt_pairs()
    example_ip_adapter()
    example_recording()
    example_loading_from_file()
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
