# Enhanced Prompt System Documentation

## Overview

The enhanced prompt system provides comprehensive support for:
- **Prompt Pairs**: Positive and negative prompts together
- **Negative Prompt Presets**: Pre-built negative prompt templates
- **IP Adapter Support**: Image-based prompting using IP Adapter models
- **Template Management**: Easy-to-use prompt templating system

## Core Classes

### PromptPair
Container for positive and negative prompt pairs.

```python
from src.services.prompt_builder import PromptPair

pair = PromptPair(
    positive="A modern living room with elegant furniture",
    negative="blurry, low quality, distorted"
)

print(pair)  # Shows both positive and negative
```

### IPAdapterConfig
Configuration for IP Adapter with validation.

```python
from src.services.prompt_builder import IPAdapterConfig

config = IPAdapterConfig(
    image_path="path/to/reference.jpg",
    weight=0.8,
    scale=1.0,
    conditioning_scale=1.5,
    ip_adapter_type="plus"  # or 'plus_face', 'full', 'full_face'
)
config.validate()  # Raises errors if invalid
```

## PromptBuilder API

### Building Prompts from Templates

```python
from src.services.prompt_builder import PromptBuilder

builder = PromptBuilder(config={})

# Build a prompt pair with default negative preset
pair = builder.build_from_template(
    "living_room",
    style="modern",
    furniture="minimalist seating",
    lighting="warm mood lighting",
    decor="plant accents",
    negative_preset="quality"  # Optional
)

print(pair.positive)   # Generated positive prompt
print(pair.negative)   # Generated negative prompt
```

### Negative Prompt Presets

Available presets:
- `quality`: Addresses image quality issues
- `artifact`: Prevents watermarks, text, logos
- `anatomy`: Handles deformations and proportions
- `professional`: Ensures professional appearance
- `style`: Avoids unwanted artistic styles
- `furniture`: Ensures good furniture quality

```python
# Use a single preset
negative = builder.build_negative_prompt(presets=["quality"])

# Combine multiple presets
negative = builder.build_negative_prompt(
    presets=["quality", "artifact", "professional"]
)

# Mix presets with custom terms
negative = builder.build_negative_prompt(
    presets=["quality"],
    custom_terms=["ugly", "bad design", "cheap materials"]
)
```

### Custom Prompts with Negative Terms

```python
# Build a prompt pair from custom text
pair = builder.build_prompt_pair(
    positive="A luxurious bedroom with marble accents",
    negative_presets=["artifact", "professional"],
    negative_custom=["dark", "gloomy", "cheap looking"]
)
```

### Adding Custom Presets

```python
# Add a domain-specific negative preset
builder.add_negative_preset(
    "furniture_quality",
    "broken furniture, worn out, torn fabrics, stained, damaged"
)

# Use it later
negative = builder.build_negative_prompt(
    presets=["furniture_quality"]
)
```

## IP Adapter Usage

IP Adapter allows using reference images to guide generation.

### Registering an IP Adapter

```python
# Register a reference image as an IP Adapter
config = builder.register_ip_adapter(
    name="living_room_style",
    image_path="examples/reference_living_room.jpg",
    weight=0.8,           # How much to influence generation (0-1)
    scale=1.0,            # Overall scale
    conditioning_scale=1.5,  # ControlNet-style conditioning
    ip_adapter_type="plus"     # "plus", "plus_face", "full", "full_face"
)
```

### Building with IP Adapter

```python
# Create a full generation config with IP Adapter
config = builder.build_with_ip_adapter(
    prompt="A modern minimalist living room",
    ip_adapter_name="living_room_style",
    negative_presets=["artifact", "professional"],
    negative_custom=["cluttered"]
)

# Returns:
# {
#     "prompt": "A modern minimalist living room",
#     "negative_prompt": "... combined negative prompt ...",
#     "ip_adapter": {
#         "image_path": "examples/reference_living_room.jpg",
#         "weight": 0.8,
#         "scale": 1.0,
#         "conditioning_scale": 1.5,
#         "ip_adapter_type": "plus"
#     }
# }
```

### Managing Multiple IP Adapters

```python
# Register multiple adapters
builder.register_ip_adapter("modern_style", "path/to/modern.jpg")
builder.register_ip_adapter("vintage_style", "path/to/vintage.jpg")
builder.register_ip_adapter("minimal_style", "path/to/minimal.jpg")

# List all registered adapters
adapters = builder.list_ip_adapters()
print(adapters)  # ['modern_style', 'vintage_style', 'minimal_style']

# Get specific adapter config
modern_config = builder.get_ip_adapter("modern_style")

# Remove an adapter
builder.remove_ip_adapter("vintage_style")
```

## Loading Prompts from Configuration Files

Create a YAML file with prompt templates and negative presets:

```yaml
# configs/prompts.yaml
templates:
  modern_living:
    "A modern living room with {furniture}, {lighting}, and {decor}"
  luxury_bedroom:
    "A luxurious bedroom with {furniture}, designer {lighting}, and {color_scheme}"

negative_templates:
  quality: "blurry, low quality, pixelated, distorted"
  professional: "amateurish, unprofessional, poorly lit, overexposed"
```

Then load them:

```python
builder = PromptBuilder(config={})
builder.load_templates_from_file("configs/prompts.yaml")

# Use the loaded templates
pair = builder.build_from_template(
    "modern_living",
    furniture="sleek sofas",
    lighting="warm lighting",
    decor="minimalist art"
)
```

## Recording and Exporting Prompts

```python
# Record a prompt for logging
builder.record_prompt(
    prompt="A modern living room",
    pipeline_type="text2img",
    parameters={
        "guidance_scale": 7.5,
        "num_inference_steps": 50,
        "height": 768,
        "width": 768
    },
    negative_prompt="blurry, low quality",
    ip_adapter_config={
        "weight": 0.8,
        "type": "plus"
    }
)

# Get all recorded prompts
history = builder.get_prompt_history()

# Export to JSON
builder.export_prompt_history("logs/prompt_history.json")
```

## Best Practices

### 1. Use Presets for Common Issues

```python
# Good: Use presets for known quality concerns
negative = builder.build_negative_prompt(
    presets=["quality", "artifact", "professional"]
)

# Instead of manually writing everything:
negative = builder.build_negative_prompt(
    custom_terms=["blurry", "low quality", "watermark", ...]
)
```

### 2. Domain-Specific Negative Presets

```python
# Create domain-specific presets for interior design
builder.add_negative_preset(
    "interior_design",
    "cheap materials, tacky decor, clashing colors, oversized furniture"
)

builder.add_negative_preset(
    "furniture",
    "broken, damaged, worn out, stained, torn fabrics"
)
```

### 3. Combine IP Adapter with Negative Prompts

```python
# IP Adapter provides style, negative prompts ensure quality
config = builder.build_with_ip_adapter(
    prompt="A living room in the same style",
    ip_adapter_name="reference_style",
    negative_presets=["quality", "artifact", "professional"]
)
```

### 4. Template Composition

```python
# Create flexible templates with many placeholders
template = "A {time_of_day} {room_type} with {style} furniture, {lighting}, {color_scheme}, and {mood}"

builder.add_template("detailed_interior", template)

pair = builder.build_from_template(
    "detailed_interior",
    time_of_day="evening",
    room_type="bedroom",
    style="modern minimalist",
    lighting="soft warm lighting",
    color_scheme="neutral earth tones",
    mood="cozy and serene"
)
```

## Advanced Usage

### Dynamic Negative Prompt Composition

```python
# Build context-aware negative prompts
def get_negative_prompt_for_style(style: str, quality_level: str):
    presets = ["quality"]  # Always include quality
    
    if style == "minimalist":
        presets.append("clutter")
    elif style == "luxury":
        presets.append("cheap")
    
    if quality_level == "professional":
        presets.append("professional")
    
    return builder.build_negative_prompt(presets=presets)

negative = get_negative_prompt_for_style("luxury", "professional")
```

### Pipeline Integration

```python
from src.services import GenerationService, PromptPair

service = GenerationService(config)

# Use prompt pairs in generation
pair = builder.build_prompt_pair(
    "Modern living room with natural light",
    negative_presets=["quality", "artifact"]
)

# Pass to pipeline
images = service.generate_text2img(
    prompt=pair.positive,
    negative_prompt=pair.negative,
    **generation_params
)
```

## Prompt Tips for Interior Design

### Effective Positive Prompts

1. **Include Style**: "modern", "minimalist", "industrial", "luxury"
2. **Add Materials**: "wood", "marble", "concrete", "brass"
3. **Describe Lighting**: "soft warm lighting", "natural daylight", "dramatic accent lighting"
4. **Specify Mood**: "cozy", "energetic", "serene", "professional"
5. **Mention Colors**: "neutral palette", "warm earth tones", "jewel tones"

### Effective Negative Prompts

Use presets to avoid:
- Quality artifacts: blurry, pixelated, distorted
- Unwanted elements: watermarks, text, logos
- Bad proportions: oversized furniture, cramped spaces
- Poor lighting: dark, overexposed, harsh shadows
- Style mismatches: cartoon, illustration, cheap looking

Example:
```python
pair = builder.build_prompt_pair(
    "A luxurious bedroom with velvet furnishings, brass fixtures, and moody lighting",
    negative_presets=["quality", "artifact", "professional"],
    negative_custom=["cheap materials", "plastic looking", "small cramped space"]
)
```

## Troubleshooting

### Issue: Generated images don't match the style

**Solution**: Use IP Adapter with a reference image

```python
builder.register_ip_adapter("my_style", "reference.jpg", weight=0.9)
config = builder.build_with_ip_adapter("Your text prompt", "my_style")
```

### Issue: Unwanted artifacts in output

**Solution**: Strengthen artifact prevention

```python
negative = builder.build_negative_prompt(
    presets=["artifact", "quality"],
    custom_terms=["watermark", "signature", "text", "border"]
)
```

### Issue: Wrong furniture styles appearing

**Solution**: Add domain-specific negative terms

```python
builder.add_negative_preset(
    "wrong_furniture",
    "plastic furniture, cheap looking, mismatched styles, outdated design"
)
```

## Performance Considerations

- Presets reduce prompt engineering overhead
- Deduplication prevents redundant negative terms
- IP Adapter provides better style control than text alone
- Combine 2-3 presets for optimal results

## References

- IP Adapter: https://github.com/tencent-ailab/IP-Adapter
- Prompt Engineering: https://huggingface.co/docs/diffusers/using-diffusers/custom_pipeline_failures
- Stable Diffusion: https://huggingface.co/models?model_type=text-to-image
