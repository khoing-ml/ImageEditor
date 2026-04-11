# SDXL Dual Prompting & Weighted Prompts Guide

## Overview

This guide covers the implementation of the prompting improvements defined in `plans/prompts.md`:

1. **SDXL Baseline** - Using SDXL for better quality
2. **Compel Integration** - Prompt weighting support
3. **Dual Prompting** - SDXL's `prompt` / `prompt_2` split
4. **ControlNet Depth** - Structure preservation
5. **IP-Adapter** - Style reference images

---

## Architecture

### SDXL Dual Prompt Structure

SDXL models use two text encoders:

- **Encoder 1** (`prompt`): Spatial information, composition, objects
- **Encoder 2** (`prompt_2`): Style, materials, lighting, realism

This separation reduces prompt overload and improves semantic clarity.

### Components

```python
from src.services import (
    DualPromptBuilder,
    InteriorDesignPromptBuilder,
    PromptWeightingParser,
    DualPrompt,
)
```

---

## Prompt Weighting

### PromptWeightingParser

Implements Compel-style weighting syntax.

```python
from src.services import PromptWeightingParser

parser = PromptWeightingParser()

# Parse weighted prompt
prompt = "(natural daylight)1.5 modern living room with (oak wood)1.3"
tokens = parser.parse_weighted_prompt(prompt)
# Output: [('natural daylight', 1.5), ('modern', 1.0), ('living', 1.0), ...]

# Emphasize token
emphasized = parser.emphasize_token("important concept", weight=1.7)
# Output: "(important concept)1.7"

# De-emphasize token
de_emphasized = parser.de_emphasize_token("unwanted element", weight=0.5)
# Output: "(unwanted element)0.5"
```

#### Weight Syntax
- `(text)weight` - Apply weight to token
- Default range: 0.5 - 2.0
- 1.0 = neutral
- >1.0 = emphasize
- <1.0 = de-emphasize

#### Best Practices

1. **Emphasize key concepts**: `(natural lighting)1.5`
2. **De-emphasize distractions**: `(clutter)0.3`
3. **Keep weights moderate**: 0.7 - 1.5 range works well
4. **Combine sparingly**: Too many weights reduce clarity

---

## Dual Prompting

### DualPrompt

Container for SDXL's dual prompt structure.

```python
from src.services import DualPrompt

pair = DualPrompt(
    prompt="modern living room, sofa, large windows",
    prompt_2="scandinavian style, oak wood, warm natural light, ultra realistic",
    negative_prompt="cluttered, dark, cheap",
    negative_prompt_2="plastic, artificial, low quality"
)

# Access components
print(pair.prompt)          # Spatial info
print(pair.prompt_2)        # Style info
print(pair.negative_prompt) # Primary negative terms
```

### DualPromptBuilder

Build structured dual prompts from components.

```python
from src.services import DualPromptBuilder

builder = DualPromptBuilder()

pair = builder.build_from_components(
    room_type="living room",
    layout="open spacious layout",
    focal_objects="sofa, coffee table",
    style="minimalist contemporary",
    materials="light oak wood",
    lighting="warm natural daylight",
    color_palette="neutral beige",
    photography_style="interior photography",
    realism_tag="ultra realistic"
)

# Get as dict for pipeline
data = builder.to_dict(pair)
# Returns: {"prompt": "...", "prompt_2": "...", "negative_prompt": "...", ...}
```

#### Method: `build_from_components()`

**Parameters:**
- `room_type` - Type of room
- `layout` - Spatial arrangement
- `focal_objects` - Main objects (comma-separated)
- `style` - Design style
- `materials` - Material descriptions
- `lighting` - Lighting setup
- `color_palette` - Color scheme
- `photography_style` - How it should be photographed
- `realism_tag` - Quality and realism level
- `emphasize` - List of components to emphasize
- `de_emphasize` - List of components to de-emphasize
- `negative_prompts` - Custom negative terms

**Returns:** `DualPrompt` object

#### Method: `emphasize_component()`

Add weight to a component in existing prompt.

```python
emphasized = builder.emphasize_component(
    original_pair,
    component="natural lighting",
    weight=1.7
)
```

#### Method: `split_prompt_by_role()`

Intelligently split a single prompt into dual structure.

```python
pair = builder.split_prompt_by_role(
    "modern living room with sofa, natural lighting, oak wood, realistic"
)
# Automatically separates spatial from style info
```

---

## Interior Design Builder

### InteriorDesignPromptBuilder

Specialized builder for interior design with presets.

```python
from src.services import InteriorDesignPromptBuilder

builder = InteriorDesignPromptBuilder()

# Build using presets
pair = builder.build_interior_prompt(
    room="living_room",          # Resolved to "living room"
    layout="open",               # Resolved to "open layout"
    focal_objects="sofa, table", # Custom text
    style="modern",              # Resolved to full style text
    materials="natural_wood",    # Resolved to material description
    lighting="natural",          # Resolved to full lighting desc
    color_palette="neutral",     # Resolved to full palette desc
    custom_negative=["dark", "cluttered"]
)
```

### Available Presets

#### Room Types
```python
builder.ROOM_TYPES
# 'living_room', 'bedroom', 'kitchen', 'bathroom', 'office', 'dining', 'entryway', 'study'
```

#### Styles
```python
builder.STYLE_PALETTES
# 'modern', 'scandinavian', 'industrial', 'luxury', 'bohemian', 'traditional',
# 'japandi', 'mid_century', 'coastal', 'farmhouse'
```

#### Materials
```python
builder.MATERIAL_PALETTES
# 'natural_wood', 'concrete', 'marble', 'velvet', 'wool', 'leather',
# 'glass', 'rattan', 'fabric'
```

#### Lighting
```python
builder.LIGHTING_OPTIONS
# 'natural', 'ambient', 'warm', 'cool', 'dramatic', 'moody', 'bright', 'mixed'
```

#### Colors
```python
builder.COLOR_PALETTES
# 'neutral', 'warm', 'cool', 'jewel', 'monochrome', 'pastel', 'bold', 'muted'
```

### Example Usage

```python
# Modern minimalist living room
pair = builder.build_interior_prompt(
    room="living_room",
    layout="open",
    focal_objects="low sofa, glass table, minimalist art",
    style="modern",
    materials="natural_wood",
    lighting="natural",
    color_palette="neutral"
)
```

---

## Pipeline Integration

### SDXL Text-to-Image

```python
from src.services import GenerationService, InteriorDesignPromptBuilder

service = GenerationService(config)
builder = InteriorDesignPromptBuilder()

# Build prompt
pair = builder.build_interior_prompt(
    room="bedroom",
    layout="spacious",
    focal_objects="bed, nightstands",
    style="luxury",
    materials="velvet",
    lighting="ambient",
    color_palette="jewel"
)

# Convert to pipeline dict
prompt_data = builder.to_dict(pair)

# Generate
images = service.generate_text2img(
    prompt=prompt_data["prompt"],
    prompt_2=prompt_data["prompt_2"],
    negative_prompt=prompt_data["negative_prompt"],
    negative_prompt_2=prompt_data["negative_prompt_2"],
    guidance_scale=7.5,
    num_inference_steps=50,
    height=1024,
    width=1024
)
```

### ControlNet Depth

```python
from src.pipelines.controlnet_depth import ControlNetDepthPipeline
from src.services import InteriorDesignPromptBuilder
from PIL import Image

# Initialize pipeline
depth_pipeline = ControlNetDepthPipeline(
    model_id="stabilityai/stable-diffusion-xl-base-1.0",
    device="cuda",
    use_sdxl=True
)

# Load depth map
depth_image = Image.open("room_depth.jpg")

# Build prompt
builder = InteriorDesignPromptBuilder()
pair = builder.build_interior_prompt(...)

# Generate with depth control
images = depth_pipeline.generate(
    prompt=pair.prompt,
    prompt_2=pair.prompt_2,
    depth_image=depth_image,
    negative_prompt=pair.negative_prompt,
    negative_prompt_2=pair.negative_prompt_2,
    controlnet_conditioning_scale=1.0  # Strength of depth control
)
```

### With IP-Adapter

```python
from src.services import InteriorDesignPromptBuilder

builder = InteriorDesignPromptBuilder()

# Register reference image
builder.register_ip_adapter(
    name="luxury_style",
    image_path="reference_luxury_room.jpg",
    weight=0.8,
    ip_adapter_type="plus"
)

# Build prompt with IP Adapter
config = builder.build_with_ip_adapter(
    prompt="A bedroom in the same style",
    ip_adapter_name="luxury_style",
    negative_presets=["quality", "artifact"],
    negative_custom=["different style"]
)

# Use config in pipeline
images = service.generate_text2img(
    prompt=config["prompt"],
    prompt_2=config["prompt_2"],
    negative_prompt=config["negative_prompt"],
    ip_adapter_config=config["ip_adapter"]
)
```

---

## Configuration

### SDXL Models

Updated `configs/default.yaml`:

```yaml
# Using SDXL as default
text2img_model_id: stabilityai/stable-diffusion-xl-base-1.0
img2img_model_id: stabilityai/stable-diffusion-xl-img2img-1.0
inpaint_model_id: stabilityai/stable-diffusion-xl-1.0

# SDXL improves output quality
default_height: 1024
default_width: 1024

# ControlNet depth
controlnet_depth_model_id: stabilityai/stable-diffusion-xl-base-1.0
controlnet_model_id: lllyasviel/control_v11f1p_sd15_depth
```

### Interior Design Presets

See `configs/interior_design_prompts.yaml` for complete room templates.

---

## Examples

### Run prompt examples:
```bash
python examples/dual_prompt_examples.py
```

### Run prompt builder examples:
```bash
python examples/prompt_examples.py
```

---

## Testing

Run comprehensive test suite:

```bash
# Test prompt weighting
pytest tests/test_prompt_weighting.py -v

# Test prompt builder
pytest tests/test_prompt_builder.py -v

# Run all tests
pytest tests/ -v
```

---

## Best Practices

### 1. Dual Prompt Split

Always split prompts by role:

✅ **Good:**
```
prompt: "modern bedroom, large bed, floor-to-ceiling windows"
prompt_2: "contemporary luxury style, natural wood, warm soft lighting"
```

❌ **Poor:**
```
prompt: "modern contemporary luxury bedroom with large bed, floor-to-ceiling windows, natural wood, warm soft lighting"
```

### 2. Emphasis Strategy

Emphasize what matters most:

```python
# Quality comes first
builder.emphasize_component(pair, "ultra realistic", weight=1.7)
builder.emphasize_component(pair, "professional lighting", weight=1.5)

# De-emphasize problems
builder.de_emphasize_token("cluttered", weight=0.3)
```

### 3. Material and Lighting Detail

Be specific in materials and lighting:

✅ **Good:**
- Materials: "white oak flooring, marble countertops, brass fixtures"
- Lighting: "warm natural daylight from large windows, accent wall lighting"

❌ **Poor:**
- Materials: "nice materials"
- Lighting: "good lighting"

### 4. Photography Language

Include realism and photography style:

```python
photography_style="interior photography"  # What it should look like
realism_tag="ultra realistic, 8k quality" # How real it should be
```

### 5. Negative Prompts

Use presets for consistency:

```python
from src.services import PromptBuilder

prompt_builder = PromptBuilder(config)
negative = prompt_builder.build_negative_prompt(
    presets=["quality", "artifact", "professional"],
    custom_terms=["wrong style"]
)
```

---

## Troubleshooting

### Issue: Generated images don't match style

**Solution:** Use IP-Adapter with reference image + dual prompting

```python
config = builder.build_with_ip_adapter(
    prompt="Your design prompt",
    ip_adapter_name="style_reference",
    negative_presets=["artifact"]
)
```

### Issue: Structure not preserved in redesign

**Solution:** Use ControlNet depth with structure control

```python
depth_pipeline.generate(
    ...,
    controlnet_conditioning_scale=1.2  # Increase control strength
)
```

### Issue: Prompts feel overloaded

**Solution:** Use prompt splitting and dual prompting

```python
pair = builder.split_prompt_by_role(full_prompt)
# Automatically separates spatial from style info
```

---

## Reference

- **Plan:** [plans/prompts.md](/home/khoi/Code/ImageEditor/plans/prompts.md)
- **Templates:** [configs/interior_design_prompts.yaml](/home/khoi/Code/ImageEditor/configs/interior_design_prompts.yaml)
- **Examples:** [examples/dual_prompt_examples.py](/home/khoi/Code/ImageEditor/examples/dual_prompt_examples.py)
- **Tests:** [tests/test_prompt_weighting.py](/home/khoi/Code/ImageEditor/tests/test_prompt_weighting.py)
