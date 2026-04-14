# SAM Integration Guide

## Overview

This project now includes integration for **Segment Anything Model (SAM)**, Facebook's powerful image segmentation model. SAM enables automatic mask generation from point prompts, bounding boxes, or full image segmentation.

## Installation

### 1. Install SAM Package

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 2. Additional Dependencies

Make sure you have the required dependencies:

```bash
pip install -r requirements.txt
```

Key requirements for SAM:
- `torch`
- `torchvision`
- `numpy`
- `pillow`
- `opencv-python`

### 3. Download SAM Models

The first time you use SAM, it will automatically download the model. You can pre-download specific models:

```python
from segment_anything import sam_model_registry
from huggingface_hub import hf_hub_download

# Download a specific model
checkpoint = hf_hub_download(
    repo_id="facebook/sam-vit-huge",
    filename="sam_vit_h_4b8939.pth"
)
sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
```

## Model Types

SAM provides 4 model variants with different trade-offs:

| Model | Size | Speed | VRAM | Best For |
|-------|------|-------|------|----------|
| `vit_h` (huge) | ~2.5GB | Slowest | ~11GB | Highest quality |
| `vit_l` (large) | ~1.3GB | Medium | ~10GB | Good balance |
| `vit_b` (base) | ~375MB | Fast | ~6GB | Most use cases (recommended) |
| `mobile_sam` | ~40MB | Fastest | ~1GB | Limited resources |

## Usage

### 1. Basic Mask Generation with Points

Generate masks by clicking points on an image:

```bash
python scripts/run_sam.py \
  --image examples/sample_inputs/room.jpg \
  --mode points \
  --points "100,100 200,200" \
  --output outputs/mask.png
```

### 2. Box-Based Segmentation

Generate masks from bounding boxes:

```bash
python scripts/run_sam.py \
  --image examples/sample_inputs/room.jpg \
  --mode box \
  --box "50,50,300,300" \
  --output outputs/mask.png
```

### 3. SAM-Assisted Inpainting

Generate a mask with SAM and immediately use it for inpainting:

```bash
python scripts/run_sam_inpaint.py \
  --image room.jpg \
  --prompt "add a modern painting on the wall" \
  --sam-mode points \
  --points "150,100 200,150" \
  --guidance-scale 7.5 \
  --num-inference-steps 50
```

### 4. Python API

Use SAM directly in your code:

```python
from src.services import GenerationService
from src.utils.io import load_config

# Load configuration
config = load_config("configs/sam.yaml")

# Create service
service = GenerationService(config)

# Generate mask with points
mask = service.generate_mask_with_sam(
    image_path="room.jpg",
    mode="points",
    points=[(100, 100), (200, 200)],
    model_type="vit_b"
)
mask.save("output_mask.png")

# Or use SAM-assisted inpainting
images = service.generate_inpaint_with_sam(
    image_path="room.jpg",
    sam_model_type="vit_b",
    sam_mode="points",
    points=[(150, 100), (200, 150)],
    prompt="add a modern painting",
    num_inference_steps=50
)

for i, img in enumerate(images):
    img.save(f"inpainted_{i}.png")
```

## Configuration

Edit `configs/sam.yaml` to customize SAM behavior:

```yaml
# Model type
model_type: vit_b  # Options: vit_h, vit_l, vit_b, mobile_sam

# Segmentation settings
segmentation:
  mode: points  # Options: points, box, all

# Post-processing of masks
mask_post_processing:
  threshold: 127          # Binary threshold
  fill_holes: true        # Fill small holes in mask
  feather_radius: 0       # Smooth edges
  dilate_kernel: 0        # Dilate mask
  erode_kernel: 0         # Erode mask
```

## Advanced Usage

### Interactive Segmentation

For interactive point-based segmentation:

```python
from src.services.sam_segmentation import SAMSegmentation
from PIL import Image

# Initialize SAM
sam = SAMSegmentation(model_type="vit_b", device="cuda")
sam.load_model()

# Load image
image = Image.open("room.jpg")
sam.set_image(image)

# Generate mask from points
mask1 = sam.generate_mask_from_points([(100, 100), (200, 200)], positive_points=True)
mask1.save("mask1.png")

# Generate mask from box
mask2 = sam.generate_mask_from_box((50, 50, 300, 300))
mask2.save("mask2.png")
```

### Combining SAM with Other Pipelines

Use SAM masks with other generation methods:

```python
service = GenerationService(config)

# SAM mask generation
mask = service.generate_mask_with_sam(
    image_path="room.jpg",
    mode="points",
    points=[(150, 150)],
)

# Use mask for inpainting
images = service.generate_inpaint(
    image_path="room.jpg",
    mask_path="temp_mask.png",  # Save and load mask
    prompt="modern decoration",
    guidance_scale=7.5
)
```

### Batch Processing

Process multiple images:

```python
import os
from pathlib import Path

image_dir = Path("examples/sample_inputs")
output_dir = Path("outputs/masks")
output_dir.mkdir(parents=True, exist_ok=True)

service = GenerationService(config)

for image_path in image_dir.glob("*.jpg"):
    mask = service.generate_mask_with_sam(
        image_path=str(image_path),
        mode="all",  # Auto-segment all objects
        model_type="vit_b"
    )
    output_path = output_dir / f"{image_path.stem}_mask.png"
    mask.save(output_path)
    print(f"Processed {image_path.stem}")
```

## Performance Tips

### Memory Optimization

1. **Use smaller models for limited VRAM:**
   ```bash
   python scripts/run_sam.py --image img.jpg --model-type vit_b  # ~6GB VRAM
   python scripts/run_sam.py --image img.jpg --model-type mobile_sam  # ~1GB VRAM
   ```

2. **Enable CPU processing if needed:**
   Edit `configs/sam.yaml`:
   ```yaml
   device: cpu  # Slower but uses no GPU VRAM
   ```

### Speed Optimization

1. Use `mobile_sam` for faster inference
2. Pre-resize images to smaller dimensions
3. Use batch processing to reuse loaded models

## Troubleshooting

### ImportError: No module named 'segment_anything'

Install SAM:
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### CUDA out of memory

1. Use a smaller model (`vit_b` or `mobile_sam`)
2. Resize input images to smaller dimensions
3. Run on CPU (slow but works)

### Mask quality issues

1. Use larger model (`vit_h`)
2. Provide more precise point locations
3. Use box prompts for better accuracy

## Integration Points

### With Inpainting Pipeline

SAM masks can be used directly with the inpainting pipeline for automated mask generation.

### With ControlNet

Combine SAM masks with ControlNet for guided generation:

```python
# Generate mask with SAM
mask = service.generate_mask_with_sam(
    image_path="room.jpg",
    mode="points",
    points=[(150, 150)]
)

# Create control map from mask
# (implementation depends on your control type)

# Generate with ControlNet guidance
images = service.generate_controlnet(
    control_image_path="room.jpg",
    prompt="modern interior",
    control_type="canny"
)
```

## Pipeline Choices

SAM can be invoked through:

1. **Direct SAM Pipeline:**
   ```python
   service.get_sam_pipeline("vit_b")
   ```

2. **SAM+Inpaint Pipeline:**
   ```python
   service.generate_inpaint_with_sam(...)
   ```

3. **Main CLI with `sam` pipeline:**
   ```bash
   python src/main.py --pipeline sam --config configs/sam.yaml
   ```

## Future Enhancements

Potential improvements:

1. **Text-based segmentation:** Combine with CLIP/Grounding DINO
2. **Interactive UI:** Gradio-based point selection
3. **Automatic object detection:** Use SAM for all objects
4. **Fine-tuning:** Train SAM on custom datasets
5. **Multi-mask composition:** Combine multiple SAM masks

## References

- **SAM Paper:** [Segment Anything](https://arxiv.org/abs/2304.02643)
- **GitHub:** https://github.com/facebookresearch/segment-anything
- **Model Cards:** https://huggingface.co/facebook/sam-vit-huge
