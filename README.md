# Interior Design Diffusion

A production-ready diffusion-model pipeline for interior design generation and editing in Python.

## Features

- **Text-to-Image**: Generate interior design images from text prompts
- **Image-to-Image**: Redesign existing rooms with controlled transformation
- **Inpainting**: Replace selected regions with masked editing
- **ControlNet Support**: Structural control using depth/canny edge detection
- **Prompt Management**: Reusable templates and example prompt library
- **Gradio UI**: Optional web interface for easy interaction

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended) or CPU fallback
- Git

### Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd interior-design-diffusion
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your preferred settings
```

### Usage

#### Text-to-Image Generation

```bash
python scripts/run_text2img.py --prompt "Modern living room with minimalist design" --num-images 4
```

#### Image-to-Image Redesign

```bash
python scripts/run_img2img.py --image path/to/room.jpg --prompt "Warm, cozy bedroom" --strength 0.7
```

#### Inpainting

```bash
python scripts/run_inpaint.py --image path/to/room.jpg --mask path/to/mask.jpg --prompt "Replace with wooden furniture"
```

#### ControlNet

```bash
python scripts/run_controlnet.py --prompt "Office space" --control-image path/to/depth.jpg --control-type depth
```

#### Interactive UI

```bash
python src/ui/gradio_app.py
```

## Project Structure

```
interior-design-diffusion/
├── configs/              # YAML configuration files
├── src/                  # Main source code
│   ├── pipelines/        # Diffusion pipeline implementations
│   ├── services/         # Business logic and utilities
│   ├── utils/            # Helper functions
│   └── ui/               # Gradio interface
├── scripts/              # Command-line entry points
├── examples/             # Example prompts and sample images
├── tests/                # Unit tests
└── outputs/              # Generated images (git-ignored)
```

## Configuration

All pipelines are configurable via YAML files in `configs/`:

- `default.yaml`: Global defaults
- `text2img.yaml`: Text-to-image parameters
- `img2img.yaml`: Image-to-image parameters
- `inpaint.yaml`: Inpainting parameters
- `controlnet.yaml`: ControlNet configuration

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting and Linting

```bash
black src/ tests/ scripts/
isort src/ tests/ scripts/
flake8 src/ tests/ scripts/
```

### Installing Development Dependencies

```bash
pip install -e ".[dev]"
```

## Technical Stack

- **PyTorch**: Deep learning framework
- **Hugging Face Diffusers**: Diffusion model implementations
- **Transformers**: CLIP text encoder and other models
- **Gradio**: Web-based UI
- **Accelerate**: Multi-GPU and mixed precision support

## Constraints & Design Decisions

- Prefer Hugging Face `diffusers` pipelines over custom implementations
- Local inference first; distributed infrastructure out of scope for v1
- GPU acceleration preferred with CPU fallback
- Modular architecture for easy model and pipeline swaps
- Configuration-driven for minimal source code changes

## Non-Goals (v1)

- Full model training from scratch
- Distributed inference infrastructure
- Mobile deployment
- 3D scene generation
- Multi-user web platform
- Advanced dataset curation

## Future Enhancements

- Multiple ControlNet modes
- Model fine-tuning support
- Batch processing
- Community prompt library
- Performance optimizations

## Contributing

TBD

## License

MIT
