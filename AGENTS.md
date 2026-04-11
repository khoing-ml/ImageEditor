# Agents and Workflows

This document describes specialized agents and workflows for the Interior Design Diffusion project.

## Overview

Agents are autonomous task handlers that can manage specific operations or problem domains. They may invoke specialized tools or coordinate complex multi-step processes.

## Agent Types

### 1. Pipeline Manager Agent
**Responsibility**: Orchestrate generation pipelines
- Load and validate configurations
- Manage device allocation (GPU/CPU)
- Coordinate model loading and caching
- Handle generation requests and output management

### 2. Image Processing Agent
**Responsibility**: Image input/output operations
- Validate image inputs
- Generate masks and control maps
- Post-process generated images
- Manage file storage and naming

### 3. Prompt Management Agent
**Responsibility**: Handle prompt-related tasks
- Load and apply prompt templates
- Validate prompt inputs
- Log prompts with generation parameters
- Manage example prompt library

### 4. Configuration Agent
**Responsibility**: Configuration management
- Load YAML configurations
- Merge defaults with user overrides
- Validate configuration values
- Provide configuration recommendations

## Workflows

### Text-to-Image Workflow
1. Parse command-line arguments → Config Agent
2. Validate output directory → Image Processing Agent
3. Load pipeline and model → Pipeline Manager Agent
4. Generate images → Pipeline Manager Agent
5. Post-process and save → Image Processing Agent
6. Log metadata → Prompt Management Agent

### Image-to-Image Workflow
1. Parse arguments and load input image → Image Processing Agent
2. Validate image format → Image Processing Agent
3. Load configuration and model → Config Agent, Pipeline Manager Agent
4. Apply transformation → Pipeline Manager Agent
5. Save output → Image Processing Agent
6. Log parameters → Prompt Management Agent

### Inpainting Workflow
1. Load image and mask → Image Processing Agent
2. Validate mask alignment → Image Processing Agent
3. Load inpainting model → Pipeline Manager Agent
4. Generate inpainted region → Pipeline Manager Agent
5. Composite and save → Image Processing Agent
6. Log operation → Prompt Management Agent

### ControlNet Workflow
1. Prepare control image → Image Processing Agent
2. Load ControlNet model → Pipeline Manager Agent
3. Extract control features → Pipeline Manager Agent
4. Generate with control guidance → Pipeline Manager Agent
5. Save output → Image Processing Agent
6. Log control parameters → Prompt Management Agent

## Tool Permissions

Agents have access to:
- File system operations (read/write images and configs)
- Model and pipeline loading
- Device management
- Logging and metadata operations

Agents do NOT have access to:
- External API calls (beyond Hugging Face Hub)
- Network operations beyond model loading
- System-level resource allocation
