# PROMPTING_PLAN.md

## Project
Interior Design with Diffusion Models

## Purpose
Define the prompt-improvement strategy for the Interior Design pipeline.  
This document focuses on improving generation quality through:

- prompt weighting
- multi-encoder prompting
- image-based prompting
- structure conditioning
- reusable style tokens

This is not a generic LLM prompt rewriting plan.  
The goal is to improve **control, consistency, and realism** in diffusion outputs.

---

## 1. Objective

Build a prompt-conditioning stack that improves:

- interior realism
- layout consistency
- style control
- material fidelity
- lighting quality
- reproducibility

The system should support both:
- pure text generation
- guided generation using images and structural conditions

---

## 2. Strategy Overview

The prompt-improvement stack will use the following components:

1. **Base Model: SDXL**
2. **Prompt Weighting: Compel**
3. **Dual Text Encoder Prompting: SDXL `prompt` + `prompt_2`**
4. **Reference Style Guidance: IP-Adapter**
5. **Structure Preservation: ControlNet**
6. **Optional Custom Style Tokens: Textual Inversion**

---

## 3. Why This Approach

Prompt quality in diffusion systems is improved more effectively by:

- weighting important tokens
- splitting semantic load across encoders
- conditioning on reference images
- conditioning on structure
- introducing learned style tokens when needed

This is more reliable than using a generic LLM to “make prompts prettier.”

---

## 4. Components

### 4.1 Base Model — SDXL

#### Role
Use SDXL as the default image generation backbone.

#### Why
- stronger prompt understanding than SD1.5
- better composition
- better material and lighting representation
- supports dual text encoders

#### Use cases
- text-to-image generation
- image-to-image redesign
- inpainting
- style-guided generation

#### Initial model
- `stabilityai/stable-diffusion-xl-base-1.0`

---

### 4.2 Prompt Weighting — Compel

#### Role
Improve prompt precision by weighting important concepts.

#### Why
Diffusion prompts often contain multiple semantic targets:
- room type
- style
- material
- lighting
- camera
- realism tags

Without weighting, important tokens may be diluted.

#### Use cases
- emphasize lighting
- emphasize material realism
- emphasize room style
- de-emphasize secondary details

#### Examples
- `(natural daylight)1.4`
- `(light oak flooring)1.3`
- `(interior photography)1.2`

#### Goal
Use Compel as the default prompt embedding layer when prompt control matters.

---

### 4.3 SDXL Dual Prompting — `prompt` and `prompt_2`

#### Role
Split prompt information across SDXL’s two text encoders.

#### Why
This reduces prompt overload and improves semantic clarity.

#### Recommended split

##### `prompt`
Use for:
- room type
- layout
- spatial composition
- main objects

##### `prompt_2`
Use for:
- style
- materials
- lighting
- color palette
- photography language
- realism tags

#### Example

`prompt`:
- modern living room, open layout, large windows, sofa, coffee table

`prompt_2`:
- Scandinavian style, light oak wood, warm daylight, beige palette, interior photography, realistic materials

#### Goal
Make dual prompting the default pattern for SDXL-based generation.

---

### 4.4 Image Prompting — IP-Adapter

#### Role
Improve style consistency using a reference image.

#### Why
Text alone is often too weak for interior design style transfer.  
A moodboard, furniture reference, or example room can carry visual information more effectively than text.

#### Use cases
- match a target aesthetic
- guide furniture style
- preserve visual mood
- align with client inspiration images

#### Inputs
- one or more reference images
- optional text prompt for layout and scene description

#### Goal
Use IP-Adapter when the user wants generation “in the style of” a room image or moodboard.

---

### 4.5 Structure Conditioning — ControlNet

#### Role
Preserve room geometry and scene structure.

#### Why
Interior design tasks often require:
- preserving walls, windows, and perspective
- redesigning a real room without changing layout
- editing while keeping structural consistency

#### Recommended first control types
- **Depth**
- **Canny**

#### Use cases

##### Depth ControlNet
Best for:
- room-aware redesign
- perspective preservation
- spatial coherence

##### Canny ControlNet
Best for:
- edge-preserving structure control
- room outlines
- object placement guidance

#### Goal
Use ControlNet when structure matters more than freeform creativity.

---

### 4.6 Textual Inversion

#### Role
Create reusable learned tokens for specific styles or aesthetics.

#### Why
Some design concepts are too specific to express reliably with plain text prompts.

#### Use cases
- custom house style
- branded interior aesthetic
- repeated client-specific visual identity

#### When to use
Only after baseline prompting, IP-Adapter, and ControlNet are already working well.

#### Goal
Treat this as an advanced extension, not an initial dependency.

---

## 5. Recommended Pipeline by Task

### 5.1 Text-to-Image Interior Generation

#### Stack
- SDXL
- Compel
- SDXL dual prompts

#### Optional
- IP-Adapter if a style reference image is available

#### Purpose
Generate new room concepts from structured prompts.

---

### 5.2 Image-to-Image Room Redesign

#### Stack
- SDXL img2img
- Compel
- dual prompts
- ControlNet depth
- optional IP-Adapter

#### Purpose
Transform an existing room while preserving layout.

---

### 5.3 Inpainting

#### Stack
- SDXL inpainting pipeline
- Compel
- localized prompt focus
- optional IP-Adapter for replacement style

#### Purpose
Replace or modify selected furniture or decor regions.

---

### 5.4 Style Transfer / Moodboard Matching

#### Stack
- SDXL
- IP-Adapter
- Compel
- optional ControlNet if structure must stay fixed

#### Purpose
Generate interiors aligned to a reference design image.

---

## 6. Priority Order

Implementation should follow this order:

1. **SDXL baseline**
2. **Compel integration**
3. **Dual prompt strategy**
4. **ControlNet depth**
5. **IP-Adapter**
6. **Textual Inversion**

Reason:
- baseline quality first
- then text control
- then structural control
- then style-reference enhancement
- then advanced learned concepts

---

## 7. Decision Rules

Use the following rules during implementation.

### If the issue is weak prompt emphasis
Use:
- **Compel**

### If the issue is overloaded or muddy prompts
Use:
- **SDXL dual prompting**

### If the issue is weak style matching
Use:
- **IP-Adapter**

### If the issue is poor layout preservation
Use:
- **ControlNet**

### If the issue is inconsistent custom aesthetics
Use:
- **Textual Inversion**

---

## 8. Prompt Design Standard

All prompts should be structured with these components:

- room type
- layout
- main furniture
- style
- materials
- lighting
- color palette
- camera / photography language
- realism tags

### Recommended breakdown

#### `prompt`
- room type
- layout
- focal objects

#### `prompt_2`
- style
- materials
- lighting
- color palette
- realism tags

### Example

#### `prompt`
modern living room, open layout, large windows, low sofa, coffee table

#### `prompt_2`
Japanese Scandinavian style, light oak flooring, warm natural daylight, neutral beige palette, interior photography, ultra realistic

---

## 9. Negative Prompt Strategy

All pipelines should support negative prompts.

### Default negative prompt
- blurry
- low quality
- distorted perspective
- bad proportions
- unrealistic lighting
- duplicate furniture
- cluttered
- watermark

### Goal
Reduce common diffusion artifacts in interior scenes.

---

## 10. Implementation Requirements

### Required support
- configurable model IDs
- weighted prompt support
- structured prompt builder
- prompt + negative prompt logging
- support for reference images
- support for control images
- modular pipeline switching

### Nice to have
- prompt templates by room type
- style presets
- material presets
- lighting presets

---

## 11. Validation Criteria

The prompt-improvement system is successful if it improves:

- prompt adherence
- visual realism
- material quality
- lighting quality
- layout consistency
- style consistency across runs

### Qualitative checks
- does the room match the requested style?
- are materials visually plausible?
- is the lighting coherent?
- is the composition stable?
- is the redesigned room faithful to input structure?

---

## 12. Risks

### Risk 1 — Overcomplicated prompts
Too many descriptors can reduce control.

**Mitigation**
- separate prompt roles
- use weighting instead of adding more adjectives

### Risk 2 — Style drift
Outputs may not match intended design language.

**Mitigation**
- use IP-Adapter
- use prompt templates
- use reference images

### Risk 3 — Geometry inconsistency
Layout can drift in redesign tasks.

**Mitigation**
- use ControlNet depth or canny

### Risk 4 — Tooling complexity
Too many modules at once can slow implementation.

**Mitigation**
- implement in phases
- keep interfaces modular

---

## 13. Phase Plan

### Phase 1 — Baseline Prompting
Deliver:
- SDXL pipeline
- prompt and negative prompt support
- structured prompt template

### Phase 2 — Prompt Weighting
Deliver:
- Compel integration
- weighted prompt syntax support
- prompt embedding path

### Phase 3 — Dual Prompting
Deliver:
- `prompt` / `prompt_2` split
- configurable prompt builder

### Phase 4 — Structure Control
Deliver:
- ControlNet depth integration
- optional canny support

### Phase 5 — Style Guidance
Deliver:
- IP-Adapter support for reference images

### Phase 6 — Advanced Styling
Deliver:
- optional textual inversion support

---

## 14. Definition of Done

This plan is complete when:

- SDXL is the default base pipeline
- Compel is integrated for prompt weighting
- dual prompting is supported
- ControlNet works for layout-preserving redesign
- IP-Adapter works for style-guided generation
- negative prompts are standardized
- prompt inputs are logged consistently
- prompt templates are documented
- the system is modular and configurable

---

## 15. Final Recommendation

The default stack for this project should be:

- **SDXL** for baseline generation quality
- **Compel** for prompt weighting
- **Dual prompting** for semantic clarity
- **ControlNet depth** for room-preserving redesign
- **IP-Adapter** for reference-driven style control

Use **Textual Inversion** later if a custom reusable house style is needed.

This should be treated as the official prompt-improvement roadmap for the Interior Design diffusion pipeline.