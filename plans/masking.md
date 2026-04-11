# MASKING_PLAN.md

## Project
Interior Design with Diffusion Models

## Purpose
Define the masking improvement strategy for the Interior Design diffusion pipeline.

This plan focuses on improving:
- object selection accuracy
- mask boundary quality
- inpainting blend quality
- automation for object-level editing
- structural consistency during masked edits

This document is the source of truth for masking-related design and implementation decisions.

---

## 1. Objective

Build a masking system that improves inpainting quality for interior design tasks.

The masking system should support:
- manual mask input
- automatic mask generation
- text-guided object masking
- mask refinement
- mask post-processing before inpainting

The final goal is to make masked editing more accurate, cleaner, and more realistic.

---

## 2. Why Masking Matters

In interior design workflows, poor masks cause:
- broken furniture boundaries
- unnatural object replacement
- edge halos
- texture bleeding
- unrealistic blending with floors, walls, and surrounding decor

Good masking is essential for:
- replacing furniture
- removing clutter
- editing decor
- modifying wall regions
- preserving room structure during localized edits

---

## 3. Core Problems to Solve

The masking pipeline must solve three distinct problems:

### 3.1 Region Selection
Find the correct object or area to edit.

Examples:
- sofa
- coffee table
- lamp
- painting
- rug
- window region
- wall section

### 3.2 Boundary Quality
Improve mask edges so the selected region aligns well with the actual object.

Examples:
- thin furniture legs
- plant leaves
- curved sofa edges
- hanging lights
- decor with irregular contours

### 3.3 Edge Blending
Prepare the mask so inpainting blends naturally with surrounding context.

Examples:
- avoid sharp seams
- avoid color discontinuities
- avoid object remnants around boundaries

---

## 4. Strategy Overview

The masking improvement stack will use the following components:

1. **Manual Mask Input**
2. **SAM for mask generation**
3. **Grounding DINO + SAM for text-driven object masking**
4. **Mask refinement utilities**
5. **Optional BiRefNet refinement**
6. **Mask-aware inpainting preparation**
7. **Optional structural control during inpainting**

---

## 5. Module Plan

---

## 5.1 Manual Mask Input

### Role
Support user-provided masks for direct editing control.

### Why
Manual masks are still useful when:
- the user knows exactly what to edit
- automatic detection fails
- object boundaries are ambiguous
- multiple objects are close together

### Requirements
- accept binary mask image input
- validate size against source image
- support white = editable region
- reject invalid dimensions
- save mask preview for debugging

### Goal
Manual masking must be supported from day one.

---

## 5.2 SAM for Automatic Mask Generation

### Role
Generate masks from interactive prompts such as clicks, points, or boxes.

### Why
SAM is the default automatic mask generator because it is flexible and general-purpose.

### Use cases
- select a sofa
- isolate a rug
- select a window area
- isolate a chair or table
- mask decor objects for replacement

### Requirements
- support point-based selection
- support box-based selection
- support one-object-at-a-time masking
- save generated masks for inspection

### Goal
SAM should be the default automatic mask generator.

---

## 5.3 Grounding DINO + SAM for Text-Driven Masking

### Role
Convert text object queries into masks.

### Why
This enables workflows like:
- "replace the sofa"
- "remove the lamp"
- "change the carpet"
- "edit the wall art"

Grounding DINO detects the target object from text.  
SAM converts that localized region into a more precise segmentation mask.

### Workflow
1. user enters object phrase
2. Grounding DINO predicts object box
3. SAM refines box into object mask
4. mask is cleaned before inpainting

### Requirements
- support phrase-based object selection
- support confidence thresholds
- support single best object or multiple candidates
- save detection preview and final mask

### Goal
This should be the main automation path for object-level inpainting.

---

## 5.4 Mask Refinement Utilities

### Role
Improve raw masks before inpainting.

### Why
Even a good segmentation model may produce:
- jagged boundaries
- tiny holes
- disconnected islands
- incomplete object coverage
- rough transitions

### Required utilities
- dilation
- erosion
- opening
- closing
- hole filling
- connected-component cleanup
- small island removal

### Why each matters

#### Dilation
Expands the editable region slightly.  
Useful for removing old object boundaries and avoiding halos.

#### Erosion
Shrinks overly large masks.  
Useful when the mask spills into surrounding objects.

#### Opening
Removes small noise regions.  
Useful for cluttered scenes.

#### Closing
Closes small gaps inside object regions.  
Useful for rugs, lamps, and thin structures.

#### Hole Filling
Ensures the selected object is fully editable.

#### Island Removal
Removes isolated mask fragments that should not be edited.

### Goal
All masks should go through a configurable cleanup step before inpainting.

---

## 5.5 Edge Feathering

### Role
Soften mask boundaries before inpainting.

### Why
Hard-edged masks often cause:
- visible seams
- unnatural replacement borders
- abrupt texture changes

### Methods
- Gaussian blur on mask edge
- alpha feathering
- soft transition ring around boundary

### Use cases
- upholstery changes
- object replacement
- decor edits
- clutter removal

### Goal
Feathering should be available as a configurable default.

---

## 5.6 Optional BiRefNet Refinement

### Role
Improve high-quality object boundaries after rough mask generation.

### Why
Some objects require better edge precision than basic segmentation provides.

Examples:
- chair legs
- hanging lamps
- plant leaves
- fine decor outlines
- mirrors and frames

### Use cases
- high-resolution inpainting
- product-level furniture replacement
- premium-quality editing workflows

### Requirements
- optional refinement stage
- applied after SAM or Grounding DINO + SAM
- output should preserve fine object boundaries

### Goal
Treat BiRefNet as an advanced refinement module, not a first dependency.

---

## 5.7 Mask-Aware Inpainting Preparation

### Role
Prepare the final mask for inpainting.

### Why
The inpainting model depends heavily on the mask quality and boundary behavior.

### Requirements
Before running inpainting:
- validate mask dimensions
- ensure binary or normalized mask format
- optionally dilate the mask
- optionally feather the mask edge
- save final mask used by pipeline

### Default preparation policy
- validate
- clean
- dilate slightly
- feather boundary
- send to inpainting

### Goal
Every masked inpainting call should use a prepared mask, not a raw one.

---

## 5.8 Optional Structural Guidance During Inpainting

### Role
Preserve room geometry while editing masked regions.

### Why
Interior scenes have strong structural constraints:
- perspective lines
- floor-wall boundaries
- window geometry
- room layout
- furniture alignment

### Methods
- ControlNet depth
- ControlNet canny

### Use cases
- replacing large furniture
- editing wall regions
- preserving room geometry during redesign
- preventing inpainting drift

### Goal
Use structural guidance when large edits risk breaking scene consistency.

---

## 6. Recommended Pipeline by Task

### 6.1 Furniture Replacement

#### Stack
- Grounding DINO
- SAM
- mask cleanup
- dilation
- feathering
- inpainting

#### Example tasks
- replace sofa
- replace dining table
- replace lamp

---

### 6.2 Decor Removal

#### Stack
- SAM or manual mask
- cleanup
- feathering
- inpainting

#### Example tasks
- remove clutter
- remove wall art
- remove small decor objects

---

### 6.3 Wall or Floor Editing

#### Stack
- manual mask or SAM
- cleanup
- optional structural control
- inpainting

#### Example tasks
- repaint wall section
- replace flooring region
- modify wall texture

---

### 6.4 High-Precision Editing

#### Stack
- Grounding DINO
- SAM
- BiRefNet refinement
- cleanup
- feathering
- inpainting

#### Example tasks
- edit thin objects
- precise decor replacement
- premium quality masking workflows

---

## 7. Priority Order

Implementation should follow this order:

1. **Manual mask support**
2. **Mask validation**
3. **Basic mask cleanup utilities**
4. **Edge feathering**
5. **SAM integration**
6. **Grounding DINO + SAM workflow**
7. **Optional ControlNet-assisted masked editing**
8. **Optional BiRefNet refinement**

Reason:
- basic reliability first
- then automatic masking
- then object-query automation
- then advanced refinement

---

## 8. Decision Rules

Use the following rules during implementation.

### If the user already provides a mask
Use:
- manual mask path
- validation
- cleanup
- inpainting preparation

### If the user wants click-based selection
Use:
- SAM

### If the user wants phrase-based selection
Use:
- Grounding DINO + SAM

### If mask edges are rough
Use:
- cleanup
- feathering
- optional BiRefNet

### If large edits break room geometry
Use:
- structural guidance with ControlNet

---

## 9. Functional Requirements

The masking system must support:

- source image input
- manual binary mask input
- generated mask preview
- text query for object localization
- automatic mask cleanup
- edge feathering
- saved final mask artifact
- compatibility with inpainting pipeline

Optional support:
- multiple mask candidates
- user selection among candidates
- mask overlay visualization
- ControlNet-assisted masked edits

---

## 10. Mask Quality Standards

A high-quality mask should:

- cover the full intended object
- exclude most surrounding context
- preserve clean boundaries
- contain minimal noise
- avoid small disconnected fragments
- blend cleanly during inpainting

A low-quality mask usually has:
- missing object regions
- spill into nearby furniture
- jagged edges
- broken silhouettes
- harsh transition lines

---

## 11. Post-Processing Policy

All auto-generated masks should optionally support the following post-processing controls:

- `dilation_radius`
- `erosion_radius`
- `blur_radius`
- `fill_holes`
- `remove_small_components`
- `keep_largest_component`

### Default recommendation
- light dilation
- light blur
- hole filling enabled
- small component removal enabled

---

## 12. Validation Criteria

The masking system is successful if it improves:

- object selection accuracy
- edge quality
- inpainting realism
- edit controllability
- visual integration with surrounding scene

### Qualitative checks
- does the mask cover the intended object correctly?
- are boundaries aligned with the object?
- does the edited region blend naturally?
- are surrounding objects preserved?
- does the room remain structurally coherent?

### Failure indicators
- object remnants remain after replacement
- edited result leaks outside target area
- seams are visible
- geometry breaks around edited region

---

## 13. Risks

### Risk 1 — Wrong Object Detection
Text-based detection may localize the wrong object.

**Mitigation**
- use thresholds
- show candidate boxes
- allow manual override

### Risk 2 — Rough Boundaries
Automatic masks may be too coarse.

**Mitigation**
- cleanup
- feathering
- optional refinement stage

### Risk 3 — Over-Expanded Mask
Aggressive dilation may repaint too much context.

**Mitigation**
- keep dilation configurable
- use conservative defaults

### Risk 4 — Under-Masked Object
Incomplete masks leave visible old object fragments.

**Mitigation**
- hole filling
- dilation
- preview generated masks

### Risk 5 — Structural Drift
Large masked edits may distort room geometry.

**Mitigation**
- use ControlNet depth or canny for large edits

---

## 14. Phase Plan

### Phase 1 — Baseline Mask Support
Deliver:
- manual mask support
- mask validation
- mask save/load utilities

### Phase 2 — Cleanup and Preparation
Deliver:
- dilation
- erosion
- blur
- hole filling
- final mask preparation pipeline

### Phase 3 — Automatic Segmentation
Deliver:
- SAM integration
- click/box-based masking

### Phase 4 — Text-Driven Masking
Deliver:
- Grounding DINO + SAM integration
- object-query-based masking

### Phase 5 — Structural Stability
Deliver:
- optional ControlNet-assisted masked editing

### Phase 6 — Advanced Refinement
Deliver:
- optional BiRefNet refinement
- premium boundary quality workflow

---

## 15. Definition of Done

This masking plan is complete when:

- manual masks are supported
- mask validation is reliable
- cleanup utilities are implemented
- edge feathering is configurable
- SAM works for automatic masking
- Grounding DINO + SAM works for phrase-based object masking
- final masks are saved for inspection
- masked edits blend cleanly in common interior design tasks
- the masking pipeline is modular and configurable

---

## 16. Final Recommendation

The default masking roadmap for this project should be:

- **Manual mask support** for full control
- **SAM** for automatic segmentation
- **Grounding DINO + SAM** for phrase-driven masking
- **Mask cleanup + dilation + feathering** before all inpainting
- **Optional ControlNet** for large structural edits
- **Optional BiRefNet** for premium boundary refinement

This should be treated as the official masking improvement roadmap for the Interior Design diffusion pipeline.c