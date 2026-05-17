---
layout: post
title: (Presentation PDF) Video Models are Zero-shot Learners and Reasoners
date: 2025-09-30 14:24:00
description: Wiedemer et al. 2025
tags: diffusion_model video_model zero_shot reasoning chain_of_frames
pdf: assets/pdf/posts/slides_video_model_zero_shot.pdf
categories: video_model
---
Wiedemer et al. 2025

## 2. Methods
- Prompt Veo
  - Versions
    - Veo 2, Veo 3
  - Props.)
    - For some tasks, the solution is likely to come from LLM (Gemini Pro)
    - Some tasks showed that a standalone LLM could not solve them.
    - The video model shouldn't be limited to Veo.
      - Any model that can be a zero-shot learners and reasoners can be supported to demonstrate the main argument.

#### Takeaway1)
- In NLP, prompting replaced task-specific training or adaptation.
- A similar paradigm shift is on the horizon in machine vision, facilitated by video models.

<br><br>

## 3. Qualitative results: Sparks of visual intelligence?

### Concept) Hierarchical Capabilities
- Components
  - [Perception](#concept-perception)
  - [Modeling](#concept-modeling)
  - [Manipulation](#concept-manipulation)
  - [Reasoning](#concept-reasoning)
- Props.)
  - Capability boundaries often overlap.


#### Concept) Success Rate 
- Def.)
  - The fraction of generated videos that solved the task
- Props.)
  - Determined by humans
  - $$\gt0$$. : the model possesses the ability to solve the task
  - $$\approx1$$. : the model reliably solves the problem irrespective of the random seed


<br>

### Concept) Perception
- Desc.)
  - A foundational ability to understand visual information
- Props.)
  - Previous video models...
    - Some backbones of the above can be adapted or fine-tuned for other tasks.
    - But training-free transfer to novel tasks is rare (limits the generalization)
  - No task-specific training required for the following tasks.
    - Object Detection
    - Edge detection
      - Quantitative
    - Segmentation
      - Quantitative
    - Keypoint localization
    - Super resolution
    - Bind blurring
    - Denoising
    - Low-light enhancing
    - Conjunctive Search
    - Interpreting ambiguous images

#### Takeaway2)
- Veo 3 shows emergent zero-shot perceptual abilities well beyond the training task.
- Just like LLMs replaced task-specific NLP models, video models will likely replace most bespoke
 models in computer vision—once they become sufficiently cheap and reliable.


<br>

### Concept) Modeling
- Desc.)
  - Based on the [perception](#concept-perception) of objects...
  - Form a **model** of a visual world
    - i.e.) Principles that governs the world.
      - e.g.) Laws of physics
- Props.)
- e.g.)
  - Physical Characteristics
    - Rigid and soft body dynamics and their surface interactions
    - Flammability
    - Air resistance affecting falling objects
    - Buoyancy
    - Optical phenomena : refraction and reflection
    - Additive / Subtractive color mixing
    - Visual Jenga task : removing objects in a physically plausible order
    - Putting objects that fits into a backpack
  - Abstract Relationships
    - Distinguishing categories
      - e.g.) toys vs laptop
    - Recognizing patterns, generating variations, and parsing larger wholes into parts
    - Maintaining a memory of the world state across time and camera movements


<br>

### Concept) Manipulation
- Desc.)
  - Based on the [perception](#concept-perception) of objects and the [model](#concept-modeling) that defines their relations...
  - Meaningfully **alter** the perceived modeled world
- Tasks)
  - Background removal
  - Style transfer
  - Colorization
  - Inpainting
  - Outpainting
  - Manipulating text elements
  - Edit images based on doodle instructions
  - Compose scenes from individual components
  - Generate novel views of objects and characters
  - Smoothly transform one object into another
  - Change of perspective, lighting, and appearance (Selfie -> Professional photograph)
  - Simulate object manipulation
  - Interpret object affordance
  - Draw a shape
  - Roll a burrito


<br>

### Concept) Reasoning
- Desc.)
  - Integrating [perception](#concept-perception), [modeling](#concept-modeling), and [manipulation](#concept-manipulation)...
  - Reasons across dimensions (space and time) over a sequence of manipulation steps
  - Applied through [CoF](#concept-chain-of-frames-cof).
- Tasks)
  - Generate a valid graph traversal
  - Perform visual BFS on a tree
  - Complete visual sequences
  - Connect matching colors
  - Fit shapes into holes
  - Sorting numbers
  - Use tools to accomplish a visual task
  - Solve simple Sudokus or visual puzzles
  - Solve mazes and navigation tasks
  - Extrapolate rules from visual examples

#### Concept) Chain of Frames (CoF)
- Desc.)
  - The changes applied frame-by-frame in a generated video
  - This is how the video models apply changes across dimensions
  - Corresponds to the chain-of-thought (CoT) in LLMs

#### Takeaway 3)
- Frame-by-frame video generation parallels chain-of-thought in language models. 
- Just like chain-of-thought (CoT) enables language models to reason with symbols, a “chain-of
frames” (CoF) enables video models to reason across time and space.


<br><br>

## 4. Quantitative results
- Evaluation Method
  - Evaluate performance separately for
    - the best frame
      - Best performance across any frame in the generated videos
      - Performance ceiling
      - The optimal frame is not non a priori
    - the last frame
      - May underestimate a model's ability
        - Why?) Veo tends to continue animating a scene even after task completion.
      - But the frame is predetermined.
  - Reference : Nano Banana
    - cf.) 
      - Image models are excellent zero-shot learners too.
      - Video models are the more general framework because of their ability to process both temporal and spatial information.




