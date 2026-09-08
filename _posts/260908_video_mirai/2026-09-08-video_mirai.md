---
layout: post
title: Video-Mirai - Autoregressive Video Diffusion Models Need Foresight
date: 2026-09-08 11:00:00
description: Yu et al.
tags: video 
categories: ar_diffusion_model
# pdf: _posts/250908_style_gan/slides_style_gan.pdf
pretty_table: true
---

[Yu et al. 2025](https://arxiv.org/abs/2607.26004)

[code](https://research.nvidia.com/labs/genair/pdd)

<br>


---

### Hozy Summary
- Idea)
  - Representation-Level Planning Gap
    - Def.)
      - States that fit the **current** segment may **discard** identity, layout, and motion information needed for a consistent future.
    - Desc.)
      - Causal video generations suffer from this gap because its standard training only asks each causal state to explain the present.
- Suggested Sol.)
  - Follow the [Causal Forcing](/blog/2026/causal_forcing) framework.
    - i.e.)
      - Train an AR teacher from the bidirectional teacher.
      - Initialize the AR student using the AR teacher.
      - Optimize the DMD loss between the bidirectional teacher and the AR student.
  - Additional supervision
    - Add additional components during training
      - Frozen Encoder
        - Prop.)
          - Bidirectional attention
          - Wan2.1-T2V-14B
        - Input
          - $$\mathbf{x}=(X_1,\ldots,X_n)$$ : the whole self rollout of the AR student generator $$G_\theta$$
        - Output
          - $$\mathbf{H}_{i+\delta}^{L'}$$ : the foresight encoder's mid-layer hidden representation corresponding to segment $$X_{i+\delta}$$, computed with access to the full rollout
        - Meaning)
          - Some hidden representation in the generated video
      - Predictor : $$\phi_\omega$$
        - Input
          - $$\mathbf{h}_i^L$$ : a hidden representation of $$G_\theta$$ of its internal $$L$$-th layer when generating the frame $$X_i$$
        - Output
          - $$\phi_\omega(\mathbf{h}_i^L)$$ : a projection/prediction of the causal generator's hidden feature into the foresight feature space
    - Create a loss of $$\ell_i^F(\delta) = 1-\cos(\phi_\omega(h_i^L), \text{stopgrad}[\mathbf{H}_{i+\delta}^{L'}])$$
    - Add this loss to the DMD loss