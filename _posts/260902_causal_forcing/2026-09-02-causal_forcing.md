---
layout: post
title: Causal Forcing - Autoregressive Diffusion Distillation Done Right for High-Quality Real-Time Interactive Video Generation
date: 2026-09-02 11:00:00
description: Zhu et al.
tags: video DMD
categories: ar_diffusion_model
# pdf: _posts/250908_style_gan/slides_style_gan.pdf
pretty_table: true
---

[Zhu et al. 2025](https://arxiv.org/abs/2602.02214)

[code](https://github.com/thu-ml/Causal-Forcing)

<br>

CAUSAL RCM   
Parallel denoising distillation   
MIRAI   
Video-Mirai: Autoregressive Video Diffusion Models Need Foresight   
flex attention   

---

### Hozy Summary
- Problem)
  - Existing distillation methods have an architectural gap between the teacher and the student model
    - s.t.
      - Teacher is trained bidirectionally
        - i.e.) attending the future frames
      - Students are trained to be AR
  - This gap causes frame-level non-injectivity in ODE distillation
    - Frame-level Injectivity
      - During teacher's training, frames attend to all frames
      - During student's distillation, frames attend to previous frames only
  - Consequently, MSE regression cannot recover the teacher's true flow map.
    - Instead, the student learns a conditional-expectation solution.
      - Needs to be filled!
- Sol.)
  1. Train the teacher as an AR diffusion model using TF
     - Teacher is AR, not bidirectional!
     - Thus, the authors argue that they satisfy the frame-level injectivity.
  2. Use the above AR diffusion model as the teacher, and perform causal ODE distillation 
     - by... 
       - sampling its PF-ODE trajectories 
       - training the AR students
     - Authors argue that since the teacher is not bidirectional, the student can accurately learn the flow map.
  3. Asymmetric DMD
     - Needs to be filled!
- Validation)
  - Comprehensive evaluation against various baseline models


---

<br><br>

#### Summarized
- 