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
  - AR distillation methods ([CausVid](/blog/2026/causvid), [Self-Forcing](/blog/2026/self_forcing)) show poorer performance compared to standard step-distillation method.
    - Symptoms)
      - Low vision quality, dynamic degree, and instruction following
  - Authors suspect the **architectural gap** between the **bidirectional teacher** and the **AR student** model in the existing methods.
    - Why?)
      - Authors argue that resolving timestep-gap is not the problem.
        - Pf.)
          - A student model that is first trained student using standard DMD and then further trained using Self Forcing showed low quality outputs compared to the standard DMD.
          - They argue that the pre-trained standard DMD resolved the timestep-gap, but latter Self Forcing caused the architectural gap. (?)
  - They further focus on the ODE distillation process ([student ODE initialization (CausVid)](/blog/2026/causvid#43-student-initialization)).
    - cf.) the MSE-loss during the initialization process
      - Recall that [ODE initialization(CausVid)](/blog/2026/causvid#43-student-initialization) was made by $$\mathcal{L}_{\text{init}} = \mathbb{E}_{x, t^i}\left\Vert G_\phi\left( x_{t^i}^i, x_{t^i}^{\lt i}, t \right) - x_{0}^i \right\Vert^2$$
    - Why?)
      - By [Rectified Flow](/blog/2025/rectified_flow/), the transport between two distributions must be injective for the MSE loss to be valid.
      - They show that the probability that injectivity does not hold is greater than 0.
      - By Bishop & Nasrabadi, authors argue that the student learns a conditional-expectation solution instead of the actual solution.
- Sol.)
  - Train an additional AR teacher model for the ODE distillation (i.e. the student initialization)
    - Use teacher forcing (TF) to train on the pre-trained bidirectional model.
    - Let this TF teacher to sample the PF-ODE-trajectory.
  - Initialize the student AR model using this trajectory.
  - Perform asymmetric DMD just as the [Self-Forcing](/blog/2026/self_forcing).
    - Why is it asymmetric?)
      - The teacher model in this stage is the original bidirectional model.




<br><br>

# Causal Forcing ++
#### Summary
1. Keep CF Stage 1: TF-trained multi-step AR diffusion teacher.
2. Replace CF Stage 2: causal ODE distillation $$\rightarrow$$ causal consistency distillation
   - same AR-conditional flow-map target
   - one online adjacent teacher ODE step
   - no offline full PF-ODE trajectories
   - smaller optimization gap
   - ~4× cheaper, no extra trajectory storage
3. Keep CF Stage 3: Self-Forcing-style asymmetric DMD with self-rollout.

- Additional:
  - targets frame-wise 1–2 step AR generation
  - causal DMD with TF as alternative initialization; worse than causal CD
  - action-conditioned world-model extension