---
layout: post
title: (DM Reconst.) [WIP] Ch.10 Distillation-Based Methods for Fast Sampling
date: 2026-06-03 11:00:00
description: Lai et al.   
tags: distillation
categories: diffusion_model
# pdf: _posts/250908_style_gan/slides_style_gan.pdf
pretty_table: true
---

Diffusion Model Conceptual Reconstruction following [The Principles of Diffusion Models](https://arxiv.org/abs/2510.21890)

<br>

---

### Hozy Summary


---

<br><br>

### Concept) Distillation
- Problem)
  - Diffusion model's trade-off between quality and efficiency.
    - DDPM's $$\mathbf{x}$$-prediction perspective.
      - We use $$\mathbf{x}_{\phi^\times}(\mathbf{x}_t, t)\approx\mathbb{E}[\mathbf{x}_0\mid\mathbf{x}_t]$$ for one-step generation, where this denoiser averages over many plausible outcomes.
      - Thus, few denoising steps leads overly smooth prediction and the blurry, low quality samples.
    - ODE/SDE Trajectory Perspective)
      - Reducing the NFE speeds up the generation but reduces the fidelity.
        - Why?)
          - Each solver step introduces an integration error of order $$\mathcal{O}(h^n)$$
            - where
              - $$h=\max_i\left\vert t_i-t_{i-1}\right\vert$$ : the maximum step size
              - $$n$$ : the solver order.
          - Thus, $$(\text{Fewer Steps})\Rightarrow h\uparrow \Rightarrow \mathcal{O}(h^n) \uparrow$$
- Idea)
  - Accelerate diffusion model sampling by teaching new generators to produce samples in only one or a few steps.
    - How?)
      - Teacher Sampler : Slow, pre-trained diffusion model
      - Student model : Learn from the teacher sampler
- Paradigms)
  - [Distribution Level Distillation](#concept-distribution-level-distillation)
  - [Flow Map Level Distillation](#concept-flow-map-level-distillation)

<br><br>


### Concept) Distribution Level Distillation
- Model)
  - Given
    - $$p_{\text{prior}}, p_{\text{data}}$$ : a prior noise distribution and the data distribution
  - We want to train a one-step generator $$G_\theta(\mathbf{z})$$
    - where
      - $$\mathbf{z}\sim p_{\text{prior}}$$ : a noisy latent
    - by optimizing as
      - $$\min_{\theta} \mathcal{D}\left(p_\theta(\hat{\mathbf{x}}), p_{\text{data}}(\hat{\mathbf{x}})\right)$$.
        - where 
          - $$\hat{\mathbf{x}} = G_\theta(\mathbf{z})$$ : a sample from $$G_\theta(\mathbf{z})$$
          - $$\mathcal{D}$$ denotes a suitable divergence measurement such as KL.
  - In practice, we align the generator's distribution with the empirical distrbution $$p_{\phi^\times}(\mathbf{x})$$
    - i.e.) $$\min_{\theta} \mathcal{D}\left(p_\theta(\hat{\mathbf{x}}), p_{\phi^\times}(\hat{\mathbf{x}})\right)$$
- Approaches)
  - Distributional Matching Distillation (DMD) (Yin et al. 2024)
  - Variational Score Distillation (VSD)
  - Score Identity Distillation (SiD)

<br><br>


### Concept) Flow Map Level Distillation
- Model)
  - Consider the [PF-ODE](/blog/2026/score_sde/#concept-probability-flow-ode-pf-ode) of
    - $$\displaystyle\frac{\text{d}\mathbf{x}(\tau)}{\text{d}\tau} = f(\tau)\mathbf{x}(\tau) - \frac{1}{2}g^2(\tau)\nabla_{\mathbf{x}} \log p_\tau(\mathbf{x}(\tau))=: \mathbf{v}^*(\mathbf{x}(\tau), \tau)$$.
  - Its solution map is denoted as
    - $$\displaystyle \Psi_{s\rightarrow t}(\mathbf{x}_s) := \mathbf{x}_s + \int_s^t v^*(\mathbf{x}(\tau), \tau)\text{d}\tau$$.
      - where
        - $$t\le s$$.
        - $$\mathbf{x}_T\sim p_{\text{prior}}, \mathbf{x}_0\sim p_{\text{data}}$$.
  - We want to learn a map $$\Psi_{T\rightarrow0}(\mathbf{x}_T)$$ that enables the one-step generation.
  - However, the PF-ODE rarely admits a closed form. Thus, we approximate it numerically during training, and denote it as the general solver of
    - $$\text{Solver}_{s\rightarrow t}(\mathbf{x}_s;\;\phi^\times)$$ or simply $$\text{Solver}_{s\rightarrow t}(\mathbf{x}_s)$$
      - i.e.) the numerical integration from $$s$$ to $$t$$ starting at $$\mathbf{x}_s$$, with teacher parameters $$\phi^\times$$
  - The loss can be denoted as
    - $$\mathcal{L}_{\text{oracle}}(\theta) := \mathbb{E}_{s,t}\mathbb{E}_{\mathbf{x}_s\sim p_s}\left[ w(s, t) \; d(G_\theta(\mathbf{x}_s, s, t), \Psi_{s\rightarrow t}(\mathbf{x}_s)) \right]$$.
      - where
        - $$w(s, t)$$ : the weight on the time pairs $$(s,t)$$
        - $$d(\cdot,\;\cdot)$$: : a distance metric
- Approaches)
  - [Knowledge Distillation](#concept-knowledge-distillation)
  - [Progressive Distillation](#concept-progressive-distillation)

<br>

#### Concept) Knowledge Distillation
Luhman and Luhman, 2021
- Goal)
  - Train a generator to imitate the output of a numerical solver evaluated along the full trajectory : $$0\rightarrow T$$.
    - i.e.) $$G_\phi(\mathbf{x}_T, T, 0) \approx\text{Solver}_{T\rightarrow0}(\mathbf{x}_T)$$ for $$\mathbf{x}_T\sim p_{\text{prior}}$$
- Loss)
  - $$\mathcal{L}_{\text{KD}} := \mathbb{E}_{\mathbf{x}_T\sim p_{\text{prior}}}\left\Vert G_\phi(\mathbf{x}_T, T, 0) - \text{Solver}_{T\rightarrow0}(\mathbf{x}_T) \right\Vert_2^2$$.
- Advantage)
  - Direct supervision from the pre-trained teacher $$\text{Solver}$$
- Drawback)
  - Cannot leverage the strong supervision from the **original training data**.
  - Computationally expensive.
    - Why?) ODE integration on the Solver is required during the $$G_\phi$$'s training.
  - The only option of the global mapping of $$T\rightarrow0$$, does not provide controllability.
    - Thus, controllable generation techniques cannot be applied.


<br>

#### Concept) Progressive Distillation
Salimans and Ho, 2021
- Goal)
  - Train a time-conditional `Student` using local supervision from `Teacher` fragments.
- Idea)
  - $$t_0=T\gt t_1\gt\cdots\gt t_N = 0$$ : the fixed time grid
  - $$\text{Teacher}_{t_k\rightarrow t_{k+1}}$$ : Teacher model that provides time-stepping maps for $$k=0,\ldots,N-1$$
  - Repeat 
    - $$\text{Student}_{t_k\rightarrow t_{k+2}} \approx \text{Teacher}_{t_k\rightarrow t_{k+1}} \circ \text{Teacher}_{t_{k+1}\rightarrow t_{k+2}}$$.
      - i.e.) Student learns the two-step skip map for $$k=0,2,4,\ldots, N-1$$.
    - $$\text{Teacher}\leftarrow\text{Student}$$.

<br><br>

{% include figure.liquid path="assets/img/blog/260526_unified_lens_on_dm/007.png" class="img-fluid rounded z-depth-1" zoomable=true %}