---
layout: post
title: Self Forcing - Bridging the Train-Test Gap in Autoregressive Video Diffusion
date: 2026-08-28 11:00:00
description: Huang et al.
tags: AR video
categories: diffusion_model
# pdf: _posts/250908_style_gan/slides_style_gan.pdf
pretty_table: true
---

[Huang et al. 2025](https://arxiv.org/abs/2506.08009)

<br>

---

### Hozy Summary


---

<br><br>

## 3. Setup & Model

### 3.1 Preliminaries: Autoregressive Video Diffusion Models
- Idea)
  - Combine AR chain-rule decomposition with denoising diffusion models for generation
- Def.)
  - $$x^{1:N} = \left( x^1, x^2, \ldots, x^N \right)$$ where $$N$$ is the length of the sequence
    - factorized using the chain rule of $$p\left( x^{1:N} \right) = \displaystyle\prod_{i=1}^N p\left( x^i \mid x^{\lt i} \right)$$
      - with each $$p\left( x^i \mid x^{\lt i} \right)$$ is modeled using a diffusion process
  - Each frame $$x^i$$ is corrupted by the forward process $$q_{t^i\mid0}\left( x_{t^i}^i \mid x_0^i \right)$$
    - s.t. $$x_{t^i}^i = \Psi\left( x^i, \epsilon^i, t^i \right) = \alpha_{t^i}x^i + \sigma_{t^i}\epsilon^i$$ : the forward process
      - where
        - $$\alpha_{t^i}, \sigma_{t^i}$$ : pre-defined noise schedule
        - $$t^i\in[0,1000]$$ : a finite time horizon
        - $$\epsilon^i\sim\mathcal{N}(0,mathbf{I})$$ : Gaussian noise
    - $$p_\theta\left( x^i \mid x^{\lt i} \right)$$ is implicitly defined as $$f_{\theta, t_1}\circ f_{\theta, t_2}\circ \cdots \circ f_{\theta, t_T}\left( x_{t_T}^i \right)$$
      - where $$f_{\theta, t_j}\left( x_{t_j}^i \right) = \Psi\bigg( \underbrace{G_\theta\left( x_{t_j}^i, j_j, x^{\lt i} \right)}_{\text{output from network}}, \epsilon_{t_{j-1}}, t_{j-1} \bigg)$$
  - Optimization Target : Epsilon prediction
    - $$\mathcal{L}_\theta^{\text{DM}} = \mathbb{E}_{x^i, t^i, \epsilon^i} \left[ w_{t^i} \Vert \hat{\epsilon}_\theta^i - \epsilon^i \Vert_2^2 \right]$$ s.t.
      - $$\hat{\epsilon}_\theta^i := G_\theta\left(x_{t^i}^i, t^i, x\right)$$ : a neural network conditioned on the context $$c$$
      - $$w_{t^i}$$ : the weighting function
- Self Forcing Set-ups
  - DiT with text conditioning on latent space encoded by a causal 3D VAE
  - AR chain-rule decomposition implemented with causal attention

#### Concept) Teacher Forcing (TF)
- Idea)
  - Train the model to predict the next token conditioned on **ground-truth tokens**
  - In video diffusion, TF denoises each frame using clean, ground-truth context frames

#### Concept) Diffusion Forcing (DF)
- Idea)
  - Train the model on **videos with noise** levels independently sampled for each frame
  - In video diffusion, TF denoises each frame based on noisy context frames

<br><br>

### 3.2 Autoregressive Diffusion Post-Training via Self-Rollout
- Training)
  - Sample a batch of videos $$\left\{x^{1:N}\right\} \sim p\left( x^{1:N} \right) = \displaystyle\prod_{i=1}^N p\left( x^i \mid x^{\lt i} \right)$$
    - conditioned on self-generated outputs including both...
      - clean context frames in the past
      - noisy frames at the current time step
    - How?)
      - At each denoising step $$t_j$$ and the frame index $$i$$, the model denoises an intermediate noisy frame $$x_{t_j}^i$$ by...
        1. Condition on previous clean frames $$x^{\lt i}$$
        2. Obtain previous timestep noisy frame $$x_{t_{j-1}}^i$$ through the forward process $$\Psi$$ and inject a Gaussian noise with lower noise level
        3. Perform few-step diffusion process
  - Employ KV caching during training as well.
  - Gradient truncation
    - i.e.) Limit the backpropagation to only the final denoising step of each frame
  - Sample a denoising timestep $$s\sim \mathcal{U}(1,T)$$, and use the $$s$$-th step output as the final output
    - Why doing this?)
      - To ensure all intermediate denoising steps receive supervision signals
  - Detach the gradients of the previous frames from the current frame by restricting gradient flow into KV cache embeddings. 

<br><br>

### 3.3 Holistic Distribution Matching Loss
- Goal)
  - Utilize distillation method to enhance the quality of AR video generation
- Method)
  - Let
    - $$p_\theta\left( x^{1:N} \right)$$ : the distribution of generated videos
    - $$p_\text{data}\left( x^{1:N} \right)$$ : the distribution of real videos
  - Inject noise to both distributions as $$p_{(\cdot), t}\left( x_t^{1:N} \right) = \displaystyle\int q_{t\mid0}\left( x_t^{1:N}\mid x^{1:N} \right) p_{(\cdot)}\left(x^{1:N}\right) \text{d} x^{1:N}$$
  - Match the whole video (not individual frames) $$p_{\theta, t}\left( x_t^{1:N} \right)$$ and $$p_{\text{data}, t}\left( x_t^{1:N} \right)$$ using...
    1. Distribution Matching Distillation (DMD) : $$\mathbb{E}_t\left[ \mathcal{D}_{\text{KL}}\left( p_{\theta, t} \Vert p_{\text{data}, t} \right) \right]$$
    2. Score Identity Distillation (SiD) : $$\mathbb{E}_{t, p_{\theta, t}}\left[ \Vert \nabla\log p_{\theta, t} - \nabla \log p_{\text{data}, t} \Vert^2 \right]$$
    3. GAN : Minimizing JS-divergence

<br><br>

### 3.4 Long Video Generation with Rolling KV Cache
- How?)
  - Maintain a fixed-size KV cache for most recent $$L$$ frames.
    - When full, remove the oldest ones, and push new ones.
  - This cause severe flickering artifacts specifically for the first latent frame.
    - Why?)
      - During training, the model is provided with clean first frames, while this is not available during the rolling KV cache procedure.
    - Sol.)
      - During training, restrict the attention window so the model cannot attend to the first chunk when denoising the final chunk

{% include figure.liquid path="assets/img/blog/260828_self_forcing/001.png" class="img-fluid rounded z-depth-1" zoomable=true %}