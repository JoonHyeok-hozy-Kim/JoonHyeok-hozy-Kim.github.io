---
layout: post
title: Score-Based Generative Modeling through Stochastic Differential Equation
date: 2025-09-31 14:24:00
description: Song et al. 2021
tags: diffusion_model score_based_model NCSN SDE ODE predictor_corrector_sampler
categories: diffusion_model
# pdf: assets/pdf/posts/slides_style_gan.pdf
pretty_table: true
---
Song et al. 2021

## 2. Background
#### Concept) Stochastic Differential Equation (SDE)
- Def.)
  - An SDE describes the evolution of a system with both deterministic dynamics and stochastic (random) noise.
- General form (Itô SDE)
  - $$\text{d}\mathbf{x} = \mathbf{f}(\mathbf{x}, t) \text{d}t + g(\mathbf{x}, t)\text{d}W_t$$.
    - where
      - $$\mathbf{f}(\mathbf{x}, t)$$. : the drift term
        - This shows the deterministic trend
        - The ordinary differential equation (ODE) has only the deterministic trend.
      - $$g(\mathbf{x}, t)$$. : the diffusion term
        - This shows the magnitude of randomness
      - $$\text{d}W_t$$. : the increment of a Wiener process (Brownian motion)
- Intuition)
  - Ordinary differential equations (ODEs) describe smooth, deterministic changes.
    - cf.) ODE : $$\text{d}\mathbf{x} = \mathbf{f}(\mathbf{x}, t) \text{d}t$$.
  - SDEs add a stochastic part.
    - Thus, the system evolves under both predictable forces and random fluctuations.
  - In the context of the generative modeling...
    - Forward process
      - Gradually corrupts data into noise using an SDE
    - Reverse process
      - Denoises by simulating the time-reversed SDE, guided by a learned score function.

#### Concept) Score
- Notation)
  - $$\nabla_\mathbf{x} \log p(\mathbf{x})$$. : the score
- Meaning)
  - A vector that shows in which direction the probability distribution $$p$$. increases near the datapoint $$\mathbf{x}$$.

#### Concept) Noise Conditional Score Network (NCSN)
- Def.)
  - A neural network $$s_\theta(\mathbf{x}, \sigma)$$. s.t. 
    - Input 
      - a noisy data sample $$\mathbf{x}$$.
      - a noise level $$\sigma$$.
    - Output 
      - $$\nabla_\mathbf{x} \log p_\sigma(\mathbf{x})$$. : the [score](#concept-score) function (the gradient of the log-density) of the **perturbed** distribution at that noise level $$\sigma$$.
- Key idea)
  - Directly estimating $$\nabla_x \log p_\text{data}(x)$$. is too hard.
  - Instead, perturb the data with Gaussian noise of different scales $$\sigma$$., and train the network to estimate
    - $$\nabla_\mathbf{x} \log p_\sigma(\mathbf{x})$$.
      - where $$p_\sigma(\mathbf{x})$$. is the data distribution convolved with Gaussian noise of variance $$\sigma^2$$..
    - How?)
      - Use the [perturbation kernel below](#21-denoising-score-matching-with-langevin-dynamics-smld) as
        - $$\tilde{\mathbf{x}} = \mathbf{x} + \sigma\cdot\epsilon,\quad\epsilon\sim\mathcal{N}(\mathbf{0,I})$$.
- Training)
  - Use denoising score matching.
    - How?)
      - Predict the score at multiple noise levels with a weighted loss across $$\sigma$$..
- Why useful)
  - By conditioning on $$\sigma$$., the network can learn a family of score functions across multiple noise scales.
  - During sampling, you start from large $$\sigma$$. (almost pure Gaussian noise) and progressively denoise using Langevin dynamics guided by $$s_\theta(\mathbf{x},\sigma)$$..
- Connection with SDE)
  - NCSN is the discrete, multi-noise-level predecessor of the continuous-time SDE formulation (Song et al. 2021).

<br>

### 2.1 Denoising Score Matching with Langevin Dynamics (SMLD)
- Problem Setting)
  - Let
    - $$p_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) := \mathcal{N} (\tilde{\mathbf{x}};\; \mathbf{x},\sigma^2\mathbf{I})$$. : the perturbation kernel
      - cf.) $$\tilde{\mathbf{x}} = \mathbf{x} + \sigma\cdot\epsilon,\quad\epsilon\sim\mathcal{N}(\mathbf{0,I})$$.
    - $$p_\sigma(\tilde{\mathbf{x}}) := \displaystyle\int p_\text{data}(\mathbf{x}) p_\sigma(\tilde{\mathbf{x}}\mid\mathbf{x}) \text{d}\mathbf{x}$$.
      - where $$p_\text{data}(\mathbf{x})$$. denotes the data distribution
    - $$\sigma_{\min} = \sigma_1 \lt \sigma_2 \lt \cdots \lt \sigma_N = \sigma_{\max}$$. : a sequence of positive noise scales
      - where
        - $$\sigma_{\min}$$. is small enough s.t. $$p_{\sigma_{\min}}(\mathbf{x})\approx p_\text{data}(\mathbf{x})$$.
        - $$\sigma_{\max}$$. is large enough s.t. $$p_{\sigma_{\max}}(\mathbf{x})\approx \mathcal{N}(\mathbf{x}; \mathbf{0},\sigma_{\max}^2\mathbf{I})$$.
- Model)
   - $$s_\theta(\mathbf{x}, \sigma)$$. : [Noise Conditional Score Network (NCSN)](#concept-noise-conditional-score-network-ncsn)
      - We want to train this neural network model!
- Training)
  - $$\boldsymbol{\theta}^* = \displaystyle\arg\min_{\boldsymbol{\theta}} \sum_{i=1}^N \sigma_i^2\; \mathbb{E}_{p_{\text{data}}(\mathbf{x})}\; \mathbb{E}_{p_{\sigma_i}(\tilde{\mathbf{x}}\mid\mathbf{x})} \left[ \Vert s_\theta(\tilde{\mathbf{x}}, \sigma) - \nabla_{\tilde{\mathbf{x}}} \log p_{\sigma_i} (\tilde{\mathbf{x}}\mid\mathbf{x}) \Vert_2^2 \right]$$.
    - Prop.)
      - Given sufficient data and model capacity, the optimal model $$s_{\boldsymbol{\theta}^*}(\mathbf{x}, \sigma) \approx \nabla_\mathbf{x} \log p_{\sigma} (\mathbf{x})$$. almost everywhere for $$\sigma \in \{\sigma_i\}_{i=1}^N$$.
      - $$\sigma_i^2$$. weight is added.
        - Why?)
          - Bigger the noise $$\sigma_i$$. is, the lower the signal-to-noise ratio, and smaller the gradient magnitude becomes.
          - Adding the weight can adjust the magnitude.
- Sampling)
  - Run $$M$$. steps of Langevin MCMC to get a sample for each $$p_{\sigma_i}(\mathbf{x})$$. sequentially as
    - $$\mathbf{x}_i^m = \mathbf{x}_i^{m-1} + \epsilon_i \mathbf{s}_{\boldsymbol{\theta^*}} (\mathbf{x}_{i}^{m-1}, \sigma_i) + \sqrt{2\epsilon_i} \mathbf{z}_i^m$$.
      - where
        - $$m=1,2,\cdots,M$$.
        - $$\epsilon_i\gt0$$. : the step size
        - $$\mathbf{z}_i^m$$. : standard normal
    - The above is repeated in $$i = N, N-1, \cdots, 2, 1$$.
      - i.e.) Reverse order from the training : $$\sigma_{\max} \to \sigma_{\min}$$. (progressive denoising)
    - Starting point of the MCMC : $$M=0$$.
      - $$\mathbf{x}_N^0 \sim\mathcal{N}(\mathbf{x}\mid \mathbf{0}, \sigma_{\max}^2\mathbf{I})$$. : the largest noise case is the Gaussian
      - $$\mathbf{x}_i^0 = \mathbf{x}_{i+1}^M$$. when $$i\lt N$$. : the $i$-th noise levels reuse the sample from the previous $$(i+1)$$.-th level
      - As $$M\rightarrow\infty, \epsilon_i\rightarrow0, \forall i$$.
        - $$\mathbf{x}_1^M$$. becomes an exact sample from $$p_{\sigma_{\min}}(\mathbf{x}) \approx p_{\text{data}}(\mathbf{x})$$.
          - cf.) Recall that $$i=1$$. was the smallest noise $$\sigma_1$$.
  - Pseudocode
    ```python
    # Initialization at largest noise : M=0, i=N
    x = sample_normal(mean=0, var=sigma[N]**2)

    for i in range(N, 0, -1):  # N, N-1, ..., 1
        for m in range(1, M+1):
            z = normal(0, I)
            x = x + eps[i] * s_theta(x, sigma[i]) + sqrt(2*eps[i]) * z
        # after finishing M steps at level i,
        # x becomes the initial point for level i-1
    ```

<br>

### 2.2 Denoising Diffusion Probabilistic Models (DDPM)
[hozy note](/blog/2025/ddpm)
- Problem Settings)
  - $$0 \lt \beta_1, \beta_2, \cdots, \beta_N \lt 1$$. : a sequence of positive noise scales
  - $$\mathbf{x}_0 \sim p_{\text{data}}(\mathbf{x})$$. : a datapoint generated by $$p_{\text{data}}$$.
  - $$\{\mathbf{x}_0, \mathbf{x}_1,\cdots,\mathbf{x}_N\}$$. : a discrete Markov chain 
    - where 
      - $$p(\mathbf{x}_i\mid\mathbf{x}_{i-1}) = \displaystyle\mathcal{N}(\mathbf{x}_i;\;\sqrt{1-\beta_i} \mathbf{x}_{i-1}, \beta_i\mathbf{I})$$.
    - Further using $$\alpha_i := \displaystyle\prod_{j=1}^i (1-\beta_j)$$., we may denote as
      - $$p_{\alpha_i}(\mathbf{x}_i\mid\mathbf{x}_0) = \mathcal{N}(\mathbf{x}_i;\;\sqrt{\alpha_i}\mathbf{x}_0, (1-\alpha_i)\mathbf{I})$$.
        - This can be treated as a perturbed distribution using $$\alpha_i$$. just like in [SMLD](#21-denoising-score-matching-with-langevin-dynamics-smld) with $$\sigma$$. as
          - $$p_{\alpha_i}(\tilde{\mathbf{x}}) := \displaystyle\int p_{\text{data}}(\mathbf{x}) p_{\alpha_i}(\tilde{\mathbf{x}}\mid\mathbf{x})\text{d}\mathbf{x}$$.
    - $$\mathbf{x}_N \sim \mathcal{N}(0,1)$$. : pure noise!
  - $$p_{\theta}(\mathbf{x}_{i-1}\mid\mathbf{x}_i) = \displaystyle\mathcal{N}(\mathbf{x}_{i-1};\;\frac{1}{\sqrt{1-\beta_i}}(\mathbf{x}_i+\beta_i \mathbf{s}_\theta (\mathbf{x}_i, i)), \beta_i\mathbf{I})$$. : a variational Markov chain in reverse direction
- Training)
  - Using the ELBO, we may get
    - $$\theta^* = \displaystyle\arg\min_\theta \sum_{i=1}^N (1-\alpha_i) \mathbf{E}_{p_{\text{data}}(\mathbf{x})} \mathbb{E}_{p_{\alpha_i}(\tilde{\mathbf{x}}\mid\mathbf{x})} \left[ \left\Vert \mathbf{s}_\theta(\tilde{\mathbf{x}}, i) - \nabla_{\tilde{\mathbf{x}}} \log p_{\alpha_i} (\tilde{\mathbf{x}}\mid\mathbf{x}) \right\Vert _2^2 \right]$$.
  - The optimal model is denoted as
    - $$\mathbf{s}_{\theta^*} (\mathbf{x}, i)$$.
- Sampling)
  - Start from $$\mathbf{x}_N \sim \mathcal{N}(\mathbf{0,I})$$..
  - Following the estimated reverse Markov chain, we have
    - $$\mathbf{x}_{i-1} = \displaystyle\frac{1}{\sqrt{1-\beta_i}}( \mathbf{x}_i + \beta_i \mathbf{s}_{\theta^*}(\mathbf{x}_i, i)) + \sqrt{\beta_i}\mathbf{z}_i, \quad i=N,N-1,\cdots,1$$.


<br><br>

## 3. Score-Based Generative Modeling with SDEs
### 3.1 Perturbing data with SDEs
- Goal)
  - Construct a diffusion process $$\{\mathbf{x}(t)\}_{t=0}^T$$. where $$t\in[0,T]$$. is a continuous time variable.
    - where
      - $$\mathbf{x}(0) \sim p_0$$. : dataset from the data distribution $$p_0$$.
      - $$\mathbf{x}(T) \sim p_T$$. : a tractable form from the prior distribution $$p_T$$.
  - This diffusion process can be modeled as the solution to an [Itô SDE](#concept-stochastic-differential-equation-sde):
    - $$\text{d}\mathbf{x} = \mathbf{f}(\mathbf{x}, t) \text{d}t + g(t)\text{d}\mathbf{w}$$.
      - where
        - $$\mathbf{w}$$. : the standard Wiener process (Brownian motion)
        - $$\mathbf{f}(\cdot, t) : \mathbb{R}^d \rightarrow \mathbb{R}^d$$. : a vector-valued function called the drift coefficient of $$\mathbf{x}(t)$$.
        - $$g(\cdot):\mathbb{R}\rightarrow\mathbb{R}$$. : a scalar function known as the diffusion coefficient of $$\mathbf{x}(t)$$.
          - For simplicity, we assume that $$g$$. is a scalar and does not depend on $$\mathbf{x}$$.
  - We want to generalize [SMLD](#21-denoising-score-matching-with-langevin-dynamics-smld) and [DDPM](#22-denoising-diffusion-probabilistic-models-ddpm) using [SDE](#concept-stochastic-differential-equation-sde).
- Notations)
  - $$p_t(\mathbf{x})$$. : the probability density of $$\mathbf{x}(t)$$.
  - $$p_{st}(\mathbf{x}(t)\mid\mathbf{x}(s))$$. : the transition kernel from $$\mathbf{x}(s)$$. to $$\mathbf{x}(t)$$.
    - where $$0\ll s\lt t\ll T$$.
  - $$p_T$$. : an unstructured prior distribution that contains no information of $$p_0$$.

<br>

### 3.2 Generating Samples by Reversing the SDE
- Ideation)
  - Now, think about the reverse process that starts from $$\mathbf{x}(T)\sim p_T$$., and obtains $$\mathbf{x}(0)\sim p_0$$. in the end.
  - The reverse of a diffusion process is also a diffusion process
    - cf.) Anderson (1982), *Reverse-time diffusion equation models.*
#### Concept) Reverse-Time SDE
- Def.)
  - $$\text{d}\mathbf{x} = [\mathbf{f}(\mathbf{x}, t) - g(t)^2 \nabla_{\mathbf{x}} \log p_t (\mathbf{x})]\text{d}t + g(t) \text{d}\bar{\mathbf{w}}$$.
    - where
      - $$\bar{\mathbf{w}}$$. : a standard Wiener process when time flows backwards : $$T\rightarrow0$$.
      - $$\text{d}t$$. : an infinitesimal negative time step
      - $$\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$$. : the [score](#concept-score) of each marginal distribution
- Prop.)
  - Once $$\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$$. is known for $$\forall t$$., we can derive the reverse diffusion process.
  - And, we may simulate it to sample from $$p_0$$. 
    - Question)
      - How do we get to know $$\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$$.?
      - We cannot.
      - Instead, we may [approximate it using the neural network](#33-estimating-scores-for-the-sde).

<br>

### 3.3 Estimating Scores for the SDE
- Goal)
  - We want to estimate the [score](#concept-score) $$\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$$..
- Model)
  - $$\theta^* = \displaystyle\arg\min_\theta \mathbb{E}_t \left\{ \lambda(t) \; \mathbb{E}_{\mathbf{x}(0)} \; \mathbb{E}_{\mathbf{x}(t) \mid \mathbf{x}(0)} \left[ \left\Vert \mathbf{s}_{\theta}(\mathbf{x}(t), t) - \nabla_{\mathbf{x}(t)} \log p_{0t} (\mathbf{x}(t) \mid \mathbf{x}(0)) \right\Vert_2^2 \right] \right\}$$.
    - where
      - $$\mathbf{s}_{\theta}(\mathbf{x}(t), t)$$. : a time-dependent score-based model
      - $$\lambda:[0,T]\rightarrow \mathbb{R}_{\gt 0}$$. : a positive weighting function
        - We choose $$\lambda \varpropto \displaystyle\frac{1}{\mathbb{E}\left[ \left\Vert \nabla_{\mathbf{x}(t)} \log p_{0t}(\mathbf{x}(t) \mid \mathbf{x}(0)) \right\Vert_2^2 \right]}$$.
      - $$t\sim\text{Uniform}(0,T)$$.
      - $$\mathbf{x}(0) \sim p_0(\mathbf{x})$$.
      - $$\mathbf{x}(t) \sim p_{0t}(\mathbf{x}(t)\mid\mathbf{x}(0))$$.
        - Props.)
          - We need to know the transition kernel $$p_{0t}(\mathbf{x}(t)\mid\mathbf{x}(0))$$. to efficiently solve the above problem.
          - e.g.) 
            - If $$\mathbf{f}(\cdot, t)$$. is affine, the transition kernel is always Gaussian.
            - Thus, we may get the closed form mean and variance.
          - e.g.) 
            - Sample from $$p_{0t}(\mathbf{x}(t)\mid\mathbf{x}(0))$$. and replace denoising score matching.

<br>

### 3.4 Examples: VE, VP SDEs and Beyond
- Idea)
  - The noise perturbations in [SMLD](#21-denoising-score-matching-with-langevin-dynamics-smld) and [DDPM](#22-denoising-diffusion-probabilistic-models-ddpm) can be viewed as discretizations of two different SDEs.

#### Model) VE SDE
- Desc.)
  - For [SMLD](#21-denoising-score-matching-with-langevin-dynamics-smld), the Markov chain is defined as
    - $$\mathbf{x}_i = \mathbf{x}_{i-1} + \sqrt{\sigma_i^2 - \sigma_{i-1}^2} \mathbf{z}_i,\quad i=1,\cdots,N$$.
  - This converges to the SDE
    - $$\text{d}\mathbf{x} = \displaystyle\sqrt{\frac{\text{d}[\sigma^2(t)]}{\text{d}t}} \text{d}\mathbf{w}$$.
  - This is called the Variance Exploding (VE) SDE
    - Why?) It's variance diverges at $$t\rightarrow\infty$$.

#### Model) VP SDE
- Desc.)
  - For [DDPM](#22-denoising-diffusion-probabilistic-models-ddpm), the discrete process is 
    - $$\mathbf{x}_i = \sqrt{1-\beta_i} \mathbf{x}_{i-1} + \sqrt{\beta_i} \mathbf{z}_{i-1},\quad i=1,\cdots,N$$.
  - This converges to the SDE
    - $$\text{d}\mathbf{x} = \displaystyle-\frac{1}{2}\beta(t)\text{d}t + \sqrt{\beta(t)} \text{d}\mathbf{w}$$.
  - This is called the Variance Preserving (VP) SDE
    - Why?) Its variance remains bounded.

#### Model) sub-VP SDE
- Def.)
  - $$\text{d}\mathbf{x} = \displaystyle-\frac{1}{2} \beta(t)\mathbf{x}\text{d}t + \sqrt{\beta(t)\left( 1- e^{-2\int_0^t \beta(s)\text{d}s} \right)} \text{d}\mathbf{w}$$.
- Prop.)
  - This process is always bounded by the [VP SDE](#model-vp-sde) and has better likelihood properties.


<br>

## 4. Solving the Reverse SDE
### 4.1 General-Purpose Numerical SDE Solvers
- Goal)
  - Approximate **trajectories** of [SDE](#concept-stochastic-differential-equation-sde)s numerically for reverse-time sampling
    - cf.) Trajectory in SDEs (Diffusion Process in [Data Perturbation](#31-perturbing-data-with-sdes))
      - Def.)
        - A continuous path $$\{\mathbf{x}_t\}_{t=0}^T$$. generated by solving [SDE](#concept-stochastic-differential-equation-sde)
      - Problem)
        - In practice, trajectories cannot be solved exactly.
        - Thus, using numerical solvers, we approximate them with discrete time steps
          - $$\mathbf{x}_{T}\rightarrow\mathbf{x}_{t_{N-1}}\rightarrow\cdots\rightarrow\mathbf{x}_{t_1}\rightarrow\mathbf{x}_{0}$$.
- Methods)
  - Use standard numerical solvers
    - e.g.)
      - Euler-Maruyama
      - stochastic Runge-Kutta
- Key Idea)
  - Any of these solvers can be applied to the reverse-time SDE to generate samples.

### 4.2 Predictor–Corrector (PC) Samplers
- Desc.)
  - We can improve sampling because we already have a score model $$\nabla_\mathbf{x} \log p_t(\mathbf{x})$$.
    - How?)
      - Combine **Predictor** (numerical SDE solver step) with a **Corrector** (score-based MCMC) step
        - **Predictor** estimates the next sample. 
        - **Corrector** adjusts the distribution to be closer to the true marginal.
  - This hybrid method generalizes both SMLD and DDPM sampling.
- Experiments)
  - Reverse diffusion samplers outperform ancestral sampling.
  - Adding a corrector step consistently improves sample quality (though it increases computation).
  - Corrector-only samplers perform worse than predictor–corrector combinations.

### 4.3 Probability Flow and Neural ODEs
- Desc.)
  - Every diffusion SDE has a deterministic counterpart: the probability flow ODE.
  - This ODE shares the same marginal distributions as the SDE.
- Benefits)
  - Enables exact likelihood computation 
    - cf.) Negative log-likelihoods can be measured exactly
  - Allows manipulation of latent representations 
    - e.g.) interpolation, image editing
  - Provides uniquely identifiable encodings, unlike many other generative models.
- Prop.)
  - Efficient sampling
    - with black-box ODE solvers, we can explicitly trade off accuracy vs. speed.

### 4.4 Architecture Improvements
- Desc.)
  - Explored new architectures for score-based models with [VE](#model-ve-sde), [VP](#model-vp-sde), and [sub-VP](#model-sub-vp-sde) SDEs.
  - Best Models)
    - NCSN++ cont ([VE](#model-ve-sde))
      - Achieves state-of-the-art FID on CIFAR-10 and high-resolution samples
    - DDPM++ cont. ([sub-VP](#model-sub-vp-sde)) 
      - Achieves exact log-likelihood of 2.99 bits/dim, a new record on CIFAR-10.
  - Improvements come from:
    - Switching to continuous-time training objectives.
    - Increasing network depth.
    - Using PC samplers together with optimized architectures.
  - Findings:
    - [VE SDE](#model-ve-sde)s give **strong sample quality**.
    - [sub-VP SDE](#model-sub-vp-sde)s achieve **higher likelihoods**.
      - i.e.) Better captures the data distribution $$p_{\text{data}}$$.
    - Indicates practitioners should choose SDEs depending on whether sample quality or likelihood is the priority.

<br><br>

## 5. Controllable Generation 
- Key Idea)
  - The continuous SDE framework also allows conditional generation
    - i.e.) If we know $$p_t(\mathbf{y}\mid\mathbf{x}(t))$$., we may get $$p_0(\mathbf{x}(0)\mid\mathbf{y})$$. 
  - This is done by solving a conditional [reverse-time SDE](#concept-reverse-time-sde)
- General Approach)
  - Following equation can be used to solve many inverse problems with score-based models.
    - $$\text{d}\mathbf{x} = \left[\mathbf{f}(\mathbf{x}, t) - g(t)^2 \left( \nabla_{\mathbf{x}} \log p_t (\mathbf{x}) + \underbrace{\nabla_{\mathbf{x}} \log p_t (\mathbf{y}\mid\mathbf{x}(t))}_{\text{cond. dist. added!}} \right)\right]\text{d}t + g(t) \text{d}\bar{\mathbf{w}}$$.
      - cf.) The conditional distribution is added to the [reverse-time SDE](#concept-reverse-time-sde) 
  - This requires the gradient of the forward process $$\nabla_{\mathbf{x}} \log p_t (\mathbf{y}\mid\mathbf{x}(t))$$..
    - How to get it)
      - Estimate by training an auxiliary model
        - e.g.) Time-dependent classifier
      - Approximate via heuristics / domain knowledge
- Applications)
  - Class-conditional generation
    - Let $$\mathbf{y}$$. be a class label.
    - Train a time-dependent classifier $$p_t (\mathbf{y}\mid\mathbf{x}(t))$$..
    - Use it inside the queation above.
  - Imputation
    - Special case of conditional generation when only part of the data point is known : $$\Omega(\mathbf{y})$$.
    - Then using the known part, we may sample the missing components with $$p(\mathbf{x}(0)\mid\Omega(\mathbf{y}))$$.
    - Can be done with unconditional score models, but accuracy improves if we transform data into a space where known/unknown dimensions are **decoupled**.
  - Colorization
    - Special case of imputation where the grayscale input is known and color channels are missing.
    - Achieved by applying the same imputation framework.