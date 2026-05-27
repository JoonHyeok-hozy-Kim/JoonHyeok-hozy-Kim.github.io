---
layout: post
title: Variational Inference with Normalizing Flows
date: 2026-02-28 11:00:00
description: Rezende et al.   
tags: LOTUS planar_flow radial_flow
categories: normalizing_flow
# pdf: _posts/250908_style_gan/slides_style_gan.pdf
pretty_table: true
---

## hozy Summary
- Start from the [ELBO Loss](#concept-approximated-posterior-distribution-for-the-latent).
  - $$\log p_\theta(\mathbf{x})\ge-\mathbb{D}_{\text{KL}}\left[q_\phi(\mathbf{z}\mid\mathbf{x}) \Vert p(\mathbf{z}) \right] + \mathbb{E}_{q}\left[ \log p_\theta(\mathbf{x}\mid\mathbf{z}) \right]=-\mathcal{F}(\mathbf{z})$$.
- Simplify the loss with [Normalizing Flow](#3-normalizing-flows) and [LOTUS](#concept-the-law-of-the-unconscious-statistician-lotus)
  - $$\begin{aligned}
      \mathbb{E}_q\left[ \log \frac{p_\theta(\mathbf{x}\mid\mathbf{z}) \; p(\mathbf{z})}{q_\phi(\mathbf{z}\mid\mathbf{x})} \right] 
      &= \mathbb{E}_q\left[ \underbrace{\log p_\theta(\mathbf{x},\mathbf{z})}_{h_1} - \underbrace{\log q_\phi(\mathbf{z}\mid\mathbf{x})}_{h_2} \right] \\
      &= \mathbb{E}_{q_0}[\log p_\theta(\mathbf{x},f(\mathbf{z}_0))] - \mathbb{E}_q\left[ \log q_\phi(\mathbf{z}\mid\mathbf{x}) \right]
    \end{aligned}$$.
- Using [Planar Flow](#concept-planar-flows) or [Radial Flow](#concept-radial-flows), we may further simplify the objective.
  - e.g.) [Optimization using Planar Flow](#42-flow-based-free-energy-bound)

---

<br>

## 2 Amortized Variational Inference
- Settings)
  - $$\mathbf{x}$$. : observations
  - $$\mathbf{z}$$. : latent variables
  - $$\theta$$. : model parameters

#### Concept) Approximated Posterior Distribution for the latent 
- Def.) 
  - $$q_\phi(\mathbf{z}\mid\mathbf{x})$$.   
- Derivation)    
  $$\begin{aligned}
    \log p_\theta(\mathbf{x}) &= \log\int p_\theta(\mathbf{x}\mid\mathbf{z}) p(\mathbf{z}) \text{d}\mathbf{z} \\
    &= \log\int\frac{q_\phi(\mathbf{z}\mid\mathbf{x})}{q_\phi(\mathbf{z}\mid\mathbf{x})} \; p_\theta(\mathbf{x}\mid\mathbf{z}) \; p(\mathbf{z}) \text{d}\mathbf{z} \\
    &= \log\int\frac{p_\theta(\mathbf{x}\mid\mathbf{z}) \; p(\mathbf{z})}{q_\phi(\mathbf{z}\mid\mathbf{x})} \; q_\phi(\mathbf{z}\mid\mathbf{x}) \text{d}\mathbf{z} \\
    &= \log \mathbb{E}_q\left[ \frac{p_\theta(\mathbf{x}\mid\mathbf{z}) \; p(\mathbf{z})}{q_\phi(\mathbf{z}\mid\mathbf{x})} \right]  \\
    &\ge \mathbb{E}_q\left[ \log \frac{p_\theta(\mathbf{x}\mid\mathbf{z}) \; p(\mathbf{z})}{q_\phi(\mathbf{z}\mid\mathbf{x})} \right]  \\
    &= \mathbb{E}_q\left[ \log p_\theta(\mathbf{x}\mid\mathbf{z}) + \log \frac{ \; p(\mathbf{z})}{q_\phi(\mathbf{z}\mid\mathbf{x})} \right] \\
    &= \mathbb{E}_q\left[ \log p_\theta(\mathbf{x}\mid\mathbf{z}) \right] - \int \log \frac{q_\phi(\mathbf{z}\mid\mathbf{x})}{p(\mathbf{z})} q_\phi(\mathbf{z}\mid\mathbf{x})\text{d}\mathbf{z} \\
    &= -\mathbb{D}_{\text{KL}}\left[q_\phi(\mathbf{z}\mid\mathbf{x}) \Vert p(\mathbf{z}) \right] + \mathbb{E}_{q}\left[ \log p_\theta(\mathbf{x}\mid\mathbf{z}) \right] \\
    &= -\mathcal{F}(\mathbf{z})
  \end{aligned}$$.
  - Other names
    - Negative Free Energy $$\mathcal{F}$$.
    - ELBO
  - Prop)
    - Consists of two terms
      1. $$-\mathbb{D}_{\text{KL}}\left[q_\phi(\mathbf{z}\mid\mathbf{x}) \Vert p(\mathbf{z}) \right]$$. 
         - KL-divergence between the approximated posterior ($$q_\phi(\mathbf{z}\mid\mathbf{x})$$.) and the prior distribution $$(p(\mathbf{z}))$$.
      2. $$\mathbb{E}_{q}\left[ \log p_\theta(\mathbf{x}\mid\mathbf{z}) \right]$$.  
         - Reconstruction Error
    - Provides the unified objective function for $$\theta$$. and $$\phi$$.

#### Tech.) Variational Inference
- How?)
  - Use... 
    - the [ELBO loss](#concept-approximated-posterior-distribution-for-the-latent)
    - mini-batch strategy
    - stochastic gradient descent
- Limit)
  - Computational cost on calculating $$\nabla_\phi \mathbb{E}_{q}\left[ \log p_\theta(\mathbf{x}\mid\mathbf{z}) \right]$$.
  - Choosing the richest, computationally feasible posterior $$q$$.
    - This paper tackles this problem!
- Practices)
  - Stochastic Backpropagation
    - Kingma et al. 2014, Stochastic Gradient Variational Bayes
      - Two steps)
        - Reparameterization
          - $$q_\theta(z) \sim\mathcal{N}(z\mid \mu,\sigma^2)$$.
        - Backpropagation with Monte Carlo
          - $$\nabla_\phi \mathbb{E}_{q}\left[ f_\theta(z) \right] \Leftrightarrow \mathbb{E}_{\mathcal{N}(\epsilon\mid0,1)}\left[ \nabla_\phi f_\theta(\mu+\sigma\epsilon) \right]$$.
  - Inference Networks
    - Goal)
      - Learn an inverse map from observations to latent variables
    - Advantage)
      - No need to compute per datapoint variational parameters
      - Instead, compute global variational parameters $$\phi$$. valid for both training and test time
        - Amortizing the cost of inference by generalizing between the posterior estimates for all latent variables through parameters of the inference network
    - e.g.)
      - Diagonal Gaussian Densities
        - $$q_\phi(\mathbf{z}\mid\mathbf{x}) = \mathcal{N}(\mathbf{z}\mid\mu_\phi(\mathbf{x}), \text{diag}(\sigma^2_\phi(\mathbf{x})))$$.
          - where $$\mu_\phi, \sigma^2_\phi$$. are specified using deep neural network
  - Deep Latent Gaussian Models (DLGM)
    - Desc.)
      - Deep directed graphical model with the $$L$$. layers of Gaussian latent variables $$\mathbf{z}_l$$.
        - where $$l=1,2,\ldots, L$$.
      - Each layer of latent variables is dependent on the layer above in a non-linear way
      - Joint Probability
        - $$p(\mathbf{x}, \mathbf{z}_1,\ldots,\mathbf{z}_L) = p(\mathbf{x}\mid f_0(\mathbf{z}_1)) \displaystyle\prod_{l=1}^L p(\mathbf{z}_l \mid f_l(\mathbf{z}_{l+1}))$$.
          - where
            - $$p(\mathbf{z}_L) = \mathcal{N}(\mathbf{0, I})$$. : the prior over latent variables
            - $$p_\theta(\mathbf{x}\mid\mathbf{z})$$. : the observation likelihood as any appropriate distribution
              - parameterized by a deep neural network $$\theta$$.

<br><br>

## 3 Normalizing Flows
- Goal)
  - We want to find the latent posterior $$q_\phi$$. s.t. 
    - $$q_\phi(\mathbf{z}\mid\mathbf{x}) \approx p_\theta(\mathbf{z}\mid\mathbf{x}) \Leftrightarrow \mathbb{D}_{\text{KL}}(q\Vert p)\approx 0$$.
- Def.)
  - Normalizing Flow
    - A **sequence** of invertible mappings that transforms an **initial simple density** (flows) into a valid, highly flexible probability distribution.

<br>

### 3.1 Finite Flows
- Settings)
  - $$f:\mathbb{R}^d\rightarrow\mathbb{R}^d$$. : an invertible smooth mapping s.t.
    - $$f^{-1} = g$$., i.e. $$g\circ f(\mathbf{z}) = \mathbf{z}$$.
      - Prop.)
        - For a random variable $$\mathbf{z}$$. and it's distribution $$q$$., and $$\mathbf{z}' = f(\mathbf{z})$$.
          - $$q_{\text{new}}(\mathbf{z}') = q_{\text{old}}(\mathbf{z})\displaystyle\left\vert\text{det}\frac{\partial f^{-1}}{\partial \mathbf{z}'}\right\vert = q_{\text{old}}(\mathbf{z})\left\vert\text{det}\frac{\partial f}{\partial \mathbf{z}}\right\vert^{-1}$$.
  - $$q_K(\mathbf{z})$$. : the density obtained by successively transforming $$\mathbf{z}_0$$. through a chain of $$K$$. transformations $$f_K$$.
    - $$\mathbf{z}_K = f_K\circ\cdots\circ f_1(\mathbf{z}_0)$$.
      - Notation)   
        $$\begin{aligned}
          \mathbf{z}_K 
          &= f_K(f_{K-1}(\cdots f_2(f_1(\mathbf{z_0})))) = f_K(f_{K-1}(\cdots f_2(\mathbf{z_1}))) & (\mathbf{z_1} = f_1(\mathbf{z_0})) \\ 
          &\quad\vdots \\
          &= f_K(f_{K-1}(\mathbf{z}_{K-2})) = f_K(\mathbf{z}_{K-1}) & (\mathbf{z}_{K-1} = f_{K-1}(\mathbf{z}_{K-2})) \\
        \end{aligned}$$.
    - $$\ln q_K(\mathbf{z}_K) = \ln q_0 (\mathbf{z}_0) - \displaystyle\sum_{k=1}^K\ln\left\vert\text{det}\frac{\partial f_k}{\partial \mathbf{z}_{k-1}}\right\vert \quad\cdots\quad(A)$$.
      - Derivation)   
        $$\begin{aligned}
          \ln q_K(\mathbf{z}_K) &= \ln q_{K-1}(f_K(\mathbf{z}_{K-1})) \\
          &= \ln \left( q_{K-1}(\mathbf{z}_{K-1}) \left\vert\text{det}\frac{\partial f_K}{\partial \mathbf{z}_{K-1}}\right\vert^{-1} \right) = \ln q_{K-1}(\mathbf{z}_{K-1}) - \ln \left\vert\text{det}\frac{\partial f_K}{\partial \mathbf{z}_{K-1}}\right\vert \\
          &= \ln q_{K-2}(\mathbf{z}_{K-2}) - \ln \left\vert\text{det}\frac{\partial f_{K-1}}{\partial \mathbf{z}_{K-2}}\right\vert - \ln \left\vert\text{det}\frac{\partial f_K}{\partial \mathbf{z}_{K-1}}\right\vert = \cdots \\
          &= \ln q_{0}(\mathbf{z}_{0}) - \sum_{k=1}^K \ln \left\vert\text{det}\frac{\partial f_k}{\partial \mathbf{z}_{k-1}}\right\vert \\
        \end{aligned}$$.

#### Concept) The Law of the Unconscious Statistician (LOTUS)
- Thm.)
  - $$\mathbb{E}_{q_K} \left[ h(\mathbf{z}_K) \right] = \mathbb{E}_{q_0} \left[ h(f_K\circ\cdots\circ f_1(\mathbf{z}_0)) \right]$$.
    - Why?)   
      $$\begin{aligned}
        \mathbb{E}_{q_K} \left[ h(\mathbf{z}_K) \right] &= \int h(\mathbf{z}_K) \; q_K(\mathbf{z}_K) \text{d} \mathbf{z}_K \\
        &= \int h(f_K\circ\cdots\circ f_1(\mathbf{z}_0)) \; q_K(\mathbf{z}_K) \text{d} \mathbf{z}_K & (\because \mathbf{z}_K \triangleq f_K\circ\cdots\circ f_1(\mathbf{z}_0)) \\
        &= \int h(f_K\circ\cdots\circ f_1(\mathbf{z}_0)) \underbrace{\left( q_0 (\mathbf{z}_0) \left\vert\text{det}\frac{\partial f}{\partial \mathbf{z}_0}\right\vert^{-1} \right)}_{=q_K(\mathbf{z}_K)} \underbrace{\left( \left\vert\text{det}\frac{\partial f}{\partial \mathbf{z}_0}\right\vert \text{d} \mathbf{z}_0 \right)}_{=\text{d} \mathbf{z}_K} & (\text{Put } f = f_K\circ\cdots\circ f_1) &\quad (\text{ cf. } \frac{\partial f}{\partial \mathbf{z}_0} = \frac{\partial f_K}{\partial \mathbf{z}_{K-1}}\frac{\partial f_{K-1}}{\partial \mathbf{z}_{K-2}}\cdots\frac{\partial f_1}{\partial \mathbf{z}_{0}}  ) \\
        &= \int h(f_K\circ\cdots\circ f_1(\mathbf{z}_0)) q_0 (\mathbf{z}_0) \text{d} \mathbf{z}_0  \\
        &= \mathbb{E}_{q_0} \left[ h(f_K\circ\cdots\circ f_1(\mathbf{z}_0)) \right]
      \end{aligned}$$.
- Meaning)
  - Expectations w.r.t. the transformed $$q_K$$. can be computed without explicitly knowing $$q_K$$..
    - i.e.) If $$h(\mathbf{z})$$. is independent on $$q_K$$., $$\mathbb{E}_{q_K}$$. does not require calculating the Jacobian terms!
- Application)
  - Recall from the [ELBO loss](#concept-approximated-posterior-distribution-for-the-latent) that   
    $$\begin{aligned}
      \mathbb{E}_q\left[ \log \frac{p_\theta(\mathbf{x}\mid\mathbf{z}) \; p(\mathbf{z})}{q_\phi(\mathbf{z}\mid\mathbf{x})} \right] 
      &= \mathbb{E}_q\left[ \underbrace{\log p_\theta(\mathbf{x},\mathbf{z})}_{h_1} - \underbrace{\log q_\phi(\mathbf{z}\mid\mathbf{x})}_{h_2} \right] 
    \end{aligned}$$.
  - $$h_1$$. can be simplified using LOTUS as...   
    $$\begin{aligned}
      \mathbb{E}_{q_K}[\log p_\theta(\mathbf{x},\mathbf{z})] 
      &= \int \log p_\theta(\mathbf{x},\mathbf{z}) q(\mathbf{z}_k) \text{d} \mathbf{z}_K \\
      &= \int \log p_\theta(\mathbf{x},f(\mathbf{z}_0)) q(\mathbf{z}_k) \text{d} \mathbf{z}_K & (\because \mathbf{z} = f_K\circ\cdots\circ f_1(\mathbf{z}_0)) \\
      &= \int \log p_\theta(\mathbf{x},f(\mathbf{z}_0)) q(\mathbf{z}_0) \text{d} \mathbf{z}_0 & (\because \text{LOTUS}) \\
      &= \mathbb{E}_{q_0}[\log p_\theta(\mathbf{x},f(\mathbf{z}_0))] \\
    \end{aligned}$$.
      - cf.) No need to calculate the Jacobian determinants.
  - $$h_2$$. needs Jacobian determinant calculation.
    - why?)
      - The transformation $$h_2=q_\phi$$., which is the $$q_K$$. itself.
  

### 3.2 Infinitesimal Flows
- Desc.)
  - The length of the normalizing flow tends to infinity.
    - Still finite!
  - Utilize the partial differential equation to describe how $$q_0(\mathbf{z}_0)$$. evolves over time(t)
    - i.e.) $$\displaystyle\frac{\partial}{\partial t} q_t(\mathbf{z}_t) = \mathcal{T}_t[q_t(\mathbf{z}_t)]$$.
- e.g.)
  - Langevin Flow
    - Def.)
      - Langevin SDE
        - $$\text{d}\mathbf{z}(t) = \mathbf{F}(\mathbf{z}(t), t)\text{d}t + \mathbf{G}(\mathbf{z}(t),t)\text{d}\xi(t)$$.
          - where
            - $$\text{d}\xi(t)$$. : a Wiener process with 
              - $$\mathbb{E}[\xi(t)] = 0$$. 
              - $$\mathbb{E}[\xi_i(t)\xi_j(t')] = \delta_{i,j}\delta(t-t')$$.
                - with 
                  - the Kronecker delta $$\delta_{i,j} = \begin{cases} 1&\text{if } i=j\\ 0&\text{otherwise} \end{cases}$$.
                  - the Dirac delta $$\delta(t-t') = \begin{cases} \infty&\text{if } t=t'\\ 0&\text{otherwise} \end{cases}$$.
            - $$\mathbf{F}$$. : the drift vector
            - $$\mathbf{D} = \mathbf{GG}^\top$$. : the diffusion matrix
      - Putting $$q_t(\mathbf{z})$$. to be the probability distribution of $$\mathbf{z}$$., we may get the Fokker-Planck eqation as
        - $$\displaystyle\frac{\partial}{\partial t} q_t(\mathbf{z}) = -\sum_i \frac{\partial}{\partial z_i}[F_i(\mathbf{z}, t)q_t] + \frac{1}{2}\sum_{i,j}\frac{\partial^2}{\partial z_i \partial z_j} [D_{i,j}(\mathbf{z}, t)q_t]$$.
    - Usage)
      - In ML, we use
        - $$F(\mathbf{z},t) = -\nabla_z\mathcal{L}(\mathbf{z})$$.
          - where $$\mathcal{L}(\mathbf{z})$$. is the unnormalized log-density of the model
        - $$G(\mathbf{z},t) = \sqrt{2}\delta_{i,j}$$.
      - Sol.)
        - Assuming the Boltzmann Distribution as $$t\rightarrow\infty$$., i.e., $$q_\infty(\mathbf{z})\varpropto e^{-\mathcal{L}(\mathbf{z})}$$., we may get the stationary solution for $$q_t(\mathbf{z})$$. at $$t\rightarrow\infty$$.
  - Hamiltonian Flow   

{% include figure.liquid path="assets/img/blog/260228_nf_variational_inference/images/001.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<br><br>

## 4 Inference with Normalizing Flows

{% include figure.liquid path="assets/img/blog/260228_nf_variational_inference/images/002.png" class="img-fluid rounded z-depth-1" zoomable=true %}

### 4.1 Invertible Linear-time Transformers
#### Concept) Planar Flows
- Settings)
  - A neural network with...
    - $$L$$. : the number of hidden layers
    - $$D$$. : the hidden dimension of the hidden layers
      - cf.) Invertible neural networks take $$O(LD^3)$$. time to calculate the Jacobians
  - A family of transformation s.t.   
    - $$f(\mathbf{z}) = \mathbf{z} + \mathbf{u}h(\mathbf{w}^\top\mathbf{z} + b)$$.
      - where
        - $$\lambda = \{\mathbf{w, u}\in\mathbb{R}^D, b\in\mathbb{R}\}$$. : free parameters
        - $$h(\cdot)$$. : a smooth element-wise non-linearity with derivative $$h'(\cdot)$$.
- Jacobian Computation Tricks
  - Using $$\psi(\mathbf{z}) = h'(\mathbf{w^\top z}+b)\mathbf{w}$$., we may get the lodget-Jacobian term as
    - $$\displaystyle\left\vert\text{det}\frac{\partial f}{\partial\mathbf{z}}\right\vert = \left\vert\text{det}(\mathbf{I} + \mathbf{u\psi(z)}^\top)\right\vert = \left\vert 1 + \mathbf{u^\top\psi(z)} \right\vert$$.
  - Thus, we may get
    - $$\ln q_K(\mathbf{z}_K) = \ln q_0(\mathbf{z}) - \displaystyle\sum_{k=1}^K\ln\left\vert 1 + \mathbf{u}_k^\top\psi_{k}(\mathbf{z}_{k-1}) \right\vert\quad\cdots\quad(A)$$.
      - where $$\mathbf{z}_K = f_K\circ\cdots\circ f_1(\mathbf{z}_0)$$.
- Desc.)
  - $$(A)$$. modifies the initial density $$q_0$$. by applying a series of contractions and expansions in the direction perpendicular to the hyperplane $$\mathbf{w^\top z}+b=0$$.
  - $$O(D)$$. : linear lodget-Jacobian computation
  - Bimodal distribution
  - Invertible

<br>

#### Concept) Radial Flows
- Settings)
  - A family of transformation s.t.
    - $$f(\mathbf{z}) = \mathbf{z}+\beta h(\alpha, r)(\mathbf{z}-\mathbf{z}_0)$$.
      - where
        - $$\mathbf{z}_0$$. : a reference point
        - $$r = \vert \mathbf{z} - \mathbf{z}_0 \vert$$.
        - $$h(\alpha, r) = \displaystyle\frac{1}{\alpha+r}$$.
        - $$\lambda = \{\mathbf{z_0}\in\mathbb{R}^D, \alpha\in\mathbb{R}^+, \beta\in\mathbb{R}\}$$. : free parameters
- Jacobian Computation Tricks
  - $$\displaystyle\left\vert\text{det}\frac{\partial f}{\partial\mathbf{z}}\right\vert = \left[ 1+\beta h(\alpha, r) \right]^{d-1} \; \left[ 1+\beta h(\alpha, r) + \beta h'(\alpha, r)r \right]$$.
- Desc.)
  - Radial contractions and expansions around the reference point
  - $$O(D)$$. : linear lodget-Jacobian computation
  - Bimodal distribution
  - Invertible

<br>

### 4.2 Flow-Based Free Energy Bound
- Goal)
  - Specify the optimization target $$\mathcal{F}(\mathbf{x})$$. : the free energy
- Settings)
  - $$q_\phi(\mathbf{z\mid x}) := q_K(\mathbf{z}_K)$$. : an approximated posterior distribution with the flow of length $$K$$.
- Derivation)
  - Recall the [ELBO loss](#concept-approximated-posterior-distribution-for-the-latent) of 
    - $$\mathbb{E}_q\left[ \log p_\theta(\mathbf{x}\mid\mathbf{z}) + \log \frac{ \; p(\mathbf{z})}{q_\phi(\mathbf{z}\mid\mathbf{x})} \right] = \mathbb{E}_q\left[ \log p_\theta(\mathbf{x},\mathbf{z}) - \log q_\phi(\mathbf{z}\mid\mathbf{x}) \right] = -\mathcal{F}(\mathbf{x})$$.
      - cf.) [LOTUS](#concept-the-law-of-the-unconscious-statistician-lotus)
  - Plugging in the [Planar Flows](#concept-planar-flows), we may get   
    $$\begin{aligned}
      \mathcal{F}(\mathbf{x})
      &= \mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}\left[ \log q_\phi(\mathbf{z}\mid\mathbf{x}) - \log p_\theta(\mathbf{x},\mathbf{z}) \right] \\
      &= \mathbb{E}_{q_K(\mathbf{z}_K)}\left[ \log q_K(\mathbf{z}_K) \right] - \mathbb{E}_{q_0(\mathbf{z}_0)}\left[  \log p_\theta(\mathbf{x},\mathbf{z}_K) \right] & (\because\text{By def. of } \mathbf{z}_K \text{and LOTUS} )  \\
      &= \mathbb{E}_{q_0(\mathbf{z}_0)}\left[ \ln q_0(\mathbf{z}) - \displaystyle\sum_{k=1}^K\ln\left\vert 1 + {\mathbf{u}_k^\top\psi_k(\mathbf{z}_{k-1})} \right\vert \right] - \mathbb{E}_{q_0(\mathbf{z}_0)}\left[  \log p_\theta(\mathbf{x},\mathbf{z}_K) \right] & (\because\text{Planar Flow}) \\
    \end{aligned}$$.
- Optimization)
  - $$\displaystyle\arg\max_{\phi,\theta} \text{ELBO} = \displaystyle\arg\min_{\phi,\theta} \mathcal{F}(\mathbf{x})$$.
    - Analysis)
      - We may rewrite as   
        $$\begin{aligned}
          -\mathcal{F}(\mathbf{x}) 
          &= \mathbb{E}_{q_{\phi}(\mathbf{z\mid x})}\left[ \ln p_\theta(\mathbf{x},\mathbf{z}) - \ln q_\phi(\mathbf{z}\mid\mathbf{x}) \right] \\
          &= \mathbb{E}_{q_{\phi}(\mathbf{z\mid x})}\left[ \ln p_\theta(\mathbf{x},\mathbf{z}) \right] - \mathbb{E}_{q_{\phi}(\mathbf{z\mid x})}\left[ \ln q_\phi(\mathbf{z}\mid\mathbf{x}) \right] \\
          &= \mathbb{E}_{q_{\phi}(\mathbf{z\mid x})}\left[ -\underbrace{\mathcal{L}(z,x)}_{\text{Energy}} \right] \underbrace{- \mathbb{E}_{q_{\phi}(\mathbf{z\mid x})}\left[ \ln q_\phi(\mathbf{z}\mid\mathbf{x}) \right]}_{\text{Entropy}} \\
        \end{aligned}$$.
        - cf.)
          - $$F = E - TS$$ : Refer to [Free energy note](/blog/2026/free_energy/)
        - why?)
          - Energy (E) : converging to a certain point
            - Recall that $$p(\mathbf{z}, \mathbf{x}) = \displaystyle\frac{e^{-\mathcal{L}(\mathbf{z}, \mathbf{x})}}{Z}$$.
              - where
                - $$Z=\displaystyle\int e^{-\mathcal{L}(\mathbf{z}, \mathbf{x})}\text{d}\mathbf{z}$$.
                - $$\mathcal{L}(\mathbf{z}, \mathbf{x})$$. is the energy of the latent $$\mathbf{z}$$., jointly distributed with the data $$\mathbf{x}$$.
              - i.e.) Higher chance (probability) that $$\mathbf{z}$$. is at the low energy state.
            - Putting $$p(\mathbf{z}, \mathbf{x}) \varpropto e^{-\mathcal{L}(\mathbf{z}, \mathbf{x})}$$., we may rewrite as 
              - $$\mathcal{L}(\mathbf{z}, \mathbf{x}) = -\ln p(\mathbf{z}, \mathbf{x})$$.
                - i.e.) Energy = - log likelihood
                - cf.) why $$p(\mathbf{z}, \mathbf{x})$$.?
                  - The goal of ML is to learn $$p(\mathbf{z}\mid\mathbf{x}) = \displaystyle\frac{p(\mathbf{z}, \mathbf{x})}{p(\mathbf{x})}\varpropto p(\mathbf{z}, \mathbf{x})$$.
          - Entropy (S) : dispersing to chaos
            - By definition the entropy of the approximated posterior $$q_\phi$$. is $$-\displaystyle\int q_\phi(\mathbf{z}) \ln q_\phi \; \text{d}\mathbf{z}$$.
        - Hence, the ELBO maximization problem is equivalent to...
          - $$\mathcal{F}(\mathbf{x})$$. minimization
          - Energy minimization
          - Entropy maximization

<br>

### 4.3 Algorithm Summary and Complexity
- Algorithm)   

{% include figure.liquid path="assets/img/blog/260228_nf_variational_inference/images/003.png" class="img-fluid rounded z-depth-1" zoomable=true %}

- Inference Time Complexity
  - $$O(LN^2 + KD)$$.
    - where
      - $$L$$. : the number of deterministic layers used to map the data to the parameters of the flow
        - cf.) the encoder depth that maintains the dimension $$D$$.
      - $$N$$. : the average hidden layer size
      - $$K$$. : the flow-length
      - $$D$$. : the dimension of the latent variables


<br><br>

## 5 Alternative Flow-based Posteriors
### Concept) Volume Preserving Flows
- Goal)
  - Its Jacobian determinant is equal to 1.
  - Allow rich posterior distributions
- Types
  - Finite
    - e.g.) [Non-linear Independent Components Estimation (NICE)](#model-non-linear-independent-components-estimation-nice)
  - Infinitesimal
    - e.g.) [Hamiltonian Variational Approximation (HVI)](#model-hamiltonian-variational-approximation-hvi)

<br>

#### Model) Non-linear Independent Components Estimation (NICE)
- Methods)
  - Partition the latent vector into $$\mathbf{z} = (\mathbf{z}_A, \mathbf{z}_B)$$.
    - e.g.)
      - $$\mathbf{z} = (\mathbf{z}_{1:d}, \mathbf{z}_{d+1:D})$$.
  - Transformation
    - $$f(\cdot)$$. : neural network s.t.
      - has easy to compute inverse $$g(\cdot)$$.
        - $$f(\mathbf{z}) = (\mathbf{z}_A, \mathbf{z}_B + h_\lambda(\mathbf{z}_A))$$.
        - $$g(\mathbf{z}') = (\mathbf{z}_A', \mathbf{z}_B' + h_\lambda(\mathbf{z}_A'))$$.
          - where
            - $$h_\lambda$$. is a neural network with parameters $$\lambda$$.
  - Alternation between $$\mathbf{z}_A$$. and $$\mathbf{z}_B$$.
    - why?)
      - To mix all components of the initial random variable $$\mathbf{z}_0$$.
  - Resulting Density
    - $$\ln q_K(f_K\circ\cdots\circ f_1(\mathbf{z}_0)) = \ln q_0(\mathbf{z}_0)$$.
    - $$\ln q_K(\mathbf{z}') = q_0(g_1\circ \cdots\circ g_K(\mathbf{z}'))$$.
- Props.)
  - Jacobian with a zero upper triangular part resulting in a determinant of 1.

<br>

#### Model) Hamiltonian Variational Approximation (HVI)

<br>