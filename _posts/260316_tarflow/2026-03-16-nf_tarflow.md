---
layout: post
title: Normalizing Flows are Capable of Generative Models (Tarflow)
date: 2026-03-16 11:00:00
description: Zhai et al.
tags: tarflow transformer
categories: normalizing_flow
# pdf: _posts/250908_style_gan/slides_style_gan.pdf
pretty_table: true
---


## 2 Method
### 2.1 Normalizing Flows
- Settings)
  - $$x\in\mathbb{R}^D$$. : the input dataset 
    - where
      - $$x\sim p_{\text{data}}$$.
  - $$p_0$$. : the prior density
    - This paper uses $$p_0 = \mathcal{N}(0, \mathbf{I}_D)$$.
- Model)
  - $$p_{\text{model}}$$. : the model density
    - i.e.) $$\displaystyle p_{\text{model}} = p_0(f(x)) \; \left\vert\text{det}\frac{\partial f(x)}{\partial x}\right\vert$$.
      - where
        - $$f:\mathbb{R}^D\rightarrow\mathbb{R}^D$$. : the transformation
- Optimization)
  - MLE objective of
    - $$\displaystyle\min_f - \log p_0(f(x)) - \log\left\vert\text{det}\frac{\partial f(x)}{\partial x}\right\vert$$.
  - With the Gaussian prior $$p_0$$., we may rewrite as
    - $$\displaystyle\min_f \log \frac{1}{2}\Vert f(x) \Vert_2^2 - \log\left\vert\text{det}\frac{\partial f(x)}{\partial x}\right\vert$$.   
  - Applying the [AR flow transformation](#model-flow-transformation) below, we may get
    - $$\displaystyle\min_f \frac{1}{2}\Vert z^T \Vert_2^2 + \sum_{i=1}^{N-1} \sum_{j=1}^{D-1} \alpha_i^t(\tilde{z}_{\lt i}^t)$$.
- Result)
  - A generative model of
    - $$x = f^{-1}(z)$$.
      - where $$z=f(x)\sim p_0(z)$$. is the latent variable

<br>

### 2.2 Block Autoregressive Flows
- Desc.)
  - Stacking multiple layers of AR flows
    - cf.)
      - IAF (Kingma et al. 2016)
      - MAF (Papamakarios et al. 2017)
- Settings)
  - $$x\in\mathbb{R}^{N\times D}$$. : the input sequence data with length $$N$$. and dimension $$D$$.
    - $$i$$. : the subscript for indexing along the sequential dimension $$\mathbb{R}^D$$.
      - i.e.) $$x_i\in\mathbb{R}^D,\quad i=1,2,\ldots,N$$.
  - $$T\in\mathbb{N}$$. : the number of flow layers in the stack of flows
  - $$\{\pi^t\}\;(t=1,\ldots,T)$$. : any fixed set of permutation functions along the sequence dimension $$\mathbb{R}^D$$.
    - cf.) In this paper, 
      - $$\pi^t(z)_i = \begin{cases} z_{N-1-i} & (\text{Reverse function}) & t\gt0 \\ z_i & (\text{Identity funcition}) & t=0 \end{cases}$$.

#### Model) Flow Transformation
- Def.)
  - $$z^T = f(x) := (f^{T-1}\circ f^{T-2}\circ\cdots f^{0})(x)$$.
    - where
      - $$f^t$$. is parameterized by two learnable functions $$\mu^t, \alpha^t : \mathbb{R}^{N\times D}\rightarrow\mathbb{R}^{N\times D}$$.
        - cf.) Both causal along the sequence dimension
          - i.e.) Causality : Refer to $$\mu_i^t(\tilde{z}_{\lt i}^t)$$. and $$\alpha_i^t(\tilde{z}_{\lt i}^t)$$. below.
    - Dynamics
      - Input
        - $$z^t = \{z_i^t\}_{i\in\{1,\ldots,N\}} \in\mathbb{R}^{N\times D}$$.
      - Output
        - $$z^{t+1} = \{z_i^{t+1}\}_{i\in\{1,\ldots,N\}} \in\mathbb{R}^{N\times D}$$.
          - where
            - $$z_i^{t+1} = \begin{cases} \tilde{z}_i^t & i=0 \\ (\tilde{z}_i^t - \mu_i^t(\tilde{z}_{\lt i}^t)) \odot \exp(-\alpha_i^t(\tilde{z}_{\lt i}^t)) & i\gt0 \end{cases}$$.
            - $$\tilde{z}^t = \pi^t(z^t)$$.
- Prop.)
  - Causality of $$\mu^t$$. and $$\alpha^t$$.
    - $$\mu_i^t(\tilde{z}_{\lt i}^t)$$. 
    - $$\alpha_i^t(\tilde{z}_{\lt i}^t)$$.
  - Inverse
    - Input 
      - $$z^{t+1}$$.
    - Output
      - $$z^t$$.
        - where
          - $$z^t = (\pi^t)^{-1}(\tilde{z}^t)$$.
          - $$\tilde{z}_i^{t} = \begin{cases} z_i^{t+1} & i=0 \\ (z_i^{t+1}) \odot \exp(\alpha_i^t(\tilde{z}_{\lt i}^t)) + \mu_i^t(\tilde{z}_{\lt i}^t) & i\gt0 \end{cases}$$.
  - $$D$$. plays a role of balancing the difficulty of modeling each position in the sequence and the length of the entire sequence.
    - why?)
      - Consider that $$N\times D$$. is fixed.
        - e.g.) $$64\times 64$$.-sized pixel image
          - Putting the patch size of $$8\times 8 = D$$. pixels, we may get the $$8^2=64$$. length of sequence.
  - Jacobian Determinant Calculation
    - Since we consider the maximum log-likelihood, we consider the log determinant.
    - $$\pi^t$$. has 0 log determinant.
      - why?)
        - $$\pi^t$$. is volume preserving $$\Rightarrow$$. det is 1 $$\Rightarrow$$. log-det is 0.
    - The AR step has the log Jacobian Determinant of...
      - $$\displaystyle \log \left\vert \text{det}\left(\frac{\partial f^t(z^t)}{\partial z^t}\right) \right\vert = -\sum_{i=1}^{N-1} \sum_{j=1}^{D-1} \alpha_i^t(\tilde{z}_{\lt i}^t)$$.
        - Prop.) Sum of linear terms
    - Then, we may get the loss of...   
      $$\begin{aligned}
        \min_f &- \log \frac{1}{2}\Vert f(x) \Vert_2^2 - \log\left\vert\text{det}\frac{\partial f(x)}{\partial x}\right\vert \\
        &=  \frac{1}{2}\Vert z^T \Vert_2^2 + \log \left\vert \text{det}\left(\frac{\partial f^t(z^t)}{\partial z^t}\right) \right\vert + \sum_{i=1}^{N-1} \sum_{j=1}^{D-1} \alpha_i^t(\tilde{z}_{\lt i}^t)
      \end{aligned}$$.




<br>

### 2.3 Transformer Autoregressive Flows
#### Concept) Parallel Implementation
- Desc.)
  - Following structure enables the parallel implementation.
    - $$z^{t+1} = \{z_i^{t+1}\}_{i\in\{1,\ldots,N\}} \in\mathbb{R}^{N\times D}$$.
      - where
        - $$z_i^{t+1} = \begin{cases} \tilde{z}_i^t & i=0 \\ (\tilde{z}_i^t - \mu_i^t(\tilde{z}_{\lt i}^t)) \odot \exp(-\alpha_i^t(\tilde{z}_{\lt i}^t)) & i\gt0 \end{cases}$$.
        - $$\tilde{z}^t = \pi^t(z^t)$$.
  - Transformer backbone
- e.g.)
  - Refer to the [image case below](#tech-dimensionality-and-modality)

#### Tech.) Dimensionality and Modality
- Desc.)
  - Recall that we had two dimensions
    - $$N$$. : the sequence length
    - $$D$$. : the dimension of a block that forms the sequence
- Application)
  - Image
    - Initial Dimension)
      - $$C\times W\times H$$.
    - Setting)
      - Let $$S$$. be the patch length.
        - i.e.) $$S\times S$$. is the patch size
    - Conversion)
      - $$N = \displaystyle\frac{HW}{S^2}$$.
      - $$D = CS^2$$.
    - Advantage)
      - Readily applicable to the Vision Transformer (ViT)
      - Apply causality mask to transformation of a single AR pass $$f^t$$.
        - [Parallelism](#concept-parallel-implementation)!

#### Analysis) Train Stability
- Arg.)
  - Tarflow enables stable training.
- Why?)
  - Tarflow is a variant of a Residual Network (ResNet).
  - Two types of ResNets
    1. Hidden layers inside the causal transformer.
       - cf.) Recall the residual connection between the self-attention and the MLP layer
    2. Latents $$z_i^t$$.
       - why?)
         - Recall that the forward dynamics goes
           - $$z_i^{t+1} = \begin{cases} \tilde{z}_i^t & i=0 \\ (\tilde{z}_i^t - \mu_i^t(\tilde{z}_{\lt i}^t)) \odot \exp(-\alpha_i^t(\tilde{z}_{\lt i}^t)) & i\gt0 \end{cases}$$.
         - Here, the permutation $$\pi$$. is volume preserving and does not destroy the residual connection.
         - $$\mu_i^t(\tilde{z}_{\lt i}^t)$$. and $$\alpha_i^t(\tilde{z}_{\lt i}^t)$$. are the affine transformation coefficients of $$\tilde{z}_i^t$$.


<br>

### 2.4 Noise Augmented Training
- Desc.)
  - Recall that [real NVP added the Uniform noise to the data](../real_nvp/summary.md#41--procedure).
    - This is equivalent to the dequantizing the discrete pixel distribution to a continuous Uniform distribution.
    - Notation)
      - Let $$p_{\text{data}}$$. be the actual data distribution.
      - Then, we model a noise augmented distribution $$\displaystyle q(y) = \int_\epsilon p_{\text{data}}(y-\epsilon) p_\epsilon(\epsilon) \text{d}\epsilon$$.
        - cf.) Finite data case : $$\displaystyle q(y) = \frac{1}{\vert\mathcal{X}\vert} \sum_{x\in\mathcal{X}} p_\epsilon(y-x)$$.
      - We may get the likelihood as
        - $$\displaystyle\tilde{p}(\tilde{x}) = \int_{\epsilon\in[0,\text{bin}]^D}\;p_{\text{model}}(\tilde{x}+\epsilon)\text{d}\epsilon$$.
          - where $$\text{bin}$$.
  - Why needed?)
    - Input data is in the discrete value.
    - Thus, the model learns the transformation between the discrete input distribution and the prior distribution.
    - Hence, the generation by the inverse model shows poor quality.
      - To the model, continuous data space is the out of distribution!
    - Meanwhile, adding noise to the data can enrich the support of the training distribution.
  - Here, the authors add Gaussian noise instead of the Uniform noise.
    - i.e.) $$\mathcal{N}(\cdot\;;\; 0, \sigma^2 \mathbf{I})$$.
    - e.g.) For the image pixel with values $$[-1,1]$$., an optimal $$\sigma$$. of $$p_\epsilon(\cdot)$$. for sample quantity is around $$0.05$$..
    - Why?)
      - Gaussian has denser input distribution
        - probability density throughout $$(-\infty, \infty)$$.
- Problem)
  - Model trained on the noisy distribution $$q(y)$$. naturally generates outputs that mimic noisy training examples.
  - Thus, the samples are less visually appealing.
    - cf.) Real NVP injected relatively small noise. Thus, the generated sample is not that much noisy but the OOD problem is not properly handled.
- Sol.)
  - [Score Based Denoising](#25-score-based-denoising) below

<br>

### 2.5 Score Based Denoising
- Motivation)
  - Samples generated from the [noise augmented training](#24-noise-augmented-training) is too noisy.
- Idea)
  - Consider the joint distribution $$(x,y)$$. s.t.
    - $$x\sim p_{\text{data}}$$. : original data
    - $$y = x + \epsilon$$. for $$\epsilon\sim\mathcal{N}(0, \sigma^2\mathbf{I})$$. : noise injected data
      - with distribution $$y\sim q(y)$$.
  - Then, by definition, $$y$$. is marginally distributed as the noisy data distribution $$q$$.
    - where $$\displaystyle q(y) = \int_\epsilon p_{\text{data}}(y-\epsilon) p_\epsilon(\epsilon) \text{d}\epsilon$$.
  - By Tweedie's formula, we have
    - $$\mathbb{E}[x\mid y] = y + \sigma^2\nabla_y \log q(y)$$.
      - Desc.)
        - Recall that the Gaussian distribution has the highest density at its origin ($\epsilon=0$).
        - Thus, $\nabla_y \log q(y)$ (the score) acts as a compass pointing toward the density center
          - i.e.) the "origin" where the noise is zero.
        - Crucially, this "origin" in the noise space corresponds to the true data location $x$ in the data space ($x = y - \epsilon$).
        - Therefore, the score $\nabla_y \log q(y)$ shows the direction toward the clean data manifold $p_{data}$.
      - Magnitude & Velocity)
        - The variance $\sigma^2$ functions as the magnitude (or step size) required to traverse back to the origin.
        - Hence, the term $\sigma^2 \nabla_y \log q(y)$ can be interpreted as the correction velocity that shifts the noisy sample $y$ back into the clean sample $x$.
          - $$y + \sigma^2 \nabla_y \log q(y) \approx \mathbb{E}[x|y]$$.
  - Thus, if we know $$\nabla_y \log q(y)$$., the gradient of log likelihood $$\log q(y)$$., we may denoise into
    - $$\hat{x}:=\mathbb{E}[x\mid y]$$.
  - Now, suppose our model is well trained and achieved the likelihood of $$p_{\text{model}}(y)$$..
  - Then, we might replace $$q(y)$$. with $$p_{\text{model}}(y)$$..
- Procedure)
  - Draw latent from the prior.
    - $$z\sim p_0$$..
  - Generate a sample using the inverse transformation.
    - $$y := f^{-1}(z)$$.
  - Score Based Denoising
    - $$x := y + \sigma^2\nabla_y \log p_{\text{model}}(y)$$.


<br>

### 2.6 Guidance
- Method) Classifier Free Guidance (CFG)
  - Settings)
    - Class conditional predictions
      - $$\mu_i^t(\cdot; c), \alpha_i^t(\cdot; c)$$. 
        - where $$c$$. is the condition
    - Unconditional predictions
      - $$\mu_i^t(\cdot; \emptyset), \alpha_i^t(\cdot; \emptyset)$$. 
        - Implementation)
          - Randomly drop out the class label during training
  - Modify the reverse function into
    - $$\tilde{z}_i^t = \tilde{z}_i^{t+1} \odot \exp\left( \tilde{\alpha}_i^t(\tilde{z}_{\lt i}^t; c, w) \right) + \tilde{\mu}_i^t(\tilde{z}_{\lt i}^t; c, w)$$..
      - where
        - $$\tilde{\mu}_i^t(\tilde{z}_{\lt i}^t; c, w) = (1+w){\mu}_i^t(\tilde{z}_{\lt i}^t; c, w) - w {\mu}_i^t(\tilde{z}_{\lt i}^t; \emptyset)$$.
        - $$\tilde{\alpha}_i^t(\tilde{z}_{\lt i}^t; c, w) = (1+w){\alpha}_i^t(\tilde{z}_{\lt i}^t; c, w) - w {\alpha}_i^t(\tilde{z}_{\lt i}^t; \emptyset)$$.

<br><br>

## 3 Experiments