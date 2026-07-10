---
layout: post
title: (Presentation PDF) Generative Modeling via Drifting
date: 2026-06-05 11:00:00
description: Deng et al.
tags: 
categories: drift_model
pdf: assets/pdf/posts/slides_drift_model.pdf
pretty_table: true
---

[Deng et al. 2026](https://arxiv.org/abs/2602.04770)

<br>

---

### Hozy Summary


---

<br><br>

### Concept) Drifting Model for Generation
- Setting)
  - $$f : \mathbb{R}^C\rightarrow\mathbb{R}^D$$ : a neural network with
    - input : $$\epsilon\sim p_\epsilon$$
    - output : $$\mathbf{x} = f(\epsilon) \in\mathbb{R}^D$$ s.t.
      - $$\mathbf{x} = f(\epsilon)\sim q$$.
    - Desc.)
      - Pushforward Relation between $$p$$ and $$q$$.
        - $$q = f_{\#}p_\epsilon$$.
      - We want to find $$f$$ s.t. $$f_{\#}p_\epsilon\approx p_{\text{data}}$$
  - Drift $$\Delta\mathbf{x}$$
    - Def.)
      - Assume an iterative training process on $$f$$
        - where 
          - $$\{f_i\}$$ : the set of training process
          - $$\{q_i\}$$ for $$q_i = [f_i]_{\#}p_\epsilon$$
      - Then, the **drift** is the residual of $$\mathbf{x}_i = f_{i}(\epsilon)$$
        - i.e.) $$\Delta\mathbf{x}_i := \mathbf{x}_{i+1}-\mathbf{x}_i = f_{i+1}(\epsilon)-f_{i}(\epsilon)$$
  - Drifting Field $$\mathbf{V}_{p,q}(\cdot)$$
    - Def.)
      - A function that computes the drift $$\Delta\mathbf{x}$$ given $$\mathbf{x} = f(\epsilon)$$ given by
        - $$\mathbf{V}_{p,q} : \mathbb{R}^d\rightarrow\mathbb{R}^d$$ s.t.
          - $$\mathbf{V}_{p,q}(\mathbf{x}_{i}) = \Delta \mathbf{x}_{i} = \mathbf{x}_{i+1} - \mathbf{x}_{i}$$.
          - Or, $$\mathbf{x}_{i+1} = \mathbf{x}_{i} + \mathbf{V}_{p,q}(\mathbf{x}_{i})$$
        - Here, $$p$$ is the target distribution, e.g. $$p=p_{\text{data}}$$
    - Instantiation)
      - [Mean Shift Method of Attracting and Repulsing](#tech-drift-field--mean-shift-method-of-attracting-and-repulsing)
- Goal)
  - Train $$f$$ until all $$\mathbf{x}$$ stop drifting
    - i.e.) $$\mathbf{V}=\mathbf{0}$$
- Method)
  - Anti-Symmetric Drifting Field
    - Def.)
      - $$\mathbf{V}_{p,q}(\mathbf{x}) = -\mathbf{V}_{q,p}(\mathbf{x}),\quad\forall\mathbf{x}$$.
    - Prop.)
      - Then we have
        - $$q=p\Rightarrow \mathbf{V}_{p,q}(\mathbf{x}) = \mathbf{0},\quad\forall\mathbf{x}$$.
- Training Objective)
  - Data space prediction
    - $$\mathcal{L} = \mathbb{E}_{\epsilon}\left[ \left\Vert \underbrace{f_\theta(\epsilon)}_{\text{pred}} - \underbrace{\text{stopgrad}\left( f_\theta(\epsilon) + \mathbf{V}_{p, q_\theta}(f_\theta(\epsilon)) \right)}_{\text{frozen target}} \right\Vert^2 \right]$$.
      - Recall that $$\mathbf{V}=\mathbf{0}$$ was the terminating condition.
      - $$\mathbf{V}$$ is not the propagation target.
        - Why?) $$\mathbf{V}$$ depends on the distribution $$q_\theta$$ which is not trivial.
        - Instead, indirectly optimize via $$\mathbf{x}=f_\theta(\epsilon)$$
  - Feature space prediction
    - $$\mathcal{L} = \mathbb{E}\left[ \left\Vert \phi(\mathbf{x}) - \text{stopgrad}\left( \phi(\mathbf{x}) + \mathbf{V}(\phi(\mathbf{x})) \right) \right\Vert^2 \right]$$.
      - where
        - $$\mathbf{x} = f_\theta(\epsilon)$$.
        - $$\mathbf{V}$$ is defined on $$\{\phi(\mathbf{x})\mid\mathbf{x} = f_\theta(\epsilon), \epsilon~\sim\mathcal{N}(\mathbf{0,I})\}$$.
- Implementation Details)
  - CFG

<br>

#### Tech.) Drift Field : Mean Shift Method of Attracting and Repulsing
- Model)   
  $$\begin{aligned}
    \mathbf{V}_{p,q}(\mathbf{x}) 
    &= \mathbb{E}_{y^+\sim p} \mathbb{E}_{y^-\sim q} \left[ \mathcal{K}(x,y^+,y^-) \right]  \\
    &= \mathbf{V}_{p}^+(\mathbf{x}) - \mathbf{V}_{q}^-(\mathbf{x})  \\
    &= \frac{1}{Z_p}\mathbb{E}_{p}\left[ k(\mathbf{x}, \mathbf{y}^+) (\mathbf{y}^+ - \mathbf{x}) \right] - \frac{1}{Z_q}\mathbb{E}_{q}\left[ k(\mathbf{x}, \mathbf{y}^-) (\mathbf{y}^- - \mathbf{x}) \right] \\
    &= \frac{1}{Z_pZ_q}\mathbb{E}_{p,q}\left[ k(\mathbf{x}, \mathbf{y}^+) k(\mathbf{x}, \mathbf{y}^-) (\mathbf{y}^+ - \mathbf{y}^-) \right] \\
  \end{aligned}$$.
    - where
      - $$\displaystyle k(\mathbf{x}, \mathbf{y}) = \exp\left(-\frac{1}{\tau}\Vert\mathbf{x}-\mathbf{y}\Vert\right)$$ : a kernel function
        - for
          - $$\tau$$ : the temperature
          - $$\Vert\cdot\Vert$$ : $$\ell_2$$ distance
      - $$\mathbf{y}^+\sim p_{\text{data}}$$.
      - $$\mathbf{y}^-=f(\epsilon)\sim q$$ where $$\epsilon~\sim\mathcal{N}(\mathbf{0,I})$$.
      - $$Z_p = \mathbb{E}_p\left[k(\mathbf{x}, \mathbf{y}^+)\right]$$.
      - $$Z_q = \mathbb{E}_q\left[k(\mathbf{x}, \mathbf{y}^-)\right]$$.
- Desc.)
  - $$p_{\text{data}}$$ attracts the field.
  - $$q$$ repulses the field.



<br><br>

#### Algorithm)

{% include figure.liquid path="assets/img/blog/260605_drift_model/001.png" class="img-fluid rounded z-depth-1" zoomable=true %}

{% include figure.liquid path="assets/img/blog/260605_drift_model/002.png" class="img-fluid rounded z-depth-1" zoomable=true %}