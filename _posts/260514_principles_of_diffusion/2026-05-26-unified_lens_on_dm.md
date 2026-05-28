---
layout: post
title: (DM Reconst.) Ch.6 A Unified and Systemic Lens on Diffusion Models
date: 2026-05-26 11:00:00
description: Lai et al.   
tags: 
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

### Concept) Four Prediction Types
- Types)
  - $\epsilon$-Prediction
  - $\mathbf{x}$-Prediction
  - Score-Prediction
  - $\mathbf{v}$-Prediction

<br>

<table>
  <thead>
    <tr>
      <th style="text-align: center;"></th>
      <th style="text-align: center;" colspan="2">Variational View</th>
      <th style="text-align: center;">Score-Based View</th>
      <th style="text-align: center;">Flow-Based View</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center;">Intractable original training objective</td>
      <td style="text-align: center;" colspan="2">$$\mathcal{J}_{\text{KL}}(\phi) := \mathbb{E}_{p_t(\mathbf{x}_t)} \left[ \mathcal{D}_{\text{KL}}(p(\mathbf{x}_{t-\Delta t}\mid\mathbf{x}_t)) \;\Vert\; p_\phi(\mathbf{x}_{t-\Delta t}\mid\mathbf{x}_t) \right]$$</td>
      <td style="text-align: center;">$$\mathcal{J}_{\text{SM}}(\phi) := \mathbb{E}_{p_t(\mathbf{x}_t)} \left[ \Vert s_\phi(\mathbf{x}_t, t) - \nabla_{\mathbf{x}}\log p_t(\mathbf{x}_t) \Vert_2^2 \right]$$</td>
      <td style="text-align: center;">$$\mathcal{J}_{\text{FM}}(\phi) := \mathbb{E}_{p_t(\mathbf{x}_t)} \left[ \Vert \mathbf{v}_\phi(\mathbf{x}_t, t) - \mathbf{v}_t(\mathbf{x}_t) \Vert_2^2 \right]$$</td>
    </tr>
    <tr>
      <td style="text-align: center;">Tractable objectives by conditioning on the data $$\mathbf{x}_0\sim p_{\text{data}}$$</td>
      <td style="text-align: center;" colspan="2">$$\mathcal{J}_{\text{KL}}(\phi) = \underbrace{\mathbb{E}_{\mathbf{x}_0} \mathbb{E}_{p_t(\mathbf{x}_t)} \left[ \mathcal{D}_{\text{KL}}(p(\mathbf{x}_{t-\Delta t}\mid\mathbf{x}_t, \mathbf{x}_0)) \;\Vert\; p_\phi(\mathbf{x}_{t-\Delta t}\mid\mathbf{x}_t) \right]}_{\mathcal{J}_{\text{KL}}(\phi) \text{ (Conditional KL)}} + C$$</td>
      <td style="text-align: center;">$$\mathcal{J}_{\text{SM}}(\phi) = \underbrace{\mathbb{E}_{\mathbf{x}_0} \mathbb{E}_{p_t(\mathbf{x}_t)} \left[ \Vert s_\phi(\mathbf{x}_t, t) - \nabla_{\mathbf{x}}\log p_t(\mathbf{x}_t \mid \mathbf{x}_0) \Vert_2^2 \right]}_{\mathcal{J}_{\text{DSM}}(\phi)} + C$$</td>
      <td style="text-align: center;">$$\mathcal{J}_{\text{FM}}(\phi) = \underbrace{\mathbb{E}_{\mathbf{x}_0} \mathbb{E}_{p_t(\mathbf{x}_t)} \left[ \Vert \mathbf{v}_\phi(\mathbf{x}_t, t) - \mathbf{v}_t(\mathbf{x}_t \mid \mathbf{x}_0) \Vert_2^2 \right]}_{\mathcal{J}_{\text{CFM}}(\phi)} + C$$</td>
    </tr>
    <tr>
      <td style="text-align: center;">Common forward perturbation kernel</td>
      <td style="text-align: center;" colspan="4">$$p_t(\mathbf{x}_t\mid\mathbf{x}_0) = \mathcal{N} \bigg( \mathbf{x}_t;\; \alpha_t\mathbf{x}_0, \sigma_t^2\mathbf{I} \bigg)$$</td>
    </tr>
    <tr>
      <td style="text-align: center;">Parameterization Target</td>
      <td style="text-align: center;">$\epsilon$-Prediction $$\epsilon_\phi(\mathbf{x}_t, t) \approx \mathbb{E}[\epsilon\mid\mathbf{x}_t] = \epsilon^*(\mathbf{x}_t, t)$$</td>
      <td style="text-align: center;">$\mathbf{x}$-Prediction $$\mathbf{x}_\phi(\mathbf{x}_t, t) \approx \mathbb{E}[\mathbf{x}_0\mid\mathbf{x}_t] = \mathbf{x}^*(\mathbf{x}_t, t)$$</td>
      <td style="text-align: center;">Score-Prediction $$\begin{aligned}
        s_\phi(\mathbf{x}_t, t) &\approx \nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t) \\
        &= \mathbb{E}[\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t\mid\mathbf{x}_0)\mid\mathbf{x}_t] \\
        &= s^*(\mathbf{x}_t, t)
      \end{aligned}$$</td>
      <td style="text-align: center;">$\mathbf{v}$-Prediction $$\mathbf{v}_\phi(\mathbf{x}_t, t) \approx \mathbb{E}\left[\frac{\text{d}\mathbf{x}_t}{\text{d}t}\mid\mathbf{x}_t\right] = \mathbf{v}^*(\mathbf{x}_t, t)$$</td>
    </tr>
    <tr>
      <td style="text-align: center;">Training objective</td>
      <td style="text-align: center;">$$\mathcal{L}_{\text{noise}}(\phi) := \mathbb{E}_t\left[ \omega(t)\;\mathbb{E}_{\mathbf{x}_0, \epsilon}\Vert \epsilon_\phi(\mathbf{x}_t, t) - \epsilon \Vert_2^2 \right]$$</td>
      <td style="text-align: center;">$$\mathcal{L}_{\text{clean}}(\phi) := \mathbb{E}_t\left[ \omega(t)\;\mathbb{E}_{\mathbf{x}_0, \epsilon}\Vert \mathbf{x}_\phi(\mathbf{x}_t, t) - \mathbf{x}_0 \Vert_2^2 \right]$$</td>
      <td style="text-align: center;">$$\mathcal{L}_{\text{score}}(\phi) := \mathbb{E}_t\left[ \omega(t)\;\mathbb{E}_{\mathbf{x}_0, \epsilon}\Vert s_\phi(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t\mid\mathbf{x}_0) \Vert_2^2 \right]$$</td>
      <td style="text-align: center;">$$\mathcal{L}_{\text{velocity}}(\phi) := \mathbb{E}_t\left[ \omega(t)\;\mathbb{E}_{\mathbf{x}_0, \epsilon}\Vert \mathbf{v}_\phi(\mathbf{x}_t, t) - \mathbf{v}_t(\mathbf{x}_t\mid\mathbf{x}_0,\epsilon) \Vert_2^2 \right]$$</td>
    </tr>
    <tr>
      <td style="text-align: center;">Conditional Parameterization Target $$A_t\mathbf{x}_0+B_t\epsilon$$</td>
      <td style="text-align: center;">$\epsilon$</td>
      <td style="text-align: center;">$\mathbf{x}_0$</td>
      <td style="text-align: center;">$$-\displaystyle\frac{1}{\sigma_t}\epsilon$$</td>
      <td style="text-align: center;">$$\alpha_t'\mathbf{x}_0 + \sigma_t'\epsilon$$</td>
    </tr>
  </tbody>
</table>

{% include figure.liquid path="assets/img/blog/260526_unified_lens_on_dm/001.png" class="img-fluid rounded z-depth-1" zoomable=true %}
{% include figure.liquid path="assets/img/blog/260526_unified_lens_on_dm/002.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<br><br>

#### Prop.) Equivalence of the Four Parameterizations

{% include figure.liquid path="assets/img/blog/260526_unified_lens_on_dm/003.png" class="img-fluid rounded z-depth-1" zoomable=true %}
{% include figure.liquid path="assets/img/blog/260526_unified_lens_on_dm/004.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<br><br>

#### Prop.) PF-ODE in each Parameterization

{% include figure.liquid path="assets/img/blog/260526_unified_lens_on_dm/005.png" class="img-fluid rounded z-depth-1" zoomable=true %}

- cf.) Although the [four parameterizations](#concept-four-prediction-types) are equivalent in principle, they differ in practice.
  - Why?)
    - Each parametrization changes has its own 
      - the stiffness of the vector field
      - the behavior of discretization error
      - the ease of optimization
  - For the fast sampling with advanced ODE solvers, $\epsilon$- or $\mathbf{x}$-prediction is preferred.
    - Why?) They align well with the solver inputs and reduce error accumulation.
  - For the limited number of function evaluations, $\mathbf{x}$- or $\mathbf{v}$-prediction is preferred.
    - Why?) They often yield smoother objectives and better step to step consistency.

<br><br>

#### Prop.) All Affine Flows are Equivalent
- Settings)
  - $$\mathbf{x}_t^{\text{FM}} = (1-t)\mathbf{x}_0 + t\epsilon = \mathbf{x}_0 + t\underbrace{(\epsilon - \mathbf{x}_0)}_{\mathbf{v}}$$.
    - i.e.) A canonical interpolation used in [CFM](/blog/2025/conditional_flow_matching/) and [RF](/blog/2025/rectified_flow/).

{% include figure.liquid path="assets/img/blog/260526_unified_lens_on_dm/006.png" class="img-fluid rounded z-depth-1" zoomable=true %}

<br><br>

#### Prop.) Conversions between the Parameterizations
- Settings)
  - $$\mathbf{x}_t = \alpha_t\mathbf{x}_0 + \sigma_t\epsilon$$ s.t.
    - $$\sigma_t\gt0$$.
    - $$(\alpha_t'\sigma_t - \alpha_t\sigma_t')\ne0$$.
  - Oracle targets are given by
    - $$\epsilon^*(\mathbf{x}_t, t) = \mathbb{E}[\epsilon\mid\mathbf{x}_t]$$.
    - $$\mathbf{x}_0^*(\mathbf{x}_t, t) = \mathbb{E}[\mathbf{x}_0\mid\mathbf{x}_t]$$.
    - $$\mathbf{v}^*(\mathbf{x}_t, t) = \mathbb{E}[\alpha_t'\mathbf{x}_0 + \sigma_t' \epsilon\mid\mathbf{x}_t]$$.
- Conversions
  - Score - Noise 
    - Parameterization
      - $$s_\phi \equiv -\frac{1}{\sigma_t}\epsilon_\phi$$.
    - Loss
      - $$\left\Vert s_\phi - \nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t) \right\Vert_2^2 = \frac{1}{\sigma_t^2} \left\Vert \epsilon_{\phi} - \epsilon^* \right\Vert_2^2$$.
  - Score - $\mathbf{x}$ 
    - Parameterization
      - $$s_\phi \equiv \frac{\alpha_t}{\sigma_t^2}\left(\mathbf{x}_\phi - \frac{\mathbf{x}_t}{\alpha_t}\right)$$.
    - Loss
      - $$\left\Vert s_\phi - \nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t) \right\Vert_2^2 = \frac{\alpha_t^2}{\sigma_t^4} \left\Vert \mathbf{x}_\phi - \mathbf{x}_0^* \right\Vert_2^2$$.
  - Score - $\mathbf{v}$
    - Parameterization
      - $$s_\phi = \frac{\alpha_t}{\sigma_t(\alpha_t'\sigma_t - \alpha_t\sigma_t')}\mathbf{v}_\phi - \frac{\alpha_t'}{\sigma_t(\alpha_t'\sigma_t - \alpha_t\sigma_t')}\mathbf{x}_t$$.
    - Loss
      - $$\left\Vert s_\phi - \nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t) \right\Vert_2^2 = \left(\frac{\alpha_t}{\sigma_t(\alpha_t'\sigma_t - \alpha_t\sigma_t')}\right) \left\Vert \mathbf{v}_\phi - \mathbf{v}^*\right\Vert_2^2$$.

<br><br>

#### Prop.) Unified perspective connecting variational, SDE, and ODE formulations through the continuity equation, where all p_t(x) evolve under a shared dynamic.

{% include figure.liquid path="assets/img/blog/260526_unified_lens_on_dm/007.png" class="img-fluid rounded z-depth-1" zoomable=true %}