# MIT PhD SOP — Research Proposal Notes

## 0. Overview: 두 연구 제안의 핵심 차이

### Antonio Torralba — Information Routing Within a Generative Model

**Core Question:** How does temporal information from video history propagate through a joint-attention video diffusion model?

- 하나의 생성 모델 내부에서 정보가 어떤 경로로 전달되고 활용되는지 분석한다.
- 특히 과거 video history의 motion, object state, identity 정보가 text-token representation에 유입되는지 조사한다.
- Text-mediated pathway와 direct video-to-video pathway를 구분한다.
- Causal intervention을 통해 이러한 정보 전달 경로가 실제 generation에 기여하는지 확인한다.
- 발견한 구조를 활용해 motion control과 long-term temporal consistency를 개선한다.

**Keywords:** Mechanistic Interpretability, Information Routing, Vision-Language Binding, Text Padding Tokens, Implicit Memory, Joint Attention, Causal Intervention, Activation Patching, AR Video Diffusion

### Phillip Isola — Shared Temporal Geometry Across Generative Models

**Core Question:** Do autoregressive language models and autoregressive video diffusion models develop shared representation geometry for temporal relationships?

- 서로 다른 modality와 generative objective를 가진 모델들 사이의 공통 representation structure를 찾는다.
- Language model의 sequential reasoning과 video model의 temporal dynamics representation이 유사한 geometry를 갖는지 검증한다.
- 단순 semantic similarity와 진정한 temporal alignment를 구분한다.
- 공통 구조가 발견된다면 이를 통해 유용한 sequential priors를 video generation에 전달할 가능성을 탐구한다.

**Keywords:** Platonic Representation Hypothesis, Shared Geometry, Rosetta Stone, Cross-Modal Alignment, Temporal Representation, Representation Geometry, Wasserstein Procrustes, Model Stitching, Sequential Priors

### 두 방향의 관계

| | Torralba | Isola |
|---|---|---|
| 분석 범위 | Within-model | Cross-model |
| 핵심 대상 | Attention / Token / Information Pathway | Representation Space / Geometry |
| 연구 질문 | 정보가 어떻게 전달되는가? | 공통적인 정보 구조가 존재하는가? |
| 검증 방식 | Causal intervention | Geometric alignment + functional validation |
| 최종 목표 | Selective control of temporal information | Transfer of useful temporal structure |
| 공통 철학 | Understanding → Intervention → Better Generation | Understanding → Transfer → Better Generation |

---

# 1. Antonio Torralba — Temporal Information Routing in AR Video Diffusion

## 1.1. 출발점: Vision-Language Binding in In-Context Image Generation

### Reference

*Vision-Language Binding in In-Context Image Generation* (2026)

- Authors: Chris Ge, Rohit Gandikota, Antonio Torralba, Tamar Rott Shaham
- Paper: https://arxiv.org/abs/2605.24624

### Background: FLUX.2의 Joint Attention

논문은 FLUX.2의 in-context image editing을 분석한다.

입력에는 다음과 같은 정보가 존재한다.

1. Reference Image: 편집 대상 이미지
2. Text Tokens: 편집 지시문
3. Noisy Image Tokens: 생성할 이미지의 noisy latent

이들은 joint attention을 통해 정보를 교환할 수 있다.

- Double-stream blocks: Text와 image가 별도의 Q/K/V projection 및 MLP를 사용하지만, attention 계산에서는 서로 정보를 교환할 수 있다.
- Single-stream blocks: 같은 Transformer block의 공유 projection을 통해 multimodal tokens를 처리한다.
- 중요한 것은 text와 image 사이에 정보가 오갈 수 있는 attention connectivity다.

### Main Finding: 두 가지 정보 전달 경로

**Finding A — Reference Image → Text Tokens → Generated Image**

- Reference image의 color, style, scene attributes 등이 text-token hidden states에 전달된다.
- 특히 의미 있는 문장 토큰뿐 아니라 padding token에서도 reference-image attributes가 발견된다.
- 즉, text tokens는 언어적 조건만 처리하는 것이 아니라 visual information을 전달하는 중간 경로 역할도 수행한다.

**Finding B — Reference Image → Generated Image**

- Identity와 같이 pixel-level로 정밀한 정보는 주로 direct image-to-image pathway를 통해 전달된다.
- Text tokens에 일부 정보가 존재하더라도, 그것이 모든 visual attributes를 전달하는 주요 경로라는 의미는 아니다.

핵심은 **서로 다른 종류의 visual information이 서로 다른 경로로 전달될 수 있다는 것**이다.

### 논문에서 사용한 분석 방법

#### A. T2I Lens

- Reference-conditioned editing run에서 text-token activations를 추출한다.
- 이 activations를 별도의 reference-free text-to-image generation run에 주입한다.
- 생성된 이미지를 통해 text representation에 어떤 reference information이 담겼는지 확인한다.

**해석상 주의:** T2I Lens는 정보가 text activation에서 복원 가능하다는 증거다. 원래 생성 과정에서 해당 경로가 실제로 사용되었는지까지 단독으로 입증하지는 않는다.

#### B. Attention Knockout

- Reference → Text attention을 차단한다.
- Reference → Generated Image attention을 차단한다.
- 각 차단이 결과에 미치는 영향을 비교한다.

목표는 각 경로의 **causal necessity**를 확인하는 것이다.

#### C. Activation Patching

- 동일한 instruction을 사용하되 서로 다른 reference image를 사용하는 두 generation run을 준비한다.
- 한 run의 text-token hidden states를 다른 run에 이식한다.
- 어떤 visual attributes가 함께 전달되는지 확인한다.

특히 text padding token과 instruction-content token을 나누어 분석한다.

**Takeaway:** Information decodability, causal necessity, and functional transfer를 구분해서 검증해야 한다.

---

## 1.2. 나의 연구 질문: Image Reference에서 Video History로

### 핵심적인 발상

기존 논문은 reference image의 visual attributes가 text-token representation으로 이동할 수 있음을 보여준다.

그렇다면 reference image 대신 **past video frames**를 제공하는 AR video diffusion에서는 어떨까?

AR video generation에서는 이전 frame들이 단순한 appearance뿐 아니라 다음과 같은 temporal information을 포함한다.

- Motion direction
- Velocity
- Acceleration
- Object position and state
- Object permanence / identity
- Past interactions
- Temporal dependencies

이 정보들은 자연어 prompt로 정확히 지정하기 어렵다.

예를 들어:

- Prompt: "A ball is rolling across the floor."
- History: 실제 공의 위치, 진행 방향, 속도, 이전의 충돌과 반사

현재 frame의 정확한 다음 상태를 예측하기 위해서는 prompt보다 visual history가 훨씬 세밀한 조건을 제공할 수 있다.

### Research Question

**Can text-token representations serve as an implicit memory channel for temporal dynamics in autoregressive video diffusion?**

더 자세히는:

1. 과거 video history의 정보가 text tokens에 유입되는가?
2. 유입된다면 어떤 temporal attributes가 표현되는가?
3. 이 정보는 실제로 future-video generation에 활용되는가?
4. Direct history-to-video pathway와 역할을 어떻게 분담하는가?
5. 이러한 경로를 조작하면 motion과 memory retention을 개선할 수 있는가?

---

## 1.3. Proposed Architecture: Joint-Attention Video Diffusion + AR Generation

### Candidate Backbone

**HunyuanVideo 계열의 joint-attention video diffusion model**

- HunyuanVideo: https://arxiv.org/abs/2412.03603
- HunyuanVideo 1.5: https://arxiv.org/abs/2511.18870

기본 구상은 기존 video diffusion architecture를 history-conditioned autoregressive generation으로 확장하는 것이다.

### 세 가지 Token Groups

1. **History Tokens (H)**
   - 이전에 생성된 clean video frames
   - 과거의 motion, appearance, object state를 담고 있는 visual context

2. **Text Tokens (T)**
   - 사용자의 text prompt
   - 잠재적으로 history-derived information을 담는 중간 representation

3. **Current Noisy Video Tokens (V)**
   - 현재 생성해야 할 future chunk
   - Noise level에 따라 denoising되는 latent representation

### 가정하는 정보 경로

```text
                 Past Video History (H)
                    /           \
                   /             \
                  v               v
           Text Tokens (T) ----> Noisy Video (V)
                  ^                   |
                  |                   |
               Prompt             Denoising
```

관심 있는 경로는 두 가지다.

**Path A — Text-mediated memory**

`History → Text → Current Video`

**Path B — Direct visual memory**

`History → Current Video`

중요한 점은 Path A의 존재 자체가 가설이라는 것이다. Joint attention이 가능하다는 것만으로 실제로 이 경로가 형성된다고 단정할 수 없다.

### Architecture Constraints

실험 전에 확인해야 하는 사항:

- Text queries가 history K/V에 attend할 수 있는가?
- History attention과 AR causal masking이 양립하는가?
- Text padding tokens가 실제로 존재하고 attention에서 masked되지 않는가?
- Text/image projection이 분리되어 있는가?
- History tokens가 현재 noisy video tokens와 어떻게 결합되는가?
- Temporal representation이 KV cache를 통해 chunk 간 유지되는가?

**주의:** HunyuanVideo 1.5는 원래 HunyuanVideo와 block 구성이 다르다. 특정 double/single-stream 구조를 전제로 실험을 설계해서는 안 된다.

또한 한 denoising step에서 text activations에 history information이 담기는 것과, 그 정보가 여러 AR chunk에 걸쳐 장기적으로 유지되는 것은 다른 문제다.

---

## 1.4. Experimental Plan

### Experiment 1 — Controlled History Pairs

**Purpose:** Temporal information과 appearance information을 분리한다.

동일한 prompt를 사용하면서 다른 video history를 제공한다.

예시:

- History A: 공이 왼쪽에서 오른쪽으로 이동
- History B: 공이 오른쪽에서 왼쪽으로 이동
- History C: 공이 정지
- History D: 공이 충돌 후 반사

가능한 한 객체 종류, 배경, 색상 등 semantic content는 동일하게 유지한다.

관찰 대상:

- Text-token activations
- History-token activations
- Current video-token activations
- Attention patterns
- Different denoising timesteps
- Different Transformer layers

**Question:** History의 temporal properties가 text representations에서도 구분되는가?

### Experiment 2 — Video Lens / Activation Decoding

원래 논문의 T2I Lens를 video setting으로 확장한다.

1. History A를 사용한 AR generation run에서 text activations를 추출한다.
2. History가 없거나 다른 history를 가진 run에 text activations를 주입한다.
3. 생성된 future video에 History A의 temporal properties가 나타나는지 관찰한다.

예상 관찰:

- Motion direction이 전달되는가?
- Velocity가 전달되는가?
- Object state가 전달되는가?
- Appearance만 전달되는가?

**Critical distinction:** 이 실험은 해당 정보가 text states에 인코딩되었다는 근거를 제공할 수 있지만, 원래 run에서 그 경로가 필수적이었다는 증거는 아니다.

### Experiment 3 — Attention Knockout

**Purpose:** Text-mediated pathway와 direct pathway의 causal contribution을 구분한다.

주요 intervention:

1. History → Text attention 차단
2. History → Current Video attention 차단
3. Text → Current Video attention 차단 또는 선택적 변경
4. 각 개입의 조합 비교

측정 대상:

- Motion continuation accuracy
- Direction consistency
- Object identity preservation
- Long-term temporal consistency
- Video generation quality

예상 가능한 시나리오:

- History → Text 차단 시 motion control이 악화된다면 text-mediated pathway의 인과적 기여 가능성
- History → Video 차단 시 정밀한 trajectory가 붕괴한다면 direct pathway의 중요성
- 두 경로가 서로 다른 temporal properties를 담당한다면 기능적 분업 가능성

단, attention knockout은 representation distribution을 교란할 수 있다. 결과를 곧바로 특정 정보의 독점적인 전달 경로로 해석하지 않도록 통제 실험이 필요하다.

### Experiment 4 — Cross-Run Activation Patching

서로 다른 motion histories를 이용한 두 generation run을 비교한다.

```text
Run A: Left-to-right motion
Run B: Right-to-left motion

Patch text activations: A → B
```

질문:

- B의 motion이 A의 방향으로 바뀌는가?
- Text activation만 바꿔도 temporal attribute를 이전할 수 있는가?
- Appearance는 유지하면서 motion만 조절할 수 있는가?
- Text padding tokens와 content tokens는 다른 역할을 하는가?

성공한다면 특정 representation pathway가 temporal information을 전달한다는 더 강한 인과적 증거가 될 수 있다.

---

## 1.5. Potential Outcomes

### H1. Text-mediated temporal pathway가 존재한다

- Text tokens에 일부 motion/state information이 담긴다.
- 특정 attention route나 activation patch가 future generation에 영향을 준다.
- 선택적 routing 또는 activation guidance를 통한 motion control 가능성을 탐구할 수 있다.

### H2. Direct history-to-video pathway가 지배적이다

- Text tokens는 appearance 또는 high-level semantics만 전달한다.
- Fine-grained motion은 direct visual pathway에서 주로 처리된다.
- 이 역시 중요한 negative finding이며, long-term memory에서 direct attention의 역할을 분석할 수 있다.

### H3. 두 pathway가 서로 다른 역할을 한다

- Text-mediated: high-level action/state, semantic temporal information
- Direct visual: object trajectory, fine-grained motion, pixel-level identity

이러한 역할 분담이 관찰된다면 속성별 intervention 전략을 개발할 수 있다.

---

## 1.6. What Is Not Yet Established?

- FLUX.2에서 발견된 text padding token의 역할이 video diffusion에도 재현된다는 보장은 없다.
- Text tokens가 history-derived motion information을 담는다는 증거는 아직 없다.
- 해당 information이 generation에 인과적으로 기여한다는 증거도 아직 없다.
- 경로를 조작한다고 반드시 video quality가 개선되는 것은 아니다.
- AR 변환과 attention mask 설계 자체가 큰 기술적 과제일 수 있다.

**연구 제안의 강점은 특정 결과를 예측하는 것이 아니라, 이 가능성을 검증할 수 있는 명확한 causal experiments를 설계한다는 점이다.**

---

## 1.7. Interview Pitch (약 45초)

> I was particularly interested in Professor Torralba's work on vision-language binding, which showed that reference-image attributes can be routed through text-token representations, while fine-grained identity information relies more on direct image-to-image attention. This made me wonder whether a similar distinction exists in autoregressive video diffusion, where past frames contain temporal information that is difficult to describe precisely in text. I would like to investigate whether text tokens can act as an implicit channel for motion and object-state information, using attention knockout and activation patching to distinguish text-mediated and direct visual pathways. Ultimately, I hope to use these insights to improve motion control and long-term temporal consistency.

---

# 2. Phillip Isola — Shared Temporal Representation Geometry Across Generative Models

## 2.1. 출발점 A: The Platonic Representation Hypothesis

### Reference

*The Platonic Representation Hypothesis* (ICML 2024)

- Authors: Minyoung Huh, Brian Cheung, Tongzhou Wang, Phillip Isola
- Paper: https://arxiv.org/abs/2405.07987

### Core Idea

서로 다른 modality와 objective로 학습된 모델들이 같은 현실 세계의 구조를 학습한다면, representation space에서도 유사한 관계 구조가 나타날 수 있다.

중요한 점:

**서로 다른 모델의 embedding vector가 동일하다는 뜻이 아니다.**

예를 들어 동일한 사건에 대한 표현이 서로 다른 차원과 좌표계를 사용하더라도, 각 모델의 representation space에서 사건들 사이의 상대적 관계가 비슷할 수 있다는 것이다.

Conceptually:

```text
Model A representation space
  Event 1 ── Event 2
      \       /
       Event 3

Model B representation space
  Event 1' ── Event 2'
      \        /
       Event 3'
```

좌표는 달라도 사건 사이의 geometric relationships는 유사할 수 있다.

주요 개념:

- Emergent representation alignment
- Shared relational structure
- Modality-independent features
- Representational convergence
- Mutual k-NN / representation similarity

---

## 2.2. 출발점 B: Shared Geometry as a Rosetta Stone

### Reference

*Shared Geometry as a Rosetta Stone: Cross-Modal Alignment Without Paired Data* (2026)

- Phillip Isola 공동저자
- Paper: https://arxiv.org/abs/2610.09411

### Core Idea

독립적으로 학습된 서로 다른 modality의 embedding space 사이에 공통된 geometry가 존재한다면, 이를 활용해 modality 간 alignment를 구축할 수 있다.

특히 paired training examples 없이도 공간의 구조를 이용해 정렬을 추정할 수 있다는 점이 중요하다.

Keywords:

- Shared geometry
- Unpaired cross-modal alignment
- Wasserstein Procrustes
- Orthogonal transformation
- Distribution / geometric correspondence

### 나에게 중요한 부분

Platonic Representation Hypothesis가 공통 구조의 존재를 묻는다면, Shared Geometry는 그 구조를 실제 alignment에 이용할 수 있는지를 탐구한다.

즉:

**Finding Shared Structure → Leveraging Shared Structure**

이것이 내 연구 아이디어의 출발점이다.

다만 geometry가 유사하다는 것과 두 모델의 계산 능력이 서로 호환된다는 것은 다른 문제다.

---

## 2.3. Related Work: Dynamic Reflections

### Reference

*Dynamic Reflections: Probing Video Representations with Text Alignment* (ICLR 2026)

- Tyler Zhu et al.
- Paper: https://arxiv.org/abs/2511.02767
- Tyler Zhu의 Google DeepMind 공동연구이며 Phillip Isola 또는 Olga Russakovsky가 공동저자인 논문은 아니다.

### Established Findings

- Video encoder와 text encoder 사이에도 emergent representation alignment가 나타난다.
- 더 많은 video frames와 다양한 captions를 제공하면 test-time alignment가 개선된다.
- Frame/caption 개수에 따른 alignment 변화를 saturation-based scaling law로 모델링했다.
- Semantic tasks뿐 아니라 일부 non-semantic video understanding tasks와도 alignment가 상관관계를 보였다.
- Temporal ordering에 대한 별도 분석도 수행했다.

### Limitations Relevant to My Proposal

1. 연구의 중심은 video-text encoder alignment다.
2. 주요 실험은 global video representations의 structural similarity에 집중한다.
3. Temporal order sensitivity는 조사했지만, velocity, trajectory, long-range causal states의 공통 representation geometry를 체계적으로 규명하지는 않았다.
4. 서로 다른 autoregressive generative paradigms 사이의 shared temporal structure는 분석 대상이 아니다.
5. Alignment를 이용해 autoregressive video generation의 temporal behavior를 개선하는 연구도 아니다.

**중요한 주의:** Dynamic Reflections가 temporal dynamics를 전혀 다루지 않았다고 설명하면 부정확하다. 정확한 한계는 fine-grained temporal geometry와 그 기능적 활용을 충분히 규명하지 않았다는 것이다.

---

## 2.4. My Research Question

### Starting Observation

AR language model은 이전 tokens를 기반으로 다음 token을 예측한다.

AR video diffusion은 이전 video chunks를 기반으로 새로운 video chunk를 생성하며, 각 chunk 내부에서는 diffusion denoising을 수행한다.

두 모델 모두 과거의 정보를 기반으로 미래를 생성하지만, 다루는 정보와 학습 목표가 다르다.

**AR Language Models**

- Token-level sequential prediction
- Event ordering
- Narrative progression
- Logical dependencies
- Long-context relationships
- High-level temporal and causal reasoning

**AR Video Diffusion Models**

- Chunk-wise autoregressive prediction
- Temporal continuity across frames
- Object motion and trajectory
- Object permanence
- History-conditioned physical state
- Denoising-based visual synthesis

### Central Hypothesis

**Could autoregressive language models and autoregressive video diffusion models develop a shared representation geometry for temporal relationships?**

핵심은 두 모델이 같은 activation을 갖는지 찾는 것이 아니다.

서로 다른 representation space에서도 temporal states와 events 사이의 상대적인 관계가 비슷하게 나타나는지 확인하는 것이다.

### 중요한 구분

"둘 다 autoregressive하므로 shared geometry가 존재할 것이다"는 성립하지 않는다.

가능한 설명은 여러 가지다.

1. 동일한 현실의 temporal structure를 학습했기 때문
2. 유사한 autoregressive prediction objective 때문
3. Semantic content가 비슷하기 때문
4. 비슷한 training data distribution 때문
5. 모델 구조나 representation extraction 방법의 영향 때문

따라서 공통 구조를 찾는 것뿐 아니라, **그 구조가 무엇에 의해 형성되었는지 구별할 수 있는 실험 설계**가 필요하다.

---

## 2.5. 핵심 실험 개념: Controlled Temporal Events

### 왜 Controlled Events인가?

Natural video와 natural language는 semantic content, appearance, temporal order, motion, causal information이 복잡하게 섞여 있다.

이 상태에서 alignment가 높다고 해서 temporal information을 공유한다고 말할 수 없다.

따라서 temporal property만 선택적으로 변경할 수 있는 controlled dataset이 필요하다.

### Candidate Temporal Properties

#### Level 1 — Event Ordering

- A occurs before B
- B occurs before A
- Same events, different order
- Identical event order, different language expressions

목표: 두 모델이 사건 순서의 차이를 representation geometry에 반영하는가?

#### Level 2 — Motion Direction

- Left → Right
- Right → Left
- Up → Down
- Stationary

목표: Appearance가 같은 상황에서 motion direction만 바꾸었을 때 geometry가 달라지는가?

#### Level 3 — Velocity and Acceleration

- Constant velocity
- Accelerating motion
- Decelerating motion
- Same initial/final position, different trajectories

목표: 연속적인 temporal quantities가 공통 구조를 형성하는가?

#### Level 4 — Causal / State Dependencies

- Object collision
- Motion reversal
- Object occlusion and reappearance
- Same present state, different histories
- Long-range event dependencies

목표: 단순한 event recognition을 넘어 temporal state와 과거 원인에 대한 정보를 표현하는가?

### Synthetic Data의 장점

- Ground-truth temporal property를 정확히 알고 있다.
- Appearance와 semantics를 통제할 수 있다.
- Counterfactual sequence를 생성할 수 있다.
- Motion direction, velocity, event ordering을 독립적으로 변경할 수 있다.
- Geometry 분석이 단순한 category similarity에 의존하는 것을 줄일 수 있다.

이 접근은 나의 diffusion grokking 연구에서 사용한 controlled setting과도 연결된다.

---

## 2.6. Experimental Plan

### Experiment 1 — Representation Extraction

두 모델에서 대응 가능한 temporal event representations를 추출한다.

**LLM side**

- 동일한 사건을 묘사한 structured natural-language sequences
- Event history나 future prediction과 관련된 hidden states
- 여러 layers 및 temporal positions 비교

**AR Video Diffusion side**

- 동일한 사건을 표현하는 video sequences
- History-conditioned hidden states
- Current chunk의 latent representations
- Transformer layers와 denoising timesteps 비교

**가장 큰 기술적 문제:** 무엇을 동일한 temporal state의 representation으로 볼 것인가?

LLM은 token position마다 representation이 존재한다.

Video diffusion은 spatial/temporal tokens에 더해 sampling-time dimension까지 존재한다.

따라서 pooling이나 readout 방식, hidden layer, noise level 선택이 geometry 결과에 미치는 영향을 확인해야 한다.

### Experiment 2 — Shared Geometry Measurement

우선 alignment 학습을 수행하기 전에 각 representation space의 구조를 비교한다.

Candidate methods:

- Representational Similarity Analysis (RSA)
- Mutual k-NN alignment
- CKA
- Distance-matrix comparison
- Procrustes / Wasserstein Procrustes alignment

이후 더 직접적인 cross-modal mapping 가능성을 살펴본다.

질문:

1. Temporal attributes가 embedding distance에 반영되는가?
2. 두 모델의 relative distance structures가 유사한가?
3. Simple linear/orthogonal mapping으로 정렬 가능한가?
4. Paired labels 없이도 alignment를 찾을 수 있는가?
5. Unseen temporal combinations에도 일반화되는가?

**주의:** Paired temporal events는 ground-truth 평가에 사용하더라도, unpaired alignment를 주장하려면 mapping 학습 단계에서는 대응 쌍을 사용하지 않아야 한다.

또한 AR video diffusion이 text conditioning을 사용한다면 공통 prompt 자체가 alignment를 유도할 수 있다. 가능한 경우 generic prompt 또는 통제된 conditioning을 사용하고, text leakage를 점검해야 한다.

### Experiment 3 — Is It Really Temporal?

높은 alignment를 발견하더라도 temporal understanding의 증거는 아닐 수 있다.

반드시 다음 조건을 비교해야 한다.

**Positive controls**

- 동일한 temporal relation
- 동일한 event ordering
- 동일한 motion state

**Negative controls**

- Same objects, reversed temporal order
- Same appearance, different motion
- Frame-shuffled sequences
- Identical final frame, different histories
- Similar semantic descriptions, different causal relationships

확인할 질문:

**Does the geometry preserve temporal relationships after semantic and appearance information are controlled for?**

이 질문이 실험의 핵심이다.

### Experiment 4 — What Makes the Geometry Shared?

공통 구조가 실제로 발견된다면, 그 원인을 더 구분한다.

가능한 비교:

- AR language model vs AR video diffusion
- AR video diffusion vs non-AR video diffusion
- Different AR video diffusion architectures
- Text models with different training objectives
- Different denoising timesteps and layers

목표는 공통 구조가 AR라는 속성과 관계있는지, 아니면 더 일반적인 temporal representation의 특성인지 분석하는 것이다.

반대로 alignment가 발견되지 않을 경우에도 다음을 구분해야 한다.

- 실제 shared geometry가 없음
- Representation extraction 방법이 부적절함
- 모델이 비교 대상 temporal property를 학습하지 못함
- Geometry가 linear/orthogonal mapping으로는 정렬되지 않음

---

## 2.7. Extension: Can Shared Geometry Improve Video Generation?

### Long-Term Goal

공통적인 temporal geometry를 발견하는 것에서 끝나지 않고, 이를 video generation 개선에 활용하고 싶다.

**If shared temporal geometry exists, can useful sequential priors from a language model be transferred to an autoregressive video generator?**

여기서 핵심적인 전제는 두 모델이 다른 강점과 약점을 가질 수 있다는 것이다.

예를 들어:

- LLM은 high-level event ordering과 long-range narrative planning에 강점을 보일 수 있다.
- Video diffusion은 연속적인 visual motion 표현에 강점을 보일 수 있다.
- 그러나 LLM이 정확한 physical trajectory를 더 잘 이해한다고 미리 가정해서는 안 된다.

따라서 실제로 어느 temporal property에서 어느 모델이 강한지 먼저 검증해야 한다.

### Potential Method — Representation Alignment / Model Stitching

Concept:

```text
AR Language Model
       |
       | Temporal representations
       v
Shared Geometry / Alignment Mapping
       |
       | Projected temporal features
       v
AR Video Diffusion Generator
       |
       v
Improved Temporal Generation?
```

가능한 접근:

1. LLM의 temporal representation을 추출한다.
2. Video model과 공유되는 geometry를 식별한다.
3. Cross-model mapping 또는 lightweight adapter를 학습한다.
4. 정렬된 temporal features를 video generator의 conditioning이나 guidance signal로 제공한다.
5. 실제 generation behavior가 개선되는지 검증한다.

Potential intervention mechanisms:

- Additional temporal conditioning tokens
- Lightweight feature adapters
- Activation-level feature injection
- History/memory-state conditioning
- Temporal guidance

이들은 모두 후보 방법론이며, 현재 입증된 접근은 아니다.

### Evaluation

**Short-range**

- Motion direction accuracy
- Trajectory consistency
- Velocity consistency
- Event ordering accuracy

**Long-range**

- Object permanence
- Event consistency
- History retention
- Temporal coherence across AR rollouts

**Generation quality**

- Visual fidelity
- Sample diversity
- Temporal controllability
- User-prompt consistency

가장 중요한 것은 alignment score 자체가 아니라 **실제 video generation의 temporal behavior가 바뀌는가**이다.

---

## 2.8. 가장 큰 기술적 난점과 반론

### Challenge A — Temporal Reasoning과 Temporal Dynamics는 다르다

LLM이 논리적인 문장 전개에 능숙하다는 사실이 정확한 물리적 움직임을 잘 표현한다는 것을 의미하지 않는다.

따라서 event-level temporal representation부터 시작한 뒤, continuous motion으로 확장하는 단계적 연구가 필요하다.

### Challenge B — Shared Geometry ≠ Shared Computation

두 모델의 representations가 유사한 geometry를 갖더라도 실제 계산 과정은 전혀 다를 수 있다.

이 연구의 첫 번째 질문은 representation-level convergence이며, computational mechanism의 동일성을 가정해서는 안 된다.

### Challenge C — Alignment ≠ Ability Transfer

공통 geometry가 발견되었다고 해서 그 representation을 이식하는 것만으로 다른 모델의 능력이 개선된다는 보장은 없다.

따라서:

1. Shared geometry 발견
2. Geometry의 temporal specificity 검증
3. Representation mapping 구축
4. Functional intervention
5. Generation improvement 검증

이라는 순서가 중요하다.

### Challenge D — Semantic Shortcut

두 모델이 동일한 temporal event를 이해한 것이 아니라 동일한 객체나 단어 의미를 공유하기 때문에 alignment가 높게 나타날 수 있다.

이를 방지하기 위해 temporal counterfactuals와 appearance-controlled experiments가 필요하다.

---

## 2.9. Connection to My Previous Research

### Grokking Project

기존 연구에서:

- Image-based modular arithmetic을 controlled task로 사용했다.
- Diffusion model의 Fourier-like periodic representations를 발견했다.
- Sampling time을 따라 representation과 기능의 변화를 조사했다.
- Natural-image generation으로 일반화하려 할 때 해석 가능한 공통 지표를 찾는 데 한계를 경험했다.

### New Direction

이번 제안은 다음 질문으로 확장된다.

**Can we identify representation structures that remain meaningful across different models, modalities, and generative processes?**

특히 temporal dynamics라는 구체적인 문제에서 공통 구조를 찾는 것이다.

기존 연구가 단일 모델에서 특정 계산이 어떻게 나타나는지 조사했다면, 이번에는 여러 모델이 학습한 temporal relationships의 공통된 구조를 탐구한다.

---

## 2.10. Interview Pitch (약 45초)

> I was fascinated by Professor Isola's work on the Platonic Representation Hypothesis and Shared Geometry, especially the idea that independently trained models can develop compatible representational structures despite differences in modality and training objectives. I wondered whether something similar might exist for temporal relationships across autoregressive language models and autoregressive video diffusion models. I would first investigate this using controlled temporal events, comparing representation geometries while separating genuine temporal information from semantic similarity. If shared temporal structures exist, I would then explore whether they could serve as a bridge for transferring useful sequential priors into video generation, potentially improving long-range temporal consistency.

---

# 3. 두 연구 제안에 공통되는 연구 철학

## 3.1. The General Research Principle

**Understand the structure of generative models, then exploit that understanding to improve generation.**

기존 연구 경험과 연결하면:

```text
Grokking in Diffusion
        |
        v
Finding Internal Representations
        |
        v
Sampling-Time Mechanistic Analysis
        |
        v
Using Model-Derived Signals
        |
        v
Guidance for Video Generation
        |
        v
More Targeted Interventions
        |
        +----------------------------+
        |                            |
        v                            v
Torralba                         Isola
Within-Model Routing             Cross-Model Geometry
        |                            |
        v                            v
Causal Intervention              Representation Transfer
        |                            |
        +-------------+--------------+
                      |
                      v
          More Reliable and
          Controllable Video Generation
                      |
                      v
          Interactive World Models
```

## 3.2. Why These Professors?

### Antonio Torralba

- 연구의 초점: 특정 모델의 내부 정보 전달 방식
- 가장 중요한 연결: Vision-language binding → Temporal history binding
- 내가 가져오는 경험: Diffusion interpretability, sampling-time analysis, AR video generation, attention-head intervention
- 제안의 강점: 구체적인 architecture와 causal intervention 방법을 제시할 수 있음

### Phillip Isola

- 연구의 초점: 서로 다른 모델의 representation에서 발견되는 공통 구조
- 가장 중요한 연결: Platonic geometry → Shared temporal geometry → Potential representation transfer
- 내가 가져오는 경험: Controlled synthetic tasks, representation analysis, temporal generation 연구
- 제안의 강점: 기존에 발견한 원리를 새로운 generative paradigm으로 확장하는 연구 질문

---

# 4. Interview Preparation — 예상 질문과 답변 핵심

## Q1. Why would text tokens contain video-history information?

**Answer:** Joint attention may allow visual history information to influence text-token hidden states. Similar routing has been observed for reference-image attributes in FLUX.2, but whether it generalizes to temporal information in video diffusion is an open question.

**Must mention:** Attention connectivity, hypothesis vs evidence, direct vs mediated pathways.

## Q2. Why can't we simply use direct video-to-video attention?

**Answer:** Direct attention may indeed be sufficient or dominant. The research goal is to understand whether different temporal attributes use different pathways and whether selectively manipulating those pathways can improve control.

**Must mention:** Functional specialization, causal knockout, no assumption that text mediation is necessary.

## Q3. How would you demonstrate that text tokens causally affect motion?

**Answer:** Construct controlled history pairs, perform attention knockout and cross-run activation patching, and test whether manipulating text representations changes generated motion while holding other conditions fixed.

**Must mention:** Activation decodability is not causal necessity.

## Q4. What if the Torralba hypothesis fails?

**Answer:** A negative result could establish that fine-grained temporal information primarily follows direct history-to-video pathways. This would still reveal how autoregressive video diffusion uses its context and could motivate targeted interventions on direct visual memory.

**Must mention:** Negative findings can clarify architectural information routing.

## Q5. Why should language and video models share temporal geometry?

**Answer:** They may learn related temporal relationships because they model different observations of the same world. However, autoregression alone does not guarantee shared geometry, so this must be tested against semantic and architectural controls.

**Must mention:** Common underlying events, no guarantee from shared AR structure.

## Q6. How would you distinguish semantic alignment from genuine temporal alignment?

**Answer:** Use controlled examples with identical objects and appearance but altered temporal order, motion direction, or velocity. If representations preserve these temporal distinctions across modalities, the evidence becomes more specific than global semantic alignment.

**Must mention:** Temporal counterfactuals, shuffled-frame controls, matched semantic content.

## Q7. Why does shared geometry imply that knowledge can be transferred?

**Answer:** It does not imply transfer by itself. Shared geometry motivates investigating whether a mapping between representation spaces can expose useful temporal features to another generator. Functional transfer must be tested independently.

**Must mention:** Alignment is necessary for the proposed bridge, but not sufficient for ability transfer.

## Q8. What if LLMs are not better at understanding motion?

**Answer:** The proposal does not require LLMs to outperform video models on every temporal property. It first identifies which temporal relations are represented by each model, then explores transfer only where one provides useful complementary information.

**Must mention:** Event-level reasoning vs continuous physical dynamics.

## Q9. How are the two proposals different?

**Answer:** The Torralba proposal studies how temporal information travels inside one generative model, while the Isola proposal studies whether different generative models share a geometric structure for temporal relationships. Both connect representation analysis to interventions, but at different levels.

## Q10. How does this relate to your PhD research goal?

**Answer:** My overarching goal is to understand internal representations and computations in generative models, then leverage those insights to make video generation more reliable and controllable. The Torralba and Isola directions offer complementary ways to pursue this goal.

---

# 5. Quick Review — 핵심만 다시 보기

## Torralba

**Starting Paper:** Vision-Language Binding in In-Context Image Generation

**Key Finding:** Reference image attributes can flow through text tokens, while finer identity information depends more on direct image pathways.

**My Hypothesis:** Past video history may similarly influence text-token representations in AR video diffusion, potentially creating an implicit temporal memory channel.

**Model:** Joint-attention video diffusion adapted to AR, history-conditioned generation.

**Experiments:** Controlled history pairs → Video Lens → Attention knockout → Activation patching.

**Main Risk:** Text-mediated motion pathways might not exist or might not be causally important.

**Goal:** Targeted control of motion and long-term memory.

## Isola

**Starting Papers:** The Platonic Representation Hypothesis; Shared Geometry as a Rosetta Stone.

**Related Work:** Dynamic Reflections — emergent video-text alignment, including limited temporal-order analysis.

**My Hypothesis:** AR language and AR video diffusion models may share representation geometry for certain temporal relationships.

**Models:** Heterogeneous generative models with different modalities and learning objectives.

**Experiments:** Controlled temporal events → Representation extraction → Geometry comparison → Temporal counterfactuals → Alignment mapping.

**Main Risk:** Similarity may come from semantics rather than temporal reasoning, and geometric alignment may not enable functional transfer.

**Goal:** Discover transferable temporal structure and use it to improve video generation.

---

# 6. Final One-Line Distinction

**Torralba:** *Where and how does temporal information flow within an autoregressive video diffusion model?*

**Isola:** *Do different autoregressive generative models share a geometry of temporal relationships, and can that structure help transfer useful temporal priors?*

**My Research Vision:** *Understanding generative representations and computations to build more reliable, temporally coherent, and controllable models of the world.*