# BU PhD SOP — Deepti Ghadiyaram & Aaron Mueller 연구 연결 노트

> 작성 기준: 2026-11-15 대화 및 BU Personal Statement 초안(`personal_statement(1).pdf`).  
> 목적: 몇 주/몇 달 뒤 다시 읽어도 **왜 이 교수들에게 관심을 가졌고, 어떤 논문에서 무슨 아이디어를 얻었으며, 내 기존 연구의 어떤 한계를 다음 연구 질문으로 발전시켰는지** 빠르게 복원하기.
>
> **표기:** `논문 내용` = 저자들이 논문에서 실제로 수행·주장한 사항 / `나의 해석·연결` = SOP 준비 중 도출한 생각 / `연구 가설·제안` = 아직 입증하지 않은 후속 연구 아이디어. 세 범주를 혼동하지 않기.

## 0. 한눈에 보는 연구 흐름

```text
[이전 연구 경험]
Diffusion grokking: internal representations / sampling-time mechanisms
          ↓
CIFAR-10: predictive representation metrics ≠ full mechanistic explanation
          ↓
AR video diffusion: uncertainty-aware motion guidance / skip-layer perturbation
          ↓
Attention-head 분석: head별 gradient signal과 생성 결과 변화 관찰
          ↓
[두 가지 PhD 연구 방향]

Ghadiyaram: 해석 가능한 motion/memory feature를 찾고 → 선택적으로 개입해 generation 개선
   Revelio (diffusion activation → k-SAE feature discovery)
       + Concept Steerers (text embedding feature → concept control)
       → attention output에 k-SAE / feature decoder direction intervention

Mueller: temporal representation을 포착하고 → 인과적 역할을 엄밀히 검증
   Priors in Time (temporal feature analysis: predictable / novel)
       + Causality is Key (association / intervention / counterfactual, identifiability)
       → AR video history-dependent representations + causal tests
```

**두 방향의 차이:** Ghadiyaram은 **intervention을 이용한 controllable generation**, Mueller는 **representation 분석과 그에 대한 정당한 causal claim**에 무게중심이 있다. 두 방향은 연결되지만 동일한 제안이 아니다.

---

## 1. 이 제안으로 이어진 내 연구 경험 (SOP 문맥)

1. **Grokking / modular arithmetic diffusion**: Fourier-like periodic representations, sampling trajectory를 따른 기능 변화, error injection을 통해 arithmetic computation과 visual refinement의 차이를 연구. 여기서 *관찰 가능한 패턴과 실제 mechanism 설명의 경계*를 고민하기 시작.
2. **CIFAR-10 확장**: class-wise feature-volume 지표가 rectification ability와 연관되었지만, 그 지표 자체가 이미지 생성의 내부 계산을 설명하는 것은 아니었음. *Predictive signal ≠ mechanistic account*라는 교훈.
3. **공동 AR video diffusion 프로젝트**: motion-aware guidance에서 **spatial/patch-wise teacher uncertainty**를 활용해 skip-layer perturbation을 다르게 적용. 배경처럼 history로 상대적으로 잘 정해진 정보와 움직임처럼 불확실할 수 있는 영역을 구분해 효율적인 generation/distillation을 도모. Anti-drifting guidance는 AR rollout 누적오차를 다루는 별도 축.
   - **기여 서술 주의:** 팀 협력 프로젝트이며 연구 전반을 단독 주도한 것으로 표현하지 않는다. 내가 참여하면서 얻은 관찰과 그로부터 발전시킨 질문을 구분한다.
4. **후속 attention-head 연구**: history-consistent / inconsistent future 조건의 attention-head parameter gradient를 비교하고, 특히 denoising 초기에 음의 cosine similarity 신호를 확인. 개별 head를 조작했을 때 **물체가 확대되는 생성 변화**도 관찰.
   - 이것은 head 개입이 output에 영향을 준다는 실제 관찰이지만, 그 head가 `motion consistency`, `memory`, `physical causality`를 담당한다는 증거와는 다르다.
   - Head 전체를 perturb하더라도 그 내부에서 **어떤 feature**가 변화의 원인인지 아직 분리되지 않음.

---

## 2. Deepti Ghadiyaram — 관련 논문 4편

### [ref5] Generative Action Tell-Tales: Assessing Human Motion in Synthesized Videos

- **링크:** https://arxiv.org/abs/2512.01803
- **SOP에서의 역할:** 연구 대상의 직접적 overlap — **video motion quality / evaluation**.
- **논문 내용:** 생성 비디오의 복잡한 사람 동작은 단순 appearance 기반 metric으로 제대로 평가하기 어렵다. 실제 인간 동작의 latent distribution을 학습하고, skeleton geometry와 appearance feature를 결합해 human-action plausibility를 평가한다. Temporal/anatomical 오류를 겨냥한 benchmark도 제안.
- **나의 연결:** 내가 개선하려는 temporal coherence 또는 motion fidelity를 *무엇으로 평가할 것인가*에 대한 문제의식과 부합. **이 논문을 SAE 기반 guidance 자체의 선행연구로 설명하면 안 됨.**

### [ref6] DDiT: Dynamic Patch Scheduling for Efficient Diffusion Transformers

- **링크:** https://arxiv.org/abs/2602.16968
- **SOP에서의 역할:** **denoising-time adaptive computation**이라는 보완적 overlap.
- **논문 내용:** DiT가 모든 denoising step에서 고정 patch 크기를 쓰는 비효율을 지적. Content complexity와 denoising timestep에 따라 inference의 patch/token size를 조절한다. 초반에는 coarse/global structure, 후반에는 fine/local detail에 더 높은 해상도를 사용한다.
- **나의 연결:** 내가 patch별 uncertainty로 guidance 강도를 조절한 것과 **입력 영역·시점의 필요에 따라 연산이나 개입을 달리한다**는 발상에서 만난다. 단, DDiT는 동적 **tokenization/compute scheduling**, 내 프로젝트는 **uncertainty-aware guidance**이므로 동일한 방법은 아님.

### [ref7] Revelio: Interpreting and Leveraging Semantic Information in Diffusion Models

- **링크:** https://arxiv.org/abs/2411.16725
- **발표:** ICCV 2025
- **SOP에서의 역할:** **직접적인 방법론적 출발점 — diffusion 내부 representation에 k-SAE 적용**.
- **논문 내용:** 서로 다른 diffusion architecture, layer, denoising timestep에 걸친 visual semantic information을 분석. Frozen diffusion activation에서 k-sparse autoencoder를 활용해 해석 가능한 finer-grained semantic features를 발견하고, transfer-learning/classifier 기반 분석으로 유용성을 검토.
- **내가 끌린 이유:** Model 내부 activation을 개별 의미 단위로 분해한다는 것. Head 전체를 perturb하는 현재 방법보다 더 작은 feature-level 단위로 내려갈 가능성을 제공.
- **정확성 주의:** Revelio의 핵심은 **feature discovery / interpretation**. 이것만으로 모든 feature의 causal role이나 steering 능력이 증명된 것은 아니다.

### [ref8] Concept Steerers: Leveraging K-Sparse Autoencoders for Controllable Generations

- **링크:** https://arxiv.org/abs/2501.19066
- **SOP에서의 역할:** **발견한 interpretable concept을 generation control로 연결**하는 보완적 연구 방향.
- **논문 내용:** Text-to-image 모델의 **text embedding / text-encoder representation**에 k-SAE를 적용해 의미 있는 concept을 찾고, test time에 concept을 강화·억제하거나 style을 바꾸는 control에 사용. Base diffusion model 자체의 재학습 없이 steering 수행.
- **내가 끌린 이유:** `Feature discovery → Targeted manipulation → Generated output 변화`라는 연구 논리.
- **정확성 주의:** **Concept Steerers는 Revelio에서 분석한 U-Net 내부 feature를 그대로 steering한 논문이 아니다.** 두 논문은 `발견`과 `제어`라는 **상보적인 연구 방향**으로 함께 인용하는 것이 정확하다.

### Ghadiyaram 논문 네 편의 역할 구분

| 역할 | 논문 | 왜 언급했는가 |
|---|---|---|
| 대상/평가 | Generative Action Tell-Tales | 사람 움직임의 시간적 타당성을 이해·평가 |
| Adaptive inference | DDiT | denoising timestep / local complexity를 고려하는 계산 |
| Representation discovery | Revelio | diffusion activation의 interpretable k-SAE features |
| Controllable generation | Concept Steerers | interpretable feature를 이용한 개념 steering |

**SOP에서 가장 중요한 연결은 [ref7] + [ref8].** [ref5]와 [ref6]은 교수 연구실의 broader fit을 보여주는 역할이므로 상세한 기술 설명을 길게 넣을 필요가 없다.

---

## 3. Ghadiyaram에게 제안한 후속 연구

### 문제의식

- 기존 **skip-layer guidance(SLG)**는 효과적인 개입 방법이지만 개별 semantic function을 식별하지는 못한다.
- 현재의 **head-level perturbation**은 layer-level보다 세밀해도 head 안의 여러 기능이 얽혀 있을 수 있다.
- 따라서 `head를 바꾸니 이미지/비디오가 변했다`는 발견에서 `어떤 feature를 바꿨기에 특정 변화가 생겼는가`로 나아가고 싶다.

### 핵심 연구 질문

> **Can k-SAEs identify disentangled motion- and memory-related features in attention outputs of AR video diffusion models, and can interventions along their decoder directions improve temporal coherence more selectively than coarse-grained perturbations?**

### 제안하는 접근 (아직 실험 완료한 내용 아님)

1. AR video diffusion Transformer의 **attention outputs**에서 activation을 수집한다. Head를 구분할 수 있는 표현(pre-output projection 등)과 aggregate attention output을 비교할 수 있다.
2. k-SAE로 sparse features를 학습해 **motion, object persistence, identity, scene memory**와 연관된 후보 feature를 찾는다.
3. Controlled video conditions에서 feature activation과 기능의 상관관계를 분석한다. **Motion이 크다 = novel/uncertain하다**고 사전에 확정하지 않는다.
4. 선택한 decoder direction에 개입하고, baseline인 layer-level 또는 head-level perturbation과 비교한다.
5. Motion improvement뿐 아니라 **selectivity / collateral changes**를 함께 평가한다. Object identity, appearance, layout, long-horizon drift가 유지되는지도 중요하다.

### 무엇이 성공 기준인가?

- 동일하거나 더 높은 temporal coherence / controllability를 **더 적은 부수효과**로 달성.
- 특정 feature 개입이 일관되게 한 속성에 영향을 주는지, seed·scene·motion 유형을 바꿔도 재현되는지 확인.
- 단순한 sparse feature의 semantic correlation이 아니라 **causal intervention의 효과**를 확보.

### 남은 위험/미해결점

- Head 내부 기능이 SAE의 개별 feature로 깔끔히 분리된다는 보장은 없음.
- SAE decoder direction이 안정적인 causal control axis라는 보장은 없음.
- Temporal coherence가 개선되어도 그 mechanism의 유일성·necessity가 입증되는 것은 아님.

---

## 4. Aaron Mueller — 관련 논문 2편

### [ref9] Causality is Key for Interpretability Claims to Generalise

- **링크:** https://arxiv.org/abs/2602.16698
- **성격:** Mechanistic interpretability에 대한 **position paper**. Shruti Joshi가 제1저자, Aaron Mueller가 공동저자.
- **SOP에서의 역할:** **내가 연구자가 되고 싶은 태도 / 주장과 증거의 기준**.

**논문 내용**

- `Pearl's causal ladder`를 interpretability claim의 근거 수준에 적용:
  - **L1 Association**: Activation–behavior correlation / probing이 보여주는 것.
  - **L2 Intervention**: Ablation, patching, steering으로 출력이 달라진다는 것.
  - **L3 Counterfactual**: 특정 사례에서 요인이 달랐다면 결과가 어땠는지 설명하기 위해 필요한 구조·가정.
- **Identifiability**: 관찰과 개입 결과를 똑같이 설명하는 여러 가설이 있다면, 특정 feature나 mechanism을 유일하게 발견했다고 단정할 수 없다. 어떤 estimand와 intervention class를 겨냥했는지 명시해야 한다.
- **Generalization / transportability**: 한 데이터·프롬프트·모델에서 발견한 효과가 다른 조건에서도 유지되는지는 별도로 평가해야 한다.

**나의 개인적 연결**

- MI를 처음 탐구하면서 **"이 실험 결과로 어디까지 mechanism을 발견했다고 주장할 수 있는가?"**라는 기준을 계속 찾고 있었음.
- 이 논문은 그 질문을 명확한 causal vocabulary로 정리해 주어 인상 깊었고, 향후 연구에서도 그런 태도를 유지하고 싶음.
- **Pearl's ladder나 identifiability를 이 논문이 새로 발명한 것은 아님.** 기존 causal inference의 개념을 MI 연구에 체계적으로 적용한 contribution.

### [ref10] Priors in Time: Missing Inductive Biases for Language Model Interpretability

- **링크:** https://arxiv.org/abs/2511.01836
- **SOP에서의 역할:** **Temporal Feature Analysis(TFA)라는 구체적 방법론적 영감**.

**논문 내용**

1. Language model activation은 sequence position에 따라 **effective intrinsic dimensionality가 증가**하고, 시간에 따른 **context correlation / non-stationarity**가 관찰된다. 이는 Transformer hidden width 자체가 변한다는 뜻은 아니다.
2. Conventional SAE의 tokenwise additive reconstruction/sparsity objective를 Bayesian MAP 관점에서 보면 latent의 **time-i.i.d. prior**가 함축된다. **실제 SAE 출력이 반드시 통계적으로 독립적이라는 뜻은 아니다.**
3. 이런 prior는 시간적으로 연결된 representation geometry를 잘 포착하지 못할 수 있고, 제한적인 sparsity budget에서 support switching / feature splitting을 유도할 수 있다.
4. TFA는 현재 activation을 다음처럼 **context-predictable + novel residual**로 분해한다.

   \[
   x_t \approx D(z_{p,t}+z_{n,t})
   \]

   - `Predictable`: 과거 activation과 현재 activation 간 관계를 attention으로 분석해 얻는 비교적 느리게 변화하는 contextual representation.
   - `Novel`: predictable 성분으로 설명되지 않는 residual에 SAE를 적용해 얻는 새 정보.
   - **정확성 주의:** Predictable part는 과거만으로 미래를 forecasting한 결과가 아님. **현재 activation도 query로 사용**하는 context-conditioned decomposition이다.
   - TFA 역시 **novel residual의 i.i.d.**라는 남은 가정을 가진다.
5. Story의 event/sub-event, garden-path 문장 구조, in-context graph geometry를 standard SAE보다 잘 드러낸다고 보고한다.
6. In-context graph task에서 **일부 counterfactual editing** 실험도 수행(9개 node 중 4개에서 기대한 변화). 그러나 **체계적인 causal evaluation을 끝냈다고 주장하지 않고**, 생성 중 code intervention의 확대를 future work로 남긴다.

**나의 연결**

- 기존 motion-aware guidance는 움직임/uncertain region을 **개입 대상으로 식별**하는 데 유용했지만, 모델이 *과거로부터 무엇을 유지하고, 무엇을 새로운 변화로 처리하는지*를 내부 representation에서 밝히지는 못했다.
- TFA의 temporal inductive bias는 그 차이를 activation 공간에서 연구할 단서가 될 수 있다.
- `TFA novel ≠ teacher uncertainty ≠ motion`: 예를 들어 일정 속도로 움직이는 공은 motion이 있지만 미래 위치는 충분히 predictable할 수 있다.

---

## 5. Mueller에게 제안한 후속 연구

### 연구 질문 1 — Temporal representations를 관찰할 수 있는가?

> **Can temporally informed representation analysis reveal persistent states, motion transitions, and history-dependent structure in AR video diffusion models?**

**분석 대상의 두 단계**

- **PoC: Pixel / VAE latent space** — 프레임·패치 간 temporal correlation과 predictable/novel 성분을 분석. 비디오 데이터 그 자체의 temporal geometry를 분해할 수 있는지 점검.
- **본격적인 MI: Diffusion Transformer internal attention activations** — 실제 AR 생성모델이 conditioning history를 내부적으로 어떻게 사용하는지 조사. **VAE에서의 temporal 구조 발견만으로 생성모델의 내부 computation을 규명했다고 할 수는 없다.**

**설계 시 중요한 점**

- Video time(프레임·chunk 진행)과 denoising time(하나의 chunk 내 sampling)을 분리. 같은 denoising step에서 video-time 변화를 먼저 비교하고, 그 후 noise level 간 차이를 분석하는 접근이 가능.
- 움직이는 객체가 고정 patch를 벗어나면 **patch displacement 때문에 residual이 커질 수 있음**. Motion-compensated/object-centric alignment를 비교해야 할 수 있다.
- `stable background`, `uniform motion`, `sudden velocity change`, `occlusion`, `identity change` 같은 controlled scenarios에서 predictable/novel code가 어떻게 반응하는지 비교.

### 연구 질문 2 — Causal intervention으로 기능을 검증할 수 있는가?

> **Do the identified temporal representations play a causal role in maintaining temporal consistency during autoregressive generation?**

- Temporal code 또는 대응 activation을 **ablate, patch, steer**하여 실제 video rollout이 어떻게 달라지는지 비교.
- 같은 시각적 변화에 대해 무작위 feature, matched-norm feature, 다른 time step, 다른 object를 쓰는 control이 필요.
- **L1:** feature가 motion/memory와 연관됨 → **L2:** feature 조작이 motion/memory에 실제 영향 → **더 강한 주장:** necessity·specificity·generalization 등 추가 evidence.
- Full mechanistic explanation을 보장한다고 주장하지 말고, **어느 수준의 causal role을 검증했는지** 엄밀히 정의.

### 연구 질문 3 — Semantically causal event relationships까지 확장할 수 있는가? (장기 목표)

**두 종류의 causality를 구분해야 함.**

1. **Causal role of a representation:** 모델 내부 feature를 바꾸면 생성 결과가 바뀌는가?
2. **Representation of causal relationships in the world:** 모델이 충돌, action→effect, object interaction 같은 원인–결과 관계를 내부에 표현하는가?

두 번째는 더 어려운 별도의 연구 문제. 예를 들어 collision 여부, 충돌 전 속도, 질량/물체 특성 등을 통제한 **counterfactual video pairs**를 구성해야 할 수 있음. 단순 motion correlation을 `physical causality`라고 부르지 않기.

**이번 SOP에서 우선 약속한 범위:** 지속적인 상태와 motion 관련 temporal representation을 찾아 **temporal consistency에 대한 causal role**을 검증해 보고 싶다는 것. **Physical causal dynamics의 완전한 해석**은 추후 확장 가능성으로 남겨두기.

---

## 6. 두 교수의 제안이 실제로 다른 지점

| 질문 | Ghadiyaram | Mueller |
|---|---|---|
| 출발한 내 연구 | SLG → head-level perturbation | uncertainty-aware guidance의 설명력 한계 + MI의 증거 기준 |
| 중심 논문 | Revelio, Concept Steerers | Priors in Time, Causality is Key |
| 해석 단위 | Attention output의 sparse feature / decoder direction | History-conditioned temporal code, geometry, causal estimand |
| 1차 목표 | Motion/memory feature를 더 선택적으로 조작 | Persistent/novel temporal structure를 관찰·식별 |
| 후속 목표 | Generation 개선, side effect 감소 | Intervention으로 temporal consistency 관련 기능 검증 |
| 핵심 문장 | **Interpretable features → selective control** | **Temporal structure → causal understanding** |

**공통된 더 큰 비전:** `Understanding internal representations → carefully validated intervention → more reliable and controllable generative models`.

**실제 지도교수 fit 유의:** Ghadiyaram은 영상 생성/해석과 직접적인 연구 overlap이 큼. Mueller는 연구 철학·방법론의 적합성이 매우 높지만 연구실의 중심 대상은 **language models**이므로, SOP/면접에서 video-only 지도 제안으로 비칠 위험이 있다. Language/video를 관통하는 방법론적 질문을 강조하되, 실제 공동연구 범위는 교수의 관심사와 맞춰 논의할 필요가 있음.

---

## 7. 면접 전 반드시 복습할 질문

- **Revelio vs Concept Steerers**: 하나는 diffusion activation 분석, 다른 하나는 text embeddings 기반 steering이라는 차이를 정확히 설명할 수 있는가?
- **SAE의 단위**: 어떤 activation을 input으로 넣을 것인가? Head output / concatenated outputs / projected attention output은 어떻게 다른가?
- **Evidence level**: Negative gradient cosine이나 object enlargement 효과로 **무엇을 주장할 수 있고 없는가?**
- **TFA의 정의**: Predictable은 미래 예측값이 아니라 현재 activation을 query로 삼는 context-conditioned component라는 점을 설명할 수 있는가?
- **Novelty vs uncertainty**: Residual이 큰 이유가 surprising motion인지, 단순 spatial misalignment인지, 표현 변화인지 어떻게 통제할 것인가?
- **Video / diffusion time**: 두 시간축을 어떻게 분리하고 평가할 것인가?
- **Causality**: Feature 조작의 output effect와 실제 사건의 cause–effect representation을 구분하고 있는가?
- **Falsification**: TFA가 video에서는 standard SAE보다 나아지지 않거나, 발견한 feature 개입이 selectivity를 높이지 못한다면 어떻게 해석할 것인가?
- **Research ownership**: FlowMap 프로젝트에서 팀의 성과와 내가 독립적으로 제안·분석·실험한 부분을 어떻게 정확히 구분할 것인가?

---

## 8. SOP 논리 복원용 두 문장

**Deepti** — *I want to move from effective layer/head perturbations to identifying the underlying interpretable motion- and memory-related features, then test whether feature-level interventions can control video generation more selectively.*

**Mueller** — *I want to understand how generative models represent temporal structure and validate the causal roles of those representations, while being explicit about what our evidence can—and cannot—justify.*

---

## 9. 논문·원문 링크 (SOP 번호 유지)

- **[ref5]** [Generative Action Tell-Tales: Assessing Human Motion in Synthesized Videos](https://arxiv.org/abs/2512.01803) — motion evaluation
- **[ref6]** [DDiT: Dynamic Patch Scheduling for Efficient Diffusion Transformers](https://arxiv.org/abs/2602.16968) — adaptive denoising-time computation
- **[ref7]** [Revelio: Interpreting and Leveraging Semantic Information in Diffusion Models](https://arxiv.org/abs/2411.16725) — k-SAE for diffusion activations
- **[ref8]** [Concept Steerers: Leveraging K-Sparse Autoencoders for Controllable Generations](https://arxiv.org/abs/2501.19066) — k-SAE text-embedding concept control
- **[ref9]** [Causality is Key for Interpretability Claims to Generalise](https://arxiv.org/abs/2602.16698) — causal evidence, identifiability
- **[ref10]** [Priors in Time: Missing Inductive Biases for Language Model Interpretability](https://arxiv.org/abs/2511.01836) — TFA, predictable/novel temporal structure

**메모:** 여기서 명시한 실험 설계는 당시의 *제안*이며, 논문에서 수행한 사실이나 현재까지 완료한 실험과 구분한다. 출판·모집 상황은 지원 시점에 따로 재확인한다.
