# Virginia Tech PhD Interview — PI & Paper Review

> 기준: VT SOP 2페이지 최종 수정본의 **Why VT** 문단 + 이전 **[Accum] PI Search 3** 논의.  
> **Pinar Yanardag = temporal memory / scene evolution**, **Chris Thomas = prompt-conditioned activation control**.  
> **논문이 보여준 결과**와 **내가 제안한 후속 연구**를 구분해서 기억할 것.

## 0. 30초 복기

- **Infinity-RoPE (Yanardag)**: **Block-Relativistic RoPE**로 시간 좌표를 sliding/re-anchor하고, **KV Flush**로 action prompt 변경에 즉각 반응하며, **RoPE Cut**으로 scene transition을 유도. Figure 6의 attention map은 intervention이 의도한 방향으로 attention allocation을 바꾼다는 것을 보여줌.
- **AdaState (Yanardag)**: 첫 frame에 지속적으로 attention이 몰리는 **attention sink** 현상을 반대로 활용. 고정된 first-frame KV 대신, 각 chunk에서 새로 denoise한 **evolving hidden state**의 KV를 sink 위치에 저장하여 **scene consistency와 dynamics**를 함께 유지.
- **SteerVLM (Thomas)**: target/converse prompt를 이용해 경량 steering module이 frozen VLM의 **hidden activations**를 layer·dimension별로 조작. 내 관심은 이를 비디오의 **text–video cross-attention에서 motion prompt가 어떻게 작동하는가**라는 질문으로 확장하는 것.

---

## 1. Pinar Yanardag — Infinity-RoPE (CVPR 2026)

[Paper](https://arxiv.org/abs/2511.20649) · [Project](https://infinity-rope.github.io/) · [Code](https://github.com/yesiltepe-hidir/infinity-rope)

### 문제 설정: 왜 long-horizon에서 문제가 생기나?

- 기반 모델은 **Self-Forcing으로 distill된 Wan2.1-T2V-1.3B**; 학습 길이는 약 5초 = **21 latent frames**, AR 생성은 **3 latent frames/block**.
- Wan의 **3D-RoPE**는 head feature를 temporal/height/width로 나누고, 각 축에 대해 **길이 1024의 position lookup table**을 준비. `1024`를 세 축이 나눠 쓰는 것이 아님.
- **구분할 것:** `1024`는 구현의 precomputed position 범위, `21`은 학습에서 경험한 temporal horizon. **RoPE 수학 자체에는 finite temporal index 상한이 없음.** 핵심 위험은 특히 **training 중 보지 못한 긴 상대적 temporal distance** (예: 고정 sink ↔ 현재 frame).
- 기존 Self-Forcing cache에서는 **sink로 보존된 첫 frame의 K/V는 sliding/eviction 대상에서 제외**되고, 이미 RoPE 적용된 K의 temporal phase도 새로 조정하지 않음. 최신 frame만 계속 증가하는 global index를 가짐.

### (A) Block-Relativistic RoPE — temporal coordinate를 재배치

- **핵심:** 새로운 block의 실제 시간은 계속 증가해도, 현재 block이 training horizon 내의 **고정된 local RoPE index**를 사용하게 함. 그에 맞춰 과거 cached frame의 **temporal index를 뒤로 이동**. **내용 latent, Q/K/V를 과거 것으로 대체하는 것이 아님.**
- 1-based 설명 예: 실제 `19–21` 생성 시 RoPE `{19,20,21}` → 실제 `22–24` 생성 시 새 block의 RoPE는 다시 `{19,20,21}`. **기존 실제 frame 19의 index는 `19→16`**으로 이동.
- **실제 frame 19가 cache에 계속 남아 있다는 가정** 아래, 생성 block별 frame 19의 index: `19–21 생성:19` → `22–24:16` → `25–27:13` → `28–30:10` → `31–33:7` → `34–36:4` → `37–39:1`.
- 구현 원리: 과거 K는 **변경된 RoPE 좌표로 재회전**해야 함(가령 unrotated K를 cache에 보관하고 attention 시 적용); V의 내용 자체를 position remap 때문에 바꾸는 것은 아님. 원본 Self-Forcing의 **이미 회전된 K를 이동만 하는 캐시**와 구별.
- **수학적 주의:** 모든 Q/K index를 동일량 이동하면 `QᵀR(j−i)K`의 상대 위치가 유지되어 score 자체는 불변. 실제 차이는 **sink의 별도 anchoring, relative distance 재조정, index clipping** 같은 부분에서 생김. 단순히 절대 index가 21을 넘었다는 것만으로 score가 바뀐다고 하면 부정확함.

### (B) Unbounded Cache / Semanticization

- **용어에 속지 말 것:** 논문의 *unbounded cache size*는 **학습 horizon(21)보다 cache capacity가 큰 경우**의 temporal encoding 설명. **물리적 GPU cache가 무한히 늘어난다는 뜻은 아님.** capacity를 고정할 수 있고 가득 차면 eviction 가능.
- 오래된 frame의 index가 0/음수가 될 상황에서는 **최소 index `1`로 collapse**: `1,1,1,…,1,2,3,…,21`(예: cache 42 frames를 이 범위로 표현). **기존 K/V tensor들을 합치거나 평균 내는 memory compression은 아님.** 오래된 *시간 순서 정보*를 희생하고 content를 유지하는 것.
- 저자 [GitHub Issue #2](https://github.com/yesiltepe-hidir/infinity-rope/issues/2): 2026-01-19 당시 **공개 구현은 bounded-cache 버전**이고, unbounded semanticization 코드는 추후 추가할 예정이라고 설명. 따라서 **논문상 개념**과 **당시 공개 코드에서 확인한 동작**은 별개로 기억.

### (C) KV Flush — prompt-responsive action control

- **문제:** 기존 scene의 K/V가 많이 남으면 새로운 action prompt를 주더라도 과거 semantic context가 영향을 주어 **prompt response가 느려짐**.
- **방법:** prompt 전환 시 오래된 KV를 flush하고 **global sink + 마지막 latent frame**만 유지(논문 기본 설명). Sink로 전역 identity를, 마지막 frame으로 국소 motion continuity를 지키면서 **새 prompt가 빠르게 반영**되도록 함.
- **중요:** 새로운 action으로의 **연속적 전환**이 목적이지, 완전히 다른 장면으로 자르는 것이 아님. Full recache보다 가벼운 inference-only cache intervention.

### (D) RoPE Cut — multi-cut scene transition

- 현재 block `B_f={f−2,f−1,f}`를 `B_{f→f+Δ}={f−2,f+Δ−1,f+Δ}`로 **불연속 remap**.
- **직접 계산한 예:** `f=22, Δ=3`: `{20,21,22} → {20,24,25}`. **24,25는 아직 생성되지 않은 미래 frame의 영상 내용이 아니라, 적용할 RoPE rotation의 *position index***일 뿐.
- 첫 token은 local anchor로 남기고 뒤의 index를 jump시켜 **temporal continuity를 깨뜨림**. 논문 Figure 5 예시는 `{4,5,6}→{4,20,21}`이며 이후 rollout에서 local index를 재사용. **scene cut 시 cache flush도 결합**(Appendix D).
- **KV Flush = action continuity 유지하며 prompt change / RoPE Cut = 의도적인 scene discontinuity**.

### (E) Figure 6 — 왜 이 시각화가 흥미로운가?

- `Q,K` shape `(B,N,H,D)` → head별 post-softmax **attention weights** `A` shape `(B,H,Nq,Nk)`. **학습된 frozen parameter가 아니라 forward-pass activation**. `A_ij`는 query `i`의 output에 key `j`에 대응하는 value `V_j`를 얼마나 가중할지를 나타냄.
- Fig. 6은 **13번째 DiT layer**의 head-averaged token attention을 spatial token들에 걸쳐 합친 **frame-to-frame heatmap**: y축=query frame, x축=key frame; 밝을수록 그 frame 쌍으로 더 큰 attention mass. 엄밀히 row-normalized 비율을 원하면 query spatial tokens 수로 나누는 정규화가 필요함.
- **(a) Diagonal band**: query가 자기 자신/최근 frame에 많이 attend → local temporal continuity. **밝은 첫 column**: 오래된 first-frame **attention sink**를 계속 attend.
- **(b) KV Flush**: 중간 과거 frame에 대한 attention이 제거/약화되고 **sink와 flush 직전 frame**으로 집중.
- **(c) RoPE Cut**: 전후 scene이 **거의 분리된 diagonal blocks**를 형성; 새 scene은 주로 자기 근방과 sink를 참고.
- **논문의 강점:** intervention의 의도와 관찰된 attention pattern을 직접 연결. **단서:** 이는 attention *allocation 시각화*이며 특정 head의 **causal mechanism 입증**까지는 아님. 이 틈이 내 후속 연구와 연결됨.

---

## 2. Pinar Yanardag — AdaState (2026)

[Paper](https://arxiv.org/abs/2605.30349) · [Project](https://adastate.github.io/)

### Infinity-RoPE와의 연결이 핵심

- Infinity-RoPE Figure 6에서는 **첫 frame/sink로의 persistent attention**을 관찰함. AdaState는 이 **같은 attention bias**를 다른 관점에서 활용: **“어차피 많이 attend하는 그 위치에 무엇을 보관할 것인가?”**
- **Static first-frame sink의 tradeoff:** 깨끗한 첫 장면은 consistency/identity를 돕지만, 계속 오래된 composition을 기준으로 삼아 **motion·camera movement·scene evolution을 억제**할 수 있음.
- AdaState의 해법은 **attention을 억지로 다른 곳으로 분산하는 것이 아니라**, 모델이 원래 많이 보는 **sink 슬롯의 내용을 매 chunk 진화시키는 것**.

### 실제 recurrence — `s_n`은 어떻게 만들어지나?

1. **Cache 입력**: 과거 clean anchor `KV(s_{n−1})`(sink 위치 0) + 새로 noise에서 시작하는 **live state** `\tilde{s}_n` + 최근 chunk들의 clean content KV + 현재 denoising 중인 noisy content `x_c^(n)`.
2. **Joint denoising**: `G_θ([ε_s, ε_c]; h_n) → [s_n, x̂_c^(n)]`. **state와 content 모두 노이즈에서 시작**하고, 같은 causal DiT가 **이전 state + 현재 content를 attend**하며 함께 denoise.
3. **서로 다른 출력 경로**: content `x̂_c^(n)`는 **video로 decode**되고 recent KV window에 들어감. **state `s_n`은 최종 비디오로 렌더링하지 않음**; 대신 **clean `KV(s_n)`을 sink position 0에 덮어써서** 다음 chunk로 넘김.
4. 다음 chunk에서 `s_n`이 이전 state가 되어 같은 과정 반복: **`s_{n−1} → (joint denoising with new content) → s_n`**. 별도의 외부 RNN/gate가 아니라 **denoiser가 transition, KV cache가 carrier**.

**정확한 해석:** `s_n`은 이전 video frame을 복사하거나 새 chunk hidden features를 단순 추출한 것이 아니라, **학습된 recurrent scene representation**. 현재 내용과 이전 상태를 통합할 수 있게 학습됨. 첫 chunk에서는 첫 clean content frame으로 `s_0`를 초기화하며, **eviction이 시작될 때 state 슬롯이 활성화**.

### Training과 relative time

- **Infinity-RoPE와 결정적 차이:** AdaState는 **training-free가 아님**. Self-Forcing backbone에서 **horizon-weighted DMD**로 fine-tuning; 뒤쪽 rollout frame에 더 큰 loss weight를 주어 accumulated drift와 state 사용을 학습. **state 전용 auxiliary loss는 없음**; content generation loss가 attention을 통해 state를 형성.
- Cached components는 **clean (`σ=0`)**, live state/content는 **현재 denoising noise level**. K는 cache에 unrotated로 저장하고 읽을 때 **block-relativistic RoPE**를 적용해 local temporal geometry를 일정하게 유지.
- **요점:** Infinity-RoPE가 *시간 좌표와 cache 사용법*을 고친다면, AdaState는 *sink 슬롯에 저장되는 scene memory의 내용*을 모델이 **스스로 갱신하도록 학습**함.

### SOP에서 제안한 다음 연구 (아직 검증되지 않은 가설)

- `s_n`에 **scene identity/consistency**와 **motion dynamics** 정보가 어떻게 나타나는지 head-level analysis 및 **causal intervention**으로 분해.
- 특정 head/state direction을 perturb했을 때 **identity 유지 vs motion·scene progression**의 변화가 있는지 비교.
- 조작 가능한 causal feature를 찾으면 **frozen video generator**를 유지한 채, action / scene transition 지시에 따라 `s_n`을 수정하는 **lightweight controller**를 학습해 control interface로 활용.
- **주의:** AdaState 논문이 이미 이런 causal head-level decomposition이나 controller를 입증했다는 뜻은 아님. 이것이 *내 연구 proposal*.

---

## 3. Chris Thomas — SteerVLM (Findings of EMNLP 2025)

[Paper](https://aclanthology.org/2025.findings-emnlp.1285/) · [arXiv](https://arxiv.org/abs/2510.26769)

### 논문 로직

- **목표:** Prompt engineering만으로 잘 제어되지 않는 **Vision-Language Model의 응답**을, backbone **weight를 freeze**한 채 activation-level intervention으로 조절.
- **Input:** 원하는 행동의 **target prompt `p+`** 및 대비되는 **converse prompt `p−`**의 hidden activations + 현재 VLM activation.
- **Steerer:** 가벼운 attention module이 어느 activation dimension을 증폭/억제할지 계산. **SteeringGate:** dimension별 intervention 강도를 조정. 여러 decoder layer에 공유 적용; steering strength `λ`로 조절.
- **수식(개념):** `z_l = x_l + λ · f(x_l, p+_l, p−_l)`. 원본 모델을 재학습하지 않고 **약 0.14% 규모의 추가 module**만 학습. 기존 fixed steering vector보다 token/layer에 적응.
- **정확한 범위:** 이 논문은 **VLM language decoder의 hidden activation**을 steer하는 것. Steerer 내부에 attention/cross-attention 계산이 있지만, **video diffusion의 text–video cross-attention 자체를 분석한 논문은 아님**.

### SOP에서 연결한 내 연구 질문

- **내 경험적 관찰:** video generation에서 text prompt는 **background·scene·object attributes**를 강하게 좌우함.
- **미검증 질문:** Motion control이 약한 이유가 prompt 표현 방식 때문인지, 아니면 **motion-related text information이 cross-attention을 통해 video latent에 전달·활용되는 방식** 때문인지?
- **후속 실험 아이디어:** motion/scene/object prompt를 통제하며 cross-attention의 token↔spatiotemporal latent 연결을 분석 → 관련 attention head/activation을 선택적으로 intervention → **motion controllability** 개선 여부와 background/identity 부작용 평가.
- Yanardag의 **temporal memory/state control**과 Thomas의 **semantic text-conditioning control**은 서로 보완적.

---

## 4. 인터뷰에서 기억할 구분 5가지

| 질문 | 핵심 답 |
|---|---|
| *Is RoPE mathematically limited to 1024?* | **No.** Wan의 1024는 precomputed table 길이; 학습에서 본 temporal 관계는 약 21 latent frames. 상대 거리 OOD가 더 실질적 문제. |
| *Does Block-Relativistic RoPE reuse old frames?* | **No.** Cached content는 유지, **temporal rotation index를 재매핑**. Cache sliding과 RoPE remap은 다름. |
| *Is unbounded cache truly infinite memory?* | **No.** 논문 용어상 training horizon보다 큰 cache. Semanticization은 **index collapse**, KV tensor 압축 아님. |
| *Why is AdaState different from Infinity-RoPE?* | Infinity-RoPE는 **inference-only position/cache intervention**, AdaState는 **training-based evolving anchor**. |
| *Did SteerVLM study video cross-attention?* | **No.** Frozen VLM decoder activation steering; 이를 **video text-conditioning/motion**으로 확장하는 것이 내 제안. |

## 5. 인터뷰용 20초 English talking points

**Yanardag:** “What interested me most was the connection between the papers. Infinity-RoPE visualizes persistent attention to the first-frame sink, while AdaState turns that same bias into an advantage by replacing the static anchor with an evolving state. I'd like to causally examine what the state encodes and whether it can become a controllable interface for motion and scene evolution.”

**Thomas:** “SteerVLM showed that prompt-derived activation interventions can control multimodal behavior. In my video experiments, prompts strongly affected scene and object attributes. I want to understand whether motion-related prompt information is underused in text–video cross-attention, and whether targeted interventions can improve motion control.”

---

### 원문 및 대조 자료

1. [Infinity-RoPE, §3.2 / §4.1–4.4 / Fig. 3·5·6 / App. D](https://arxiv.org/html/2511.20649)
2. [Infinity-RoPE GitHub Issue #2 — bounded vs unbounded cache](https://github.com/yesiltepe-hidir/infinity-rope/issues/2)
3. [AdaState, §3.1–3.3 / Fig. 2·3](https://arxiv.org/html/2605.30349)
4. [SteerVLM, §3 / Fig. 2](https://arxiv.org/html/2510.26769)
5. VT 제출용 `Academic and Professional Goals Statement` 2페이지 버전 및 이전 `[Accum] PI Search 3` 검토 대화.
