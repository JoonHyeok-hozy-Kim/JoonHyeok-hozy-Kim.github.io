### Direction 1 — Understanding generative models

Question: How does generative computation emerge and evolve over sampling time?

→ diffusion grokking
→ sampling-time dynamics 발견
→ probe의 explanatory limitation
→ realistic image-generation settings에서 더 principled quantity를 찾고 싶음

### Direction 2 — Improving generative models

Question: How should generative models be trained when the states they encounter during generation differ from those seen during training?

→ video generation
→ teacher forcing의 distribution mismatch / π-Flow
→ self-forcing + DMD
→ 여기서 네가 실제로 발견하게 될 limitation
→ 더 principled distillation / on-policy generative training / trajectory-aware objective 같은 future question