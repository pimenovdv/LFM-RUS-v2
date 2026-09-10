# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-75)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM, включая базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, Mixture of Experts (MoE), Retrieval-Augmented Generation (RAG), Continuous Time Diffusion, дистилляцию консистентности (Consistency Models Distillation), интеграцию FlashAttention-2 (Шаг 71), Latent Masked Diffusion (Шаг 72), Discrete Flow Matching (Шаг 73), Contrastive Decoding (Шаг 74) и поддержку Self-Conditioning во время инференса (Шаг 75).

## [x] Шаг 76: Поддержка Joint Self-Conditioning во время обучения MDLM
**Цель:** Внедрить поддержку Self-Conditioning (самообусловливания) в процессе обучения (`def forward`) для `DiffusionModelForConditionalGeneration`. Это позволит модели обучаться извлекать выгоду из предсказаний на предыдущем шаге диффузии.
**Детали:**
- В методе `forward` класса `DiffusionModelForConditionalGeneration` при условии `use_self_conditioning=True` с вероятностью 50% (`torch.rand(1).item() > 0.5`) выполнять предварительный проход без вычисления градиентов (`torch.no_grad()`).
- В этом предварительном проходе вычислять `prev_logits = outputs.logits`.
- Затем проецировать `prev_logits` через `self.self_conditioning_proj(torch.softmax(prev_logits, dim=-1))` и добавлять полученные эмбеддинги к `inputs_embeds` для основного прохода, где уже будут считаться градиенты.
- Написать и обновить тесты в `tests/test_diffusion_self_conditioning.py`.
