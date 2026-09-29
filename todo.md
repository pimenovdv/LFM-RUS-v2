# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-93)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM, включая базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, Mixture of Experts (MoE), Retrieval-Augmented Generation (RAG), Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat Sampling, DRY Sampling, Min-K% Prob Sampling, XTC Sampling, интеграцию дополнительных фильтров (Min-K% Prob, DRY Sampling), а также интеграцию различных вероятностных фильтров (Top-K, Top-P, Min-P, сглаживание логитов, добавление гауссовского шума, Typical Sampling, TFS, Top-A, Epsilon Cutoff, Eta Cutoff, Top-N Tokens, Mirostat, Gumbel Noise, Dynamic Temperature Entropy) в механизм Continuous Batching.

## [x] Шаг 94: Внедрение Logit Softcapping в MDLM
**Цель:** Добавить поддержку Logit Softcapping для генерации и обучения MDLM, как в Gemma/Grok.
**Детали:**
- Добавить параметры `logit_softcapping` (float, по умолчанию 0.0) в `DiffusionConfig`.
- В `DiffusionModelForConditionalGeneration.forward` применять `logits = (logits / logit_softcapping).tanh() * logit_softcapping` если `logit_softcapping > 0.0`.
- Написать тесты для проверки применения Logit Softcapping во время forward pass.
