# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-94)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM, включая базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, Mixture of Experts (MoE), Retrieval-Augmented Generation (RAG), Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat Sampling, DRY Sampling, Min-K% Prob Sampling, XTC Sampling, интеграцию дополнительных фильтров (Min-K% Prob, DRY Sampling), а также интеграцию различных вероятностных фильтров (Top-K, Top-P, Min-P, сглаживание логитов, добавление гауссовского шума, Typical Sampling, TFS, Top-A, Epsilon Cutoff, Eta Cutoff, Top-N Tokens, Mirostat, Gumbel Noise, Dynamic Temperature Entropy) в механизм Continuous Batching, и поддержку Logit Softcapping для генерации и обучения MDLM.

## [x] Шаг 95: Внедрение Stochastic Unmasking (Случайное размаскирование) в MDLM
**Цель:** Добавить поддержку Stochastic Unmasking для генерации MDLM, чтобы повысить вариативность.
**Детали:**
- Добавить параметр `stochastic_unmasking` (bool, по умолчанию False) в `DiffusionModelForConditionalGeneration.generate` и в датакласс `MDLMRequest`.
- При вычислении уверенности для выбора токенов (`confidence`) добавлять случайный шум, прибавляя `torch.rand_like(...)` перед выбором (top-k), если `stochastic_unmasking=True`.
- Написать тесты для проверки генерации со случайным размаскированием.
