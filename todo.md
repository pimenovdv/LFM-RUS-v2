# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-98)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM, включая базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, Mixture of Experts (MoE), Retrieval-Augmented Generation (RAG), Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat Sampling, DRY Sampling, Min-K% Prob Sampling, XTC Sampling, Quantile Sampling, Thresholding Sampling, интеграцию различных вероятностных фильтров в механизм Continuous Batching (включая Rank Penalty Sampling), поддержку Logit Softcapping и Stochastic Unmasking.

## [x] Шаг 99: Внедрение Uniform Noise Sampling в MDLM
**Цель:** Добавить поддержку добавления равномерного шума к логитам перед сэмплингом, что может быть полезно для улучшения разнообразия генерации и исследования других распределений.
**Детали:**
- Добавить параметры `uniform_noise_scale` (float, по умолчанию 0.0), `uniform_noise_schedule` (str, по умолчанию 'constant') и `min_uniform_noise_scale` (float, по умолчанию 0.0) в метод `DiffusionModelForConditionalGeneration.generate` и датакласс `MDLMRequest`.
- При генерации и в механизме `MDLMContinuousBatchingManager`: перед этапом сэмплинга (но после штрафов и до XTC/Top-P/Top-K и т.д.) добавлять к логитам равномерный шум из распределения U(-scale, scale), где `scale` — текущее значение `uniform_noise_scale`. Если scale = 0, шум не добавляется.
- Добавить тесты для проверки влияния Uniform Noise Sampling при стандартной генерации и непрерывном батчинге.
- Выполнить pre-commit шаги.

## [ ] Шаг 100: Внедрение Tail Free Sampling (TFS) Schedule Scaling
**Цель:** Добавить более гибкую настройку TFS за счет масштабирования.
**Детали:**
- Опционально (будет зависеть от результатов Uniform Noise).
