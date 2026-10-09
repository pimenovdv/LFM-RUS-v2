# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-100)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM, включая базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, Mixture of Experts (MoE), Retrieval-Augmented Generation (RAG), Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat Sampling, DRY Sampling, Min-K% Prob Sampling, XTC Sampling, Quantile Sampling, Thresholding Sampling, интеграцию различных вероятностных фильтров в механизм Continuous Batching (включая Rank Penalty Sampling), поддержку Logit Softcapping, Stochastic Unmasking и Uniform Noise Sampling.

## [x] Шаг 101: Внедрение Repetition, Frequency и Presence Penalties в Continuous Batching
**Цель:** Добавить поддержку стандартных штрафов за повторение (`repetition_penalty`, `frequency_penalty`, `presence_penalty`) в механизм непрерывного батчинга для обеспечения функционального паритета со стандартной генерацией.
**Детали:**
- Добавить параметры `repetition_penalty`, `frequency_penalty`, `presence_penalty`, их динамические расписания, минимальные значения и параметр `penalty_range` в датакласс `MDLMRequest`.
- В методе `MDLMContinuousBatchingManager.step()` извлекать токены контекста без учета маски и применять штрафы к `req_logits`.
- Добавить тесты для проверки влияния добавленных штрафов.
