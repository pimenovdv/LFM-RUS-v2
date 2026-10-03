# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-96)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM, включая базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, Mixture of Experts (MoE), Retrieval-Augmented Generation (RAG), Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat Sampling, DRY Sampling, Min-K% Prob Sampling, XTC Sampling, Quantile Sampling, интеграцию различных вероятностных фильтров в механизм Continuous Batching, поддержку Logit Softcapping и Stochastic Unmasking.

## [ ] Шаг 97: Внедрение Rank Penalty Sampling в MDLM
**Цель:** Добавить поддержку Rank Penalty Sampling (пенализация на основе ранга токена) для генерации MDLM.
**Детали:**
- Добавить параметры `rank_penalty` (float, по умолчанию 0.0), `rank_penalty_schedule` (str, по умолчанию 'constant') и `min_rank_penalty` (float, по умолчанию 0.0) в `DiffusionModelForConditionalGeneration.generate` и датакласс `MDLMRequest`.
- При сэмплинге: вычислять ранги токенов (1 для самого вероятного и т.д.) и вычитать из логитов значение `rank_penalty * log(rank)`.
- Интегрировать логику в базовую генерацию и `MDLMContinuousBatchingManager`.
- Добавить тесты для проверки корректности Rank Penalty Sampling при генерации и batching.
- Выполнить pre-commit шаги.
