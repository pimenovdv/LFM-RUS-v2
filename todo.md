# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-97)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM, включая базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, Mixture of Experts (MoE), Retrieval-Augmented Generation (RAG), Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat Sampling, DRY Sampling, Min-K% Prob Sampling, XTC Sampling, Quantile Sampling, интеграцию различных вероятностных фильтров в механизм Continuous Batching (включая Rank Penalty Sampling), поддержку Logit Softcapping и Stochastic Unmasking.

## [x] Шаг 98: Внедрение Thresholding Sampling в MDLM
**Цель:** Добавить Thresholding Sampling для отсечения токенов с вероятностью ниже заданного порога.
**Детали:**
- Добавить параметры `prob_threshold_val` (float, по умолчанию 0.0), `prob_threshold_schedule` (str, по умолчанию 'constant') и `min_prob_threshold_val` (float, по умолчанию 0.0) в `DiffusionModelForConditionalGeneration.generate` и датакласс `MDLMRequest`.
- При сэмплинге: вычислять вероятности через softmax и отсекать токены (устанавливать логиты в -inf), чьи вероятности строго меньше `prob_threshold_val`.
- Интегрировать логику в базовую генерацию и `MDLMContinuousBatchingManager`.
- Добавить тесты для проверки корректности Thresholding Sampling при генерации и batching.
- Выполнить pre-commit шаги.
