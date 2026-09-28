# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-92)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM, включая базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, Mixture of Experts (MoE), Retrieval-Augmented Generation (RAG), Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat Sampling, DRY Sampling, Min-K% Prob Sampling, XTC Sampling, интеграцию дополнительных фильтров (Min-K% Prob, DRY Sampling), а также интеграцию различных вероятностных фильтров (Top-K, Top-P, Min-P, сглаживание логитов, добавление гауссовского шума, Typical Sampling, TFS, Top-A, Epsilon Cutoff, Eta Cutoff, Top-N Tokens, Mirostat, Gumbel Noise) в механизм Continuous Batching.

## [x] Шаг 93: Поддержка Dynamic Temperature Entropy в Continuous Batching
**Цель:** Добавить поддержку Dynamic Temperature Entropy (DTE) для режима Continuous Batching.
**Детали:**
- В класс `MDLMRequest` добавить параметры `dynamic_temperature_entropy`, `dynamic_temperature_entropy_schedule`, `min_dynamic_temperature_entropy`.
- В `MDLMContinuousBatchingManager.step` внедрить логику изменения `temperature` в зависимости от энтропии логитов.
- Написать и запустить тесты для проверки корректности применения.
