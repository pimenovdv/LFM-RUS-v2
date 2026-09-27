# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-90)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM, включая базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, Mixture of Experts (MoE), Retrieval-Augmented Generation (RAG), Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat Sampling, DRY Sampling, Min-K% Prob Sampling, XTC Sampling, интеграцию дополнительных фильтров (Min-K% Prob, DRY Sampling), а также интеграцию различных вероятностных фильтров (Top-K, Top-P, Min-P, сглаживание логитов, добавление гауссовского шума, Typical Sampling, TFS, Top-A, Epsilon Cutoff, Eta Cutoff, Top-N Tokens, Mirostat) в механизм Continuous Batching.

## [x] Шаг 92: Поддержка Gumbel Noise в Continuous Batching
**Цель:** Добавить поддержку Gumbel Noise Injection (Gumbel Temperature) для режима Continuous Batching.
**Детали:**
- В класс `MDLMRequest` добавить параметры `gumbel_temperature`, `gumbel_temperature_schedule`, `min_gumbel_temperature`.
- В `MDLMContinuousBatchingManager.step` внедрить логику добавления шума Гамбеля к логитам в лог-масштабе.
- Написать и запустить тесты для проверки корректности применения.