# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-90)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM, включая базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, Mixture of Experts (MoE), Retrieval-Augmented Generation (RAG), Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat Sampling, DRY Sampling, Min-K% Prob Sampling, XTC Sampling, а также интеграцию различных вероятностных фильтров (Top-K, Top-P, Min-P, сглаживание логитов, добавление гауссовского шума, Typical Sampling, TFS, Top-A, Epsilon Cutoff, Eta Cutoff, Top-N Tokens, Mirostat) в механизм Continuous Batching.

## [x] Шаг 91: Поддержка XTC Sampling, Min-K% Prob Sampling и DRY Sampling фильтров в Continuous Batching
**Цель:** Добавить поддержку оставшихся фильтров сэмплирования, реализованных в основном методе `generate`, для режима Continuous Batching.
**Детали:**
- В класс `MDLMRequest` добавить параметры `xtc_threshold`, `xtc_probability`, `cutoff_min_percent`, `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_sequence_breakers`, `original_prompt_len` и их расписания.
- В `MDLMContinuousBatchingManager.add_request` прокинуть эти новые аргументы, вычисляя `original_prompt_len` как `input_ids.shape[-1]`.
- В `MDLMContinuousBatchingManager.step` внедрить логику XTC Sampling, Min-K% Prob Sampling и DRY Sampling.
- Написать и запустить тесты для проверки корректности применения новых фильтров в Continuous Batching.
