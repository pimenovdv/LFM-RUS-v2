# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-88)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM, включая базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, Mixture of Experts (MoE), Retrieval-Augmented Generation (RAG), Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat Sampling, DRY Sampling, Min-K% Prob Sampling, XTC Sampling, а также интеграцию различных вероятностных фильтров (Top-K, Top-P, Min-P, сглаживание логитов, добавление гауссовского шума) в механизм Continuous Batching.

## [x] Шаг 89: Поддержка Typical Sampling, Tail Free Sampling (TFS) и Top-A фильтров в Continuous Batching
**Цель:** Добавить поддержку вероятностных фильтров Typical Sampling, Tail Free Sampling (TFS) и Top-A для режима Continuous Batching.
**Детали:**
- В класс `MDLMRequest` добавить параметры `typical_p`, `typical_p_schedule`, `min_typical_p`, `tfs`, `tfs_schedule`, `min_tfs`, `top_a`, `top_a_schedule`, `min_top_a`.
- В методе `MDLMContinuousBatchingManager.step` рассчитывать их текущие значения на основе шага (`step_ratio`).
- Применять фильтры Typical Sampling, TFS и Top-A к логитам запроса (заменяя отфильтрованные значения на `-inf`), следуя алгоритмам, уже реализованным в основном методе `generate`.
- Написать тесты для проверки корректности применения Typical Sampling, TFS и Top-A в Continuous Batching.
