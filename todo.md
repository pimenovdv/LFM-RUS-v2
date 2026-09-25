# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-89)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM, включая базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, Mixture of Experts (MoE), Retrieval-Augmented Generation (RAG), Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat Sampling, DRY Sampling, Min-K% Prob Sampling, XTC Sampling, а также интеграцию различных вероятностных фильтров (Top-K, Top-P, Min-P, сглаживание логитов, добавление гауссовского шума, Typical Sampling, TFS, Top-A) в механизм Continuous Batching.

## [x] Шаг 90: Поддержка Epsilon Cutoff, Eta Cutoff, Top-N Tokens и Mirostat Sampling фильтров в Continuous Batching
**Цель:** Добавить поддержку дополнительных вероятностных фильтров (Epsilon Cutoff, Eta Cutoff, Top-N Tokens и Mirostat Sampling) для режима Continuous Batching в MDLM.
**Детали:**
- В класс `MDLMRequest` добавить параметры для `epsilon_cutoff`, `eta_cutoff`, `top_n_tokens`, `mirostat_mode`, `mirostat_tau`, `mirostat_eta`, `mirostat_mu` и их расписаний.
- В методе `MDLMContinuousBatchingManager.step` рассчитывать текущие значения параметров и применять соответствующие фильтры к логитам, следуя алгоритмам из основного метода `generate`.
- Обновлять `mirostat_mu` после выбора токена, если `mirostat_mode > 0`.
- Написать тесты для проверки корректности работы Epsilon Cutoff, Eta Cutoff, Top-N Tokens и Mirostat Sampling в Continuous Batching.
