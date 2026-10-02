# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-95)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM, включая базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, Mixture of Experts (MoE), Retrieval-Augmented Generation (RAG), Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat Sampling, DRY Sampling, Min-K% Prob Sampling, XTC Sampling, интеграцию дополнительных фильтров (Min-K% Prob, DRY Sampling), интеграцию различных вероятностных фильтров (Top-K, Top-P, Min-P, сглаживание логитов, добавление гауссовского шума, Typical Sampling, TFS, Top-A, Epsilon Cutoff, Eta Cutoff, Top-N Tokens, Mirostat, Gumbel Noise, Dynamic Temperature Entropy) в механизм Continuous Batching, поддержку Logit Softcapping для генерации и обучения MDLM, и внедрение Stochastic Unmasking (Случайное размаскирование) для повышения вариативности.

## [x] Шаг 96: Внедрение Quantile Sampling в MDLM
**Цель:** Добавить поддержку Quantile Sampling (сэмплинг по квантилям) для генерации MDLM.
**Детали:**
- Добавить параметры `quantile_low` (float, по умолчанию 0.0) и `quantile_high` (float, по умолчанию 1.0) и соответствующие расписания в `DiffusionModelForConditionalGeneration.generate` и в датакласс `MDLMRequest`.
- В процессе сэмплинга после вычисления вероятностей токенов отсортировать их по убыванию, вычислить кумулятивную сумму и отфильтровать (замаскировать) токены, кумулятивная вероятность которых выходит за пределы `[quantile_low, quantile_high]`.
- Написать тесты для проверки корректности Quantile Sampling при базовой генерации, continuous batching и speculative decoding.
