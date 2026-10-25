# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-118)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM. Включает базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, MoE, RAG, Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat, DRY Sampling, Min-K%, XTC, Quantile, Thresholding, вероятностные фильтры для Continuous Batching, Logit Softcapping, Stochastic Unmasking, Uniform/Laplacian/Cauchy/Student-T/Pareto/Logistic/Weibull/Rayleigh/Log-Normal/Exponential/Gamma/Beta/Von Mises/Dirichlet/Poisson Noise, Penalties, SmoothK, Positional Temperature, CAT, TDS, Rank-Based Sampling, Epsilon-Greedy, Logit Noising, Gumbel-Softmax Noise Injection, Tsallis Entropy Sampling, и Adaptive Masking Ratio Schedule.

## [ ] Шаг 119: Внедрение Guidance Rescale для Continuous Batching
**Цель:** Добавить динамическое масштабирование guidance rescale для запросов в непрерывном батчинге.
**Детали:**
- Добавить `guidance_rescale` в конфигурацию запросов.
- Интегрировать в `MDLMContinuousBatchingManager.step`.
- Создать тесты.

## [ ] Шаг 120: Внедрение Adaptive Token-Level Temperature
**Цель:** Температура, изменяющаяся на уровне отдельных токенов в зависимости от их энтропии в Continuous Batching.
**Детали:**
- Добавить параметры и логику расчета температуры для каждого токена.
- Интегрировать в цикл сэмплинга.
- Создать тесты.
