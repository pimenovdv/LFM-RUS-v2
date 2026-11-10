# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-134)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM. Включает базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, MoE, RAG, Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat, DRY Sampling, Min-K%, XTC, Quantile, Thresholding, вероятностные фильтры, Logit Softcapping, Stochastic Unmasking, Noise Injections (в т.ч. Chi-Square, Half-Normal, Fisher-Snedecor, Half-Cauchy), Penalties, SmoothK, Positional Temperature, CAT, TDS, Rank-Based Sampling, Epsilon-Greedy, Logit Noising, Gumbel-Softmax Noise Injection, Tsallis Entropy Sampling, Adaptive Masking Ratio Schedule, Length-Scaled Temperature, Entropy-Guided Top-K Filtering, Adaptive Sequence-Length Guidance, Information Gain Masking, Adaptive Iterative Decoding (AID), Contrastive Masking, Harmonic Temperature Oscillation (HTO), Information-Guided Repetition Penalty (IGRP), Token-Level Entropic Smoothing (TLES) и Adaptive Iterative Denoising (AID) Momentum.

## [x] Шаг 135: Внедрение Beta Prime Noise Injection
**Цель:** Добавить возможность вносить шум из распределения Бета Прайм (Beta Prime) в логиты перед сэмплированием для контроля разнообразия генерации.
**Детали:**
- Добавить параметры `beta_prime_noise_scale`, `beta_prime_noise_schedule`, `beta_prime_alpha`, `beta_prime_beta`, `min_beta_prime_noise_scale` в конфигурацию.
- Реализовать расписание масштаба шума в методе `generate` и менеджере continuous batching.
- Использовать `torch.distributions.beta.Beta` (с трансформацией `X / (1 - X)`) для генерации шума.
- Покрыть новую функциональность тестами.
