# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-135)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM. Включает базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, MoE, RAG, Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat, DRY Sampling, Min-K%, XTC, Quantile, Thresholding, вероятностные фильтры, Logit Softcapping, Stochastic Unmasking, Noise Injections (в т.ч. Chi-Square, Half-Normal, Fisher-Snedecor, Half-Cauchy, Beta Prime), Penalties, SmoothK, Positional Temperature, CAT, TDS, Rank-Based Sampling, Epsilon-Greedy, Logit Noising, Gumbel-Softmax Noise Injection, Tsallis Entropy Sampling, Adaptive Masking Ratio Schedule, Length-Scaled Temperature, Entropy-Guided Top-K Filtering, Adaptive Sequence-Length Guidance, Information Gain Masking, Adaptive Iterative Decoding (AID), Contrastive Masking, Harmonic Temperature Oscillation (HTO), Information-Guided Repetition Penalty (IGRP), Token-Level Entropic Smoothing (TLES) и Adaptive Iterative Denoising (AID) Momentum.

## [x] Шаг 137: Внедрение Gumbel-Max Noise Injection
**Цель:** Добавить возможность вносить шум из распределения Гумбеля (Gumbel-Max) в логиты перед сэмплированием.
**Детали:**
- Добавить параметры `gumbel_max_noise_scale`, `gumbel_max_noise_schedule`, `min_gumbel_max_noise_scale`, `gumbel_max_loc`, `gumbel_max_scale` в конфигурацию.
- Реализовать расписание масштаба шума в методе `generate` и менеджере continuous batching.
- Использовать Inverse Transform Sampling с равномерным распределением (`loc - scale * torch.log(-torch.log(u))`) для генерации шума Gumbel-Max.
- Покрыть новую функциональность тестами.
