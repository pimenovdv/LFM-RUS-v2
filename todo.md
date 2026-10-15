# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-106)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM. Включает базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, MoE, RAG, Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat, DRY Sampling, Min-K%, XTC, Quantile, Thresholding, вероятностные фильтры для Continuous Batching, Logit Softcapping, Stochastic Unmasking, Uniform/Laplacian/Cauchy Noise, Penalties, SmoothK, Positional Temperature, Confidence-Aware Temperature (CAT), Token Drop Sampling (TDS), Rank-Based Sampling, Epsilon-Greedy Sampling, Student-T Noise Injection и Pareto Noise Injection Sampling.

## [ ] Шаг 107: Внедрение Logit Noising Sampling (Gaussian и Gumbel)
**Цель:** Добавить новые методы вероятностного зашумления логитов на основе Гауссова распределения (Gaussian Logit Noising) и распределения Гамбеля (Gumbel Logit Noising) для управления вариативностью генерации.
**Детали:**
- **Gaussian Logit Noising:** Внедрение шума из нормального (Гауссова) распределения напрямую в логиты.
  *Параметры:* `gaussian_logit_noise_scale` (float), `gaussian_logit_noise_schedule` (str), `min_gaussian_logit_noise_scale` (float).
- **Gumbel Logit Noising:** Добавление Гамбелевского шума к логитам независимо от основного `gumbel_temperature`, для более гибкого сэмплирования.
  *Параметры:* `gumbel_logit_noise_scale` (float), `gumbel_logit_noise_schedule` (str), `min_gumbel_logit_noise_scale` (float).
- Интеграция динамических расписаний в методы `generate` и `step` (continuous batching).
- Покрытие кода тестами.
