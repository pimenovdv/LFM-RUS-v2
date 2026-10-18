# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-108)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM. Включает базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, MoE, RAG, Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat, DRY Sampling, Min-K%, XTC, Quantile, Thresholding, вероятностные фильтры для Continuous Batching, Logit Softcapping, Stochastic Unmasking, Uniform/Laplacian/Cauchy/Student-T/Pareto/Logistic/Weibull Noise, Penalties, SmoothK, Positional Temperature, CAT, TDS, Rank-Based Sampling, Epsilon-Greedy, и Logit Noising.

## [x] Шаг 109: Внедрение Rayleigh и Log-Normal Noise Injection Sampling
**Цель:** Добавить инъекцию шума из распределения Рэлея и Лог-нормального распределения в confidence scores для маскированной диффузии.
**Детали:**
- **Rayleigh Noise Injection:** Добавление шума из распределения Рэлея.
  *Параметры:* `rayleigh_noise_scale` (float), `rayleigh_noise_schedule` (str), `min_rayleigh_noise_scale` (float).
- **Log-Normal Noise Injection:** Добавление шума из Лог-нормального распределения.
  *Параметры:* `log_normal_noise_scale` (float), `log_normal_noise_schedule` (str), `min_log_normal_noise_scale` (float), `log_normal_noise_mean` (float).
- Поддержка динамических расписаний.
- Интеграция в `generate` и continuous batching (`MDLMRequest` и `step`).
- Покрытие тестами.
