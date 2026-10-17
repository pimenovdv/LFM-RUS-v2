# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-106)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM. Включает базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, MoE, RAG, Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat, DRY Sampling, Min-K%, XTC, Quantile, Thresholding, вероятностные фильтры для Continuous Batching, Logit Softcapping, Stochastic Unmasking, Uniform/Laplacian/Cauchy Noise, Penalties, SmoothK, Positional Temperature, Confidence-Aware Temperature (CAT), Token Drop Sampling (TDS), Rank-Based Sampling, Epsilon-Greedy Sampling, Student-T Noise Injection и Pareto Noise Injection Sampling, and Gaussian/Gumbel Logit Noising.


## [x] Шаг 108: Внедрение Logistic и Weibull Noise Injection Sampling
**Цель:** Добавить инъекцию шума из Логистического распределения и распределения Вейбулла в confidence scores для маскированной диффузии.
**Детали:**
- **Logistic Noise Injection:** Добавление шума из логистического распределения.
  *Параметры:* `logistic_noise_scale` (float), `logistic_noise_schedule` (str), `min_logistic_noise_scale` (float).
- **Weibull Noise Injection:** Добавление шума из распределения Вейбулла.
  *Параметры:* `weibull_noise_scale` (float), `weibull_noise_shape` (float), `weibull_noise_schedule` (str), `min_weibull_noise_scale` (float).
- Поддержка динамических расписаний.
- Интеграция в `generate` и continuous batching (`MDLMRequest` и `step`).
- Покрытие тестами.
