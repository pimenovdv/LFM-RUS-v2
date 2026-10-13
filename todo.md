# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-104)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM. Включает базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, MoE, RAG, Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat, DRY Sampling, Min-K%, XTC, Quantile, Thresholding, вероятностные фильтры для Continuous Batching, Logit Softcapping, Stochastic Unmasking, Uniform Noise, Penalties, SmoothK, Positional Temperature, Confidence-Aware Temperature (CAT), Token Drop Sampling (TDS), Rank-Based Sampling и Epsilon-Greedy Sampling.

## [x] Шаг 105: Внедрение Laplacian Noise Injection и Cauchy Noise Injection Sampling
**Цель:** Добавить новые вероятностные фильтры для внедрения шума с более тяжелыми хвостами, стимулирующие креативность и вариативность при генерации.
**Детали:**
- **Laplacian Noise Injection Sampling:** Добавление шума из распределения Лапласа к логитам.
  *Параметры:* `laplacian_noise_scale` (float), `laplacian_noise_schedule` (str), `min_laplacian_noise_scale` (float).
- **Cauchy Noise Injection Sampling:** Добавление шума из распределения Коши к логитам, обеспечивающее экстремальные выбросы.
  *Параметры:* `cauchy_noise_scale` (float), `cauchy_noise_schedule` (str), `min_cauchy_noise_scale` (float).
- Интеграция динамических расписаний (linear, cosine, exponential, cyclic) в методы `generate` и `step` (continuous batching).
- Покрытие тестами.
