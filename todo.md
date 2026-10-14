# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-105)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM. Включает базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, MoE, RAG, Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat, DRY Sampling, Min-K%, XTC, Quantile, Thresholding, вероятностные фильтры для Continuous Batching, Logit Softcapping, Stochastic Unmasking, Uniform/Laplacian/Cauchy Noise, Penalties, SmoothK, Positional Temperature, Confidence-Aware Temperature (CAT), Token Drop Sampling (TDS), Rank-Based Sampling и Epsilon-Greedy Sampling.

## [ ] Шаг 106: Внедрение Student-T Noise Injection и Pareto Noise Injection Sampling
**Цель:** Добавить новые вероятностные фильтры на основе распределений Стьюдента и Парето для управления генерацией и добавления тяжелого шума.
**Детали:**
- **Student-T Noise Injection Sampling:** Внедрение шума из t-распределения Стьюдента.
  *Параметры:* `student_t_noise_scale` (float), `student_t_df` (float), `student_t_noise_schedule` (str), `min_student_t_noise_scale` (float).
- **Pareto Noise Injection Sampling:** Внедрение шума из распределения Парето.
  *Параметры:* `pareto_noise_scale` (float), `pareto_alpha` (float), `pareto_noise_schedule` (str), `min_pareto_noise_scale` (float).
- Интеграция динамических расписаний в методы `generate` и `step` (continuous batching).
- Покрытие кода тестами.
