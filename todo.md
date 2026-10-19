# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-109)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM. Включает базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, MoE, RAG, Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat, DRY Sampling, Min-K%, XTC, Quantile, Thresholding, вероятностные фильтры для Continuous Batching, Logit Softcapping, Stochastic Unmasking, Uniform/Laplacian/Cauchy/Student-T/Pareto/Logistic/Weibull/Rayleigh/Log-Normal Noise, Penalties, SmoothK, Positional Temperature, CAT, TDS, Rank-Based Sampling, Epsilon-Greedy, и Logit Noising.

## [x] Шаг 110: Внедрение Exponential Noise Injection Sampling
**Цель:** Добавить инъекцию шума из экспоненциального распределения в confidence scores (logits) для маскированной диффузии.
**Детали:**
- **Exponential Noise Injection:** Добавление шума из экспоненциального распределения.
  *Параметры:* `exponential_noise_scale` (float), `exponential_noise_schedule` (str), `min_exponential_noise_scale` (float).
- Поддержка динамических расписаний (linear, cosine, exponential, cyclic).
- Интеграция в метод `generate`.
- Интеграция в continuous batching (`MDLMRequest` и `MDLMContinuousBatchingManager.step()`).
- Покрытие тестами (проверка параметров и самого сэмплинга).

## [ ] Шаг 111: Внедрение Gamma Noise Injection Sampling
**Цель:** Добавить инъекцию шума из гамма-распределения в confidence scores (logits) для маскированной диффузии.

## [ ] Шаг 112: Внедрение Beta Noise Injection Sampling
**Цель:** Добавить инъекцию шума из бета-распределения в confidence scores (logits) для маскированной диффузии.

## [ ] Шаг 113: Внедрение Von Mises Noise Injection Sampling
**Цель:** Добавить инъекцию шума из распределения фон Мизеса в confidence scores (logits) для маскированной диффузии.
