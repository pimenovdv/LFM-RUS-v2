# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-113)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM. Включает базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, MoE, RAG, Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat, DRY Sampling, Min-K%, XTC, Quantile, Thresholding, вероятностные фильтры для Continuous Batching, Logit Softcapping, Stochastic Unmasking, Uniform/Laplacian/Cauchy/Student-T/Pareto/Logistic/Weibull/Rayleigh/Log-Normal/Exponential/Gamma/Beta/Von Mises Noise, Penalties, SmoothK, Positional Temperature, CAT, TDS, Rank-Based Sampling, Epsilon-Greedy, и Logit Noising.

## [ ] Шаг 114: Внедрение Dirichlet Noise Injection Sampling
**Цель:** Добавить инъекцию шума из распределения Дирихле в confidence scores (logits) для маскированной диффузии.
**Детали:**
- Добавить параметры: `dirichlet_noise_scale` (float), `dirichlet_noise_schedule` (str), `min_dirichlet_noise_scale` (float), `dirichlet_noise_alpha` (float или list/tensor).
- Интегрировать в методы `generate` и непрерывный батчинг.
- Создать тесты.

## [ ] Шаг 115: Внедрение Poisson Noise Injection Sampling
**Цель:** Добавить инъекцию пуассоновского шума в confidence scores (logits) для маскированной диффузии.
**Детали:**
- Добавить параметры `poisson_noise_scale`, `poisson_noise_schedule`, `min_poisson_noise_scale`, `poisson_noise_rate`.
- Интегрировать в методы `generate` и непрерывный батчинг.
- Создать тесты.

## [ ] Шаг 116: Внедрение Gumbel-Softmax Noise Injection
**Цель:** Добавить инъекцию шума через репараметризацию Gumbel-Softmax.
**Детали:**
- Добавить параметры для Gumbel-Softmax (температура, масштаб, расписания).
- Интегрировать в сэмплинг.
- Создать тесты.
