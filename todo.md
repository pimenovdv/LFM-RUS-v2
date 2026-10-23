# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-115)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM. Включает базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, MoE, RAG, Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat, DRY Sampling, Min-K%, XTC, Quantile, Thresholding, вероятностные фильтры для Continuous Batching, Logit Softcapping, Stochastic Unmasking, Uniform/Laplacian/Cauchy/Student-T/Pareto/Logistic/Weibull/Rayleigh/Log-Normal/Exponential/Gamma/Beta/Von Mises/Dirichlet/Poisson Noise, Penalties, SmoothK, Positional Temperature, CAT, TDS, Rank-Based Sampling, Epsilon-Greedy, и Logit Noising.

## [x] Шаг 116: Внедрение Gumbel-Softmax Noise Injection
**Цель:** Добавить инъекцию шума через репараметризацию Gumbel-Softmax.
**Детали:**
- Добавить параметры для Gumbel-Softmax (температура, масштаб, расписания).
- Интегрировать в сэмплинг (generate) и непрерывный батчинг.
- Создать тесты.

## [ ] Шаг 117: Внедрение Tsallis Entropy Sampling
**Цель:** Добавить динамическую настройку температуры на основе обобщенной энтропии Тсаллиса для управления генерацией.
**Детали:**
- Добавить параметры `tsallis_temperature_entropy` (масштаб), `tsallis_q` (параметр неэкстенсивности), расписания.
- Изменять температуру пропорционально энтропии Тсаллиса: $T_{new} = T_{base} + \alpha * S_q(p)$.
- Интегрировать в `generate` и непрерывный батчинг.
- Создать тесты.

## [ ] Шаг 118: Внедрение Adaptive Masking Ratio Schedule
**Цель:** Динамически изменять количество размаскируемых токенов на основе текущей уверенности модели (confidence).
**Детали:**
- Добавить метод `adaptive_masking` в `unmasking_schedule`.
- Использовать среднюю уверенность для определения шага: если модель уверена, размаскировать больше токенов; если нет - меньше (сохраняя бюджет шагов).
- Интегрировать в цикл сэмплинга и непрерывный батчинг.
- Создать тесты.
