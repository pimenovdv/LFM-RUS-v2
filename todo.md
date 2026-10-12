# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-103)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM. Включает базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, MoE, RAG, Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat, DRY Sampling, Min-K%, XTC, Quantile, Thresholding, вероятностные фильтры для Continuous Batching, Logit Softcapping, Stochastic Unmasking, Uniform Noise, Penalties, SmoothK, Positional Temperature, Confidence-Aware Temperature (CAT) и Token Drop Sampling (TDS).

## [x] Шаг 104: Внедрение Rank-Based Sampling и Epsilon-Greedy Sampling
**Цель:** Добавить новые методы сэмплинга, расширяющие возможности генерации и контролируемого исследования.
**Детали:**
- **Rank-Based Sampling:** Метод сэмплинга, при котором вероятности токенов вычисляются исключительно на основе их ранга, а не значений логитов. Вероятность $p \propto 1 / (rank^{\alpha})$.
  *Параметры:* `rank_based_sampling_alpha` (float), `rank_based_sampling_schedule` (str), `min_rank_based_sampling_alpha` (float).
- **Epsilon-Greedy Sampling:** С вероятностью `epsilon` происходит случайный (uniform) выбор токена, а с вероятностью `1 - epsilon` — жадный выбор (argmax).
  *Параметры:* `epsilon_greedy_prob` (float), `epsilon_greedy_schedule` (str), `min_epsilon_greedy_prob` (float).
- Интеграция в `DiffusionModelForConditionalGeneration.generate()` и `MDLMContinuousBatchingManager.step()`.
- Покрытие тестами.
