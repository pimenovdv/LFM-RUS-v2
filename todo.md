# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-122)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM. Включает базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, MoE, RAG, Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat, DRY Sampling, Min-K%, XTC, Quantile, Thresholding, вероятностные фильтры, Logit Softcapping, Stochastic Unmasking, Noise Injections, Penalties, SmoothK, Positional Temperature, CAT, TDS, Rank-Based Sampling, Epsilon-Greedy, Logit Noising, Gumbel-Softmax Noise Injection, Tsallis Entropy Sampling, Adaptive Masking Ratio Schedule, Length-Scaled Temperature, и Entropy-Guided Top-K Filtering.


## [ ] Шаг 123: Внедрение Adaptive Sequence-Length Guidance
**Цель:** Динамически масштабировать `cfg_scale` на основе сгенерированной длины, чтобы усиливать guidance к концу предложения.
**Детали:**
- Добавить параметры `length_cfg_scale`, `min_length_cfg_scale`, `length_cfg_schedule`.
- Применить логику `current_cfg_scale = current_cfg_scale * (1.0 + length_cfg_scale * step_ratio)`.
- Интегрировать в `modeling_diffusion.py` и создать тесты.

## [ ] Шаг 124: Внедрение Information Gain Masking
**Цель:** Маскировать токены не случайно, а на основе их информативности (Information Gain) относительно контекста.
**Детали:**
- Добавить стратегию `remasking="info_gain"`.
- Считать расхождение KL или другую метрику между условными и безусловными логитами.
- Интегрировать в `modeling_diffusion.py` и покрыть тестами.
