# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-126)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM. Включает базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, MoE, RAG, Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat, DRY Sampling, Min-K%, XTC, Quantile, Thresholding, вероятностные фильтры, Logit Softcapping, Stochastic Unmasking, Noise Injections, Penalties, SmoothK, Positional Temperature, CAT, TDS, Rank-Based Sampling, Epsilon-Greedy, Logit Noising, Gumbel-Softmax Noise Injection, Tsallis Entropy Sampling, Adaptive Masking Ratio Schedule, Length-Scaled Temperature, Entropy-Guided Top-K Filtering, Adaptive Sequence-Length Guidance, Information Gain Masking, Adaptive Iterative Decoding (AID) и Contrastive Masking.

## [x] Шаг 127: Внедрение Harmonic Temperature Oscillation (HTO)
**Цель:** Циклическое изменение температуры генерации во время диффузионных шагов, чтобы чередовать фазы "исследования" (высокая температура) и "эксплуатации" (низкая температура).
**Детали:**
- Добавить параметры `temperature_oscillation_amplitude` (float) и `temperature_oscillation_frequency` (float) в `DiffusionConfig` и `MDLMRequest`.
- Итоговая температура: `T = base_temperature + amplitude * sin(frequency * pi * step_ratio)`.
- Интегрировать в `modeling_diffusion.py` и `MDLMContinuousBatchingManager`.
- Написать тесты.

## [ ] Шаг 128: Внедрение Information-Guided Repetition Penalty (IGRP)
**Цель:** Применять штраф за повторение только к токенам с высоким Information Gain (низкой уверенностью), или наоборот, чтобы лучше контролировать лексическое разнообразие.
**Детали:**
- Добавить параметры `igrp_scale` (float) и `igrp_schedule`.
- Модифицировать функцию расчета repetition_penalty.
- Покрыть тестами.

## [ ] Шаг 129: Внедрение Token-Level Entropic Smoothing (TLES)
**Цель:** Сглаживать логиты только для тех токенов, где исходное распределение имеет очень низкую энтропию (модель "слишком" уверена).
**Детали:**
- Добавить параметры `entropic_smoothing_alpha`.
- `temperature_per_token = 1.0 + entropic_smoothing_alpha / (entropy + 1e-5)`.
- Интегрировать в генерацию и покрыть тестами.
