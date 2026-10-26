# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-120)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM. Включает базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching (с поддержкой Guidance Rescale и Adaptive Token-Level Temperature), Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, MoE, RAG, Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat, DRY Sampling, Min-K%, XTC, Quantile, Thresholding, вероятностные фильтры, Logit Softcapping, Stochastic Unmasking, Noise Injections, Penalties, SmoothK, Positional Temperature, CAT, TDS, Rank-Based Sampling, Epsilon-Greedy, Logit Noising, Gumbel-Softmax Noise Injection, Tsallis Entropy Sampling, и Adaptive Masking Ratio Schedule.

## [ ] Шаг 121: Внедрение Length-Scaled Temperature
**Цель:** Масштабировать базовую температуру в зависимости от текущей длины сгенерированной последовательности.
**Детали:**
- Добавить параметры `length_temperature_scale`, `length_temperature_schedule` в `MDLMRequest` и конфигурацию.
- Внедрить логику расчета: `current_temperature *= (1.0 + length_temperature_scale * (current_seq_len / max_new_tokens))` в `modeling_diffusion.py`.
- Создать тесты.

## [ ] Шаг 122: Внедрение Entropy-Guided Top-K Filtering
**Цель:** Динамически адаптировать K в Top-K в зависимости от энтропии распределения вероятностей токенов.
**Детали:**
- Добавить параметры `entropy_top_k_scale`, `min_entropy_top_k`.
- Интегрировать логику в `modeling_diffusion.py`: `k = max(min_entropy_top_k, int(base_k * (1.0 + entropy * entropy_top_k_scale)))`.
- Создать тесты.
