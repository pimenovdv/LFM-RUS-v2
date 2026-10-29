# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-122)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM. Включает базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, MoE, RAG, Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat, DRY Sampling, Min-K%, XTC, Quantile, Thresholding, вероятностные фильтры, Logit Softcapping, Stochastic Unmasking, Noise Injections, Penalties, SmoothK, Positional Temperature, CAT, TDS, Rank-Based Sampling, Epsilon-Greedy, Logit Noising, Gumbel-Softmax Noise Injection, Tsallis Entropy Sampling, Adaptive Masking Ratio Schedule, Length-Scaled Temperature, и Entropy-Guided Top-K Filtering.


## [x] Шаг 123: Внедрение Adaptive Sequence-Length Guidance
**Цель:** Динамически масштабировать `cfg_scale` на основе сгенерированной длины, чтобы усиливать guidance к концу предложения.
**Детали:**
- Добавить параметры `length_cfg_scale`, `min_length_cfg_scale`, `length_cfg_schedule`.
- Применить логику `current_cfg_scale = current_cfg_scale * (1.0 + length_cfg_scale * step_ratio)`.
- Интегрировать в `modeling_diffusion.py` и создать тесты.

## [x] Шаг 124: Внедрение Information Gain Masking
**Цель:** Маскировать токены не случайно, а на основе их информативности (Information Gain) относительно контекста.
**Детали:**
- Добавить стратегию `remasking="info_gain"`.
- Считать расхождение KL или другую метрику между условными и безусловными логитами.
- Интегрировать в `modeling_diffusion.py` и покрыть тестами.

## [x] Шаг 125: Внедрение Adaptive Iterative Decoding (AID)
**Цель:** Динамически изменять количество шагов диффузии (и масок за шаг) в зависимости от уверенности модели, чтобы ускорить генерацию простых токенов.
**Детали:**
- Добавить параметры `use_aid` (bool) и `aid_threshold` (float).
- Если средняя уверенность (или энтропия) на текущем шаге превышает `aid_threshold`, увеличивать количество снимаемых масок (уменьшать оставшиеся шаги).
- Реализовать в `modeling_diffusion.py` и написать тесты.

## [x] Шаг 126: Внедрение Contrastive Masking
**Цель:** Улучшить качество генерации, штрафуя токены, которые предсказываются любительской (amateur) моделью, непосредственно во время принятия решения о снятии маски.
**Детали:**
- Использовать `contrastive_alpha` и `amateur_model` при вычислении `confidence` (а не только при сэмплинге конечных токенов).
- Реализовать логику `confidence = main_confidence - contrastive_alpha * amateur_confidence`.
- Интегрировать в `modeling_diffusion.py` и покрыть тестами.
