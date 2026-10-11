# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-102)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM, включая базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, Mixture of Experts (MoE), Retrieval-Augmented Generation (RAG), Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat Sampling, DRY Sampling, Min-K% Prob Sampling, XTC Sampling, Quantile Sampling, Thresholding Sampling, интеграцию различных вероятностных фильтров в механизм Continuous Batching, поддержку Logit Softcapping, Stochastic Unmasking, Uniform Noise Sampling, Repetition/Frequency/Presence Penalties, SmoothK Sampling и Positional Temperature.

## [x] Шаг 103: Внедрение Confidence-Aware Temperature (CAT) и Token Drop Sampling (TDS)
**Цель:** Расширить возможности сэмплинга за счет адаптивного температурного контроля и стохастического ограничения словаря.
**Детали:**
- **Confidence-Aware Temperature (CAT):** Динамически изменяет температуру (умножает логиты) в зависимости от максимальной вероятности в распределении. Если модель слишком не уверена (max prob < `cat_min_prob`), логиты умножаются на `cat_sharpen_factor`. Если слишком уверена (max prob > `cat_max_prob`), логиты умножаются на `cat_flatten_factor`.
  *Параметры:* `cat_min_prob`, `cat_max_prob`, `cat_sharpen_factor`, `cat_flatten_factor`, `cat_schedule`, `min_cat_sharpen_factor`, `min_cat_flatten_factor`.
- **Token Drop Sampling (TDS):** Случайным образом исключает часть словаря (задает логитам -inf) с вероятностью `token_drop_prob`, стимулируя модель к использованию альтернативных токенов (исключая токен с максимальным логитом для стабильности).
  *Параметры:* `token_drop_prob`, `token_drop_schedule`, `min_token_drop_prob`.
- Интегрировать новые функции в `MDLMContinuousBatchingManager.step()` и `DiffusionModelForConditionalGeneration.generate()`.
- Покрыть новые возможности тестами в `tests/test_diffusion_cat_tds.py`.
