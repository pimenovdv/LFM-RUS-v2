# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-100)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM, включая базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, Mixture of Experts (MoE), Retrieval-Augmented Generation (RAG), Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat Sampling, DRY Sampling, Min-K% Prob Sampling, XTC Sampling, Quantile Sampling, Thresholding Sampling, интеграцию различных вероятностных фильтров в механизм Continuous Batching (включая Rank Penalty Sampling), поддержку Logit Softcapping, Stochastic Unmasking, Uniform Noise Sampling, а также интеграцию Repetition, Frequency и Presence Penalties в механизм непрерывного батчинга.

## [x] Шаг 102: Внедрение SmoothK Sampling и Positional Temperature в стандартную генерацию и Continuous Batching
**Цель:** Добавить новые методы для повышения вариативности и управляемости генерации.
**Детали:**
- **SmoothK Sampling** позволяет применять мягкий штраф к токенам вне top-k вместо жесткого отсечения (путем вычитания `smooth_k_alpha` из логитов). Добавить параметры `smooth_k`, `smooth_k_alpha`, `smooth_k_schedule`, `min_smooth_k_alpha`.
- **Positional Temperature** позволяет линейно масштабировать температуру в зависимости от позиции токена в генерируемой последовательности. Добавить параметры `positional_temperature`, `positional_temperature_schedule`, `min_positional_temperature`.
- Интегрировать новые функции в `MDLMContinuousBatchingManager.step()` и `DiffusionModelForConditionalGeneration.generate()`.
- Покрыть новые возможности тестами в `tests/test_diffusion_smooth_k_positional_temp.py`.
