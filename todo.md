# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-79)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM, включая базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, Mixture of Experts (MoE), Retrieval-Augmented Generation (RAG), Continuous Time Diffusion, дистилляцию консистентности (Consistency Models Distillation), интеграцию FlashAttention-2 (Шаг 71), Latent Masked Diffusion (Шаг 72), Discrete Flow Matching (Шаг 73), Contrastive Decoding (Шаг 74), поддержку Self-Conditioning во время инференса (Шаг 75), во время обучения (Шаг 76), алгоритм сэмплирования Mirostat (Шаг 77), Min-K% Prob Sampling (Шаг 78) и Repetition Penalization Penalty по времени (Шаг 79).

## [x] Шаг 80: Добавление поддержки DRY (Don't Repeat Yourself) Sampling
**Цель:** Добавить поддержку DRY сэмплирования, которое предотвращает повторение длинных последовательностей токенов путем динамического регулирования штрафов на основе совпадений подстрок в контексте.

## [x] Шаг 81: Добавление расписаний для Min-P Sampling (min_p_schedule)
**Цель:** Расширить логику Min-P Sampling в `generate`, добавив динамические расписания (linear, cosine, exponential, cyclic) для `min_p`.
**Детали:**
- Добавление расписаний для Min-P Sampling позволит динамически управлять порогом отсечения токенов на разных этапах генерации.

## [x] Шаг 82: Добавление расписаний для Top-K Guidance (tkg_schedule)
**Цель:** Добавить поддержку динамических расписаний (linear, cosine, exponential, cyclic) для `tkg_scale` (Top-K Guidance).
**Детали:**
- В `generate` добавить логику изменения `tkg_scale` в зависимости от шага. Это позволит контролировать силу наведения по Top-K на разных этапах генерации (например, уменьшать к концу генерации для большей креативности).

## [x] Шаг 83: Добавление расписаний для XTC (Exclude Top Choices) Sampling
**Цель:** Внедрить расписания `xtc_threshold_schedule` и `xtc_probability_schedule` для XTC Sampling.
**Детали:**
- XTC Sampling исключает наиболее вероятные токены. Добавление расписаний (linear, cosine, exponential, cyclic) для порога отсечения и вероятности применения позволит динамически изменять агрессивность XTC Sampling в процессе генерации.

## [x] Шаг 84: Добавление расписаний для DRY (Don't Repeat Yourself) Sampling
**Цель:** Добавить поддержку динамических расписаний (linear, cosine, exponential, cyclic) для `dry_multiplier`.
**Детали:**
- Добавление расписания для `dry_multiplier` позволит динамически управлять силой штрафования за повторение длинных последовательностей в процессе генерации.

## [x] Шаг 85: Добавление расписаний для Mirostat Sampling
**Цель:** Добавить поддержку динамических расписаний (linear, cosine, exponential, cyclic) для `mirostat_tau` и `mirostat_eta` в алгоритме Mirostat.
**Детали:**
- В алгоритме Mirostat используются параметры `mirostat_tau` (целевая энтропия) и `mirostat_eta` (скорость обучения).
- Добавить параметры `mirostat_tau_schedule`, `min_mirostat_tau` и `mirostat_eta_schedule`, `min_mirostat_eta` в метод `generate`.
- Внутри цикла `steps`, обновлять `current_mirostat_tau` и `current_mirostat_eta` в зависимости от `step_ratio` (так же как это делается для `temperature` и других параметров).
- Это позволит управлять тем, насколько быстро Mirostat реагирует на неожиданность (surprisal) на разных этапах генерации (например, увеличивая целевую энтропию в конце или меняя learning rate).
