# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-79)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM, включая базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, Mixture of Experts (MoE), Retrieval-Augmented Generation (RAG), Continuous Time Diffusion, дистилляцию консистентности (Consistency Models Distillation), интеграцию FlashAttention-2 (Шаг 71), Latent Masked Diffusion (Шаг 72), Discrete Flow Matching (Шаг 73), Contrastive Decoding (Шаг 74), поддержку Self-Conditioning во время инференса (Шаг 75), во время обучения (Шаг 76), алгоритм сэмплирования Mirostat (Шаг 77), Min-K% Prob Sampling (Шаг 78) и Repetition Penalization Penalty по времени (Шаг 79).

## [x] Шаг 80: Добавление поддержки DRY (Don't Repeat Yourself) Sampling
**Цель:** Добавить поддержку DRY сэмплирования, которое предотвращает повторение длинных последовательностей токенов путем динамического регулирования штрафов на основе совпадений подстрок в контексте.

## [ ] Шаг 81: Добавление расписаний для Min-P Sampling (min_p_schedule)
**Цель:** Расширить логику Min-P Sampling в `generate`, добавив динамические расписания (linear, cosine, exponential, cyclic) для `min_p`.
**Детали:**
- Добавление расписаний для Min-P Sampling позволит динамически управлять порогом отсечения токенов на разных этапах генерации.
