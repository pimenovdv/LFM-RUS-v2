# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-77)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM, включая базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, Mixture of Experts (MoE), Retrieval-Augmented Generation (RAG), Continuous Time Diffusion, дистилляцию консистентности (Consistency Models Distillation), интеграцию FlashAttention-2 (Шаг 71), Latent Masked Diffusion (Шаг 72), Discrete Flow Matching (Шаг 73), Contrastive Decoding (Шаг 74), поддержку Self-Conditioning во время инференса (Шаг 75), во время обучения (Шаг 76) и алгоритм сэмплирования Mirostat (Шаг 77).

## [x] Шаг 78: Добавление поддержки Min-K% Prob Sampling (cutoff_min_percent)
**Цель:** Добавить новую стратегию сэмплирования Min-K% Prob Sampling в метод `generate` для уменьшения галлюцинаций и улучшения когерентности путем отсечения K% наименее вероятных токенов.
**Детали:**
- Добавлены параметры `cutoff_min_percent: float = 0.0`, `cutoff_min_percent_schedule: str = "constant"`, и `min_cutoff_min_percent: float = 0.0` в сигнатуру функции `generate`.
- В процессе генерации на каждом шаге отсекаются `k` наименее вероятных токенов с использованием `torch.kthvalue`.
- Написаны тесты в `tests/test_diffusion_cutoff_min_percent.py` для тестирования различных расписаний ("linear", "cosine", "exponential", "cyclic").

## [ ] Шаг 79: Добавление поддержки Repetition Penalization Penalty (RPP) по времени
**Цель:** Добавить поддержку временного штрафа (Decay Penalty) для повторяющихся токенов, где штраф ослабевает со временем (с расстоянием от текущего токена).

## [ ] Шаг 80: Добавление поддержки DRY (Don't Repeat Yourself) Sampling
**Цель:** Добавить поддержку DRY сэмплирования, которое предотвращает повторение длинных последовательностей токенов путем динамического регулирования штрафов на основе совпадений подстрок в контексте.
