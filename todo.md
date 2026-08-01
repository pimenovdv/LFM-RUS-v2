# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-32)
**Сжатое описание:**
Реализована базовая интеграция MDLM с обширным набором методов сэмплинга и параметров генерации:
- Поддержка CFG, Guidance Rescale, Activation Steering.
- Динамические расписания для temperature, cfg, guidance_rescale, top_k/p/a, min_p, typical_p, epsilon, eta, tfs_z, tkg, штрафов (repetition, frequency, presence) и параметров XTC (threshold, probability).
- Реализованы алгоритмы сэмплинга: Top-k/p/a, Min-p, Typical, Epsilon, Eta, TFS, XTC Sampling, Top-K Guidance.
- Добавлены ранние остановки (Early Stopping) и ограничения на длину.
- Добавлено принудительное декодирование (forced_decoder_ids, forced_eos_token_id), подавление токенов (suppress_tokens) и logit_bias.
- Реализовано окно применения штрафов (penalty_range) и сглаживание логитов (logit_smoothing).

## [x] Шаг 33: Динамические расписания размаскирования (Dynamic Unmasking Schedules)
**Цель:** Добавить поддержку нелинейных расписаний (`cosine`, `square`, `exponential`) для определения количества токенов, размаскируемых на каждом шаге.
* **Детали реализации:**
  * Обновить функцию `get_num_transfer_tokens`, добавив параметр `schedule="linear"` по умолчанию.
  * Реализовать расчет доли размаскируемых токенов для нелинейных расписаний, как это делается для других параметров генерации.
  * В метод `generate` добавить параметр `unmasking_schedule: str = "linear"` и пробрасывать его в `get_num_transfer_tokens`.
  * Написать тесты для новых параметров и алгоритмов сэмплинга.
