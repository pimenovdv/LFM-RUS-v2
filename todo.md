# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-28)
**Сжатое описание:**
Реализована базовая интеграция MDLM с обширным набором методов сэмплинга и параметров генерации:
- Поддержка CFG и Guidance Rescale.
- Динамические расписания (linear, cosine, exponential) для temperature, cfg, top_k, top_p, min_p, typical_p, top_a, tfs_z, tkg и guidance_rescale.
- Реализованы алгоритмы сэмплинга: Top-k/p/a, Min-p, Typical, Epsilon, Eta, TFS, XTC Sampling, Top-K Guidance.
- Добавлены настройки логитов: Penalties (repetition, frequency, presence), Dynamic Temperature, Activation Steering.
- Реализованы ранние остановки (Early Stopping) и ограничения на длину.

## [x] Шаг 29: Динамические расписания для Penalties
**Цель:** Добавление динамического расписания для `repetition_penalty`, `frequency_penalty` и `presence_penalty`.
* **Детали реализации:**
  * Добавлены параметры для расписаний (например, `repetition_penalty_schedule`, `min_repetition_penalty`) в метод `generate` в `src/models/diffusion/modeling_diffusion.py`.
  * Реализована логика вычисления значений по аналогии с другими расписаниями (`linear`, `cosine`, `exponential`).
  * Написаны тесты для проверки корректности применения динамических штрафов.
