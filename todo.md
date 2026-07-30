# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-30)
**Сжатое описание:**
Реализована базовая интеграция MDLM с обширным набором методов сэмплинга и параметров генерации:
- Поддержка CFG и Guidance Rescale.
- Динамические расписания (linear, cosine, exponential) для temperature, cfg, top_k, top_p, min_p, typical_p, top_a, epsilon, eta, tfs_z, tkg и guidance_rescale.
- Реализованы алгоритмы сэмплинга: Top-k/p/a, Min-p, Typical, Epsilon, Eta, TFS, XTC Sampling, Top-K Guidance.
- Добавлены настройки логитов: Penalties (repetition, frequency, presence), Dynamic Temperature, Activation Steering.
- Реализованы динамические расписания для штрафов (repetition_penalty, frequency_penalty, presence_penalty).
- Реализованы ранние остановки (Early Stopping) и ограничения на длину.

## [x] Шаг 31: Динамические расписания для параметров XTC Sampling
**Цель:** Добавление динамического расписания для `xtc_threshold` и `xtc_probability`.
* **Детали реализации:**
  * Добавление параметров `xtc_threshold_schedule`, `min_xtc_threshold`, `xtc_probability_schedule`, `min_xtc_probability` в метод `generate` в `src/models/diffusion/modeling_diffusion.py`.
  * Реализация логики вычисления `current_xtc_threshold` и `current_xtc_probability` с помощью расписаний (`linear`, `cosine`, `exponential`).
  * Написание тестов для проверки корректности применения динамических расписаний.
