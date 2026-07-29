# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-29)
**Сжатое описание:**
Реализована базовая интеграция MDLM с обширным набором методов сэмплинга и параметров генерации:
- Поддержка CFG и Guidance Rescale.
- Динамические расписания (linear, cosine, exponential) для temperature, cfg, top_k, top_p, min_p, typical_p, top_a, tfs_z, tkg и guidance_rescale.
- Реализованы алгоритмы сэмплинга: Top-k/p/a, Min-p, Typical, Epsilon, Eta, TFS, XTC Sampling, Top-K Guidance.
- Добавлены настройки логитов: Penalties (repetition, frequency, presence), Dynamic Temperature, Activation Steering.
- Реализованы динамические расписания для штрафов (repetition_penalty, frequency_penalty, presence_penalty).
- Реализованы ранние остановки (Early Stopping) и ограничения на длину.

## [x] Шаг 30: Динамические расписания для Epsilon и Eta cutoff
**Цель:** Добавление динамического расписания для `epsilon_cutoff` и `eta_cutoff`.
* **Детали реализации:**
  * Добавлены параметры для расписаний (например, `epsilon_cutoff_schedule`, `min_epsilon_cutoff`, `eta_cutoff_schedule`, `min_eta_cutoff`) в метод `generate` в `src/models/diffusion/modeling_diffusion.py`.
  * Реализована логика вычисления значений по аналогии с другими расписаниями (`linear`, `cosine`, `exponential`).
  * Написаны тесты для проверки корректности применения динамических расписаний для Epsilon и Eta cutoff.
