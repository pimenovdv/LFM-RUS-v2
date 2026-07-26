# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-27)
**Цель:** Базовая интеграция MDLM, реализация обширного набора методов генерации: CFG, Guidance Rescale, Schedule (temperature, cfg, top_k, top_p, min_p, typical_p, top_a, tfs_z, tkg), Top-k/p/a, Min-p, Typical, Epsilon, Eta, Penalties (включая repetition, frequency, presence), Dynamic Temperature, Early Stopping, TFS, Activation Steering, XTC Sampling, Top-K Guidance (TKG) и другие ограничения и настройки логитов.

## [ ] Шаг 28: Поддержка CFG Rescale Schedule
**Цель:** Добавление динамического расписания для Guidance Rescale.
* **Детали реализации:**
  * Добавить параметры `guidance_rescale_schedule: str = "constant"` и `min_guidance_rescale: float = 0.0` в метод `generate` в `src/models/diffusion/modeling_diffusion.py`.
  * Реализовать логику для `guidance_rescale_schedule` (`linear`, `cosine`, `exponential`).
  * Написать тесты для проверки динамического Guidance Rescale.
