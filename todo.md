# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-26)
**Цель:** Базовая интеграция MDLM и реализация обширного набора методов генерации: CFG, Guidance Rescale, Schedule (temperature, cfg, top_k, top_p, min_p, typical_p, top_a, tfs_z), Top-k/p/a, Min-p, Typical, Epsilon, Eta, Penalties (включая repetition, frequency, presence), Dynamic Temperature, Early Stopping, TFS, Activation Steering, XTC Sampling, и другие ограничения и настройки логитов.

## [ ] Шаг 27: Поддержка Top-K Guidance (TKG) и Dynamic Schedule
**Цель:** Добавление Top-K Guidance и schedule для него.
* **Детали реализации:**
  * Добавить параметры `tkg_scale: float = 0.0`, `tkg_schedule: str = "constant"`, `tkg_min_scale: float = 0.0` в метод `generate` в `src/models/diffusion/modeling_diffusion.py`.
  * Реализовать логику для `tkg_schedule`.
  * Добавить применение Top-K Guidance к логитам после CFG.
  * Написать тесты для проверки `tkg_schedule` (`linear`, `cosine`, `exponential`) в `tests/test_mdlm_generation.py`.
