# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-23)
**Цель:** Базовая интеграция MDLM и реализация обширного набора методов генерации: CFG, Guidance Rescale, различные Schedule, Top-k/p/a, Min-p, Typical, Epsilon, Eta, Penalties (включая repetition, frequency, presence), Dynamic Temperature, Early Stopping, TFS, Activation Steering, XTC Sampling, и другие ограничения и настройки логитов (включая динамический typical_p).

## [x] Шаг 24: Добавление `top_p_schedule` и динамического top_p
**Цель:** Добавление возможности динамически изменять значение top_p во время генерации (Nucleus Sampling) для большего контроля над разнообразием на разных этапах деноизинга.
* **Детали реализации:**
  * Добавить параметры `top_p_schedule: str = "constant"` и `min_top_p: float = 0.0` в метод `generate` в `src/models/diffusion/modeling_diffusion.py`.
  * Реализовать логику для `top_p_schedule`, аналогичную `typical_p_schedule` (расчет `current_top_p` в зависимости от `step_ratio`).
  * Использовать `current_top_p` при фильтрации логитов.
  * Написать тесты для проверки `top_p_schedule` (`linear`, `cosine`, `exponential`) в `tests/test_mdlm_generation.py`.
