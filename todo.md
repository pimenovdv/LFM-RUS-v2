# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-24)
**Цель:** Базовая интеграция MDLM и реализация обширного набора методов генерации: CFG, Guidance Rescale, различные Schedule, Top-k/p/a, Min-p, Typical, Epsilon, Eta, Penalties (включая repetition, frequency, presence), Dynamic Temperature, Early Stopping, TFS, Activation Steering, XTC Sampling, и другие ограничения и настройки логитов (включая динамические typical_p и top_p).

## [x] Шаг 25: Добавление `top_k_schedule` и `min_p_schedule`
**Цель:** Добавление возможности динамически изменять значения top_k и min_p во время генерации для большего контроля на разных этапах деноизинга.
* **Детали реализации:**
  * Добавить параметры `top_k_schedule: str = "constant"`, `min_top_k: int = 0`, `min_p_schedule: str = "constant"`, `min_min_p: float = 0.0` в метод `generate` в `src/models/diffusion/modeling_diffusion.py`.
  * Реализовать логику для `top_k_schedule` и `min_p_schedule` (расчет `current_top_k` и `current_min_p` в зависимости от `step_ratio`).
  * Использовать `current_top_k` и `current_min_p` при фильтрации логитов вместо статических.
  * Написать тесты для проверки `top_k_schedule` и `min_p_schedule` (`linear`, `cosine`, `exponential`) в `tests/test_mdlm_generation.py`.
