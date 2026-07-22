# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-22)
**Цель:** Базовая интеграция MDLM и реализация обширного набора методов генерации: CFG, Guidance Rescale, различные Schedule, Top-k/p/a, Min-p, Typical, Epsilon, Eta, Penalties (включая repetition, frequency, presence), Dynamic Temperature, Early Stopping, TFS, Activation Steering, XTC Sampling, `min_new_tokens`, `no_repeat_ngram_size`, `bad_words_ids`, `max_time`, `remove_invalid_values`, `forced_decoder_ids`, `forced_eos_token_id` и `renormalize_logits`.

## [x] Шаг 23: Добавление `typical_p_schedule` и динамического typical_p
**Цель:** Добавление возможности динамически изменять значение typical_p во время генерации.
* **Детали реализации:**
  * Добавить параметры `typical_p_schedule: str = "constant"` и `min_typical_p: float = 0.0` в метод `generate` в `src/models/diffusion/modeling_diffusion.py`.
  * Реализовать логику для `typical_p_schedule`, аналогичную `temperature_schedule` или `cfg_schedule` (расчет `current_typical_p` в зависимости от `step_ratio`).
  * Написать тесты для проверки `typical_p_schedule` (`linear`, `cosine`, `exponential`) в `tests/test_mdlm_generation.py`.
