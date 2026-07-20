# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-20)
**Цель:** Базовая интеграция MDLM и реализация обширного набора методов генерации: CFG, Guidance Rescale, различные Schedule, Top-k/p/a, Min-p, Typical, Epsilon, Eta, Penalties (включая repetition, frequency, presence), Dynamic Temperature, Early Stopping, TFS, Activation Steering, XTC Sampling, `min_new_tokens`, `no_repeat_ngram_size` и `bad_words_ids`.

## [x] Шаг 21: Добавление `max_time` и `remove_invalid_values`
**Цель:** Добавление возможности прерывать генерацию по тайм-ауту и очищать логиты от `NaN`/`Inf`.
* **Детали реализации:**
  * Добавить аргументы `max_time: Optional[float] = None` и `remove_invalid_values: bool = False` в метод `generate` в `src/models/diffusion/modeling_diffusion.py`.
  * Если `remove_invalid_values=True`, очищать логиты от NaN/Inf.
  * Если `max_time` задан, прерывать цикл генерации при превышении времени выполнения.
  * Написать тесты `test_max_time` и `test_remove_invalid_values` в `tests/test_diffusion.py`.