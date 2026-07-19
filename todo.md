# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-19)
**Цель:** Базовая интеграция MDLM и реализация обширного набора методов генерации: CFG, Guidance Rescale, различные Schedule, Top-k/p/a, Min-p, Typical, Epsilon, Eta, Penalties (включая repetition, frequency, presence), Dynamic Temperature, Early Stopping, TFS, Activation Steering, XTC Sampling, `min_new_tokens` и `no_repeat_ngram_size`.

## [x] Шаг 20: Добавление `bad_words_ids`
**Цель:** Добавление возможности запрещать генерацию конкретных последовательностей токенов.
* **Детали реализации:**
  * Добавить аргумент `bad_words_ids: Optional[list[list[int]]] = None` в метод `generate` в `src/models/diffusion/modeling_diffusion.py`.
  * Внутри цикла генерации проверять, не является ли уже сгенерированный префикс в текущем блоке началом одной из запрещенных последовательностей.
  * Если да, то токен, завершающий последовательность (или продолжающий), нужно заблокировать (логит в `-inf`).
  * Написать тест `test_bad_words_ids` в `tests/test_diffusion.py`.
