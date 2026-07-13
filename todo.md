# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-12)
**Цель:** Базовая интеграция MDLM, адаптация пайплайнов (CPT/SFT/Alignment), реализация основных и расширенных методов сэмплирования (Dynamic CFG, Guidance Rescale, Exponential Schedule, Top-k/p, Min-p, Penalties), управление генерацией на уровне токенов (Logit Bias и Suppress Tokens), а также поддержка Activation Steering при генерации.

## [x] Шаг 13: Динамическое расписание температуры (Dynamic Temperature Scheduling)
**Цель:** Внедрение `temperature_schedule` ("constant", "linear", "cosine", "exponential") и `min_temperature` для управления температурой в процессе диффузии.
* **Детали реализации:**
  * Обновить метод `generate` в `src/models/diffusion/modeling_diffusion.py` аргументами `temperature_schedule` и `min_temperature`.
  * Вычислять `current_temperature` на каждом шаге на основе `step_ratio`.
  * Добавить тесты.

## [x] Шаг 14: Поддержка ранней остановки (Early Stopping via EOS)
**Цель:** Поддержка `eos_token_id` и `pad_token_id` для досрочного завершения блоков и обрезки результата.
* **Детали реализации:**
  * Добавить аргументы `eos_token_id` и `pad_token_id` в `generate`.
  * Добавить проверку `eos_token_id` после генерации блока, обрезать последовательности и делать паддинг при необходимости, с досрочным выходом из цикла.
  * Добавить тесты.
