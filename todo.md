# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-11)
**Цель:** Базовая интеграция MDLM, адаптация пайплайнов (CPT/SFT/Alignment), реализация основных и расширенных методов сэмплирования (Dynamic CFG, Guidance Rescale, Exponential Schedule, Top-k/p, Min-p, Penalties), а также управление генерацией на уровне токенов (Logit Bias и Suppress Tokens).

## [x] Шаг 12: Добавление поддержки Activation Steering при генерации
**Цель:** Позволить пользователям направлять процесс диффузионной генерации, передавая `steering_vector`, `steering_layer_name` и `steering_scale` в метод `generate`.
* **Детали реализации:**
  * Добавить аргументы `steering_vector`, `steering_layer_name` и `steering_scale` (по умолчанию 1.0) в функцию `generate` файла `src/models/diffusion/modeling_diffusion.py`.
  * Пробросить эти аргументы при вызове `self(...)` внутри цикла генерации, как для основного, так и для unconditional прохода (если применяется CFG).
  * Покрыть тестами в `tests/test_diffusion_steering.py`.
