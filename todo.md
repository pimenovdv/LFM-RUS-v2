# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-14)
**Цель:** Базовая интеграция MDLM, адаптация пайплайнов (CPT/SFT/Alignment), реализация основных и расширенных методов сэмплирования (Dynamic CFG, Guidance Rescale, Exponential Schedule, Top-k/p, Min-p, Penalties, Dynamic Temperature, Early Stopping), управление генерацией на уровне токенов (Logit Bias и Suppress Tokens), а также поддержка Activation Steering при генерации.

## [x] Шаг 15: Добавление Typical Sampling и Top-A Sampling
**Цель:** Внедрение продвинутых методов фильтрации токенов (typical_p, top_a) в метод генерации для улучшения качества текста.
* **Детали реализации:**
  * Обновить метод `generate` в `src/models/diffusion/modeling_diffusion.py` аргументами `typical_p` и `top_a`.
  * Реализовать логику Typical Sampling на основе энтропии распределения вероятностей.
  * Реализовать логику Top-A Sampling на основе максимальной вероятности.
  * Написать unit-тесты для обеих стратегий.
