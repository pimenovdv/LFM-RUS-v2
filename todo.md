# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-17)
**Цель:** Базовая интеграция MDLM, пайплайнов и реализация основных методов сэмплирования (Dynamic CFG, Guidance Rescale, Exponential Schedule, Top-k/p/a, Min-p, Typical, Epsilon, Eta, Penalties, Dynamic Temperature, Early Stopping, TFS, Dynamic Entropy Temperature), управление генерацией и Activation Steering.

## [x] Шаг 18: Интеграция XTC Sampling и min_new_tokens
**Цель:** Добавление метода XTC (Exclude Top Choices) Sampling для подавления наиболее вероятных токенов (увеличение разнообразия) и поддержка аргумента `min_new_tokens` для ограничения минимальной длины генерации.
* **Детали реализации:**
  * Добавить аргументы `xtc_threshold: float = 0.0`, `xtc_probability: float = 0.0` и `min_new_tokens: Optional[int] = None` в функцию `generate` (`src/models/diffusion/modeling_diffusion.py`).
  * Реализовать логику `min_new_tokens`: если сгенерировано меньше токенов, чем `min_new_tokens`, приравнивать вероятность `eos_token_id` к `-inf`.
  * Реализовать логику XTC: если вероятность самого вероятного токена превышает `xtc_threshold`, то с вероятностью `xtc_probability` исключать его из распределения (присваивая `-inf`).
  * Написать тесты `test_xtc_sampling` и `test_min_new_tokens` в `tests/test_diffusion.py` для обеспечения полного покрытия нового кода.
