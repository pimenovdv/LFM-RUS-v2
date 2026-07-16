# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-16)
**Цель:** Базовая интеграция MDLM, пайплайнов и реализация основных методов сэмплирования (Dynamic CFG, Guidance Rescale, Exponential Schedule, Top-k/p/a, Min-p, Typical, Epsilon, Eta, Penalties, Dynamic Temperature, Early Stopping), управление генерацией и Activation Steering.

## [x] Шаг 17: Добавление Tail Free Sampling (TFS) и Dynamic Entropy Temperature
**Цель:** Внедрение метода фильтрации хвоста распределения (TFS) и динамической температуры на основе энтропии в метод `generate`.
* **Детали реализации:**
  * Добавить аргументы `tfs_z` и `dynamic_temperature_entropy` в функцию `generate` (`src/models/diffusion/modeling_diffusion.py`).
  * Реализовать логику TFS: отсечение токенов на основе второй производной вероятностей.
  * Реализовать логику Dynamic Entropy Temperature: корректировка температуры генерации пропорционально локальной энтропии вероятностей токенов.
  * Написать тесты для проверки корректности алгоритмов `test_tfs_sampling` и `test_dynamic_entropy_temperature` в `tests/test_diffusion.py`.
