# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-15)
**Цель:** Базовая интеграция MDLM, адаптация пайплайнов (CPT/SFT/Alignment), реализация всех основных методов сэмплирования (Dynamic CFG, Guidance Rescale, Exponential Schedule, Top-k/p/a, Min-p, Typical, Penalties, Dynamic Temperature, Early Stopping), управление генерацией (Logit Bias, Suppress Tokens) и Activation Steering.

## [x] Шаг 16: Добавление Epsilon и Eta Sampling
**Цель:** Внедрение методов обрезки распределения на основе Epsilon (абсолютный порог) и Eta (энтропийно-зависимый порог) в метод `generate`.
* **Детали реализации:**
  * Добавить аргументы `epsilon_cutoff` и `eta_cutoff` в функцию `generate` (`src/models/diffusion/modeling_diffusion.py`).
  * Реализовать логику Epsilon-отсечения: удаление токенов с вероятностью ниже абсолютного порога `epsilon_cutoff`.
  * Реализовать логику Eta-отсечения: удаление токенов с вероятностью ниже адаптивного порога `min(eta_cutoff, sqrt(eta_cutoff) * exp(-entropy))`.
  * Написать тесты для проверки корректности алгоритмов в `tests/test_diffusion.py`.
