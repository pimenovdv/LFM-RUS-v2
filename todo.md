# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-33)
**Сжатое описание:**
Реализована базовая интеграция MDLM с обширным набором методов сэмплинга и параметров генерации:
- Поддержка CFG, Guidance Rescale, Activation Steering.
- Динамические расписания для temperature, cfg, guidance_rescale, top_k/p/a, min_p, typical_p, epsilon, eta, tfs_z, tkg, штрафов и параметров XTC.
- Реализованы алгоритмы сэмплинга (Top-k/p/a, Min-p, Typical, Epsilon, Eta, TFS, XTC, TKG).
- Добавлено принудительное декодирование, подавление токенов, logit_bias.
- Реализовано окно применения штрафов и сглаживание логитов.
- Внедрены динамические расписания размаскирования (linear, cosine, square, exponential) для определения доли токенов на каждом шаге.

## [x] Шаг 34: Поддержка Negative Prompting (negative_prompt_ids) для Classifier-Free Guidance
**Цель:** Позволить использование отрицательных промптов (`negative_prompt_ids`) в качестве безусловного входа (`unconditional_input_ids`) при сэмплировании.
* **Детали реализации:**
  * Добавлен параметр `negative_prompt_ids: Optional[torch.Tensor] = None` в функцию `generate` модели `DiffusionModelForConditionalGeneration`.
  * Если `negative_prompt_ids` передан, он присваивается переменной `unconditional_input_ids`.
  * Написаны и пройдены тесты (включая mock для `forward`) для подтверждения работы negative prompting с CFG.
