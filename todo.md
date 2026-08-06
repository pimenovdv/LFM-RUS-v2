# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-38)
**Сжатое описание:**
Реализована базовая интеграция MDLM с обширным набором методов сэмплинга и параметров генерации:
- Поддержка CFG, Guidance Rescale, Activation Steering, Negative Prompting.
- Динамические расписания для temperature, cfg, guidance_rescale, top_k/p/a, min_p, typical_p, epsilon, eta, tfs_z, tkg, штрафов и параметров XTC.
- Реализованы алгоритмы сэмплинга (Top-k/p/a, Min-p, Typical, Epsilon, Eta, TFS, XTC, TKG).
- Добавлено принудительное декодирование, подавление токенов (`suppress_tokens`), `logit_bias`.
- Окно применения штрафов, сглаживание логитов, поддержка `num_return_sequences`.
- Поддержка `return_dict_in_generate`, `output_scores`, `logits_processor` и `stopping_criteria` для пайплайнов RLHF.

## [x] Шаг 39: Поддержка `begin_suppress_tokens`
**Цель:** Добавить поддержку передачи `begin_suppress_tokens` в метод `generate` для подавления указанных токенов исключительно на первом шаге генерации нового текста (сразу после промпта).
