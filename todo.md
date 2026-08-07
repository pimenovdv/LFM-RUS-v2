# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-39)
**Сжатое описание:**
Реализована базовая интеграция MDLM с обширным набором методов сэмплинга и параметров генерации:
- Поддержка CFG, Guidance Rescale, Activation Steering, Negative Prompting.
- Динамические расписания для temperature, cfg, guidance_rescale, top_k/p/a, min_p, typical_p, epsilon, eta, tfs_z, tkg, штрафов и параметров XTC.
- Реализованы алгоритмы сэмплинга (Top-k/p/a, Min-p, Typical, Epsilon, Eta, TFS, XTC, TKG).
- Добавлено принудительное декодирование, подавление токенов (`suppress_tokens`), `begin_suppress_tokens`, `logit_bias`.
- Окно применения штрафов, сглаживание логитов, поддержка `num_return_sequences`.
- Поддержка `return_dict_in_generate`, `output_scores`, `logits_processor` и `stopping_criteria` для пайплайнов RLHF.

## [x] Шаг 40: Поддержка нескольких `eos_token_id`
**Цель:** Добавить возможность передавать как `int`, так и `list[int]` в параметр `eos_token_id` для поддержки моделей с несколькими токенами остановки (например, Llama 3). Логика `min_new_tokens` должна подавлять все токены из списка, пока нужное количество токенов не будет сгенерировано.

## [x] Шаг 41: Поддержка списка `forced_eos_token_id`
**Цель:** Добавить поддержку передачи `Union[int, list[int]]` для `forced_eos_token_id` (принудительная вставка EOS токена в конец). Если передан список, вставляется его первый элемент.
