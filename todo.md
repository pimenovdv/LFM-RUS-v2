# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-41)
**Сжатое описание:**
Реализована базовая интеграция MDLM с обширным набором методов сэмплинга и параметров генерации:
- Поддержка CFG, Guidance Rescale, Activation Steering, Negative Prompting.
- Динамические расписания для temperature, cfg, guidance_rescale, top_k/p/a, min_p, typical_p, epsilon, eta, tfs_z, tkg, штрафов и параметров XTC.
- Реализованы алгоритмы сэмплинга (Top-k/p/a, Min-p, Typical, Epsilon, Eta, TFS, XTC, TKG).
- Добавлено принудительное декодирование, подавление токенов, `eos_token_id` (списки) и `forced_eos_token_id` (списки).
- Окно применения штрафов, сглаживание логитов, поддержка `num_return_sequences`.
- Поддержка `return_dict_in_generate`, `output_scores`, `logits_processor` и `stopping_criteria` для пайплайнов RLHF.

## [x] Шаг 42: Поддержка `encoder_repetition_penalty`
**Цель:** Добавить поддержку штрафа за повторение токенов из промпта (по аналогии с энкодером в seq2seq моделях).

## [x] Шаг 43: Поддержка `encoder_no_repeat_ngram_size`
**Цель:** Добавить поддержку запрета на генерацию n-грамм, которые уже присутствуют в промпте (исходном контексте).
