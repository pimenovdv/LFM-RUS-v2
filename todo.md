# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-45)
**Сжатое описание:**
Реализована базовая интеграция MDLM с обширным набором методов сэмплинга и параметров генерации:
- Поддержка CFG, Guidance Rescale, Activation Steering, Negative Prompting.
- Динамические расписания для temperature, cfg, guidance_rescale, top_k/p/a, min_p, typical_p, epsilon, eta, tfs_z, tkg, штрафов и параметров XTC.
- Реализованы алгоритмы сэмплинга (Top-k/p/a, Min-p, Typical, Epsilon, Eta, TFS, XTC, TKG).
- Добавлено принудительное декодирование, подавление токенов, `eos_token_id` и `forced_eos_token_id`.
- Окно применения штрафов, сглаживание логитов, поддержка `num_return_sequences`.
- Интеграция стандартных параметров HF: `return_dict_in_generate`, `output_scores`, `logits_processor`, `stopping_criteria`, `max_length` и `min_length`.
- Поддержка штрафов `encoder_repetition_penalty` и запрета `encoder_no_repeat_ngram_size` на основе промпта.
- Поддержка `cyclic` расписаний для всех динамических параметров для улучшения эксплорации.

## [x] Шаг 46: Поддержка `sigmoid` расписания для `unmasking_schedule`
**Цель:** Добавить `sigmoid` расписание в функцию анмаскинга токенов для более плавного перехода в начале и конце процесса генерации.

## [x] Шаг 47: Внедрение `encoder_frequency_penalty` и `encoder_presence_penalty`
**Цель:** Добавить аналоги частотных штрафов, применяемые к токенам исходного промпта, для более гибкого управления лексикой, основанной на входных данных.
