# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-49)
**Сжатое описание:**
Реализована базовая интеграция MDLM с обширным набором методов сэмплинга и параметров генерации:
- Поддержка CFG, Guidance Rescale, Activation Steering, Negative Prompting.
- Динамические расписания для temperature, cfg, guidance_rescale, top_k/p/a, min_p, typical_p, epsilon, eta, tfs_z, tkg, штрафов и параметров XTC.
- Добавлено `sigmoid` расписание для `unmasking_schedule`.
- Реализованы алгоритмы сэмплинга (Top-k/p/a, Min-p, Typical, Epsilon, Eta, TFS, XTC, TKG).
- Добавлено принудительное декодирование, подавление токенов, `eos_token_id` и `forced_eos_token_id`.
- Окно применения штрафов, сглаживание логитов, поддержка `num_return_sequences`.
- Интеграция стандартных параметров HF: `return_dict_in_generate`, `output_scores`, `logits_processor`, `stopping_criteria`, `max_length` и `min_length`.
- Поддержка штрафов `encoder_repetition_penalty`, `encoder_frequency_penalty`, `encoder_presence_penalty` и запрета `encoder_no_repeat_ngram_size` на основе промпта.
- Поддержка `cyclic` расписаний для всех динамических параметров для улучшения эксплорации.
- Реализована стратегия ремаскинга `entropy` (`remasking="entropy"`), использующая энтропию логитов.
- Добавлен независимый параметр `gumbel_temperature` с поддержкой динамических расписаний (`gumbel_temperature_schedule`).

## [ ] Шаг 50: Поддержка `dynamic_temperature_entropy` с расписаниями
**Цель:** Добавить поддержку динамических расписаний (например, `dynamic_temperature_entropy_schedule`) для параметра `dynamic_temperature_entropy`, чтобы управлять влиянием энтропии на температуру в течение процесса генерации.

## [ ] Шаг 51: Добавление `top_n_tokens` для усечения в методах сэмплинга
**Цель:** Внедрить параметр `top_n_tokens`, который будет ограничивать выборку N наиболее вероятными токенами перед применением специфических алгоритмов сэмплинга (как дополнительный фильтр перед типичным сэмплингом или XTC).
