# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-54)
**Сжатое описание:**
Реализована базовая интеграция MDLM с обширным набором методов сэмплинга и параметров генерации:
- Поддержка CFG, Guidance Rescale, Activation Steering, Negative Prompting.
- Динамические расписания для temperature, cfg, guidance_rescale, top_k/p/a, min_p, typical_p, epsilon, eta, tfs_z, tkg, штрафов, параметров XTC и length_penalty.
- Добавлено `sigmoid` расписание для `unmasking_schedule`.
- Реализованы алгоритмы сэмплинга (Top-k/p/a, Min-p, Typical, Epsilon, Eta, TFS, XTC, TKG).
- Добавлено принудительное декодирование, подавление токенов, `eos_token_id` и `forced_eos_token_id`.
- Окно применения штрафов, сглаживание логитов, поддержка `num_return_sequences`.
- Интеграция стандартных параметров HF: `return_dict_in_generate`, `output_scores`, `logits_processor`, `stopping_criteria`, `max_length` и `min_length`.
- Поддержка штрафов: `encoder_repetition_penalty`, `encoder_frequency_penalty`, `encoder_presence_penalty`, `encoder_no_repeat_ngram_size`, `no_repeat_ngram_size`, `ngram_penalty`, а также `length_penalty`.
- Поддержка `cyclic` расписаний для всех динамических параметров для улучшения эксплорации.
- Реализована стратегия ремаскинга `entropy` (`remasking="entropy"`), использующая энтропию логитов.
- Добавлен независимый параметр `gumbel_temperature` с поддержкой динамических расписаний (`gumbel_temperature_schedule`).
- Реализована поддержка динамических расписаний для `dynamic_temperature_entropy` (`dynamic_temperature_entropy_schedule` и `min_dynamic_temperature_entropy`).
- Добавлено жесткое усечение токенов `top_n_tokens` с расписаниями перед применением других методов сэмплинга.
- Поддержка `temperature` на уровне отдельных токенов/слоев (`temperature_mask`).

## [ ] Шаг 55: Интеграция Watermarking алгоритмов
**Цель:** Внедрение водяных знаков в распределение логитов для детекции сгенерированного текста (например, SynthID, KGW).

## [ ] Шаг 56: Реализация Classifier-Guided Sampling
**Цель:** Внедрение возможности использовать внешний классификатор для навигации процесса генерации (добавление градиентов классификатора к логитам во время демаскирования).
