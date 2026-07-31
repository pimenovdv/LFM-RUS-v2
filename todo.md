# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-31)
**Сжатое описание:**
Реализована базовая интеграция MDLM с обширным набором методов сэмплинга и параметров генерации:
- Поддержка CFG, Guidance Rescale, Activation Steering.
- Динамические расписания для temperature, cfg, guidance_rescale, top_k/p/a, min_p, typical_p, epsilon, eta, tfs_z, tkg, штрафов (repetition, frequency, presence) и параметров XTC (threshold, probability).
- Реализованы алгоритмы сэмплинга: Top-k/p/a, Min-p, Typical, Epsilon, Eta, TFS, XTC Sampling, Top-K Guidance.
- Добавлены ранние остановки (Early Stopping) и ограничения на длину.
- Добавлено принудительное декодирование (forced_decoder_ids, forced_eos_token_id), подавление токенов (suppress_tokens) и logit_bias.

## [x] Шаг 32: Окно применения штрафов (Penalty Range) и Сглаживание логитов (Logit Smoothing)
**Цель:** Ограничить применение штрафов последними N сгенерированными токенами и добавить возможность сглаживания распределения логитов.
* **Детали реализации:**
  * Добавить `penalty_range: Optional[int] = None` и `logit_smoothing: float = 0.0` в метод `generate`.
  * При расчёте контекста токенов для штрафов за повторение/частоту использовать срез `context_tokens[-penalty_range:]`.
  * Реализовать формулу `logits = logits * (1 - logit_smoothing) + logits.mean(dim=-1, keepdim=True) * logit_smoothing` после применения всех остальных модификаторов.
  * Написать тесты для новых параметров.
