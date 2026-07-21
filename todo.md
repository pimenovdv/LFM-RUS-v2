# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-21)
**Цель:** Базовая интеграция MDLM и реализация обширного набора методов генерации: CFG, Guidance Rescale, различные Schedule, Top-k/p/a, Min-p, Typical, Epsilon, Eta, Penalties (включая repetition, frequency, presence), Dynamic Temperature, Early Stopping, TFS, Activation Steering, XTC Sampling, `min_new_tokens`, `no_repeat_ngram_size`, `bad_words_ids`, `max_time` и `remove_invalid_values`.

## [x] Шаг 22: Добавление `forced_decoder_ids`, `forced_eos_token_id` и `renormalize_logits`
**Цель:** Добавление возможности форсировать токены в генерации и нормализовывать логиты перед генерацией.
* **Детали реализации:**
  * Добавить параметры `forced_decoder_ids: Optional[List[List[int]]] = None`, `forced_eos_token_id: Optional[int] = None` и `renormalize_logits: bool = False` в метод `generate` в `src/models/diffusion/modeling_diffusion.py`.
  * Реализовать применение `forced_decoder_ids` (набор токенов и их относительных индексов от начала генерации), чтобы они подставлялись в тензор перед началом генерации, и маска на этих позициях снималась.
  * Если `forced_eos_token_id` задан, подставить его в самый конец сгенерированной последовательности (индекс max_new_tokens - 1).
  * Если `renormalize_logits=True`, делать `logits = logits - logits.logsumexp(dim=-1, keepdim=True)` перед дальнейшими вычислениями.
  * Написать тесты `test_forced_decoder_ids`, `test_forced_eos_token_id` и `test_renormalize_logits`.
