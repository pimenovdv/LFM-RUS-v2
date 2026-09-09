# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-74)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM, включая базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, Mixture of Experts (MoE), Retrieval-Augmented Generation (RAG) и Continuous Time Diffusion. Внедрена дистилляция консистентности (Consistency Models Distillation). Добавлена интеграция FlashAttention-2 для двунаправленного маскирования (Шаг 71). Реализована Latent Masked Diffusion (LMDLM) через автоэнкодер и векторное квантование (Шаг 72). Интегрирован подход Discrete Flow Matching (`compute_flow_matching_loss`) как альтернатива классической диффузии (Шаг 73). Добавлена поддержка Contrastive Decoding для повышения специфичности генерации с использованием amateur-модели (Шаг 74).

## [x] Шаг 75: Реализация поддержки Self-Conditioning для MDLM во время инференса
**Цель:** Реализовать поддержку Self-Conditioning (самообусловливания) в процессе генерации для `DiffusionModelForConditionalGeneration`. Эта техника позволяет модели на текущем шаге диффузии использовать свой собственный прогноз логитов (или вероятностей) с предыдущего шага, подавая его как дополнительное условие. Это улучшает согласованность и качество генерации.
**Детали:**
- Добавить флаг `use_self_conditioning` в `DiffusionConfig`.
- Добавить линейный слой (проекцию) в `DiffusionModelForConditionalGeneration` для проецирования предсказанных логитов (размерности vocab_size) в пространство скрытых состояний (hidden_size).
- Модифицировать метод `generate`: на каждой итерации сохранять логиты с текущего шага и добавлять их проекцию к `inputs_embeds` на следующем шаге, если `use_self_conditioning=True`.
- Написать тесты для проверки работы Self-Conditioning, убедившись в корректном влиянии логитов с предыдущего шага на результат генерации.