# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-73)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM, включая базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, Mixture of Experts (MoE), Retrieval-Augmented Generation (RAG) и Continuous Time Diffusion. Внедрена дистилляция консистентности (Consistency Models Distillation). Добавлена интеграция FlashAttention-2 для двунаправленного маскирования (Шаг 71). Реализована Latent Masked Diffusion (LMDLM) через автоэнкодер и векторное квантование (Шаг 72). Интегрирован подход Discrete Flow Matching (`compute_flow_matching_loss`) как альтернатива классической диффузии (Шаг 73).

## [x] Шаг 74: Внедрение Contrastive Decoding для MDLM
**Цель:** Реализовать поддержку Contrastive Decoding в процессе генерации для `DiffusionModelForConditionalGeneration`. Идея заключается в корректировке логитов основной (сильной) модели путем вычитания взвешенных логитов слабой (amateur) модели. Это позволяет уменьшить вероятность выбора "общих" и скучных токенов, повышая специфичность и качество генерации.
**Детали:**
- Добавить аргументы `amateur_model` и `contrastive_alpha` в метод `generate`.
- На каждой итерации размаскирования вычислять логиты через `amateur_model` (параллельно с основной моделью).
- Корректировать логиты основной модели: `logits = main_logits - contrastive_alpha * amateur_logits` перед вычислением вероятностей и выбором токенов.
- Написать тесты для проверки корректности работы генерации с `amateur_model`.