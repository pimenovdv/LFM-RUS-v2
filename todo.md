# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-74)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM, включая базовый сэмплинг, динамические расписания, CFG, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, MoE, RAG и Continuous Time Diffusion. Внедрена дистилляция консистентности (Consistency Models Distillation). Добавлена интеграция FlashAttention-2 для двунаправленного маскирования. Реализована Latent Masked Diffusion (LMDLM). Интегрирован подход Discrete Flow Matching (`compute_flow_matching_loss`) как альтернатива классической диффузии. Добавлена поддержка Contrastive Decoding для корректировки логитов через amateur-модель (Шаг 74).

## [x] Шаг 75: Внедрение Self-Conditioning для MDLM
**Цель:** Реализовать поддержку Self-Conditioning в процессе прямого прохода и генерации для `DiffusionModelForConditionalGeneration`. Self-Conditioning улучшает качество генерации, позволяя модели использовать свои собственные предсказания (логиты) из предыдущего шага в качестве дополнительного условия.
**Детали:**
- Добавить аргумент `use_self_conditioning` в `DiffusionConfig`.
- В `DiffusionModelForConditionalGeneration` добавить линейный проекционный слой `self_cond_proj` (из размера словаря в `hidden_size`).
- В методе `forward` при `use_self_conditioning=True` с вероятностью 50% (или если переданы `self_cond_states`) делать предварительный forward pass без градиентов для получения логитов, пропускать их через `self_cond_proj` и добавлять к `inputs_embeds` основного прохода.
- Написать тесты для проверки корректности работы Self-Conditioning.
