# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-72)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM, включая базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, Mixture of Experts (MoE), Retrieval-Augmented Generation (RAG) и Continuous Time Diffusion. Внедрена дистилляция консистентности (Consistency Models Distillation). Добавлена интеграция FlashAttention-2 для двунаправленного маскирования (Шаг 71). Реализована Latent Masked Diffusion (LMDLM) через автоэнкодер и векторное квантование для сжатия контекста (Шаг 72).

## [ ] Шаг 73: Интеграция Discrete Flow Matching для MDLM
**Цель:** Внедрить подход Discrete Flow Matching как альтернативу классическому диффузионному процессу. Это включает в себя добавление функции потерь `compute_flow_matching_loss` на базе условных вероятностей и интерполированного распределения в `DiffusionModelForConditionalGeneration`, а также флага `use_flow_matching` в `DiffusionConfig`.