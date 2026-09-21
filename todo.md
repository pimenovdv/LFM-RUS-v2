# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-85)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM, включая базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, Mixture of Experts (MoE), Retrieval-Augmented Generation (RAG), Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat Sampling, DRY Sampling, Min-K% Prob Sampling, XTC Sampling, и динамические расписания для большинства параметров сэмплирования.

## [x] Шаг 86: Добавление поддержки Logits Momentum Sampling
**Цель:** Добавить механизм сглаживания логитов (Exponential Moving Average) между шагами диффузии.
**Детали:**
- Ввести параметры `logits_momentum` (float, от 0 до 1, по умолчанию 0.0), `logits_momentum_schedule` и `min_logits_momentum` в метод `generate`.
- Если `logits_momentum > 0.0`, то перед применением различных фильтров (CFG, Temperature и т.д.) и штрафов сглаживать сырые логиты: `running_logits = logits_momentum * running_logits + (1 - logits_momentum) * current_logits`.
- Хранить состояние `running_logits` между шагами `for step in range(steps):` внутри `generate`.
