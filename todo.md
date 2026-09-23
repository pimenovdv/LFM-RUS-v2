# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-87)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM, включая базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, Mixture of Experts (MoE), Retrieval-Augmented Generation (RAG), Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat Sampling, DRY Sampling, Min-K% Prob Sampling, XTC Sampling, динамические расписания для большинства параметров сэмплирования, а также механизм сглаживания логитов (Logits Momentum Sampling) и Gaussian Noise Injection Sampling.

## [x] Шаг 88: Поддержка Top-K, Top-P, Min-P фильтров и вероятностного сэмплирования в Continuous Batching
**Цель:** Добавить поддержку популярных фильтров (Top-K, Top-P, Min-P) и вероятностного выбора токенов (через `torch.multinomial` при `temperature > 0`) для режима Continuous Batching.
**Детали:**
- Добавить параметры `top_k`, `top_k_schedule`, `min_top_k`, `top_p`, `top_p_schedule`, `min_top_p`, `min_p`, `min_p_schedule`, `min_min_p` в класс `MDLMRequest`.
- В методе `MDLMContinuousBatchingManager.step` рассчитывать их текущие значения на основе шага (`step_ratio`).
- Применять фильтры Top-K, Top-P, Min-P к логитам запроса (заменяя отфильтрованные значения на `-inf`).
- Изменить алгоритм выбора `x0`: если `temperature > 0`, использовать `torch.multinomial` для выбора по вероятностям `F.softmax`, иначе применять жадный `torch.argmax`.
