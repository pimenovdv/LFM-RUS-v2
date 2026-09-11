# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-76)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM, включая базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, Mixture of Experts (MoE), Retrieval-Augmented Generation (RAG), Continuous Time Diffusion, дистилляцию консистентности (Consistency Models Distillation), интеграцию FlashAttention-2 (Шаг 71), Latent Masked Diffusion (Шаг 72), Discrete Flow Matching (Шаг 73), Contrastive Decoding (Шаг 74), поддержку Self-Conditioning во время инференса (Шаг 75) и во время обучения (Шаг 76).

## [x] Шаг 77: Внедрение алгоритма сэмплирования Mirostat
**Цель:** Добавить поддержку сэмплирования Mirostat во время генерации в `DiffusionModelForConditionalGeneration`. Mirostat позволяет динамически регулировать уровень "сюрприза" (perplexity) текста, что приводит к более качественной и когерентной генерации.
**Детали:**
- Добавить параметры `mirostat_mode: int = 0`, `mirostat_tau: float = 5.0`, `mirostat_eta: float = 0.1`, `mirostat_mu: Optional[float] = None` в `generate`.
- На каждом шаге генерации, если `mirostat_mode == 1`, вычислять вероятности `p = softmax(logits)` и `surprise = -log2(p)`.
- Оставлять только токены, для которых `surprise <= mu` (остальным задавать `-inf`).
- В конце каждого шага обновлять `mu`: `mu = mu - eta * (surprise_of_sampled_token - tau)`.
- Покрыть новую функциональность тестами в `tests/test_diffusion_mirostat.py`.
