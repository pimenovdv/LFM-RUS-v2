# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-86)
**Сжатое описание:**
Реализована полнофункциональная интеграция MDLM, включая базовый сэмплинг, динамические расписания, Classifier-Free Guidance, Watermarking, Classifier-Guided Sampling, Continuous Batching, Dynamic Batching, Beam Search, Speculative Decoding, LoRA, оптимизацию памяти, RLAIF, Mixture of Experts (MoE), Retrieval-Augmented Generation (RAG), Continuous Time Diffusion, дистилляцию консистентности, Latent Masked Diffusion, Discrete Flow Matching, Contrastive Decoding, Mirostat Sampling, DRY Sampling, Min-K% Prob Sampling, XTC Sampling, динамические расписания для большинства параметров сэмплирования, а также механизм сглаживания логитов (Logits Momentum Sampling).

## [x] Шаг 87: Добавление поддержки Gaussian Noise Injection Sampling
**Цель:** Добавить механизм добавления гауссовского шума к сырым логитам для контролируемого увеличения случайности и разнообразия.
**Детали:**
- Добавить параметры `gaussian_noise_std` (float, по умолчанию 0.0), `gaussian_noise_schedule` (строка, по умолчанию "constant") и `min_gaussian_noise_std` (float, по умолчанию 0.0) в метод `generate` и класс `MDLMRequest`.
- Если `gaussian_noise_std > 0.0`, то перед применением фильтров, но после Logits Momentum, вычислять `current_std` согласно расписанию и добавлять `torch.randn_like(logits) * current_std` к логитам.
- Написать тесты, чтобы покрыть новый код и поддержать покрытие выше 90%.
