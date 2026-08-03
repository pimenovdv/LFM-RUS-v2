# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-34)
**Сжатое описание:**
Реализована базовая интеграция MDLM с обширным набором методов сэмплинга и параметров генерации:
- Поддержка CFG, Guidance Rescale, Activation Steering, и Negative Prompting (`negative_prompt_ids`).
- Динамические расписания для temperature, cfg, guidance_rescale, top_k/p/a, min_p, typical_p, epsilon, eta, tfs_z, tkg, штрафов и параметров XTC.
- Реализованы алгоритмы сэмплинга (Top-k/p/a, Min-p, Typical, Epsilon, Eta, TFS, XTC, TKG).
- Добавлено принудительное декодирование, подавление токенов, logit_bias.
- Реализовано окно применения штрафов и сглаживание логитов.
- Внедрены динамические расписания размаскирования (linear, cosine, square, exponential).

## [x] Шаг 35: Поддержка `num_return_sequences` при генерации
**Цель:** Добавить поддержку возврата нескольких последовательностей на один входной промпт (параметр `num_return_sequences`), что необходимо для пайплайнов RLHF, таких как Best-of-N и Rejection Sampling.
* **Детали реализации:**
  * Добавить параметр `num_return_sequences: int = 1` в метод `generate` в файле `src/models/diffusion/modeling_diffusion.py`.
  * Реализовать логику: если `num_return_sequences > 1`, применить `repeat_interleave` для дублирования входных тензоров (`input_ids`, `attention_mask`, `unconditional_input_ids`, `negative_prompt_ids`) в размерности батча перед началом цикла генерации.
