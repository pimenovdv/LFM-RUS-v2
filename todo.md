# План внедрения обучения Diffusion Language Models (MDLM) в пайплайн LFM-RUS-v2

Этот документ содержит пошаговый план разработки для интеграции маскированной дискретной диффузии в существующий пайплайн обучения.

## [x] Завершенные этапы (Шаги 1-35)
**Сжатое описание:**
Реализована базовая интеграция MDLM с обширным набором методов сэмплинга и параметров генерации:
- Поддержка CFG, Guidance Rescale, Activation Steering, и Negative Prompting (`negative_prompt_ids`).
- Динамические расписания для temperature, cfg, guidance_rescale, top_k/p/a, min_p, typical_p, epsilon, eta, tfs_z, tkg, штрафов и параметров XTC.
- Реализованы алгоритмы сэмплинга (Top-k/p/a, Min-p, Typical, Epsilon, Eta, TFS, XTC, TKG).
- Добавлено принудительное декодирование, подавление токенов, logit_bias.
- Реализовано окно применения штрафов и сглаживание логитов.
- Внедрены динамические расписания размаскирования (linear, cosine, square, exponential).

- Поддержка `num_return_sequences` при генерации для генерации нескольких последовательностей на один промпт (полезно для Best-of-N).

## [x] Шаг 36: Поддержка `return_dict_in_generate` и `output_scores`
**Цель:** Добавить поддержку возврата результатов генерации в виде словаря с логитами (scores), что необходимо для пайплайнов RLHF (Reward Modeling, PPO) для вычисления логпробов и других метрик.
* **Детали реализации:**
  * Добавить параметры `return_dict_in_generate: bool = False` и `output_scores: bool = False` в метод `generate` в `src/models/diffusion/modeling_diffusion.py`.
  * Если `output_scores=True`, сохранять логиты (logits_full) на каждом шаге итерации.
  * Если `return_dict_in_generate=True`, возвращать словарь `{"sequences": input_ids, "scores": tuple(scores)}` вместо тензора.
