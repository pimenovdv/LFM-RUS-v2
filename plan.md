1. **Интеграция Watermarking алгоритмов в DiffusionModelForConditionalGeneration**:
   - `WatermarkLogitsProcessor` из библиотеки `transformers` не поддерживает 3D тензоры напрямую, так как он применяет разные сиды для разных элементов батча и делает цикл `for b_idx, input_seq in enumerate(input_ids)`.
   - В диффузионном процессе генерации мы обновляем логиты параллельно для всех маскированных токенов (формы `batch_size x seq_len x vocab_size`).
   - Нам нужно импортировать `WatermarkingConfig` и `WatermarkLogitsProcessor` из `transformers`.
   - Если `watermarking_config` предоставлен в функцию `generate` (например, `watermarking_config=None`), мы инициализируем `WatermarkLogitsProcessor` в начале метода.
   - Внутри цикла `steps_per_block`, после получения `logits` и перед применением других `logits_processor`, мы проверяем: `if watermarking_config is not None and watermark_processor is not None:`.
   - Мы можем векторизованно/параллельно применять вотермарк: для каждого батча и каждого замаскированного токена, мы берем `context_width` предыдущих токенов (из `x`) и получаем `greenlist`. Так как это диффузия, токены генерируются независимо, и вотермарк будет работать, если использовать уже предсказанные не маскированные (или маскированные) контекстные токены.
   - Более того, для простоты мы можем итерироваться по `b_idx` и по замаскированным позициям `seq_idx`, извлекая контекст из `x[b_idx, :seq_idx]` (или до текущего индекса), применяя `watermark_processor` к этому конкретному срезу: `scores = logits[b_idx, seq_idx].unsqueeze(0)`, затем `new_scores = watermark_processor(context, scores)`, и записывать обратно. Однако, это может быть медленно.
   - Так как метод HF не оптимизирован для 3D, напишем небольшой хелпер внутри `generate` (или снаружи), который будет обрабатывать 3D логиты.
   - По условиям задачи нам нужно просто использовать алгоритм из `transformers`. Мы можем вызывать стандартный `watermark_processor`, подготавливая для него `input_ids` и `scores`.
   - Создаем "псевдо-батч", где каждый элемент — это `(b_idx, seq_idx)`, который мы обновляем.
     ```python
     mask_positions = mask_index.nonzero(as_tuple=True) # (batch_indices, seq_indices)
     if len(mask_positions[0]) > 0:
         # Для каждого замаскированного токена, его "input_ids" - это префикс x[b_idx, :seq_idx]
         # Однако WatermarkLogitsProcessor из-за реализации делает enumerate(input_ids) и ожидает 2D-тензор.
         # Создадим псевдо-input_ids и псевдо-scores.
         pseudo_input_ids = []
         pseudo_scores = []
         valid_indices = []

         for b, s in zip(mask_positions[0], mask_positions[1]):
             # Проверяем, есть ли достаточный контекст
             if s >= watermark_processor.context_width:
                 pseudo_input_ids.append(x[b, :s])
                 pseudo_scores.append(logits[b, s])
                 valid_indices.append((b, s))

         if len(pseudo_input_ids) > 0:
             # Но длины pseudo_input_ids разные. Pad их?
             # WatermarkLogitsProcessor делает input_seq[-self.context_width:]
             # Значит мы можем просто отрезать последние context_width токенов
             cw = watermark_processor.context_width
             pseudo_input_ids = torch.stack([seq[-cw:] for seq in pseudo_input_ids])
             pseudo_scores = torch.stack(pseudo_scores)

             new_pseudo_scores = watermark_processor(pseudo_input_ids, pseudo_scores)

             for i, (b, s) in enumerate(valid_indices):
                 logits[b, s] = new_pseudo_scores[i]
     ```

2. **Шаги плана**:
   - Изменить `generate` в `modeling_diffusion.py` добавив аргумент `watermarking_config: Optional[WatermarkingConfig] = None`.
   - Импортировать `WatermarkingConfig` и `WatermarkLogitsProcessor` из `transformers`.
   - Внутри `generate` перед циклом инициализировать `watermark_processor = WatermarkLogitsProcessor(vocab_size=self.config.vocab_size, device=device, **watermarking_config.to_dict())` (или просто использовать `watermarking_config` параметры).
   - Внутри цикла обновления `logits` (перед `temperature` и сэмплингом) применять `watermark_processor` к 3D логитам, используя метод с `mask_positions`.
   - Написать тест в `tests/test_diffusion.py`, который вызывает генерацию с `watermarking_config` и проверяет, что вотермарк меняет логиты/вывод (например, не падает).
