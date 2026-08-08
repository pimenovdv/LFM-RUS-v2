1. **Обновление аргументов функции `generate`**:
   - В `src/models/diffusion/modeling_diffusion.py` добавить в метод `generate` два новых аргумента: `encoder_repetition_penalty: float = 1.0` и `encoder_no_repeat_ngram_size: int = 0`.
2. **Реализация `encoder_repetition_penalty`**:
   - Там же, где применяется `repetition_penalty` (около строки 541), добавить проверку `if encoder_repetition_penalty != 1.0`.
   - Если параметр включен, нужно взять токены из промпта (то есть `x[b, :T]`), отфильтровать их (можно опционально применить `penalty_range` или брать весь промпт).
   - Вычислить штрафы к логитам для уникальных токенов промпта, аналогично `repetition_penalty`: `penalized_score = torch.where(score < 0, score * encoder_repetition_penalty, score / encoder_repetition_penalty)`.
3. **Реализация `encoder_no_repeat_ngram_size`**:
   - Там же, где применяется `no_repeat_ngram_size` (около строки 472), добавить блок для `encoder_no_repeat_ngram_size > 0`.
   - Логика немного отличается: нам нужно запретить генерировать токены, которые создают n-грамму, уже существующую в **промпте** (`x[b, :T]`).
   - Для каждой позиции `pos` в текущем блоке:
     - Берем `pos - (encoder_no_repeat_ngram_size - 1)` токенов, предшествующих `pos`. Если среди них есть `mask_id`, пропускаем.
     - Ищем эту последовательность (префикс) в промпте (`x[b, :T]`). Если находим, то токен, который следует за ней в промпте, добавляется в `banned_tokens`.
     - Зануляем вероятности (ставим `-float("Inf")`) для всех `banned_tokens`.
4. **Тестирование**:
   - В `tests/test_mdlm_generation.py` добавить тесты `test_encoder_repetition_penalty` и `test_encoder_no_repeat_ngram_size`.
   - Убедиться, что при включенном `encoder_repetition_penalty` логиты токенов из промпта уменьшаются.
   - Убедиться, что при включенном `encoder_no_repeat_ngram_size` модель не генерирует n-граммы, присутствующие в промпте.
5. **Pre-commit и Submit**:
   - Запустить pre commit скрипты.
   - Закоммитить изменения.
