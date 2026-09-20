1. **Сжать выполненные пункты и добавить новый пункт (Шаг 85) в `todo.md`**
   - Это я уже сделал в предыдущем вызове `cat << 'EOF' > todo.md`.
2. **Реализовать расписания для Mirostat (Шаг 85) в `src/models/diffusion/modeling_diffusion.py`**
   - Добавить параметры `mirostat_tau_schedule="constant"`, `min_mirostat_tau=0.0`, `mirostat_eta_schedule="constant"`, `min_mirostat_eta=0.0` в `generate`.
   - В цикле генерации вычислять `current_mirostat_tau` и `current_mirostat_eta` с использованием расписаний (`linear`, `cosine`, `exponential`, `cyclic`), аналогично другим параметрам.
   - Использовать `current_mirostat_tau` и `current_mirostat_eta` вместо статичных параметров для обновления `current_mirostat_mu`.
3. **Написать тесты для расписаний Mirostat в `tests/test_diffusion_mirostat.py` (или аналогичном файле)**
   - Проверить, что генерация с расписаниями `linear`, `cosine`, `exponential`, `cyclic` для `mirostat_tau` и `mirostat_eta` проходит успешно без ошибок и возвращает тензоры правильной формы.
4. **Выполнить проверки (pre-commit)**
   - Проверить, что тесты проходят успешно с использованием `uv run pytest tests/ -k "mirostat"`.
   - Убедиться, что покрытие кода тестами остается выше 90%.
5. **Отметить пункт в `todo.md` как выполненный.**
6. **Закоммитить изменения и завершить задачу.**
