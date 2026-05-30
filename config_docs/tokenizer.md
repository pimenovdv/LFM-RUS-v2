# Настройки Tokenizer (tokenizer.yaml)

Этот файл описывает конфигурацию для этапа обучения или инициализации токенизатора.

## Параметры

* **`stage`** (строка): Определяет этап пайплайна.
  * *Пример*: `"tokenizer"`
* **`model_name`** (строка): Имя или путь к базовой модели, токенизатор которой можно использовать как основу или для переноса.
  * *Пример*: `"sshleifer/tiny-gpt2"`
* **`vocab_size`** (целое число): Желаемый размер словаря (vocabulary size) токенизатора.
  * *Пример*: `50`
* **`focus`** (булево): Флаг, определяющий, нужно ли "фокусировать" токенизатор на определенном языке или домене.
  * *Пример*: `true`
* **`use_translation_init`** (булево): Использовать ли инициализацию эмбеддингов для новых токенов на основе перевода существующих токенов (из базовой модели).
  * *Пример*: `true`
* **`dataset_ratios`** (словарь чисел): Пропорции датасетов для обучения токенизатора (сумма должна быть равна 1.0).
  * *Пример*:
    ```yaml
    ru: 0.40
    en: 0.30
    code: 0.15
    math: 0.15
    ```
* **`dataset_paths`** (словарь строк): Пути к датасетам (или названия датасетов в библиотеке Hugging Face `datasets`), соответствующие ключам из `dataset_ratios`.
  * *Пример*:
    ```yaml
    ru: "oscar"
    en: "wikitext"
    code: "codeparrot/github-code"
    math: "math-dataset/math-dataset"
    ```
* **`translation_dict`** (словарь строк): Словарь переводов слов с целевого языка на базовый (используется, если `use_translation_init: true`). Ключи - новые слова, значения - известные слова из базовой модели.
  * *Пример*:
    ```yaml
    "кошка": "cat"
    "собака": "dog"
    ```
* **`save_path`** (строка или null): Путь для сохранения обученного токенизатора. Если `null`, токенизатор может не сохраняться локально или сохраняться в путь по умолчанию.
  * *Пример*: `null`

## Пример использования
```yaml
stage: tokenizer
model_name: "sshleifer/tiny-gpt2"
vocab_size: 50
focus: true
use_translation_init: true
dataset_ratios:
  ru: 0.40
  en: 0.30
  code: 0.15
  math: 0.15
dataset_paths:
  ru: "oscar"
  en: "wikitext"
  code: "codeparrot/github-code"
  math: "math-dataset/math-dataset"
translation_dict:
  "кошка": "cat"
  "собака": "dog"
  "нейросеть": "neural network"
  "обучение": "learning"
save_path: null
```
