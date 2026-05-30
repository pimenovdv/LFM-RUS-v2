# Настройки Data Prep (data_prep.yaml)

Этот файл описывает конфигурацию для этапа подготовки данных (очистка, фильтрация, дедупликация) с использованием библиотеки DataTrove.

## Параметры

* **`stage`** (строка): Определяет этап пайплайна.
  * *Пример*: `"data_prep"`
* **`method`** (строка): Метод, используемый для дедупликации данных.
  * *Пример*: `"minhash_lsh"`
* **`input_path`** (строка): Путь к директории с исходными (сырыми) данными.
  * *Пример*: `"./data/raw"`
* **`output_path`** (строка): Путь к директории для сохранения обработанных и дедуплицированных данных.
  * *Пример*: `"./data/deduplicated"`
* **`minhash_base_path`** (строка): Директория для хранения промежуточных файлов MinHash.
  * *Пример*: `"./data/minhash"`
* **`minhash_config`** (словарь): Настройки алгоритма MinHash.
  * `n_grams` (целое число): Размер n-грамм для разбиения текста.
  * `num_buckets` (целое число): Количество корзин (buckets) для LSH.
  * `hashes_per_bucket` (целое число): Количество хешей в одной корзине.
  * `precision` (целое число): Точность хешей (в битах).
  * *Пример*:
    ```yaml
    n_grams: 5
    num_buckets: 14
    hashes_per_bucket: 8
    precision: 64
    ```
* **`filters`** (словарь): Настройки различных фильтров очистки данных (интеграция с DataTrove).
  * `remove_seo` (булево): Удаление SEO-спама.
  * `remove_logs` (булево): Удаление логов.
  * `remove_cyclic` (булево): Удаление зацикленного текста.
  * `fasttext_spam` (словарь): Настройки фильтра спама на основе FastText.
    * `enabled` (булево): Включить фильтр.
    * `model_url` (строка): Ссылка на модель (например, на HF Hub).
  * `fineweb_quality` (словарь): Настройки фильтрации качества текста, специфичной для FineWeb.
    * `enabled` (булево): Включить фильтр.
  * `transformers_classifier` (словарь): Настройки фильтра классификатора на базе библиотеки `transformers`.
    * `enabled` (булево): Включить фильтр.
    * `model_name` (строка): Имя модели классификатора на HF Hub.
    * `batch_size` (целое число): Размер пакета для инференса.
    * `keep_labels` (список списков): Метки классов и пороги вероятности для сохранения текста (например, сохранить классы "4" и "5" с вероятностью выше 0.5).

## Пример использования
```yaml
stage: data_prep
method: "minhash_lsh"
input_path: "./data/raw"
output_path: "./data/deduplicated"
minhash_base_path: "./data/minhash"
minhash_config:
  n_grams: 5
  num_buckets: 14
  hashes_per_bucket: 8
  precision: 64
filters:
  remove_seo: true
  remove_logs: true
  remove_cyclic: true
  fasttext_spam:
    enabled: true
    model_url: "hf://datatrove/fasttext-spam/model.bin"
  fineweb_quality:
    enabled: true
  transformers_classifier:
    enabled: true
    model_name: "HuggingFaceTB/fineweb-edu-classifier"
    batch_size: 32
    keep_labels:
      - ["4", 0.5]
      - ["5", 0.5]
```
