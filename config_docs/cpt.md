# Настройки CPT (cpt.yaml)

Этот файл описывает конфигурацию для этапа непрерывного предварительного обучения (Continual Pre-Training).

## Параметры

* **`stage`** (строка): Определяет этап пайплайна.
  * *Пример*: `"cpt"`
* **`model_name`** (строка): Имя или путь к базовой модели (например, с Hugging Face Hub).
  * *Пример*: `"sshleifer/tiny-gpt2"`
* **`dataset_paths`** (словарь строк): Пути к датасетам для разных языков и доменов.
  * *Пример*:
    ```yaml
    ru: "data/ru_corpus"
    en: "data/en_corpus"
    ```
* **`dataset_ratios`** (словарь чисел): Пропорции (веса) для смешивания датасетов, указанных в `dataset_paths`. Должны в сумме давать 1.0.
  * *Пример*:
    ```yaml
    ru: 0.40
    en: 0.30
    ```
* **`embedding_warmup`** (словарь): Настройки фазы предварительного прогрева (warm-up) эмбеддингов, при которой веса остальной части модели заморожены.
  * `enabled` (булево): Включен ли warmup.
  * `epochs` (целое число): Количество эпох warmup.
  * `learning_rate` (число с плавающей точкой): Скорость обучения во время warmup.
  * *Пример*:
    ```yaml
    enabled: true
    epochs: 1
    learning_rate: 0.001
    ```
* **`output_dir`** (строка): Директория для сохранения результатов (модели, чекпоинтов).
  * *Пример*: `"./cpt-output"`
* **`epochs`** (целое число): Общее количество эпох основного обучения.
  * *Пример*: `3`
* **`learning_rate`** (число с плавающей точкой): Скорость обучения для основного этапа CPT.
  * *Пример*: `0.0001`
* **`per_device_train_batch_size`** (целое число): Размер пакета на каждое устройство при обучении.
  * *Пример*: `4`
* **`save_steps`** (целое число): Интервал в шагах для сохранения чекпоинтов.
  * *Пример*: `1000`

## Пример использования
```yaml
stage: cpt
model_name: "sshleifer/tiny-gpt2"
dataset_paths:
  ru: "data/ru_corpus"
  en: "data/en_corpus"
  code: "data/code_corpus"
  math: "data/math_corpus"
dataset_ratios:
  ru: 0.40
  en: 0.30
  code: 0.15
  math: 0.15
embedding_warmup:
  enabled: true
  epochs: 1
  learning_rate: 0.001
output_dir: "./cpt-output"
epochs: 3
learning_rate: 0.0001
per_device_train_batch_size: 4
save_steps: 1000
```
