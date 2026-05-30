# Настройки Task SFT (task_sft.yaml)

Этот файл содержит конфигурацию для целевого дообучения модели (Task SFT), с поддержкой адаптации через LoRA (Low-Rank Adaptation).

## Параметры

* **`stage`** (строка): Определяет этап пайплайна.
  * *Пример*: `"task_sft"`
* **`use_lora`** (булево): Использовать ли LoRA для обучения (parameter-efficient fine-tuning).
  * *Пример*: `true`
* **`model_name`** (строка): Имя или путь к базовой модели (например, с Hugging Face Hub).
  * *Пример*: `"sshleifer/tiny-gpt2"`
* **`max_seq_length`** (целое число): Максимальная длина последовательности токенов для обучения.
  * *Пример*: `512`
* **`packing`** (булево): Использовать ли упаковку последовательностей (sequence packing).
  * *Пример*: `true`
* **`learning_rate`** (число с плавающей точкой): Скорость обучения.
  * *Пример*: `2.0e-5`
* **`epochs`** (целое число): Количество эпох обучения.
  * *Пример*: `1`
* **`batch_size`** (целое число): Размер пакета (batch size) на устройство для обучения.
  * *Пример*: `4`
* **`max_steps`** (целое число): Максимальное количество шагов обучения (если больше 0, переопределяет epochs).
  * *Пример*: `1000`
* **`save_steps`** (целое число): Интервал (в шагах) сохранения чекпоинтов модели.
  * *Пример*: `1000`
* **`logging_steps`** (целое число): Интервал (в шагах) для вывода метрик (loss, learning rate и др.).
  * *Пример*: `10`
* **`output_dir`** (строка): Директория для сохранения обученной модели и чекпоинтов.
  * *Пример*: `"./task-sft-output"`
* **`lora`** (словарь): Конфигурация для LoRA (если `use_lora: true`).
  * `r` (целое число): Ранг матриц обновления LoRA.
  * `lora_alpha` (целое число): Параметр масштабирования LoRA.
  * `target_modules` (список строк): Модули архитектуры трансформера, к которым будет применена LoRA (например, attention проекции).
  * `lora_dropout` (число с плавающей точкой): Вероятность dropout для слоев LoRA.
  * `bias` (строка): Указывает, нужно ли обучать смещения (biases). Обычно `"none"`.
  * `task_type` (строка): Тип задачи, для `peft`. Для языковых моделей обычно `"CAUSAL_LM"`.
  * *Пример*:
    ```yaml
    r: 8
    lora_alpha: 32
    target_modules: ["c_attn", "c_proj"]
    lora_dropout: 0.1
    bias: "none"
    task_type: "CAUSAL_LM"
    ```
* **`dataset_paths`** (словарь строк): Пути к датасетам для конкретных задач (task).
  * *Пример*:
    ```yaml
    task1: "path/to/task1"
    ```

## Пример использования
```yaml
stage: task_sft
use_lora: true
model_name: "sshleifer/tiny-gpt2"
max_seq_length: 512
packing: true
learning_rate: 2.0e-5
epochs: 1
batch_size: 4
max_steps: 1000
save_steps: 1000
logging_steps: 10
output_dir: "./task-sft-output"
lora:
  r: 8
  lora_alpha: 32
  target_modules: ["c_attn", "c_proj"]
  lora_dropout: 0.1
  bias: "none"
  task_type: "CAUSAL_LM"
dataset_paths:
  task1: "path/to/task1"
```
