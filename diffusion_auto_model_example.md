Чтобы сделать диффузионную обертку абсолютно универсальной для любой архитектуры (Llama, Qwen, GPT2, LFM и др.) и обеспечить корректную загрузку через AutoModel.from_pretrained(path, trust_remote_code=True), необходимо использовать два главных механизма экосистемы Hugging Face:
 1. **Абстракции стандартных методов HF** (get_input_embeddings() и аргумент inputs_embeds), которые одинаковы для всех моделей.
 2. **Систему auto_map**, которая автоматически сохраняет код вашего кастомного класса прямо в папку с весами модели, избавляя от необходимости импортировать его вручную в коде инференса.
Ниже представлена готовая архитектура универсального класса-обертки.
### 1. Реализация универсального конфигуратора и модели
Код ниже полностью абстрагирован от конкретных имен слоев (вроде wte или embed_tokens), так как использует стандартный интерфейс PreTrainedModel.
```python
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel, AutoConfig, AutoModel

# ==========================================
# 1. УНИВЕРСАЛЬНЫЙ КОНФИГ
# ==========================================
class UniversalDiffusionConfig(PretrainedConfig):
    model_type = "universal_diffusion_lm"

    def __init__(
        self,
        base_config_dict=None,
        timestep_dim=256,
        max_timesteps=1000,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_config_dict = base_config_dict or {}
        self.timestep_dim = timestep_dim
        self.max_timesteps = max_timesteps

# ==========================================
# 2. УНИВЕРСАЛЬНАЯ МОДЕЛЬ-ОБЕРТКА
# ==========================================
class UniversalDiffusionLM(PreTrainedModel):
    config_class = UniversalDiffusionConfig

    def __init__(self, config: UniversalDiffusionConfig):
        super().__init__(config)
        
        # Динамически загружаем конфиг базовой модели (Llama, Qwen, GPT2 и т.д.)
        base_config = AutoConfig.for_model(**config.base_config_dict)
        
        # Инициализируем "голую" базовую модель без LM-головы
        self.base_model = AutoModel.from_config(base_config)
        
        # Отключаем причинную (causal) маску внимания для двунаправленности
        self._disable_causal_mask(base_config.model_type)
        
        # Универсальный эмбеддер шагов времени t
        self.timestep_embedder = nn.Sequential(
            nn.Linear(1, config.timestep_dim),
            nn.SiLU(),
            nn.Linear(config.timestep_dim, base_config.hidden_size)
        )
        
        # Голова денойзинга (проекция скрытых состояний обратно в словарь токенов)
        self.lm_head = nn.Linear(base_config.hidden_size, base_config.vocab_size, bias=False)
        
        # Привязываем веса головы к входным эмбеддингам (Weight Tying), если это необходимо
        if getattr(base_config, "tie_word_embeddings", False):
            self.lm_head.weight = self.base_model.get_input_embeddings().weight

    def _disable_causal_mask(self, model_type):
        """
        Динамический патчинг базовой модели для отключения казуальности.
        Разные архитектуры в HF реализуют маску по-разному.
        """
        # Для современных моделей на базе Llama (Llama 2/3, Mistral, Qwen2, Gemma)
        if hasattr(self.base_model, "_update_causal_mask"):
            # Подменяем метод генерации маски на "пустышку", возвращающую обычную маску
            self.base_model._update_causal_mask = lambda *args, **kwargs: args[1] if len(args) > 1 else kwargs.get("attention_mask")
            
        # Для старых моделей вроде GPT2
        elif model_type == "gpt2":
            for module in self.base_model.modules():
                if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor):
                    # Заполняем треугольную маску единицами, делая внимание bidirectional
                    module.bias.data.fill_(1)
                    
        # Для других архитектур логику сброса маски можно расширять здесь...

    def forward(self, input_ids, timesteps, attention_mask=None):
        # ВАЖНО: Получаем эмбеддинги через универсальный метод HF API
        embeddings_layer = self.base_model.get_input_embeddings()
        inputs_embeds = embeddings_layer(input_ids)
        
        # Добавляем эмбеддинг времени t к эмбеддингам токенов
        t_embed = self.timestep_embedder(timesteps.unsqueeze(-1).float()) # [batch, hidden_size]
        hidden_states = inputs_embeds + t_embed.unsqueeze(1)              # [batch, seq_len, hidden_size]
        
        # Передаем данные через `inputs_embeds`. Этот аргумент поддерживают 99% моделей в HF.
        outputs = self.base_model(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Получаем логиты через денойзинг-голову
        logits = self.lm_head(outputs.last_hidden_state)
        return logits

```
### 2. Как правильно сохранить модель, чтобы сработал trust_remote_code
Чтобы Hugging Face автоматически создал файлы кода в целевой папке и прописал их в config.json (создав карту auto_map), нужно использовать метод .register_for_auto_class().
Выполните этот шаг **перед** сохранением обученной модели:
```python
# 1. Регистрируем кастомные классы в глобальном реестре автоклассов Hugging Face
UniversalDiffusionConfig.register_for_auto_class("AutoConfig")
UniversalDiffusionLM.register_for_auto_class("AutoModel")

# 2. Создаем конфиг нашей диффузионной модели, передавая внутрь конфиг базовой архитектуры
base_model_name = "meta-llama/Meta-Llama-3-8B" # Или "Qwen/Qwen2-0.5B", или ваша LFM-модель
base_config = AutoConfig.from_pretrained(base_model_name)

diffusion_config = UniversalDiffusionConfig(
    base_config_dict=base_config.to_dict(),
    timestep_dim=256,
    max_timesteps=1000
)

# 3. Инициализируем модель (или берем уже обученную в вашем фреймворке)
model = UniversalDiffusionLM(diffusion_config)

# ... [Процесс обучения вашего Diffusion LM] ...

# 4. Сохранение
output_dir = "./universal_diffusion_model_checkpoint"
model.save_pretrained(output_dir)

```
**Что произойдет при сохранении:**
Библиотека transformers заметит регистрацию и автоматически выполнит следующие действия:
 1. Скопирует исходный код классов UniversalDiffusionConfig и UniversalDiffusionLM в файлы configuration_universal_diffusion_lm.py и modeling_universal_diffusion_lm.py прямо внутри папки ./universal_diffusion_model_checkpoint.
 2. В файле config.json автоматически появится секция "auto_map", указывающая на эти файлы:
```json
"auto_map": {
  "AutoConfig": "configuration_universal_diffusion_lm.UniversalDiffusionConfig",
  "AutoModel": "modeling_universal_diffusion_lm.UniversalDiffusionLM"
}

```
### 3. Как теперь выглядит универсальный инференс
После такого сохранения абсолютно не важно, какая модель была внутри (Llama, GPT2 или Qwen). Любой сторонний пользователь (или ваш скрипт инференса) может загрузить её стандартной командой без импорта кастомных python-файлов:
```python
from transformers import AutoModel, AutoTokenizer
import torch

path = "./universal_diffusion_model_checkpoint"

# Модель автоматически подтянет правильную архитектуру через trust_remote_code
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModel.from_pretrained(path, trust_remote_code=True)

model.to("cuda")
model.eval()

# Пример подачи данных (прямой проход денойзинга)
input_ids = tokenizer("Пример зашумленного текста", return_tensors="pt").input_ids.to("cuda")
timesteps = torch.tensor([500]).to("cuda") # Шаг диффузии t=500

with torch.no_grad():
    logits = model(input_ids=input_ids, timesteps=timesteps)
    print(logits.shape) # [batch_size, seq_len, vocab_size]

```
