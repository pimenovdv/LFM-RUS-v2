Реализация Lexical Initialization (также известной как Cross-lingual Word Embedding Alignment) — это одна из самых изящных задач в NLP. Суть в том, чтобы передать новому русскому токену "опыт" его английского брата, не обучая модель с нуля.
Архитектурно процесс выглядит так:
 1. Очистка русских токенов
   Новые токены часто содержат служебные символы (например, Ġ для пробела в BPE). Перед переводом их нужно удалить. Токен Ġкошка должен стать словом кошка.
 2. Массовый перевод токенов
   Инструмент: MarianMT, DeepL API или FastText
   Для каждого чистого русского токена мы получаем английский перевод. Сложность здесь в морфемах: если токен — это целое слово ("собака" \rightarrow "dog"), все отлично. Если это кусок слова ("ство", "тся"), переводчик выдаст мусор. Такие "непереводимые" токены нужно отфильтровать.
 3. Токенизация английского перевода базовым словарем
   Мы берем английское слово (например, "cat") и прогоняем его через старый токенизатор модели. Важно понимать, что английское слово тоже может разбиться на 2-3 саб-токена, если оно редкое.
 4. Усреднение и копирование весов
   Если английское слово разбилось на токены [A, B, C], мы берем их векторы из базовой матрицы эмбеддингов, складываем, делим на 3 (получаем средний смысл) и записываем этот вектор в ячейку нашего нового русского токена.
 5. Fallback (Запасной план)
   Все те русские окончания, слоги и знаки, которые не удалось адекватно перевести, получают "средний вектор по больнице" (усредненное значение всех английских токенов), чтобы они не начинали обучение с деструктивного шума.
## Рабочий код на PyTorch и Transformers
Вот готовый скрипт, который выполняет эту операцию. В реальном проекте вместо заглушки-словаря translation_dict вы будете использовать вызов локальной модели-переводчика (например, Helsinki-NLP/opus-mt-ru-en) для массового перевода.
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Загружаем модель и базовый токенизатор
model_name = "liquid-lfm-base" # Замените на вашу модель
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

old_vocab_size = len(tokenizer)

# 2. Допустим, мы уже обучили новый русский токенизатор 
# и отфильтровали 10 000 новых кириллических токенов
new_ru_tokens = ["Ġкошка", "Ġсобака", "ство", "Ġнейросеть", ...] 
tokenizer.add_tokens(new_ru_tokens)
model.resize_token_embeddings(len(tokenizer))

# 3. Получаем доступ к весам
input_embeddings = model.get_input_embeddings().weight.data
output_embeddings = model.get_output_embeddings().weight.data # Голова модели

# Вычисляем дефолтный средний вектор для "непереводимых" токенов
default_mean_in = input_embeddings[:old_vocab_size].mean(dim=0)
default_mean_out = output_embeddings[:old_vocab_size].mean(dim=0)

# 4. Имитация словаря переводов (на практике здесь работает скрипт-переводчик)
# Формат: {чистый_русский_токен: английский_перевод}
translation_dict = {
    "кошка": "cat",
    "собака": "dog",
    "нейросеть": "neural network"
}

# 5. Главный цикл маппинга
for new_tok in new_ru_tokens:
    # Получаем ID нового токена в расширенном словаре
    ru_tok_id = tokenizer.convert_tokens_to_ids(new_tok)
    
    # Очищаем токен от спецсимволов токенизатора (SentencePiece / BPE)
    clean_ru_word = new_tok.replace("Ġ", "").replace(" ", "").strip()
    
    # Ищем перевод
    en_translation = translation_dict.get(clean_ru_word)
    
    if en_translation:
        # Токенизируем английский перевод СТАРЫМ словарем
        # add_special_tokens=False, чтобы не прихватить токены <s> или [CLS]
        en_tok_ids = tokenizer.encode(en_translation, add_special_tokens=False)
        
        # Защита: если переводчик выдал что-то странное
        if len(en_tok_ids) > 0:
            # Извлекаем эмбеддинги всех английских саб-токенов
            en_embeds_in = input_embeddings[en_tok_ids]
            en_embeds_out = output_embeddings[en_tok_ids]
            
            # Усредняем их
            mapped_vec_in = en_embeds_in.mean(dim=0)
            mapped_vec_out = en_embeds_out.mean(dim=0)
            
            # Копируем веса в новый русский токен
            input_embeddings[ru_tok_id] = mapped_vec_in
            output_embeddings[ru_tok_id] = mapped_vec_out
            continue # Успешно смапили, идем к следующему
            
    # Если перевода нет или это просто морфема (Fallback)
    input_embeddings[ru_tok_id] = default_mean_in
    output_embeddings[ru_tok_id] = default_mean_out

print("Lexical Initialization завершена!")
model.save_pretrained("./lfm-russian-lexical")
tokenizer.save_pretrained("./lfm-russian-lexical")

```
> **Важный технический нюанс:** Не забудьте обновить веса обеих матриц. Матрица input_embeddings переводит текст в векторы, а output_embeddings (часто называется lm_head) переводит внутреннее состояние модели обратно в вероятности слов. Если вы обновите только первую, модель научится *читать* по-русски, но будет генерировать мусор.
> 
## Что происходит с моделью после такого скрипта?
Если вы сразу же попробуете с ней поговорить, она все еще не сможет свободно общаться на русском. Но! На этапе Continual Pre-Training ее loss (ошибка) упадет до приемлемых значений не за 5 000 шагов, а за 200–300. Вы сэкономите десятки часов аренды дорогих GPU. Модель буквально скажет: *"А, так 'собака' — это то же самое, что 'dog', только склоняется иначе! Понял!"*
