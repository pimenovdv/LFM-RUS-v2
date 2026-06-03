import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Digits, Whitespace, Sequence
from tokenizers.trainers import BpeTrainer
import fasttext
from entmax import sparsemax
from typing import Iterator, List, Dict, Any, Tuple, Optional

def build_tokenizer(data_iterator: Iterator[str], vocab_size: int = 5000) -> Tokenizer:
    """
    Обучает новый BPE токенизатор на основе переданных данных.

    Args:
        data_iterator (Iterator[str]): Итератор по текстовым данным для обучения.
        vocab_size (int, optional): Целевой размер словаря. По умолчанию 5000.

    Returns:
        Tokenizer: Обученный токенизатор.
    """
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Sequence([
        Whitespace(),
        Digits(individual_digits=True)
    ])
    trainer = BpeTrainer(special_tokens=["<unk>", "<s>", "</s>", "<pad>"], vocab_size=vocab_size)
    tokenizer.train_from_iterator(data_iterator, trainer=trainer)
    return tokenizer

def run_lexical_initialization(
    model_name: str,
    new_tokens: List[str],
    config: Dict[str, Any],
    fasttext_model_path: Optional[str] = None,
    save_path: Optional[str] = "./lfm-russian-lexical"
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Выполняет лексическую инициализацию (добавление новых токенов и инициализацию их эмбеддингов).
    Использует перевод (translation_dict), подход FOCUS (sparsemax поверх fasttext эмбеддингов) или fallback (среднее).

    Args:
        model_name (str): Имя или путь к базовой модели (например, на HuggingFace).
        new_tokens (List[str]): Список новых токенов для добавления в словарь.
        config (Dict[str, Any]): Словарь конфигурации (содержит настройки use_translation_init, translation_dict).
        fasttext_model_path (Optional[str]): Путь к обученной модели fastText для инициализации FOCUS.
        save_path (Optional[str]): Путь для сохранения модели и токенизатора после инициализации.

    Returns:
        Tuple[AutoTokenizer, AutoModelForCausalLM]: Обновленный токенизатор и модель.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    old_vocab_size = len(tokenizer)
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data if model.get_output_embeddings() is not None else None

    # Overlapping tokens mapping (we skip actually verifying them all here, we assume anything in original vocab is overlapping)
    # We will compute fallback mean for those completely absent
    default_mean_in = input_embeddings[:old_vocab_size].mean(dim=0)
    if output_embeddings is not None:
        default_mean_out = output_embeddings[:old_vocab_size].mean(dim=0)

    use_translation_init = config.get("use_translation_init", False)
    translation_dict = config.get("translation_dict", {})

    ft_model = None
    if fasttext_model_path:
        ft_model = fasttext.load_model(fasttext_model_path)

    overlapping_tokens = list(tokenizer.get_vocab().keys())[:old_vocab_size]

    # Gather overlapping token FT embeddings if ft_model is available
    overlap_ft_embs = None
    if ft_model is not None:
        # Build tensor of fasttext embeddings for all old tokens
        overlap_embs = []
        for t in overlapping_tokens:
            word = t.replace('Ġ', '').replace(' ', '').strip()
            if not word:
                overlap_embs.append(torch.zeros(ft_model.get_dimension()))
            else:
                overlap_embs.append(torch.tensor(ft_model.get_word_vector(word)))
        overlap_ft_embs = torch.stack(overlap_embs)
        # normalize
        overlap_ft_embs = F.normalize(overlap_ft_embs, p=2, dim=1)

    for new_tok in new_tokens:
        ru_tok_id = tokenizer.convert_tokens_to_ids(new_tok)
        clean_word = new_tok.replace("Ġ", "").replace(" ", "").strip()

        # 1. Exact match translation (override)
        if use_translation_init and clean_word in translation_dict:
            en_trans = translation_dict[clean_word]
            en_ids = tokenizer.encode(en_trans, add_special_tokens=False)
            if len(en_ids) > 0:
                # Need to make sure en_ids are within old_vocab bounds for stability,
                # but usually they are since they are base english tokens.
                valid_ids = [eid for eid in en_ids if eid < old_vocab_size]
                if valid_ids:
                    input_embeddings[ru_tok_id] = input_embeddings[valid_ids].mean(dim=0)
                    if output_embeddings is not None:
                        output_embeddings[ru_tok_id] = output_embeddings[valid_ids].mean(dim=0)
                    continue

        # 2. FOCUS sparsemax
        if ft_model is not None and clean_word:
            new_ft_vec = torch.tensor(ft_model.get_word_vector(clean_word)).unsqueeze(0)
            new_ft_vec = F.normalize(new_ft_vec, p=2, dim=1)

            # Cosine similarity
            # overlap_ft_embs is [V_old, D], new_ft_vec is [1, D]
            cos_sims = torch.mm(new_ft_vec, overlap_ft_embs.T).squeeze(0) # [V_old]

            # Applying sparsemax
            weights = sparsemax(cos_sims, dim=0) # [V_old]

            # New embedding is weighted sum
            mapped_in = torch.sum(weights.unsqueeze(1) * input_embeddings[:old_vocab_size], dim=0)
            input_embeddings[ru_tok_id] = mapped_in
            if output_embeddings is not None:
                mapped_out = torch.sum(weights.unsqueeze(1) * output_embeddings[:old_vocab_size], dim=0)
                output_embeddings[ru_tok_id] = mapped_out
            continue

        # 3. Fallback
        input_embeddings[ru_tok_id] = default_mean_in
        if output_embeddings is not None:
            output_embeddings[ru_tok_id] = default_mean_out

    print("Lexical Initialization завершена!")
    if save_path:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    return tokenizer, model
