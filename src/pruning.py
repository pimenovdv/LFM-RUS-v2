import json
import torch
import torch.nn as nn
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os
import tempfile
from typing import List, Dict, Any, Tuple

def prune_tokenizer_and_model(
    model_name: str,
    dataset: List[Dict[str, Any]],
    min_freq: int,
    output_dir: str
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Уменьшает размер словаря токенизатора и соответствующих эмбеддингов модели,
    оставляя только токены, встречающиеся не реже min_freq раз в dataset.

    Args:
        model_name (str): Имя или путь к базовой модели (на HuggingFace).
        dataset (List[Dict[str, Any]]): Датасет, на котором считаются частоты токенов. Должен содержать поле 'text' или 'content'.
        min_freq (int): Минимальная частота встречаемости токена для его сохранения.
        output_dir (str): Директория для сохранения урезанной модели и токенизатора.

    Returns:
        Tuple[AutoTokenizer, AutoModelForCausalLM]: Обновленный токенизатор и модель.
    """
    print("1. Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    print("2. Counting token frequencies...")
    token_counts = Counter()
    for item in tqdm(dataset):
        text = item.get("text", item.get("content", ""))
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_counts.update(ids)

    print("3. Selecting tokens to keep...")
    keep_ids = []
    special_tokens_ids = set(tokenizer.all_special_ids)
    for old_id in range(tokenizer.vocab_size):
        if old_id in special_tokens_ids or token_counts[old_id] >= min_freq:
            keep_ids.append(old_id)

    new_vocab_size = len(keep_ids)
    print(f"Old vocab size: {tokenizer.vocab_size}")
    print(f"New vocab size: {new_vocab_size}")
    print(f"To be removed: {tokenizer.vocab_size - new_vocab_size} tokens.")

    print("4. Tokenizer surgery (JSON editing)...")
    with tempfile.TemporaryDirectory() as temp_dir:
        tokenizer.save_pretrained(temp_dir)

        with open(f"{temp_dir}/tokenizer.json", "r", encoding="utf-8") as f:
            tok_data = json.load(f)

        old_vocab = tok_data["model"]["vocab"]
        id_to_tok_str = {v: k for k, v in old_vocab.items()}

        new_vocab = {}
        new_id = 0
        valid_tokens = set()
        for old_id in keep_ids:
            tok_str = id_to_tok_str[old_id]
            new_vocab[tok_str] = new_id
            valid_tokens.add(tok_str)
            new_id += 1

        tok_data["model"]["vocab"] = new_vocab

        # Also prune merges if it's BPE
        if "merges" in tok_data.get("model", {}):
            new_merges = []
            for merge in tok_data["model"]["merges"]:
                # sometimes merges are lists instead of strings? No, in tokenizer.json they should be strings like "Ġ t"
                # let's be careful
                if isinstance(merge, str):
                    parts = merge.split()
                elif isinstance(merge, (list, tuple)):
                    parts = list(merge)
                    merge = " ".join(parts) # recreate the string just in case
                else:
                    continue

                if len(parts) == 2 and parts[0] in valid_tokens and parts[1] in valid_tokens and "".join(parts) in valid_tokens:
                    new_merges.append(merge)

            tok_data["model"]["merges"] = new_merges

        with open(f"{temp_dir}/tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(tok_data, f, ensure_ascii=False, indent=2)

        pruned_tokenizer = AutoTokenizer.from_pretrained(temp_dir)

    print("5. Model matrices surgery (PyTorch)...")
    keep_indices_tensor = torch.tensor(keep_ids, dtype=torch.long)

    # 5.1 Truncate input embeddings
    old_embeddings = model.get_input_embeddings().weight.data
    new_embeddings_data = old_embeddings[keep_indices_tensor]

    new_embeddings = nn.Embedding(new_vocab_size, model.config.hidden_size, dtype=model.dtype)
    new_embeddings.weight.data = new_embeddings_data
    model.set_input_embeddings(new_embeddings)

    # 5.2 Truncate output head (if exists)
    if model.get_output_embeddings() is not None:
        old_lm_head = model.get_output_embeddings().weight.data
        new_lm_head_data = old_lm_head[keep_indices_tensor]

        new_lm_head = nn.Linear(model.config.hidden_size, new_vocab_size, bias=False, dtype=model.dtype)
        new_lm_head.weight.data = new_lm_head_data
        model.set_output_embeddings(new_lm_head)

    # 5.3 Update config
    model.config.vocab_size = new_vocab_size

    print("6. Saving final model...")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        pruned_tokenizer.save_pretrained(output_dir)
        print(f"Success! Pruned model saved to {output_dir}.")

    return pruned_tokenizer, model
