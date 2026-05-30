import torch
import fasttext
import tempfile
import os
from src.tokenizer import build_tokenizer, run_lexical_initialization

def test_build_tokenizer_digits():
    data = ["test 1234"]
    tokenizer = build_tokenizer(data, vocab_size=50)
    encoded = tokenizer.encode("1234")
    tokens = encoded.tokens

    for token in tokens:
        stripped = token.replace('Ġ', '').strip()
        if stripped.isdigit():
            assert len(stripped) == 1, f"Digit not split properly: {token}"

def test_run_lexical_initialization_focus():
    config = {
        "use_translation_init": False
    }

    model_name = "sshleifer/tiny-gpt2"
    new_tokens = ["Ġкошка", "ство"]

    # Create a dummy fasttext model
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write("cat кошка dog собака\n")
        f.write("tree дерево\n")
        f.write("test ство\n")
        tmp_name = f.name

    ft_model = fasttext.train_unsupervised(tmp_name, model='cbow', dim=16, epoch=1, minCount=1)
    ft_model_path = tmp_name + ".bin"
    ft_model.save_model(ft_model_path)

    try:
        tokenizer, model = run_lexical_initialization(
            model_name,
            new_tokens,
            config,
            fasttext_model_path=ft_model_path,
            save_path=None
        )

        assert "Ġкошка" in tokenizer.get_vocab()

        ru_id = tokenizer.convert_tokens_to_ids("Ġкошка")
        input_embeddings = model.get_input_embeddings().weight.data

        # Check that it is not nan or zero
        assert not torch.isnan(input_embeddings[ru_id]).any()
        assert torch.norm(input_embeddings[ru_id]) > 0
    finally:
        os.remove(tmp_name)
        os.remove(ft_model_path)
