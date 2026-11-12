import pytest
from src.tokenizer import run_lexical_initialization

def test_run_lexical_initialization_focus_no_output_embeddings(mocker):
    config = {
        "use_focus": True
    }

    mock_model = mocker.MagicMock()
    mock_model.get_output_embeddings.return_value = None

    # We need to setup get_input_embeddings
    import torch
    mock_input_embeds = mocker.MagicMock()
    mock_input_embeds.weight.data = torch.ones((11, 5))
    mock_model.get_input_embeddings.return_value = mock_input_embeds

    mock_tokenizer = mocker.MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.__len__.return_value = 10

    # mock get_vocab
    mock_tokenizer.get_vocab.return_value = {str(i): i for i in range(10)}
    mock_tokenizer.convert_tokens_to_ids.return_value = 10

    mocker.patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=mock_model)
    mocker.patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer)

    class MockFastText:
        def get_word_vector(self, word):
            import numpy as np
            return np.ones(5)
        def get_dimension(self):
            return 5

    mocker.patch('fasttext.load_model', return_value=MockFastText())

    run_lexical_initialization("dummy_model", ["new_token"], config, fasttext_model_path="dummy_path")

def test_run_lexical_initialization_translation_with_output_embeddings(mocker):
    config = {
        "use_translation_init": True,
        "translation_dict": {"new_token": "translated"}
    }

    mock_model = mocker.MagicMock()

    import torch
    mock_input_embeds = mocker.MagicMock()
    mock_input_embeds.weight.data = torch.ones((11, 5))
    mock_model.get_input_embeddings.return_value = mock_input_embeds

    mock_output_embeds = mocker.MagicMock()
    mock_output_embeds.weight.data = torch.ones((11, 5))
    mock_model.get_output_embeddings.return_value = mock_output_embeds

    mock_tokenizer = mocker.MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.__len__.return_value = 10
    mock_tokenizer.encode.return_value = [0]

    # mock get_vocab
    mock_tokenizer.get_vocab.return_value = {str(i): i for i in range(10)}
    mock_tokenizer.convert_tokens_to_ids.return_value = 10

    mocker.patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=mock_model)
    mocker.patch('transformers.AutoTokenizer.from_pretrained', return_value=mock_tokenizer)

    run_lexical_initialization("dummy_model", ["new_token"], config)
