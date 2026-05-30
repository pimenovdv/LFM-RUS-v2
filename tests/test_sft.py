import pytest
from unittest.mock import MagicMock
from src.sft import run_sft

def test_run_sft_dummy_data(mocker):
    mock_tokenizer = mocker.patch("src.sft.AutoTokenizer.from_pretrained")
    mock_model = mocker.patch("src.sft.AutoModelForCausalLM.from_pretrained")
    mock_trainer = mocker.patch("src.sft.SFTTrainer")
    mock_sft_config = mocker.patch("src.sft.SFTConfig")
    mocker.patch('transformers.PreTrainedModel.push_to_hub', return_value=None)
    mocker.patch('transformers.PreTrainedTokenizerBase.push_to_hub', return_value=None)
    mocker.patch('trl.SFTTrainer.push_to_hub', return_value=None)

    mock_tok_inst = MagicMock()
    mock_tok_inst.pad_token = None
    mock_tok_inst.eos_token = "<|endoftext|>"
    mock_tok_inst.chat_template = None
    mock_tokenizer.return_value = mock_tok_inst

    mock_mod_inst = MagicMock()
    mock_model.return_value = mock_mod_inst

    cfg = {
        "model_name": "dummy_model",
        "max_seq_length": 128,
        "packing": True,
        "output_dir": "./dummy_sft_output",
        "learning_rate": 0.0001,
        "epochs": 1,
        "batch_size": 2,
        "save_steps": 10,
        "logging_steps": 5
    }

    cfg['push_to_hub'] = 'dummy/sft'
    run_sft(cfg, dummy_data=True)

    mock_tokenizer.assert_called_once_with("dummy_model")
    mock_model.assert_called_once_with("dummy_model")
    mock_trainer.assert_called_once()
    mock_trainer.return_value.train.assert_called_once()
    # Check that chat template is set correctly for EOS control
    assert mock_tok_inst.chat_template is not None
    mock_mod_inst.resize_token_embeddings.assert_called_once()


def test_run_sft_real_data(mocker):
    mock_tokenizer = mocker.patch("src.sft.AutoTokenizer.from_pretrained")
    mock_model = mocker.patch("src.sft.AutoModelForCausalLM.from_pretrained")
    mock_trainer = mocker.patch("src.sft.SFTTrainer")
    mock_sft_config = mocker.patch("src.sft.SFTConfig")
    mocker.patch('transformers.PreTrainedModel.push_to_hub', return_value=None)
    mocker.patch('transformers.PreTrainedTokenizerBase.push_to_hub', return_value=None)
    mocker.patch('trl.SFTTrainer.push_to_hub', return_value=None)
    mock_load_dataset = mocker.patch("src.sft.load_dataset")
    mock_interleave = mocker.patch("src.sft.interleave_datasets")

    mock_ds = MagicMock()
    mock_interleave.return_value = mock_ds
    mock_load_dataset.return_value = mock_ds

    cfg = {
        "model_name": "dummy_model",
        "dataset_paths": {
            "ru": "path/ru"
        }
    }

    run_sft(cfg, dummy_data=False)

    mock_load_dataset.assert_called_once_with("path/ru", split="train", streaming=True)
    mock_interleave.assert_called_once()
    mock_trainer.return_value.train.assert_called_once()

def test_run_sft_no_datasets(mocker):
    mock_tokenizer = mocker.patch("src.sft.AutoTokenizer.from_pretrained")

    cfg = {
        "model_name": "dummy_model",
        "dataset_paths": {}
        # missing dataset_path will trigger value error
    }

    with pytest.raises(ValueError, match="No datasets configured for SFT."):
        run_sft(cfg, dummy_data=False)
