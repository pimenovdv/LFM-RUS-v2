import pytest
from unittest.mock import MagicMock
from src.sft import run_sft

def test_run_sft_dummy_data(mocker):
    mock_tokenizer = mocker.patch("src.sft.AutoTokenizer.from_pretrained")
    mock_model = mocker.patch("src.sft.AutoModelForCausalLM.from_pretrained")
    mock_trainer = mocker.patch("src.sft.SFTTrainer")
    mocker.patch("src.sft.SFTConfig")
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
    mocker.patch("src.sft.AutoTokenizer.from_pretrained")
    mocker.patch("src.sft.AutoModelForCausalLM.from_pretrained")
    mock_trainer = mocker.patch("src.sft.SFTTrainer")
    mocker.patch("src.sft.SFTConfig")
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
    mocker.patch("src.sft.AutoTokenizer.from_pretrained")

    cfg = {
        "model_name": "dummy_model",
        "dataset_paths": {}
        # missing dataset_path will trigger value error
    }

    with pytest.raises(ValueError, match="No datasets configured for SFT."):
        run_sft(cfg, dummy_data=False)


import torch
from src.sft import SFTDiffusionDataCollator
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration


def test_sft_diffusion_collator(mocker):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|im_start|>', '<|im_end|>']})

    collator = SFTDiffusionDataCollator(tokenizer, mask_token_id=0, max_timesteps=10, block_size=2)

    input_text = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nWorld<|im_end|>"
    input_ids = tokenizer.encode(input_text)

    batch = [{"input_ids": input_ids}]

    output = collator(batch)

    assert "input_ids" in output
    assert "labels" in output
    assert "timesteps" in output
    assert "attention_mask" in output

    seq_len = len(input_ids)
    assert output["input_ids"].shape == (1, seq_len)
    assert output["labels"].shape == (1, seq_len)
    assert output["timesteps"].shape == (1,)
    assert output["attention_mask"].shape == (1, 1, seq_len, seq_len)


def test_run_sft_diffusion(mocker):
    mock_tokenizer = mocker.patch("src.sft.AutoTokenizer.from_pretrained")
    mock_model = mocker.patch("src.sft.AutoModelForCausalLM.from_pretrained")
    mock_trainer = mocker.patch("src.sft.SFTTrainer")
    mocker.patch("src.sft.SFTConfig")

    mock_tok_inst = MagicMock()
    mock_tok_inst.pad_token_id = 0
    mock_tokenizer.return_value = mock_tok_inst

    mock_mod_inst = MagicMock(spec=DiffusionModelForConditionalGeneration)
    mock_mod_inst.config = MagicMock()
    mock_mod_inst.config.mask_token_id = 99
    mock_mod_inst.config.max_timesteps = 100
    mock_model.return_value = mock_mod_inst

    cfg = {
        "model_name": "dummy_model",
        "packing": True,
        "block_diffusion": {"enabled": True, "block_size": 16}
    }

    run_sft(cfg, dummy_data=True)

    # Assert trainer was called with data_collator
    args, kwargs = mock_trainer.call_args
    assert "data_collator" in kwargs
    assert isinstance(kwargs["data_collator"], SFTDiffusionDataCollator)
