import pytest
from unittest.mock import MagicMock
from src.task_sft import run_task_sft

def test_run_task_sft_dummy_data_full_ft(mocker):
    mock_tokenizer = mocker.patch("src.task_sft.AutoTokenizer.from_pretrained")
    mock_model = mocker.patch("src.task_sft.AutoModelForCausalLM.from_pretrained")
    mock_trainer = mocker.patch("src.task_sft.SFTTrainer")
    mocker.patch("src.task_sft.SFTConfig")
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
        "use_lora": False,
        "output_dir": "./dummy_task_sft_output"
    }

    cfg['push_to_hub'] = 'dummy/task_sft'
    run_task_sft(cfg, dummy_data=True)

    mock_tokenizer.assert_called_once_with("dummy_model")
    mock_model.assert_called_once_with("dummy_model")
    mock_trainer.assert_called_once()
    mock_trainer.return_value.train.assert_called_once()
    assert mock_tok_inst.chat_template is not None
    mock_mod_inst.resize_token_embeddings.assert_called_once()

def test_run_task_sft_dummy_data_lora(mocker):
    mock_tokenizer = mocker.patch("src.task_sft.AutoTokenizer.from_pretrained")
    mock_model = mocker.patch("src.task_sft.AutoModelForCausalLM.from_pretrained")
    mock_trainer = mocker.patch("src.task_sft.SFTTrainer")
    mocker.patch("src.task_sft.SFTConfig")
    mocker.patch('transformers.PreTrainedModel.push_to_hub', return_value=None)
    mocker.patch('transformers.PreTrainedTokenizerBase.push_to_hub', return_value=None)
    mocker.patch('trl.SFTTrainer.push_to_hub', return_value=None)
    mock_get_peft = mocker.patch("peft.get_peft_model")
    mock_lora_config = mocker.patch("peft.LoraConfig")

    mock_tok_inst = MagicMock()
    mock_tok_inst.pad_token = None
    mock_tok_inst.eos_token = "<|endoftext|>"
    mock_tok_inst.chat_template = None
    mock_tokenizer.return_value = mock_tok_inst

    mock_mod_inst = MagicMock()
    mock_model.return_value = mock_mod_inst
    mock_get_peft.return_value = mock_mod_inst

    cfg = {
        "model_name": "dummy_model",
        "use_lora": True,
        "lora": {
            "r": 16,
            "target_modules": ["q_proj", "v_proj"]
        }
    }

    cfg['push_to_hub'] = 'dummy/task_sft'
    run_task_sft(cfg, dummy_data=True)

    mock_lora_config.assert_called_once()
    mock_get_peft.assert_called_once_with(mock_mod_inst, mock_lora_config.return_value)
    mock_trainer.assert_called_once()
    mock_trainer.return_value.train.assert_called_once()

def test_run_task_sft_real_data(mocker):
    mocker.patch("src.task_sft.AutoTokenizer.from_pretrained")
    mocker.patch("src.task_sft.AutoModelForCausalLM.from_pretrained")
    mock_trainer = mocker.patch("src.task_sft.SFTTrainer")
    mocker.patch("src.task_sft.SFTConfig")
    mocker.patch('transformers.PreTrainedModel.push_to_hub', return_value=None)
    mocker.patch('transformers.PreTrainedTokenizerBase.push_to_hub', return_value=None)
    mocker.patch('trl.SFTTrainer.push_to_hub', return_value=None)
    mock_load_dataset = mocker.patch("src.task_sft.load_dataset")
    mock_interleave = mocker.patch("src.task_sft.interleave_datasets")

    mock_ds = MagicMock()
    mock_interleave.return_value = mock_ds
    mock_load_dataset.return_value = mock_ds

    cfg = {
        "model_name": "dummy_model",
        "use_lora": False,
        "dataset_paths": {
            "task1": "path/task1"
        }
    }

    run_task_sft(cfg, dummy_data=False)

    mock_load_dataset.assert_called_once_with("path/task1", split="train", streaming=True)
    mock_interleave.assert_called_once()
    mock_trainer.return_value.train.assert_called_once()

def test_run_task_sft_no_datasets(mocker):
    mocker.patch("src.task_sft.AutoTokenizer.from_pretrained")

    cfg = {
        "model_name": "dummy_model",
        "dataset_paths": {}
    }

    with pytest.raises(ValueError, match="No datasets configured for Task SFT."):
        run_task_sft(cfg, dummy_data=False)
