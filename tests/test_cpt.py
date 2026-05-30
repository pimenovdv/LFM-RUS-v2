import pytest
from unittest.mock import patch, MagicMock
from src.cpt import run_cpt

def test_run_cpt_dummy_data(mocker):
    # Mocking HF components
    mock_tokenizer = mocker.patch("src.cpt.AutoTokenizer.from_pretrained")
    mock_model = mocker.patch("src.cpt.AutoModelForCausalLM.from_pretrained")
    mock_trainer = mocker.patch("src.cpt.Trainer")
    mock_training_args = mocker.patch("src.cpt.TrainingArguments")
    mock_data_collator = mocker.patch("src.cpt.DataCollatorForLanguageModeling")
    mocker.patch('transformers.PreTrainedModel.push_to_hub', return_value=None)
    mocker.patch('transformers.PreTrainedTokenizerBase.push_to_hub', return_value=None)
    mocker.patch('src.cpt.Trainer.push_to_hub', return_value=None)

    mock_tokenizer.return_value.pad_token = None
    mock_tokenizer.return_value.eos_token = "<|endoftext|>"
    mock_tokenizer.return_value.return_value = {"input_ids": [[1, 2, 3]] * 400}
    mock_tokenizer.return_value.return_value = {"input_ids": [[1, 2, 3]] * 400}

    cfg = {
        "model_name": "dummy_model",
        "max_seq_length": 16,
        "output_dir": "./dummy_output",
        "learning_rate": 0.001,
        "epochs": 1,
        "per_device_train_batch_size": 2,
        "save_steps": 10
    }

    # Run with dummy data
    cfg['push_to_hub'] = 'dummy/cpt'
    run_cpt(cfg, dummy_data=True)

    # Check assertions
    mock_tokenizer.assert_called_once_with("dummy_model")
    mock_model.assert_called_once_with("dummy_model")
    mock_trainer.assert_called_once()
    mock_trainer.return_value.train.assert_called_once()


def test_run_cpt_real_data(mocker):
    # Mocking HF components
    mock_tokenizer = mocker.patch("src.cpt.AutoTokenizer.from_pretrained")
    mock_model = mocker.patch("src.cpt.AutoModelForCausalLM.from_pretrained")
    mock_trainer = mocker.patch("src.cpt.Trainer")
    mock_load_dataset = mocker.patch("src.cpt.load_dataset")
    mock_interleave = mocker.patch("src.cpt.interleave_datasets")

    # Mock the return values for dataset functions
    mock_ds = MagicMock()
    # Mock map so it just returns a mock
    mock_ds.map.return_value = mock_ds
    mock_interleave.return_value = mock_ds
    mock_load_dataset.return_value = mock_ds

    cfg = {
        "model_name": "dummy_model",
        "dataset_paths": {
            "ru": "path/ru",
            "en": "path/en"
        },
        "dataset_ratios": {
            "ru": 0.6,
            "en": 0.4
        }
    }

    run_cpt(cfg, dummy_data=False)

    # Check that datasets were loaded and interleaved
    assert mock_load_dataset.call_count == 2
    mock_interleave.assert_called_once()
    mock_trainer.return_value.train.assert_called_once()


def test_run_cpt_no_datasets(mocker):
    # Mocking HF components
    mock_tokenizer = mocker.patch("src.cpt.AutoTokenizer.from_pretrained")

    cfg = {
        "model_name": "dummy_model",
        "dataset_paths": {} # No paths should cause failure
    }

    with pytest.raises(ValueError, match="No datasets loaded successfully."):
        run_cpt(cfg, dummy_data=False)

def test_run_cpt_embedding_warmup(mocker):
    # Mocking HF components
    mock_tokenizer = mocker.patch("src.cpt.AutoTokenizer.from_pretrained")
    mock_model = mocker.patch("src.cpt.AutoModelForCausalLM.from_pretrained")
    mock_trainer = mocker.patch("src.cpt.Trainer")
    mock_training_args = mocker.patch("src.cpt.TrainingArguments")
    mock_data_collator = mocker.patch("src.cpt.DataCollatorForLanguageModeling")

    mock_tokenizer.return_value.pad_token = None
    mock_tokenizer.return_value.eos_token = "<|endoftext|>"
    mock_tokenizer.return_value.return_value = {"input_ids": [[1, 2, 3]] * 400}

    # Setup mock parameters for the model
    mock_param_1 = mocker.MagicMock()
    mock_param_1.requires_grad = True
    mock_param_2 = mocker.MagicMock()
    mock_param_2.requires_grad = True

    mock_model_instance = mock_model.return_value
    mock_model_instance.parameters.return_value = [mock_param_1, mock_param_2]

    mock_in_emb = mocker.MagicMock()
    mock_in_emb.parameters.return_value = [mock_param_1]
    mock_model_instance.get_input_embeddings.return_value = mock_in_emb

    mock_out_emb = mocker.MagicMock()
    mock_out_emb.parameters.return_value = [mock_param_2]
    mock_model_instance.get_output_embeddings.return_value = mock_out_emb

    cfg = {
        "model_name": "dummy_model",
        "max_seq_length": 16,
        "output_dir": "./dummy_output",
        "learning_rate": 0.001,
        "epochs": 1,
        "per_device_train_batch_size": 2,
        "save_steps": 10,
        "embedding_warmup": {
            "enabled": True,
            "epochs": 1,
            "learning_rate": 0.005
        }
    }

    run_cpt(cfg, dummy_data=True)

    # Trainer should be instantiated twice: once for warmup, once for main CPT
    assert mock_trainer.call_count == 2
    # Train should be called twice
    assert mock_trainer.return_value.train.call_count == 2

    # Check that input and output embeddings were processed
    mock_model_instance.get_input_embeddings.assert_called_once()
    mock_model_instance.get_output_embeddings.assert_called_once()

    # Finally, all parameters should have requires_grad = True before main CPT phase
    assert mock_param_1.requires_grad == True
    assert mock_param_2.requires_grad == True
