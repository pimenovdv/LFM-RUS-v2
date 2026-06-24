import pytest
from unittest.mock import MagicMock
from src.cpt import run_cpt

def test_run_cpt_dummy_data(mocker):
    # Mocking HF components
    mock_tokenizer = mocker.patch("src.cpt.AutoTokenizer.from_pretrained")
    mock_model = mocker.patch("src.cpt.AutoModelForCausalLM.from_pretrained")
    mock_trainer = mocker.patch("src.cpt.Trainer")
    mocker.patch("src.cpt.TrainingArguments")
    mocker.patch("src.cpt.DataCollatorForLanguageModeling")
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
    mocker.patch("src.cpt.AutoTokenizer.from_pretrained")
    mocker.patch("src.cpt.AutoModelForCausalLM.from_pretrained")
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
    mocker.patch("src.cpt.AutoTokenizer.from_pretrained")

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
    mocker.patch("src.cpt.TrainingArguments")
    mocker.patch("src.cpt.DataCollatorForLanguageModeling")

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
    assert mock_param_1.requires_grad
    assert mock_param_2.requires_grad

def test_run_cpt_wsd(mocker):
    # Mocking HF components
    mock_tokenizer = mocker.patch("src.cpt.AutoTokenizer.from_pretrained")
    mock_model = mocker.patch("src.cpt.AutoModelForCausalLM.from_pretrained")
    mock_trainer = mocker.patch("src.cpt.Trainer")
    mock_ta = mocker.patch("src.cpt.TrainingArguments")
    mocker.patch("src.cpt.DataCollatorForLanguageModeling")

    mock_tokenizer.return_value.pad_token = None
    mock_tokenizer.return_value.eos_token = "<|endoftext|>"
    mock_tokenizer.return_value.return_value = {"input_ids": [[1, 2, 3]] * 400}

    cfg = {
        "model_name": "dummy_model",
        "wsd": {
            "enabled": True,
            "warmup_steps": 100,
            "decay_steps": 50
        }
    }

    run_cpt(cfg, dummy_data=True)

    # Check TrainingArguments call to contain wsd config
    _, kwargs = mock_ta.call_args
    assert kwargs.get("lr_scheduler_type") == "warmup_stable_decay"
    assert kwargs.get("warmup_steps") == 100
    assert kwargs.get("lr_scheduler_kwargs") == {"num_decay_steps": 50}

def test_run_cpt_block_diffusion(mocker):
    # Mocking HF components
    mock_tokenizer = mocker.patch("src.cpt.AutoTokenizer.from_pretrained")
    mock_model = mocker.patch("src.cpt.AutoModelForCausalLM.from_pretrained")
    mock_trainer = mocker.patch("src.cpt.Trainer")
    mocker.patch("src.cpt.TrainingArguments")
    mock_bd_collator = mocker.patch("src.cpt.BlockDiffusionDataCollator")

    mock_tokenizer.return_value.pad_token = None
    mock_tokenizer.return_value.eos_token = "<|endoftext|>"
    mock_tokenizer.return_value.return_value = {"input_ids": [[1, 2, 3]] * 400}

    cfg = {
        "model_name": "dummy_model",
        "block_diffusion": {
            "enabled": True,
            "block_size": 32
        }
    }

    run_cpt(cfg, dummy_data=True)

    mock_bd_collator.assert_called_once()
    _, kwargs = mock_bd_collator.call_args
    assert kwargs.get("block_size") == 32

def test_run_cpt_top_k_merge(mocker):
    # Mocking HF components
    mock_tokenizer = mocker.patch("src.cpt.AutoTokenizer.from_pretrained")
    mock_model = mocker.patch("src.cpt.AutoModelForCausalLM.from_pretrained")
    mock_trainer = mocker.patch("src.cpt.Trainer")
    mocker.patch("src.cpt.TrainingArguments")
    mocker.patch("src.cpt.DataCollatorForLanguageModeling")

    mock_tokenizer.return_value.pad_token = None
    mock_tokenizer.return_value.eos_token = "<|endoftext|>"
    mock_tokenizer.return_value.return_value = {"input_ids": [[1, 2, 3]] * 400}

    mock_merge = mocker.patch("src.cpt.merge_top_k_checkpoints")

    cfg = {
        "model_name": "dummy_model",
        "output_dir": "./dummy_out",
        "merge_top_k": 3
    }

    run_cpt(cfg, dummy_data=True)

    mock_merge.assert_called_once_with("./dummy_out", 3)


def test_block_diffusion_collator(mocker):
    from src.cpt import BlockDiffusionDataCollator
    import torch
    from unittest.mock import MagicMock

    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token_id = 0

    # Patch the superclass call
    mocker.patch("transformers.DataCollatorForLanguageModeling.__call__",
                 return_value={"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])})

    collator = BlockDiffusionDataCollator(block_size=4, tokenizer=mock_tokenizer, mlm=False)

    examples = [{"input_ids": [1, 2, 3, 4, 5, 6, 7, 8]}]
    batch = collator(examples)

    assert "attention_mask" in batch
    mask = batch["attention_mask"]

    # Check shape: batch=1, heads=1, seq=8, seq=8
    assert mask.shape == (1, 1, 8, 8)

    # Block 1 (idx 0-3) should attend to itself
    assert torch.all(mask[0, 0, 0:4, 0:4] == 1.0)
    # Block 1 shouldn't attend to Block 2 (idx 4-7)
    assert torch.all(mask[0, 0, 0:4, 4:8] == 0.0)
    # Block 2 (idx 4-7) should attend to Block 1 and Block 2
    assert torch.all(mask[0, 0, 4:8, 0:4] == 1.0)
    assert torch.all(mask[0, 0, 4:8, 4:8] == 1.0)

def test_merge_top_k_checkpoints(mocker, tmp_path):
    from src.cpt import merge_top_k_checkpoints
    import torch
    import safetensors.torch
    import os

    # Create mock checkpoints
    ckpt1 = tmp_path / "checkpoint-10"
    ckpt1.mkdir()
    ckpt2 = tmp_path / "checkpoint-20"
    ckpt2.mkdir()
    ckpt3 = tmp_path / "checkpoint-30"
    ckpt3.mkdir()

    state1 = {"weight": torch.tensor([1.0, 2.0])}
    state2 = {"weight": torch.tensor([3.0, 4.0])}

    safetensors.torch.save_file(state1, str(ckpt1 / "model.safetensors"))
    torch.save(state2, str(ckpt2 / "pytorch_model.bin"))

    # Run merge for top 2 (ckpt2 and ckpt3 - but ckpt3 has no weights, so it will skip it, but logic says wait
    # let's put weights in ckpt3
    state3 = {"weight": torch.tensor([5.0, 6.0])}
    safetensors.torch.save_file(state3, str(ckpt3 / "model.safetensors"))

    merge_top_k_checkpoints(str(tmp_path), 2)

    # Check if merged_model is created
    merged_dir = tmp_path / "merged_model"
    assert merged_dir.exists()

    merged_file = merged_dir / "model.safetensors"
    assert merged_file.exists()

    merged_state = safetensors.torch.load_file(str(merged_file))

    # Merged of state2 ([3.0, 4.0]) and state3 ([5.0, 6.0]) -> [4.0, 5.0]
    # Wait, the code divides by num_ckpts = 2
    assert torch.allclose(merged_state["weight"], torch.tensor([4.0, 5.0]))

def test_merge_top_k_checkpoints_no_checkpoints(tmp_path, caplog):
    from src.cpt import merge_top_k_checkpoints
    merge_top_k_checkpoints(str(tmp_path), 2)
    assert "No checkpoints found to merge." in caplog.text
