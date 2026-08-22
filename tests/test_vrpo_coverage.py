import pytest
import torch
from src.alignment.vrpo import VRPOConfig, VRPOTrainer
from transformers import PreTrainedTokenizerFast
from datasets import Dataset

def create_mock_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    tokenizer_obj = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_obj, pad_token="[PAD]")
    return tokenizer

def test_prepare_diffusion_trajectory_no_input_ids(mocker):
    config = VRPOConfig(diffusion_steps=10, use_cpu=True)
    tokenizer = create_mock_tokenizer()
    dataset = Dataset.from_dict({"prompt": ["test"]})
    trainer = VRPOTrainer.__new__(VRPOTrainer)
    trainer.args = config
    with pytest.raises(ValueError, match="input_ids not provided for diffusion trajectory"):
        trainer._prepare_diffusion_trajectory({})

def test_compute_loss_no_alignment(mocker):
    config = VRPOConfig(trajectory_alignment=False, use_cpu=True)
    trainer = VRPOTrainer.__new__(VRPOTrainer)
    trainer.args = config
    mocker.patch('trl.GRPOTrainer.compute_loss', return_value=1.0)
    loss = trainer.compute_loss(mocker.MagicMock(), {"input_ids": torch.tensor([[1, 2]])})
    assert loss == 1.0

def test_compute_loss_with_alignment(mocker):
    config = VRPOConfig(trajectory_alignment=True, diffusion_steps=5, use_cpu=True)
    trainer = VRPOTrainer.__new__(VRPOTrainer)
    trainer.args = config
    mocker.patch('trl.GRPOTrainer.compute_loss', return_value=2.0)
    inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
    loss = trainer.compute_loss(mocker.MagicMock(), inputs)
    assert loss == 2.0
    assert "trajectory_states" in inputs
    assert inputs["trajectory_states"].shape == (1, 5, 3)
