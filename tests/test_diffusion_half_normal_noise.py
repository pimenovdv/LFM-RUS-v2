import torch
import pytest
from unittest.mock import MagicMock

from src.models.diffusion.configuration_diffusion import DiffusionConfig
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, MDLMContinuousBatchingManager

@pytest.fixture
def dummy_model(mocker):
    config = DiffusionConfig(
        vocab_size=100,
        model_type="gpt2",
        base_config_dict={"vocab_size": 100, "model_type": "gpt2"},
        mask_token_id=0,
        diffusion_steps=10
    )
    mocker.patch("src.models.diffusion.modeling_diffusion.AutoConfig.for_model")

    mock_model = MagicMock()
    mock_model.get_input_embeddings.return_value.weight = torch.nn.Parameter(torch.randn(100, 100))
    mocker.patch("src.models.diffusion.modeling_diffusion.AutoModel.from_config", return_value=mock_model)


    model = DiffusionModelForConditionalGeneration(config)

    class Output:
        def __init__(self, logits):
            self.logits = logits

    def mock_forward(input_ids, **kwargs):
        batch_size, seq_len = input_ids.shape
        # Create predictable logits
        logits = torch.zeros(batch_size, seq_len, 100)
        return Output(logits)

    model.forward = mock_forward
    return model

def test_half_normal_noise_injection_generate(dummy_model, mocker):
    input_ids = torch.tensor([[1, 2, 3]])

    mock_sample = mocker.patch("torch.distributions.half_normal.HalfNormal.sample", side_effect=lambda shape: torch.ones(shape))

    dummy_model.generate(
        input_ids,
        max_new_tokens=1,
        steps=2,
        half_normal_noise_scale=2.0,
        half_normal_noise_schedule="constant",
        min_half_normal_noise_scale=0.1
    )

    assert mock_sample.call_count > 0

def test_half_normal_noise_injection_continuous_batching(dummy_model, mocker):
    manager = MDLMContinuousBatchingManager(dummy_model)

    req_config = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "max_new_tokens": 1,
        "total_steps": 2,
        "half_normal_noise_scale": 2.0,
        "half_normal_noise_schedule": "constant",
        "min_half_normal_noise_scale": 0.1
    }

    manager.add_request(**req_config)

    mock_sample = mocker.patch("torch.distributions.half_normal.HalfNormal.sample", side_effect=lambda shape: torch.ones(shape))

    manager.step()

    assert mock_sample.call_count > 0

def test_half_normal_noise_schedules(dummy_model, mocker):
    input_ids = torch.tensor([[1, 2, 3]])

    for schedule in ["linear", "cosine", "exponential", "cyclic"]:
        mock_sample = mocker.patch("torch.distributions.half_normal.HalfNormal.sample", side_effect=lambda shape: torch.ones(shape))
        dummy_model.generate(
            input_ids,
            max_new_tokens=1,
            steps=2,
            half_normal_noise_scale=2.0,
            half_normal_noise_schedule=schedule,
            min_half_normal_noise_scale=0.1
        )
        assert mock_sample.call_count > 0
