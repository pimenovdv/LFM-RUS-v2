import pytest
import torch
import math
from transformers import AutoConfig
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, MDLMContinuousBatchingManager, MDLMRequest
from src.models.diffusion.configuration_diffusion import DiffusionConfig

@pytest.fixture
def mock_model_with_chi_square_noise(mocker):
    config = DiffusionConfig(
        vocab_size=100,
        base_config_dict={"model_type": "gpt2", "vocab_size": 100},
        chi_square_noise_scale=2.0,
        chi_square_noise_schedule="constant",
        chi_square_df=2.0,
        mask_token_id=0
    )
    model = DiffusionModelForConditionalGeneration(config)

    class Output:
        def __init__(self, logits):
            self.logits = logits

    def mock_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        return Output(torch.randn(input_ids.shape[0], input_ids.shape[1], config.vocab_size))

    mocker.patch.object(model, "forward", side_effect=mock_forward)
    return model, config

def test_chi_square_noise_generate(mocker, mock_model_with_chi_square_noise):
    model, config = mock_model_with_chi_square_noise
    input_ids = torch.tensor([[1, 2, 3]])

    chi_square_mock = mocker.patch("torch.distributions.Chi2.sample")
    chi_square_mock.side_effect = lambda shape: torch.ones(shape)

    model.generate(
        input_ids,
        max_new_tokens=2,
        chi_square_noise_scale=2.0,
        chi_square_noise_schedule="constant",
        chi_square_df=2.0,
        gumbel_temperature=0.0
    )

    assert chi_square_mock.called

def test_chi_square_noise_schedules(mocker, mock_model_with_chi_square_noise):
    model, config = mock_model_with_chi_square_noise
    input_ids = torch.tensor([[1, 2, 3]])

    chi_square_mock = mocker.patch("torch.distributions.Chi2.sample")
    chi_square_mock.side_effect = lambda shape: torch.ones(shape)

    for schedule in ["linear", "cosine", "exponential", "cyclic"]:
        model.generate(
            input_ids,
            max_new_tokens=2,
            chi_square_noise_scale=2.0,
            chi_square_noise_schedule=schedule,
            chi_square_df=2.0,
            gumbel_temperature=0.0
        )
        assert chi_square_mock.called
        chi_square_mock.reset_mock()

def test_chi_square_noise_continuous_batching(mocker, mock_model_with_chi_square_noise):
    model, config = mock_model_with_chi_square_noise
    manager = MDLMContinuousBatchingManager(model)

    chi_square_mock = mocker.patch("torch.distributions.Chi2.sample")
    chi_square_mock.side_effect = lambda shape: torch.ones(shape)

    req_config = {
        "request_id": "req1",
        "input_ids": torch.tensor([[1, 2]]),
        "max_new_tokens": 2,
        "total_steps": 10,
        "chi_square_noise_scale": 2.0,
        "chi_square_noise_schedule": "constant",
        "chi_square_df": 2.0
    }

    manager.add_request(**req_config)
    manager.step()

    assert chi_square_mock.called

def test_chi_square_noise_continuous_batching_schedules(mocker, mock_model_with_chi_square_noise):
    model, config = mock_model_with_chi_square_noise

    chi_square_mock = mocker.patch("torch.distributions.Chi2.sample")
    chi_square_mock.side_effect = lambda shape: torch.ones(shape)

    for i, schedule in enumerate(["linear", "cosine", "exponential", "cyclic"]):
        manager = MDLMContinuousBatchingManager(model)
        req_config = {
            "request_id": f"req_{i}",
                "input_ids": torch.tensor([[1, 2]]),
            "max_new_tokens": 2,
            "total_steps": 10,
            "chi_square_noise_scale": 2.0,
            "chi_square_noise_schedule": schedule,
            "chi_square_df": 2.0
        }
        manager.add_request(**req_config)
        manager.step()
        assert chi_square_mock.called
        chi_square_mock.reset_mock()
