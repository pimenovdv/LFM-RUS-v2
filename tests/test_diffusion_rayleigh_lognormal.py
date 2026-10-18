import pytest
import torch
import math
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, DiffusionConfig
from tests.test_diffusion_continuous_batching_sampling import get_mock_model
from src.models.diffusion.modeling_diffusion import MDLMRequest, MDLMContinuousBatchingManager
from tests.test_diffusion_smooth_k_positional_temp import dummy_model

def test_rayleigh_noise_generation(dummy_model, mocker):
    mock_rand = mocker.patch("torch.rand")
    mock_rand.side_effect = lambda shape, **kwargs: torch.full(shape, 0.5, **kwargs)

    input_ids = torch.tensor([[2, 0, 0, 3]])

    dummy_model.generate(
        input_ids=input_ids,
        max_new_tokens=2,
        rayleigh_noise_scale=1.0,
        rayleigh_noise_schedule="linear",
        min_rayleigh_noise_scale=0.1
    )
    assert mock_rand.called

def test_log_normal_noise_generation(dummy_model, mocker):
    mock_log_normal_dist = mocker.patch("torch.distributions.log_normal.LogNormal")
    mock_log_normal_instance = mocker.MagicMock()
    mock_log_normal_instance.sample.side_effect = lambda shape: torch.ones(shape)
    mock_log_normal_dist.return_value = mock_log_normal_instance

    input_ids = torch.tensor([[2, 0, 0, 3]])

    dummy_model.generate(
        input_ids=input_ids,
        max_new_tokens=2,
        log_normal_noise_scale=1.0,
        log_normal_noise_schedule="cosine",
        log_normal_noise_mean=0.0,
        min_log_normal_noise_scale=0.1
    )
    assert mock_log_normal_dist.called

def test_rayleigh_log_normal_continuous_batching(mocker):
    mock_model = get_mock_model(mocker)
    manager = MDLMContinuousBatchingManager(mock_model)

    req1 = {
        "request_id": "req_1",
        "input_ids": torch.tensor([[2]]),
        "total_steps": 2,
        "max_new_tokens": 2,
        "rayleigh_noise_scale": 1.0,
        "rayleigh_noise_schedule": "exponential",
        "min_rayleigh_noise_scale": 0.1
    }

    req2 = {
        "request_id": "req_2",
        "input_ids": torch.tensor([[2]]),
        "total_steps": 2,
        "max_new_tokens": 2,
        "log_normal_noise_scale": 1.0,
        "log_normal_noise_schedule": "cyclic",
        "log_normal_noise_mean": 0.5,
        "min_log_normal_noise_scale": 0.1
    }

    manager.add_request(**req1)
    manager.add_request(**req2)

    mock_rand = mocker.patch("torch.rand")
    mock_rand.side_effect = lambda shape, **kwargs: torch.full(shape, 0.5, **kwargs)

    mock_log_normal_dist = mocker.patch("torch.distributions.log_normal.LogNormal")
    mock_log_normal_instance = mocker.MagicMock()
    mock_log_normal_instance.sample.side_effect = lambda shape: torch.ones(shape)
    mock_log_normal_dist.return_value = mock_log_normal_instance

    manager.step()

    assert mock_rand.called
    assert mock_log_normal_dist.called
