import pytest
import torch
import torch.nn.functional as F
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, MDLMContinuousBatchingManager, MDLMRequest
from src.models.diffusion.configuration_diffusion import DiffusionConfig
import math


@pytest.fixture
def model_and_config():
    config = DiffusionConfig(
        vocab_size=100,
        base_config_dict={"model_type": "gpt2", "vocab_size": 100},
        mask_token_id=0,
        pad_token_id=1,
    )

    model = DiffusionModelForConditionalGeneration(config)

    # Mocking self.forward logic inside generate to just return a dummy tensor
    # where all logits are random to test the sampling methods.
    def mock_forward(input_ids, **kwargs):
        class Output:
            logits = torch.randn((input_ids.shape[0], input_ids.shape[1], config.vocab_size))
        return Output()

    model.forward = mock_forward
    return model, config


def test_laplacian_and_cauchy_noise_in_generate(mocker, model_and_config):
    model, config = model_and_config
    input_ids = torch.full((1, 5), config.mask_token_id)

    # We must patch so that the mock sample shape matches the logits shape exactly during generation
    # The generation loop adds tokens so length is changing.
    mock_laplace_sample = mocker.patch("torch.distributions.laplace.Laplace.sample")
    def mock_laplace_sampler(shape):
        return torch.ones(shape)
    mock_laplace_sample.side_effect = mock_laplace_sampler

    mock_cauchy_sample = mocker.patch("torch.distributions.cauchy.Cauchy.sample")
    def mock_cauchy_sampler(shape):
        return torch.ones(shape)
    mock_cauchy_sample.side_effect = mock_cauchy_sampler

    # Mocking self.forward logic inside generate to handle correct input_ids shape during generation
    def mock_forward(input_ids, **kwargs):
        class Output:
            logits = torch.randn((input_ids.shape[0], input_ids.shape[1], config.vocab_size))
        return Output()

    model.forward = mock_forward

    # Test Laplacian noise
    model.generate(
        input_ids,
        laplacian_noise_scale=0.5,
        laplacian_noise_schedule="constant",
        steps_per_block=1,
        max_new_tokens=5
    )
    assert mock_laplace_sample.called
    assert not mock_cauchy_sample.called

    mock_laplace_sample.reset_mock()
    mock_cauchy_sample.reset_mock()

    # Test Cauchy noise
    model.generate(
        input_ids,
        cauchy_noise_scale=0.5,
        cauchy_noise_schedule="constant",
        steps_per_block=1,
        max_new_tokens=5
    )
    assert not mock_laplace_sample.called
    assert mock_cauchy_sample.called


def test_laplacian_and_cauchy_noise_in_continuous_batching(mocker, model_and_config):
    model, config = model_and_config

    def mock_forward(input_ids, **kwargs):
        class Output:
            logits = torch.randn((input_ids.shape[0], input_ids.shape[1], config.vocab_size))
        return Output()

    model.forward = mock_forward

    manager = MDLMContinuousBatchingManager(model)

    # Add requests with laplace and cauchy noise
    manager.add_request(
        input_ids=torch.full((1, 5), config.mask_token_id),
        max_new_tokens=5,
        total_steps=1,
        laplacian_noise_scale=0.5,
    )
    manager.add_request(
        input_ids=torch.full((1, 5), config.mask_token_id),
        max_new_tokens=5,
        total_steps=1,
        cauchy_noise_scale=0.5,
    )

    mock_laplace_sample = mocker.patch("torch.distributions.laplace.Laplace.sample")
    # manager steps pad and combine inputs, mock needs a matching return shape for tensor ops
    def mock_laplace_sampler(shape):
        return torch.ones(shape)
    mock_laplace_sample.side_effect = mock_laplace_sampler

    mock_cauchy_sample = mocker.patch("torch.distributions.cauchy.Cauchy.sample")
    def mock_cauchy_sampler(shape):
        return torch.ones(shape)
    mock_cauchy_sample.side_effect = mock_cauchy_sampler

    manager.step()

    # Because req1 has laplacian_noise_scale > 0 and req2 has cauchy_noise_scale > 0,
    # both distribution samplers should be called
    assert mock_laplace_sample.called
    assert mock_cauchy_sample.called
