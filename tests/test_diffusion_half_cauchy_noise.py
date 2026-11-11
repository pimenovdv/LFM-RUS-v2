import pytest
import torch
from src.models.diffusion.configuration_diffusion import DiffusionConfig
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, MDLMContinuousBatchingManager

class MockOutput:
    def __init__(self, logits):
        self.logits = logits

@pytest.fixture
def mock_diffusion_model(mocker):
    config = DiffusionConfig(
        vocab_size=100,
        model_type="gpt2",
        block_size=16,
        mask_token_id=0,
        half_cauchy_noise_scale=2.0,
        base_config_dict={"vocab_size": 100, "model_type": "gpt2"}
    )
    model = DiffusionModelForConditionalGeneration(config)

    def mock_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is not None:
            return MockOutput(torch.randn(input_ids.shape[0], input_ids.shape[1], config.vocab_size))
        return MockOutput(torch.randn(1, 16, config.vocab_size))

    mocker.patch.object(model, "forward", side_effect=mock_forward)
    return model, config

def test_half_cauchy_noise_generation(mocker, mock_diffusion_model):
    model, _ = mock_diffusion_model
    input_ids = torch.randint(0, 100, (1, 10))

    mock_sample = mocker.patch("torch.distributions.HalfCauchy.sample", side_effect=lambda shape: torch.ones(shape))

    model.generate(
        input_ids,
        half_cauchy_noise_scale=2.0,
        half_cauchy_noise_schedule="constant"
    )

    assert mock_sample.called

def test_half_cauchy_noise_schedules(mocker, mock_diffusion_model):
    model, _ = mock_diffusion_model
    input_ids = torch.randint(0, 100, (1, 10))

    for schedule in ["linear", "cosine", "exponential", "cyclic"]:
        mock_sample = mocker.patch("torch.distributions.HalfCauchy.sample", side_effect=lambda shape: torch.ones(shape))

        model.generate(
            input_ids,
            half_cauchy_noise_scale=2.0,
            half_cauchy_noise_schedule=schedule
        )

        assert mock_sample.called

def test_half_cauchy_noise_continuous_batching(mocker, mock_diffusion_model):
    model, config = mock_diffusion_model
    manager = MDLMContinuousBatchingManager(model, max_batch_size=8)

    mock_sample = mocker.patch("torch.distributions.HalfCauchy.sample", side_effect=lambda shape: torch.ones(shape))

    manager.add_request(
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=5,
        total_steps=5,
        half_cauchy_noise_scale=2.0,
        half_cauchy_noise_schedule="constant"
    )

    manager.step()
    assert mock_sample.called

def test_half_cauchy_noise_continuous_batching_schedules(mocker, mock_diffusion_model):
    model, config = mock_diffusion_model

    for schedule in ["linear", "cosine", "exponential", "cyclic"]:
        manager = MDLMContinuousBatchingManager(model, max_batch_size=8)
        mock_sample = mocker.patch("torch.distributions.HalfCauchy.sample", side_effect=lambda shape: torch.ones(shape))

        manager.add_request(
            input_ids=torch.tensor([[1, 2, 3]]),
            max_new_tokens=5,
            total_steps=5,
            half_cauchy_noise_scale=2.0,
            half_cauchy_noise_schedule=schedule
        )

        manager.step()
        assert mock_sample.called
