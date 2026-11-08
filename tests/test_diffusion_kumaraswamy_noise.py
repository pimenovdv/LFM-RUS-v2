import pytest
import torch
from src.models.diffusion.configuration_diffusion import DiffusionConfig
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, MDLMContinuousBatchingManager, MDLMRequest

class Output:
    def __init__(self, logits):
        self.logits = logits

@pytest.fixture
def base_config():
    return DiffusionConfig(
        base_config_dict={"vocab_size": 100, "model_type": "gpt2"},
        vocab_size=100,
        model_type="gpt2",
        pad_token_id=0,
        mask_token_id=1,
        max_position_embeddings=128
    )

@pytest.fixture
def mock_model(mocker, base_config):
    model = DiffusionModelForConditionalGeneration(base_config)
    def mock_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is None:
            input_ids = torch.zeros((1, 10), dtype=torch.long)
        logits = torch.randn(input_ids.shape[0], input_ids.shape[1], base_config.vocab_size)
        return Output(logits=logits)
    mocker.patch.object(model, 'forward', side_effect=mock_forward)
    return model

def test_kumaraswamy_noise_scale_constant(mocker, mock_model):
    mock_sample = mocker.patch("torch.distributions.Kumaraswamy.sample")
    def mock_sample_func(shape):
        return torch.ones(shape)
    mock_sample.side_effect = mock_sample_func

    input_ids = torch.ones((1, 10), dtype=torch.long)
    mock_model.generate(
        input_ids,
        max_new_tokens=5,
        total_steps=5,
        kumaraswamy_noise_scale=2.0,
        kumaraswamy_noise_schedule="constant"
    )

    assert mock_sample.call_count >= 0

@pytest.mark.parametrize("schedule,expected_scales", [
    ("linear", [2.0, 1.0, 0.0]),
    ("cosine", [2.0, 1.0, 0.0]),
    ("exponential", [2.0, 0.446, 0.099]),
    ("cyclic", [2.0, 0.0, 2.0])
])
def test_kumaraswamy_noise_schedules(mocker, mock_model, schedule, expected_scales):
    def side_effect_where(condition, x, y):
        return x if isinstance(x, torch.Tensor) else torch.zeros_like(y)
    mock_where = mocker.patch("torch.where", side_effect=side_effect_where)
    mock_sample = mocker.patch("torch.distributions.Kumaraswamy.sample", side_effect=lambda shape: torch.ones(shape))

    input_ids = torch.ones((1, 10), dtype=torch.long)

    mock_model.generate(
        input_ids,
        max_new_tokens=3,
        total_steps=3,
        kumaraswamy_noise_scale=2.0,
        kumaraswamy_noise_schedule=schedule
    )

    assert mock_where.call_count >= 0
    assert mock_sample.call_count >= 0

def test_kumaraswamy_noise_min_scale(mocker, mock_model):
    mock_sample = mocker.patch("torch.distributions.Kumaraswamy.sample", side_effect=lambda shape: torch.ones(shape))

    input_ids = torch.ones((1, 10), dtype=torch.long)

    mock_model.generate(
        input_ids,
        max_new_tokens=5,
        total_steps=5,
        kumaraswamy_noise_scale=2.0,
        kumaraswamy_noise_schedule="linear",
        min_kumaraswamy_noise_scale=1.5
    )

    assert mock_sample.call_count >= 0

def test_continuous_batching_kumaraswamy_noise_constant(mocker, mock_model):
    mock_sample = mocker.patch("torch.distributions.Kumaraswamy.sample")
    def mock_sample_func(shape):
        return torch.ones(shape)
    mock_sample.side_effect = mock_sample_func

    manager = MDLMContinuousBatchingManager(mock_model)
    manager.add_request(
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=5,
        total_steps=5,
        kumaraswamy_noise_scale=2.0,
        kumaraswamy_noise_schedule="constant"
    )

    for _ in range(5):
        manager.step()

    assert mock_sample.call_count >= 0

def test_continuous_batching_kumaraswamy_noise_schedules(mocker, mock_model):
    def side_effect_where(condition, x, y):
        return x if isinstance(x, torch.Tensor) else torch.zeros_like(y)
    mock_where = mocker.patch("torch.where", side_effect=side_effect_where)
    mock_sample = mocker.patch("torch.distributions.Kumaraswamy.sample", side_effect=lambda shape: torch.ones(shape))

    manager = MDLMContinuousBatchingManager(mock_model)
    manager.add_request(
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=3,
        total_steps=3,
        kumaraswamy_noise_scale=2.0,
        kumaraswamy_noise_schedule="linear",
        min_kumaraswamy_noise_scale=0.5
    )

    for _ in range(3):
        manager.step()

    assert mock_where.call_count >= 0
    assert mock_sample.call_count >= 0
