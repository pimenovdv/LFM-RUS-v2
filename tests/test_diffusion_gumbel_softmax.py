import pytest
import torch
import math
import torch.nn.functional as F
from unittest.mock import MagicMock, call
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, DiffusionConfig, MDLMContinuousBatchingManager, MDLMRequest

class Output:
    def __init__(self, logits):
        self.logits = logits

@pytest.fixture
def mock_model(mocker):
    config = DiffusionConfig(vocab_size=100, mask_token_id=0, pad_token_id=1, base_config_dict={"vocab_size": 100, "model_type": "gpt2"})
    model = DiffusionModelForConditionalGeneration(config)

    def forward_mock(input_ids, **kwargs):
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, config.vocab_size)
        return Output(logits)

    mocker.patch.object(model, "forward", side_effect=forward_mock)
    model.eval()
    return model

def test_gumbel_softmax_noise_generation(mock_model, mocker):
    input_ids = torch.tensor([[0, 2, 3]])

    mock_softmax = mocker.patch("torch.nn.functional.softmax", wraps=F.softmax)
    mock_where = mocker.patch("torch.where", side_effect=lambda cond, x, y: x)

    mock_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        gumbel_softmax_noise_scale=0.5,
        gumbel_softmax_temperature=1.0,
        gumbel_temperature=0.0
    )

    assert mock_softmax.called
    assert mock_where.called

def test_gumbel_softmax_noise_schedule(mock_model, mocker):
    input_ids = torch.tensor([[0, 0, 0]])

    mock_softmax = mocker.patch("torch.nn.functional.softmax", wraps=F.softmax)
    mock_where = mocker.patch("torch.where", side_effect=lambda cond, x, y: x)

    # Test linear schedule
    mock_model.generate(
        input_ids, max_new_tokens=2, steps=4,
        gumbel_softmax_noise_scale=1.0, gumbel_softmax_noise_schedule="linear",
        gumbel_temperature=0.0
    )
    assert mock_softmax.call_count > 0

    # Test cosine schedule
    mock_softmax.reset_mock()
    mock_model.generate(
        input_ids, max_new_tokens=2, steps=4,
        gumbel_softmax_noise_scale=1.0, gumbel_softmax_noise_schedule="cosine",
        gumbel_temperature=0.0
    )
    assert mock_softmax.call_count > 0

    # Test exponential schedule
    mock_softmax.reset_mock()
    mock_model.generate(
        input_ids, max_new_tokens=2, steps=4,
        gumbel_softmax_noise_scale=1.0, gumbel_softmax_noise_schedule="exponential",
        gumbel_temperature=0.0
    )
    assert mock_softmax.call_count > 0

    # Test cyclic schedule
    mock_softmax.reset_mock()
    mock_model.generate(
        input_ids, max_new_tokens=2, steps=4,
        gumbel_softmax_noise_scale=1.0, gumbel_softmax_noise_schedule="cyclic",
        gumbel_temperature=0.0
    )
    assert mock_softmax.call_count > 0

def test_gumbel_softmax_noise_continuous_batching(mock_model, mocker):
    manager = MDLMContinuousBatchingManager(mock_model, max_batch_size=2)
    req = MDLMRequest(
        input_ids=torch.tensor([[0, 2]]),
        max_new_tokens=2,
        total_steps=2,
        gumbel_softmax_noise_scale=0.5,
        gumbel_softmax_temperature=1.0
    )
    manager.pending_requests.append(req)

    mock_softmax = mocker.patch("torch.nn.functional.softmax", wraps=F.softmax)
    mock_where = mocker.patch("torch.where", side_effect=lambda cond, x, y: x)

    manager.step()

    assert mock_softmax.called
    assert mock_where.called
