import torch
import pytest
from src.models.diffusion.configuration_diffusion import DiffusionConfig
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, MDLMContinuousBatchingManager, MDLMRequest
import math

class Output:
    def __init__(self, logits):
        self.logits = logits

@pytest.fixture
def mock_model(mocker):
    config = DiffusionConfig(
        vocab_size=100,
        diffusion_steps=10,
        mask_token_id=99,
        base_config_dict={"vocab_size": 100, "model_type": "gpt2"}
    )
    model = DiffusionModelForConditionalGeneration(config)

    # Mocking the forward method
    def mock_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and "inputs_embeds" in kwargs:
            inputs_embeds = kwargs["inputs_embeds"]
            batch_size, seq_len = inputs_embeds.shape[:2]
        else:
            batch_size, seq_len = input_ids.shape[:2]

        logits = torch.randn(batch_size, seq_len, 100)
        return Output(logits)

    mocker.patch.object(model, "forward", side_effect=mock_forward)
    return model

def test_generate_with_exponential_noise(mock_model, mocker):
    input_ids = torch.tensor([[1, 2, 3]])

    # Mock exponential distribution
    mock_sample = mocker.patch("torch.distributions.exponential.Exponential.sample")
    mock_sample.side_effect = lambda shape: torch.ones(shape)

    # Run generate
    output = mock_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        exponential_noise_scale=1.0,
        exponential_noise_schedule="linear",
        min_exponential_noise_scale=0.1
    )

    assert output is not None
    assert output.shape[1] == 5 # 3 prompt + 2 generated
    assert mock_sample.called

def test_continuous_batching_with_exponential_noise(mock_model, mocker):
    manager = MDLMContinuousBatchingManager(mock_model, max_batch_size=2)

    input_ids = torch.tensor([[1, 2, 3]])
    config = {
        "input_ids": input_ids,
        "max_new_tokens": 2,
        "total_steps": 2,
        "exponential_noise_scale": 1.0,
        "exponential_noise_schedule": "cosine",
        "min_exponential_noise_scale": 0.1
    }

    # Mock exponential distribution
    mock_sample = mocker.patch("torch.distributions.exponential.Exponential.sample")
    # Due to padding and context length: batch_size=1, seq_len=5, vocab_size=100
    mock_sample.side_effect = lambda shape: torch.ones(shape)

    req_id = manager.add_request(**config)

    while manager.active_requests or manager.pending_requests:
        manager.step()

    assert len(manager.completed_requests) == 1
    assert manager.completed_requests[0].result.shape[1] == 5
    assert mock_sample.called

def test_mdlm_request_exponential_noise_defaults():
    input_ids = torch.tensor([[1, 2, 3]])
    req = MDLMRequest(input_ids=input_ids, max_new_tokens=5, total_steps=5)

    assert req.exponential_noise_scale == 0.0
    assert req.exponential_noise_schedule == "constant"
    assert req.min_exponential_noise_scale == 0.0

def test_exponential_noise_schedules_generate(mock_model, mocker):
    input_ids = torch.tensor([[1, 2, 3]])

    mock_sample = mocker.patch("torch.distributions.exponential.Exponential.sample")
    mock_sample.side_effect = lambda shape: torch.ones(shape)

    schedules = ["linear", "cosine", "exponential", "cyclic", "constant"]

    for schedule in schedules:
        output = mock_model.generate(
            input_ids,
            max_new_tokens=2,
            steps=2,
            exponential_noise_scale=2.0,
            exponential_noise_schedule=schedule,
            min_exponential_noise_scale=0.5
        )
        assert output is not None

def test_exponential_noise_schedules_continuous_batching(mock_model, mocker):
    manager = MDLMContinuousBatchingManager(mock_model, max_batch_size=2)
    input_ids = torch.tensor([[1, 2, 3]])

    mock_sample = mocker.patch("torch.distributions.exponential.Exponential.sample")
    mock_sample.side_effect = lambda shape: torch.ones(shape)

    schedules = ["linear", "cosine", "exponential", "cyclic", "constant"]

    for schedule in schedules:
        config = {
            "input_ids": input_ids,
            "max_new_tokens": 2,
            "total_steps": 2,
            "exponential_noise_scale": 2.0,
            "exponential_noise_schedule": schedule,
            "min_exponential_noise_scale": 0.5
        }
        manager.add_request(**config)

    while manager.active_requests or manager.pending_requests:
        manager.step()

    assert len(manager.completed_requests) == 5
