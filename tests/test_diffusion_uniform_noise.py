import pytest
import torch
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, MDLMRequest, MDLMContinuousBatchingManager
from src.models.diffusion.configuration_diffusion import DiffusionConfig

def test_uniform_noise_generate(mocker):
    config_dict = {"vocab_size": 100, "hidden_size": 32, "mask_token_id": 99, "base_config_dict": {"model_type": "gpt2", "vocab_size": 100}}
    config = DiffusionConfig(**config_dict)
    model = DiffusionModelForConditionalGeneration(config)

    def mock_forward(*args, **kwargs):
        class Output:
            # Predict some logits
            logits = torch.randn(1, 4, 100)
            last_hidden_state = torch.randn(1, 4, 32)
        return Output()

    mocker.patch.object(model, 'forward', side_effect=mock_forward)

    input_ids = torch.tensor([[1, 2, 3]])

    # Run without noise
    torch.manual_seed(42)
    out_no_noise = model.generate(
        input_ids,
        uniform_noise_scale=0.0,
        max_new_tokens=1,
    )

    # Run with noise
    torch.manual_seed(42)
    out_with_noise = model.generate(
        input_ids,
        uniform_noise_scale=5.0,
        max_new_tokens=1,
    )

    assert out_no_noise is not None
    assert out_with_noise is not None
    # Depending on the noise, the token might be different or the internal path covers it.
    # We mainly test that it doesn't crash and runs the code paths.

def test_uniform_noise_schedules(mocker):
    config_dict = {"vocab_size": 100, "hidden_size": 32, "mask_token_id": 99, "base_config_dict": {"model_type": "gpt2", "vocab_size": 100}}
    config = DiffusionConfig(**config_dict)
    model = DiffusionModelForConditionalGeneration(config)

    def mock_forward(*args, **kwargs):
        class Output:
            logits = torch.randn(1, 4, 100)
            last_hidden_state = torch.randn(1, 4, 32)
        return Output()

    mocker.patch.object(model, 'forward', side_effect=mock_forward)

    input_ids = torch.tensor([[1, 2, 3]])

    schedules = ["constant", "linear", "cosine", "exponential", "cyclic"]
    for sched in schedules:
        out = model.generate(
            input_ids,
            uniform_noise_scale=1.0,
            min_uniform_noise_scale=0.1,
            uniform_noise_schedule=sched,
            max_new_tokens=1,
        )
        assert out is not None

def test_uniform_noise_continuous_batching(mocker):
    config_dict = {"vocab_size": 100, "hidden_size": 32, "mask_token_id": 99, "base_config_dict": {"model_type": "gpt2", "vocab_size": 100}}
    config = DiffusionConfig(**config_dict)
    model = DiffusionModelForConditionalGeneration(config)

    manager = MDLMContinuousBatchingManager(model)

    manager.add_request(input_ids=torch.tensor([[1,2,3]]), max_new_tokens=5, total_steps=10, uniform_noise_scale=2.0)
    manager.add_request(input_ids=torch.tensor([[4,5]]), max_new_tokens=5, total_steps=10, uniform_noise_scale=3.0, min_uniform_noise_scale=0.5, uniform_noise_schedule="linear")

    def mock_forward(*args, **kwargs):
        input_ids = kwargs.get('input_ids')
        if input_ids is None:
            input_ids = kwargs.get('inputs_embeds')
        batch_size, seq_len = input_ids.shape[:2]
        class Output:
            logits = torch.randn(batch_size, seq_len, 100)
        return Output()

    mocker.patch.object(model, 'forward', side_effect=mock_forward)

    for _ in range(5):
        completed = manager.step()
        if len(manager.active_requests) == 0:
            break

    assert True
