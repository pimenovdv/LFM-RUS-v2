import torch
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, DiffusionConfig
from unittest.mock import MagicMock

def get_mock_model(mocker):
    config = DiffusionConfig(
        base_config_dict={"model_type": "gpt2", "vocab_size": 100, "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2},
        diffusion_steps=4,
        mask_token_id=99,
        max_timesteps=100
    )
    model = DiffusionModelForConditionalGeneration(config)

    def mock_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        batch_size, seq_len = input_ids.shape

        if not hasattr(mock_forward, "call_count"):
            mock_forward.call_count = 0
        mock_forward.call_count += 1

        logits = torch.ones((batch_size, seq_len, config.base_config_dict["vocab_size"])) * mock_forward.call_count
        return MagicMock(logits=logits)

    mocker.patch.object(model, 'forward', side_effect=mock_forward)
    return model

def test_gaussian_noise_basic(mocker):
    model = get_mock_model(mocker)
    input_ids = torch.tensor([[1, 2, 3]])

    # Seed is not set, we just want to ensure it runs without errors
    model.generate(input_ids, max_new_tokens=1, steps=4, gaussian_noise_std=0.5)
    assert model.forward.call_count == 4

def test_gaussian_noise_schedules(mocker):
    model = get_mock_model(mocker)
    input_ids = torch.tensor([[1, 2, 3]])

    schedules = ["linear", "cosine", "exponential", "cyclic"]
    for schedule in schedules:
        model = get_mock_model(mocker)
        model.generate(input_ids, max_new_tokens=1, steps=4, gaussian_noise_std=0.5, gaussian_noise_schedule=schedule)
        assert model.forward.call_count == 4

def test_gaussian_noise_continuous_batching(mocker):
    model = get_mock_model(mocker)

    requests = [
        {"input_ids": torch.tensor([[1, 2, 3]]), "max_new_tokens": 1, "total_steps": 4, "gaussian_noise_std": 0.5},
        {"input_ids": torch.tensor([[4, 5]]), "max_new_tokens": 2, "total_steps": 2, "gaussian_noise_std": 0.8, "gaussian_noise_schedule": "linear"}
    ]

    results = model.generate_dynamic_batch(requests)
    assert len(results) == 2
