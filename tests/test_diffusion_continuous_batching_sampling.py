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

        # We generate deterministic logits to make testing stable
        # Using arange ensures different probabilities for different tokens
        logits = torch.arange(config.base_config_dict["vocab_size"], dtype=torch.float).unsqueeze(0).unsqueeze(0)
        logits = logits.expand(batch_size, seq_len, -1)
        return MagicMock(logits=logits)

    mocker.patch.object(model, 'forward', side_effect=mock_forward)
    return model

def test_continuous_batching_top_k(mocker):
    model = get_mock_model(mocker)

    requests = [
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "top_k": 5},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "top_k": 10, "top_k_schedule": "linear", "min_top_k": 2}
    ]

    results = model.generate_dynamic_batch(requests)
    assert len(results) == 2
    assert results[0] is not None
    assert results[1] is not None

def test_continuous_batching_top_p(mocker):
    model = get_mock_model(mocker)

    requests = [
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "top_p": 0.9},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "top_p": 0.8, "top_p_schedule": "cosine", "min_top_p": 0.2}
    ]

    results = model.generate_dynamic_batch(requests)
    assert len(results) == 2
    assert results[0] is not None
    assert results[1] is not None

def test_continuous_batching_min_p(mocker):
    model = get_mock_model(mocker)

    requests = [
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "min_p": 0.1},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "min_p": 0.2, "min_p_schedule": "exponential", "min_min_p": 0.05}
    ]

    results = model.generate_dynamic_batch(requests)
    assert len(results) == 2
    assert results[0] is not None
    assert results[1] is not None

def test_continuous_batching_temperature_sampling(mocker):
    model = get_mock_model(mocker)

    requests = [
        # Multinomial sampling route
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "temperature": 0.5, "top_k": 10},
        # Argmax route
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "temperature": 0.0, "top_p": 0.9}
    ]

    results = model.generate_dynamic_batch(requests)
    assert len(results) == 2
    assert results[0] is not None
    assert results[1] is not None
