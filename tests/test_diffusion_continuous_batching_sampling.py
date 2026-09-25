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
import torch
from tests.test_diffusion_continuous_batching_sampling import get_mock_model

def test_coverage(mocker):
    model = get_mock_model(mocker)
    requests = [
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "top_k": 5, "top_k_schedule": "cosine"},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "top_k": 5, "top_k_schedule": "exponential"},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "top_k": 5, "top_k_schedule": "cyclic"},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "top_p": 0.5, "top_p_schedule": "linear"},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "top_p": 0.5, "top_p_schedule": "exponential"},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "top_p": 0.5, "top_p_schedule": "cyclic"},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "min_p": 0.5, "min_p_schedule": "linear"},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "min_p": 0.5, "min_p_schedule": "cosine"},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "min_p": 0.5, "min_p_schedule": "cyclic"}
    ]
    model.generate_dynamic_batch(requests)

def test_continuous_batching_typical_p(mocker):
    model = get_mock_model(mocker)

    requests = [
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "typical_p": 0.5},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "typical_p": 0.5, "typical_p_schedule": "linear", "min_typical_p": 0.1},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "typical_p": 0.5, "typical_p_schedule": "cosine", "min_typical_p": 0.1},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "typical_p": 0.5, "typical_p_schedule": "exponential", "min_typical_p": 0.1},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "typical_p": 0.5, "typical_p_schedule": "cyclic", "min_typical_p": 0.1}
    ]

    results = model.generate_dynamic_batch(requests)
    assert len(results) == 5
    for res in results:
        assert res is not None

def test_continuous_batching_epsilon_cutoff(mocker):
    model = get_mock_model(mocker)

    requests = [
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "epsilon_cutoff": 0.05},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "epsilon_cutoff": 0.05, "epsilon_cutoff_schedule": "linear", "min_epsilon_cutoff": 0.01},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "epsilon_cutoff": 0.05, "epsilon_cutoff_schedule": "cosine", "min_epsilon_cutoff": 0.01},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "epsilon_cutoff": 0.05, "epsilon_cutoff_schedule": "exponential", "min_epsilon_cutoff": 0.01},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "epsilon_cutoff": 0.05, "epsilon_cutoff_schedule": "cyclic", "min_epsilon_cutoff": 0.01}
    ]

    results = model.generate_dynamic_batch(requests)
    assert len(results) == 5
    for res in results:
        assert res is not None

def test_continuous_batching_eta_cutoff(mocker):
    model = get_mock_model(mocker)

    requests = [
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "eta_cutoff": 0.05},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "eta_cutoff": 0.05, "eta_cutoff_schedule": "linear", "min_eta_cutoff": 0.01},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "eta_cutoff": 0.05, "eta_cutoff_schedule": "cosine", "min_eta_cutoff": 0.01},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "eta_cutoff": 0.05, "eta_cutoff_schedule": "exponential", "min_eta_cutoff": 0.01},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "eta_cutoff": 0.05, "eta_cutoff_schedule": "cyclic", "min_eta_cutoff": 0.01}
    ]

    results = model.generate_dynamic_batch(requests)
    assert len(results) == 5
    for res in results:
        assert res is not None

def test_continuous_batching_top_n_tokens(mocker):
    model = get_mock_model(mocker)

    requests = [
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "top_n_tokens": 5},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "top_n_tokens": 10, "top_n_tokens_schedule": "linear", "min_top_n_tokens": 2},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "top_n_tokens": 10, "top_n_tokens_schedule": "cosine", "min_top_n_tokens": 2},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "top_n_tokens": 10, "top_n_tokens_schedule": "exponential", "min_top_n_tokens": 2},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "top_n_tokens": 10, "top_n_tokens_schedule": "cyclic", "min_top_n_tokens": 2}
    ]

    results = model.generate_dynamic_batch(requests)
    assert len(results) == 5
    for res in results:
        assert res is not None

def test_continuous_batching_mirostat(mocker):
    model = get_mock_model(mocker)

    requests = [
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "mirostat_mode": 1},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "mirostat_mode": 1, "mirostat_tau_schedule": "linear", "min_mirostat_tau": 0.5},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "mirostat_mode": 1, "mirostat_tau_schedule": "cosine", "min_mirostat_tau": 0.5},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "mirostat_mode": 1, "mirostat_tau_schedule": "exponential", "min_mirostat_tau": 0.5},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "mirostat_mode": 1, "mirostat_tau_schedule": "cyclic", "min_mirostat_tau": 0.5}
    ]

    results = model.generate_dynamic_batch(requests)
    assert len(results) == 5
    for res in results:
        assert res is not None

def test_continuous_batching_tfs(mocker):
    model = get_mock_model(mocker)

    requests = [
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "tfs": 0.5},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "tfs": 0.5, "tfs_schedule": "linear", "min_tfs": 0.1},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "tfs": 0.5, "tfs_schedule": "cosine", "min_tfs": 0.1},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "tfs": 0.5, "tfs_schedule": "exponential", "min_tfs": 0.1},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "tfs": 0.5, "tfs_schedule": "cyclic", "min_tfs": 0.1}
    ]

    results = model.generate_dynamic_batch(requests)
    assert len(results) == 5
    for res in results:
        assert res is not None

def test_continuous_batching_top_a(mocker):
    model = get_mock_model(mocker)

    requests = [
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "top_a": 0.5},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "top_a": 0.5, "top_a_schedule": "linear", "min_top_a": 0.1},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "top_a": 0.5, "top_a_schedule": "cosine", "min_top_a": 0.1},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "top_a": 0.5, "top_a_schedule": "exponential", "min_top_a": 0.1},
        {"input_ids": torch.tensor([[1, 2]]), "max_new_tokens": 1, "total_steps": 2, "top_a": 0.5, "top_a_schedule": "cyclic", "min_top_a": 0.1}
    ]

    results = model.generate_dynamic_batch(requests)
    assert len(results) == 5
    for res in results:
        assert res is not None
