import pytest
import torch
import math
from src.models.diffusion.configuration_diffusion import DiffusionConfig
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, MDLMContinuousBatchingManager, MDLMRequest
import torch.nn.functional as F

@pytest.fixture
def mock_model(mocker):
    config_dict = {
        "vocab_size": 100,
        "mask_token_id": 99,
        "base_config_dict": {
            "model_type": "llama",
            "vocab_size": 100,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "max_position_embeddings": 128,
        }
    }
    config = DiffusionConfig(**config_dict)
    model = DiffusionModelForConditionalGeneration(config)

    def mock_forward(input_ids, *args, **kwargs):
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, config.vocab_size)
        return type('Output', (), {'logits': logits})()

    mocker.patch.object(model, 'forward', side_effect=mock_forward)
    return model

def test_rank_based_sampling_generate(mock_model, mocker):
    input_ids = torch.full((1, 5), mock_model.config.mask_token_id, dtype=torch.long)

    mock_log = mocker.patch("torch.log", side_effect=torch.log)

    mock_model.generate(
        input_ids,
        max_new_tokens=5,
        total_steps=10,
        rank_based_sampling_alpha=1.0,
        rank_based_sampling_schedule="linear"
    )

    assert mock_log.called

def test_epsilon_greedy_sampling_generate(mock_model, mocker):
    input_ids = torch.full((1, 5), mock_model.config.mask_token_id, dtype=torch.long)

    mock_rand = mocker.patch("torch.rand", return_value=torch.zeros(1)) # force random path
    mock_multinomial = mocker.patch("torch.multinomial", side_effect=lambda x, num_samples: torch.zeros((x.shape[0], num_samples), dtype=torch.long))

    out = mock_model.generate(
        input_ids,
        max_new_tokens=5,
        total_steps=10,
        epsilon_greedy_prob=0.9,
    )

    assert mock_rand.called


def test_continuous_batching_rank_based(mock_model, mocker):
    manager = MDLMContinuousBatchingManager(model=mock_model)
    input_ids = torch.full((1, 5), mock_model.config.mask_token_id, dtype=torch.long)

    req_id = manager.add_request(
        input_ids=input_ids,
        max_new_tokens=5,
        total_steps=10,
        rank_based_sampling_alpha=1.5,
        rank_based_sampling_schedule="constant",
    )

    mock_log = mocker.patch("torch.log", side_effect=torch.log)
    manager.step()

    assert mock_log.called

def test_continuous_batching_epsilon_greedy(mock_model, mocker):
    manager = MDLMContinuousBatchingManager(model=mock_model)
    input_ids = torch.full((1, 5), mock_model.config.mask_token_id, dtype=torch.long)

    req_id = manager.add_request(
        input_ids=input_ids,
        max_new_tokens=5,
        total_steps=10,
        epsilon_greedy_prob=0.9,
    )

    mock_rand = mocker.patch("torch.rand", return_value=torch.zeros(1)) # force random path
    mock_multinomial = mocker.patch("torch.multinomial", side_effect=lambda x, num_samples: torch.zeros((x.shape[0], num_samples), dtype=torch.long))

    manager.step()

    assert mock_rand.called
