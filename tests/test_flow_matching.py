import pytest
import torch
from src.models.diffusion.configuration_diffusion import DiffusionConfig
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration

@pytest.fixture
def mock_config():
    return DiffusionConfig(
        base_config_dict={
            "model_type": "gpt2",
            "vocab_size": 100,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2
        },
        use_flow_matching=True,
        mask_token_id=0,
        max_timesteps=100
    )

def test_flow_matching_loss(mock_config):
    model = DiffusionModelForConditionalGeneration(mock_config)
    input_ids = torch.randint(1, 100, (2, 10))
    # mask some tokens
    input_ids[:, 5:] = mock_config.mask_token_id

    timesteps = torch.tensor([10, 50])
    target_ids = torch.randint(1, 100, (2, 10))

    loss = model.compute_flow_matching_loss(input_ids, timesteps, target_ids)

    assert loss.shape == torch.Size([])
    assert loss.item() > 0.0

def test_flow_matching_loss_no_masks(mock_config):
    model = DiffusionModelForConditionalGeneration(mock_config)
    # no masks
    input_ids = torch.randint(1, 100, (2, 10))
    timesteps = torch.tensor([10, 50])
    target_ids = torch.randint(1, 100, (2, 10))

    loss = model.compute_flow_matching_loss(input_ids, timesteps, target_ids)

    assert loss.shape == torch.Size([])
    assert loss.item() == 0.0
