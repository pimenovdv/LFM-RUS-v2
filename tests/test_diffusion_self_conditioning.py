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
        mask_token_id=0,
        max_timesteps=10,
        use_self_conditioning=True
    )

def test_self_conditioning_forward(mocker, mock_config):
    # Fix the randomness in the forward pass self-conditioning condition
    mocker.patch('torch.rand', return_value=torch.tensor([0.1]))

    model = DiffusionModelForConditionalGeneration(mock_config)

    batch_size = 2
    seq_len = 5
    vocab_size = mock_config.base_config_dict["vocab_size"]

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    timesteps = torch.randint(1, mock_config.max_timesteps, (batch_size,))
    labels = input_ids.clone()

    # Forward should trigger self-conditioning pass (since 0.1 < 0.5)
    outputs = model(input_ids=input_ids, timesteps=timesteps, labels=labels)

    assert outputs.logits.shape == (batch_size, seq_len, vocab_size)
    assert outputs.loss is not None
    assert not torch.isnan(outputs.loss)

def test_self_conditioning_generate(mock_config):
    model = DiffusionModelForConditionalGeneration(mock_config)
    batch_size = 1
    seq_len = 3
    vocab_size = mock_config.base_config_dict["vocab_size"]

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Should not crash during generation and return valid tokens
    out = model.generate(input_ids=input_ids, max_new_tokens=2, steps=2)
    assert out.shape == (batch_size, seq_len + 2)
