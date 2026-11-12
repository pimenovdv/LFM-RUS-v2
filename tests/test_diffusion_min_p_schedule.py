import pytest
import torch
import math
from src.models.diffusion.configuration_diffusion import DiffusionConfig
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration

def test_min_p_schedules(mocker):
    # Mocking the inner AutoModel and AutoConfig
    mock_auto_model = mocker.patch("src.models.diffusion.modeling_diffusion.AutoModel")
    mocker.patch("src.models.diffusion.modeling_diffusion.AutoConfig")
    mock_inner = mocker.MagicMock()

    class FakeConfig:
        is_causal = True
        model_type = "gpt2"

    mock_inner.config = FakeConfig()
    mock_auto_model.from_config.return_value = mock_inner
    mocker.patch("src.models.diffusion.modeling_diffusion.getattr", return_value=False)

    config = DiffusionConfig(
        base_config_dict={"hidden_size": 12, "vocab_size": 10},
        timestep_dim=8,
        mask_token_id=0,
        max_timesteps=10,
        diffusion_steps=3,
        block_size=1
    )
    model = DiffusionModelForConditionalGeneration(config)
    model.lm_head = torch.nn.Linear(12, 10, bias=False)
    model.eval()

    input_ids = torch.tensor([[1, 2]])

    def side_effect(*args, **kwargs):
        if 'return_dict' in kwargs:
            del kwargs['return_dict']
        class MockOut:
            def __init__(self):
                self.last_hidden_state = torch.rand((1, 3, 12))
        return MockOut()

    mock_inner.side_effect = side_effect

    # Test linear
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=1,
        steps=3,
        min_p=0.2,
        min_p_schedule="linear",
        min_min_p=0.05
    )
    assert outputs is not None

    # Test cosine
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=1,
        steps=3,
        min_p=0.2,
        min_p_schedule="cosine"
    )
    assert outputs is not None

    # Test exponential
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=1,
        steps=3,
        min_p=0.2,
        min_p_schedule="exponential"
    )
    assert outputs is not None

    # Test cyclic
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=1,
        steps=3,
        min_p=0.2,
        min_p_schedule="cyclic"
    )
    assert outputs is not None
