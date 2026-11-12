import pytest
import torch
import math
from transformers import AutoConfig
from src.models.diffusion.configuration_diffusion import DiffusionConfig
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, MDLMContinuousBatchingManager

@pytest.fixture
def dummy_model():
    base_config = AutoConfig.for_model("gpt2")
    base_config.n_layer = 2
    base_config.n_head = 2
    base_config.n_embd = 32
    base_config.vocab_size = 100
    config = DiffusionConfig(
        base_config_dict=base_config.to_dict(),
        mask_token_id=0,
        diffusion_steps=4,
        block_size=2,
        timestep_dim=16
    )
    model = DiffusionModelForConditionalGeneration(config)
    return model

def test_length_temperature_scale(dummy_model, mocker):
    input_ids = torch.randint(0, 100, (2, 5))
    attention_mask = torch.ones_like(input_ids)

    mock_forward = mocker.patch.object(dummy_model, "forward")
    class Output:
        def __init__(self, logits):
            self.logits = logits

    mock_forward.side_effect = lambda *args, **kwargs: Output(torch.randn(kwargs['input_ids'].size(0), kwargs['input_ids'].size(1), 100))

    generated_tokens = dummy_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=5,
        steps=5,
        temperature=1.0,
        length_temperature_scale=0.5,
        length_temperature_schedule="linear",
        min_length_temperature_scale=0.1
    )

    assert generated_tokens.shape == (2, 10)

def test_length_temperature_scale_continuous_batching(dummy_model, mocker):
    manager = MDLMContinuousBatchingManager(dummy_model, max_batch_size=2)

    mock_forward = mocker.patch.object(dummy_model, "forward")
    class Output:
        def __init__(self, logits):
            self.logits = logits

    manager.add_request(
        input_ids=torch.randint(0, 100, (1, 5)),
        max_new_tokens=5,
        total_steps=5,
        temperature=1.0,
        length_temperature_scale=0.5,
        length_temperature_schedule="cosine",
        min_length_temperature_scale=0.1
    )

    mock_forward.side_effect = lambda *args, **kwargs: Output(torch.randn(kwargs.get('input_ids', args[0] if args else None).size(0), kwargs.get('input_ids', args[0] if args else None).size(1), 100))
    has_active = manager.step()

    assert has_active is True
    assert manager.active_requests[0].length_temperature_scale == 0.5
