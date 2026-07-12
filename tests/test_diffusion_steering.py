import torch
import pytest
from transformers import AutoConfig
from src.models.diffusion.configuration_diffusion import DiffusionConfig
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration

@pytest.fixture
def dummy_model():
    base_config = AutoConfig.for_model("gpt2")
    # Make it very small for tests
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

def test_diffusion_cfg(dummy_model):
    batch_size = 2
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))
    uncond_ids = torch.randint(1, 100, (batch_size, seq_len))

    # Generate without CFG
    out_normal = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2
    )

    # Generate with CFG
    out_cfg = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        cfg_scale=2.0,
        unconditional_input_ids=uncond_ids
    )

    assert out_normal.shape == out_cfg.shape
    assert out_cfg.shape == (batch_size, seq_len + 2)

def test_diffusion_steering(dummy_model):
    batch_size = 1
    seq_len = 4
    positive_ids = torch.randint(1, 100, (batch_size, seq_len))
    negative_ids = torch.randint(1, 100, (batch_size, seq_len))

    # Get steering vector
    steering_vector = dummy_model.get_steering_vector(
        positive_ids,
        negative_ids,
        layer_name="h.0"
    )

    assert steering_vector.shape == (batch_size, dummy_model.config.base_config_dict['n_embd'])

    input_ids = torch.randint(1, 100, (batch_size, seq_len))

    # Forward pass without steering
    out_unsteered = dummy_model(input_ids)

    # Forward pass with steering
    out_steered = dummy_model(
        input_ids,
        steering_vector=steering_vector,
        steering_layer_name="h.0",
        steering_scale=5.0
    )

    # Ensure logits are different
    assert not torch.allclose(out_unsteered.logits, out_steered.logits, atol=1e-5)

def test_generate_with_steering(dummy_model):
    batch_size = 1
    seq_len = 4
    positive_ids = torch.randint(1, 100, (batch_size, seq_len))
    negative_ids = torch.randint(1, 100, (batch_size, seq_len))

    # Get steering vector
    steering_vector = dummy_model.get_steering_vector(
        positive_ids,
        negative_ids,
        layer_name="h.0"
    )

    input_ids = torch.randint(1, 100, (batch_size, seq_len))

    # Generate without steering
    out_unsteered = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2
    )

    # Generate with steering
    out_steered = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        steering_vector=steering_vector,
        steering_layer_name="h.0",
        steering_scale=5.0
    )

    assert out_unsteered.shape == out_steered.shape
    assert out_steered.shape == (batch_size, seq_len + 2)
