import pytest
import torch
from src.models.diffusion.configuration_diffusion import DiffusionConfig
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, DiffusionControlNetModel

@pytest.fixture
def dummy_config():
    return DiffusionConfig(
        base_config_dict={
            "model_type": "bert",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 32,
            "vocab_size": 100,
        },
        mask_token_id=0,
        max_timesteps=10
    )

def test_controlnet_initialization(dummy_config):
    model = DiffusionControlNetModel(dummy_config)
    assert hasattr(model, "zero_projections")
    assert len(model.zero_projections) == 2
    # Verify zero init
    for proj in model.zero_projections:
        assert torch.all(proj.weight == 0)
        assert torch.all(proj.bias == 0)

def test_controlnet_forward(dummy_config):
    model = DiffusionControlNetModel(dummy_config)
    input_ids = torch.randint(0, 100, (2, 5))
    timesteps = torch.zeros((2,), dtype=torch.long)
    control_states = model(input_ids=input_ids, timesteps=timesteps)
    assert len(control_states) == 2
    for state in control_states:
        # Since zero projections are 0, output should be exact zeros
        assert torch.all(state == 0)
        assert state.shape == (2, 5, 16)

def test_main_model_forward_with_control_states(dummy_config):
    model = DiffusionModelForConditionalGeneration(dummy_config)
    control_model = DiffusionControlNetModel(dummy_config)

    input_ids = torch.randint(0, 100, (1, 5))
    timesteps = torch.zeros((1,), dtype=torch.long)

    # 1. Forward without control states
    outputs_baseline = model(input_ids=input_ids, timesteps=timesteps)

    model.eval()
    control_model.eval()

    # 2. Forward with exact zeros control states
    with torch.no_grad():
        outputs_baseline = model(input_ids=input_ids, timesteps=timesteps)
        control_states = control_model(input_ids=input_ids, timesteps=timesteps)
        outputs_with_zeros = model(input_ids=input_ids, timesteps=timesteps, control_states=control_states)

    # Assert logits are identical since control is zero
    assert torch.allclose(outputs_baseline.logits, outputs_with_zeros.logits, atol=1e-5)

    # 3. Forward with non-zero control states (simulate non-trivial injection)
    non_zero_control_states = [torch.randn_like(s) for s in control_states]
    outputs_with_noise = model(input_ids=input_ids, timesteps=timesteps, control_states=non_zero_control_states)

    # Assert logits changed
    assert not torch.allclose(outputs_baseline.logits, outputs_with_noise.logits)

def test_main_model_generate_with_controlnet(dummy_config):
    model = DiffusionModelForConditionalGeneration(dummy_config)
    control_model = DiffusionControlNetModel(dummy_config)

    input_ids = torch.randint(1, 100, (1, 10))
    # Add masks
    input_ids[0, 5:] = 0

    # Test generation fallback to main input
    generated = model.generate(
        input_ids=input_ids.clone(),
        steps=2,
        control_model=control_model,
        control_scale=1.0,
        max_new_tokens=0,
    )
    assert generated.shape == (1, 10)

    # Test generation with explicit control inputs
    control_input_ids = torch.randint(1, 100, (1, 10))
    generated_explicit = model.generate(
        input_ids=input_ids.clone(),
        steps=2,
        control_model=control_model,
        control_input_ids=control_input_ids,
        control_scale=1.5,
        max_new_tokens=0,
    )
    assert generated_explicit.shape == (1, 10)

    # With CFG
    generated_cfg = model.generate(
        input_ids=input_ids.clone(),
        steps=2,
        control_model=control_model,
        control_scale=1.0,
        cfg_scale=1.5,
        unconditional_input_ids=torch.randint(1, 100, (1, 5)),
        max_new_tokens=0,
    )
    assert generated_cfg.shape == (1, 10)

def test_main_model_generate_with_speculative_controlnet(dummy_config):
    model = DiffusionModelForConditionalGeneration(dummy_config)
    draft_model = DiffusionModelForConditionalGeneration(dummy_config)
    control_model = DiffusionControlNetModel(dummy_config)

    input_ids = torch.randint(1, 100, (1, 10))
    input_ids[0, 5:] = 0

    generated = model.generate(
        input_ids=input_ids.clone(),
        steps=2,
        control_model=control_model,
        draft_model=draft_model,
        speculative_steps=2,
        max_new_tokens=0,
    )
    assert generated.shape == (1, 10)
