import torch
import pytest
from transformers import AutoConfig
from src.models.diffusion.configuration_diffusion import DiffusionConfig
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration

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

def test_generate_dynamic_cfg(dummy_model, mocker):
    batch_size = 2
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))
    uncond_ids = torch.randint(1, 100, (batch_size, seq_len))

    # We mock the math.cos to verify cosine is called or we just test they run without crashing
    out_linear = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        cfg_scale=2.0,
        cfg_schedule="linear",
        unconditional_input_ids=uncond_ids
    )

    out_cosine = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        cfg_scale=2.0,
        cfg_schedule="cosine",
        unconditional_input_ids=uncond_ids
    )

    assert out_linear.shape == (batch_size, seq_len + 2)
    assert out_cosine.shape == (batch_size, seq_len + 2)

def test_generate_top_k_top_p(dummy_model):
    batch_size = 2
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))

    out_top_k = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        top_k=5
    )

    out_top_p = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        top_p=0.9
    )

    out_both = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        top_k=5,
        top_p=0.9
    )

    assert out_top_k.shape == (batch_size, seq_len + 2)
    assert out_top_p.shape == (batch_size, seq_len + 2)
    assert out_both.shape == (batch_size, seq_len + 2)

def test_generate_temperature(dummy_model):
    batch_size = 2
    seq_len = 4
    input_ids = torch.randint(1, 100, (batch_size, seq_len))

    out_temp = dummy_model.generate(
        input_ids,
        max_new_tokens=2,
        steps=2,
        temperature=0.8
    )

    assert out_temp.shape == (batch_size, seq_len + 2)
