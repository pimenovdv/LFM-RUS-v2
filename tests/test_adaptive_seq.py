import pytest
import torch
import math
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, DiffusionConfig
from transformers import AutoModelForCausalLM, AutoConfig

class MockOutput:
    def __init__(self, logits):
        self.logits = logits
        self.last_hidden_state = logits

class MockAutoModelForCausalLM(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, **kwargs):
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, self.config.vocab_size)
        return MockOutput(logits)

def test_adaptive_sequence_length_guidance(mocker):
    config = DiffusionConfig(
        vocab_size=100,
        model_type="gpt2",
        diffusion_steps=4,
        block_size=16,
        mask_token_id=99,
        base_config_dict={"vocab_size": 100, "model_type": "gpt2"}
    )

    hf_config = AutoConfig.for_model("gpt2", vocab_size=100)
    inner_model = MockAutoModelForCausalLM(hf_config)

    model = DiffusionModelForConditionalGeneration(config)
    model.eval()
    mocker.patch.object(model, "forward", return_value=MockOutput(torch.randn(1, 15, 100)))

    input_ids = torch.zeros((1, 5), dtype=torch.long)
    unconditional_input_ids = torch.zeros((1, 5), dtype=torch.long)

    # We just run generate to ensure no crashes
    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=10,
        cfg_scale=2.0,
        length_cfg_scale=0.5,
        length_cfg_schedule="linear",
        unconditional_input_ids=unconditional_input_ids
    )
    assert out.shape == (1, 15)

    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=10,
        cfg_scale=2.0,
        length_cfg_scale=0.5,
        length_cfg_schedule="cosine",
        unconditional_input_ids=unconditional_input_ids
    )
    assert out.shape == (1, 15)

    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=10,
        cfg_scale=2.0,
        length_cfg_scale=0.5,
        length_cfg_schedule="exponential",
        unconditional_input_ids=unconditional_input_ids
    )
    assert out.shape == (1, 15)

    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=10,
        cfg_scale=2.0,
        length_cfg_scale=0.5,
        length_cfg_schedule="cyclic",
        unconditional_input_ids=unconditional_input_ids
    )
    assert out.shape == (1, 15)

def test_information_gain_masking(mocker):
    config = DiffusionConfig(
        vocab_size=100,
        model_type="gpt2",
        diffusion_steps=4,
        block_size=16,
        mask_token_id=99,
        base_config_dict={"vocab_size": 100, "model_type": "gpt2"}
    )

    hf_config = AutoConfig.for_model("gpt2", vocab_size=100)
    inner_model = MockAutoModelForCausalLM(hf_config)

    model = DiffusionModelForConditionalGeneration(config)
    model.eval()
    mocker.patch.object(model, "forward", return_value=MockOutput(torch.randn(1, 15, 100)))

    input_ids = torch.zeros((1, 5), dtype=torch.long)

    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=10,
        remasking="info_gain"
    )

    assert out.shape == (1, 15)

def test_information_gain_masking_with_uncond(mocker):
    config = DiffusionConfig(
        vocab_size=100,
        model_type="gpt2",
        diffusion_steps=4,
        block_size=16,
        mask_token_id=99,
        base_config_dict={"vocab_size": 100, "model_type": "gpt2"}
    )

    hf_config = AutoConfig.for_model("gpt2", vocab_size=100)
    inner_model = MockAutoModelForCausalLM(hf_config)

    model = DiffusionModelForConditionalGeneration(config)
    model.eval()
    mocker.patch.object(model, "forward", return_value=MockOutput(torch.randn(1, 15, 100)))

    input_ids = torch.zeros((1, 5), dtype=torch.long)
    unconditional_input_ids = torch.zeros((1, 5), dtype=torch.long)

    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=10,
        remasking="info_gain",
        cfg_scale=2.0,
        unconditional_input_ids=unconditional_input_ids
    )

    assert out.shape == (1, 15)
