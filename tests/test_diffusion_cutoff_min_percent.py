import torch
import pytest
from src.models.diffusion.configuration_diffusion import DiffusionConfig
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration

def test_diffusion_cutoff_min_percent(mocker):
    config = DiffusionConfig(
        mask_token_id=0,
        diffusion_steps=5,
        base_config_dict={"vocab_size": 100, "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 1, "model_type": "gpt2"},
    )
    model = DiffusionModelForConditionalGeneration(config)

    input_ids = torch.tensor([[1, 2, 3]])

    # Spy on torch.argmax to verify generation still works
    argmax_spy = mocker.spy(torch, "argmax")

    # Constant schedule
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        steps=5,
        cutoff_min_percent=0.1
    )
    assert output.shape == (1, 8)
    assert argmax_spy.call_count > 0

    argmax_spy.reset_mock()

    # Linear schedule
    output_linear = model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        steps=5,
        cutoff_min_percent=0.2,
        cutoff_min_percent_schedule="linear",
        min_cutoff_min_percent=0.05
    )
    assert output_linear.shape == (1, 8)
    assert argmax_spy.call_count > 0

    argmax_spy.reset_mock()

    # Cosine schedule
    output_cosine = model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        steps=5,
        cutoff_min_percent=0.15,
        cutoff_min_percent_schedule="cosine"
    )
    assert output_cosine.shape == (1, 8)
    assert argmax_spy.call_count > 0

    argmax_spy.reset_mock()

    # Exponential schedule
    output_exp = model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        steps=5,
        cutoff_min_percent=0.3,
        cutoff_min_percent_schedule="exponential"
    )
    assert output_exp.shape == (1, 8)
    assert argmax_spy.call_count > 0

    argmax_spy.reset_mock()

    # Cyclic schedule
    output_cyclic = model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        steps=5,
        cutoff_min_percent=0.25,
        cutoff_min_percent_schedule="cyclic"
    )
    assert output_cyclic.shape == (1, 8)
    assert argmax_spy.call_count > 0
