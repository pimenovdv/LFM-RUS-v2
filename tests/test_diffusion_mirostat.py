import torch
import pytest
from src.models.diffusion.configuration_diffusion import DiffusionConfig
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration

def test_diffusion_mirostat(mocker):
    config = DiffusionConfig(
        mask_token_id=0,
        diffusion_steps=5,
        base_config_dict={"vocab_size": 100, "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 1, "model_type": "gpt2"},
    )
    model = DiffusionModelForConditionalGeneration(config)

    input_ids = torch.tensor([[1, 2, 3]])

    # Spy on torch.argmax to verify generation still works
    argmax_spy = mocker.spy(torch, "argmax")

    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        steps=5,
        mirostat_mode=1,
        mirostat_tau=5.0,
        mirostat_eta=0.1
    )

    assert output.shape == (1, 8)
    # Ensure argmax was called (i.e., generation loop ran)
    assert argmax_spy.call_count > 0

    # Also test with a specific mu to see if it doesn't crash
    output_mu = model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        steps=5,
        mirostat_mode=1,
        mirostat_tau=5.0,
        mirostat_eta=0.1,
        mirostat_mu=10.0
    )

    assert output_mu.shape == (1, 8)

def test_diffusion_mirostat_schedules(mocker):
    config = DiffusionConfig(
        mask_token_id=0,
        diffusion_steps=5,
        base_config_dict={"vocab_size": 100, "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 1, "model_type": "gpt2"},
    )
    model = DiffusionModelForConditionalGeneration(config)

    input_ids = torch.tensor([[1, 2, 3]])

    schedules = ["linear", "cosine", "exponential", "cyclic"]

    for schedule in schedules:
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=5,
            steps=5,
            mirostat_mode=1,
            mirostat_tau=5.0,
            mirostat_tau_schedule=schedule,
            min_mirostat_tau=1.0,
            mirostat_eta=0.1,
            mirostat_eta_schedule=schedule,
            min_mirostat_eta=0.01
        )
        assert output.shape == (1, 8)
