import torch
import pytest
from src.models.diffusion.configuration_diffusion import DiffusionConfig
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration

def test_diffusion_self_conditioning(mocker):
    config = DiffusionConfig(
        mask_token_id=0,
        diffusion_steps=5,
        base_config_dict={"vocab_size": 100, "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 1, "model_type": "gpt2"},
        use_self_conditioning=True
    )
    model = DiffusionModelForConditionalGeneration(config)

    # We need to mock forward to track if inputs_embeds is passed
    original_forward = model.forward
    forward_spy = mocker.spy(model, "forward")

    # Mocking to make it faster and predictable if needed, but since we use small config,
    # real forward is fast enough and actually tests the embeddings projection logic.

    input_ids = torch.tensor([[1, 2, 3]])

    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        steps=5
    )

    assert output.shape == (1, 8)

    # Check that forward was called with inputs_embeds at least once (i > 0)
    called_with_embeds = False
    for call in forward_spy.call_args_list:
        args, kwargs = call
        if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
            called_with_embeds = True
            break

    assert called_with_embeds, "Forward was not called with inputs_embeds, so self-conditioning did not trigger."
    assert not hasattr(model, "prev_logits"), "Model should not save prev_logits as instance attribute."

def test_diffusion_no_self_conditioning(mocker):
    config = DiffusionConfig(
        mask_token_id=0,
        diffusion_steps=5,
        base_config_dict={"vocab_size": 100, "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 1, "model_type": "gpt2"},
        use_self_conditioning=False
    )
    model = DiffusionModelForConditionalGeneration(config)

    forward_spy = mocker.spy(model, "forward")

    input_ids = torch.tensor([[1, 2, 3]])

    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        steps=5
    )

    assert output.shape == (1, 8)

    # Check that forward was never called with inputs_embeds
    called_with_embeds = False
    for call in forward_spy.call_args_list:
        args, kwargs = call
        if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
            called_with_embeds = True
            break

    assert not called_with_embeds, "Forward should not be called with inputs_embeds when use_self_conditioning is False."

def test_diffusion_self_conditioning_training(mocker):
    config = DiffusionConfig(
        mask_token_id=0,
        diffusion_steps=5,
        base_config_dict={"vocab_size": 100, "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 1, "model_type": "gpt2"},
        use_self_conditioning=True
    )
    model = DiffusionModelForConditionalGeneration(config)
    model.train() # Make sure model is in training mode

    # We want to force rand() to be > 0.5 to trigger self conditioning
    mocker.patch("torch.rand", return_value=torch.tensor([0.8]))

    # Spy on the inner_model's forward pass to see if it gets called twice
    # (Once for the preliminary pass, once for the actual pass)
    inner_forward_spy = mocker.spy(model.inner_model, "forward")

    input_ids = torch.tensor([[1, 2, 3]])
    timesteps = torch.tensor([1])

    model(input_ids=input_ids, timesteps=timesteps)

    assert inner_forward_spy.call_count == 2, "Inner model should be called twice when self-conditioning is triggered during training"

def test_diffusion_self_conditioning_training_no_trigger(mocker):
    config = DiffusionConfig(
        mask_token_id=0,
        diffusion_steps=5,
        base_config_dict={"vocab_size": 100, "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 1, "model_type": "gpt2"},
        use_self_conditioning=True
    )
    model = DiffusionModelForConditionalGeneration(config)
    model.train()

    # Force rand() to be <= 0.5 to NOT trigger self conditioning
    mocker.patch("torch.rand", return_value=torch.tensor([0.2]))

    inner_forward_spy = mocker.spy(model.inner_model, "forward")

    input_ids = torch.tensor([[1, 2, 3]])
    timesteps = torch.tensor([1])

    model(input_ids=input_ids, timesteps=timesteps)

    assert inner_forward_spy.call_count == 1, "Inner model should be called once when self-conditioning is not triggered"
