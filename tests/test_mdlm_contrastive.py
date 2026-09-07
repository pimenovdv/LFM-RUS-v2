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
        max_timesteps=10
    )

def test_mdlm_contrastive_decoding(mocker, mock_config):
    # Mocking amateur and main models
    main_model = DiffusionModelForConditionalGeneration(mock_config)
    amateur_model = DiffusionModelForConditionalGeneration(mock_config)

    # Mock forwards to return deterministic random logits based on shape to ensure stable generation
    def mock_forward(*args, **kwargs):
        input_ids = kwargs.get('input_ids')
        if input_ids is None and len(args) > 0:
            input_ids = args[0]
        batch, seq_len = input_ids.shape
        vocab_size = mock_config.base_config_dict["vocab_size"]

        class OutputMock:
            def __init__(self):
                # Ensure deterministic behavior and predictable shapes
                self.logits = torch.randn(batch, seq_len, vocab_size)
        return OutputMock()

    mocker.patch.object(main_model, 'forward', side_effect=mock_forward)
    mocker.patch.object(amateur_model, 'forward', side_effect=mock_forward)

    input_ids = torch.tensor([[1, 2, 3]])

    # Generate without amateur
    output_standard = main_model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        steps=2
    )

    # Generate with amateur
    output_contrastive = main_model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        steps=2,
        amateur_model=amateur_model,
        contrastive_alpha=0.5
    )

    # Both should complete successfully and return tensors
    assert isinstance(output_standard, torch.Tensor)
    assert isinstance(output_contrastive, torch.Tensor)

    # Asserting outputs have correct shapes
    assert output_standard.shape[1] == input_ids.shape[1] + 5
    assert output_contrastive.shape[1] == input_ids.shape[1] + 5

    # Note: Since the logits are randomized in the mock (randn),
    # we just need to ensure the logic runs without crashing.
    # To truly test the subtraction happened, we'll setup a small deterministic test below.

def test_contrastive_alpha_effect(mocker, mock_config):
    main_model = DiffusionModelForConditionalGeneration(mock_config)
    amateur_model = DiffusionModelForConditionalGeneration(mock_config)

    main_logits_tensor = torch.zeros(1, 8, 100)
    # Set one specific token to be very likely
    main_logits_tensor[0, :, 10] = 10.0

    amateur_logits_tensor = torch.zeros(1, 8, 100)
    # Amateur strongly agrees
    amateur_logits_tensor[0, :, 10] = 5.0

    def mock_main_forward(*args, **kwargs):
        class OutputMock:
            def __init__(self):
                self.logits = main_logits_tensor.clone()
        return OutputMock()

    def mock_amateur_forward(*args, **kwargs):
        class OutputMock:
            def __init__(self):
                self.logits = amateur_logits_tensor.clone()
        return OutputMock()

    mocker.patch.object(main_model, 'forward', side_effect=mock_main_forward)
    mocker.patch.object(amateur_model, 'forward', side_effect=mock_amateur_forward)

    input_ids = torch.tensor([[1, 2, 3]])

    # Without amateur, the model should pick token 10
    output_standard = main_model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        steps=2
    )
    assert (output_standard[0, 3:] == 10).all()

    # With a large contrastive alpha, the amateur's prediction cancels out the main model's preference,
    # and another token might be picked.
    # Here alpha=2.5. main = 10, amateur = 5. 10 - 2.5*5 = -2.5.
    # Now token 10 has logit -2.5, while others have 0.
    # It should pick something else (mostly token 0 or random, depending on argmax of zeros).
    output_contrastive = main_model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        steps=2,
        amateur_model=amateur_model,
        contrastive_alpha=2.5
    )

    assert not (output_contrastive[0, 3:] == 10).any()
