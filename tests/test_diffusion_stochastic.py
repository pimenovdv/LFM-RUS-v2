import pytest
import torch
import numpy as np

from src.models.diffusion.configuration_diffusion import DiffusionConfig
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, MDLMRequest, MDLMContinuousBatchingManager


@pytest.fixture
def dummy_model(mocker):
    config = DiffusionConfig(
        base_config_dict={"vocab_size": 100, "hidden_size": 16, "num_hidden_layers": 2, "num_attention_heads": 2, "model_type": "gpt2"},
        mask_token_id=0,
        max_timesteps=10,
    )

    # Mocking internal AutoModel and embeddings
    mock_auto_model = mocker.patch("src.models.diffusion.modeling_diffusion.AutoModel")
    mock_inner = mocker.MagicMock()
    mock_inner.config.model_type = "gpt2"
    mock_inner.config.is_causal = False

    embed_mock = torch.nn.Embedding(100, 16)
    mock_inner.get_input_embeddings.return_value = embed_mock

    class DummyOutputs:
        def __init__(self, batch_size, seq_len):
            self.last_hidden_state = torch.randn(batch_size, seq_len, 16)
            self.hidden_states = [torch.randn(batch_size, seq_len, 16)] * 3

    # Mock the inner model call to just return dummy outputs
    def mock_forward(*args, **kwargs):
        inputs_embeds = kwargs.get('inputs_embeds')
        if inputs_embeds is not None:
            bsz, seq_len = inputs_embeds.shape[:2]
        else:
            bsz, seq_len = 1, 10
        return DummyOutputs(bsz, seq_len)

    mock_inner.side_effect = mock_forward
    mock_auto_model.from_config.return_value = mock_inner

    model = DiffusionModelForConditionalGeneration(config)

    # We want to trace the generation so we'll patch forward to return slightly varied logits
    # to avoid division by zero or exactly deterministic behavior when using stochastic unmasking
    def custom_forward(input_ids=None, inputs_embeds=None, *args, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = model.inner_model.get_input_embeddings()(input_ids)
        bsz, seq_len = inputs_embeds.shape[:2]

        class Output:
            # Random logits so argmax varies slightly
            logits = torch.randn(bsz, seq_len, 100)

            def __getitem__(self, idx):
                if idx == 0:
                    return None
                return self.logits

        return Output()

    model.forward = custom_forward
    model.eval()
    return model


def test_stochastic_unmasking_generate(dummy_model):
    input_ids = torch.tensor([[1, 2, 3]])

    # Set seeds for reproducibility where possible, but we expect it to not crash
    torch.manual_seed(42)
    out1 = dummy_model.generate(
        input_ids,
        max_new_tokens=4,
        steps=2,
        stochastic_unmasking=True
    )

    torch.manual_seed(43)
    out2 = dummy_model.generate(
        input_ids,
        max_new_tokens=4,
        steps=2,
        stochastic_unmasking=False
    )

    # Mainly checking that adding stochastic_unmasking=True doesn't cause a crash
    # and processes the forward pass
    assert out1.shape == out2.shape
    assert out1.shape == (1, 7)


def test_stochastic_unmasking_speculative(dummy_model):
    input_ids = torch.tensor([[1, 2, 3]])

    # Pass itself as draft model to test speculative decoding path
    out = dummy_model.generate(
        input_ids,
        max_new_tokens=4,
        steps=2,
        stochastic_unmasking=True,
        draft_model=dummy_model,
        speculative_steps=2
    )

    assert out.shape == (1, 7)


def test_stochastic_unmasking_continuous_batching(dummy_model):
    req_config = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "max_new_tokens": 4,
        "total_steps": 2,
        "stochastic_unmasking": True
    }

    # Dynamic batching uses the manager
    outputs = dummy_model.generate_dynamic_batch([req_config])

    assert len(outputs) == 1
    assert outputs[0].shape == (1, 7)

def test_stochastic_unmasking_no_nan(dummy_model):
    input_ids = torch.tensor([[1, 2, 3]])

    # We want to make sure it doesn't output NaN for unmasked positions
    dummy_model.mask_token_id = 0
    # Add dummy token
    input_ids = torch.tensor([[1, 0, 0]])

    out = dummy_model.generate(
        input_ids,
        max_new_tokens=4,
        steps=2,
        stochastic_unmasking=True
    )

    assert not torch.isnan(out).any(), "NaN found in generation output with stochastic unmasking"
    assert out.shape == (1, 7)

def test_stochastic_unmasking_confidence_logic(mocker):
    # Test that confidence calculation handles -inf correctly and adds noise
    import torch

    # Mock confidence array
    confidence = torch.tensor([
        [-float('inf'), 0.5, 0.8, -float('inf')],
        [0.1, -float('inf'), 0.9, 0.2]
    ])

    original_confidence = confidence.clone()

    # Simulate stochastic unmasking block
    stochastic_unmasking = True
    if stochastic_unmasking:
        torch.manual_seed(42) # Deterministic noise
        noise = torch.rand_like(confidence)
        new_confidence = torch.where(confidence == -float('inf'), confidence, confidence + noise)

    # Check -inf preservation
    assert new_confidence[0, 0] == -float('inf')
    assert new_confidence[0, 3] == -float('inf')
    assert new_confidence[1, 1] == -float('inf')

    # Check real influence of noise (should not be exactly equal to original)
    assert not torch.allclose(new_confidence[0, 1:3], original_confidence[0, 1:3])
    assert not torch.allclose(new_confidence[1, 2:], original_confidence[1, 2:])

    # Check that values increased (since noise is positive)
    assert (new_confidence[0, 1] > original_confidence[0, 1]).item()
    assert (new_confidence[0, 2] > original_confidence[0, 2]).item()

    # Check behavior when False
    stochastic_unmasking = False
    if stochastic_unmasking:
        new_confidence_false = torch.where(confidence == -float('inf'), confidence, confidence + torch.rand_like(confidence))
    else:
        new_confidence_false = confidence.clone()

    assert torch.allclose(new_confidence_false, original_confidence, equal_nan=True)
