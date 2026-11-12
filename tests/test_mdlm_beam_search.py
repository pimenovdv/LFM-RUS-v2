import pytest
import torch
from unittest.mock import MagicMock
from src.models.diffusion.modeling_diffusion import DiffusionModelForConditionalGeneration, DiffusionConfig

def test_mdlm_beam_search_num_beams():
    # Setup mock config
    config = DiffusionConfig(
        base_config_dict={"model_type": "gpt2", "vocab_size": 100, "n_embd": 32, "n_layer": 2, "n_head": 2},
        mask_token_id=0,
        diffusion_steps=2,
        block_size=5
    )

    # Initialize model
    model = DiffusionModelForConditionalGeneration(config)

    # Mock the forward pass directly
    # We need to simulate logits for the beam search logic
    def mock_forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids", args[0] if len(args) > 0 else None)
        if input_ids is None:
            raise ValueError("No input_ids provided")

        batch_size, seq_len = input_ids.shape
        # Create dummy logits
        logits = torch.randn(batch_size, seq_len, config.base_config_dict["vocab_size"])

        # Make the "best" token obvious to avoid randomness issues
        logits[:, :, 1] += 5.0

        mock_output = MagicMock()
        mock_output.logits = logits
        mock_output.loss = None
        mock_output.__getitem__.side_effect = lambda idx: logits if idx == 0 else None

        return mock_output

    model.forward = mock_forward

    # Input ids: batch_size = 2, seq_len = 3
    input_ids = torch.tensor([[10, 11, 12], [20, 21, 22]])
    original_batch_size = input_ids.shape[0]

    # Test generation with num_beams > 1
    num_beams = 3
    num_return_sequences = 2
    max_new_tokens = 5

    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        steps=2,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences
    )

    # Verify shape: (original_batch_size * num_return_sequences, seq_len + max_new_tokens)
    expected_batch_size = original_batch_size * num_return_sequences
    expected_seq_len = input_ids.shape[1] + max_new_tokens

    assert output.shape == (expected_batch_size, expected_seq_len)

    # Test generation with num_beams > 1 but num_return_sequences = 1
    num_return_sequences = 1
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        steps=2,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences
    )

    expected_batch_size = original_batch_size * num_return_sequences
    assert output.shape == (expected_batch_size, expected_seq_len)
